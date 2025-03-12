import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

from simple_clip import CLIP
from simple_clip.utils import get_image_encoder, get_text_encoder, get_dataset
from simple_clip.custom_datasets.clip_datasets import get_image_tranforms
from torch.nn import functional as F

def load_model(model_path, image_encoder_name, text_encoder_name, use_siglip=True, device="cpu", unfreeze=False):
    """Load the trained CLIP model."""
    print(f"Loading model from {model_path}...")
    
    # Initialize with the same parameters as in training
    if not use_siglip:
        init_tau, init_b = np.log(5), 0
    else:
        init_tau, init_b = np.log(10), -10
    
    image_encoder = get_image_encoder(image_encoder_name, unfreeze=unfreeze)
    text_encoder = get_text_encoder(text_encoder_name, unfreeze=unfreeze)
    
    # Initialize model with same parameters as training
    model = CLIP(image_encoder, text_encoder, init_tau=init_tau, init_b=init_b)
    
    # Load the saved weights with explicit device mapping
    try:
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        print(f"Model temperature: {model.t_prime.exp().item():.4f}")
        print(f"Model bias: {model.b.item():.4f}")
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Attempting to load with strict=False...")
        try:
            model_state = torch.load(model_path, map_location=device)
            model.load_state_dict(model_state, strict=False)
            model = model.to(device)
            model.eval()
            print(f"Model temperature: {model.t_prime.exp().item():.4f}")
            print(f"Model bias: {model.b.item():.4f}")
            print("Model loaded successfully with strict=False!")
        except Exception as e:
            print(f"Still failed to load model: {e}")
            exit(1)
        
    return model


def extract_all_features(model, test_dataset, device, batch_size=256):
    """Extract features for all unique images and all captions"""
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Extract dataset structure
    unique_img_indices = set(test_dataset.image_ids)
    img_idx_to_captions = test_dataset.image_to_annotations
    
    # Collect all caption embeddings
    all_caption_embeddings = []
    all_image_embeddings = []  # This will store embeddings for ALL captions (including duplicates)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'image']}
            
            # Extract features
            image_features = model.extract_image_features(batch['image'])
            text_features = model.extract_text_features(batch['input_ids'], batch['attention_mask'])
            
            # Normalize features
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            
            all_image_embeddings.append(image_features.cpu())
            all_caption_embeddings.append(text_features.cpu())
    
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_caption_embeddings = torch.cat(all_caption_embeddings)
    
    # Now, extract unique image embeddings using the first caption for each image
    unique_image_embeddings = {}
    
    # For each unique image index
    for img_idx in unique_img_indices:
        # Get the first caption index for this image
        caption_idx = img_idx_to_captions[img_idx][0]
        # Store the image embedding for this unique image
        unique_image_embeddings[img_idx] = all_image_embeddings[caption_idx]
    
    # Convert to tensors
    unique_img_list = list(unique_img_indices)
    unique_img_tensor = torch.stack([unique_image_embeddings[idx] for idx in unique_img_list])
    
    return unique_img_tensor, all_caption_embeddings, unique_img_list, test_dataset.image_ids


def extract_features(model, dataset, device):
    """Extract embeddings with proper ID handling between images and captions"""
    # First, get actual image IDs for each annotation (fixing the index vs ID issue)
    actual_image_ids = []
    for i in range(len(dataset)):
        ann_idx = i
        img_idx = dataset.annotation_to_image[ann_idx]
        entry = dataset.data['images'][img_idx]
        actual_image_ids.append(entry['id'])  # Store ACTUAL image ID
    
    # Extract unique images (one per unique ID)
    unique_images = {}  # Map actual_image_id -> image
    unique_id_to_idx = {}  # Map actual_image_id -> index in dataset
    
    for ann_idx, img_id in enumerate(actual_image_ids):
        if img_id not in unique_images:
            caption = dataset[ann_idx]
            unique_images[img_id] = caption['image'].unsqueeze(0)
            unique_id_to_idx[img_id] = dataset.annotation_to_image[ann_idx]
    
    # Process unique images
    image_embeddings = []
    image_ids = []  # Store ACTUAL image IDs
    
    print(f"Processing {len(unique_images)} unique images...")
    for img_id, image in tqdm(unique_images.items()):
        with torch.no_grad():
            embedding = model.extract_image_features(image.to(device))
            embedding = F.normalize(embedding, p=2, dim=-1)
            image_embeddings.append(embedding.cpu())
            image_ids.append(img_id)
    
    image_embeddings = torch.cat(image_embeddings)
    
    # Process all captions, mapping to ACTUAL image IDs
    caption_embeddings = []
    caption_img_ids = []  # Store ACTUAL image IDs for each caption
    
    print(f"Processing all {len(dataset)} captions...")
    for i in tqdm(range(len(dataset))):
        caption = dataset[i]
        img_id = actual_image_ids[i]  # Get ACTUAL image ID
        
        with torch.no_grad():
            embedding = model.extract_text_features(
                caption['input_ids'].unsqueeze(0).to(device),
                caption['attention_mask'].unsqueeze(0).to(device)
            )
            embedding = F.normalize(embedding, p=2, dim=-1)
            caption_embeddings.append(embedding.cpu())
            caption_img_ids.append(img_id)
    
    caption_embeddings = torch.cat(caption_embeddings)
    
    # Verify ID matching
    match_count = sum(1 for img_id in caption_img_ids if img_id in image_ids)
    print(f"Caption-Image ID match check: {match_count}/{len(caption_img_ids)} captions have matching images")
    
    return image_embeddings, caption_embeddings, image_ids, caption_img_ids


def precision_at_k(relevance, k):
    if k == 0:
        return 0.0
    return np.sum(relevance[:k]) / k


def recall_at_k(relevance, k, total_relevant):
    if total_relevant == 0:
        return 0.0
    return np.sum(relevance[:k]) / total_relevant


def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(relevance):
    if np.sum(relevance) == 0:
        return 0.0
    cum_precision = np.cumsum(relevance) / (np.arange(len(relevance)) + 1)
    return np.sum(cum_precision * relevance) / np.sum(relevance)


def ndcg_at_k(relevance, k):
    if k == 0:
        return 0.0
    dcg = np.sum(relevance[:k] / np.log2(np.arange(2, k + 2)))
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = np.sum(ideal_relevance[:k] / np.log2(np.arange(2, k + 2)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def print_retrieval_metrics(metrics, k_values=[1, 5]):
    print("\n" + "="*80)
    print(" "*30 + "RETRIEVAL EVALUATION")
    print("="*80)
    
    print("\nTop-K Accuracy (%):")
    print("-" * 40)
    print(f"Image->Text Top1: {metrics['i2t']['top1']:.2f}%")
    print(f"Image->Text Top5: {metrics['i2t']['top5']:.2f}%")
    print(f"Text->Image Top1: {metrics['t2i']['top1']:.2f}%")
    print(f"Text->Image Top5: {metrics['t2i']['top5']:.2f}%")
    print(f"Mean Top1: {(metrics['i2t']['top1'] + metrics['t2i']['top1']) / 2:.2f}%")
    print(f"Mean Top5: {(metrics['i2t']['top5'] + metrics['t2i']['top5']) / 2:.2f}%")
    
    print("\nDetailed Retrieval Metrics:")
    for task, task_name in [("i2t", "Image-to-Text"), ("t2i", "Text-to-Image")]:
        print(f"\n{task_name} Retrieval:")
        print("-" * 80)
        
        headers = ["Metric"] + [f"@{k}" for k in k_values]
        row_format = "{:15}" + "{:15}" * len(k_values)
        
        print(row_format.format(*headers))
        print("-" * 80)
        
        for metric_name in ["precision", "recall", "f1", "ndcg"]:
            metric_display = metric_name.upper()
            values = [metrics[task][metric_name][k] for k in k_values]
            formatted_values = [f"{v:.4f}" for v in values]
            print(row_format.format(metric_display, *formatted_values))
        
        print(f"MAP: {metrics[task]['mAP']:.4f}")
    
    print("\n" + "="*80)


def evaluate_retrieval(image_embeddings, caption_embeddings, unique_img_list, caption_img_ids, 
                        temperature=1.0, bias=0.0, k_values=[1, 5]):
    """Simple evaluation for CLIP retrieval performance with correct ID matching"""
    
    # 1. Compute similarity matrix using RAW cosine similarity
    print("Computing similarity matrix...")
    similarity = torch.matmul(image_embeddings, caption_embeddings.t())
    
    # Print raw similarity stats
    print(f"Raw similarity stats: min={similarity.min().item():.4f}, max={similarity.max().item():.4f}, mean={similarity.mean().item():.4f}")
    
    similarity = similarity * temperature + bias
    similarity = similarity.cpu().numpy()

    # 2. Create mapping from image IDs to positions in similarity matrix
    img_idx_to_pos = {img_idx: pos for pos, img_idx in enumerate(unique_img_list)}
    
    # 3. Initialize metrics
    metrics = {
        'i2t': {'top1': 0.0, 'top5': 0.0, 'mAP': 0.0,
                'precision': {k: 0.0 for k in k_values},
                'recall': {k: 0.0 for k in k_values},
                'f1': {k: 0.0 for k in k_values},
                'ndcg': {k: 0.0 for k in k_values}},
        't2i': {'top1': 0.0, 'top5': 0.0, 'mAP': 0.0,
                'precision': {k: 0.0 for k in k_values},
                'recall': {k: 0.0 for k in k_values},
                'f1': {k: 0.0 for k in k_values},
                'ndcg': {k: 0.0 for k in k_values}}
    }
    
    n_images = len(unique_img_list)
    n_captions = len(caption_embeddings)
    
    # Count valid samples for stats
    valid_images = 0
    valid_captions = 0
    
    # 4. Image-to-text evaluation
    print("Evaluating image-to-text retrieval...")
    for i, img_idx in enumerate(tqdm(unique_img_list)):
        # Get relevant caption indices for this image
        relevant_captions = [j for j, cap_img_id in enumerate(caption_img_ids) if cap_img_id == img_idx]
        
        # Skip if no relevant captions (shouldn't happen but just in case)
        if not relevant_captions:
            continue
        
        valid_images += 1
        
        # Get similarity scores for this image
        i2t_scores = similarity[i, :]
        
        # Sort by score (descending)
        i2t_ranks = np.argsort(-i2t_scores)
        
        # Create relevance array (1 = caption belongs to this image)
        relevance = np.zeros(n_captions)
        for j in relevant_captions:
            relevance[j] = 1
        
        # Apply ranking
        relevance = relevance[i2t_ranks]
        
        # Calculate top-k accuracy
        metrics['i2t']['top1'] += (np.sum(relevance[:1]) > 0)
        metrics['i2t']['top5'] += (np.sum(relevance[:5]) > 0)
        
        # Calculate other metrics
        total_relevant = np.sum(relevance)
        for k in k_values:
            p_k = precision_at_k(relevance, k)
            r_k = recall_at_k(relevance, k, total_relevant)
            metrics['i2t']['precision'][k] += p_k
            metrics['i2t']['recall'][k] += r_k
            metrics['i2t']['f1'][k] += f1_at_k(p_k, r_k)
            metrics['i2t']['ndcg'][k] += ndcg_at_k(relevance, k)
        
        metrics['i2t']['mAP'] += average_precision(relevance)
    
    # 5. Text-to-image evaluation
    print("Evaluating text-to-image retrieval...")
    for caption_idx in tqdm(range(n_captions)):
        # Get image ID for this caption
        img_idx = caption_img_ids[caption_idx]
        
        # Skip if image ID not in our list of unique images
        if img_idx not in img_idx_to_pos:
            continue
            
        valid_captions += 1
        
        # Get position in similarity matrix
        img_pos = img_idx_to_pos[img_idx]
        
        # Get similarity scores for this caption
        t2i_scores = similarity[:, caption_idx]
        
        # Sort by score (descending)
        t2i_ranks = np.argsort(-t2i_scores)
        
        # Create relevance array (1 = image belongs to this caption)
        relevance = np.zeros(n_images)
        relevance[img_pos] = 1
        relevance = relevance[t2i_ranks]
        
        # Calculate top-k accuracy
        metrics['t2i']['top1'] += (relevance[0] > 0)
        metrics['t2i']['top5'] += (np.sum(relevance[:5]) > 0)
        
        # Calculate other metrics
        for k in k_values:
            p_k = precision_at_k(relevance, k)
            r_k = recall_at_k(relevance, k, 1)  # Only 1 relevant image
            metrics['t2i']['precision'][k] += p_k
            metrics['t2i']['recall'][k] += r_k
            metrics['t2i']['f1'][k] += f1_at_k(p_k, r_k)
            metrics['t2i']['ndcg'][k] += ndcg_at_k(relevance, k)
        
        metrics['t2i']['mAP'] += average_precision(relevance)
    
    print(f"\nFound {valid_images}/{n_images} valid images and {valid_captions}/{n_captions} valid captions for evaluation")
    
    # 6. Normalize metrics by valid samples
    if valid_images > 0:
        for metric in ['top1', 'top5', 'mAP']:
            metrics['i2t'][metric] /= valid_images
        
        for k in k_values:
            for metric in ['precision', 'recall', 'f1', 'ndcg']:
                metrics['i2t'][metric][k] /= valid_images
    
    if valid_captions > 0:
        for metric in ['top1', 'top5', 'mAP']:
            metrics['t2i'][metric] /= valid_captions
        
        for k in k_values:
            for metric in ['precision', 'recall', 'f1', 'ndcg']:
                metrics['t2i'][metric][k] /= valid_captions
    
    # Convert accuracy to percentage
    metrics['i2t']['top1'] *= 100
    metrics['i2t']['top5'] *= 100
    metrics['t2i']['top1'] *= 100
    metrics['t2i']['top5'] *= 100
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='CLIP Evaluation')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--image_encoder', default='mobile_net_v3_small', help='Image encoder type')
    parser.add_argument('--text_encoder', default='phobert-base', help='Text encoder type')
    parser.add_argument('--dataset_name', default='ktvic', help='Dataset name')
    parser.add_argument('--dataset_path', default='./simple_clip/custom_datasets', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--no_siglip', action='store_true', help='Use contrastive loss instead of SigLIP')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze the model')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    use_siglip = not args.no_siglip
    model = load_model(args.model_path, args.image_encoder, args.text_encoder, use_siglip, device, args.unfreeze)
    
    # Get image transforms
    transforms = get_image_tranforms((args.image_size, args.image_size))
    
    # Load test dataset
    test_dataset = get_dataset(args.text_encoder, args.dataset_name, args.dataset_path, 
                             transforms=transforms, split="test")
    
    print(f"Test dataset size: {len(test_dataset)} captions")
    print(f"Loading ALL features at once for evaluation...")

        # Debug dataset structure
    print("\n==== DATASET STRUCTURE DEBUG ====")
    print(f"First 5 annotations-to-image mappings:")
    for i in range(5):
        ann_idx = i
        img_idx = test_dataset.annotation_to_image[ann_idx]
        img_id = test_dataset.data['images'][img_idx]['id']
        print(f"  Annotation {i} → Image Index {img_idx} → Actual Image ID {img_id}")
        
    print(f"\nImage-to-annotations mappings (first 3 images):")
    counter = 0
    for img_idx, ann_idxs in test_dataset.image_to_annotations.items():
        img_id = test_dataset.data['images'][img_idx]['id']
        print(f"  Image Index {img_idx} (ID: {img_id}) → Annotations: {ann_idxs}")
        counter += 1
        if counter >= 3:
            break
        
    # Extract all features
    unique_img_embeddings, caption_embeddings, unique_img_list, caption_img_ids = extract_features(
        model, test_dataset, device
    )
    
    # Get model temperature and bias
    temperature = model.t_prime.exp().cpu().detach()
    bias = model.b.cpu().detach()

    # Add this right before evaluate_retrieval() call in main()
    print(f"First 5 unique image IDs: {unique_img_list[:5]}")
    print(f"First 5 caption image IDs: {caption_img_ids[:5]}")

    # Count matches (should be multiple per image)
    matches = 0
    for cap_img_id in caption_img_ids[:100]:  # Check first 100 captions
        if cap_img_id in unique_img_list:
            matches += 1
    print(f"Found {matches}/100 matching image IDs between captions and images")
        
    # Evaluate retrieval
    metrics = evaluate_retrieval(
        unique_img_embeddings, caption_embeddings, unique_img_list, caption_img_ids, 
        temperature=temperature, bias=bias
    )
    
    # Print results
    print_retrieval_metrics(metrics)
    
    # Save results to file
    os.makedirs('./results', exist_ok=True)
    model_name = os.path.basename(args.model_path).split('.')[0]
    result_file = f"./results/{model_name}_results.txt"
    
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Temperature: {temperature.item():.4f}\n")
        f.write(f"Bias: {bias.item():.4f}\n\n")
        
        f.write("Top-K Accuracy (%):\n")
        f.write(f"Image->Text Top1: {metrics['i2t']['top1']:.2f}%\n")
        f.write(f"Image->Text Top5: {metrics['i2t']['top5']:.2f}%\n")
        f.write(f"Text->Image Top1: {metrics['t2i']['top1']:.2f}%\n")
        f.write(f"Text->Image Top5: {metrics['t2i']['top5']:.2f}%\n")
        f.write(f"Mean Top1: {(metrics['i2t']['top1'] + metrics['t2i']['top1']) / 2:.2f}%\n")
        f.write(f"Mean Top5: {(metrics['i2t']['top5'] + metrics['t2i']['top5']) / 2:.2f}%\n\n")
        
        for task, task_name in [("i2t", "Image-to-Text"), ("t2i", "Text-to-Image")]:
            f.write(f"{task_name} Detailed Metrics:\n")
            for k in [1, 5]:
                f.write(f"  Precision@{k}: {metrics[task]['precision'][k]:.4f}\n")
                f.write(f"  Recall@{k}: {metrics[task]['recall'][k]:.4f}\n")
                f.write(f"  F1@{k}: {metrics[task]['f1'][k]:.4f}\n")
                f.write(f"  NDCG@{k}: {metrics[task]['ndcg'][k]:.4f}\n")
            f.write(f"  MAP: {metrics[task]['mAP']:.4f}\n\n")
    
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
