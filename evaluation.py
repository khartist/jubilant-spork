import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
from pathlib import Path

from simple_clip import CLIP
from simple_clip.utils import get_image_encoder, get_text_encoder, get_dataset
from simple_clip.custom_datasets.clip_datasets import get_image_tranforms


def load_model(model_path, image_encoder_name, text_encoder_name, use_siglip=True, device="cpu"):
    """Load the trained CLIP model."""
    print(f"Loading model from {model_path}...")
    
    # Initialize with the same parameters as in training
    if not use_siglip:
        init_tau, init_b = np.log(5), 0
    else:
        init_tau, init_b = np.log(10), -10
    
    image_encoder = get_image_encoder(image_encoder_name)
    text_encoder = get_text_encoder(text_encoder_name)
    
    # Initialize model with same parameters as training
    model = CLIP(image_encoder, text_encoder, init_tau=init_tau, init_b=init_b)
    
    # Load the saved weights
    try:
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Attempting to load with strict=False...")
        try:
            model.load_state_dict(model_state, strict=False)
            model = model.to(device)
            model.eval()
            print("Model loaded successfully with strict=False!")
        except Exception as e:
            print(f"Still failed to load model: {e}")
            exit(1)
        
    return model


def compute_embeddings(model, test_loader, device):
    """Compute image and text embeddings for the entire dataset."""
    image_embeddings = []
    text_embeddings = []
    all_image_ids = []
    all_text_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Computing embeddings")):
            # Handle batch based on the specific structure from your dataset
            if 'image' in batch:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Generate sequential IDs since they're not provided in the dataset
                batch_size = images.size(0)
                img_ids = [f"{batch_idx}_{j}" for j in range(batch_size)]
                txt_ids = [f"{batch_idx}_{j}" for j in range(batch_size)]
                
                # Get embeddings
                image_emb = model.extract_image_features(images)
                text_emb = model.extract_text_features(input_ids, attention_mask)
                
                # Store embeddings and IDs
                image_embeddings.append(image_emb.cpu())
                text_embeddings.append(text_emb.cpu())
                all_image_ids.extend(img_ids)
                all_text_ids.extend(txt_ids)
            else:
                # This is a fallback for any other format, though it shouldn't be needed
                print(f"Warning: Unknown batch format. Available keys: {batch.keys() if isinstance(batch, dict) else 'Not a dictionary'}")
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    return image_embeddings, text_embeddings, all_image_ids, all_text_ids

def precision_at_k(relevance, k):
    """Calculate precision@k."""
    if k == 0:
        return 0.0
    return np.sum(relevance[:k]) / k


def recall_at_k(relevance, k, total_relevant):
    """Calculate recall@k."""
    if total_relevant == 0:
        return 0.0
    return np.sum(relevance[:k]) / total_relevant


def f1_at_k(precision, recall):
    """Calculate F1@k."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(relevance):
    """Calculate average precision."""
    if np.sum(relevance) == 0:
        return 0.0
    
    cum_precision = np.cumsum(relevance) / (np.arange(len(relevance)) + 1)
    ap = np.sum(cum_precision * relevance) / np.sum(relevance)
    return ap


def ndcg_at_k(relevance, k):
    """Calculate NDCG@k."""
    dcg = np.sum(relevance[:k] / np.log2(np.arange(2, k + 2)))
    
    # Calculate ideal DCG (IDCG)
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = np.sum(ideal_relevance[:k] / np.log2(np.arange(2, k + 2)))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(image_embs, text_embs, k_values=[1, 5]):
    """
    Evaluate cross-modal retrieval performance with various metrics.
    
    Args:
        image_embs (torch.Tensor): Normalized image embeddings
        text_embs (torch.Tensor): Normalized text embeddings
        k_values (list): Values of k for which to compute metrics
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Compute similarity matrix
    sim_matrix = torch.matmul(image_embs, text_embs.t()).numpy()
    
    n = sim_matrix.shape[0]
    metrics = {}
    
    # Image-to-Text Retrieval
    i2t_precision = {k: 0.0 for k in k_values}
    i2t_recall = {k: 0.0 for k in k_values}
    i2t_f1 = {k: 0.0 for k in k_values}
    i2t_map = {k: 0.0 for k in k_values}
    i2t_ndcg = {k: 0.0 for k in k_values}
    
    # Text-to-Image Retrieval
    t2i_precision = {k: 0.0 for k in k_values}
    t2i_recall = {k: 0.0 for k in k_values}
    t2i_f1 = {k: 0.0 for k in k_values}
    t2i_map = {k: 0.0 for k in k_values}
    t2i_ndcg = {k: 0.0 for k in k_values}
    
    for i in range(n):
        # For image-to-text retrieval
        i2t_scores = sim_matrix[i, :]
        i2t_ranks = np.argsort(-i2t_scores)  # Sort in descending order
        
        # Ground truth is that the i-th text is relevant
        i2t_relevance = np.zeros(n)
        i2t_relevance[i] = 1
        i2t_relevance = i2t_relevance[i2t_ranks]  # Reorder based on ranking
        
        # For text-to-image retrieval
        t2i_scores = sim_matrix[:, i]
        t2i_ranks = np.argsort(-t2i_scores)  # Sort in descending order
        
        # Ground truth is that the i-th image is relevant
        t2i_relevance = np.zeros(n)
        t2i_relevance[i] = 1
        t2i_relevance = t2i_relevance[t2i_ranks]  # Reorder based on ranking
        
        # Calculate metrics for each k
        for k in k_values:
            # Image-to-Text
            i2t_p = precision_at_k(i2t_relevance, k)
            i2t_r = recall_at_k(i2t_relevance, k, 1)
            i2t_precision[k] += i2t_p
            i2t_recall[k] += i2t_r
            i2t_f1[k] += f1_at_k(i2t_p, i2t_r)
            i2t_map[k] += average_precision(i2t_relevance[:k])
            i2t_ndcg[k] += ndcg_at_k(i2t_relevance, k)
            
            # Text-to-Image
            t2i_p = precision_at_k(t2i_relevance, k)
            t2i_r = recall_at_k(t2i_relevance, k, 1)
            t2i_precision[k] += t2i_p
            t2i_recall[k] += t2i_r
            t2i_f1[k] += f1_at_k(t2i_p, t2i_r)
            t2i_map[k] += average_precision(t2i_relevance[:k])
            t2i_ndcg[k] += ndcg_at_k(t2i_relevance, k)
    
    # Average over all queries
    for k in k_values:
        # Image-to-Text
        i2t_precision[k] /= n
        i2t_recall[k] /= n
        i2t_f1[k] /= n
        i2t_map[k] /= n
        i2t_ndcg[k] /= n
        
        # Text-to-Image
        t2i_precision[k] /= n
        t2i_recall[k] /= n
        t2i_f1[k] /= n
        t2i_map[k] /= n
        t2i_ndcg[k] /= n
    
    # Compile metrics
    metrics["i2t"] = {
        "precision": i2t_precision,
        "recall": i2t_recall,
        "f1": i2t_f1,
        "mAP": i2t_map,
        "ndcg": i2t_ndcg
    }
    
    metrics["t2i"] = {
        "precision": t2i_precision,
        "recall": t2i_recall,
        "f1": t2i_f1,
        "mAP": t2i_map,
        "ndcg": t2i_ndcg
    }
    
    return metrics


def print_metrics(metrics, k_values=[1, 5]):
    """Print the evaluation metrics in a formatted table."""
    print("\n" + "="*80)
    print(" "*30 + "EVALUATION RESULTS")
    print("="*80)
    
    for task, task_name in [("i2t", "Image-to-Text"), ("t2i", "Text-to-Image")]:
        print(f"\n{task_name} Retrieval:")
        print("-" * 80)
        
        headers = ["Metric"] + [f"@{k}" for k in k_values]
        row_format = "{:15}" + "{:15}" * len(k_values)
        
        print(row_format.format(*headers))
        print("-" * 80)
        
        for metric_name in ["precision", "recall", "f1", "mAP", "ndcg"]:
            metric_display = metric_name.upper()
            values = [metrics[task][metric_name][k] for k in k_values]
            formatted_values = [f"{v:.4f}" for v in values]
            print(row_format.format(metric_display, *formatted_values))
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='CLIP Model Evaluation')
    parser.add_argument('--model_path', default='./models/clip_model.pth', help='Path to the trained model')
    parser.add_argument('--image_encoder', default='mobile_net_v3_small', choices=['mobile_net_v3_small', 'tiny_vit_5m'], help='Image encoder architecture')
    parser.add_argument('--text_encoder', default='phobert-base', choices=['phobert-base', 'sentence_transformer'], help='Text encoder architecture')
    parser.add_argument('--dataset_name', default='ktvic', choices=['ktvic'], help='Dataset name for evaluation')
    parser.add_argument('--image_size', default=224, type=int, help='Image size for evaluation')
    parser.add_argument('--use_siglip', action='store_true', help='Use SigLIP loss instead of contrastive loss')
    parser.add_argument('--dataset_path', default='./simple_clip/custom_datasets', help='Path to the dataset')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with the appropriate parameters
    model = load_model(args.model_path, args.image_encoder, args.text_encoder, args.use_siglip, device)
    model.eval()
    
    # Prepare transforms
    transforms = get_image_tranforms((args.image_size, args.image_size))
    
    # Load dataset
    test_dataset = get_dataset(args.text_encoder, args.dataset_name, args.dataset_path, 
                               transforms=transforms, split="test")
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Compute embeddings
    image_embeddings, text_embeddings, image_ids, text_ids = compute_embeddings(
        model, test_loader, device
    )
    
    # Normalize embeddings for cosine similarity
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    
    # Evaluate retrieval performance
    metrics = evaluate_retrieval(image_embeddings, text_embeddings, k_values=[1, 5])
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results to file
    results_dir = "./evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/clip_evaluation_results.txt", "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Image Encoder: {args.image_encoder}\n")
        f.write(f"Text Encoder: {args.text_encoder}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Using SigLIP: {args.use_siglip}\n\n")
        
        for task, task_name in [("i2t", "Image-to-Text"), ("t2i", "Text-to-Image")]:
            f.write(f"{task_name} Retrieval:\n")
            f.write("-" * 40 + "\n")
            
            for metric in ["precision", "recall", "f1", "mAP", "ndcg"]:
                for k in [1, 5]:
                    f.write(f"{metric.upper()}@{k}: {metrics[task][metric][k]:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to {results_dir}/clip_evaluation_results.txt")


if __name__ == "__main__":
    main()
