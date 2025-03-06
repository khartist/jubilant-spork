import torch
import argparse
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from simple_clip import CLIP
from simple_clip.utils import get_image_encoder, get_text_encoder
from simple_clip.custom_datasets.clip_datasets import get_image_tranforms


def load_model(model_path, image_encoder_name, text_encoder_name, device="cpu"):
    """Load the trained CLIP model."""
    print(f"Loading model from {model_path}...")
    
    image_encoder = get_image_encoder(image_encoder_name)
    text_encoder = get_text_encoder(text_encoder_name)
    
    # Initialize model
    model = CLIP(image_encoder, text_encoder)
    
    # Load the saved weights
    try:
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)
        
    return model


def load_images(image_dir, transform):
    """Load all images from a directory and apply transforms."""
    print(f"Loading images from {image_dir}...")
    image_paths = []
    images = []
    
    # Get all image files with jpg, jpeg, or png extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], []
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    return image_paths, torch.stack(images)


def encode_text(model, tokenizer, texts, device):
    """Encode text into embeddings using the model."""
    encoded_texts = tokenizer(
        texts,
        padding=True, 
        truncation=True, 
        max_length=100, 
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    
    # Extract text features
    with torch.no_grad():
        text_features = model.extract_text_features(input_ids, attention_mask)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    
    return text_features


def encode_images(model, images, device):
    """Encode images into embeddings using the model."""
    # Move to device
    images = images.to(device)
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_images = images.shape[0]
    image_features = []
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_images = images[i:i+batch_size]
            batch_features = model.extract_image_features(batch_images)
            batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=-1)
            image_features.append(batch_features)
    
    # Concatenate all batches
    image_features = torch.cat(image_features, dim=0)
    return image_features


def search_images_with_text(model, tokenizer, text_query, image_paths, images, device, top_k=5):
    """Search for images matching a text query."""
    print(f'Searching for images matching: "{text_query}"')
    
    # Encode the text query
    text_features = encode_text(model, tokenizer, [text_query], device)
    
    # Encode all images
    image_features = encode_images(model, images, device)
    
    # Calculate similarities
    similarities = (100.0 * text_features @ image_features.T).squeeze()
    
    # Get top matches
    if similarities.dim() == 0:  # If there's only one image
        top_indices = [0] if similarities.item() > 0 else []
        top_scores = [similarities.item()] if similarities.item() > 0 else []
    else:
        values, indices = similarities.topk(min(top_k, len(image_paths)))
        top_indices = indices.cpu().numpy()
        top_scores = values.cpu().numpy()
    
    # Return results
    results = []
    for i, score in zip(top_indices, top_scores):
        results.append({
            "image_path": image_paths[i],
            "similarity_score": float(score),
            "image_index": int(i)
        })
    
    return results


def search_text_with_image(model, tokenizer, image_path, texts, device, top_k=5):
    """Search for text captions matching an image."""
    print(f'Searching for texts matching image: "{image_path}"')
    
    # Load and transform the image
    transform = get_image_tranforms((224, 224))
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Encode the image
    with torch.no_grad():
        image_features = model.extract_image_features(img_tensor)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    
    # Encode all texts
    text_features = encode_text(model, tokenizer, texts, device)
    
    # Calculate similarities
    similarities = (100.0 * image_features @ text_features.T).squeeze()
    
    # Get top matches
    if similarities.dim() == 0:  # If there's only one text
        top_indices = [0] if similarities.item() > 0 else []
        top_scores = [similarities.item()] if similarities.item() > 0 else []
    else:
        values, indices = similarities.topk(min(top_k, len(texts)))
        top_indices = indices.cpu().numpy()
        top_scores = values.cpu().numpy()
    
    # Return results
    results = []
    for i, score in zip(top_indices, top_scores):
        results.append({
            "text": texts[i],
            "similarity_score": float(score),
            "text_index": int(i)
        })
    
    return results


def calculate_similarity(model, tokenizer, image_path, text, device):
    """Calculate similarity between a single image and text."""
    print(f'Calculating similarity between image "{image_path}" and text "{text}"')
    
    # Load and transform the image
    transform = get_image_tranforms((224, 224))
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Encode the image
    with torch.no_grad():
        image_features = model.extract_image_features(img_tensor)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    
    # Encode the text
    text_features = encode_text(model, tokenizer, [text], device)
    
    # Calculate similarity
    similarity = (100.0 * (image_features @ text_features.T)).item()
    
    return similarity


def visualize_results(results, mode="text-to-image", texts=None):
    """Visualize search results."""
    if mode == "text-to-image":
        # Display images with their scores
        num_images = len(results)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        
        # Handle the case with only one result
        if num_images == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            img = Image.open(result["image_path"])
            axes[i].imshow(img)
            axes[i].set_title(f"Score: {result['similarity_score']:.2f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    elif mode == "image-to-text":
        # Display text results
        print("\nTop matching texts:")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['similarity_score']:.2f}")
            print(f"   {result['text']}")
            print()


def main():
    parser = argparse.ArgumentParser(description='CLIP Inference')
    parser.add_argument('--model_path', default='./models/clip_model.pth', help='Path to the trained model')
    parser.add_argument('--image_encoder', default='mobile_net_v3_small', choices=['tiny_vit_5m', 'mobile_net_v3_small'], help='Image encoder architecture')
    parser.add_argument('--text_encoder', default='phobert-base', help='Text encoder architecture')
    parser.add_argument('--image_dir', default=None, help='Directory containing images to search through')
    parser.add_argument('--image_path', default=None, help='Path to a single image for inference')
    parser.add_argument('--text', default=None, help='Text query or caption for inference')
    parser.add_argument('--texts_file', default=None, help='Path to a file containing text captions (one per line)')
    parser.add_argument('--mode', choices=['text-to-image', 'image-to-text', 'similarity'], required=True, help='Inference mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    
    args = parser.parse_args()
    
    # Check for required arguments based on mode
    if args.mode == 'text-to-image' and (not args.text or not args.image_dir):
        parser.error('text-to-image mode requires --text and --image_dir')
    elif args.mode == 'image-to-text' and (not args.image_path or (not args.texts_file and not args.text)):
        parser.error('image-to-text mode requires --image_path and either --texts_file or --text')
    elif args.mode == 'similarity' and (not args.image_path or not args.text):
        parser.error('similarity mode requires --image_path and --text')
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.image_encoder, args.text_encoder, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base" if args.text_encoder == "phobert-base" else args.text_encoder)
    
    # Perform inference based on the mode
    if args.mode == 'text-to-image':
        # Get image transform
        transform = get_image_tranforms((224, 224))
        
        # Load images
        image_paths, images = load_images(args.image_dir, transform)
        
        if len(image_paths) == 0:
            print("No images found. Exiting.")
            return
        
        # Search images with text
        results = search_images_with_text(model, tokenizer, args.text, image_paths, images, device, args.top_k)
        
        # Print results
        print("\nTop matching images:")
        for i, result in enumerate(results):
            print(f"{i+1}. {Path(result['image_path']).name}: Score {result['similarity_score']:.2f}")
        
        # Visualize if requested
        if args.visualize:
            visualize_results(results, mode="text-to-image")
            
    elif args.mode == 'image-to-text':
        # Load texts
        texts = []
        if args.texts_file:
            with open(args.texts_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        elif args.text:
            texts = [args.text]
        
        if not texts:
            print("No texts provided. Exiting.")
            return
        
        # Search texts with image
        results = search_text_with_image(model, tokenizer, args.image_path, texts, device, args.top_k)
        
        # Print and visualize results
        if args.visualize:
            visualize_results(results, mode="image-to-text", texts=texts)
        else:
            print("\nTop matching texts:")
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result['similarity_score']:.2f}")
                print(f"   {result['text']}")
                print()
                
    elif args.mode == 'similarity':
        # Calculate similarity
        similarity = calculate_similarity(model, tokenizer, args.image_path, args.text, device)
        print(f"\nSimilarity score: {similarity:.2f}/100")


if __name__ == "__main__":
    main()