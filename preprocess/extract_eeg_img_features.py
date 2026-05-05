"""
Extract EEG image features from Things-EEG2 dataset
Data structure:
- Train: 16540 images (10 subjects × ~1654 images each)
- Test: 200 images (10 subjects × 20 images each)
- Feature dim: 1024
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# Configuration
# MODEL_NAME = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
# PRETRAINED = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
# MODEL_NAME = "hf-hub:timm/PE-Core-bigG-14-448"
# PRETRAINED = "hf-hub:timm/PE-Core-bigG-14-448"
# MODEL_NAME = "hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K"
# PRETRAINED = "hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K"
# MODEL_NAME = "hf-hub:OFA-Sys/chinese-clip-rn50"
# PRETRAINED = "hf-hub:OFA-Sys/chinese-clip-rn50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
OUTPUT_DIR = r"./EEG2image"
IMAGE_ROOT = r"./Things_EEG2/Image_set"
CENTER_IMAGES_DIR = os.path.join(IMAGE_ROOT, "center_images")
TRAINING_IMAGES_DIR = os.path.join(IMAGE_ROOT, "training_images")
TEST_IMAGES_DIR = os.path.join(IMAGE_ROOT, "test_images")

# # Load the model and tokenizer from the specified CLIP model
# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K')
# tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K')

def load_model():
    """Load chinese-clip-rn50 model and preprocessing (official chinese-clip)"""
    print("Loading OFA-Sys/chinese-clip-rn50 with official chinese-clip...")
    # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    #     MODEL_NAME
    # )
    # tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    # model = model.to(DEVICE)
    # model.eval()
    # return model, preprocess_val, tokenizer

    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name, available_models
    model, preprocess = load_from_name("RN50")
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, None

def get_image_files(directory):
    """Get all image files from directory recursively"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []
    
    if os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, filename))
    
    return sorted(image_files)

def extract_features(model, preprocess, image_paths, batch_size=32):
    """Extract features for a list of images (chinese-clip)"""
    features_list = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = preprocess(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        if not images:
            continue
        images_batch = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(images_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features_list.append(image_features.cpu())
    if features_list:
        all_features = torch.cat(features_list, dim=0)
    else:
        all_features = torch.tensor([])
    return all_features

def extract_training_features(model, preprocess):
    """Extract features from training images"""
    print("\n" + "="*60)
    print("Extracting training image features...")
    print("="*60)
    
    training_images = get_image_files(TRAINING_IMAGES_DIR)
    print(f"Found {len(training_images)} training images")
    
    if len(training_images) > 0:
        features = extract_features(model, preprocess, training_images, batch_size=32)
        print(f"Extracted training features shape: {features.shape}")
        return features
    else:
        print("No training images found!")
        return None

def extract_test_features(model, preprocess):
    """Extract features from test images"""
    print("\n" + "="*60)
    print("Extracting test image features...")
    print("="*60)
    
    test_images = get_image_files(TEST_IMAGES_DIR)
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) > 0:
        features = extract_features(model, preprocess, test_images, batch_size=16)
        print(f"Extracted test features shape: {features.shape}")
        return features
    else:
        print("No test images found!")
        return None

def extract_center_images_features(model, preprocess):
    """Extract features from center images (one per category)"""
    print("\n" + "="*60)
    print("Extracting center images features...")
    print("="*60)
    
    center_images = []
    
    # Get all category folders
    if os.path.isdir(CENTER_IMAGES_DIR):
        categories = sorted([d for d in os.listdir(CENTER_IMAGES_DIR) 
                            if os.path.isdir(os.path.join(CENTER_IMAGES_DIR, d))])
        
        for category in categories:
            category_path = os.path.join(CENTER_IMAGES_DIR, category)
            images = get_image_files(category_path)
            if images:
                # Usually take the first image as center image
                center_images.append(images[0])
    
    print(f"Found {len(center_images)} center images")
    
    if len(center_images) > 0:
        features = extract_features(model, preprocess, center_images, batch_size=16)
        print(f"Extracted center images features shape: {features.shape}")
        return features
    else:
        print("No center images found!")
        return None

def main():
    
    model, preprocess, tokenizer = load_model()
    train_features = extract_training_features(model, preprocess)
    test_features = extract_test_features(model, preprocess)
    
    # Extract center images features (for image set)
    center_features = extract_center_images_features(model, preprocess)
    
    
    # Save training features
    if train_features is not None:
        # train_path = os.path.join(OUTPUT_DIR, "ViT-H-14-378_features_train.pt")
        train_path = os.path.join(OUTPUT_DIR, "clip-rn50_features_train.pt")
        torch.save({'img_features': train_features}, train_path)
        print(f" Saved training features to {train_path}")
        print(f"  Shape: {train_features.shape}")
    
    # Save test features
    if test_features is not None:
        test_path = os.path.join(OUTPUT_DIR, "clip-rn50_features_test.pt")
        torch.save({'img_features': test_features}, test_path)
        print(f" Saved test features to {test_path}")
        print(f"  Shape: {test_features.shape}")
    
    # Save center images features (as numpy for compatibility)
    if center_features is not None:
        center_npy_path = os.path.join(OUTPUT_DIR, "center_all_image_clip-rn50.npy")
        np.save(center_npy_path, center_features.numpy())
        print(f" Saved center images features to {center_npy_path}")
        print(f"  Shape: {center_features.shape}")
    
if __name__ == "__main__":
    main()

