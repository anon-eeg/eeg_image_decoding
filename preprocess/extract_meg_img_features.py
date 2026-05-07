"""
Extract chinese-clip-rn50 image features from Things-MEG dataset
Data structure:
- Train: training images from things-meg/Image_set/training_images
- Test: test images from things-meg/Image_set/test_images
- Feature dim: 1024
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
    
# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
OUTPUT_DIR = r"./things-meg"
IMAGE_ROOT = r"./things-meg/Image_set"
TRAINING_IMAGES_DIR = os.path.join(IMAGE_ROOT, "training_images")
TEST_IMAGES_DIR = os.path.join(IMAGE_ROOT, "test_images")

def load_model():
    """Load chinese-clip-rn50 model and preprocessing (official chinese-clip)"""
    print("Available models:", available_models())
    model, preprocess = load_from_name("RN50")
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess

def get_image_files(directory):
    """Get all image files from directory recursively"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
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
            # chinese-clip official API
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
    print("Extracting MEG training image features...")
    print("="*60)
    
    train_path = os.path.join(OUTPUT_DIR, "meg_clip-rn50_features_train.pt")
    
    # Check if training features already exist
    if os.path.exists(train_path):
        print(f"Training features already exist at {train_path}")
        print("Loading existing features...")
        try:
            data = torch.load(train_path)
            features = data['img_features']
            print(f"Loaded training features shape: {features.shape}")
            return features
        except Exception as e:
            print(f"Error loading existing features: {e}")
            print("Re-extracting features...")
    
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
    print("Extracting MEG test image features...")
    print("="*60)
    
    test_path = os.path.join(OUTPUT_DIR, "meg_clip-rn50_features_test.pt")
    
    # Check if test features already exist
    if os.path.exists(test_path):
        print(f"Test features already exist at {test_path}")
        print("Loading existing features...")
        try:
            data = torch.load(test_path)
            features = data['img_features']
            print(f"Loaded test features shape: {features.shape}")
            return features
        except Exception as e:
            print(f"Error loading existing features: {e}")
            print("Re-extracting features...")
    
    test_images = get_image_files(TEST_IMAGES_DIR)
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) > 0:
        features = extract_features(model, preprocess, test_images, batch_size=16)
        print(f"Extracted test features shape: {features.shape}")
        return features
    else:
        print("No test images found!")
        return None

def main():

    model, preprocess = load_model()
    train_features = extract_training_features(model, preprocess)
    test_features = extract_test_features(model, preprocess)
    
    # Save training features
    if train_features is not None:
        train_path = os.path.join(OUTPUT_DIR, "meg_clip-rn50_features_train.pt")
        torch.save({'img_features': train_features}, train_path)
        print(f" Saved training features to {train_path}")
        print(f"  Shape: {train_features.shape}")
    
    # Save test features
    if test_features is not None:
        test_path = os.path.join(OUTPUT_DIR, "meg_clip-rn50_features_test.pt")
        torch.save({'img_features': test_features}, test_path)
        print(f" Saved test features to {test_path}")
        print(f"  Shape: {test_features.shape}")
    
    print("\n" + "="*60)
    print("MEG image feature extraction completed!")
    print("="*60)

if __name__ == "__main__":
    main()
