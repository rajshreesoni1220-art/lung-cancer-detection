"""
extract_features.py
--------------------
STEP 1 of the pipeline: turn every CT scan image into a numeric "feature vector"
using a pretrained VGG16 network.

Why do this instead of training a CNN from scratch?
VGG16 was trained on 1.4 million everyday photos (ImageNet) and already learned
to recognize edges, textures, and shapes. We reuse ("transfer") that ability
instead of trying to teach a network to see from zero using only ~1,000 CT images
(which is not nearly enough data to train a deep CNN well).

What this script does:
1. Walks through dataset/Lung/<class_name>/*.png (or .jpg)
2. Resizes every image to 224x224 (the input size VGG16 expects)
3. Normalizes pixel values the same way ImageNet images were normalized
4. Passes each image through VGG16 with the final classification layer removed,
   producing a long vector of numbers (features) per image
5. Saves all the feature vectors + their labels to disk as .npy files, so you
   never have to redo this slow step again.

Run it with:
    python extract_features.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG - change these if your folders are named differently
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset/Lung"          # expects subfolders like "Benign cases", etc.
CLASS_NAMES = ["Benign cases", "Malignant cases", "Normal cases"]
OUTPUT_DIR = "models"
IMAGE_SIZE = 224

# Use the GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. Build the VGG16 feature extractor
# ---------------------------------------------------------------------------
# Load VGG16 with weights pretrained on ImageNet.
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# VGG16 normally ends with a classifier that outputs 1000 ImageNet classes.
# We don't want that - we want the features that feed INTO that classifier.
# vgg16.classifier is a stack of Linear/ReLU/Dropout layers; we drop the very
# last Linear layer (the 1000-class output) and keep everything before it.
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])

# Freeze all the weights - we are NOT training VGG16, just using it as-is.
for param in vgg16.parameters():
    param.requires_grad = False

vgg16 = vgg16.to(device)
vgg16.eval()  # inference mode (disables dropout etc.)

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
# These mean/std values are the standard ImageNet normalization stats.
# We use them because VGG16 was trained with images normalized this way.
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def extract_feature(image_path):
    """Load one image and return its VGG16 feature vector as a numpy array."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        features = vgg16(tensor)
    return features.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Walk the dataset folder and extract features for every image
# ---------------------------------------------------------------------------
def main():
    all_features = []
    all_labels = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: folder not found, skipping: {class_dir}")
            continue

        image_files = [f for f in os.listdir(class_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        print(f"Extracting features for '{class_name}' ({len(image_files)} images)...")
        for filename in tqdm(image_files):
            path = os.path.join(class_dir, filename)
            try:
                feat = extract_feature(path)
                all_features.append(feat)
                all_labels.append(label_idx)
            except Exception as e:
                print(f"  Skipping {filename}: {e}")

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), all_features)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), all_labels)

    print(f"\nDone. Extracted {all_features.shape[0]} feature vectors "
          f"of length {all_features.shape[1]} each.")
    print(f"Saved to {OUTPUT_DIR}/features.npy and {OUTPUT_DIR}/labels.npy")


if __name__ == "__main__":
    main()
