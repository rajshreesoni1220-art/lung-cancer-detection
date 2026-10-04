"""
app.py
------
Serves the trained pipeline (VGG16 feature extractor -> top-K feature
selection -> MLP classifier) as a Flask web API.

Prerequisite: run extract_features.py then train.py first, so that the
models/ folder contains:
    mlp_model.pt        (trained MLP weights)
    top_k_indices.npy   (which VGG16 features to keep)
    class_names.pkl     (["Benign", "Malignant", "Normal"])

Run it with:
    python app.py

Then test it with:
    curl -X POST -F "file=@sample_ct_scan.png" http://127.0.0.1:5000/predict
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
import joblib
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
IMAGE_SIZE = 224
HIDDEN_1 = 100
HIDDEN_2 = 70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load everything ONCE at startup (not on every request - that would be slow)
# ---------------------------------------------------------------------------
class_names = joblib.load(os.path.join(MODELS_DIR, "class_names.pkl"))
top_k_indices = np.load(os.path.join(MODELS_DIR, "top_k_indices.npy"))
top_k_indices_t = torch.tensor(top_k_indices, dtype=torch.long)

# VGG16 feature extractor - identical setup to extract_features.py
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
for param in vgg16.parameters():
    param.requires_grad = False
vgg16 = vgg16.to(device)
vgg16.eval()

# MLP classifier - must match the architecture used in train.py exactly
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


mlp_model = MLPClassifier(len(top_k_indices), HIDDEN_1, HIDDEN_2, len(class_names)).to(device)
mlp_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mlp_model.pt"), map_location=device))
mlp_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def classify_image(image_bytes):
    """Run the full pipeline on one image and return (label, confidence)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: VGG16 feature extraction (full ~25k-length vector)
        full_features = vgg16(tensor).squeeze()

        # Step 2: keep only the features selected by ExtraTrees during training
        reduced_features = full_features[top_k_indices_t.to(device)].unsqueeze(0)

        # Step 3: MLP classification
        outputs = mlp_model(reduced_features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    label = class_names[predicted_idx.item()]
    return label, confidence.item()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    # Renders templates/index.html if you build a simple upload form there.
    # If you haven't created that file yet, this route will error - that's fine,
    # /predict below works independently via curl or any HTTP client.
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send it as form field 'file'."}), 400

    file = request.files["file"]
    image_bytes = file.read()

    try:
        label, confidence = classify_image(image_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": label, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    app.run(debug=True)
