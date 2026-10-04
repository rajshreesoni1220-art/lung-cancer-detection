"""
train.py
--------
STEPS 2 and 3 of the pipeline: feature selection + MLP training.

Prerequisite: run extract_features.py first so that models/features.npy and
models/labels.npy exist.

What this script does:
1. Loads the raw VGG16 features (usually ~25,088 numbers per image)
2. Splits into train/test sets
3. Trains an ExtraTreesClassifier to rank feature importance, and keeps only
   the top N most useful features (default 4323, matching the original project)
4. Trains a small MLP (neural network) on the reduced features to classify
   Benign / Malignant / Normal
5. Evaluates on the held-out test set (accuracy, classification report,
   confusion matrix)
6. Saves everything needed for inference later: the MLP weights, the
   ExtraTrees feature selector, and the class names

Run it with:
    python train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
TOP_K_FEATURES = 4096          # how many features to keep after selection
HIDDEN_1 = 100
HIDDEN_2 = 70
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# 1. Load extracted features
# ---------------------------------------------------------------------------
features = np.load(os.path.join(MODELS_DIR, "features.npy"))
labels = np.load(os.path.join(MODELS_DIR, "labels.npy"))
print(f"Loaded {features.shape[0]} samples, {features.shape[1]} raw features each.")

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)

# ---------------------------------------------------------------------------
# 2. Feature selection with ExtraTreesClassifier
# ---------------------------------------------------------------------------
# ExtraTrees (Extremely Randomized Trees) is an ensemble of decision trees.
# After training, it can tell us how "important" each input feature was for
# making correct predictions. We use that to throw away the least useful
# features - this reduces noise and speeds up the next stage.
print("Training ExtraTrees for feature selection...")
selector = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
selector.fit(X_train, y_train)

importances = selector.feature_importances_
top_k_indices = np.argsort(importances)[-TOP_K_FEATURES:]  # indices of best features

X_train_reduced = X_train[:, top_k_indices]
X_test_reduced = X_test[:, top_k_indices]
print(f"Reduced feature count from {features.shape[1]} to {TOP_K_FEATURES}.")

# ---------------------------------------------------------------------------
# 3. Define the MLP classifier
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    """A small feed-forward network: input -> hidden(100) -> hidden(70) -> 3 classes."""

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


model = MLPClassifier(TOP_K_FEATURES, HIDDEN_1, HIDDEN_2, len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Convert numpy arrays to PyTorch tensors and build simple data loaders
X_train_t = torch.tensor(X_train_reduced, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_reduced, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# 4. Train the MLP
# ---------------------------------------------------------------------------
train_losses, train_accuracies = [], []

print("Training MLP classifier...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Evaluate on the test set
# ---------------------------------------------------------------------------
def evaluate_mlp(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels


test_acc, preds, true_labels = evaluate_mlp(model, test_loader, device)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(classification_report(true_labels, preds, target_names=CLASS_NAMES))

# Confusion matrix plot
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
print(f"Saved confusion matrix plot to {MODELS_DIR}/confusion_matrix.png")

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "training_curves.png"))
print(f"Saved training curves to {MODELS_DIR}/training_curves.png")

# ---------------------------------------------------------------------------
# 6. Save everything needed for inference
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "mlp_model.pt"))
np.save(os.path.join(MODELS_DIR, "top_k_indices.npy"), top_k_indices)
joblib.dump(CLASS_NAMES, os.path.join(MODELS_DIR, "class_names.pkl"))

print(f"\nSaved trained MLP to {MODELS_DIR}/mlp_model.pt")
print(f"Saved selected feature indices to {MODELS_DIR}/top_k_indices.npy")
print("Training complete. You can now run app.py to serve predictions.")
