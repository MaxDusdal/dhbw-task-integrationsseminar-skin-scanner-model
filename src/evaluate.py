"""
Evaluation script — run AFTER training to generate:
  1. Training/validation loss & accuracy curves over epochs
  2. Confusion matrix on the validation set
  3. Per-class classification report

Outputs are saved as PNG images in the models/ directory and printed to stdout.

Run:  python src/evaluate.py
"""

import os, json, pickle, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # headless backend — works without display
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from glob import glob

# ── Config (must match train.py) ─────────────────────────────────────────────
DATA_DIR    = "raw_data/image_data"
META        = "raw_data/HAM10000_metadata.csv"
MODEL_PATH  = "models/densenet121_ft.pth"
LABELS_PATH = "models/classes.pkl"
HISTORY_PATH = "models/training_history.json"
IMG_SIZE    = 224
BATCH_SIZE  = 32

CURVES_OUT    = "models/training_curves.png"
CONFUSION_OUT = "models/confusion_matrix.png"

# ── Device ───────────────────────────────────────────────────────────────────
device = (torch.device("cuda")  if torch.cuda.is_available()
     else torch.device("mps")   if torch.backends.mps.is_available()
     else torch.device("cpu"))

# ── 1. Plot training curves ─────────────────────────────────────────────────
def plot_training_curves(history_path, output_path):
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Loss
    ax1.plot(epochs, h["train_loss"], "b-o", markersize=4, label="Train loss")
    ax1.plot(epochs, h["val_loss"],   "r-o", markersize=4, label="Val loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("DenseNet-121 — Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, h["train_acc"], "b-o", markersize=4, label="Train acc")
    ax2.plot(epochs, h["val_acc"],   "r-o", markersize=4, label="Val acc")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate best val accuracy
    best_epoch = int(np.argmax(h["val_acc"])) + 1
    best_acc   = max(h["val_acc"])
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    ax2.annotate(f"Best: {best_acc:.4f} (epoch {best_epoch})",
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 1, best_acc - 0.05),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=10, color="green")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Training curves saved → {output_path}")
    plt.close()


# ── 2. Confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, output_path, normalize=False,
                          title="Confusion Matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Confusion matrix saved → {output_path}")
    plt.close()


# ── Recreate the val set (same split logic & seed as train.py) ───────────────
def get_val_loader(norm_mean, norm_std):
    all_image_path = glob(os.path.join(DATA_DIR, "*", "*.jpg"))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in all_image_path}

    lesion_type_dict = {
        'nv': 'Melanocytic nevi', 'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions', 'df': 'Dermatofibroma',
    }

    df = pd.read_csv(META)
    df['path']      = df['image_id'].map(imageid_path_dict.get)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])
    df = df.dropna(subset=['path']).reset_index(drop=True)

    # Same lesion-ID split as train.py
    img_per_lesion = df.groupby('lesion_id')['image_id'].count()
    single_ids = img_per_lesion[img_per_lesion == 1].index
    df_single  = df[df['lesion_id'].isin(single_ids)]
    _, df_val  = train_test_split(df_single, test_size=0.2,
                                  stratify=df_single['dx'], random_state=42)
    df_val = df_val.reset_index(drop=True)

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    class HAM10000(Dataset):
        def __init__(self, df, transform):
            self.df = df
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            X = Image.open(self.df['path'][idx]).convert("RGB")
            y = torch.tensor(int(self.df['label'][idx]))
            return self.transform(X), y

    loader = DataLoader(HAM10000(df_val, val_transform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, le


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check files exist
    for p in [MODEL_PATH, LABELS_PATH, HISTORY_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run `python src/train.py` first.")
            exit(1)

    # Load history and get norm values
    with open(HISTORY_PATH) as f:
        history = json.load(f)

    norm_mean = history["norm_mean"]
    norm_std  = history["norm_std"]

    # 1. Training curves
    print("\n--- Training Curves ---")
    plot_training_curves(HISTORY_PATH, CURVES_OUT)

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc   = max(history["val_acc"])
    print(f"  Best val accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"  Final train acc:   {history['train_acc'][-1]:.4f}")
    print(f"  Final val acc:     {history['val_acc'][-1]:.4f}")

    # 2. Load model
    print("\n--- Loading best model ---")
    with open(LABELS_PATH, "rb") as f:
        le = pickle.load(f)

    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, len(le.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Run inference on val set
    print("Running inference on validation set ...")
    val_loader, _ = get_val_loader(norm_mean, norm_std)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.max(1, keepdim=True)[1]
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy().squeeze())

    # 4. Classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # 5. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, le.classes_.tolist(), CONFUSION_OUT,
                          title="DenseNet-121 — Confusion Matrix (Validation)")

    # Also save normalised version
    norm_out = CONFUSION_OUT.replace(".png", "_normalised.png")
    plot_confusion_matrix(cm, le.classes_.tolist(), norm_out, normalize=True,
                          title="DenseNet-121 — Normalised Confusion Matrix")

    print("\n--- Summary ---")
    print(f"  Best val accuracy : {best_acc:.4f}")
    print(f"  Training curves   : {CURVES_OUT}")
    print(f"  Confusion matrix  : {CONFUSION_OUT}")
    print(f"  Normalised CM     : {norm_out}")
