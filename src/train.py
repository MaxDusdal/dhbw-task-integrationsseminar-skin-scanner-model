"""
Fine-tuning DenseNet-121 on HAM10000 — 7 skin lesion classes.

Closely follows the approach that achieved >90% val accuracy:
  - DenseNet-121 pretrained on ImageNet, ALL parameters trainable from epoch 1
  - Dataset-specific normalization (mean/std computed from HAM10000)
  - Lesion-ID aware train/val split (no data leakage)
  - Simple Adam + CrossEntropyLoss — no Focal Loss, no label smoothing
  - Data augmentation: flip, rotation, colour jitter
  - 25 epochs, best model saved by val accuracy

Resume support:
  - A full checkpoint (model + optimizer + epoch + history) is saved after
    every epoch that improves val accuracy.
  - If you stop training (Ctrl+C, time limit, etc.) and re-run this script,
    it automatically resumes from the last best checkpoint.
  - To force a fresh start, delete models/checkpoint.pth

Run:  python src/train.py
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(10)
torch.manual_seed(10)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "raw_data/image_data"
META        = "raw_data/HAM10000_metadata.csv"
MODEL_OUT      = "models/densenet121_ft.pth"
LABELS_OUT     = "models/classes.pkl"
HISTORY_OUT    = "models/training_history.json"
CHECKPOINT_OUT = "models/checkpoint.pth"       # full checkpoint for resume
IMG_SIZE    = 224
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 50
NUM_WORKERS = 4   # Windows + NVIDIA GPU: parallel data loading while GPU trains

# ── Device ───────────────────────────────────────────────────────────────────
device = (torch.device("cuda")  if torch.cuda.is_available()
     else torch.device("mps")   if torch.backends.mps.is_available()
     else torch.device("cpu"))

# ── Dataset class ────────────────────────────────────────────────────────────
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).convert("RGB")
        y = torch.tensor(int(self.df['label'][index]))
        if self.transform:
            X = self.transform(X)
        return X, y

# ── Training helpers ─────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

# ── Compute dataset mean / std (HAM10000-specific, not ImageNet) ─────────────
def compute_img_mean_std(image_paths):
    """Per-channel mean and std over entire dataset (RGB, 0-1 range)."""
    print("Computing dataset mean and std ...")
    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels     = 0
    for path in tqdm(image_paths):
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float64) / 255.0        # (H, W, 3)
        pixel_sum    += arr.sum(axis=(0, 1))
        pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        n_pixels     += IMG_SIZE * IMG_SIZE
    mean = pixel_sum / n_pixels
    std  = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)
    print(f"  mean = {mean.tolist()}")
    print(f"  std  = {std.tolist()}")
    return mean.tolist(), std.tolist()


def train_epoch(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    for i, (images, labels) in enumerate(train_loader):
        N = images.size(0)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.max(1, keepdim=True)[1]
        acc_meter.update(preds.eq(labels.view_as(preds)).sum().item() / N)
        loss_meter.update(loss.item())

        if (i + 1) % 100 == 0:
            print(f'    [epoch {epoch}] [iter {i+1}/{len(train_loader)}] '
                  f'[loss {loss_meter.avg:.5f}] [acc {acc_meter.avg:.5f}]')
    return loss_meter.avg, acc_meter.avg


def validate_epoch(epoch, model, val_loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    with torch.no_grad():
        for images, labels in val_loader:
            N = images.size(0)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.max(1, keepdim=True)[1]
            acc_meter.update(preds.eq(labels.view_as(preds)).sum().item() / N)
            loss_meter.update(criterion(outputs, labels).item())
    print(f'    [epoch {epoch}] [val loss {loss_meter.avg:.5f}] [val acc {acc_meter.avg:.5f}]')
    return loss_meter.avg, acc_meter.avg


# ── Entry point — required on Windows for multiprocessing (NUM_WORKERS > 0) ──
if __name__ == '__main__':

    print(f"Device: {device}")

    # ── Build image-path dictionary ──────────────────────────────────────────
    all_image_path = glob(os.path.join(DATA_DIR, "*", "*.jpg"))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    print(f"Found {len(all_image_path)} images")

    norm_mean, norm_std = compute_img_mean_std(all_image_path)

    # ── Load & prepare metadata ──────────────────────────────────────────────
    lesion_type_dict = {
        'nv':    'Melanocytic nevi',
        'mel':   'Melanoma',
        'bkl':   'Benign keratosis-like lesions',
        'bcc':   'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc':  'Vascular lesions',
        'df':    'Dermatofibroma',
    }

    df_original = pd.read_csv(META)
    df_original['path']      = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)

    # Label encoding — use dx codes directly (alphabetical: akiec bcc bkl df mel nv vasc)
    le = LabelEncoder()
    df_original['label'] = le.fit_transform(df_original['dx'])
    n_classes = len(le.classes_)

    # Drop rows where image was not found
    df_original = df_original.dropna(subset=['path']).reset_index(drop=True)
    print(f"\nDataset: {len(df_original)} samples, {n_classes} classes")
    print(df_original['cell_type'].value_counts().to_string())

    # ── Lesion-ID aware train/val split ──────────────────────────────────────
    # Multiple images of the same lesion exist in HAM10000.  Naive random splits
    # leak the same lesion into both sets → inflated val accuracy.
    # Fix: only single-image lesions are eligible for the val pool.
    img_per_lesion = df_original.groupby('lesion_id')['image_id'].count()
    single_lesion_ids = img_per_lesion[img_per_lesion == 1].index

    df_single = df_original[df_original['lesion_id'].isin(single_lesion_ids)]
    _, df_val  = train_test_split(df_single, test_size=0.2,
                                  stratify=df_single['dx'], random_state=42)
    df_train = df_original[~df_original.index.isin(df_val.index)]

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    print(f"\nTrain: {len(df_train)}  |  Val: {len(df_val)}")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(HAM10000(df_train, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(HAM10000(df_val, val_transform),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model — DenseNet-121 (ALL parameters trainable from the start) ───────
    model_ft = models.densenet121(weights="IMAGENET1K_V1")
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, n_classes)
    model = model_ft.to(device)

    # Simple Adam + CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss().to(device)

    # ReduceLROnPlateau: halves lr when val_loss doesn't improve for 4 epochs.
    # Chosen over StepLR (blind/time-based) and CosineAnnealingLR (breaks on resume)
    # because val_loss oscillates here and we want data-driven reduction.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )

    os.makedirs("models", exist_ok=True)

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch  = 1
    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
        "norm_mean": norm_mean, "norm_std": norm_std,
        "classes": le.classes_.tolist(),
    }

    if os.path.exists(CHECKPOINT_OUT):
        print(f"\n*** Resuming from checkpoint: {CHECKPOINT_OUT} ***")
        ckpt = torch.load(CHECKPOINT_OUT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        history      = ckpt["history"]
        print(f"    Resumed at epoch {start_epoch}, best val acc so far: {best_val_acc:.4f}")
    else:
        print("\nNo checkpoint found — starting fresh.")

    # ── Training loop ─────────────────────────────────────────────────────────
    remaining = EPOCHS - start_epoch + 1
    print("\n" + "=" * 60)
    print(f"Training DenseNet-121 — full fine-tuning")
    print(f"  Epochs {start_epoch}→{EPOCHS}  ({remaining} remaining)")
    print(f"  LR={LR}  Batch={BATCH_SIZE}  Device={device}")
    print("=" * 60)

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        loss_train, acc_train = train_epoch(epoch, model, train_loader, optimizer, criterion, device)
        loss_val,   acc_val   = validate_epoch(epoch, model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(loss_train)
        history["train_acc"].append(acc_train)
        history["val_loss"].append(loss_val)
        history["val_acc"].append(acc_val)

        # Step scheduler based on val_loss — reduces lr by 0.5 after 4 epochs of no improvement
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss_val)
        curr_lr = optimizer.param_groups[0]['lr']
        lr_info = f" | lr: {prev_lr:.2e} → {curr_lr:.2e}" if curr_lr != prev_lr else f" | lr: {curr_lr:.2e}"

        print(f'  Epoch {epoch:2d}/{EPOCHS} | '
              f'Train acc: {acc_train:.4f} | Val acc: {acc_val:.4f} | {elapsed:.1f}s{lr_info}')

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), MODEL_OUT)
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc":         best_val_acc,
                "history":              history,
            }, CHECKPOINT_OUT)
            print(f'  *** Best model + checkpoint saved (val acc: {best_val_acc:.4f}) ***')

        # Save history after every epoch (so evaluate.py can plot partial progress)
        with open(HISTORY_OUT, "w") as f:
            json.dump(history, f, indent=2)

    # ── Save final artifacts ──────────────────────────────────────────────────
    with open(LABELS_OUT, "wb") as f:
        pickle.dump(le, f)

    print("\n" + "=" * 60)
    print(f"Training complete — best val acc: {best_val_acc:.4f}")
    print(f"  Model      → {MODEL_OUT}  (for inference)")
    print(f"  Checkpoint → {CHECKPOINT_OUT}  (for resume)")
    print(f"  Labels     → {LABELS_OUT}")
    print(f"  History    → {HISTORY_OUT}")
    print("=" * 60)
    print(f"\nTo continue training: just run `python src/train.py` again.")
    print(f"To start fresh:       delete {CHECKPOINT_OUT} first.")
