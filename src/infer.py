"""
Inference script for the fine-tuned DenseNet-121 skin lesion classifier.

Features:
  - Uses HAM10000-specific normalisation (mean/std from training_history.json)
  - Test-Time Augmentation (TTA): averages 5 augmented views for higher accuracy
  - Importable as a module: from src.infer import predict

Usage:
  python src/infer.py path/to/image.jpg
  python src/infer.py path/to/image.jpg --no-tta
  python src/infer.py --batch path/to/dir/
"""

import os, json, pickle, argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH   = "models/densenet121_ft.pth"
LABELS_PATH  = "models/classes.pkl"
HISTORY_PATH = "models/training_history.json"

CLASS_INFO = {
    "akiec": "Actinic Keratosis       (pre-cancerous)",
    "bcc":   "Basal Cell Carcinoma    (most common skin cancer)",
    "bkl":   "Benign Keratosis        (seborrheic keratosis / solar lentigo)",
    "df":    "Dermatofibroma          (benign fibrous nodule)",
    "mel":   "Melanoma                (aggressive skin cancer)",
    "nv":    "Melanocytic Nevus       (common mole — usually benign)",
    "vasc":  "Vascular Lesion         (angioma / pyogenic granuloma)",
}
HIGH_RISK = {"mel", "bcc", "akiec"}

# ── Model + transform cache ─────────────────────────────────────────────────
_cache: dict = {}


def _load():
    """Load DenseNet-121, label encoder, and build transforms once."""
    if _cache:
        return _cache["model"], _cache["le"], _cache["tta"], _cache["preprocess"]

    for p in [MODEL_PATH, LABELS_PATH, HISTORY_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run:  python src/train.py")

    # Load label encoder
    with open(LABELS_PATH, "rb") as f:
        le = pickle.load(f)

    # Load norm values from training history (HAM10000-specific)
    with open(HISTORY_PATH) as f:
        h = json.load(f)
    mean, std = h["norm_mean"], h["norm_std"]

    # Load model
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, len(le.classes_))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Build TTA transforms using the training normalisation
    tta = [
        transforms.Compose([                                    # 1. Original
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([                                    # 2. H-flip
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([                                    # 3. V-flip
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([                                    # 4. 90-deg rotation
            transforms.Resize((244, 244)),
            transforms.CenterCrop(224),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([                                    # 5. Zoom-in crop
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]

    _cache.update(model=model, le=le, tta=tta, preprocess=tta[0])
    return model, le, tta, tta[0]


# ── Public API ────────────────────────────────────────────────────────────────
def predict_pil(img: Image.Image, tta: bool = True) -> dict:
    """
    Classify a skin lesion from a PIL Image (no disk I/O required).

    Returns:
        {
          "top_class":     str   — e.g. "mel"
          "top_name":      str   — full name
          "confidence":    float — probability 0-1
          "high_risk":     bool  — True for mel / bcc / akiec
          "probabilities": dict  — {class: probability} sorted desc
          "tta":           bool
        }
    """
    model, le, tta_transforms, preprocess = _load()
    img = img.convert("RGB")

    if tta:
        tensors = torch.stack([tf(img) for tf in tta_transforms])
        with torch.no_grad():
            probs = torch.softmax(model(tensors), dim=1).mean(0).tolist()
    else:
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].tolist()

    prob_dict = dict(zip(le.classes_, probs))
    top_cls   = max(prob_dict, key=prob_dict.get)

    return {
        "top_class":     top_cls,
        "top_name":      CLASS_INFO.get(top_cls, top_cls),
        "confidence":    prob_dict[top_cls],
        "high_risk":     top_cls in HIGH_RISK,
        "probabilities": dict(sorted(prob_dict.items(), key=lambda x: -x[1])),
        "tta":           tta,
    }


def predict(image_path: str, tta: bool = True) -> dict:
    """Classify a skin lesion image from a file path (convenience wrapper)."""
    return predict_pil(Image.open(image_path), tta=tta)


def predict_batch(image_dir: str, tta: bool = True) -> list:
    """Classify all .jpg/.png images in a directory."""
    exts  = {".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(image_dir)
             if os.path.splitext(f)[1].lower() in exts]
    results = []
    for fname in sorted(files):
        r = predict(os.path.join(image_dir, fname), tta=tta)
        r["file"] = fname
        results.append(r)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def _print_result(r: dict):
    W = 65
    mode = "TTA (5 views)" if r["tta"] else "single pass"
    print("\n" + "=" * W)
    print(f"  DERMASENSE — Skin Lesion Classifier  [{mode}]")
    print("-" * W)
    print(f"  Diagnosis  : {r['top_name']}")
    print(f"  Confidence : {r['confidence']:.1%}")
    risk = "*** HIGH RISK — please consult a doctor! ***" if r["high_risk"] else "Low risk"
    print(f"  Risk       : {risk}")
    print("-" * W)
    for cls, prob in r["probabilities"].items():
        bar  = "#" * int(prob * 40)
        flag = " <-- HIGH RISK" if cls in HIGH_RISK and prob > 0.05 else ""
        print(f"  {cls:5s}  {prob:5.1%}  {bar}{flag}")
    print("=" * W)
    print("  WARNING: Not a substitute for professional medical diagnosis!")
    print("=" * W + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DermaSense — Skin Lesion Inference")
    p.add_argument("input", help="Image file or directory (with --batch)")
    p.add_argument("--batch",  action="store_true")
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args()

    use_tta = not args.no_tta
    if args.batch:
        results = predict_batch(args.input, tta=use_tta)
        high = [r for r in results if r["high_risk"]]
        mode = "TTA" if use_tta else "single"
        print(f"\n  {'FILE':<45} {'CLASS':<6} {'CONF':>6}  [{mode}]")
        print("  " + "-" * 65)
        for r in results:
            flag = "[!]" if r["high_risk"] else "[ ]"
            print(f"  {flag} {r['file']:<43} {r['top_class']:<6} {r['confidence']:>5.1%}")
        print(f"\n  {len(results)} images | {len(high)} high-risk\n")
    else:
        _print_result(predict(args.input, tta=use_tta))
