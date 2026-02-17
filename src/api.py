"""
DermaSense REST API — FastAPI service for skin lesion classification.

The image is read into memory, classified, and immediately discarded.

Run:
  uvicorn src.api:app --host 0.0.0.0 --port 8000

"""

import io
import logging
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from src.infer import predict_pil, _load, CLASS_INFO, HIGH_RISK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dermasense.api")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ── Pydantic response models ────────────────────────────────────────────────
class PredictionResult(BaseModel):
    top_class: str
    top_name: str
    confidence: float
    high_risk: bool
    probabilities: dict[str, float]
    tta: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Lifespan: warm the model on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model into memory …")
    _load()
    logger.info("Model ready.")
    yield


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermaSense API",
    version="1.0.0",
    description="Skin lesion classification powered by DenseNet-121 (HAM10000).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    from src.infer import _cache
    return HealthResponse(status="ok", model_loaded=bool(_cache))


@app.post("/predict", response_model=PredictionResult)
async def classify(
    file: UploadFile = File(..., description="Skin lesion image (JPEG, PNG, or WebP)"),
    tta: bool = True,
):
    """
    Upload a skin lesion image and receive a classification.

    - **file**: image file (max 10 MB, JPEG / PNG / WebP)
    - **tta**: enable Test-Time Augmentation for higher accuracy (default true)
    """
    t_start = time.perf_counter()
    logger.info("predict request: file=%s content_type=%s tta=%s", file.filename, file.content_type, tta)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    raw = await file.read()
    logger.info("  read %d bytes (%.1f KB)", len(raw), len(raw) / 1024)
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        img = Image.open(io.BytesIO(raw))
        logger.info("  decoded image: size=%s mode=%s", img.size, img.mode)
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Could not decode image.")

    logger.info("  running inference (tta=%s) …", tta)
    t_infer = time.perf_counter()
    result = predict_pil(img, tta=tta)
    t_infer_ms = (time.perf_counter() - t_infer) * 1000

    del raw, img

    t_total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "  done: top=%s confidence=%.1f%% high_risk=%s | inference=%.0fms total=%.0fms",
        result["top_class"], result["confidence"] * 100, result["high_risk"],
        t_infer_ms, t_total_ms,
    )

    return PredictionResult(**result)


@app.post("/predict/demo", response_model=PredictionResult)
async def classify_demo(
    file: UploadFile = File(..., description="Skin lesion image (JPEG, PNG, or WebP)"),
    tta: bool = True,
):
    """
    **Demo endpoint** — identical contract to `POST /predict` but returns
    realistic fake data without running the model.  Useful for front-end
    development and integration testing before the model is available.

    - **file**: image file (max 10 MB, JPEG / PNG / WebP) — validated but not
      passed to the model
    - **tta**: mirrored in the response so the client sees the flag it sent
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()          # confirm it is a real image
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Could not decode image.")

    del raw, img

    # ── Build a plausible fake probability distribution ───────────────────
    all_classes = list(CLASS_INFO.keys())

    # Seed from the filename so repeated calls with the same file are stable
    rng = random.Random(file.filename)

    # Pick a "winner" class and assign it a high-confidence score
    top_cls = rng.choice(all_classes)
    top_prob = rng.uniform(0.55, 0.92)

    # Distribute the remaining probability among the other classes
    remaining = 1.0 - top_prob
    cuts = sorted(rng.random() * remaining for _ in range(len(all_classes) - 2))
    cuts = [0.0] + cuts + [remaining]
    other_probs = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]

    other_classes = [c for c in all_classes if c != top_cls]
    rng.shuffle(other_classes)

    prob_dict: dict[str, float] = {top_cls: top_prob}
    for cls, p in zip(other_classes, other_probs):
        prob_dict[cls] = p

    # Sort descending by probability (mirrors real endpoint)
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

    return PredictionResult(
        top_class=top_cls,
        top_name=CLASS_INFO[top_cls],
        confidence=top_prob,
        high_risk=top_cls in HIGH_RISK,
        probabilities=prob_dict,
        tta=tta,
    )
