# ── DermaSense — Training & Inference Image ───────────────────────────────────
# Model: DenseNet-121 fine-tuned on HAM10000 (7 skin lesion classes, >90% acc)
# Base:  official PyTorch image with CUDA 12.1 support.
# Falls back to CPU automatically if no GPU is available at runtime.
#
# Build:
#   docker build -t dermasense .
#
# Train (GPU):
#   docker run --gpus all -v $(pwd)/raw_data:/app/raw_data:ro \
#              -v $(pwd)/models:/app/models dermasense
#
# Infer (single image):
#   docker run -v $(pwd)/models:/app/models:ro \
#              -v $(pwd)/data:/app/data:ro \
#              dermasense python src/infer.py /app/data/image.jpg
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# System libraries needed by Pillow (JPEG/PNG decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-turbo8 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model weights into the image
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Default: run the API server
# Override with: docker run ... dermasense python src/train.py
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
