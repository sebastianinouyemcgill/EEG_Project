# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — small footprint, matches typical CI images
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \       
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy requirements first so Docker cache skips re-installing on code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .

# ── Data dirs (will be mounted as volumes in production) ──────────────────────
RUN mkdir -p data/raw data/processed checkpoints mlruns

# ── Default command: train with CNN config ────────────────────────────────────
# Override at runtime:
#   docker run ... python -m src.training.evaluate --config configs/cnn.yaml
CMD ["python", "-m", "src.training.train", "--config", "configs/cnn.yaml"]
