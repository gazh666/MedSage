# syntax=docker/dockerfile:1
# MedSage（睿医）Backend — production image
# Build:  docker build -t medsage-backend ./backend
# Run:    docker run -p 8000:8000 --env-file backend/.env medsage-backend

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libraries required by the AI/vision/audio/RAG/PDF stack:
#   ffmpeg      -> pydub / torchaudio audio decoding
#   libsm6/libxext6/libgl1/libglib2.0-0 -> opencv-python-headless
#   libgomp1    -> PyTorch OpenMP runtime
#   libpango/libcairo/libgdk-pixbuf/libffi -> weasyprint PDF generation
#   curl        -> runtime healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libpango-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source (weights/large artifacts are mounted as volumes at runtime).
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

# Runtime data/log/storage dirs (may also be bind-mounted).
RUN mkdir -p /app/logs /app/data /app/storage /app/backups

ENV PORT=8000
EXPOSE 8000

# NOTE: runs as root by default. For multi-tenant hardening, switch to a dedicated
# non-root user here (ensure volume ownership and $HOME font cache for weasyprint).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]