# ============================================================================
# RunPod Serverless Dockerfile — RAG Backend + Ollama (GPU)
# ============================================================================
# Base: NVIDIA CUDA 12.4 runtime on Ubuntu 22.04
# Includes: Ollama, Python 3.11, all RAG dependencies
# Pre-pulls Ollama models at build time so they're baked into the image.
# ============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ── Environment ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    OLLAMA_HOST=0.0.0.0:11434 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── System Dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libgl1 libglib2.0-0 \
    build-essential curl wget ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ollama ───────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python Dependencies ─────────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first (Docker layer cache)
COPY requirements.txt .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# ── Copy Application Code ───────────────────────────────────────────────────
COPY . .

# ── Startup Script ───────────────────────────────────────────────────────────
COPY start_runpod.sh /start_runpod.sh
RUN chmod +x /start_runpod.sh

# ── Expose Ports ─────────────────────────────────────────────────────────────
# 8001 = Unified API Gateway
# 8002 = RAG V1 Engine
# 8003 = RAG V2 Engine
# 11434 = Ollama API
EXPOSE 8001 8002 8003 11434

CMD ["/start_runpod.sh"]
