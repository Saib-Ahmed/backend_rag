#!/bin/bash
set -e

echo "============================================"
echo "  RAG Backend — RunPod Serverless Startup"
echo "============================================"

# ── 1. Setup Persistent Caching Directories ─────────────────────────────
export OLLAMA_MODELS="/runpod-volume/ollama_models"
export HF_HOME="/runpod-volume/hf_cache"
export HF_HUB_DISABLE_SYMLINKS=1

echo "       Persistent Volume Ollama Cache:     $OLLAMA_MODELS"
echo "       Persistent Volume Hugging Face Cache: $HF_HOME"

mkdir -p "$OLLAMA_MODELS"
mkdir -p "$HF_HOME"

# ── 2. Start Ollama in background ──────────────────────────────────────
echo "[1/3] Starting Ollama server..."
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait until Ollama is ready (max 120 seconds)
echo "       Waiting for Ollama to become healthy..."
MAX_WAIT=120
ELAPSED=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Ollama failed to start within ${MAX_WAIT}s"
        echo "=== OLLAMA LOGS ==="
        cat /var/log/ollama.log || true
        echo "==================="
        exit 1
    fi
done
echo "       Ollama is ready! (took ${ELAPSED}s)"

# ── 3. Verify / Cache Ollama Models ────────────────────────────────────
echo "[2/3] Verifying/Caching models on persistent storage..."
MODELS=("qwen3.5:9b" "qwen3-embedding:4b" "dengcao/Qwen3-Reranker-4B:Q4_K_M")

for model in "${MODELS[@]}"; do
    if ! ollama list | grep -q "$model"; then
        echo "       → Pulling $model to persistent volume (one-time operation)..."
        ollama pull "$model"
    else
        echo "       → $model is already cached on persistent volume."
    fi
done

# ── 3. Launch all backend services ────────────────────────────────────
echo "[3/3] Starting backend services..."

cd /app

echo "       → Unified API Gateway (port 8001)"
python -m uvicorn unified_app:app --host 0.0.0.0 --port 8001 &
GATEWAY_PID=$!

echo "       → RAG V1 Engine (port 8002)"
cd /app/RAG_system
python -m uvicorn app:app --host 0.0.0.0 --port 8002 &
RAG1_PID=$!

echo "       → RAG V2 Engine (port 8003)"
cd /app
python -m uvicorn final_rag.api:app --host 0.0.0.0 --port 8003 &
RAG2_PID=$!

echo ""
echo "============================================"
echo "  All services are running!"
echo "  Gateway:  http://0.0.0.0:8001"
echo "  RAG V1:   http://0.0.0.0:8002"
echo "  RAG V2:   http://0.0.0.0:8003"
echo "  Ollama:   http://0.0.0.0:11434"
echo "============================================"

# ── Graceful shutdown handler ─────────────────────────────────────────
shutdown() {
    echo ""
    echo "Shutting down all services..."
    kill $GATEWAY_PID $RAG1_PID $RAG2_PID $OLLAMA_PID 2>/dev/null
    wait $GATEWAY_PID $RAG1_PID $RAG2_PID $OLLAMA_PID 2>/dev/null
    echo "Shutdown complete."
    exit 0
}

trap shutdown SIGTERM SIGINT

# Keep container alive — wait for any child to exit
wait -n $GATEWAY_PID $RAG1_PID $RAG2_PID $OLLAMA_PID
echo "WARNING: A service exited unexpectedly. Shutting down..."
shutdown
