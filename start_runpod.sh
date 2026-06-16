#!/bin/bash
set -e

echo "============================================"
echo "  RAG Backend — RunPod Serverless Startup"
echo "============================================"

# ── 1. Start Ollama in background ──────────────────────────────────────
echo "[1/3] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is ready (max 60 seconds)
echo "       Waiting for Ollama to become healthy..."
MAX_WAIT=60
ELAPSED=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Ollama failed to start within ${MAX_WAIT}s"
        exit 1
    fi
done
echo "       Ollama is ready! (took ${ELAPSED}s)"

# ── 2. Verify models are available ────────────────────────────────────
echo "[2/3] Verifying pre-pulled models..."
ollama list

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
