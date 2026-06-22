#!/bin/bash
set -e

echo "============================================"
echo "  RAG Backend — RunPod Serverless Startup"
echo "============================================"

# ── 1. Setup Persistent Caching Directories ─────────────────────────────
export OLLAMA_MODELS="/runpod-volume/ollama_models"
export HF_HOME="/runpod-volume/hf_cache"
export HF_HUB_DISABLE_SYMLINKS=1
export YOLO_MODEL_PATH="/runpod-volume/models/YOLO_Layout_Model/doclayout_yolo_docstructbench_imgsz1024.pt"
export YOLO_MODEL_PATH_V1="/runpod-volume/models/YOLO_Layout_Model/doclayout_yolo_docstructbench_imgsz1024.pt"
export TRANSFORMER_MODEL_PATH="/runpod-volume/models/Table_Trans_Model"
export TRANSFORMER_MODEL_PATH_V1="/runpod-volume/models/Table_Trans_Model"
export QDRANT_STORAGE_PATH="/runpod-volume/qdrant_db/final_rag"
export QDRANT_STORAGE_PATH_V1="/runpod-volume/qdrant_db/rag_system"
export MD_OUTPUT_DIR="/runpod-volume/md_output/final_rag"
export RAG_TMP_DIR="/runpod-volume/md_output/rag_system"

echo "       Persistent Volume Ollama Cache:     $OLLAMA_MODELS"
echo "       Persistent Volume Hugging Face Cache: $HF_HOME"

mkdir -p "$OLLAMA_MODELS"
mkdir -p "$HF_HOME"
mkdir -p "$MD_OUTPUT_DIR"
mkdir -p "$RAG_TMP_DIR"

# ── 1.5 Auto-heal Missing Models on /runpod-volume ─────────────────────
echo "Checking for missing models on persistent volume..."

heal_file() {
    local env_val="$1"
    local local_src="$2"
    local label="$3"
    
    if [ -n "$env_val" ]; then
        if [[ "$env_val" == /runpod-volume/* ]]; then
            if [ ! -f "$env_val" ]; then
                echo "       → [Auto-heal] $label missing at $env_val. Copying from $local_src..."
                mkdir -p "$(dirname "$env_val")"
                cp "$local_src" "$env_val"
                echo "       → [Auto-heal] Finished copying $label."
            else
                echo "       → $label found at $env_val."
            fi
        fi
    fi
}

heal_dir() {
    local env_val="$1"
    local local_src="$2"
    local label="$3"
    
    if [ -n "$env_val" ]; then
        if [[ "$env_val" == /runpod-volume/* ]]; then
            if [ ! -d "$env_val" ] || [ -z "$(ls -A "$env_val" 2>/dev/null)" ]; then
                echo "       → [Auto-heal] $label missing or empty at $env_val. Copying from $local_src..."
                mkdir -p "$env_val"
                cp -r "$local_src"/. "$env_val"/
                echo "       → [Auto-heal] Finished copying $label."
            else
                echo "       → $label found at $env_val."
            fi
        fi
    fi
}

heal_file "$YOLO_MODEL_PATH" "/app/final_rag/ingestion/YOLO_Layout_Model/doclayout_yolo_docstructbench_imgsz1024.pt" "YOLO V2 Model"
heal_file "$YOLO_MODEL_PATH_V1" "/app/RAG_system/new_ingestion/YOLO_Layout_Model/doclayout_yolo_docstructbench_imgsz1024.pt" "YOLO V1 Model"
heal_dir "$TRANSFORMER_MODEL_PATH" "/app/final_rag/ingestion/Table_Trans_Model" "Table Transformer V2 Model"
heal_dir "$TRANSFORMER_MODEL_PATH_V1" "/app/RAG_system/new_ingestion/Table_Trans_Model" "Table Transformer V1 Model"


# ── 2. Start Ollama in background ──────────────────────────────────────
echo "[1/3] Starting Ollama server..."
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "       Ollama is already running!"
    OLLAMA_PID=""
else
    ollama serve > /var/log/ollama.log 2>&1 &
    OLLAMA_PID=$!
fi

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

# ── 3. Launch all backend services ────────────────────────────────────
echo "[2/3] Starting backend services immediately..."

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

# ── 4. Verify / Cache Ollama Models in background ─────────────────────
echo "[3/3] Starting model caching in background..."
MODELS=("qwen3.5:9b" "qwen3-embedding:4b")

pull_models() {
    for model in "${MODELS[@]}"; do
        if ! ollama list | grep -q "$model"; then
            echo "       → Pulling $model to persistent volume (one-time operation)..."
            ollama pull "$model" || echo "WARNING: Failed to pull $model"
        else
            echo "       → $model is already cached on persistent volume."
        fi
    done
    echo "       → All background model downloads complete!"
}
pull_models &
PULL_PID=$!

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
PIDS_TO_WAIT=("$GATEWAY_PID" "$RAG1_PID" "$RAG2_PID")
if [ -n "$OLLAMA_PID" ]; then
    PIDS_TO_WAIT+=("$OLLAMA_PID")
fi

wait -n "${PIDS_TO_WAIT[@]}"
echo "WARNING: A service exited unexpectedly. Shutting down..."
shutdown
