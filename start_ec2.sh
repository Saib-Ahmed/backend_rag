#!/bin/bash
set -e

echo "============================================"
echo "  RAG Backend — AWS EC2 Startup"
echo "============================================"

# ── 1. Setup Persistent Storage Root (4-tier folder hierarchy) ────────────
export STORAGE_ROOT="${STORAGE_ROOT:-/data}"

export OLLAMA_MODELS="$STORAGE_ROOT/3_models/ollama"
export HF_HOME="$STORAGE_ROOT/3_models/huggingface"
export HF_HUB_DISABLE_SYMLINKS=1
export FASTEMBED_CACHE_PATH="$STORAGE_ROOT/3_models/fastembed"

export YOLO_MODEL_PATH="$STORAGE_ROOT/3_models/yolo_layout/doclayout_yolo_docstructbench_imgsz1024.pt"
export YOLO_MODEL_PATH_V1="$STORAGE_ROOT/3_models/yolo_layout/doclayout_yolo_docstructbench_imgsz1024.pt"
export TRANSFORMER_MODEL_PATH="$STORAGE_ROOT/3_models/table_transformer"
export TRANSFORMER_MODEL_PATH_V1="$STORAGE_ROOT/3_models/table_transformer"

export QDRANT_STORAGE_PATH="$STORAGE_ROOT/4_qdrant_db"
export QDRANT_STORAGE_PATH_V1="$STORAGE_ROOT/4_qdrant_db/rag_system"
export SME_QDRANT_PATH="$STORAGE_ROOT/4_qdrant_db"

export MD_OUTPUT_DIR="$STORAGE_ROOT/1_markdown_storage/msme/rag2"
export RAG_TMP_DIR="$STORAGE_ROOT/1_markdown_storage/msme/rag1"

echo "       Storage Root:       $STORAGE_ROOT"
echo "       Ollama Models:      $OLLAMA_MODELS"
echo "       HuggingFace Cache:  $HF_HOME"
echo "       FastEmbed Cache:    $FASTEMBED_CACHE_PATH"
echo "       Qdrant Database:    $QDRANT_STORAGE_PATH"

# ── 1.1 Create all required directory tiers ──────────────────────────────
mkdir -p "$STORAGE_ROOT/1_markdown_storage/msme/rag1" "$STORAGE_ROOT/1_markdown_storage/msme/rag2" "$STORAGE_ROOT/1_markdown_storage/msme/backup"
mkdir -p "$STORAGE_ROOT/1_markdown_storage/sme/rag1" "$STORAGE_ROOT/1_markdown_storage/sme/rag2" "$STORAGE_ROOT/1_markdown_storage/sme/backup"

mkdir -p "$STORAGE_ROOT/2_pdf_storage/msme/rag1" "$STORAGE_ROOT/2_pdf_storage/msme/rag2" "$STORAGE_ROOT/2_pdf_storage/msme/backup"
mkdir -p "$STORAGE_ROOT/2_pdf_storage/sme/rag1" "$STORAGE_ROOT/2_pdf_storage/sme/rag2" "$STORAGE_ROOT/2_pdf_storage/sme/backup"

mkdir -p "$OLLAMA_MODELS" "$HF_HOME" "$FASTEMBED_CACHE_PATH"
mkdir -p "$(dirname "$YOLO_MODEL_PATH")" "$TRANSFORMER_MODEL_PATH"
mkdir -p "$QDRANT_STORAGE_PATH"

# Backward compatibility symlink for any legacy tools looking for /runpod-volume
if [ ! -d "/runpod-volume" ] && [ -w "/" ]; then
    ln -s "$STORAGE_ROOT" /runpod-volume 2>/dev/null || true
fi

# ── 1.2 Clean Stale Qdrant Lock Files ────────────────────────────────────
echo "Checking for and cleaning stale Qdrant lock files..."
find "$STORAGE_ROOT/4_qdrant_db" -name ".lock" -delete 2>/dev/null || true
find "$STORAGE_ROOT/4_qdrant_db" -name "meta.json.lock" -delete 2>/dev/null || true

# ── 1.3 Auto-heal Missing Model Files onto EBS ───────────────────────────
echo "Checking for missing model weights on EBS..."

heal_file() {
    local target_file="$1"
    local local_src="$2"
    local label="$3"
    
    if [ ! -f "$target_file" ]; then
        if [ -f "$local_src" ]; then
            echo "       → [Auto-heal] $label missing at $target_file. Copying from $local_src..."
            mkdir -p "$(dirname "$target_file")"
            cp "$local_src" "$target_file"
            echo "       → [Auto-heal] Finished copying $label."
        fi
    else
        echo "       → $label found at $target_file."
    fi
}

heal_dir() {
    local target_dir="$1"
    local local_src="$2"
    local label="$3"
    
    if [ ! -d "$target_dir" ] || [ -z "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        if [ -d "$local_src" ]; then
            echo "       → [Auto-heal] $label missing or empty at $target_dir. Copying from $local_src..."
            mkdir -p "$target_dir"
            cp -r "$local_src"/. "$target_dir"/
            echo "       → [Auto-heal] Finished copying $label."
        fi
    else
        echo "       → $label found at $target_dir."
    fi
}

heal_file "$YOLO_MODEL_PATH" "/app/final_rag/ingestion/YOLO_Layout_Model/doclayout_yolo_docstructbench_imgsz1024.pt" "YOLO Layout Model"
heal_dir "$TRANSFORMER_MODEL_PATH" "/app/final_rag/ingestion/Table_Trans_Model" "Table Transformer Model"

# ── 1.4 Verify CUDA / GPU Availability ───────────────────────────────────
echo "Checking GPU / CUDA environment..."
nvidia-smi || echo "nvidia-smi is not accessible"

# ── 2. Start Ollama Server ───────────────────────────────────────────────
echo "[1/3] Starting Ollama server..."
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "       Ollama is already running!"
    OLLAMA_PID=""
else
    ollama serve > /var/log/ollama.log 2>&1 &
    OLLAMA_PID=$!
fi

echo "       Waiting for Ollama to become healthy..."
MAX_WAIT=120
ELAPSED=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Ollama failed to start within ${MAX_WAIT}s"
        cat /var/log/ollama.log || true
        exit 1
    fi
done
echo "       Ollama is ready! (took ${ELAPSED}s)"

# ── 3. Launch Backend Services ──────────────────────────────────────────
echo "[2/3] Starting RAG backend microservices..."

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

# ── 4. Verify / Cache Ollama Models in background ────────────────────────
echo "[3/3] Checking Ollama models on EBS..."
MODELS=("qwen3.5:9b" "qwen3-embedding:4b")

pull_models() {
    for model in "${MODELS[@]}"; do
        if ! ollama list | grep -q "$model"; then
            echo "       → Pulling $model to EBS storage (one-time operation)..."
            ollama pull "$model" || echo "WARNING: Failed to pull $model"
        else
            echo "       → $model is already cached on EBS storage."
        fi
    done
    echo "       → Ollama model check complete!"
}
pull_models &
PULL_PID=$!

echo ""
echo "============================================"
echo "  All services are running on AWS EC2!"
echo "  Gateway:  http://0.0.0.0:8001"
echo "  RAG V1:   http://0.0.0.0:8002"
echo "  RAG V2:   http://0.0.0.0:8003"
echo "  Ollama:   http://0.0.0.0:11434"
echo "============================================"

# ── Graceful shutdown handler ────────────────────────────────────────────
shutdown() {
    echo ""
    echo "Shutting down services..."
    kill $GATEWAY_PID $RAG1_PID $RAG2_PID $OLLAMA_PID 2>/dev/null
    wait $GATEWAY_PID $RAG1_PID $RAG2_PID $OLLAMA_PID 2>/dev/null
    echo "Shutdown complete."
    exit 0
}

trap shutdown SIGTERM SIGINT

PIDS_TO_WAIT=("$GATEWAY_PID" "$RAG1_PID" "$RAG2_PID")
if [ -n "$OLLAMA_PID" ]; then
    PIDS_TO_WAIT+=("$OLLAMA_PID")
fi

wait -n "${PIDS_TO_WAIT[@]}"
echo "WARNING: A backend service exited unexpectedly. Shutting down..."
shutdown
