# AWS EC2 & S3 Deployment & Migration Report

**Project:** LexAI Backend RAG System  
**Migration Source:** RunPod Serverless  
**Migration Target:** AWS EC2 (`g4dn.xlarge` GPU) + AWS S3 (`ap-south-1`)  
**Deployment Date:** August 26, 2026  
**Status:** **Live & Operational**

---

## 1. AWS Infrastructure Specifications

| Component | Configuration | Purpose |
| :--- | :--- | :--- |
| **Instance Name / ID** | `lex_ai` (`i-0041f8c1b7fc6ec1f`) | GPU Compute Host |
| **Public HTTPS Domain** | **`https://43-204-39-6.sslip.io`** | Direct Let's Encrypt SSL (Port 443) |
| **Public IPv4 Address** | `43.204.39.6` (Static Elastic IP) | Permanent Public API Gateway Host |
| **Public DNS** | `ec2-3-110-156-73.ap-south-1.compute.amazonaws.com` | Hostname |
| **AWS Region** | `ap-south-1 (Mumbai)` | Same region for low latency & free S3 data transfer |
| **Instance Type** | `g4dn.xlarge` | 4 vCPUs, 16 GB System RAM, NVIDIA Tesla T4 (16 GB VRAM) |
| **Operating System** | Ubuntu 24.04.1 LTS (x86_64) | Deep Learning Base AMI with CUDA 13.2 |
| **AWS S3 Bucket** | `lexai-documents-storage-2026` | Cloud Object Storage for PDFs and Markdown |
| **IAM Role** | `EC2-LexAI-S3-Role` | Grants `AmazonS3FullAccess` via Instance Metadata |
| **SSH Key File** | `lex_ai.pem` (in local combine folder) | Private Key for root/ubuntu SSH access |

---

## 2. Server Connection & Access Commands

### SSH Login
From PowerShell / Terminal:
```powershell
ssh -i "c:\Users\welcome\Downloads\combine\lex_ai.pem" ubuntu@3.110.156.73
```

### Essential Management Commands on EC2:
```bash
# Navigate to project
cd /home/ubuntu/app

# View live container logs
docker compose logs -f

# Check GPU status and VRAM utilization
nvidia-smi

# Check running microservices
docker compose ps

# Restart backend container
docker compose restart

# View Ollama models on EBS
docker exec combine_rag_backend ollama list
```

---

## 3. Persistent 4-Tier Storage Layout on EBS (`/data`)

The storage architecture on the persistent EBS disk is organized as follows:

```text
/data/
│
├── 📁 1_markdown_storage/                  [Markdown files + S3 Sync]
│   ├── 📁 msme/
│   │   ├── 📂 rag1/                       --> RAG V1 parsed markdowns
│   │   ├── 📂 rag2/                       --> RAG V2 parsed markdowns
│   │   └── 📂 backup/                     --> Master unedited markdown archive
│   └── 📁 sme/
│       ├── 📂 rag1/                       --> SME RAG V1 parsed markdowns
│       ├── 📂 rag2/                       --> SME RAG V2 parsed markdowns
│       └── 📂 backup/                     --> SME master markdown archive
│
├── 📁 2_pdf_storage/                       [Raw PDF documents + S3 Sync]
│   ├── 📁 msme/
│   │   ├── 📂 rag1/                       --> MSME RAG V1 document PDFs
│   │   ├── 📂 rag2/                       --> MSME RAG V2 document PDFs
│   │   └── 📂 backup/                     --> MSME original uploaded PDFs
│   └── 📁 sme/
│       ├── 📂 rag1/                       --> SME RAG V1 document PDFs
│       ├── 📂 rag2/                       --> SME RAG V2 document PDFs
│       └── 📂 backup/                     --> SME original uploaded PDFs
│
├── 📁 3_models/                            [Cached on EBS for fast GPU boot]
│   ├── 📂 ollama/                         --> qwen3.5:9b (6.6 GB) & qwen3-embedding:4b (2.5 GB)
│   ├── 📂 huggingface/                    --> Qwen3-Reranker-0.6B weights cache
│   ├── 📂 fastembed/                      --> FastEmbed BM25 sparse model cache
│   ├── 📂 yolo_layout/                    --> doclayout_yolo_docstructbench_imgsz1024.pt
│   └── 📂 table_transformer/              --> Table_Trans_Model weights
│
└── 📁 4_qdrant_db/                         [Single unified Qdrant Database]
    ├── (Collection: msme_documents)
    └── (Collection: sme_documents)
```

---

## 4. Codebase Components & Changes

### 1. `s3_service.py` (New)
* Complete AWS S3 helper built with `boto3`.
* Handles file uploads, byte uploads, pre-signed URL generation (for frontend PDF viewing), and byte streaming.
* Automatically uses the EC2 IAM Role (`EC2-LexAI-S3-Role`) without hardcoded credentials.

### 2. `start_ec2.sh` (New)
* Startup entrypoint script for Docker.
* Creates all directory tiers in `/data`.
* Cleans stale Qdrant lock files (`.lock`, `meta.json.lock`).
* Auto-heals missing model weights from repository to `/data/3_models`.
* Initializes Ollama server in background and pre-caches `qwen3.5:9b` and `qwen3-embedding:4b`.
* Concurrently launches:
  * Unified API Gateway (`8001`)
  * RAG V1 Engine (`8002`)
  * RAG V2 Engine (`8003`)

### 3. `unified_app.py`
* Replaced hardcoded `/runpod-volume` references with configurable `STORAGE_ROOT` (`/data`).
* Integrated S3 uploading into `backup_pdf_file` and `backup_markdown_file`.
* Stored `s3_key` in MongoDB `document_metadata`.
* Added S3 fallback streaming in `/documents/{file_name}/pdf` route.

### 4. `final_rag/config.py`
* Updated default directories for `MD_OUTPUT_DIR`, `PDF_STORAGE_DIR`, `QDRANT_STORAGE_PATH`, `YOLO_MODEL_PATH`, and `TRANSFORMER_MODEL_PATH` to point to `/data`.

### 5. `Dockerfile` & `docker-compose.yml`
* Updated to use `start_ec2.sh`.
* Configured Docker Compose with NVIDIA GPU reservation and `/data` volume binding.

### 6. `requirements.txt`
* Added `boto3>=1.34.0` and `botocore>=1.34.0`.

---

## 5. Live Service Endpoints

| Service | Port | Endpoint URL | Status |
| :--- | :--- | :--- | :--- |
| **Unified API Gateway** | `8001` | `http://3.110.156.73:8001/ping` | 🟢 **HTTP 200 `{"status":"ok"}`** |
| **PDF Path Inspector** | `8001` | `http://3.110.156.73:8001/debug/pdf-paths` | 🟢 **Verified** |
| **RAG V1 Engine** | `8002` | `http://3.110.156.73:8002/` | 🟢 **Running** |
| **RAG V2 Engine** | `8003` | `http://3.110.156.73:8003/` | 🟢 **Running** |
| **Ollama API** | `11434` | `http://3.110.156.73:11434/api/tags` | 🟢 **Running (`qwen3.5:9b`, `qwen3-embedding:4b`)** |

---

## 6. Server Environment Configuration (`/home/ubuntu/app/.env`)

```env
# Storage Root on EBS
STORAGE_ROOT=/data

# AWS S3 Storage
S3_BUCKET_NAME=lexai-documents-storage-2026
AWS_REGION=ap-south-1

# Security
JWT_SECRET=b63c89df9012a45e9821ef34ba8921c563e41209bca7892341eec780912ab45e
ADMIN_EMAIL=admin@lexai.com
ADMIN_SEED_PASSWORD=AdminSecurePass2026!

# CORS
ALLOWED_ORIGINS=*

# Optional: Remote MongoDB Atlas URI
# MONGO_URI=mongodb+srv://<user>:<password>@cluster0.mongodb.net/rag_database?retryWrites=true&w=majority

# Optional: External LLM & Grounding API Keys
# CLAUDE_API_KEY=sk-ant-api...
# GEMINI_API_KEY=AIzaSy...
# NVIDIA_API_KEY=nvapi-...
```

---

## 7. Frontend Integration (`lexAI`)

To connect the mobile or web frontend (`lexAI`) to your new backend on EC2, set:

```env
EXPO_PUBLIC_API_URL=http://3.110.156.73:8001
```
