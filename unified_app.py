import sys
import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, BackgroundTasks, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import requests
import tempfile
import subprocess
import speech_recognition as sr
import imageio_ffmpeg
from jose import JWTError, jwt
from datetime import datetime, timedelta
from cachetools import TTLCache

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from msme_extractor import MsmeExtractor

from unified_db import (
    create_user, verify_password, get_user, create_session,
    get_user_sessions, append_message, get_chat_history,
    get_chat_history_formatted_for_llm, update_session_title, delete_session,
    save_document_metadata, get_all_document_metadata, delete_document_metadata,
    update_document_metadata,
)

app = FastAPI(title="Unified RAG API")


@app.on_event("startup")
def startup_event():
    import torch
    try:
        logging.info("--- GPU DIAGNOSTICS AT API STARTUP ---")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            logging.warning("CUDA is NOT available in python environment at startup!")
        logging.info("--------------------------------------")
    except Exception as diag_e:
        logging.error(f"Failed to run GPU diagnostics in API startup: {diag_e}")



@app.get("/ping")
def ping():
    return {"status": "ok"}


class _ExcludeUploadStatusAccessLog(logging.Filter):
    """Hide high-frequency polling access logs for /upload/status/* endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn access logger usually keeps request path in record.args[2].
        try:
            if isinstance(record.args, tuple) and len(record.args) >= 3:
                path = str(record.args[2])
                if "/upload/status/" in path:
                    return False
        except Exception:
            pass

        # Fallback for formatted-message based handlers.
        return "/upload/status/" not in record.getMessage()


def _configure_access_log_filters() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _ExcludeUploadStatusAccessLog) for f in access_logger.filters):
        access_logger.addFilter(_ExcludeUploadStatusAccessLog())


_configure_access_log_filters()

# Global task registry for background uploads
# Format: { "task_id": {"status": "processing" | "success" | "failed" | "already_exists", "data": {...}, "error": "..."} }
upload_tasks = TTLCache(maxsize=1000, ttl=3600)
msme_upload_tasks = TTLCache(maxsize=1000, ttl=3600)

# Ensure backup_markdown directory exists (immutable archive)
# If running on RunPod serverless, store inside the persistent network volume at /runpod-volume
RUNPOD_VOLUME = "/runpod-volume"
if os.path.exists(RUNPOD_VOLUME) and os.access(RUNPOD_VOLUME, os.W_OK):
    BACKUP_MD_DIR = os.path.join(RUNPOD_VOLUME, "backup_markdown")
    BACKUP_PDF_DIR = os.path.join(RUNPOD_VOLUME, "backup_pdf")
else:
    BACKUP_MD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_markdown")
    BACKUP_PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_pdf")
os.makedirs(BACKUP_MD_DIR, exist_ok=True)
os.makedirs(BACKUP_PDF_DIR, exist_ok=True)

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "https://lexai-proxy.suyashjai2010.workers.dev,http://localhost:8081,http://localhost:19006"
)
ALLOWED_ORIGINS_LIST = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_request_source(request: Request, call_next):
    path = request.url.path
    if path in ("/ping", "/stats"):
        x_forwarded_for = request.headers.get("x-forwarded-for")
        x_real_ip = request.headers.get("x-real-ip")
        user_agent = request.headers.get("user-agent")
        client_host = request.client.host if request.client else "unknown"
        logging.info(
            "[Request Trace] Path: %s | Client: %s | X-Forwarded-For: %s | X-Real-IP: %s | UA: %s",
            path, client_host, x_forwarded_for, x_real_ip, user_agent
        )
    response = await call_next(request)
    return response

# ─── JWT Configuration ───────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET environment variable is not set. "
        "Generate one with: openssl rand -hex 32"
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload.update({"exp": expire})
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None

security = HTTPBearer(auto_error=False)

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    payload = decode_access_token(credentials.credentials)
    return payload

def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """Strict auth guard — raises HTTP 401 if token is missing or invalid."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

# ── Global Auth Enforcement Middleware ────────────────────────────────────────────
# Paths that are accessible without a JWT token
_PUBLIC_PATHS = {"/ping", "/stats", "/docs", "/openapi.json", "/redoc", "/debug/pdf-paths"}

@app.middleware("http")
async def enforce_authentication(request: Request, call_next):
    """Block every non-public route that does not carry a valid JWT Bearer token."""
    path = request.url.path
    # Allow OPTIONS preflight, public endpoints, and all /auth/* routes
    if request.method == "OPTIONS" or path in _PUBLIC_PATHS or path.startswith("/auth/"):
        return await call_next(request)
    auth_header = request.headers.get("Authorization", "")
    x_auth = request.headers.get("x-jwt-authorization", "")
    token = None
    
    # 1. Prioritize x-jwt-authorization header set by proxy worker
    if x_auth.startswith("Bearer "):
        token = x_auth.split(" ", 1)[1].strip()
    
    # 2. Check query parameter token for browser PDF streaming requests
    if not token and path.endswith("/pdf"):
        token = request.query_params.get("token")
        
    # 3. Fall back to standard Authorization header only if x-jwt-authorization was not provided
    if not token and not x_auth and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()

    if not token:
        return JSONResponse(status_code=401, content={"detail": "Authentication required"})
    payload = decode_access_token(token)
    if not payload:
        return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})
    # Attach verified user to request state so route handlers can read it
    request.state.current_user = payload
    return await call_next(request)
# ─────────────────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str = Field(..., max_length=8000)
    model: str = "v1"  # "v1" (RAG_system) or "v2" (final_rag)
    user_id: Optional[str] = None  # Ignored — server derives user_id from the verified JWT
    use_claude: bool = False

_ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")

@app.post("/auth/register")
def register(req: RegisterRequest):
    user_id = create_user(req.username, req.email, req.password)
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or Email already exists")
    role = "admin" if (_ADMIN_EMAIL and req.email == _ADMIN_EMAIL) else "user"
    token = create_access_token({"sub": user_id, "username": req.username, "email": req.email, "role": role})
    return {"user_id": user_id, "username": req.username, "email": req.email, "role": role, "access_token": token, "token_type": "bearer"}

@app.post("/auth/login")
def login(req: AuthRequest):
    user = get_user(req.username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    role = user.get("role", "user")  # Role is read from DB — not determined by email string comparison
    token = create_access_token({"sub": user["user_id"], "username": user["username"], "email": user.get("email"), "role": role})
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user.get("email"),
        "role": role,
        "access_token": token,
        "token_type": "bearer"
    }

@app.get("/auth/verify")
def verify_token(current_user: Optional[dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"valid": True, "user_id": current_user.get("sub"), "username": current_user.get("username"), "role": current_user.get("role")}

@app.get("/sessions")
def get_sessions(request: Request):
    user_id = request.state.current_user["sub"]
    sessions = get_user_sessions(user_id)
    return [{"id": s["session_id"], "title": s["title"]} for s in sessions]

@app.get("/sessions/{session_id}/history")
def get_history(session_id: str, request: Request):
    current_user = request.state.current_user
    user_id = current_user["sub"]
    if not any(s["session_id"] == session_id for s in get_user_sessions(user_id)):
        raise HTTPException(status_code=403, detail="Forbidden: Session does not belong to you")
    history = get_chat_history(session_id)
    messages = [
        {
            "role": msg["role"],
            "text": msg["content"],
            "sources": msg.get("sources", [])
        }
        for msg in history
    ]
    return {"session_id": session_id, "messages": messages}

class RenameSessionRequest(BaseModel):
    title: str

@app.patch("/sessions/{session_id}/title")
def rename_session_route(session_id: str, req: RenameSessionRequest, request: Request):
    current_user = request.state.current_user
    user_id = current_user["sub"]
    if not any(s["session_id"] == session_id for s in get_user_sessions(user_id)):
        raise HTTPException(status_code=403, detail="Forbidden: Session does not belong to you")
    update_session_title(session_id, req.title)
    return {"status": "success"}

@app.delete("/sessions/{session_id}")
def delete_session_route(session_id: str, request: Request):
    current_user = request.state.current_user
    user_id = current_user["sub"]
    if not any(s["session_id"] == session_id for s in get_user_sessions(user_id)):
        raise HTTPException(status_code=403, detail="Forbidden: Session does not belong to you")
    delete_session(session_id)
    return {"status": "success"}

@app.get("/stats")
def get_combined_stats():
    stats = {
        "v1": {"status": "offline", "num_chunks": 0},
        "v2": {"status": "offline", "num_chunks": 0, "graph": {"entities": 0, "relationships": 0}},
        "hardware": {
            "type": "CPU",
            "name": "CPU"
        }
    }
    try:
        import torch
        if torch.cuda.is_available():
            stats["hardware"] = {
                "type": "GPU",
                "name": torch.cuda.get_device_name(0)
            }
    except Exception as e:
        logging.warning(f"GPU stats failure: {e}")
    try:
        res = requests.get("http://127.0.0.1:8002/api/stats", timeout=15)
        if res.status_code == 200:
            stats["v1"] = res.json()
    except Exception as e:
        logging.warning(f"V1 Stats Error: {e}")
        pass
    
    try:
        res = requests.get("http://127.0.0.1:8003/api/stats", timeout=15)
        if res.status_code == 200:
            stats["v2"] = res.json()
    except Exception as e:
        logging.warning(f"V2 Stats Error: {e}")
        pass
    
    return stats

@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    user_id = request.state.current_user["sub"]
    session_id = req.session_id
    if not session_id:
        title = req.query[:30] + "..." if len(req.query) > 30 else req.query
        session_id = create_session(user_id, title)

    append_message(session_id, "user", req.query, req.model, user_id=user_id)

    def generate_v1():
        # Proxy to RAG_system (port 8002) with true real-time SSE streaming
        try:
            res = requests.post("http://127.0.0.1:8002/api/query/stream", json={
                "question": req.query,
                "session_id": session_id,
                "use_live_search": True,
                "enable_graph_rag": False,
                "history_limit": 6
            }, stream=True, timeout=900)
            
            for line in res.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    yield decoded + "\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def generate_v2():
        # Proxy to final_rag (port 8003)
        try:
            res = requests.post("http://127.0.0.1:8003/chat/stream", json={
                "query": req.query,
                "session_id": session_id,
                "use_claude": req.use_claude or (req.model == "cloud"),
                "model": req.model
            }, stream=True, timeout=900)
            
            full_answer = []
            for line in res.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    yield decoded + "\n\n"
                    if decoded.startswith("data: "):
                        try:
                            data = json.loads(decoded.replace("data: ", ""))
                            if "token" in data:
                                full_answer.append(data["token"])
                        except:
                            pass
            
            complete_answer = "".join(full_answer).strip()
            # We don't call append_message here because final_rag/api.py already saves the assistant message with metadata
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # In UI, models are 'version1' and 'version2'
    generator = generate_v1() if req.model == "version1" else generate_v2()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

def backup_markdown_file(filename: str, rag_version: str):
    try:
        import shutil
        from pathlib import Path
        stem = Path(filename).stem
        md_source = None

        if rag_version in ["version2", "v2"]:
            env_dir = os.environ.get("MD_OUTPUT_DIR")
            if env_dir and os.path.exists(os.path.join(env_dir, f"{stem}.md")):
                md_source = os.path.join(env_dir, f"{stem}.md")
            else:
                candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_rag", "md_output", f"{stem}.md")
                if os.path.exists(candidate):
                    md_source = candidate
        else:
            import re
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
            env_dir = os.environ.get("RAG_TMP_DIR")
            if env_dir and os.path.exists(os.path.join(env_dir, f"{safe_name}_extraction.md")):
                md_source = os.path.join(env_dir, f"{safe_name}_extraction.md")
            else:
                candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG_system", "tmp", f"{safe_name}_extraction.md")
                if os.path.exists(candidate):
                    md_source = candidate

        if md_source:
            v_suffix = "_v2" if rag_version in ["version2", "v2"] else "_v1"
            backup_dest = os.path.join(BACKUP_MD_DIR, f"{stem}{v_suffix}.md")
            # If file already exists in backup, add timestamp suffix to keep both
            if os.path.exists(backup_dest):
                from datetime import datetime
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                backup_dest = os.path.join(BACKUP_MD_DIR, f"{stem}{v_suffix}_{ts}.md")
            shutil.copy2(md_source, backup_dest)
            logging.info(f"Backed up md to: {backup_dest}")
        else:
            logging.warning(f"No .md source file found for backup of {filename}")
    except Exception as backup_err:
        logging.error(f"Failed to create backup for {filename}: {backup_err}")

def backup_pdf_file(filename: str, file_content: bytes) -> str:
    """Save original PDF to backup_pdf directory and return the backup file path."""
    try:
        pdf_dest = os.path.join(BACKUP_PDF_DIR, filename)
        with open(pdf_dest, "wb") as f:
            f.write(file_content)
        logging.info(f"Backed up PDF to: {pdf_dest}")
        return pdf_dest
    except Exception as err:
        logging.error(f"Failed to backup PDF {filename}: {err}")
        return ""

def process_upload_background(task_id: str, filename: str, file_content: bytes, content_type: str, buildGraph: bool, rag_version: str,
                               doc_type: str = "PDF", source: str = "public", source_description: str = "", creation_date: str = ""):
    try:
        logging.info(f"[Background Task {task_id}] Started for file: {filename}")
        if rag_version in ["version2", "v2"]:
            target_url = "http://127.0.0.1:8003/upload"
            files_payload = {"file": (filename, file_content, content_type)}
            res = requests.post(target_url, files=files_payload, timeout=900)
        else:
            target_url = "http://127.0.0.1:8002/api/ingest"
            files_payload = {"files": (filename, file_content, content_type)}
            data_payload = {
                "parsing_mode": "SMART",
                "build_graph": str(buildGraph).lower()
            }
            res = requests.post(target_url, files=files_payload, data=data_payload, timeout=900)
        
        logging.info(f"[Background Task {task_id}] Downstream response: status={res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            if data.get("total_chunks") == 0:
                upload_tasks[task_id] = {"status": "already_exists", "data": data}
            else:
                upload_tasks[task_id] = {"status": "success", "data": data}

                # ── Save PDF to backup_pdf ──
                pdf_backup_path = backup_pdf_file(filename, file_content)

                # ── Save metadata to MongoDB ──
                try:
                    import hashlib
                    file_hash = hashlib.md5(file_content).hexdigest()
                    save_document_metadata(
                        file_name=filename,
                        doc_type=doc_type,
                        source=source,
                        source_description=source_description,
                        creation_date=creation_date,
                        rag_version=rag_version,
                        file_hash=file_hash,
                        pdf_path=f"doc_input/{filename}" if rag_version in ["v2", "version2"] else f"pdf_storage/{filename}",
                        pdf_backup_path=pdf_backup_path,
                        s3_key=None,
                    )
                except Exception as meta_err:
                    logging.error(f"[Background Task {task_id}] Failed to save metadata: {meta_err}")

                # ── Copy .md to backup_markdown/ (immutable archive) ──
                backup_markdown_file(filename, rag_version)
        else:
            logging.error(f"[Background Task {task_id}] Downstream error: status={res.status_code} body={res.text[:500]}")
            upload_tasks[task_id] = {"status": "failed", "error": f"Backend returned {res.status_code}: {res.text}"}
            
    except requests.exceptions.Timeout:
        logging.error(f"[Background Task {task_id}] Downstream request timed out after 900s for {filename}")
        upload_tasks[task_id] = {"status": "failed", "error": "Ingestion timed out after 15 minutes."}
    except requests.exceptions.ConnectionError as e:
        logging.error(f"[Background Task {task_id}] Cannot reach downstream service: {e}")
        upload_tasks[task_id] = {"status": "failed", "error": f"Backend service unreachable: {e}"}
    except Exception as e:
        logging.error(f"[Background Task {task_id}] Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        upload_tasks[task_id] = {"status": "failed", "error": str(e)}

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard cap
_ALLOWED_MIME_TYPES = {
    "application/pdf", "text/plain", "text/markdown",
    "image/jpeg", "image/png", "image/tiff", "image/webp",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

@app.post("/upload")
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    buildGraph: bool = Form(False),
    rag_version: str = Form("v1"),
    doc_type: str = Form("PDF"),
    source: str = Form("public"),
    source_description: str = Form(""),
    creation_date: str = Form(""),
    request: Request = None,
):
    current_user = request.state.current_user
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Only admins can upload documents")
    try:
        # Validate MIME type against allowlist
        mime = (file.content_type or "").split(";")[0].strip()
        if mime and mime not in _ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{mime}'. Allowed: PDF, plain text, images, Word documents."
            )
        # Read with hard size cap to prevent memory exhaustion
        file_content = file.file.read(_MAX_UPLOAD_BYTES + 1)
        if len(file_content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 50 MB.")
        task_id = str(uuid.uuid4())
        
        # Register the task
        upload_tasks[task_id] = {"status": "processing"}
        
        # Dispatch to background thread
        background_tasks.add_task(
            process_upload_background, 
            task_id, 
            file.filename, 
            file_content, 
            file.content_type, 
            buildGraph, 
            rag_version,
            doc_type,
            source,
            source_description,
            creation_date,
        )
        
        logging.info(f"[Upload] Dispatched background task {task_id} for file: {file.filename}")
        
        return {"status": "processing", "task_id": task_id, "message": "Document ingestion started in background"}
    except Exception as e:
        logging.error(f"[Upload Init] Failed to start upload task: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/upload/status/{task_id}")
def get_upload_status(task_id: str):
    if task_id not in upload_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # You could optionally pop the task from the dictionary once it's success/failed 
    # to free memory, but keeping it allows the client to retry fetching status if needed.
    return upload_tasks[task_id]

@app.get("/documents/search")
def search_documents_route(q: str = Query(..., min_length=1), rag_version: str = Query("version1")):
    try:
        if rag_version == "version1":
            target_url = "http://127.0.0.1:8002/api/documents/search"
        else:
            target_url = "http://127.0.0.1:8003/api/documents/search"
            
        res = requests.get(target_url, params={"q": q}, timeout=90)
        
        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend proxy error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/documents/{file_name}/content")
def get_document_content_route(file_name: str, rag_version: str = Query("version1")):
    try:
        import urllib.parse
        encoded_name = urllib.parse.quote(file_name)
        if rag_version == "version1":
            target_url = f"http://127.0.0.1:8002/api/documents/{encoded_name}/content"
        else:
            target_url = f"http://127.0.0.1:8003/api/documents/{encoded_name}/content"
            
        res = requests.get(target_url, timeout=90)
        
        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend proxy error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/documents/{file_name}/pdf")
def get_document_pdf_route(file_name: str, page: int = Query(None)):
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.resolve()
        import os
        import urllib.parse
        import fitz
        import io
        from fastapi.responses import StreamingResponse

        search_names = {
            file_name,
            file_name.replace(" ", "_"),
            file_name.replace("_", " "),
            file_name.replace("..", ".")
        }
        extra_variations = list(search_names)
        for name in extra_variations:
            search_names.add(name.replace("..", "."))
            search_names.add(name.replace(" ", "_"))
            search_names.add(name.replace("_", " "))
            search_names.add(urllib.parse.quote(name))
            search_names.add(urllib.parse.unquote(name))

        directories = [
            Path(BACKUP_PDF_DIR),
            Path(RUNPOD_VOLUME) / "pdf_storage",
            Path(RUNPOD_VOLUME) / "backup_pdf",
            base_dir / "backup_pdf",
            base_dir / "final_rag" / "doc_input",
            base_dir / "final_rag" / "pdf_storage",
            base_dir / "RAG_system" / "pdf_storage",
        ]
        for d in directories:
            if not d.exists():
                continue
            try:
                dir_files = os.listdir(d)
                for f in dir_files:
                    f_lower = f.lower()
                    for target in search_names:
                        if f_lower == target.lower():
                            target_path = d / f
                            if target_path.is_file():
                                # If page parameter is supplied, extract and return only that page
                                if page is not None and page > 0:
                                    try:
                                        doc = fitz.open(str(target_path))
                                        page_num = page - 1  # convert to 0-indexed
                                        if 0 <= page_num < len(doc):
                                            new_doc = fitz.open()
                                            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                                            pdf_bytes = new_doc.write()
                                            new_doc.close()
                                            doc.close()
                                            return StreamingResponse(
                                                io.BytesIO(pdf_bytes),
                                                media_type="application/pdf",
                                                headers={"Content-Disposition": f"inline; filename=\"page_{page}_{f}\""}
                                            )
                                        else:
                                            doc.close()
                                            raise HTTPException(status_code=400, detail=f"Page {page} is out of range for PDF '{f}' (total pages: {len(doc)}).")
                                    except HTTPException:
                                        raise
                                    except Exception as page_err:
                                        logging.error(f"Error extracting page {page} from {f}: {page_err}")
                                        raise HTTPException(status_code=500, detail=f"Failed to extract page {page}: {page_err}")
                                
                                return FileResponse(
                                    path=str(target_path),
                                    media_type="application/pdf",
                                    filename=f,
                                    headers={"Content-Disposition": f"inline; filename=\"{f}\""}
                                )
            except Exception as dir_err:
                logging.warning(f"Error reading directory {d} for PDF search: {dir_err}")

        raise HTTPException(status_code=404, detail=f"PDF file '{file_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/pdf-paths")
def debug_pdf_paths():
    from pathlib import Path
    base_dir = Path(__file__).parent.resolve()
    directories = {
        "BACKUP_PDF_DIR": BACKUP_PDF_DIR,
        "runpod_volume_pdf_storage": str(Path(RUNPOD_VOLUME) / "pdf_storage"),
        "runpod_volume_backup_pdf": str(Path(RUNPOD_VOLUME) / "backup_pdf"),
        "backup_pdf": str(base_dir / "backup_pdf"),
        "doc_input": str(base_dir / "final_rag" / "doc_input"),
        "final_rag_pdf_storage": str(base_dir / "final_rag" / "pdf_storage"),
        "RAG_system_pdf_storage": str(base_dir / "RAG_system" / "pdf_storage"),
    }
    results = {}
    for name, path_str in directories.items():
        p = Path(path_str)
        exists = p.exists()
        is_dir = p.is_dir() if exists else False
        files = []
        if exists and is_dir:
            try:
                files = os.listdir(p)
            except Exception as e:
                files = [f"Error listing: {e}"]
        results[name] = {
            "path": path_str,
            "exists": exists,
            "is_dir": is_dir,
            "file_count": len(files),
            "files": files[:100]  # Show up to 100 files for debugging
        }
        
    return results

class DocumentContentUpdateUnified(BaseModel):
    content: str

@app.put("/documents/{file_name}/content")
def update_document_content_route(file_name: str, req: DocumentContentUpdateUnified, request: Request, rag_version: str = Query("version1")):
    current_user = request.state.current_user
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Only admins can modify documents")
    try:
        import urllib.parse
        encoded_name = urllib.parse.quote(file_name)
        if rag_version == "version1":
            target_url = f"http://127.0.0.1:8002/api/documents/{encoded_name}/content"
        else:
            target_url = f"http://127.0.0.1:8003/api/documents/{encoded_name}/content"
            
        res = requests.put(target_url, json={"content": req.content}, timeout=300)
        
        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend proxy error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.delete("/documents/{file_name}")
def delete_document_route(file_name: str, request: Request, rag_version: str = Query("version1")):
    current_user = request.state.current_user
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Only admins can delete documents")
    try:
        # Determine target URL based on rag_version
        import urllib.parse
        encoded_name = urllib.parse.quote(file_name)
        if rag_version == "version1":
            target_url = f"http://127.0.0.1:8002/api/documents/{encoded_name}"
        else:
            target_url = f"http://127.0.0.1:8003/documents/{encoded_name}"
            
        res = requests.delete(target_url, timeout=90)
        
        # Also delete metadata from MongoDB (but NOT from backup_markdown)
        try:
            delete_document_metadata(file_name)
        except Exception as meta_err:
            logging.error(f"Failed to delete metadata for '{file_name}': {meta_err}")
        
        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/documents/metadata")
def get_documents_metadata_route():
    """Return metadata for all ingested documents."""
    try:
        return get_all_document_metadata()
    except Exception as e:
        logging.error(f"Failed to get document metadata: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


class UpdateDocumentMetadataRequest(BaseModel):
    doc_type: Optional[str] = None
    source: Optional[str] = None
    source_description: Optional[str] = None
    creation_date: Optional[str] = None
    ingestion_date: Optional[str] = None
    rag_version: Optional[str] = None


@app.put("/documents/{file_name}/metadata")
def update_document_metadata_route(file_name: str, req: UpdateDocumentMetadataRequest):
    try:
        success = update_document_metadata(
            file_name=file_name,
            doc_type=req.doc_type,
            source=req.source,
            source_description=req.source_description,
            creation_date=req.creation_date,
            ingestion_date=req.ingestion_date,
            rag_version=req.rag_version,
        )
        return {"status": "success" if success else "no_change"}
    except Exception as e:
        logging.error(f"Failed to update metadata for '{file_name}': {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.delete("/documents")
def clear_all_documents_route(rag_version: str = Query("version1")):
    try:
        if rag_version == "version1":
            target_url = "http://127.0.0.1:8002/api/documents"
        else:
            target_url = "http://127.0.0.1:8003/api/documents"
            
        res = requests.delete(target_url, timeout=120)
        
        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.post("/documents/preview-parse")
async def preview_parse_route(
    file: UploadFile = File(...),
    rag_version: str = Query("version1"),
):
    """Proxy: parse a document to markdown without saving."""
    try:
        file_bytes = await file.read()
        if rag_version == "version1":
            target_url = "http://127.0.0.1:8002/api/documents/preview-parse"
        else:
            target_url = "http://127.0.0.1:8003/api/documents/preview-parse"

        files_payload = {"file": (file.filename, file_bytes, file.content_type)}
        res = requests.post(target_url, files=files_payload, timeout=900)

        if res.status_code == 200:
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend proxy error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.post("/documents/replace")
async def replace_document_route(
    file: UploadFile = File(...),
    old_file_name: str = Form(...),
    rag_version: str = Query("version1"),
):
    """Proxy: delete old document and ingest new one."""
    try:
        file_bytes = await file.read()
        if rag_version == "version1":
            target_url = "http://127.0.0.1:8002/api/documents/replace"
        else:
            target_url = "http://127.0.0.1:8003/api/documents/replace"

        files_payload = {"file": (file.filename, file_bytes, file.content_type)}
        data_payload = {"old_file_name": old_file_name}
        res = requests.post(target_url, files=files_payload, data=data_payload, timeout=900)

        if res.status_code == 200:
            # ── Update MongoDB Metadata & Create Backup ──
            try:
                from unified_db import get_document_metadata, delete_document_metadata, save_document_metadata
                
                old_meta = get_document_metadata(old_file_name) or {}
                doc_type = old_meta.get("doc_type", "PDF" if file.filename.lower().endswith(".pdf") else "Text")
                source = old_meta.get("source", "public")
                source_description = old_meta.get("source_description", "")
                creation_date = old_meta.get("creation_date", "")
                
                if old_file_name != file.filename:
                    delete_document_metadata(old_file_name)
                    
                save_document_metadata(
                    file_name=file.filename,
                    doc_type=doc_type,
                    source=source,
                    source_description=source_description,
                    creation_date=creation_date,
                    rag_version=rag_version
                )
                
                # Backup the new replaced file markdown
                backup_markdown_file(file.filename, rag_version)
            except Exception as meta_err:
                logging.error(f"Failed to update metadata/backup during file replacement: {meta_err}")
                
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend proxy error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/query_trace")
def get_query_trace_route(rag_version: str = Query("version1")):
    try:
        if rag_version == "version1":
            target_url = "http://127.0.0.1:8002/api/query_trace"
            res = requests.get(target_url, timeout=90)
            if res.status_code == 200:
                return res.json()
            else:
                return {"status": "error", "trace": {"stages": []}, "metrics": {}}
        else:
            # Version 2 doesn't support complex tracing yet
            return {"status": "success", "trace": {"stages": []}, "metrics": {}}
    except Exception as e:
        return {"status": "error", "trace": {"stages": []}, "metrics": {}}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    tmp_in_path = ""
    tmp_out_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_in:
            tmp_in.write(await file.read())
            tmp_in_path = tmp_in.name
            
        tmp_out_path = tmp_in_path.replace(".m4a", ".wav")
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, "-y", "-i", tmp_in_path, tmp_out_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_out_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path): os.remove(tmp_out_path)
        
        return {"text": text}
    except sr.UnknownValueError:
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path): os.remove(tmp_out_path)
        return {"text": ""}
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        if os.path.exists(tmp_in_path): os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path): os.remove(tmp_out_path)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

# /debug/ollama endpoint removed — it exposed the full system process list (ps aux)
# and system logs without any authentication. Do not restore in production.

# ==========================================
# MSME Extraction Endpoints
# ==========================================

class MsmeChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    user_id: Optional[str] = None  # Ignored — server derives user_id from the verified JWT


@app.post("/chat/msme")
def chat_msme_stream(req: MsmeChatRequest, request: Request):
    """Streaming MSME extraction from text (voice transcript or typed description)."""
    user_id = request.state.current_user["sub"]
    session_id = req.session_id
    if not session_id:
        title = req.query[:30] + "..." if len(req.query) > 30 else req.query
        session_id = create_session(user_id, title)

    append_message(session_id, "user", req.query, "msme", user_id=user_id)

    def generate():
        try:
            nl = "\n\n"
            data_init = json.dumps({"token": "\ud83d\udccb Analyzing your input for MSME form fields..."})
            yield f"data: {data_init}\n\n"
            data_nl = json.dumps({"token": nl})
            yield f"data: {data_nl}\n\n"

            extractor = MsmeExtractor(session_id)
            result = extractor.extract_from_text(req.query)

            if result["status"] == "error":
                msg = "\u274c Extraction failed: " + result["message"]
                data_str = json.dumps({'token': msg})
                yield f"data: {data_str}\n\n"
            elif result["status"] == "complete":
                msg = "\u2705 All form fields are already filled! No new extraction needed."
                data_str = json.dumps({'token': msg})
                yield f"data: {data_str}\n\n"
            else:
                fu = result["fields_updated"]
                prov = result.get("provider", "LLM")
                msg1 = f"\u2705 Successfully extracted **{fu}** new fields using {prov}."
                yield f"data: {json.dumps({'token': msg1})}\n\n"

                ff = result["filled_fields"]
                tf = result["total_fields"]
                pc = result["percent_complete"]
                msg2 = f"{nl}\ud83d\udcca **Progress:** {ff}/{tf} fields filled ({pc}%)"
                yield f"data: {json.dumps({'token': msg2})}\n\n"

                mfc = result["missing_fields_count"]
                if mfc > 0:
                    msg3 = f"{nl}\ud83d\udcc4 {mfc} fields still missing. Upload more documents or describe additional details."
                    yield f"data: {json.dumps({'token': msg3})}\n\n"
                else:
                    msg3 = f"{nl}\ud83c\udf89 **All fields filled!** Your MSME form is ready."
                    yield f"data: {json.dumps({'token': msg3})}\n\n"

            # Send extraction metadata for frontend to render result cards
            meta_payload = {"msme_result": result, "form_url": f"/msme/form/{session_id}"}
            yield f"data: {json.dumps(meta_payload)}\n\n"

            # Save assistant response
            summary = f"Extracted {result.get('fields_updated', 0)} fields. Progress: {result.get('filled_fields', 0)}/{result.get('total_fields', 0)}"
            append_message(session_id, "assistant", summary, "msme", user_id=user_id)

            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
        except Exception as e:
            logging.error(f"MSME chat error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def process_msme_extract_background(task_id: str, file_bytes: bytes, filename: str, content_type: str, session_id: str):
    try:
        logging.info(f"[MSME Background Task {task_id}] Started for: {filename}")
        extractor = MsmeExtractor(session_id)
        result = extractor.extract(file_bytes, filename, content_type)
        
        if result["status"] == "error":
            msme_upload_tasks[task_id] = {"status": "failed", "error": result["message"]}
        else:
            msme_upload_tasks[task_id] = {
                "status": "success",
                "data": {
                    "status": result["status"],
                    "message": result["message"],
                    "fields_updated": result.get("fields_updated", 0),
                    "total_filled": result.get("filled_fields", 0),
                    "total_fields": result.get("total_fields", 0),
                    "missing_fields_count": result.get("missing_fields_count", 0),
                    "percent_complete": result.get("percent_complete", 0),
                    "form_available": result.get("filled_fields", 0) > 0,
                }
            }
        logging.info(f"[MSME Background Task {task_id}] Completed with status: {result['status']}")
    except Exception as e:
        logging.error(f"[MSME Background Task {task_id}] Unexpected error: {e}", exc_info=True)
        msme_upload_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/msme/extract")
async def msme_extract_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Upload a file (PDF/image/text) for MSME form extraction with active heartbeat streaming."""
    import re
    UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
    if not UUID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    try:
        import time
        import threading
        import queue

        file_bytes = await file.read()
        filename = file.filename
        content_type = file.content_type

        # Global task registry update (for backwards compatibility if any client polls it)
        task_id = str(uuid.uuid4())
        msme_upload_tasks[task_id] = {"status": "processing"}

        q = queue.Queue()

        def run_extraction():
            try:
                logging.info(f"[MSME Sync Stream] Starting extraction for: {filename}")
                extractor = MsmeExtractor(session_id)
                res = extractor.extract(file_bytes, filename, content_type)
                
                # Format the response exactly as expected by the frontend
                if res["status"] == "error":
                    final_res = {
                        "status": "error",
                        "message": res["message"],
                        "fields_updated": 0,
                        "total_filled": res.get("filled_fields", 0),
                        "total_fields": res.get("total_fields", 0),
                        "missing_fields_count": res.get("missing_fields_count", 0),
                        "percent_complete": res.get("percent_complete", 0),
                        "form_available": res.get("filled_fields", 0) > 0,
                    }
                    msme_upload_tasks[task_id] = {"status": "failed", "error": res["message"]}
                else:
                    final_res = {
                        "status": res["status"],
                        "message": res["message"],
                        "fields_updated": res.get("fields_updated", 0),
                        "total_filled": res.get("filled_fields", 0),
                        "total_fields": res.get("total_fields", 0),
                        "missing_fields_count": res.get("missing_fields_count", 0),
                        "percent_complete": res.get("percent_complete", 0),
                        "form_available": res.get("filled_fields", 0) > 0,
                    }
                    msme_upload_tasks[task_id] = {"status": "success", "data": final_res}
                
                q.put(("success", final_res))
            except Exception as e:
                logging.error(f"[MSME Sync Stream] Extraction thread exception: {e}", exc_info=True)
                error_res = {
                    "status": "error",
                    "message": str(e),
                    "fields_updated": 0,
                    "total_filled": 0,
                    "total_fields": 0,
                    "missing_fields_count": 0,
                    "percent_complete": 0,
                    "form_available": False,
                }
                msme_upload_tasks[task_id] = {"status": "failed", "error": str(e)}
                q.put(("error", error_res))

        t = threading.Thread(target=run_extraction)
        t.start()

        def generate():
            # Yield spaces/newlines to keep connection active and prevent RunPod from freezing container
            while t.is_alive():
                yield " "
                time.sleep(1.5)

            t.join()

            try:
                status, val = q.get_nowait()
                yield json.dumps(val)
            except Exception as e:
                yield json.dumps({
                    "status": "error",
                    "message": f"Failed to retrieve extraction result: {str(e)}",
                    "fields_updated": 0,
                    "total_filled": 0,
                    "total_fields": 0,
                    "missing_fields_count": 0,
                    "percent_complete": 0,
                    "form_available": False,
                })

        return StreamingResponse(
            generate(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        logging.error(f"[MSME Extract Init] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@app.get("/msme/extract/status/{task_id}")
def get_msme_extract_status(task_id: str):
    """Poll the status of an MSME extraction background task."""
    if task_id not in msme_upload_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return msme_upload_tasks[task_id]


@app.get("/msme/form/{session_id}")
def get_msme_form(session_id: str):
    """Get the filled MSME form as rendered markdown."""
    try:
        extractor = MsmeExtractor(session_id)
        markdown = extractor.get_filled_form()
        progress = extractor.get_progress()
        return {
            "session_id": session_id,
            "markdown": markdown,
            "progress": progress,
        }
    except Exception as e:
        logging.error(f"MSME form fetch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@app.delete("/msme/session/{session_id}")
def reset_msme_session(session_id: str):
    """Clear/reset extraction state for a session."""
    try:
        extractor = MsmeExtractor(session_id)
        extractor.reset()
        return {"status": "success", "message": f"MSME session {session_id} reset."}
    except Exception as e:
        logging.error(f"MSME session reset error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

# ── Citation Verification (proxy to final_rag on port 8003) ────────────
class VerifyClaimRequest(BaseModel):
    claim: str
    source_chunk: str
    query: str = ""

@app.post("/verify_claim")
@app.post("/api/verify_claim")
def verify_claim_proxy(req: VerifyClaimRequest):
    """Proxy verify_claim to final_rag (port 8003)."""
    try:
        res = requests.post(
            "http://127.0.0.1:8003/verify_claim",
            json=req.dict(),
            timeout=60,
        )
        return JSONResponse(status_code=res.status_code, content=res.json())
    except Exception as e:
        logging.error(f"verify_claim proxy error: {e}")
        return JSONResponse(status_code=500, content={
            "verdict": "UNCHECKED",
            "score": 0.0,
            "unsupported_claims": [],
            "reasoning": f"Proxy error: {e}",
            "provider": "none",
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
