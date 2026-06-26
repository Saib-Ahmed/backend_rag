import sys
import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, BackgroundTasks
import uuid
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
import tempfile
import subprocess
import speech_recognition as sr
import imageio_ffmpeg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from msme_extractor import MsmeExtractor

from unified_db import (
    create_user, verify_password, get_user, create_session,
    get_user_sessions, append_message, get_chat_history,
    get_chat_history_formatted_for_llm, update_session_title, delete_session,
    save_document_metadata, get_all_document_metadata, delete_document_metadata,
)

app = FastAPI(title="Unified RAG API")


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
upload_tasks = {}
msme_upload_tasks = {}

# Ensure backup_markdown directory exists (immutable archive)
# If running on RunPod serverless, store inside the persistent network volume at /runpod-volume
RUNPOD_VOLUME = "/runpod-volume"
if os.path.exists(RUNPOD_VOLUME) and os.access(RUNPOD_VOLUME, os.W_OK):
    BACKUP_MD_DIR = os.path.join(RUNPOD_VOLUME, "backup_markdown")
else:
    BACKUP_MD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_markdown")
os.makedirs(BACKUP_MD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    model: str = "v1" # "v1" (RAG_system) or "v2" (final_rag)
    user_id: str = "default_user"

@app.post("/auth/register")
def register(req: RegisterRequest):
    user_id = create_user(req.username, req.email, req.password)
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or Email already exists")
    return {"user_id": user_id, "username": req.username}

@app.post("/auth/login")
def login(req: AuthRequest):
    user = get_user(req.username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user["user_id"], "username": user["username"]}

@app.get("/sessions")
def get_sessions(user_id: str = "default_user"):
    sessions = get_user_sessions(user_id)
    return [{"id": s["session_id"], "title": s["title"]} for s in sessions]

@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
    history = get_chat_history(session_id)
    messages = [{"role": msg["role"], "text": msg["content"]} for msg in history]
    return {"session_id": session_id, "messages": messages}

class RenameSessionRequest(BaseModel):
    title: str

@app.patch("/sessions/{session_id}/title")
def rename_session_route(session_id: str, req: RenameSessionRequest):
    update_session_title(session_id, req.title)
    return {"status": "success"}

@app.delete("/sessions/{session_id}")
def delete_session_route(session_id: str):
    delete_session(session_id)
    return {"status": "success"}

@app.get("/stats")
def get_combined_stats():
    stats = {
        "v1": {"status": "offline", "num_chunks": 0},
        "v2": {"status": "offline", "num_chunks": 0, "graph": {"entities": 0, "relationships": 0}}
    }
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
def chat_stream(req: ChatRequest):
    session_id = req.session_id
    if not session_id:
        title = req.query[:30] + "..." if len(req.query) > 30 else req.query
        session_id = create_session(req.user_id, title)
    
    append_message(session_id, "user", req.query, req.model, user_id=req.user_id)

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
                "session_id": session_id
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

                # ── Save metadata to MongoDB ──
                try:
                    save_document_metadata(
                        file_name=filename,
                        doc_type=doc_type,
                        source=source,
                        source_description=source_description,
                        creation_date=creation_date,
                        rag_version=rag_version,
                    )
                except Exception as meta_err:
                    logging.error(f"[Background Task {task_id}] Failed to save metadata: {meta_err}")

                # ── Copy .md to backup_markdown/ (immutable archive) ──
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
                        backup_dest = os.path.join(BACKUP_MD_DIR, f"{stem}.md")
                        # If file already exists in backup, add timestamp suffix to keep both
                        if os.path.exists(backup_dest):
                            from datetime import datetime
                            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                            backup_dest = os.path.join(BACKUP_MD_DIR, f"{stem}_{ts}.md")
                        shutil.copy2(md_source, backup_dest)
                        logging.info(f"[Background Task {task_id}] Backed up md to: {backup_dest}")
                    else:
                        logging.warning(f"[Background Task {task_id}] No .md source file found for backup")
                except Exception as backup_err:
                    logging.error(f"[Background Task {task_id}] Failed to create backup: {backup_err}")
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
):
    try:
        # Read the file content synchronously before returning the response so it's available in the background
        file_content = file.file.read()
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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

class DocumentContentUpdateUnified(BaseModel):
    content: str

@app.put("/documents/{file_name}/content")
def update_document_content_route(file_name: str, req: DocumentContentUpdateUnified, rag_version: str = Query("version1")):
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
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{file_name}")
def delete_document_route(file_name: str, rag_version: str = Query("version1")):
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/metadata")
def get_documents_metadata_route():
    """Return metadata for all ingested documents."""
    try:
        return get_all_document_metadata()
    except Exception as e:
        logging.error(f"Failed to get document metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
            return res.json()
        else:
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend proxy error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/ollama")
def debug_ollama():
    try:
        import subprocess
        # Check running processes
        ps_out = subprocess.check_output(["ps", "aux"], text=True)
        # Check ollama status
        try:
            ollama_ps = subprocess.check_output(["ollama", "ps"], text=True)
        except Exception as e:
            ollama_ps = f"Ollama ps failed: {e}"
            
        # Read last 100 lines of ollama log
        log_out = ""
        if os.path.exists("/var/log/ollama.log"):
            with open("/var/log/ollama.log", "r") as f:
                lines = f.readlines()
                log_out = "".join(lines[-100:])
        else:
            log_out = "/var/log/ollama.log not found"
            
        return {
            "ps": ps_out,
            "ollama_ps": ollama_ps,
            "ollama_log": log_out
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# MSME Extraction Endpoints
# ==========================================

class MsmeChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    user_id: str = "default_user"


@app.post("/chat/msme")
def chat_msme_stream(req: MsmeChatRequest):
    """Streaming MSME extraction from text (voice transcript or typed description)."""
    session_id = req.session_id
    if not session_id:
        title = req.query[:30] + "..." if len(req.query) > 30 else req.query
        session_id = create_session(req.user_id, title)

    append_message(session_id, "user", req.query, "msme", user_id=req.user_id)

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
            append_message(session_id, "assistant", summary, "msme", user_id=req.user_id)

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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/msme/session/{session_id}")
def reset_msme_session(session_id: str):
    """Clear/reset extraction state for a session."""
    try:
        extractor = MsmeExtractor(session_id)
        extractor.reset()
        return {"status": "success", "message": f"MSME session {session_id} reset."}
    except Exception as e:
        logging.error(f"MSME session reset error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
