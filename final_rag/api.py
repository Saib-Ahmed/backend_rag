"""
api.py
FastAPI — RAG Pipeline Gateway
"""

import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from final_rag.qdrant_storage.store import QdrantManager
from final_rag.agent.orchestrator import get_orchestrator
from final_rag.ingestion.embedder import get_embedder
from final_rag.ingestion.parser import DocumentParser
from final_rag.ingestion.chunker import DocumentChunker
from final_rag.db.database import (
    create_tables,
    fetch_all_sessions,
    fetch_conversation_history,
    insert_conversation,
    upsert_session_title,
    delete_session,
    get_document_by_filename,
    insert_document,
    update_document_status,
    cleanup_stuck_documents,
    delete_document_record,
    list_documents,
    clear_all_documents,
    health_check_db,
)
import final_rag.config as config

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="c-net RAG API", version="1.0.0")

_raw_origins    = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Globals ────────────────────────────────────────────────────────────
db           = None
embedder     = None
orchestrator = None
doc_parser   = None
doc_chunker  = None


# ── Startup ────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global db, embedder, orchestrator, doc_parser, doc_chunker
    create_tables()

    db           = QdrantManager()
    db.setup_database()
    embedder     = get_embedder(db=db)
    orchestrator = get_orchestrator(embedder=embedder)
    doc_parser   = DocumentParser(output_dir=Path("md_output"))
    doc_chunker  = DocumentChunker()

    # Cleanup stuck records from previous crashed runs
    cleanup_stuck_documents()
    logger.info("Cleaned up stuck processing records on startup")

    logger.info("API Ready | CORS origins: %s", ALLOWED_ORIGINS)


# ── Shutdown ───────────────────────────────────────────────────────────
@app.on_event("shutdown")
def shutdown():
    logger.info("API shutting down, explicitly clearing memory and connections...")
    global db
    try:
        if db and hasattr(db, "client") and db.client:
            db.client.close()
    except Exception as e:
        logger.error("Error closing Qdrant: %s", e)

    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
    except ImportError:
        pass


# ── Request schemas ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    query:      str


class RenameTitleRequest(BaseModel):
    title: str


# ── POST /upload ───────────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    existing_doc = get_document_by_filename(file.filename)
    if existing_doc:
        logger.info("File already exists, skipping: %s", file.filename)
        return {
            "status":  "already_exists",
            "message": "File is already stored in the database.",
        }

    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=500, detail="Could not read file.")

    doc_id   = str(uuid.uuid4())
    doc_type = Path(file.filename).suffix.lower()

    try:
        insert_document(
            document_id = doc_id,
            file_name   = file.filename,
            doc_type    = doc_type,
            file_data   = file_bytes,
            status      = "processing",
        )
    except Exception as e:
        logger.error("Failed to create document placeholder: %s", e)
        raise HTTPException(status_code=500, detail="Database error.")

    try:
        logger.info("Starting ingestion for: %s", file.filename)

        parsed_result = doc_parser.parse_bytes(file_bytes, file.filename)
        if not parsed_result.success:
            raise Exception(f"Parsing failed: {parsed_result.error}")

        chunks = doc_chunker.chunk(parsed_result)
        embedder.embed_and_store(chunks)

        update_document_status(file.filename, "ingested")

        logger.info("Successfully ingested: %s", file.filename)
        return {
            "status":      "success",
            "message":     "File successfully uploaded and processed.",
            "document_id": doc_id,
            "file_name":   file.filename,
        }

    except Exception as e:
        logger.error("Upload pipeline failed for %s: %s", file.filename, e)

        try:
            db.delete_document(file.filename)
        except Exception as rollback_err:
            logger.critical(
                "CRITICAL: Qdrant rollback failed for %s: %s — manual cleanup needed",
                file.filename, rollback_err,
            )

        try:
            update_document_status(file.filename, "failed")
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(e))


# ── POST /chat/stream ──────────────────────────────────────────────────
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    history_turns = fetch_conversation_history(req.session_id, limit=6)
    history = [
        {"question": turn.question, "answer": turn.answer}
        for turn in history_turns
    ]
    # Remove the very last message if it matches the current query, to prevent LLM seeing it twice
    if history and history[-1]["question"] == req.query and not history[-1]["answer"]:
        history.pop()

    def event_generator():
        full_answer    = []
        metadata_chunk = ""
        source_name    = ""
        page_label     = ""
        page_no        = 0
        stream_error   = None

        try:
            for token in orchestrator.run(
                query           = req.query,
                history         = history,
                active_document = None,
            ):
                if token.startswith("__METADATA__:"):
                    metadata_chunk = token
                    continue
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            logger.error("Streaming error: %s", e)
            stream_error = str(e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        complete_answer = "".join(full_answer).strip()

        if metadata_chunk:
            try:
                _, metadata_part = metadata_chunk.split("__METADATA__:")
                sources = json.loads(metadata_part.strip())
                if sources:
                    top_source  = sources[0]
                    source_name = top_source.get("file_name", "")
                    pages       = top_source.get("pages", [])
                    if pages:
                        page_label = str(pages[0])
                        page_no    = int(pages[0]) if str(pages[0]).isdigit() else 0
            except Exception as e:
                logger.error("Failed to parse metadata: %s", e)

        try:
            answer_to_save = (
                f"[Error: {stream_error}]" if stream_error
                else complete_answer or "[no answer generated]"
            )
            insert_conversation(
                session_id = req.session_id,
                question   = req.query,
                answer     = answer_to_save,
                source     = source_name,
                page       = page_no,
                page_label = page_label,
            )
        except Exception as e:
            logger.error("Failed to persist conversation: %s", e)

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /sessions ──────────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    sessions = fetch_all_sessions()
    return [{"id": s.session_id, "title": s.title} for s in sessions]


# ── GET /sessions/{session_id}/history ────────────────────────────────
@app.get("/sessions/{session_id}/history")
def get_history(session_id: str, limit: int = Query(default=100, le=200)):
    turns    = fetch_conversation_history(session_id, limit=limit)
    messages = []
    for turn in turns:
        messages.append({"role": "user",      "text": turn.question})
        messages.append({"role": "assistant", "text": turn.answer})
    return {"session_id": session_id, "messages": messages}


# ── PATCH /sessions/{session_id}/title ────────────────────────────────
@app.patch("/sessions/{session_id}/title")
def rename_session(session_id: str, body: RenameTitleRequest):
    try:
        upsert_session_title(session_id, body.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"session_id": session_id, "title": body.title}


# ── DELETE /sessions/{session_id} ─────────────────────────────────────
@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    try:
        delete_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": session_id}


# ── DELETE /documents/{file_name} ─────────────────────────────────────
@app.delete("/documents/{file_name}")
def delete_document_endpoint(file_name: str):
    qdrant_deleted   = False
    postgres_deleted = False

    try:
        if db:
            db.delete_document(file_name)
            qdrant_deleted = True
            logger.info("Cleared vectors for '%s' from Qdrant.", file_name)

        postgres_deleted = delete_document_record(file_name)
        if postgres_deleted:
            logger.info("Deleted '%s' from Database.", file_name)
        else:
            logger.warning(
                "'%s' not found in Database — Qdrant only.", file_name
            )

        return {
            "status":           "success",
            "message":          f"'{file_name}' purged from pipeline.",
            "qdrant_deleted":   qdrant_deleted,
            "postgres_deleted": postgres_deleted,
        }

    except Exception as e:
        logger.error("Failed to delete %s: %s", file_name, e)

        if qdrant_deleted and not postgres_deleted:
            logger.critical(
                "CORRUPTION: '%s' removed from Qdrant but not Database. "
                "Manual cleanup required.",
                file_name,
            )

        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents")
def clear_all_documents_endpoint():
    try:
        # 1. Delete and setup the database collection fresh to clear it completely and safely
        db.client.delete_collection(db.collection_name)
        db.setup_database()
        qdrant_cleared = True
    except Exception as e:
        logger.error("Failed to clear Qdrant collection: %s", e)
        qdrant_cleared = False

    try:
        # 2. Truncate the Database documents table
        postgres_cleared = clear_all_documents()
    except Exception as e:
        logger.error("Failed to truncate Database documents: %s", e)
        postgres_cleared = False

    return {
        "status": "success",
        "message": "All documents cleared.",
        "qdrant_cleared": qdrant_cleared,
        "postgres_cleared": postgres_cleared,
    }


# ── GET /documents ─────────────────────────────────────────────────────
@app.get("/documents")
def list_documents_endpoint():
    try:
        return list_documents()
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
def get_stats():
    # 1. Chunks
    try:
        num_chunks = db.client.count(db.collection_name).count
    except Exception as e:
        logger.error("Failed to get Qdrant count: %s", e)
        num_chunks = 0
        
    # 2. Files from Qdrant payloads
    files = set()
    try:
        points, _ = db.client.scroll(
            collection_name=db.collection_name, 
            limit=10000, 
            with_payload=True, 
            with_vectors=False
        )
        for point in points:
            src = point.payload.get("source_file") or point.payload.get("source") or point.payload.get("file_name") or point.payload.get("source_name")
            if src:
                files.add(str(src))
    except Exception as e:
        logger.error("Failed to get Qdrant files: %s", e)
        pass

    return {
        "status": "online",
        "num_chunks": num_chunks,
        "files": list(files),
        "graph": {"connected": False, "entities": 0, "relationships": 0}
    }

# ── GET /health ────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    try:
        db.get_client().get_collections()

        health_check_db()

        return {
            "status":   "healthy",
            "qdrant":   "connected",
            "database": "connected",
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error":  str(e),
        }