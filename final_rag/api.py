"""
final_rag/api.py

FastAPI — RAG Pipeline Gateway for final_rag (MSME & Legal Document Intelligence).
Integrated with Gemini Multimodal OCR, GLiNER 2.5 entity chunking, Qwen3 Hybrid retrieval,
active document routing, and MongoDB Atlas for session & document management.
"""

import sys
sys.modules['torchcodec'] = None

import json
import logging
import os
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

from final_rag.qdrant_storage.store import QdrantManager
from final_rag.agent.orchestrator import get_orchestrator
from final_rag.agent.grounding_checker import get_grounding_checker
from final_rag.ingestion.embedder import get_embedder
from final_rag.ingestion.parser import DocumentParser
from final_rag.ingestion.chunker import DocumentChunker
from final_rag.ingestion.chunker_metadata import build_chunk_meta_batch
from final_rag.db.database import (
    create_tables,
    fetch_all_sessions,
    fetch_conversation_history,
    insert_conversation,
    upsert_session_title,
    delete_session,
    get_document_by_filename,
    get_document_by_hash,
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

app = FastAPI(title="c-net RAG API", version="2.0.0")


@app.get("/ping")
def ping():
    return {"status": "ok"}


_raw_origins    = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8081")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline Singletons ───────────────────────────────────────────────────────
db               = QdrantManager()
embedder         = get_embedder(db)
orchestrator     = get_orchestrator(embedder)
doc_parser       = DocumentParser()
doc_chunker      = DocumentChunker()
try:
    grounding_checker = get_grounding_checker()
except Exception as e:
    logger.warning("Grounding checker initialization skipped: %s", e)
    grounding_checker = None


@app.on_event("startup")
def startup():
    create_tables()
    cleanup_stuck_documents()
    logger.info("final_rag API startup complete | Collections & MongoDB tables ready.")


# ── Request Models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:                str
    session_id:           Optional[str] = None
    stream:               bool          = False
    use_claude:           bool          = True
    model:                Optional[str] = None
    active_doc_id:        Optional[str] = None
    active_doc_filename:  Optional[str] = None
    active_doc_ids:       Optional[List[str]] = None
    active_doc_filenames: Optional[List[str]] = None
    active_documents:     Optional[List[str]] = None
    files:                Optional[List[Dict[str, Any]]] = None


def _resolve_active_docs(req: QueryRequest) -> tuple[Optional[str], List[str]]:
    single_doc = req.active_doc_filename
    multi_docs = list(req.active_doc_filenames or req.active_documents or [])

    if req.files:
        extracted = [f.get("name") for f in req.files if isinstance(f, dict) and f.get("name")]
        if len(extracted) == 1 and not single_doc:
            single_doc = extracted[0]
        elif len(extracted) > 1:
            multi_docs.extend(extracted)

    return single_doc, list(set(multi_docs))


# ── POST /api/v2/generate (and /chat) ─────────────────────────────────────────
@app.post("/api/v2/generate")
@app.post("/api/generate")
@app.post("/chat")
def generate(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    raw_history = fetch_conversation_history(session_id)
    history = [
        {"question": h.question, "answer": h.answer}
        for h in raw_history
    ]

    single_doc, multi_docs = _resolve_active_docs(req)

    result = orchestrator.run(
        query                = req.query,
        history              = history,
        active_doc_id        = req.active_doc_id,
        active_doc_filename  = single_doc,
        active_doc_ids       = req.active_doc_ids,
        active_doc_filenames = multi_docs,
        active_documents     = multi_docs,
    )

    sources_payload = [s.model_dump() for s in result.sources]
    insert_conversation(
        session_id = session_id,
        question   = req.query,
        answer     = result.answer,
        sources    = sources_payload,
    )

    return {
        "session_id":     session_id,
        "query":          req.query,
        "answer":         result.answer,
        "sources":        sources_payload,
        "cleaned_query":  result.cleaned_query,
        "pipeline_times": result.pipeline_times,
        "total_time_sec": result.total_time_sec,
    }


# ── POST /api/v2/generate/stream (and /chat/stream) ───────────────────────────
@app.post("/api/v2/generate/stream")
@app.post("/api/generate/stream")
@app.post("/chat/stream")
def generate_stream(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    raw_history = fetch_conversation_history(session_id)
    history = [
        {"question": h.question, "answer": h.answer}
        for h in raw_history
    ]

    single_doc, multi_docs = _resolve_active_docs(req)

    def event_generator():
        collected_tokens = []
        metadata_payload = None

        token_stream = orchestrator.stream(
            query                = req.query,
            history              = history,
            active_doc_id        = req.active_doc_id,
            active_doc_filename  = single_doc,
            active_doc_ids       = req.active_doc_ids,
            active_doc_filenames = multi_docs,
            active_documents     = multi_docs,
        )

        for chunk in token_stream:
            if chunk.startswith("__METADATA__:"):
                metadata_payload = chunk[len("__METADATA__:"):]
                yield f"event: metadata\ndata: {metadata_payload}\n\n"
            else:
                collected_tokens.append(chunk)
                yield f"data: {json.dumps({'token': chunk})}\n\n"

        full_answer = "".join(collected_tokens)
        sources = json.loads(metadata_payload) if metadata_payload else []

        insert_conversation(
            session_id = session_id,
            question   = req.query,
            answer     = full_answer,
            sources    = sources,
        )
        yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'metadata': sources, 'sources': sources})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── POST /api/v2/upload ───────────────────────────────────────────────────────
@app.post("/api/v2/upload")
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    file_bytes = await file.read()
    suffix = Path(file.filename).suffix.lower()

    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {config.SUPPORTED_EXTENSIONS}",
        )

    doc_id = str(uuid.uuid4())
    resolved_type = doc_type or suffix.lstrip(".")

    try:
        await run_in_threadpool(lambda: insert_document(
            document_id = doc_id,
            file_name   = file.filename,
            doc_type    = resolved_type,
            file_data   = file_bytes,
            status      = "processing",
        ))
    except Exception as e:
        logger.error("Failed to register document in DB: %s", e)

    try:
        # 1. Parse via Gemini Multimodal OCR
        parsed_result = await run_in_threadpool(
            lambda: doc_parser.parse_bytes(file_bytes, file.filename)
        )
        if not parsed_result.success:
            raise RuntimeError(f"Parsing failed: {parsed_result.error}")

        # 2. Chunk with semantic boundaries & tables
        chunks = await run_in_threadpool(lambda: doc_chunker.chunk(parsed_result))
        if not chunks:
            raise RuntimeError("Document yielded 0 embeddable chunks.")

        # 3. Enrich chunks with GLiNER entities & metadata
        metas = await run_in_threadpool(
            lambda: build_chunk_meta_batch(chunks, parsed_result.meta)
        )

        # 4. Embed & Store into Qdrant
        await run_in_threadpool(
            lambda: embedder.embed_and_store(chunks, metas)
        )

        # 5. Mark as indexed in MongoDB
        await run_in_threadpool(
            lambda: update_document_status(file.filename, "indexed")
        )

        return {
            "status":         "success",
            "file_name":      file.filename,
            "document_id":    doc_id,
            "chunks_indexed": len(chunks),
            "pages":          parsed_result.total_pages,
            "doc_lang":       parsed_result.meta.doc_lang if parsed_result.meta else "en",
            "doc_year":       parsed_result.meta.doc_year if parsed_result.meta else "",
        }

    except Exception as e:
        logger.error("Upload pipeline failed for %s: %s", file.filename, e)
        await run_in_threadpool(lambda: update_document_status(file.filename, "failed"))
        raise HTTPException(status_code=500, detail=str(e))


# ── DELETE /api/v2/documents/{filename} ───────────────────────────────────────
@app.delete("/api/v2/documents/{filename}")
@app.delete("/api/documents/{filename}")
@app.delete("/documents/{filename}")
async def delete_doc(filename: str):
    try:
        await run_in_threadpool(lambda: db.delete_document(filename))
        await run_in_threadpool(lambda: delete_document_record(filename))
        return {"status": "success", "deleted": filename}
    except Exception as e:
        logger.error("Failed to delete document %s: %s", filename, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/v2/documents ─────────────────────────────────────────────────────
@app.get("/api/v2/documents")
@app.get("/api/documents")
@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


# ── GET /api/v2/sessions ──────────────────────────────────────────────────────
@app.get("/api/v2/sessions")
@app.get("/api/sessions")
def get_sessions():
    return {"sessions": [s.__dict__ if hasattr(s, "__dict__") else s for s in fetch_all_sessions()]}


# ── GET /api/v2/sessions/{session_id}/history ─────────────────────────────────
@app.get("/api/v2/sessions/{session_id}/history")
@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    history = fetch_conversation_history(session_id)
    return {
        "session_id": session_id,
        "history": [
            {
                "question": h.question,
                "answer":   h.answer,
                "sources":  getattr(h, "sources", []),
            }
            for h in history
        ],
    }


# ── DELETE /api/v2/sessions/{session_id} ──────────────────────────────────────
@app.delete("/api/v2/sessions/{session_id}")
@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"status": "success", "deleted_session": session_id}


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        db.get_client().get_collections()
        health_check_db()
        return {
            "status":   "healthy",
            "qdrant":   "connected",
            "database": "connected",
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ── POST /api/verify_claim ────────────────────────────────────────────────────
class VerifyClaimRequest(BaseModel):
    claim:        Optional[str] = None
    answer:       Optional[str] = None
    source_chunk: Optional[str] = None
    chunks:       Optional[List[str]] = None
    query:        str = ""


@app.post("/verify_claim")
@app.post("/api/verify_claim")
def verify_claim_endpoint(req: VerifyClaimRequest):
    if not grounding_checker:
        raise HTTPException(status_code=500, detail="Grounding checker not initialized")

    answer_text = req.answer or req.claim or ""
    chunk_list  = req.chunks or ([req.source_chunk] if req.source_chunk else [])

    try:
        gr = grounding_checker.check(
            answer = answer_text,
            chunks = chunk_list,
            query  = req.query,
        )
        return gr.to_dict() if hasattr(gr, "to_dict") else gr
    except Exception as e:
        return {
            "verdict": "UNCHECKED",
            "score": 0.0,
            "unsupported_claims": [],
            "claims": [],
            "reasoning": str(e),
            "provider": "none",
        }