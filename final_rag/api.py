"""
api.py
FastAPI — RAG Pipeline Gateway
"""

import sys
# Mock torchcodec to prevent import runtime crashes in environments with missing FFmpeg
sys.modules['torchcodec'] = None

import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
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


@app.get("/ping")
def ping():
    return {"status": "ok"}

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
    doc_parser   = DocumentParser(output_dir=config.MD_OUTPUT_DIR)
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

        # Delete intermediate .md file
        try:
            stem = Path(file_name).stem
            md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
            if md_path.exists():
                md_path.unlink()
                logger.info("Deleted intermediate md file: %s", md_path)
        except Exception as file_err:
            logger.error("Failed to delete intermediate md file for %s: %s", file_name, file_err)

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

    try:
        # 3. Clear all intermediate .md files from cache folder
        for md_file in config.MD_OUTPUT_DIR.glob("*.md"):
            try:
                md_file.unlink()
            except Exception as file_err:
                logger.error("Failed to delete md file %s: %s", md_file, file_err)
    except Exception as e:
        logger.error("Failed to clear md_output folder: %s", e)

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

class DocumentContentUpdate(BaseModel):
    content: str

@app.get("/api/documents/search")
def search_documents(q: str = Query(..., min_length=1)):
    query_lower = q.lower()
    results = []
    
    md_dir = config.MD_OUTPUT_DIR
    if not md_dir.exists():
        return results
        
    for md_path in md_dir.glob("*.md"):
        file_name = md_path.stem + ".pdf"
        matches = 0
        snippets = []
        
        name_match = query_lower in file_name.lower()
        if name_match:
            matches += 1
            
        try:
            content = md_path.read_text(encoding="utf-8")
            import re
            content_lower = content.lower()
            for match in re.finditer(re.escape(query_lower), content_lower):
                matches += 1
                match_start = match.start()
                match_end = match.end()
                
                para_start = content.rfind("\n\n", 0, match_start)
                if para_start == -1:
                    para_start = 0
                else:
                    para_start += 2
                    
                para_end = content.find("\n\n", match_end)
                if para_end == -1:
                    para_end = len(content)
                    
                paragraph = content[para_start:para_end].strip()
                
                if len(paragraph) > 400:
                    rel_start = match_start - para_start
                    rel_end = match_end - para_start
                    
                    crop_start = max(0, rel_start - 180)
                    crop_end = min(len(paragraph), rel_end + 180)
                    
                    snippet_text = paragraph[crop_start:crop_end].strip()
                    if crop_start > 0:
                        snippet_text = "... " + snippet_text
                    if crop_end < len(paragraph):
                        snippet_text = snippet_text + " ..."
                else:
                    snippet_text = paragraph
                
                snippet_clean = re.sub(r'\s+', ' ', snippet_text).strip()
                if snippet_clean and snippet_clean not in snippets:
                    snippets.append(snippet_clean)
        except Exception:
            pass
            
        if matches > 0:
            results.append({
                "file_name": file_name,
                "matches": matches,
                "snippets": snippets
            })
            
    return results

@app.get("/api/documents/{file_name}/content")
def get_document_content(file_name: str):
    stem = Path(file_name).stem
    md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
    if not md_path.exists():
        # Fallback: Reconstruct from Qdrant chunks
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            points, _ = db.client.scroll(
                collection_name=db.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=file_name),
                        )
                    ]
                ),
                limit=10000,
                with_payload=["chunk_index", "text"],
                with_vectors=False,
            )
            if points:
                sorted_points = sorted(points, key=lambda p: p.payload.get("chunk_index", 0))
                reconstructed_content = "\n\n".join([p.payload.get("text", "") for p in sorted_points])
                try:
                    config.MD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    md_path.write_text(reconstructed_content, encoding="utf-8")
                except Exception as save_err:
                    logger.warning("Failed to save reconstructed markdown cache: %s", save_err)
                return {"file_name": file_name, "content": reconstructed_content}
        except Exception as e:
            logger.error("Failed to reconstruct document from Qdrant: %s", e)
        raise HTTPException(status_code=404, detail="Content not found")
    content = md_path.read_text(encoding="utf-8")
    return {"file_name": file_name, "content": content}

@app.put("/api/documents/{file_name}/content")
def update_document_content(file_name: str, payload: DocumentContentUpdate):
    stem = Path(file_name).stem
    md_path = config.MD_OUTPUT_DIR / f"{stem}.md"
    
    md_path.parent.mkdir(exist_ok=True, parents=True)
    md_path.write_text(payload.content, encoding="utf-8")
    
    try:
        db.delete_document(file_name)
    except Exception as e:
        logger.error("Error deleting old chunks for %s: %s", file_name, e)
        
    try:
        from final_rag.ingestion.parser import ParseResult, BlockRecord, DocumentMeta, ExtractionMethod
        import hashlib
        import re
        
        file_hash = hashlib.md5(payload.content.encode('utf-8')).hexdigest()
        
        meta = DocumentMeta(
            doc_id=file_hash[:12],
            file_name=file_name,
            file_path=str(md_path),
            file_type=".md",
            file_size_kb=len(payload.content) / 1024,
            page_count=1,
            has_tables=False,
            parse_success=True,
            filename_tokens=re.split(r'[_\-\\s]+', stem.lower()),
        )
        
        blocks = [BlockRecord(block_type="text", content=payload.content, page_no=1, page_label="1")]
        
        parse_res = ParseResult(
            file_name=file_name,
            file_type=".md",
            method_used=ExtractionMethod.YOLO,
            markdown=payload.content,
            meta=meta,
            total_pages=1,
            success=True,
            blocks=blocks,
            doc_id=meta.doc_id,
            filename_tokens=meta.filename_tokens
        )
        
        chunks = doc_chunker.chunk(parse_res)
        embedder.embed_and_store(chunks)
        
        doc_record = get_document_by_filename(file_name)
        if doc_record:
            update_document_status(doc_record.document_id, "success")
        
        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        logger.error("Error re-ingesting %s: %s", file_name, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/documents/preview-parse ──────────────────────────────────
@app.post("/api/documents/preview-parse")
async def preview_parse_document(file: UploadFile = File(...)):
    """Parse a document to markdown in memory without saving or ingesting."""
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file for preview: %s", e)
        raise HTTPException(status_code=500, detail="Could not read file.")

    try:
        # Use a parser with output_dir=None so _save() is a no-op
        preview_parser = DocumentParser(output_dir=None)
        parsed_result = preview_parser.parse_bytes(file_bytes, file.filename)
        if not parsed_result.success:
            raise Exception(f"Parsing failed: {parsed_result.error}")

        return {
            "file_name": file.filename,
            "content": parsed_result.markdown,
        }
    except Exception as e:
        logger.error("Preview parse failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/documents/replace ────────────────────────────────────────
@app.post("/api/documents/replace")
async def replace_document(
    file: UploadFile = File(...),
    old_file_name: str = Form(...),
):
    """Delete an existing document and ingest a new one in its place."""
    # Step 1: Delete old document from Qdrant
    try:
        db.delete_document(old_file_name)
        logger.info("Deleted old vectors for '%s' from Qdrant.", old_file_name)
    except Exception as e:
        logger.error("Failed to delete old vectors for %s: %s", old_file_name, e)

    # Step 2: Delete old document DB record
    try:
        delete_document_record(old_file_name)
        logger.info("Deleted old DB record for '%s'.", old_file_name)
    except Exception as e:
        logger.error("Failed to delete old DB record for %s: %s", old_file_name, e)

    # Step 3: Delete old .md file
    try:
        old_stem = Path(old_file_name).stem
        old_md_path = config.MD_OUTPUT_DIR / f"{old_stem}.md"
        if old_md_path.exists():
            old_md_path.unlink()
            logger.info("Deleted old md file: %s", old_md_path)
    except Exception as e:
        logger.error("Failed to delete old md file for %s: %s", old_file_name, e)

    # Step 4: Ingest the new file (full pipeline: parse → chunk → embed → save md)
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read new file.")

    doc_id = str(uuid.uuid4())
    doc_type = Path(file.filename).suffix.lower()

    try:
        insert_document(
            document_id=doc_id,
            file_name=file.filename,
            doc_type=doc_type,
            file_data=file_bytes,
            status="processing",
        )
    except Exception as e:
        logger.error("Failed to create document placeholder for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail="Database error.")

    try:
        parsed_result = doc_parser.parse_bytes(file_bytes, file.filename)
        if not parsed_result.success:
            raise Exception(f"Parsing failed: {parsed_result.error}")

        chunks = doc_chunker.chunk(parsed_result)
        embedder.embed_and_store(chunks)
        update_document_status(file.filename, "ingested")

        logger.info("Successfully replaced '%s' with '%s' (%d chunks)", old_file_name, file.filename, len(chunks))
        return {
            "status": "success",
            "old_deleted": old_file_name,
            "new_file": file.filename,
            "chunks_indexed": len(chunks),
        }
    except Exception as e:
        logger.error("Replace pipeline failed for %s: %s", file.filename, e)
        try:
            db.delete_document(file.filename)
        except Exception:
            pass
        try:
            update_document_status(file.filename, "failed")
        except Exception:
            pass
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
            with_payload=["source_file"], 
            with_vectors=False
        )
        for point in points:
            src = point.payload.get("source_file")
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