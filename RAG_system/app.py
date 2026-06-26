"""FastAPI backend for the Local RAG System."""

from __future__ import annotations

import sys
# Mock torchcodec to prevent import runtime crashes in environments with missing FFmpeg
sys.modules['torchcodec'] = None

import io
import json
import logging
import os
import re
import time
from threading import Lock
from typing import Any, Dict, List, Optional

# Configure logging to ensure debug statements and prints are visible in the terminal
# We use force=True to override any existing logging configuration from imported modules
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    force=True
)

# Silence noisy library internals that flood the terminal and cause lag
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("neo4j").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

import requests
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import unified_db
except ImportError:
    unified_db = None
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import RAGEngine
import config
from new_ingestion.parser import ParsingMode
from live_search import LiveSearchEngine, load_live_mappings, save_live_mappings
from session_store import (
    load_sessions as load_session_store,
    save_sessions as save_session_store,
)
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
    OLLAMA_BASE_URL,
)


HIST_FILE = "chat_history_sessions.json"
ENGINE_LOCK = Lock()


class EngineConfig(BaseModel):
    llm_model: str = LLM_MODEL
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = LLM_MAX_TOKENS
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    retrieval_top_k: int = RETRIEVAL_TOP_K
    retrieval_score_threshold: float = RETRIEVAL_SCORE_THRESHOLD
    lazy_loading_mode: bool = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    metadata_filters: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    use_live_search: bool = True
    history_limit: int = 6
    enable_graph_rag: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    session_id: str
    live_answer: Optional[str] = None
    live_sources: Optional[List[Dict[str, Any]]] = None


class LiveMappingRequest(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    name: Optional[str] = None


def _ensure_history_file() -> None:
    if not os.path.exists(HIST_FILE):
        with open(HIST_FILE, "w", encoding="utf-8") as handle:
            json.dump({}, handle)


def _load_sessions() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_history_file()
    return load_session_store(HIST_FILE)


def _save_sessions(sessions: Dict[str, List[Dict[str, Any]]]) -> None:
    _ensure_history_file()
    save_session_store(HIST_FILE, sessions)


def _create_session_id(question: str) -> str:
    prefix = (question[:30].strip() + "...") if len(question) > 30 else question.strip()
    prefix = prefix or "New Session"
    stamp = int(time.time() * 1000) % 10000
    return f"{prefix} ({stamp})"


def _parse_parsing_mode(mode: str) -> ParsingMode:
    return ParsingMode.SMART


def _keyword_matches(text: str, keyword: str) -> bool:
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, text) is not None


class _UploadedFileAdapter:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._buffer = io.BytesIO(data)

    def getvalue(self) -> bytes:
        return self._buffer.getvalue()

    def getbuffer(self) -> memoryview:
        return self._buffer.getbuffer()

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def seek(self, pos: int) -> None:
        self._buffer.seek(pos)


ENGINE_STATE = {
    "config": EngineConfig(),
    "engine": None,
}


def _build_engine(config: EngineConfig) -> RAGEngine:
    return RAGEngine(
        llm_model=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        retrieval_top_k=config.retrieval_top_k,
        retrieval_score_threshold=config.retrieval_score_threshold,
        auto_load_models=not config.lazy_loading_mode,
    )


def get_engine() -> RAGEngine:
    with ENGINE_LOCK:
        engine = ENGINE_STATE.get("engine")
        if engine is None:
            ENGINE_STATE["engine"] = _build_engine(ENGINE_STATE["config"])
        return ENGINE_STATE["engine"]


app = FastAPI(title="Local RAG API", version="1.0")


@app.get("/ping")
def ping():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def shutdown_event():
    """Explicitly clear memory and close connections when the terminal is closed."""
    logger.info("Server shutting down, explicitly clearing memory and connections...")
    with ENGINE_LOCK:
        engine = ENGINE_STATE.get("engine")
        if engine:
            try:
                if getattr(engine, "db", None) and getattr(engine.db, "client", None):
                    engine.db.client.close()
            except Exception as e:
                logger.error("Error closing Qdrant: %s", e)
                
            try:
                if getattr(engine, "graph_rag", None) and getattr(engine.graph_rag, "driver", None):
                    engine.graph_rag.driver.close()
            except Exception as e:
                logger.error("Error closing Neo4j: %s", e)
    
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
    except ImportError:
        pass


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=EngineConfig)
def get_config() -> EngineConfig:
    return ENGINE_STATE["config"]


@app.post("/api/config", response_model=EngineConfig)
def update_config(config: EngineConfig) -> EngineConfig:
    with ENGINE_LOCK:
        ENGINE_STATE["config"] = config
        ENGINE_STATE["engine"] = _build_engine(config)
    return config


@app.get("/api/models")
def get_available_models() -> List[str]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model.get("name") for model in data.get("models", []) if model.get("name")]
            return sorted(models)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to fetch models: %s", exc)
    return []


@app.get("/api/sources")
def get_sources() -> List[str]:
    engine = get_engine()
    return engine.get_available_sources()


@app.get("/api/sections")
def get_sections() -> List[str]:
    engine = get_engine()
    return engine.get_available_sections()


@app.delete("/api/sources/{source_name}")
def delete_source(source_name: str) -> Dict[str, Any]:
    engine = get_engine()
    ok = engine.delete_source(source_name)
    if not ok:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "deleted", "source": source_name}


@app.post("/api/clear")
def clear_database() -> Dict[str, str]:
    engine = get_engine()
    engine.clear_database()
    return {"status": "cleared"}


from fastapi import BackgroundTasks

@app.post("/api/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    parsing_mode: str = Form("SMART"),
    build_graph: bool = Form(False),
) -> Dict[str, Any]:
    engine = get_engine()
    mode = _parse_parsing_mode(parsing_mode)
    
    file_data = []
    for file in files:
        data = await file.read()
        file_data.append((file.filename, data))
        
    total_chunks = 0
    errors = []
    
    for filename, data in file_data:
        adapter = _UploadedFileAdapter(filename, data)
        try:
            print(f"🚀 Starting ingestion for {filename}...")
            chunks_indexed, _ = await engine.ingest_file(
                uploaded_file=adapter,
                mode=mode,
                build_graph=build_graph,
            )
            total_chunks += chunks_indexed
            print(f"✅ Successfully finished ingesting {filename}")
        except Exception as exc:
            print(f"❌ Failed to ingest {filename}: {exc}")
            errors.append(f"{filename}: {str(exc)}")

    if errors:
        raise HTTPException(status_code=500, detail="Failed: " + "; ".join(errors))
    
    return {"status": "success", "total_chunks": total_chunks}



@app.get("/api/sessions")
def list_sessions() -> List[Dict[str, Any]]:
    if unified_db:
        try:
            # Assuming 'default_user' for now as there's no auth in RAG_system
            mongo_sessions = unified_db.get_user_sessions("default_user")
            payload = [{"id": s["session_id"], "message_count": 0} for s in mongo_sessions]
            return payload
        except Exception as e:
            logger.error("Failed to list sessions from MongoDB: %s", e)

    sessions = _load_sessions()
    payload = []
    for session_id, messages in sessions.items():
        payload.append({"id": session_id, "message_count": len(messages)})
    return list(reversed(payload))


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    if unified_db:
        try:
            mongo_hist = unified_db.get_chat_history(session_id)
            messages = [{"role": msg["role"], "content": msg["content"], "sources": msg.get("sources", [])} for msg in mongo_hist]
            return {"id": session_id, "messages": messages}
        except Exception as e:
            logger.error("Failed to get session from MongoDB: %s", e)

    sessions = _load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session_id, "messages": sessions.get(session_id, [])}


@app.post("/api/sessions")
def create_session(request: SessionCreateRequest) -> Dict[str, str]:
    sessions = _load_sessions()
    session_id = request.name or _create_session_id("New Session")
    if session_id in sessions:
        msgs = sessions.pop(session_id)
        sessions[session_id] = msgs
    else:
        sessions[session_id] = []
    _save_sessions(sessions)
    return {"id": session_id}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    if unified_db:
        try:
            unified_db.delete_session(session_id)
            return {"status": "deleted"}
        except Exception as e:
            logger.error("Failed to delete session from MongoDB: %s", e)

    sessions = _load_sessions()
    if session_id in sessions:
        del sessions[session_id]
        _save_sessions(sessions)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/live-mappings")
def list_live_mappings() -> Dict[str, Any]:
    return load_live_mappings()


@app.post("/api/live-mappings")
def add_live_mapping(request: LiveMappingRequest) -> Dict[str, Any]:
    mappings = load_live_mappings().get("mappings", [])
    keywords = [k.strip().lower() for k in request.keywords if k.strip()]
    urls = [u.strip() for u in request.urls if u.strip()]
    if not keywords or not urls:
        raise HTTPException(status_code=400, detail="keywords and urls are required")
    mappings.append({"keywords": keywords, "urls": urls})
    save_live_mappings({"mappings": mappings})
    return {"mappings": mappings}


@app.delete("/api/live-mappings/{keyword}")
def delete_live_mapping(keyword: str) -> Dict[str, Any]:
    keyword = keyword.strip().lower()
    mappings = load_live_mappings().get("mappings", [])
    filtered = [m for m in mappings if keyword not in [k.lower() for k in m.get("keywords", [])]]
    save_live_mappings({"mappings": filtered})
    return {"mappings": filtered}


@app.post("/api/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    engine = get_engine()
    session_id = request.session_id or _create_session_id(request.question)

    # Always load sessions so the variable exists for session persistence later
    sessions = _load_sessions()

    history = []
    if unified_db:
        try:
            mongo_hist = unified_db.get_chat_history(session_id, limit=max(1, int(request.history_limit)))
            history = [{"role": msg["role"], "content": msg["content"]} for msg in mongo_hist]
            logger.info("Loaded %d history turns from MongoDB", len(history))
        except Exception as e:
            logger.error("Failed to load history from MongoDB: %s", e)
            
    # Fallback to local sessions if MongoDB is not available or failed
    if not history:
        existing = sessions.get(session_id, [])
        if existing:
            history = existing[-max(1, int(request.history_limit)) :]

    live_answer = None
    live_sources = []
    if request.use_live_search:
        mappings = load_live_mappings().get("mappings", [])
        prompt_lower = request.question.lower()
        triggered_urls = []
        for mapping in mappings:
            keywords = []
            if "keyword" in mapping:
                keywords.append(mapping["keyword"].lower())
            if "keywords" in mapping:
                keywords.extend([k.lower() for k in mapping.get("keywords", [])])
            if any(_keyword_matches(prompt_lower, kw) for kw in keywords):
                triggered_urls.extend(mapping.get("urls", []))
        triggered_urls = list(dict.fromkeys(triggered_urls))
        if triggered_urls:
            live_engine = LiveSearchEngine(engine.embeddings, engine.llm)
            live_answer, live_sources_docs = await asyncio.to_thread(
                live_engine.query_live_data,
                request.question,
                triggered_urls,
            )
            live_sources = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in live_sources_docs
            ]

    answer, sources = await engine.query(
        request.question,
        metadata_filters=request.metadata_filters,
        chat_history=history,
        enable_graph_rag=request.enable_graph_rag,
    )
    source_payload = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in sources
    ]
    metrics = getattr(engine, "last_query_metrics", {}) or {}

    if unified_db:
        try:
            unified_db.append_message(
                session_id=session_id,
                role="user",
                content=request.question,
                rag_version="rag_system"
            )
            unified_db.append_message(
                session_id=session_id,
                role="assistant",
                content=str(answer),
                rag_version="rag_system",
                sources=source_payload,
                metrics=metrics
            )
        except Exception as e:
            logger.error("Failed to save history to MongoDB: %s", e)
    else:
        # Fallback to local sessions
        if session_id in sessions:
            msgs = sessions.pop(session_id)
            sessions[session_id] = msgs
        else:
            sessions[session_id] = []

        sessions[session_id].append({"role": "user", "content": request.question})
        sessions[session_id].append(
            {
                "role": "assistant",
                "content": answer,
                "metrics": metrics,
                "sources": source_payload,
            }
        )
        _save_sessions(sessions)

    return QueryResponse(
        answer=str(answer),
        sources=source_payload,
        metrics=metrics,
        session_id=session_id,
        live_answer=live_answer,
        live_sources=live_sources or None,
    )



@app.post("/api/query/stream")
async def run_query_stream(request: QueryRequest):
    engine = get_engine()
    session_id = request.session_id or _create_session_id(request.question)

    sessions = _load_sessions()

    history = []
    if unified_db:
        try:
            mongo_hist = unified_db.get_chat_history(session_id, limit=max(1, int(request.history_limit)))
            history = [{"role": msg["role"], "content": msg["content"]} for msg in mongo_hist]
        except Exception as e:
            logger.error("Failed to load history from MongoDB: %s", e)
            
    if not history:
        existing = sessions.get(session_id, [])
        if existing:
            history = existing[-max(1, int(request.history_limit)) :]

    async def event_generator():
        full_answer = []
        metadata_payload = None
        
        try:
            async for token in engine.orchestrator.async_run(
                query=request.question,
                history=history,
                enable_graph_rag=request.enable_graph_rag
            ):
                if token.startswith("__METADATA__:"):
                    metadata_payload = token
                    yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'metadata': token})}\n\n"
                    continue
                
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
                
            if not metadata_payload:
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            complete_answer = "".join(full_answer).strip()
            
            source_payload = []
            if metadata_payload:
                try:
                    raw_meta = metadata_payload[len("__METADATA__:"):]
                    source_payload = json.loads(raw_meta)
                except Exception:
                    pass
            
            if unified_db:
                try:
                    unified_db.append_message(
                        session_id=session_id,
                        role="assistant",
                        content=complete_answer,
                        rag_version="rag_system",
                        sources=source_payload,
                        metrics={}
                    )
                except Exception as e:
                    logger.error("Failed to save streaming message to MongoDB: %s", e)
            else:
                local_sessions = _load_sessions()
                if session_id not in local_sessions:
                    local_sessions[session_id] = []
                local_sessions[session_id].append({"role": "user", "content": request.question})
                local_sessions[session_id].append({
                    "role": "assistant",
                    "content": complete_answer,
                    "metrics": {},
                    "sources": source_payload
                })
                _save_sessions(local_sessions)
                
        except Exception as e:
            logger.error("Streaming failed in event_generator: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


class DocumentContentUpdate(BaseModel):
    content: str

@app.get("/api/documents/search")
def search_documents(q: str = Query(..., min_length=1)) -> List[Dict[str, Any]]:
    engine = get_engine()
    sources = engine.get_available_sources()
    query_lower = q.lower()
    results = []
    
    for source in sources:
        safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', source)
        filepath = os.path.join(config.RAG_TMP_DIR, f"{safe_name}_extraction.md")
        matches = 0
        snippets = []
        
        name_match = query_lower in source.lower()
        if name_match:
            matches += 1
            
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
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
                    
        if matches > 0:
            results.append({
                "file_name": source,
                "matches": matches,
                "snippets": snippets
            })
            
    return results

@app.get("/api/documents/{file_name}/content")
def get_document_content(file_name: str) -> Dict[str, Any]:
    safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_name)
    filepath = os.path.join(config.RAG_TMP_DIR, f"{safe_name}_extraction.md")
    if not os.path.exists(filepath):
        # Fallback: Reconstruct from Qdrant chunks
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            engine = get_engine()
            points, _ = engine.db.client.scroll(
                collection_name=engine.db.collection_name,
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
                    os.makedirs(config.RAG_TMP_DIR, exist_ok=True)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(reconstructed_content)
                except Exception as save_err:
                    logger.warning(f"Failed to save reconstructed markdown cache in v1: {save_err}")
                return {"file_name": file_name, "content": reconstructed_content}
        except Exception as e:
            logger.error(f"Failed to reconstruct document from Qdrant in v1: {e}")
        raise HTTPException(status_code=404, detail="Content not found")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return {"file_name": file_name, "content": content}

@app.put("/api/documents/{file_name}/content")
def update_document_content(file_name: str, payload: DocumentContentUpdate) -> Dict[str, Any]:
    engine = get_engine()
    safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_name)
    filepath = os.path.join(config.RAG_TMP_DIR, f"{safe_name}_extraction.md")
    
    os.makedirs(config.RAG_TMP_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(payload.content)
        
    try:
        engine.delete_source(file_name)
    except Exception as e:
        logger.error(f"Error deleting old chunks for {file_name}: {e}")
        
    try:
        from new_ingestion.parser import ParseResult, BlockRecord, DocumentMeta, ExtractionMethod
        import hashlib
        
        file_hash = hashlib.sha256(payload.content.encode('utf-8')).hexdigest()
        
        meta = DocumentMeta(
            doc_id=file_hash[:12],
            file_name=file_name,
            file_path=filepath,
            file_type=".md",
            file_size_kb=len(payload.content) / 1024,
            page_count=1,
            has_tables=False,
            parse_success=True,
            filename_tokens=re.split(r'[_\-\s]+', os.path.splitext(file_name)[0].lower()),
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
        
        chunks = engine.chunker.chunk(parse_res)
        for c in chunks:
            c.file_hash = file_hash
            
        engine.embedder_instance.embed_and_store(chunks)
        
        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        logger.error(f"Error re-ingesting {file_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{file_name}")
def delete_document(file_name: str) -> Dict[str, Any]:
    engine = get_engine()
    try:
        engine.db.delete_document(file_name)
        if engine.graph_rag.available:
            engine.graph_rag.delete_document(file_name)

        # Delete intermediate extraction and chunk md files
        try:
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_name)
            for suffix in ["_extraction.md", "_chunks.md"]:
                old_path = os.path.join(config.RAG_TMP_DIR, f"{safe_name}{suffix}")
                if os.path.exists(old_path):
                    os.remove(old_path)
                    logger.info(f"Deleted tmp file: {old_path}")
        except Exception as file_err:
            logger.error(f"Failed to delete tmp files for {file_name}: {file_err}")

        return {"status": "success", "message": f"Deleted {file_name}"}
    except Exception as e:
        logger.error(f"Error deleting document {file_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents")
def clear_all_documents() -> Dict[str, Any]:
    engine = get_engine()
    try:
        # Delete and setup the database collection fresh to clear it completely and safely
        engine.db.client.delete_collection(engine.db.collection_name)
        engine.db.setup_database()
        if engine.graph_rag.available:
            engine.graph_rag.clear_all_documents()

        # Clear all intermediate md files in the tmp directory
        try:
            for f in os.listdir(config.RAG_TMP_DIR):
                if f.endswith(".md"):
                    try:
                        os.remove(os.path.join(config.RAG_TMP_DIR, f))
                    except Exception as file_err:
                        logger.error(f"Failed to delete {f}: {file_err}")
        except Exception as dir_err:
            logger.error(f"Failed to clear RAG_TMP_DIR: {dir_err}")

        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Error clearing all documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/preview-parse")
async def preview_parse_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parse a document to markdown in memory without saving or ingesting."""
    engine = get_engine()
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file for preview: {e}")
        raise HTTPException(status_code=500, detail="Could not read file.")

    try:
        adapter = _UploadedFileAdapter(file.filename, file_bytes)
        parse_res = engine.parser.parse_bytes(adapter.getvalue(), file.filename, generate_metadata=False)
        if not parse_res.success:
            raise Exception(f"Parsing failed: {parse_res.error}")

        return {
            "file_name": file.filename,
            "content": parse_res.markdown,
        }
    except Exception as e:
        logger.error(f"Preview parse failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/replace")
async def replace_document(
    file: UploadFile = File(...),
    old_file_name: str = Form(...),
) -> Dict[str, Any]:
    """Delete an existing document and ingest a new one in its place."""
    engine = get_engine()

    # Step 1: Delete old document from Qdrant + Neo4j asynchronously
    try:
        await engine.db.async_delete_document(old_file_name)
        if engine.graph_rag.available:
            await asyncio.to_thread(engine.graph_rag.delete_document, old_file_name)
        logger.info(f"Deleted old document '{old_file_name}' from vector stores.")
    except Exception as e:
        logger.error(f"Failed to delete old document {old_file_name}: {e}")

    # Step 2: Delete old tmp files
    try:
        safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', old_file_name)
        for suffix in ["_extraction.md", "_chunks.md"]:
            old_path = os.path.join(config.RAG_TMP_DIR, f"{safe_name}{suffix}")
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info(f"Deleted old file: {old_path}")
    except Exception as e:
        logger.error(f"Failed to delete old tmp files for {old_file_name}: {e}")

    # Step 3: Ingest the new file (full pipeline) asynchronously
    try:
        file_bytes = await file.read()
        adapter = _UploadedFileAdapter(file.filename, file_bytes)
        chunks_indexed, source_name = await engine.ingest_file(
            uploaded_file=adapter,
            mode=_parse_parsing_mode("SMART"),
            build_graph=False,
        )
        logger.info(f"Successfully replaced '{old_file_name}' with '{file.filename}' ({chunks_indexed} chunks)")
        return {
            "status": "success",
            "old_deleted": old_file_name,
            "new_file": file.filename,
            "chunks_indexed": chunks_indexed,
        }
    except Exception as e:
        logger.error(f"Replace pipeline failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/query_trace")
def get_query_trace() -> Dict[str, Any]:
    engine = get_engine()
    trace = engine.orchestrator.retriever.last_query_trace if hasattr(engine.orchestrator.retriever, "last_query_trace") else {"stages": []}
    metrics = engine.last_query_metrics if hasattr(engine, "last_query_metrics") else {}
    return {
        "status": "success",
        "trace": trace,
        "metrics": metrics
    }

@app.get("/api/stats")
def get_stats() -> Dict[str, Any]:
    engine = get_engine()
    
    # 1. Chunks count
    try:
        num_chunks = engine.client.count(engine.db.collection_name).count
    except Exception as e:
        num_chunks = 0
        
    # 2. Extract unique filenames from payloads
    files = set()
    try:
        points, _ = engine.client.scroll(
            collection_name=engine.db.collection_name, 
            limit=10000, 
            with_payload=["source_file"], 
            with_vectors=False
        )
        for point in points:
            src = point.payload.get("source_file")
            if src:
                files.add(str(src))
    except Exception as e:
        logger.error(f"Error fetching Qdrant files: {e}")
        pass

    # 3. Build files_meta with real mtime from extraction md files
    files_meta = {}
    for fname in files:
        try:
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', fname)
            filepath = os.path.join(config.RAG_TMP_DIR, f"{safe_name}_extraction.md")
            if os.path.exists(filepath):
                from datetime import datetime, timezone
                mtime = os.path.getmtime(filepath)
                files_meta[fname] = {"upload_time": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()}
        except Exception as e:
            logger.error(f"Error getting mtime for {fname}: {e}")
        
    # 4. Neo4j Status & Stats
    graph_stats = {"connected": False, "entities": 0, "relationships": 0}
    if engine.graph_rag.enabled and engine.graph_rag.driver:
        try:
            with engine.graph_rag.driver.session() as session:
                entities = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                relationships = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
                graph_stats["entities"] = entities
                graph_stats["relationships"] = relationships
                graph_stats["connected"] = True
        except Exception as e:
            logger.error(f"Error fetching Neo4j stats: {e}")
            pass

    return {
        "status": "online",
        "num_chunks": num_chunks,
        "files": list(files),
        "files_meta": files_meta,
        "graph": graph_stats
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
