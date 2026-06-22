import os
import shutil
import time
import json
import logging
import hashlib
import asyncio
import concurrent.futures
import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional, Generator
from collections import OrderedDict

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

import config
from new_ingestion.parser import DocumentParser
from new_ingestion.chunker import DocumentChunker
from new_ingestion.embedder import OllamaEmbedder
from qdrant_storage.store import QdrantManager
from agent.orchestrator import get_orchestrator
from agent.validator import get_validator
from graph_rag import GraphRAGManager

logger = logging.getLogger("rag_engine")

class QueryCache:
    def __init__(self, db_path: str = "./qdrant_db/query_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        cache_key TEXT PRIMARY KEY,
                        answer TEXT,
                        metrics_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize query cache SQLite DB: %s", e)

    def get(self, cache_key: str) -> Optional[Tuple[str, dict]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT answer, metrics_json FROM query_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0], json.loads(row[1])
        except Exception as e:
            logger.error("Failed to read query cache: %s", e)
        return None

    def set(self, cache_key: str, answer: str, metrics: dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO query_cache (cache_key, answer, metrics_json) VALUES (?, ?, ?)",
                    (cache_key, answer, json.dumps(metrics))
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to write query cache: %s", e)

    def clear(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM query_cache")
                conn.commit()
        except Exception as e:
            logger.error("Failed to clear query cache: %s", e)

class EmbeddingsAdapter:
    def __init__(self, embedder: OllamaEmbedder):
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        dense, _ = self.embedder.embed_query(text)
        return dense

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"passage: {t}" for t in texts]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.embedder._encode_passages_async(prefixed_texts))
                vectors = future.result()
            return [v.tolist() for v in vectors if v is not None]
        except Exception as e:
            logger.error("Embed documents failed in adapter: %s", e)
            return []

    async def async_embed_query(self, text: str) -> list[float]:
        dense, _ = await self.embedder.async_embed_query(text)
        return dense

    async def async_embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"passage: {t}" for t in texts]
        try:
            vectors = await self.embedder._encode_passages_async(prefixed_texts)
            return [v.tolist() for v in vectors if v is not None]
        except Exception as e:
            logger.error("Async embed documents failed in adapter: %s", e)
            return []


class LLMAdapter:
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=300.0)
        self.model = config.GENERATOR_MODEL

    def invoke(self, prompt: str, options: dict = None, model: str = None, raise_on_error: bool = False) -> str:
        try:
            default_options = {
                "temperature": 0.0,
                "num_ctx": 8192
            }
            if options:
                default_options.update(options)
            response = self.client.generate(
                model=model or self.model,
                prompt=prompt,
                think=False,
                options=default_options
            )
            return response.get("response", "")
        except Exception as e:
            logger.error("LLM invoke failed in adapter: %s", e)
            if raise_on_error:
                raise
            return ""

class RAGEngine:
    def __init__(self, **kwargs):
        # Initialize modular database connection
        self.db = QdrantManager()
        self.db.setup_database()
        self.client = self.db.get_client()

        # Build embedder and adapters
        self.embedder_instance = OllamaEmbedder(self.db)
        self.embeddings = EmbeddingsAdapter(self.embedder_instance)
        self.llm = LLMAdapter()

        # GraphRAG Setup
        self.graph_rag = GraphRAGManager(
            enabled=config.ENABLE_GRAPH_RAG,
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE,
            domain=config.GRAPH_DOMAIN,
            domain_description=config.GRAPH_DOMAIN_DESCRIPTION,
            preferred_entity_types=config.GRAPH_PREFERRED_ENTITY_TYPES,
            preferred_relation_types=config.GRAPH_PREFERRED_RELATION_TYPES,
            max_hops=config.GRAPH_MAX_HOPS,
            max_query_entities=config.GRAPH_MAX_QUERY_ENTITIES,
            max_graph_chunks=config.GRAPH_TOP_K,
            max_chunks_per_entity=config.GRAPH_MAX_CHUNKS_PER_ENTITY,
            semantic_merge_enabled=config.GRAPH_SEMANTIC_MERGE_ENABLED,
            semantic_merge_threshold=config.GRAPH_SEMANTIC_MERGE_THRESHOLD,
            extraction_timeout=config.GRAPH_EXTRACTION_TIMEOUT,
            extraction_retries=config.GRAPH_EXTRACTION_RETRIES,
        )

        # Agent Orchestrator and Validator
        self.orchestrator = get_orchestrator(embedder=self.embedder_instance, graph_rag=self.graph_rag)
        self.validator = get_validator()

        # Ingestion processors
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()

        # Telemetry
        self.telemetry_log_path = getattr(config, "TELEMETRY_LOG_PATH", "./qdrant_db/telemetry.jsonl")
        self.enable_telemetry_logging = getattr(config, "ENABLE_TELEMETRY_LOGGING", True)
        if self.enable_telemetry_logging:
            telemetry_dir = os.path.dirname(self.telemetry_log_path)
            if telemetry_dir:
                os.makedirs(telemetry_dir, exist_ok=True)

        self.last_query_metrics: dict = {}
        self.query_cache = QueryCache(getattr(config, "QUERY_CACHE_DB_PATH", "./qdrant_db/query_cache.db"))

    def _collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.db.collection_name for c in collections)
        except Exception:
            return False

    async def ingest_file(
        self,
        file_path: str = None,
        uploaded_file = None,
        progress_callback: Optional[callable] = None,
        mode = "SMART",
        build_graph: bool = False,
    ) -> Tuple[int, str]:
        file_hash = None
        source_name = uploaded_file.name if uploaded_file else os.path.basename(file_path)

        if uploaded_file:
            content = uploaded_file.getvalue()
            file_hash = hashlib.sha256(content).hexdigest()
            uploaded_file.seek(0)
        elif file_path:
            def read_file_hash():
                with open(file_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
            file_hash = await asyncio.to_thread(read_file_hash)

        # Deduplication Check
        if self._collection_exists() and file_hash:
            points, _ = await self.db.async_client.scroll(
                collection_name=self.db.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
                ),
                limit=1,
                with_payload=True
            )
            if points:
                logger.info("File %s already ingested (Hash Match). Skipping.", source_name)
                return 0, source_name

        # Parsing step
        if progress_callback:
            progress_callback("Parsing document structural layout...", 0.2)
        if uploaded_file:
            parse_res = await asyncio.to_thread(self.parser.parse_bytes, content, source_name)
        else:
            parse_res = await asyncio.to_thread(self.parser.parse_file, file_path)
        if not parse_res.success:
            raise RuntimeError(f"Parsing failed: {parse_res.error}")

        # --- Debug dump extraction ---
        try:
            import re
            os.makedirs(config.RAG_TMP_DIR, exist_ok=True)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', source_name)
            
            def write_debug_md():
                with open(config.RAG_TMP_DIR / "latest_extraction.md", "w", encoding="utf-8") as f:
                    f.write(parse_res.markdown)
                with open(config.RAG_TMP_DIR / f"{safe_name}_extraction.md", "w", encoding="utf-8") as f:
                    f.write(parse_res.markdown)
            await asyncio.to_thread(write_debug_md)
        except Exception as e:
            logger.warning("Could not save extraction dump: %s", e)

        # Chunking step
        if progress_callback:
            progress_callback("Generating optimal chunks and deduping...", 0.5)
        chunks = await asyncio.to_thread(self.chunker.chunk, parse_res, self.embeddings.embed_documents)
        if not chunks:
            return 0, source_name

        # --- Debug dump chunks ---
        try:
            import re
            import json
            os.makedirs(config.RAG_TMP_DIR, exist_ok=True)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', source_name)
            
            def write_debug_chunks():
                with open(config.RAG_TMP_DIR / "latest_chunks.md", "w", encoding="utf-8") as f:
                    for c in chunks:
                        f.write(f"--- Chunk {c.chunk_index} ({c.token_count} tokens) ---\n")
                        f.write(f"Tags: Page {c.page_label} | Section: {c.section} | Table: {c.is_table}\n\n")
                        f.write(c.text + "\n\n")

                with open(config.RAG_TMP_DIR / f"{safe_name}_chunks.md", "w", encoding="utf-8") as f:
                    for c in chunks:
                        f.write(f"## Chunk {c.chunk_index}\n\n")
                        f.write("```json\n")
                        meta = {
                            "chunk_id": c.chunk_id,
                            "chunk_index": c.chunk_index,
                            "page_no": c.page_no,
                            "page_label": c.page_label,
                            "page_range": c.page_range,
                            "section": c.section,
                            "is_table": c.is_table,
                            "token_count": c.token_count,
                            "chunk_year": c.chunk_year,
                            "doc_id": c.doc_id,
                            "source_file": c.source_file,
                            "filename_tokens": c.filename_tokens,
                            "doc_year": c.doc_year,
                            "has_tables": c.has_tables,
                            "keywords": c.keywords
                        }
                        f.write(json.dumps(meta, indent=2) + "\n")
                        f.write("```\n\n")
                        f.write("### Text\n\n")
                        f.write(c.text + "\n\n")
                        f.write("---\n\n")
            await asyncio.to_thread(write_debug_chunks)
        except Exception as e:
            logger.warning("Could not save chunk debug files: %s", e)

        # Embedding & Storage
        if progress_callback:
            progress_callback("Embedding chunks and building indices...", 0.8)
        for c in chunks:
            c.file_hash = file_hash
        await self.embedder_instance.async_embed_and_store(chunks)

        # GraphRAG building
        if build_graph and self.graph_rag.available:
            if progress_callback:
                progress_callback("Building Knowledge Graph (GraphRAG)...", 0.9)
            langchain_docs = []
            for c in chunks:
                langchain_docs.append(Document(
                    page_content=c.text,
                    metadata={
                        "source": c.source_file,
                        "chunk_id": c.chunk_id,
                        "page_label": c.page_label,
                        "file_hash": file_hash,
                        "full_table_context": getattr(c, "full_table_context", "")
                    }
                ))
            def index_in_thread():
                return self.graph_rag.index_documents(
                    langchain_docs,
                    llm_invoke=self._graph_llm_invoke,
                    batch_embedding_fn=self.embeddings.embed_documents
                )
            stats = await asyncio.to_thread(index_in_thread)
            logger.info("GraphRAG Indexing Stats: %s", stats)

        await asyncio.to_thread(self.query_cache.clear)
        import gc
        gc.collect()
        return len(chunks), source_name

    def _graph_llm_invoke(self, prompt: str, timeout_sec: float = config.GRAPH_LLM_TIMEOUT_SEC, retries: int = 1) -> str:
        graph_model: str = str(getattr(config, "GRAPH_LLM_MODEL", "") or self.llm.model or "")

        def _invoke():
            return self.llm.invoke(prompt, options={
                "temperature": config.GRAPH_LLM_TEMPERATURE,
                "num_ctx": getattr(config, "GRAPH_LLM_NUM_CTX", 8192),
                "num_predict": getattr(config, "GRAPH_LLM_NUM_PREDICT", 2048),
            }, model=graph_model, raise_on_error=True)
        
        last_exc = RuntimeError("LLM invoke failed: unknown error")
        total_attempts = max(1, retries)
        
        for attempt in range(total_attempts):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(_invoke)
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                last_exc = TimeoutError(f"Ollama invoke timed out after {timeout_sec}s (attempt {attempt+1}/{total_attempts})")
                logger.error(f"[GraphRAG] LLM timeout on attempt {attempt+1}/{total_attempts}")
            except Exception as e:
                import traceback
                last_exc = e
                logger.error(f"[GraphRAG] LLM error on attempt {attempt+1}/{total_attempts}: {e}")
                logger.error(traceback.format_exc())
            finally:
                executor.shutdown(wait=False)
        
        raise last_exc

    def retrieve(self, question: str, metadata_filters: Optional[dict] = None) -> List[Document]:
        if not self._collection_exists():
            return []
        cleaned = self.orchestrator.cleaner.clean(question)
        if metadata_filters:
            for k, v in metadata_filters.items():
                if v:
                    cleaned.filter_hints[k] = v
        chunks = self.orchestrator.retriever.retrieve(cleaned, enable_graph_rag=self.graph_rag.available)
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={"source": chunk.source_file, "chunk_id": getattr(chunk, 'chunk_id', str(chunk.chunk_index))}
            )
            documents.append(doc)
        return documents

    async def query(
        self,
        question: str,
        metadata_filters: Optional[dict] = None,
        chat_history: Optional[List[dict]] = None,
        status_callback: Optional[callable] = None,
        enable_graph_rag: bool = True,
    ) -> Tuple[str, List[Document]]:
        if not self._collection_exists():
            return "No documents have been uploaded yet. Please upload a document first.", []

        # Step 1: Query analysis & Cleaner
        print("\n=== DEBUG: Entering query engine ===")
        print(f"DEBUG: Question: '{question}'")
        if status_callback:
            status_callback("Analyzing query intent and keywords...", 0.2)
        
        print("DEBUG: Calling query cleaner...")
        cleaned = await asyncio.to_thread(self.orchestrator.cleaner.clean, question)
        print("DEBUG: Query cleaner completed successfully!")

        # Merge manual frontend filters into cleaner hints
        if metadata_filters:
            for k, v in metadata_filters.items():
                if v:
                    cleaned.filter_hints[k] = v

        # Step 2: Context Retrieval
        if status_callback:
            status_callback("Retrieving relevant context...", 0.5)
        chunks = await self.orchestrator.retriever.async_retrieve(cleaned, enable_graph_rag=enable_graph_rag)

        # Convert to langchain documents
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source_file,
                    "source_file": chunk.source_file,
                    "page_no": chunk.page_no,
                    "page_label": chunk.page_label,
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section,
                    "is_table": chunk.is_table,
                    "doc_year": chunk.doc_year,
                    "doc_id": chunk.doc_id,
                    "chunk_year": chunk.chunk_year,
                    "token_count": chunk.token_count,
                    "retrieval_score": chunk.rerank_score,
                    "rerank_score": chunk.rerank_score,
                    "is_weak_match": chunk.is_weak_match,
                    "source_tag": chunk.source_tag,
                    "arm_label": chunk.arm_label,
                    "summary": chunk.summary,
                    "graph_score": chunk.graph_score,
                    "graph_hops": chunk.graph_hops,
                    "graph_path": chunk.graph_path,
                    "retrieval_channel": chunk.retrieval_channel,
                }
            )
            documents.append(doc)

        # Check Cache
        doc_ids = [doc.metadata.get("chunk_id", "") for doc in documents]
        filters_str = json.dumps(metadata_filters or {}, sort_keys=True)
        history_str = json.dumps(chat_history or [], sort_keys=True)
        docs_str = ",".join(sorted(doc_ids))
        raw_key = f"{question}||{filters_str}||{history_str}||{enable_graph_rag}||{docs_str}"
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        cached_val = await asyncio.to_thread(self.query_cache.get, cache_key)
        if cached_val is not None:
            cached_answer, cached_metrics = cached_val
            logger.info("Query cache hit for question: '%s'", question)
            self.last_query_metrics = cached_metrics
            return cached_answer, documents

        # Step 3: Assembly
        assembled = await asyncio.to_thread(self.orchestrator.assembler.assemble, cleaned, chunks)
        if assembled.not_found:
            self.last_query_metrics = {
                "confidence": 0.0,
                "validation_passed": True,
                "validation_reason": "not_found",
                "source_count": 0,
                "filters": metadata_filters or {},
                "query_keyword_overlap": 0.0,
            }
            return config.FALLBACK_ANSWER, []

        # Step 4: Generation
        if status_callback:
            status_callback("Expert reasoning in progress...", 0.8)
        history_str = self.orchestrator._format_history(chat_history or [])

        logger.debug("Assembled context length: %d chars", len(assembled.context_block))

        def run_generation():
            generator = self.orchestrator._stream_generate(
                original_query=question,
                improved_query=cleaned.improved_query,
                detected_language=cleaned.detected_language,
                assembled=assembled,
                history_str=history_str
            )
            return "".join(generator)
            
        answer = await asyncio.to_thread(run_generation)

        print("\n\n" + "="*50)
        print("DEBUG: RAW LLM OUTPUT:")
        print(answer)
        print("="*50 + "\n\n")

        if not answer or not answer.strip():
            answer = config.FALLBACK_ANSWER

        # Step 5: Validation & Metrics
        # Skip expensive validation when disabled
        if getattr(config, 'ENABLE_ANSWER_VALIDATION', False):
            context_text = assembled.context_block
            validation_passed = await asyncio.to_thread(self.validator.validate, answer, context_text)
            validation_reason = self.validator.last_reason

            if not validation_passed and validation_reason == "citation_format":
                validation_passed = True
            elif not validation_passed and documents:
                answer = self._build_extractive_answer(documents)
                validation_reason = f"fallback_after_{validation_reason or 'validation_failed'}"

            confidence = self._compute_confidence(answer, documents, validation_passed, context_text)
        else:
            validation_passed = True
            validation_reason = "disabled"
            # Lightweight confidence from retrieval scores only
            top_scores = [float(doc.metadata.get("retrieval_score", 0.0)) for doc in documents[:3]]
            confidence = round(sum(top_scores) / max(1, len(top_scores)), 3)

        query_overlap = self._query_context_overlap(question, documents)

        self.last_query_metrics = {
            "confidence": confidence,
            "validation_passed": validation_passed,
            "validation_reason": validation_reason,
            "source_count": len(documents),
            "filters": metadata_filters or {},
            "query_keyword_overlap": round(query_overlap, 3),
        }

        await asyncio.to_thread(
            self._log_query_telemetry,
            question=question,
            answer=answer,
            docs=documents,
            validation_passed=validation_passed,
            confidence=confidence,
            metadata_filters=metadata_filters,
        )

        await asyncio.to_thread(self.query_cache.set, cache_key, answer, self.last_query_metrics)
        return answer, documents

    def _build_extractive_answer(self, docs: List[Document]) -> str:
        lines = []
        for doc in docs[:3]:
            source = doc.metadata.get("source", "Unknown")
            page_label = doc.metadata.get("page_label", doc.metadata.get("page_no", "?"))
            text = doc.page_content.strip()
            text = " ".join(text.split())
            if len(text) > 800:
                text = text[:800] + "..."
            lines.append(f"- {text} [Source: {source}, Page {page_label}]")
        return "\n".join(lines)

    def _query_context_overlap(self, query: str, docs: List[Document]) -> float:
        query_tokens = [t for t in self.validator._tokenize_for_keyword(query) if len(t) > 2]
        if not query_tokens:
            return 1.0
        context_text = "\n".join(doc.page_content for doc in docs)
        context_tokens = set(t for t in self.validator._tokenize_for_keyword(context_text) if len(t) > 2)
        if not context_tokens:
            return 0.0
        unique_query_tokens = set(query_tokens)
        matched = sum(1 for token in unique_query_tokens if token in context_tokens)
        return matched / max(1, len(unique_query_tokens))

    def _compute_confidence(
        self,
        answer: str,
        docs: List[Document],
        validation_passed: bool,
        context_text: str,
    ) -> float:
        top_scores = [float(doc.metadata.get("retrieval_score", 0.0)) for doc in docs[:3]]
        retrieval_strength = sum(top_scores) / max(1, len(top_scores))
        return self.validator.compute_confidence(answer, context_text, retrieval_strength, validation_passed)

    def _log_query_telemetry(
        self,
        question: str,
        answer: str,
        docs: List[Document],
        validation_passed: bool,
        confidence: float,
        metadata_filters: Optional[dict] = None,
    ) -> None:
        if not self.enable_telemetry_logging:
            return
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "question": question,
            "answer": answer,
            "validation_passed": validation_passed,
            "confidence": confidence,
            "metadata_filters": metadata_filters or {},
            "sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "?"),
                    "page_no": doc.metadata.get("page_no", 1),
                    "page_label": doc.metadata.get("page_label", "1"),
                    "retrieval_score": float(doc.metadata.get("retrieval_score", 0.0)),
                }
                for doc in docs
            ],
        }
        try:
            with open(self.telemetry_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Telemetry logging failed: %s", e)

    def get_available_sources(self) -> List[str]:
        if not self._collection_exists():
            return []
        sources = set()
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.db.collection_name,
                limit=config.QDRANT_SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                source = payload.get("source_file") or payload.get("source")
                if source:
                    sources.add(str(source))
            if next_offset is None:
                break
            offset = next_offset
        return sorted(sources)

    def get_available_sections(self) -> List[str]:
        if not self._collection_exists():
            return []
        sections_with_pages = {}
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.db.collection_name,
                limit=config.QDRANT_SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                section = payload.get("section")
                page = payload.get("page_label") or payload.get("page_no")
                if section and section not in sections_with_pages:
                    sections_with_pages[str(section)] = str(page) if page else "?"
            if next_offset is None:
                break
            offset = next_offset
        formatted = []
        for sec, pg in sections_with_pages.items():
            formatted.append(f"{sec} (Page {pg})")
        return sorted(formatted)

    def get_collection_stats(self) -> dict:
        if not self._collection_exists():
            graph_status = self.graph_rag.status() if getattr(self, "graph_rag", None) else {}
            return {"status": "empty", "num_chunks": 0, "graph": graph_status}
        try:
            collection_info = self.client.get_collection(self.db.collection_name)
            count = collection_info.points_count
            graph_status = self.graph_rag.status() if getattr(self, "graph_rag", None) else {}
            return {
                "status": "ready",
                "mode": "hybrid",
                "num_chunks": count,
                "persist_directory": config.QDRANT_PATH,
                "graph": graph_status,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clear_database(self):
        if getattr(self, "graph_rag", None):
            self.graph_rag.clear()
        try:
            self.client.close()
        except Exception as e:
            logger.warning("Warning during client close: %s", e)

        qdrant_path = config.QDRANT_PATH
        if os.path.exists(qdrant_path):
            for attempt in range(3):
                try:
                    shutil.rmtree(qdrant_path, ignore_errors=True)
                    if not os.path.exists(qdrant_path):
                        break
                except Exception as e:
                    logger.warning("Could not remove Qdrant DB: %s", e)
                time.sleep(1.0)
        
        self.db.setup_database()
        self.client = self.db.get_client()
        self.query_cache.clear()
        logger.info("Database cleared and setup fresh.")

    def delete_source(self, source_name: str) -> bool:
        if not self._collection_exists():
            return False
        try:
            if getattr(self, "graph_rag", None):
                self.graph_rag.delete_source(
                    source_name,
                    qdrant_client=self.client,
                    batch_embedding_fn=self.embeddings.embed_documents
                )
            self.client.delete(
                collection_name=self.db.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_name)
                        )
                    ]
                )
            )
            self.query_cache.clear()
            logger.info("Deleted source '%s' successfully.", source_name)
            return True
        except Exception as e:
            logger.error("Error deleting source '%s': %s", source_name, e)
            return False
