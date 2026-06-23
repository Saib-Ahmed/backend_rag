"""
ingestion/embedder.py
Qwen3-Embedding:4b via Ollama — dense vectors
FastEmbed BM25 — sparse vectors
Hybrid RAG pipeline for Qdrant
"""

from __future__ import annotations
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict
import concurrent.futures
import httpx
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector

from new_ingestion.chunker import ChunkResult
from qdrant_storage.store import QdrantManager
import config
import threading

logger = logging.getLogger("ingestion.embedder")

_EMBEDDING_CACHE = []
_EMBEDDING_CACHE_LOCK = threading.Lock()


# ── Constants ──────────────────────────────────────────────────────────
MODEL_NAME         = config.EMBED_MODEL
OLLAMA_URL         = config.OLLAMA_EMBED_URL
DEFAULT_BATCH_SIZE = config.EMBED_BATCH_SIZE
MAX_TOKENS         = config.EMBED_MAX_TOKENS
SPARSE_MODEL_NAME  = config.SPARSE_MODEL_NAME
QUERY_PREFIX       = "query: "
PASSAGE_PREFIX     = "passage: "


# ── Tiktoken loader ────────────────────────────────────────────────────
def _load_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

_TOKENIZER = _load_tokenizer()

def _count_tokens(text: str) -> int:
    if _TOKENIZER is None:
        return len(text.split())
    return len(_TOKENIZER.encode(text))

def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if _TOKENIZER is None:
        words = text.split()
        return " ".join(words[:max_tokens]) if len(words) > max_tokens else text
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


# ── Data class ─────────────────────────────────────────────────────────
@dataclass
class EmbeddedChunk:
    chunk_index:     int
    text:            str
    page_no:         int             = 0
    page_label:      str             = ""
    page_range:      tuple[int, int] = (0, 0)
    section:         str             = ""
    is_table:        bool            = False
    token_count:     int             = 0
    chunk_year:      list[str]       = field(default_factory=list)
    doc_id:          str             = ""
    chunk_id:        str             = ""
    source_file:     str             = ""
    filename_tokens: list[str]       = field(default_factory=list)
    doc_year:        str             = ""
    has_tables:      bool            = False
    summary:         str             = ""
    keywords:        list[str]       = field(default_factory=list)
    file_hash:       str             = ""
    dense_vector:    Optional[np.ndarray] = field(default=None, repr=False)
    sparse_indices:  list[int]            = field(default_factory=list, repr=False)
    sparse_values:   list[float]          = field(default_factory=list, repr=False)

    @classmethod
    def from_chunk(cls, chunk: ChunkResult, **kwargs) -> "EmbeddedChunk":
        return cls(
            chunk_index     = chunk.chunk_index,
            text            = chunk.text,
            page_no         = chunk.page_no,
            page_range      = tuple(chunk.page_range),
            page_label      = chunk.page_label,
            section         = chunk.section,
            is_table        = chunk.is_table,
            chunk_year      = chunk.chunk_year,
            doc_id          = chunk.doc_id,
            chunk_id        = chunk.chunk_id,
            source_file     = chunk.source_file,
            filename_tokens = chunk.filename_tokens,
            doc_year        = chunk.doc_year,
            has_tables      = chunk.has_tables,
            summary         = chunk.summary,
            keywords        = chunk.keywords,
            file_hash       = getattr(chunk, "file_hash", kwargs.get("file_hash", "")),
            **kwargs,
        )


# ── Main embedder ──────────────────────────────────────────────────────
class OllamaEmbedder:

    def __init__(
        self,
        db:         QdrantManager,
        model_name: str = MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        ollama_url: str = OLLAMA_URL,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.ollama_url = ollama_url

        logger.info("Loading FastEmbed BM25 sparse model...")
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        logger.info("FastEmbed BM25 ready | model=%s", SPARSE_MODEL_NAME)

        self.db = db
        self._verify_ollama()
        self._query_client = httpx.Client(timeout=300.0)

        logger.info(
            "OllamaEmbedder ready | model=%s | batch=%d | url=%s",
            model_name, batch_size, ollama_url,
        )

    # ── Verify Ollama + dimension check ───────────────────────────────
    def _verify_ollama(self) -> None:
        base_url   = self.ollama_url.rsplit("/api/", 1)[0]
        health_url = f"{base_url}/api/tags"

        try:
            r = httpx.get(health_url, timeout=10.0)
            r.raise_for_status()
            logger.info("Ollama reachable | %s", health_url)
        except Exception as e:
            logger.error("Ollama not reachable | %s", e)
            raise RuntimeError(
                f"Ollama not reachable at {health_url}. "
                f"Run: ollama serve && ollama pull {self.model_name}"
            )

        try:
            probe_payload = {"model": self.model_name, "prompt": "test"}
            r = httpx.post(self.ollama_url, json=probe_payload, timeout=300.0)
            r.raise_for_status()
            actual_dim = len(r.json()["embedding"])
            if actual_dim != config.EMBED_DIMENSIONS:
                raise RuntimeError(
                    f"Model '{self.model_name}' returns {actual_dim}-dim vectors "
                    f"but config.EMBED_DIMENSIONS={config.EMBED_DIMENSIONS}. "
                    f"Update config or switch model."
                )
            logger.info("Embedding dimension verified | %d", actual_dim)
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Could not verify embedding dimensions | %s", e)

    # ── Build embed input — CHANGE: passage prefix, no year stamp ─────
    def _build_embed_input(self, chunk: ChunkResult) -> str:
        return PASSAGE_PREFIX + chunk.text

    # ── Sparse model guard helper ──────────────────────────────────────
    def _ensure_sparse_model(self) -> None:
        if self.sparse_model is None:
            logger.info("Reloading sparse model...")
            self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    # ── Sparse vector — FastEmbed BM25 ────────────────────────────────
    def _sparse_embed(self, text: str) -> tuple[list[int], list[float]]:
        self._ensure_sparse_model()  # CHANGE: guard added
        result = list(self.sparse_model.embed([text]))[0]
        return result.indices.tolist(), result.values.tolist()

    def _sparse_query(self, query: str) -> tuple[list[int], list[float]]:
        self._ensure_sparse_model()
        result = list(self.sparse_model.query_embed([query]))[0]
        return result.indices.tolist(), [float(v) for v in result.values]

    # ── Ingest: embed + store in Qdrant ───────────────────────────────
    def embed_and_store(self, chunks: list[ChunkResult], checkpoint_file=None) -> None:
        if not chunks:
            logger.warning("No chunks to embed. Skipping.")
            return

        source_file = chunks[0].source_file
        total       = len(chunks)
        logger.info("Embedding | file=%s | chunks=%d", source_file, total)

        # Step 1 — sort by token_count for batch efficiency
        indexed  = list(enumerate(chunks))
        sorted_  = sorted(indexed, key=lambda x: x[1].token_count)
        orig_idx = [i for i, _ in sorted_]

        # Step 2 — build enriched texts + recalculate token counts
        texts                 = [self._build_embed_input(c) for _, c in sorted_]
        enriched_token_counts = [_count_tokens(t) for t in texts]

        # Step 3 — sparse vectors via FastEmbed BM25
        sparse_results = [self._sparse_embed(t) for t in texts]

        # Step 4 — dense vectors via Qwen3 async
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future     = pool.submit(asyncio.run, self._encode_passages_async(texts))
            dense_vecs = future.result()

        # Step 5 — reorder back to original index
        reordered = [None] * total
        for j, orig_i in enumerate(orig_idx):
            indices, values = sparse_results[j]
            chunk           = chunks[orig_i]
            reordered[orig_i] = EmbeddedChunk.from_chunk(
                chunk,
                dense_vector   = dense_vecs[j],
                sparse_indices = indices,
                sparse_values  = values,
                token_count    = enriched_token_counts[j],
            )

        # Step 6 — delete ONLY after all embeddings are ready
        self.db.delete_document(source_file)

        # Step 7 — upsert in batches
        points_stored = 0
        for start in range(0, total, self.batch_size):
            batch  = reordered[start: start + self.batch_size]
            points = [self._build_point(ec) for ec in batch if ec is not None]
            self.db.client.upsert(
                collection_name=config.QDRANT_COLLECTION_NAME,
                points=points, 
            )
            points_stored += len(points)
            logger.info("  Upserted %d/%d", min(start + self.batch_size, total), total)

        logger.info(
            "✓ Ingestion complete | file=%s | stored=%d | skipped=%d | all steps passed",
            source_file, points_stored, total - points_stored,
        )

    # ── Batch ingestion ────────────────────────────────────────────────
    def embed_batch(self, chunks_by_file: dict[str, list[ChunkResult]]) -> None:
        for file_name, chunks in chunks_by_file.items():
            self.embed_and_store(chunks)
        total = sum(len(c) for c in chunks_by_file.values())
        logger.info(
            "Batch complete | files=%d | total_chunks=%d",
            len(chunks_by_file), total,
        )
        self.cleanup_sparse_model()  # CHANGE: actually call after batch done

    def cleanup_sparse_model(self) -> None:
        try:
            del self.sparse_model
            self.sparse_model = None
            logger.info("Sparse model cleaned up")
        except Exception as e:
            logger.warning("Cleanup failed: %s", e)

    # ── Query embedding ────────────────────────────────────────────────
    def embed_query(self, query: str) -> tuple[list[float], Dict[int, float]]:
        cleaned_query = query.strip()
        
        # 1. Exact string match check against _EMBEDDING_CACHE
        with _EMBEDDING_CACHE_LOCK:
            for entry in _EMBEDDING_CACHE:
                if entry["query"] == cleaned_query:
                    logger.info("Cache hit (exact match) for query: '%s'", cleaned_query)
                    return entry["dense"], entry["sparse"]
                    
        # 2. Fetch the dense embedding from Ollama
        prefixed = QUERY_PREFIX + cleaned_query
        prefixed = _truncate_by_tokens(prefixed, MAX_TOKENS)

        payload         = {"model": self.model_name, "prompt": prefixed}
        r               = self._query_client.post(self.ollama_url, json=payload)
        r.raise_for_status()
        vec             = np.array(r.json()["embedding"], dtype=np.float32)
        dense           = _normalize(vec)
        dense_list      = dense.tolist()
        
        # 3. Compare the dot product of this dense embedding with the dense embeddings of all items in _EMBEDDING_CACHE
        with _EMBEDDING_CACHE_LOCK:
            for entry in _EMBEDDING_CACHE:
                sim = np.dot(dense, np.array(entry["dense"], dtype=np.float32))
                if sim >= config.GRAPH_EMBEDDING_SIMILARITY_THRESHOLD:
                    logger.info("Cache hit (semantic match, similarity=%f) for query: '%s'", sim, cleaned_query)
                    return entry["dense"], entry["sparse"]
        
        # 4. If no semantic match, compute the sparse vector via FastEmbed, save the new query payload to the cache, and return
        indices, values = self._sparse_query(cleaned_query)
        sparse_dict = dict(zip(indices, values))
        
        new_entry = {
            "query": cleaned_query,
            "dense": dense_list,
            "sparse": sparse_dict
        }
        with _EMBEDDING_CACHE_LOCK:
            _EMBEDDING_CACHE.append(new_entry)
            
        return dense_list, sparse_dict

    # ── Async query embedding ──────────────────────────────────────────
    async def async_embed_query(self, query: str) -> tuple[list[float], Dict[int, float]]:
        cleaned_query = query.strip()
        
        # 1. Exact string match check against _EMBEDDING_CACHE
        with _EMBEDDING_CACHE_LOCK:
            for entry in _EMBEDDING_CACHE:
                if entry["query"] == cleaned_query:
                    logger.info("Cache hit (exact match) for query: '%s'", cleaned_query)
                    return entry["dense"], entry["sparse"]
                    
        # 2. Fetch the dense embedding from Ollama asynchronously
        prefixed = QUERY_PREFIX + cleaned_query
        prefixed = _truncate_by_tokens(prefixed, MAX_TOKENS)

        payload = {"model": self.model_name, "prompt": prefixed}
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(self.ollama_url, json=payload)
            r.raise_for_status()
        vec = np.array(r.json()["embedding"], dtype=np.float32)
        dense = _normalize(vec)
        dense_list = dense.tolist()
        
        # 3. Compare the dot product of this dense embedding with the dense embeddings of all items in _EMBEDDING_CACHE
        with _EMBEDDING_CACHE_LOCK:
            for entry in _EMBEDDING_CACHE:
                sim = np.dot(dense, np.array(entry["dense"], dtype=np.float32))
                if sim >= config.GRAPH_EMBEDDING_SIMILARITY_THRESHOLD:
                    logger.info("Cache hit (semantic match, similarity=%f) for query: '%s'", sim, cleaned_query)
                    return entry["dense"], entry["sparse"]
        
        # 4. Compute sparse vector using asyncio.to_thread to prevent blocking the event loop
        indices, values = await asyncio.to_thread(self._sparse_query, cleaned_query)
        sparse_dict = dict(zip(indices, values))
        
        new_entry = {
            "query": cleaned_query,
            "dense": dense_list,
            "sparse": sparse_dict
        }
        with _EMBEDDING_CACHE_LOCK:
            _EMBEDDING_CACHE.append(new_entry)
            
        return dense_list, sparse_dict

    # ── Async Ingest: embed + store in Qdrant ──────────────────────────
    async def async_embed_and_store(self, chunks: list[ChunkResult]) -> None:
        if not chunks:
            logger.warning("No chunks to embed. Skipping.")
            return

        source_file = chunks[0].source_file
        total       = len(chunks)
        logger.info("Async Embedding | file=%s | chunks=%d", source_file, total)

        # Step 1 — sort by token_count for batch efficiency
        indexed  = list(enumerate(chunks))
        sorted_  = sorted(indexed, key=lambda x: x[1].token_count)
        orig_idx = [i for i, _ in sorted_]

        # Step 2 — build enriched texts + recalculate token counts
        texts                 = [self._build_embed_input(c) for _, c in sorted_]
        enriched_token_counts = [_count_tokens(t) for t in texts]

        # Step 3 — sparse vectors via FastEmbed BM25 in a background thread to avoid CPU block
        sparse_results = await asyncio.to_thread(lambda: [self._sparse_embed(t) for t in texts])

        # Step 4 — dense vectors via Qwen3 async
        dense_vecs = await self._encode_passages_async(texts)

        # Step 5 — reorder back to original index
        reordered = [None] * total
        for j, orig_i in enumerate(orig_idx):
            indices, values = sparse_results[j]
            chunk           = chunks[orig_i]
            reordered[orig_i] = EmbeddedChunk.from_chunk(
                chunk,
                dense_vector   = dense_vecs[j],
                sparse_indices = indices,
                sparse_values  = values,
                token_count    = enriched_token_counts[j],
            )

        # Step 6 — delete asynchronously
        await self.db.async_delete_document(source_file)

        # Step 7 — upsert in batches asynchronously
        points_stored = 0
        for start in range(0, total, self.batch_size):
            batch  = reordered[start: start + self.batch_size]
            points = [self._build_point(ec) for ec in batch if ec is not None]
            await asyncio.to_thread(
                self.db.client.upsert,
                collection_name=config.QDRANT_COLLECTION_NAME,
                points=points,
            )
            points_stored += len(points)
            logger.info("  Async Upserted %d/%d", min(start + self.batch_size, total), total)

        logger.info(
            "✓ Async Ingestion complete | file=%s | stored=%d | skipped=%d",
            source_file, points_stored, total - points_stored,
        )


    # ── Async dense encoding ───────────────────────────────────────────
    async def _encode_passages_async(self, texts: list[str]) -> list[np.ndarray]:
        all_vecs = []
        
        # Determine whether to use the newer /api/embed (which supports batch input)
        # or fallback to /api/embeddings (which is what OLLAMA_EMBED_URL currently points to)
        batch_url = self.ollama_url.replace("/api/embeddings", "/api/embed")
        is_legacy = "embeddings" in self.ollama_url and batch_url == self.ollama_url
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start: start + self.batch_size]
                batch_truncated = [_truncate_by_tokens(t, MAX_TOKENS) for t in batch]
                
                try:
                    payload = {"model": self.model_name, "input": batch_truncated}
                    r = await client.post(batch_url, json=payload)
                    r.raise_for_status()
                    
                    # /api/embed returns {"embeddings": [[...], [...]]}
                    embeddings = r.json().get("embeddings", [])
                    for vec_list in embeddings:
                        vec = np.array(vec_list, dtype=np.float32)
                        all_vecs.append(_normalize(vec))
                        
                except Exception as e:
                    logger.warning("Batch embed failed, falling back to sequential: %s", e)
                    # Fallback to sequential /api/embeddings
                    for text in batch_truncated:
                        try:
                            payload = {"model": self.model_name, "prompt": text}
                            r = await client.post(self.ollama_url, json=payload)
                            r.raise_for_status()
                            vec = np.array(r.json()["embedding"], dtype=np.float32)
                            all_vecs.append(_normalize(vec))
                        except Exception as fallback_err:
                            logger.error("Fallback embed failed: %s", fallback_err)
                            all_vecs.append(None)
                            
        return all_vecs

    # ── Build Qdrant point ─────────────────────────────────────────────
    def _build_point(self, ec: EmbeddedChunk) -> Optional[PointStruct]:
        # CHANGE: skip None vectors instead of storing zero vectors
        if ec is None or ec.dense_vector is None:
            logger.warning(
                "Skipping chunk | source=%s | index=%d | reason=no dense vector",
                ec.source_file if ec else "unknown", ec.chunk_index if ec else -1,
            )
            return None

        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{ec.source_file}::{ec.chunk_index}",
        ))

        return PointStruct(
            id     = point_id,
            vector = {
                "dense": ec.dense_vector.tolist(),
                "sparse": SparseVector(
                    indices = ec.sparse_indices,
                    values  = ec.sparse_values,
                ),
            },
            payload = {
                "chunk_index":    ec.chunk_index,
                "text":           ec.text,
                "page_no":        ec.page_no,
                "page_label":     ec.page_label,
                "page_range":     list(ec.page_range),
                "section":        ec.section,
                "is_table":       ec.is_table,
                "token_count":    ec.token_count,
                "chunk_year":     ec.chunk_year,
                "doc_id":         ec.doc_id,
                "chunk_id":       ec.chunk_id,
                "source_file":    ec.source_file,
                "filename_tokens": ec.filename_tokens,
                "doc_year":       ec.doc_year,
                "has_tables":     ec.has_tables,
                "summary":        ec.summary,
                "keywords":       ec.keywords,
                "file_hash":      ec.file_hash,
            },
        )


# ── Helpers ────────────────────────────────────────────────────────────
def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def get_embedder(db: QdrantManager) -> OllamaEmbedder:
    return OllamaEmbedder(db=db)