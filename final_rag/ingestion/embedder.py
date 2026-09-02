"""
final_rag/ingestion/embedder.py

Qwen3-Embedding via Ollama (dense vectors with prefix instructions) + FastEmbed BM25 (sparse vectors).
Hybrid ingestion and query embedding for Qdrant storage.
"""

from __future__ import annotations
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import httpx
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector

try:
    import final_rag.config as config
except ImportError:
    from .. import config

from .chunker import ChunkResult
from .chunker_metadata import ChunkMeta, build_chunk_meta_batch
from ..qdrant_storage.store import QdrantManager

logger = logging.getLogger("ingestion.embedder")

MODEL_NAME         = config.EMBED_MODEL
OLLAMA_URL         = config.OLLAMA_EMBED_URL
DEFAULT_BATCH_SIZE = config.EMBED_BATCH_SIZE
MAX_TOKENS         = config.EMBED_MAX_TOKENS
SPARSE_MODEL_NAME  = config.SPARSE_MODEL_NAME

_prefixes      = getattr(config, "EMBED_PREFIXES", {}).get(MODEL_NAME, {"query": "query: ", "passage": "passage: "})
QUERY_PREFIX   = _prefixes.get("query", "query: ")
PASSAGE_PREFIX = _prefixes.get("passage", "passage: ")


def _load_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

_TOKENIZER = _load_tokenizer()


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if not _TOKENIZER:
        words = text.split()
        return " ".join(words[:max_tokens]) if len(words) > max_tokens else text
    tokens = _TOKENIZER.encode(text)
    return _TOKENIZER.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text


def _truncate_and_normalize(vec: np.ndarray, target_dim: int = None) -> np.ndarray:
    target_dim = target_dim or getattr(config, "EMBED_DIMENSIONS", 2560)
    truncated  = vec[:target_dim]
    norm       = np.linalg.norm(truncated)
    return truncated.astype(np.float32) if norm == 0 else (truncated / norm).astype(np.float32)


@dataclass
class EmbeddedChunk:
    chunk_id:        str
    chunk_index:     int
    text:            str
    page_no:         int
    page_label:      str
    page_range:      list[int]
    heading:         str
    is_table:        bool
    token_count:     int
    doc_id:          str
    source_file:     str
    dense_vector:    Optional[np.ndarray]
    sparse_indices:  list[int]
    sparse_values:   list[float]
    filename_tokens: list[str] = field(default_factory=list)
    doc_year:        str       = ""
    doc_lang:        str       = "en"
    chunk_years:     list[str] = field(default_factory=list)
    entities:        list[str] = field(default_factory=list)


class OllamaEmbedder:
    def __init__(
        self,
        db:                 QdrantManager,
        model_name:         str = MODEL_NAME,
        ollama_url:         str = OLLAMA_URL,
        batch_size:         int = DEFAULT_BATCH_SIZE,
        sparse_model_name:  str = SPARSE_MODEL_NAME,
    ):
        self.db                 = db
        self.model_name         = model_name
        self.ollama_url         = ollama_url
        self.batch_size         = batch_size
        self.sparse_model_name  = sparse_model_name
        self._sparse_model      = None
        self._query_client      = httpx.Client(timeout=getattr(config, "EMBED_HTTP_TIMEOUT_SEC", 80.0))

    @property
    def sparse_model(self):
        if self._sparse_model is None:
            logger.info("Initializing Sparse Model: %s", self.sparse_model_name)
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    @sparse_model.setter
    def sparse_model(self, value):
        self._sparse_model = value

    def _sparse_query(self, query: str) -> tuple[list[int], list[float]]:
        vec = list(self.sparse_model.query_embed(query))[0]
        return vec.indices.tolist(), vec.values.tolist()

    def embed_and_store(
        self,
        chunks:  list[ChunkResult],
        metas:   Optional[list[ChunkMeta]] = None,
        domain:  str = "",
    ) -> None:
        if not chunks:
            logger.warning("embed_and_store: empty chunk list received")
            return

        source_file = chunks[0].source_file
        total       = len(chunks)
        logger.info("Embedding | file=%s | chunks=%d", source_file, total)

        texts = [
            _truncate_by_tokens(PASSAGE_PREFIX + c.text.strip(), MAX_TOKENS)
            for c in chunks
        ]

        # 1. Sparse BM25
        sparse_gen = self.sparse_model.passage_embed(texts)
        sparse_vecs = [
            (sv.indices.tolist(), sv.values.tolist())
            for sv in sparse_gen
        ]

        # 2. Dense Qwen3
        dense_vecs = asyncio.run(self._encode_passages_async(texts))

        embedded: list[EmbeddedChunk] = []
        for i, chunk in enumerate(chunks):
            sparse_idx, sparse_val = sparse_vecs[i]
            dense_vec = dense_vecs[i]
            meta = metas[i] if metas and i < len(metas) else None

            embedded.append(EmbeddedChunk(
                chunk_id        = chunk.chunk_id,
                chunk_index     = chunk.chunk_index,
                text            = chunk.text,
                page_no         = chunk.page_no,
                page_label      = chunk.page_label,
                page_range      = list(chunk.page_range),
                heading         = chunk.heading,
                is_table        = chunk.is_table,
                token_count     = chunk.token_count,
                doc_id          = chunk.doc_id,
                source_file     = chunk.source_file,
                dense_vector    = dense_vec,
                sparse_indices  = sparse_idx,
                sparse_values   = sparse_val,
                filename_tokens = meta.filename_tokens if meta else [],
                doc_year        = meta.doc_year if meta else "",
                doc_lang        = meta.doc_lang if meta else "en",
                chunk_years     = meta.chunk_years if meta else [],
                entities        = meta.entities if meta else [],
            ))

        successful_count = sum(1 for ec in embedded if ec.dense_vector is not None)
        if successful_count == 0:
            raise RuntimeError(f"All {total} chunks failed to embed for file={source_file}")

        # Delete existing points for file
        self.db.delete_document(source_file, domain=domain)

        points_stored = 0
        for start in range(0, total, self.batch_size):
            batch = embedded[start: start + self.batch_size]
            points = [p for p in (self._build_point(ec) for ec in batch) if p is not None]
            if points:
                self.db.upsert(points, domain=domain)
            points_stored += len(points)

        logger.info("Ingestion complete | file=%s | stored=%d | skipped=%d", source_file, points_stored, total - points_stored)

    def embed_query(self, query: str) -> tuple[list[float], Dict[int, float]]:
        prefixed = _truncate_by_tokens(QUERY_PREFIX + query.strip(), MAX_TOKENS)
        payload = {"model": self.model_name, "prompt": prefixed}
        r = self._query_client.post(self.ollama_url, json=payload)
        r.raise_for_status()
        vec = np.array(r.json()["embedding"], dtype=np.float32)
        dense = _truncate_and_normalize(vec)

        indices, values = self._sparse_query(query)
        return dense.tolist(), dict(zip(indices, values))

    async def _encode_passages_async(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        all_vecs = []
        timeout = getattr(config, "EMBED_HTTP_TIMEOUT_SEC", 80.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch   = texts[start: start + self.batch_size]
                tasks   = [self._single_embed_async(t, client) for t in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning("Embed failed chunk %d | retrying | %s", start + i, result)
                        try:
                            retried = await self._single_embed_async(texts[start + i], client)
                            all_vecs.append(retried)
                        except Exception as retry_err:
                            logger.error("Embed retry failed chunk %d: %s", start + i, retry_err)
                            all_vecs.append(None)
                    else:
                        all_vecs.append(result)
        return all_vecs

    async def _single_embed_async(self, text: str, client: httpx.AsyncClient) -> np.ndarray:
        text    = _truncate_by_tokens(text, MAX_TOKENS)
        payload = {"model": self.model_name, "prompt": text}
        r = await client.post(self.ollama_url, json=payload)
        r.raise_for_status()
        return _truncate_and_normalize(np.array(r.json()["embedding"], dtype=np.float32))

    def _build_point(self, ec: EmbeddedChunk) -> Optional[PointStruct]:
        if ec is None or ec.dense_vector is None:
            return None

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ec.doc_id}::{ec.chunk_index}"))

        return PointStruct(
            id     = point_id,
            vector = {
                "dense":  ec.dense_vector.tolist(),
                "sparse": SparseVector(indices=ec.sparse_indices, values=ec.sparse_values),
            },
            payload = {
                "chunk_index":     ec.chunk_index,
                "text":            ec.text,
                "page_no":         ec.page_no,
                "page_label":      ec.page_label,
                "page_range":      list(ec.page_range),
                "heading":         ec.heading,
                "is_table":        ec.is_table,
                "token_count":     ec.token_count,
                "doc_id":          ec.doc_id,
                "chunk_id":        ec.chunk_id,
                "source_file":     ec.source_file,
                "filename_tokens": ec.filename_tokens,
                "doc_year":        ec.doc_year,
                "doc_lang":        ec.doc_lang,
                "chunk_years":     ec.chunk_years,
                "entities":        ec.entities,
            },
        )


def get_embedder(db: QdrantManager) -> OllamaEmbedder:
    return OllamaEmbedder(db=db)