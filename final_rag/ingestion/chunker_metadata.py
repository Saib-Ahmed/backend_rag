"""
final_rag/ingestion/chunker_metadata.py

Chunk-level metadata enrichment: inherits document metadata (language, filename tokens, year)
plus per-chunk regex year extraction and GLiNER entity extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chunker import ChunkResult
    from .parser import DocumentMeta

from .gliner_engine import extract_gliner_entities

_YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')


@dataclass
class ChunkMeta:
    chunk_id:        str
    doc_id:          str
    file_name:       str
    filename_tokens: list[str] = field(default_factory=list)
    doc_year:        str       = ""
    doc_lang:        str       = "en"
    chunk_years:     list[str] = field(default_factory=list)
    entities:        list[str] = field(default_factory=list)


def _extract_chunk_years(text: str) -> list[str]:
    return sorted(set(_YEAR_RE.findall(text)))


def build_chunk_meta(chunk: Any, doc_meta: Any) -> ChunkMeta:
    return ChunkMeta(
        chunk_id        = getattr(chunk, "chunk_id", ""),
        doc_id          = getattr(doc_meta, "doc_id", ""),
        file_name       = getattr(doc_meta, "file_name", ""),
        filename_tokens = getattr(doc_meta, "filename_tokens", []),
        doc_year        = getattr(doc_meta, "doc_year", ""),
        doc_lang        = getattr(doc_meta, "doc_lang", "en"),
        chunk_years     = _extract_chunk_years(getattr(chunk, "text", "")),
        entities        = extract_gliner_entities(getattr(chunk, "text", "")),
    )


def build_chunk_meta_batch(chunks: list[Any], doc_meta: Any) -> list[ChunkMeta]:
    return [build_chunk_meta(c, doc_meta) for c in chunks]
