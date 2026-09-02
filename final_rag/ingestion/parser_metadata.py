"""
final_rag/ingestion/parser_metadata.py

File-level metadata extraction — pure regex, no LLM, no network calls.
Extracts filename tokens, document year, and detected language (Hindi / English / Mixed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from final_rag.config import (
        DOC_LANG_HINDI_CHAR_RATIO_THRESHOLD,
        DOC_LANG_MIXED_THRESHOLD,
        DOC_LANG_SAMPLE_CHARS,
    )
except ImportError:
    from ..config import (
        DOC_LANG_HINDI_CHAR_RATIO_THRESHOLD,
        DOC_LANG_MIXED_THRESHOLD,
        DOC_LANG_SAMPLE_CHARS,
    )


@dataclass
class DocMetadataResult:
    filename_tokens: list[str] = field(default_factory=list)
    doc_year: str = ""
    doc_lang: str = "en"  # "hi" | "en" | "mixed"


YEAR_PATTERN     = re.compile(r'\b(19\d{2}|20\d{2})\b')
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')


def _tokenize_filename(file_name: str) -> list[str]:
    stem   = Path(file_name).stem.lower()
    tokens = re.split(r'[_\-\s]+', stem)
    return [t for t in tokens if t]


def _extract_year(file_name: str) -> str:
    name = Path(file_name).stem
    year = YEAR_PATTERN.search(name)
    return year.group(1) if year else ""


def _sample_document(markdown: str, window: int) -> str:
    n = len(markdown)
    if n <= window * 3:
        return markdown

    mid_start = (n // 2) - (window // 2)
    return (
        markdown[:window]
        + markdown[mid_start:mid_start + window]
        + markdown[-window:]
    )


def _detect_doc_lang(markdown: str) -> str:
    if not markdown:
        return "en"

    sample      = _sample_document(markdown, DOC_LANG_SAMPLE_CHARS)
    total_chars = len(re.sub(r'\s', '', sample))
    if total_chars == 0:
        return "en"

    hindi_chars = len(DEVANAGARI_RANGE.findall(sample))
    ratio       = hindi_chars / total_chars

    if ratio >= DOC_LANG_HINDI_CHAR_RATIO_THRESHOLD:
        return "hi" if ratio > DOC_LANG_MIXED_THRESHOLD else "mixed"
    return "en"


def extract_doc_metadata(file_name: str, markdown: str = "") -> DocMetadataResult:
    return DocMetadataResult(
        filename_tokens = _tokenize_filename(file_name),
        doc_year        = _extract_year(file_name),
        doc_lang        = _detect_doc_lang(markdown),
    )
