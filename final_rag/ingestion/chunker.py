"""
final_rag/ingestion/chunker.py

Semantic Chunking: Converts BlockRecord list -> ChunkResult list.
Splitting, buffering, sentence formatting for tables, and smart deduplication.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

try:
    from final_rag.config import (
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MIN_CHARS,
        MIN_WORDS,
        TABLE_MAX_TOKENS,
    )
except ImportError:
    from ..config import (
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MIN_CHARS,
        MIN_WORDS,
        TABLE_MAX_TOKENS,
    )

from .parser import BlockRecord, ParseResult

logger = logging.getLogger("ingestion.chunker")

# Tuning constants
DEDUP_THRESHOLD                 = 0.85
DEDUP_SHINGLE_SIZE              = 6
DEDUP_SHINGLE_SIZE_LONG         = 8
DEDUP_LONG_TEXT_THRESHOLD_CHARS = 1200
HINDI_TOKEN_SCALE_FACTOR        = 0.5
MIN_CHARS_FACTUAL               = 20
MAX_BOND_CHARS                  = 120
NOISE_RATIO_THRESHOLD           = 0.6


@dataclass
class ChunkResult:
    chunk_id:    str
    chunk_index: int
    text:        str
    page_no:     int        = 0
    page_label:  str        = ""
    page_range:  list[int]  = field(default_factory=lambda: [0, 0])
    heading:     str        = ""
    is_table:    bool       = False
    token_count: int        = 0
    doc_id:      str        = ""
    source_file: str        = ""


def _load_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning("tiktoken unavailable, falling back to word count | %s", exc)
        return None


def _count_tokens(text: str, tokenizer) -> int:
    return len(text.split()) if tokenizer is None else len(tokenizer.encode(text))


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return hashlib.md5(f"{doc_id}{chunk_index}".encode()).hexdigest()[:12]


def _sanitize_whitespace(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        lines.append(line.strip() if "|" in line else re.sub(r" {3,}", " ", line).strip())
    return "\n".join(lines).strip()


_NOISE_RUN_RE = re.compile(r'[.\-_]{5,}')


def _is_embeddable(text: str, is_table: bool = False) -> bool:
    if is_table:
        return True
    stripped = text.strip()
    non_heading = [l.strip() for l in stripped.split("\n") if l.strip() and not l.strip().startswith("#")]
    if not non_heading:
        return False

    non_ws_len = len(re.sub(r'\s', '', stripped))
    if non_ws_len > 0:
        noise_len = sum(len(m) for m in _NOISE_RUN_RE.findall(stripped))
        if noise_len / non_ws_len >= NOISE_RATIO_THRESHOLD:
            return False

    factual = re.compile(r'\d+|rs\.?|inr|₹|crore|lakh|\b(19|20)\d{2}\b', re.IGNORECASE)
    if factual.search(stripped):
        return len(stripped) >= MIN_CHARS_FACTUAL
    return len(stripped.split()) >= MIN_WORDS and len(stripped) >= MIN_CHARS


def _table_to_sentences(table_markdown: str, tokenizer=None, max_tokens: int = TABLE_MAX_TOKENS) -> list[str]:
    lines = [l.strip() for l in table_markdown.strip().split("\n") if l.strip() and not re.match(r"^\|[-| :]+\|$", l.strip())]
    if len(lines) < 2:
        return [table_markdown]
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    sentences, dropped = [], 0
    for row in lines[1:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) != len(headers):
            dropped += 1
            continue
        pairs = ", ".join(f"{h}={c}" for h, c in zip(headers, cells) if c)
        if pairs:
            sentences.append(pairs)
    if dropped:
        logger.warning("_table_to_sentences: dropped %d row(s) — column mismatch", dropped)
    if not sentences:
        return [table_markdown]

    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sent_tokens = _count_tokens(sentence, tokenizer)
        if current and current_tokens + sent_tokens > max_tokens:
            groups.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sent_tokens
    if current:
        groups.append("\n".join(current))

    return groups


def _extract_entity_fingerprint(text: str) -> str:
    numbers  = re.findall(r'\d+', text)
    years    = re.findall(r'\b(19|20)\d{2}\b', text)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    sig = "|".join(sorted(set(numbers + years + entities[:5])))
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def _dedup_exact(chunks: list[ChunkResult]) -> list[ChunkResult]:
    seen, out = set(), []
    for c in chunks:
        h = hashlib.md5(re.sub(r"\s+", " ", c.text.strip().lower()).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out


def _get_shingles(text: str, size: int) -> set[str]:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return {norm[i:i+size] for i in range(len(norm) - size + 1)}


def _dedup_near_smart(chunks: list[ChunkResult], threshold: float = DEDUP_THRESHOLD) -> list[ChunkResult]:
    kept, kept_shingles, kept_fp = [], [], []
    for c in chunks:
        if c.is_table:
            kept.append(c)
            continue
        size = DEDUP_SHINGLE_SIZE_LONG if len(c.text) > DEDUP_LONG_TEXT_THRESHOLD_CHARS else DEDUP_SHINGLE_SIZE
        shingles, fp = _get_shingles(c.text, size), _extract_entity_fingerprint(c.text)
        dup = False
        for i, existing in enumerate(kept_shingles):
            if fp != kept_fp[i]:
                continue
            union = len(shingles | existing)
            if union and len(shingles & existing) / union >= threshold:
                dup = True
                break
        if not dup:
            kept.append(c)
            kept_shingles.append(shingles)
            kept_fp.append(fp)
    return kept


def _dominant_heading(blocks: list[BlockRecord]) -> str:
    names = [b.section for b in blocks if b.section]
    return Counter(names).most_common(1)[0][0] if names else ""


class DocumentChunker:
    def __init__(
        self,
        chunk_size:    int   = CHUNK_SIZE,
        chunk_overlap: int   = CHUNK_OVERLAP,
        tokenizer            = None,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer     = tokenizer or _load_tokenizer()
        self._splitter     = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=self.tokenizer.encode if self.tokenizer else None,
        )

        self._hindi_eff_size    = int(self.chunk_size * HINDI_TOKEN_SCALE_FACTOR)
        self._hindi_eff_overlap = int(self.chunk_overlap * HINDI_TOKEN_SCALE_FACTOR)
        self._hindi_splitter    = SentenceSplitter(
            chunk_size=self._hindi_eff_size,
            chunk_overlap=self._hindi_eff_overlap,
            tokenizer=self.tokenizer.encode if self.tokenizer else None,
        )

    def _effective_splitter(self, doc_lang: str) -> tuple[SentenceSplitter, int, int]:
        if doc_lang in ("hi", "mixed"):
            return self._hindi_splitter, self._hindi_eff_size, self._hindi_eff_overlap
        return self._splitter, self.chunk_size, self.chunk_overlap

    def chunk(self, parse_result: ParseResult) -> list[ChunkResult]:
        if not parse_result.success or not parse_result.blocks:
            logger.warning("Skipping | %s — no success/blocks", parse_result.file_name)
            return []

        doc_lang = parse_result.meta.doc_lang if parse_result.meta else "en"
        splitter, eff_size, eff_overlap = self._effective_splitter(doc_lang)
        if doc_lang in ("hi", "mixed"):
            logger.info(
                "Language-scaled chunk sizing | %s | lang=%s | size=%d | overlap=%d",
                parse_result.file_name, doc_lang, eff_size, eff_overlap,
            )

        chunks = self._process_blocks(
            blocks      = parse_result.blocks,
            doc_id      = parse_result.doc_id,
            source_file = parse_result.file_name,
            splitter    = splitter,
            chunk_size  = eff_size,
        )
        chunks = self._deduplicate(chunks, parse_result.file_name)
        logger.info("Done | %s | chunks=%d | tables=%d", parse_result.file_name, len(chunks), sum(1 for c in chunks if c.is_table))
        return chunks

    def chunk_batch(self, parse_results: list[ParseResult]) -> dict[str, list[ChunkResult]]:
        results = {pr.file_name: self.chunk(pr) for pr in parse_results}
        logger.info("Batch done | files=%d | total_chunks=%d", len(results), sum(len(v) for v in results.values()))
        return results

    def _deduplicate(self, chunks: list[ChunkResult], file_name: str) -> list[ChunkResult]:
        before = len(chunks)
        chunks = _dedup_near_smart(_dedup_exact(chunks))
        if before - len(chunks):
            logger.info("Dedup removed %d | file=%s", before - len(chunks), file_name)
        return chunks

    def _split_text(self, text: str, splitter: SentenceSplitter) -> list[str]:
        try:
            nodes = splitter.get_nodes_from_documents([Document(text=text)])
        except Exception as exc:
            logger.warning("SentenceSplitter failed: %s", exc)
            return [text]
        return [(n.get_content() if hasattr(n, "get_content") else n.text).strip() for n in nodes]

    def _process_blocks(
        self,
        blocks: list[BlockRecord],
        doc_id: str,
        source_file: str,
        splitter: SentenceSplitter,
        chunk_size: int,
    ) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        chunk_index = 0
        text_buffer: list[BlockRecord] = []
        buffer_tokens = 0

        def flush_buffer():
            nonlocal chunk_index
            if not text_buffer:
                return
            merged = "\n".join(_sanitize_whitespace(b.content) for b in text_buffer)
            if not _is_embeddable(merged):
                text_buffer.clear()
                return
            heading    = _dominant_heading(text_buffer)
            start_page = text_buffer[0].page_no
            end_page   = text_buffer[-1].page_no
            page_label = text_buffer[0].page_label

            for split_text in self._split_text(merged, splitter):
                if not _is_embeddable(split_text):
                    continue
                display = f"[{heading}]\n{split_text}" if heading else split_text
                chunks.append(ChunkResult(
                    chunk_id=_make_chunk_id(doc_id, chunk_index), chunk_index=chunk_index,
                    text=display, page_no=start_page, page_label=page_label,
                    page_range=[start_page, end_page], heading=heading, is_table=False,
                    token_count=_count_tokens(display, self.tokenizer),
                    doc_id=doc_id, source_file=source_file,
                ))
                chunk_index += 1
            text_buffer.clear()

        for i, block in enumerate(blocks):
            content = _sanitize_whitespace(block.content)

            if block.block_type in ("heading", "title"):
                flush_buffer()
                buffer_tokens = 0
                text_buffer.append(block)
                continue

            if block.is_table:
                preceding = blocks[i-1] if i > 0 else None
                bonded = ""
                if (
                    preceding
                    and preceding.block_type == "text"
                    and preceding.content.strip()
                    and len(preceding.content.strip()) <= MAX_BOND_CHARS
                    and text_buffer
                    and text_buffer[-1] is preceding
                ):
                    bonded = _sanitize_whitespace(preceding.content)
                    text_buffer.pop()

                flush_buffer()
                buffer_tokens = 0

                if _count_tokens(content, self.tokenizer) > TABLE_MAX_TOKENS:
                    table_parts = _table_to_sentences(content, self.tokenizer, TABLE_MAX_TOKENS)
                else:
                    table_parts = [content]

                for part_idx, part in enumerate(table_parts):
                    if bonded and part_idx == 0:
                        full_text = f"{bonded}\n{part}".strip()
                    else:
                        full_text = part
                    if not _is_embeddable(full_text, is_table=True):
                        continue
                    display = f"[{block.section}]\n{full_text}" if block.section else full_text
                    chunks.append(ChunkResult(
                        chunk_id=_make_chunk_id(doc_id, chunk_index), chunk_index=chunk_index,
                        text=display, page_no=block.page_no, page_label=block.page_label,
                        page_range=[block.page_no, block.page_no], heading=block.section, is_table=True,
                        token_count=_count_tokens(display, self.tokenizer),
                        doc_id=doc_id, source_file=source_file,
                    ))
                    chunk_index += 1
                continue

            if not _is_embeddable(content):
                continue

            if text_buffer and block.section != text_buffer[-1].section:
                flush_buffer()
                buffer_tokens = 0

            text_buffer.append(block)
            buffer_tokens += _count_tokens(content, self.tokenizer)
            if buffer_tokens >= chunk_size:
                flush_buffer()
                buffer_tokens = 0

        flush_buffer()
        return chunks