"""
ingestion/chunker.py

Chunker — consumes structured BlockRecord list from parser.
Table detection comes from parser metadata (no regex detection).
Stack: SentenceSplitter + tiktoken
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import config
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHARS,
    MIN_WORDS,
    TABLE_MAX_TOKENS,
    DEDUP_THRESHOLD,
    DEDUP_SHINGLE_SIZE,
    MIN_CHUNK_TOKENS,
    MIN_EMBEDDABLE_TOKENS,
)
from new_ingestion.parser import BlockRecord, PageRecord, ParseResult

logger = logging.getLogger("ingestion.chunker")
_re_table_sep = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    chunk_id:        str
    chunk_index:     int
    text:            str
    full_table_context: str = ""
    page_no:         int       = 0
    page_label:      str       = ""
    page_range: list[int] = field(default_factory=lambda: [0, 0])
    section:         str       = ""
    heading_path:    str       = ""
    is_table:        bool      = False
    token_count:     int       = 0
    chunk_year:      list[str] = field(default_factory=list)
    keywords:        list[str] = field(default_factory=list)
    doc_id:          str       = ""
    source_file:     str       = ""
    filename_tokens: list[str] = field(default_factory=list)
    doc_year:        str       = ""
    has_tables:      bool      = False
    summary:         str       = ""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _load_tokenizer():
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        logger.info("tiktoken loaded — cl100k_base")
        return encoder
    except Exception as exc:
        logger.warning("tiktoken unavailable, falling back to word count | %s", exc)
        return None


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# chunk_id generator
# ---------------------------------------------------------------------------

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _sanitize_whitespace(text: str) -> str:
    # Strip digital signature artifacts
    text = re.sub(r"(?i)Signature Not Verified.*?(?:Reason:|Date:).*?\d{4}.*?(?=\n|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)Digitally signed by.*?(?:Date:|Reason:).*?\d{4}.*?(?=\n|$)", "", text, flags=re.DOTALL)
    
    cleaned_lines = []
    for line in text.split("\n"):
        if "|" in line:
            cleaned_lines.append(line.strip())
        else:
            cleaned_lines.append(re.sub(r" {3,}", " ", line).strip())
    return "\n".join(cleaned_lines).strip()


def _extract_years(text: str) -> list[str]:
    return list(set(re.findall(r'\b(19\d{2}|20\d{2})\b', text)))


def _is_embeddable(text: str, is_table: bool = False) -> bool:
    if is_table:
        return True

    stripped = text.strip()
    non_heading_lines = [
        line.strip()
        for line in stripped.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    if not non_heading_lines:
        return False

    FACTUAL_PATTERN = config.FACTUAL_PATTERN
    has_factual_content = bool(FACTUAL_PATTERN.search(stripped))

    if has_factual_content:
        return len(stripped) >= 20

    return len(stripped.split()) >= MIN_WORDS and len(stripped) >= MIN_CHARS


# ---------------------------------------------------------------------------
# Per-chunk keyword extraction (production-level BM25 support)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was',
    'were', 'been', 'have', 'has', 'had', 'not', 'but', 'which', 'their',
    'they', 'will', 'would', 'could', 'should', 'shall', 'may', 'can',
    'also', 'such', 'any', 'each', 'other', 'into', 'than', 'its',
    'being', 'does', 'did', 'these', 'those', 'above', 'below', 'under',
    'over', 'between', 'about', 'upon', 'through', 'after', 'before',
})

def _extract_chunk_keywords(text: str) -> list[str]:
    """Extract significant lexical terms per chunk for BM25-style matching."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return list(set(w for w in words if w not in _STOPWORDS))[:25]


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _process_table_to_json(table_markdown: str) -> Tuple[List[dict], List[str]]:
    lines = [line.strip() for line in table_markdown.splitlines() if line.strip()]
    separator_idx = -1
    for i, line in enumerate(lines):
        if _re_table_sep.match(line):
            separator_idx = i
            break
    if separator_idx <= 0 or separator_idx >= len(lines) - 1:
        return [], []
    header_line = lines[separator_idx - 1]
    raw_headers = [h.strip() for h in header_line.strip("|").split("|")]
    # Preserve empty headers as named placeholders to prevent column shifting
    headers = []
    for i, h in enumerate(raw_headers):
        if h:
            headers.append(h)
        else:
            headers.append(f"UNNAMED_COLUMN_{i}")
    if not any(h for h in raw_headers):  # All headers empty → truly malformed
        return [], []
    json_rows = []
    for row_line in lines[separator_idx + 1:]:
        if _re_table_sep.match(row_line):
            continue
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        if len(cells) != len(headers):
            logger.debug(
                "Table column mismatch: headers=%d cells=%d",
                len(headers), len(cells),
            )
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[:len(headers)]  # Truncate excess cells
        row_dict = {headers[idx]: cells[idx] for idx in range(len(headers))}
        if any(v.strip() for v in row_dict.values()):
            json_rows.append(row_dict)
    return json_rows, headers


# ---------------------------------------------------------------------------
# Entity fingerprint for smart dedup
# ---------------------------------------------------------------------------

def _extract_entity_fingerprint(text: str) -> str:
    numbers  = re.findall(r'\d+', text)
    years    = re.findall(r'\b(19|20)\d{2}\b', text)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    signature = "|".join(sorted(set(numbers + years + entities[:5])))
    return hashlib.md5(signature.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_exact(chunks: list[ChunkResult]) -> list[ChunkResult]:
    seen_hashes: set[str] = set()
    unique_chunks: list[ChunkResult] = []

    for chunk in chunks:
        normalised = re.sub(r"\s+", " ", chunk.text.strip().lower())
        md5_hash   = hashlib.md5(normalised.encode()).hexdigest()

        if md5_hash not in seen_hashes:
            seen_hashes.add(md5_hash)
            unique_chunks.append(chunk)
        else:
            logger.debug(
                "Exact dedup dropped | file=%s | index=%d",
                chunk.source_file, chunk.chunk_index,
            )

    return unique_chunks


def _get_shingles(text: str, size: int = DEDUP_SHINGLE_SIZE) -> set[str]:
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return {normalised[i:i + size] for i in range(len(normalised) - size + 1)}


def _dedup_near_smart(
    chunks: list[ChunkResult],
    threshold: float = DEDUP_THRESHOLD,
) -> list[ChunkResult]:
    kept_chunks: list[ChunkResult]    = []
    kept_shingle_sets: list[set[str]] = []
    kept_fingerprints: list[str]      = []

    for chunk in chunks:
        if chunk.is_table:
            kept_chunks.append(chunk)
            continue

        shingle_size          = config.LARGE_CHUNK_SHINGLE_SIZE if len(chunk.text) > config.LARGE_CHUNK_THRESHOLD else config.DEDUP_SHINGLE_SIZE
        candidate_shingles    = _get_shingles(chunk.text, shingle_size)
        candidate_fingerprint = _extract_entity_fingerprint(chunk.text)

        is_near_duplicate = False

        for i, existing_shingles in enumerate(kept_shingle_sets):
            if candidate_fingerprint != kept_fingerprints[i]:
                continue

            union_size = len(candidate_shingles | existing_shingles)
            if union_size == 0:
                continue
            jaccard = len(candidate_shingles & existing_shingles) / union_size
            if jaccard >= threshold:
                is_near_duplicate = True
                logger.debug(
                    "Near dedup dropped | file=%s | index=%d | jaccard=%.2f",
                    chunk.source_file, chunk.chunk_index, jaccard,
                )
                break

        if not is_near_duplicate:
            kept_chunks.append(chunk)
            kept_shingle_sets.append(candidate_shingles)
            kept_fingerprints.append(candidate_fingerprint)

    return kept_chunks


# ---------------------------------------------------------------------------
# Small chunk merging (production-level: prevents noisy tiny embeddings)
# ---------------------------------------------------------------------------

def _merge_small_chunks(
    chunks: list[ChunkResult],
    tokenizer,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[ChunkResult]:
    """Merge chunks below min_tokens with their next neighbor."""
    if len(chunks) <= 1:
        return chunks

    merged: list[ChunkResult] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        # If current chunk is too small, not a table, and next chunk is compatible
        if (
            current.token_count < min_tokens
            and not current.is_table
            and i + 1 < len(chunks)
            and not chunks[i + 1].is_table
            and chunks[i + 1].source_file == current.source_file
        ):
            nxt = chunks[i + 1]
            merged_text = current.text + "\n" + nxt.text
            merged_keywords = list(set(current.keywords + nxt.keywords))[:25]
            merged.append(ChunkResult(
                chunk_id        = current.chunk_id,
                chunk_index     = current.chunk_index,
                text            = merged_text,
                full_table_context = current.full_table_context,
                page_no         = current.page_no,
                page_label      = current.page_label,
                page_range      = [current.page_range[0], nxt.page_range[1]],
                section         = current.section or nxt.section,
                heading_path    = getattr(current, "heading_path", "") or getattr(nxt, "heading_path", ""),
                is_table        = False,
                token_count     = _count_tokens(merged_text, tokenizer),
                chunk_year      = list(set(current.chunk_year + nxt.chunk_year)),
                keywords        = merged_keywords,
                doc_id          = current.doc_id,
                source_file     = current.source_file,
                filename_tokens = current.filename_tokens,
                doc_year        = current.doc_year,
                has_tables      = current.has_tables,
                summary         = current.summary,
            ))
            i += 2  # Skip the merged neighbor
            logger.debug(
                "Merged small chunk | file=%s | index=%d + %d | tokens=%d",
                current.source_file, current.chunk_index, nxt.chunk_index,
                merged[-1].token_count,
            )
        else:
            merged.append(current)
            i += 1

    return merged


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _determine_heading_level(content: str) -> int:
    content = content.strip()
    m = re.match(r'^(\d+(?:\.\d+)+)\.?\s', content)
    if m:
        parts = m.group(1).split('.')
        return len(parts)
    
    m = re.match(r'^\d+\.?\s', content)
    if m:
        return 1
        
    lower_content = content.lower()
    if lower_content.startswith("chapter") or lower_content.startswith("part"):
        return 1
    if lower_content.startswith("section"):
        m = re.match(r'^section\s+(\d+(?:\.\d+)*)', lower_content)
        if m:
            parts = m.group(1).split('.')
            return len(parts) + 1
        return 2
    if lower_content.startswith("sub-section") or lower_content.startswith("subsection"):
        return 3
    if lower_content.startswith("annex") or lower_content.startswith("appendix"):
        return 1

    if re.match(r'^(?:IX|IV|V?I{0,3})\.?\s', content, re.IGNORECASE):
        return 1
    
    if re.match(r'^[A-Z]\.?\s', content):
        return 2
        
    return 2


def semantic_split(text: str, embed_fn, threshold: float = 0.6, max_chunk_size: int = 500, tokenizer = None) -> list[str]:
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    if len(sentences) <= 1:
        return [text]
        
    try:
        embeddings = embed_fn(sentences)
    except Exception as e:
        logger.warning("Failed to get sentence embeddings for semantic chunking: %s. Falling back to default split.", e)
        return []

    if not embeddings or len(embeddings) != len(sentences):
        return []

    import numpy as np
    emb_arr = [np.array(e) for e in embeddings]
    
    norms = [np.linalg.norm(e) for e in emb_arr]
    normalized_embs = []
    for e, norm in zip(emb_arr, norms):
        if norm == 0:
            normalized_embs.append(e)
        else:
            normalized_embs.append(e / norm)
            
    similarities = []
    for i in range(len(normalized_embs) - 1):
        sim = float(np.dot(normalized_embs[i], normalized_embs[i+1]))
        similarities.append(sim)
        
    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_tokens = len(tokenizer.encode(sentences[0])) if tokenizer else len(sentences[0].split())
    
    for i in range(len(similarities)):
        sim = similarities[i]
        next_sentence = sentences[i+1]
        next_tokens = len(tokenizer.encode(next_sentence)) if tokenizer else len(next_sentence.split())
        
        if sim < threshold or current_tokens + next_tokens > max_chunk_size:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [next_sentence]
            current_tokens = next_tokens
        else:
            current_chunk_sentences.append(next_sentence)
            current_tokens += next_tokens
            
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks


def _dominant_section(blocks: list[BlockRecord]) -> str:
    section_names = [block.section for block in blocks if block.section]
    if not section_names:
        return ""
    return Counter(section_names).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class DocumentChunker:

    def __init__(
        self,
        chunk_size: int    = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer     = _load_tokenizer()

        self._splitter = SentenceSplitter(
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            tokenizer     = self.tokenizer.encode if self.tokenizer else None,
        )

        logger.info(
            "DocumentChunker initialised | size=%d | overlap=%d",
            chunk_size, chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, parse_result: ParseResult, embed_fn = None) -> list[ChunkResult]:
        if not parse_result.success:
            logger.warning("Skipping failed ParseResult | %s", parse_result.file_name)
            return []

        if not parse_result.blocks:
            logger.warning(
                "No blocks in ParseResult | %s — nothing to chunk",
                parse_result.file_name,
            )
            return []

        logger.info(
            "Chunking | file=%s | blocks=%d",
            parse_result.file_name, len(parse_result.blocks),
        )

        chunks = self._process_blocks(
            blocks          = parse_result.blocks,
            page_records    = parse_result.pages,
            doc_id          = parse_result.doc_id,
            source_file     = parse_result.file_name,
            filename_tokens = parse_result.filename_tokens,
            doc_year        = parse_result.doc_year,
            has_tables      = parse_result.meta.has_tables if parse_result.meta else False,
            summary         = parse_result.summary,
            keywords        = parse_result.keywords,
            embed_fn        = embed_fn,
        )

        chunks = self._deduplicate(chunks, parse_result.file_name)

        # Production-level: merge small chunks with neighbors
        before_merge = len(chunks)
        chunks = _merge_small_chunks(chunks, self.tokenizer)
        if len(chunks) < before_merge:
            logger.info(
                "Small-chunk merge | file=%s | before=%d after=%d",
                parse_result.file_name, before_merge, len(chunks),
            )

        # Production-level: drop chunks below absolute minimum
        chunks = [c for c in chunks if c.token_count >= MIN_EMBEDDABLE_TOKENS or c.is_table]

        self._log_summary(chunks, parse_result.file_name)
        return chunks

    def chunk_batch(
        self, parse_results: list[ParseResult]
    ) -> dict[str, list[ChunkResult]]:
        results     = {pr.file_name: self.chunk(pr) for pr in parse_results}
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(
            "Batch done | files=%d | total_chunks=%d",
            len(results), total_chunks,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, chunks: list[ChunkResult], file_name: str) -> list[ChunkResult]:
        before  = len(chunks)
        chunks  = _dedup_exact(chunks)
        chunks  = _dedup_near_smart(chunks)
        removed = before - len(chunks)
        if removed:
            logger.info(
                "Dedup removed %d chunks | file=%s | before=%d after=%d",
                removed, file_name, before, len(chunks),
            )
        return chunks

    def _log_summary(self, chunks: list[ChunkResult], file_name: str) -> None:
        logger.info(
            "Done | %s | chunks=%d | tables=%d",
            file_name,
            len(chunks),
            sum(1 for c in chunks if c.is_table),
        )

    def _split_text(self, text: str, embed_fn = None) -> list[str]:
        semantic_enabled = getattr(config, "SEMANTIC_CHUNKING_ENABLED", False)
        if semantic_enabled and embed_fn:
            threshold = getattr(config, "SEMANTIC_SPLIT_THRESHOLD", 0.6)
            chunks = semantic_split(
                text=text,
                embed_fn=embed_fn,
                threshold=threshold,
                max_chunk_size=self.chunk_size,
                tokenizer=self.tokenizer
            )
            if chunks:
                return chunks

        # Temporarily hide numbered list periods from LlamaIndex
        protected_text = re.sub(r"\b(\d{1,3})\.(?=\s+[A-Za-z])", r"\1<DOT>", text)
        try:
            nodes = self._splitter.get_nodes_from_documents([Document(text=protected_text)])
        except Exception as exc:
            logger.warning("SentenceSplitter failed: %s", exc)
            return [text]

        return [
            (node.get_content() if hasattr(node, "get_content") else node.text).strip().replace("<DOT>", ".")
            for node in nodes
        ]

    # ------------------------------------------------------------------
    # Core block processing
    # ------------------------------------------------------------------

    def _process_blocks(
        self,
        blocks:          list[BlockRecord],
        page_records:    list[PageRecord],
        doc_id:          str       = "",
        source_file:     str       = "",
        filename_tokens: list      = None,
        doc_year:        str       = "",
        has_tables:      bool      = False,
        summary:         str       = "",
        keywords:        list      = None,
        embed_fn                  = None,
    ) -> list[ChunkResult]:

        chunks: list[ChunkResult] = []
        chunk_index               = 0

        text_buffer: list[BlockRecord] = []
        buffer_token_count             = 0
        heading_stack: list[Tuple[int, str]] = []

        # ------------------------------------------------------------------
        # flush_buffer — with cross-flush overlap (production-level)
        # ------------------------------------------------------------------
        def flush_buffer(keep_overlap: bool = True) -> int:
            """Flush buffer into chunks. Returns overlap token count carried over."""
            nonlocal chunk_index

            if not text_buffer:
                return 0

            merged_text = "\n".join(
                _sanitize_whitespace(block.content) for block in text_buffer
            )

            if not _is_embeddable(merged_text):
                text_buffer.clear()
                return 0

            section = _dominant_section(text_buffer)
            heading_path = " > ".join(title for lvl, title in heading_stack) if heading_stack else ""
            if not heading_path:
                heading_path = section or ""

            start_page = text_buffer[0].page_no
            end_page   = text_buffer[-1].page_no
            page_label = text_buffer[0].page_label

            for split_text in self._split_text(merged_text, embed_fn=embed_fn):
                if not _is_embeddable(split_text):
                    continue

                display_text = f"[{heading_path}]\n{split_text}" if heading_path else split_text
                chunk_keywords = _extract_chunk_keywords(split_text)
                # Merge document-level keywords with chunk-level keywords
                all_keywords = list(set((keywords or []) + chunk_keywords))[:30]

                chunks.append(ChunkResult(
                    chunk_id        = _make_chunk_id(doc_id, chunk_index),
                    chunk_index     = chunk_index,
                    text            = display_text,
                    page_no         = start_page,
                    page_label      = page_label,
                    page_range      = [start_page, end_page],
                    section         = heading_path,
                    heading_path    = heading_path,
                    is_table        = False,
                    token_count     = _count_tokens(display_text, self.tokenizer),
                    chunk_year      = _extract_years(split_text),
                    doc_id          = doc_id,
                    source_file     = source_file,
                    filename_tokens = filename_tokens or [],
                    doc_year        = doc_year,
                    has_tables      = has_tables,
                    summary         = summary,
                    keywords        = all_keywords,
                ))
                chunk_index += 1

            # Cross-flush overlap: carry last block(s) into next buffer
            if keep_overlap and self.chunk_overlap > 0 and text_buffer:
                overlap_blocks = []
                overlap_tokens = 0
                for block in reversed(text_buffer):
                    block_tokens = _count_tokens(block.content, self.tokenizer)
                    if overlap_tokens + block_tokens <= self.chunk_overlap:
                        overlap_blocks.insert(0, block)
                        overlap_tokens += block_tokens
                    else:
                        break
                text_buffer.clear()
                text_buffer.extend(overlap_blocks)
                return overlap_tokens
            else:
                text_buffer.clear()
                return 0

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        for block_index, block in enumerate(blocks):
            content = _sanitize_whitespace(block.content)

            # Heading starts a new section — flush with no overlap
            if block.block_type in ("heading", "title"):
                flush_buffer(keep_overlap=False)
                buffer_token_count = 0
                
                # Track heading stack
                level = _determine_heading_level(content)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, content))
                
                text_buffer.append(block)
                continue

            # table block
            if block.is_table:
                start_page = block.page_no
                end_page   = block.page_no

                preceding_block = blocks[block_index - 1] if block_index > 0 else None
                bonded_context  = ""
                if (
                    preceding_block
                    and preceding_block.block_type == "text"
                    and preceding_block.content.strip()
                ):
                    bonded_context = _sanitize_whitespace(preceding_block.content)

                table_title = bonded_context.strip() if len(bonded_context) < 150 else ""
                
                # Prevent table title from being indexed as a standalone text chunk
                if table_title and text_buffer and text_buffer[-1] is preceding_block:
                    text_buffer.pop()
                    buffer_token_count -= _count_tokens(preceding_block.content, self.tokenizer)
                    buffer_token_count = max(0, buffer_token_count)

                flush_buffer(keep_overlap=False)
                buffer_token_count = 0

                heading_path = " > ".join(title for lvl, title in heading_stack) if heading_stack else ""
                if not heading_path:
                    heading_path = block.section or ""

                json_rows, headers = _process_table_to_json(content)
                if not json_rows:
                    continue

                # Large table guard: if too many rows, batch them into grouped chunks
                total_table_tokens = _count_tokens(content, self.tokenizer)
                if total_table_tokens > TABLE_MAX_TOKENS and len(json_rows) > 10:
                    # Group rows into batches that fit within token budget
                    batch_rows = []
                    batch_token_count = 0
                    for row in json_rows:
                        row_prose = ", ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
                        row_tokens = _count_tokens(row_prose, self.tokenizer)
                        if batch_token_count + row_tokens > TABLE_MAX_TOKENS and batch_rows:
                            # Flush current batch
                            embed_parts = []
                            if table_title:
                                embed_parts.append(f"Table Title: {table_title}")
                            if heading_path:
                                embed_parts.append(f"Section: {heading_path}")
                            embed_parts.extend(batch_rows)
                            display_text = "\n".join(embed_parts)
                            table_keywords = _extract_chunk_keywords(display_text)
                            chunks.append(ChunkResult(
                                chunk_id        = _make_chunk_id(doc_id, chunk_index),
                                chunk_index     = chunk_index,
                                text            = display_text,
                                full_table_context = content,
                                page_no         = start_page,
                                page_label      = block.page_label,
                                page_range      = [start_page, end_page],
                                section         = heading_path,
                                heading_path    = heading_path,
                                is_table        = True,
                                token_count     = _count_tokens(display_text, self.tokenizer),
                                chunk_year      = _extract_years(display_text),
                                doc_id          = doc_id,
                                source_file     = source_file,
                                filename_tokens = filename_tokens or [],
                                doc_year        = doc_year,
                                has_tables      = has_tables,
                                summary         = summary,
                                keywords        = list(set((keywords or []) + table_keywords))[:30],
                            ))
                            chunk_index += 1
                            batch_rows = []
                            batch_token_count = 0
                        batch_rows.append(row_prose)
                        batch_token_count += row_tokens
                    # Flush remaining batch
                    if batch_rows:
                        embed_parts = []
                        if table_title:
                            embed_parts.append(f"Table Title: {table_title}")
                        if heading_path:
                            embed_parts.append(f"Section: {heading_path}")
                        embed_parts.extend(batch_rows)
                        display_text = "\n".join(embed_parts)
                        table_keywords = _extract_chunk_keywords(display_text)
                        chunks.append(ChunkResult(
                            chunk_id        = _make_chunk_id(doc_id, chunk_index),
                            chunk_index     = chunk_index,
                            text            = display_text,
                            full_table_context = content,
                            page_no         = start_page,
                            page_label      = block.page_label,
                            page_range      = [start_page, end_page],
                            section         = heading_path,
                            heading_path    = heading_path,
                            is_table        = True,
                            token_count     = _count_tokens(display_text, self.tokenizer),
                            chunk_year      = _extract_years(display_text),
                            doc_id          = doc_id,
                            source_file     = source_file,
                            filename_tokens = filename_tokens or [],
                            doc_year        = doc_year,
                            has_tables      = has_tables,
                            summary         = summary,
                            keywords        = list(set((keywords or []) + table_keywords))[:30],
                        ))
                        chunk_index += 1
                else:
                    # Normal row-by-row chunking for small/medium tables
                    for row_index, row in enumerate(json_rows):
                        embed_parts = []
                        if table_title:
                            embed_parts.append(f"Table Title: {table_title}")
                        if heading_path:
                            embed_parts.append(f"Section: {heading_path}")
                        
                        row_prose = ", ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
                        embed_parts.append(row_prose)
                        display_text = "\n".join(embed_parts)
                        table_keywords = _extract_chunk_keywords(display_text)

                        chunks.append(ChunkResult(
                            chunk_id        = _make_chunk_id(doc_id, chunk_index),
                            chunk_index     = chunk_index,
                            text            = display_text,
                            full_table_context = content,
                            page_no         = start_page,
                            page_label      = block.page_label,
                            page_range      = [start_page, end_page],
                            section         = heading_path,
                            heading_path    = heading_path,
                            is_table        = True,
                            token_count     = _count_tokens(display_text, self.tokenizer),
                            chunk_year      = _extract_years(display_text),
                            doc_id          = doc_id,
                            source_file     = source_file,
                            filename_tokens = filename_tokens or [],
                            doc_year        = doc_year,
                            has_tables      = has_tables,
                            summary         = summary,
                            keywords        = list(set((keywords or []) + table_keywords))[:30],
                        ))
                        chunk_index += 1
                continue

            # text block
            if not _is_embeddable(content):
                continue

            section_changed = (
                text_buffer and block.section != text_buffer[-1].section
            )
            if section_changed:
                flush_buffer(keep_overlap=False)
                buffer_token_count = 0

            text_buffer.append(block)
            buffer_token_count += _count_tokens(content, self.tokenizer)

            if buffer_token_count >= self.chunk_size:
                overlap_tokens = flush_buffer(keep_overlap=True)
                buffer_token_count = overlap_tokens

        flush_buffer(keep_overlap=False)

        return chunks