"""
final_rag/agent/assembler.py

Context Assembler: Fits retrieved chunks into the LLM context window (45,000 tokens),
formats structured citation blocks (direct, compare, synthesize), and produces SourceInfo metadata.
"""

from __future__ import annotations
import logging
import time
from collections import defaultdict
from typing import List

try:
    import final_rag.config as config
except ImportError:
    from .. import config

from .models import CleanedQuery, RetrievedChunk, SourceInfo, AssembledResult

logger = logging.getLogger("agent.assembler")

MAX_CONTEXT_TOKENS = getattr(config, "MAX_CONTEXT_TOKENS", 45000)
TOKENS_PER_WORD    = getattr(config, "TOKENS_PER_WORD", 1.3)


class Assembler:
    def __init__(self, max_context_tokens: int = MAX_CONTEXT_TOKENS):
        self.max_context_tokens = max_context_tokens

    def assemble(
        self,
        chunks:  list[RetrievedChunk],
        cleaned: CleanedQuery,
    ) -> AssembledResult:
        start = time.perf_counter()
        if not chunks:
            return self._not_found_result(start, cleaned.answer_structure)

        # Budget tokens
        selected, was_trimmed = self._budget_chunks(chunks)
        if not selected:
            return self._not_found_result(start, cleaned.answer_structure)

        # Order chunks logically
        ordered = self._order_chunks(selected, cleaned.answer_structure)

        # Build context block based on answer structure
        if cleaned.answer_structure == "compare":
            context_block = self._build_compare_block(ordered)
        elif cleaned.answer_structure == "synthesize":
            context_block = self._build_synthesize_block(ordered)
        else:
            context_block = self._build_direct_block(ordered)

        sources = self._build_sources(ordered)
        elapsed = round(time.perf_counter() - start, 3)

        return AssembledResult(
            context_block       = context_block,
            sources             = sources,
            chunks_used         = len(ordered),
            has_weak_match      = any(c.is_weak_match for c in ordered),
            has_tables          = any(c.is_table for c in ordered),
            not_found           = False,
            was_trimmed         = was_trimmed,
            processing_time_sec = elapsed,
            answer_structure    = cleaned.answer_structure,
            sources_count       = len(sources),
        )

    def _budget_chunks(self, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], bool]:
        selected = []
        current_tokens = 0
        was_trimmed = False

        for chunk in chunks:
            tokens = self._chunk_tokens(chunk)
            if current_tokens + tokens > self.max_context_tokens:
                was_trimmed = True
                break
            selected.append(chunk)
            current_tokens += tokens

        return selected, was_trimmed

    def _order_chunks(self, chunks: list[RetrievedChunk], answer_structure: str) -> list[RetrievedChunk]:
        if answer_structure == "compare":
            return sorted(chunks, key=lambda c: (c.arm_label or "General", c.source_file, c.page_no, c.chunk_index))
        return sorted(chunks, key=lambda c: (c.source_file, c.page_no, c.chunk_index))

    def _build_direct_block(self, chunks: list[RetrievedChunk]) -> str:
        lines = ["=== KNOWLEDGE BASE CONTEXT ==="]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"\n--- [Context {i}] ---")
            lines.append(self._format_chunk(chunk, show_file=True))
        lines.append("\n=== END OF CONTEXT ===")
        return "\n".join(lines)

    def _build_compare_block(self, chunks: list[RetrievedChunk]) -> str:
        arm_groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            arm_groups[chunk.arm_label or "General"].append(chunk)

        lines = ["=== COMPARISON CONTEXT ==="]
        for arm_label, arm_chunks in arm_groups.items():
            sample = arm_chunks[0]
            lines.append(f"\n─── {arm_label.upper()} ───")
            lines.append(f"Source: {sample.source_file}")
            if sample.doc_year:
                lines.append(f"Year: {sample.doc_year}")
            for chunk in arm_chunks:
                lines.append(self._format_chunk(chunk, show_file=False))
        lines.append("=== END OF COMPARISON CONTEXT ===")
        return "\n".join(lines)

    def _build_synthesize_block(self, chunks: list[RetrievedChunk]) -> str:
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.source_file].append(chunk)

        lines = ["=== SYNTHESIZED CONTEXT FROM MULTIPLE DOCUMENTS ==="]
        for source_file, doc_chunks in grouped.items():
            sample = doc_chunks[0]
            header = f"{source_file} ({sample.doc_year})" if sample.doc_year else source_file
            lines.append(f"\n─── {header} ───")
            for chunk in doc_chunks:
                lines.append(self._format_chunk(chunk, show_file=False))
        lines.append("=== END OF SYNTHESIZED CONTEXT ===")
        return "\n".join(lines)

    def _format_chunk(self, chunk: RetrievedChunk, show_file: bool = True) -> str:
        page_str = chunk.page_label if chunk.page_label else (str(chunk.page_no) if chunk.page_no > 0 else "Unknown")
        data_type = "[TABLE]" if chunk.is_table else "[TEXT]"
        section   = chunk.heading or "General"

        header_parts = []
        if show_file:
            header_parts.append(f"File: {chunk.source_file}")
        header_parts.append(f"Page {page_str} | {section} {data_type}")

        header = " | ".join(header_parts)
        return f"{header}\n{chunk.text.strip()}\n"

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[SourceInfo]:
        grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.source_file].append(chunk)

        sources = []
        for file_name, file_chunks in grouped.items():
            pages_set = set()
            for c in file_chunks:
                label = c.page_label if c.page_label else (str(c.page_no) if c.page_no > 0 else "1")
                pages_set.add(label)

            pages = sorted(list(pages_set), key=lambda x: int(x) if x.isdigit() else 9999)
            sources.append(SourceInfo(
                file_name   = file_name,
                pages       = pages,
                chunk_count = len(file_chunks),
            ))
        return sources

    def _not_found_result(self, start: float, answer_structure: str) -> AssembledResult:
        return AssembledResult(
            context_block       = "No relevant matches found in the knowledge base.",
            sources             = [],
            chunks_used         = 0,
            has_weak_match      = False,
            has_tables          = False,
            not_found           = True,
            was_trimmed         = False,
            processing_time_sec = round(time.perf_counter() - start, 3),
            answer_structure    = answer_structure,
            sources_count       = 0,
        )

    @staticmethod
    def _chunk_tokens(chunk: RetrievedChunk) -> int:
        if getattr(chunk, "token_count", 0):
            return chunk.token_count
        return int(len(chunk.text.split()) * TOKENS_PER_WORD)


def get_assembler() -> Assembler:
    return Assembler()