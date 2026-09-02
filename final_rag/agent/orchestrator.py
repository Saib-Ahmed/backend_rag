"""
final_rag/agent/orchestrator.py

Orchestrator: Coordinates Query Cleaner, Retriever, Assembler, and Claude Generator.
Includes multi-turn history sanitization (350 char excerpt cap, 1500 token history budget),
active document routing, chitchat bypass, and token streaming.
"""

from __future__ import annotations
import json
import logging
import re
import time
import traceback
from typing import Generator, List, Dict, Any, Optional

try:
    import final_rag.config as config
except ImportError:
    from .. import config

from .models import OrchestratorResult, SourceInfo, AssembledResult, CleanedQuery
from .query_cleaner import get_cleaner
from .retriever import get_retriever
from .assembler import get_assembler
from .claude_client import ClaudeClient
from ..ingestion.embedder import OllamaEmbedder
from ..prompts.generator_prompt import get_generator_prompt, get_chitchat_prompt

logger = logging.getLogger("agent.orchestrator")

MODEL                    = getattr(config, "GENERATOR_MODEL", "claude-3-5-sonnet-20241022")
MAX_HISTORY_TURNS        = getattr(config, "MAX_HISTORY_TURNS", 3)
MAX_TOKENS               = getattr(config, "CLAUDE_CHAT_STREAM_MAX_TOKENS", 9000)
CHITCHAT_MAX_TOKENS      = getattr(config, "CHITCHAT_MAX_TOKENS", 512)
MAX_HISTORY_ANSWER_CHARS = getattr(config, "MAX_HISTORY_ANSWER_CHARS", 350)
MAX_HISTORY_TOTAL_TOKENS = getattr(config, "MAX_HISTORY_TOTAL_TOKENS", 1500)

CLARIFICATION_MESSAGE  = "I couldn't find specific information about that in the uploaded documents. Could you please rephrase or specify which case/order you are looking for?"
FALLBACK_ERROR_MESSAGE = "An unexpected error occurred while processing your request. Please try again."


class Orchestrator:
    def __init__(self, embedder: OllamaEmbedder):
        self.cleaner   = get_cleaner()
        self.retriever = get_retriever(embedder)
        self.assembler = get_assembler()
        self.client    = ClaudeClient(model=MODEL)

    def run(
        self,
        query: str,
        history: list[dict] = None,
        active_doc_id: Optional[str] = None,
        active_doc_filename: Optional[str] = None,
        active_doc_ids: Optional[List[str]] = None,
        active_doc_filenames: Optional[List[str]] = None,
        active_documents: Optional[List[str]] = None,
        domain: str = "",
    ) -> OrchestratorResult:
        total_start = time.perf_counter()
        times = {}

        try:
            # 1. Cleaner
            t = time.perf_counter()
            cleaned = self.cleaner.clean(
                query,
                active_documents    = active_documents,
                active_doc_id       = active_doc_id,
                active_doc_filename = active_doc_filename,
                active_doc_ids      = active_doc_ids,
                active_doc_filenames = active_doc_filenames,
            )
            times["cleaner"] = round(time.perf_counter() - t, 3)

            history_str = self._format_history(history or [])

            # Chitchat branch
            if cleaned.is_chitchat:
                t = time.perf_counter()
                prompt = get_chitchat_prompt().format(
                    history_str       = history_str,
                    original_query    = query,
                    detected_language = cleaned.detected_language,
                )
                answer = self.client.generate(prompt=prompt, max_tokens=CHITCHAT_MAX_TOKENS)
                times["generator"] = round(time.perf_counter() - t, 3)
                total = round(time.perf_counter() - total_start, 3)
                return OrchestratorResult(
                    answer         = answer,
                    sources        = [],
                    cleaned_query  = cleaned.primary_query,
                    pipeline_times = times,
                    total_time_sec = total,
                )

            # 2. Retriever
            t = time.perf_counter()
            chunks = self.retriever.retrieve(cleaned, domain=domain)
            times["retriever"] = round(time.perf_counter() - t, 3)

            # 3. Assembler
            t = time.perf_counter()
            assembled = self.assembler.assemble(chunks, cleaned)
            times["assembler"] = round(time.perf_counter() - t, 3)

            if assembled.not_found:
                total = round(time.perf_counter() - total_start, 3)
                return OrchestratorResult(
                    answer               = CLARIFICATION_MESSAGE,
                    sources              = [],
                    clarification_needed = True,
                    cleaned_query        = cleaned.primary_query,
                    pipeline_times       = times,
                    total_time_sec       = total,
                )

            # 4. Generator
            t = time.perf_counter()
            template = get_generator_prompt(assembled.answer_structure)
            prompt   = template.format(
                history_str       = history_str,
                original_query    = query,
                detected_language = cleaned.detected_language,
                context_block     = assembled.context_block,
            )
            answer = self.client.generate(prompt=prompt, max_tokens=MAX_TOKENS)
            times["generator"] = round(time.perf_counter() - t, 3)

            total = round(time.perf_counter() - total_start, 3)
            return OrchestratorResult(
                answer         = answer,
                sources        = assembled.sources,
                cleaned_query  = cleaned.primary_query,
                pipeline_times = times,
                total_time_sec = total,
            )

        except Exception as e:
            logger.error("Orchestrator error: %s\n%s", e, traceback.format_exc())
            return OrchestratorResult(
                answer = FALLBACK_ERROR_MESSAGE,
                error  = str(e),
                total_time_sec = round(time.perf_counter() - total_start, 3),
            )

    def stream(
        self,
        query: str,
        history: list[dict] = None,
        active_doc_id: Optional[str] = None,
        active_doc_filename: Optional[str] = None,
        active_doc_ids: Optional[List[str]] = None,
        active_doc_filenames: Optional[List[str]] = None,
        active_documents: Optional[List[str]] = None,
        domain: str = "",
    ) -> Generator[str, None, None]:
        total_start = time.perf_counter()
        times = {}

        try:
            t = time.perf_counter()
            cleaned = self.cleaner.clean(
                query,
                active_documents     = active_documents,
                active_doc_id        = active_doc_id,
                active_doc_filename  = active_doc_filename,
                active_doc_ids       = active_doc_ids,
                active_doc_filenames = active_doc_filenames,
            )
            times["cleaner"] = round(time.perf_counter() - t, 3)
            history_str = self._format_history(history or [])

            if cleaned.is_chitchat:
                t = time.perf_counter()
                yield from self._stream_chitchat(query, cleaned.detected_language, history_str, t)
                return

            t = time.perf_counter()
            chunks = self.retriever.retrieve(cleaned, domain=domain)
            times["retriever"] = round(time.perf_counter() - t, 3)

            t = time.perf_counter()
            assembled = self.assembler.assemble(chunks, cleaned)
            times["assembler"] = round(time.perf_counter() - t, 3)

            if assembled.not_found:
                yield CLARIFICATION_MESSAGE
                return

            t = time.perf_counter()
            yield from self._stream_generate(
                original_query    = query,
                improved_query    = cleaned.primary_query,
                detected_language = cleaned.detected_language,
                assembled         = assembled,
                history_str       = history_str,
                stage_start       = t,
            )
            times["generator"] = round(time.perf_counter() - t, 3)

            if assembled.sources:
                yield self._build_metadata_payload(assembled.sources)

        except Exception as e:
            logger.error("Streaming error: %s\n%s", e, traceback.format_exc())
            yield FALLBACK_ERROR_MESSAGE

    def _stream_generate(
        self,
        original_query:    str,
        improved_query:    str,
        detected_language: str,
        assembled:         AssembledResult,
        history_str:       str,
        stage_start:       float,
    ) -> Generator[str, None, None]:
        template = get_generator_prompt(assembled.answer_structure)
        prompt   = template.format(
            history_str       = history_str,
            original_query    = original_query,
            detected_language = detected_language,
            context_block     = assembled.context_block,
        )

        try:
            stream = self.client.chat_stream(
                messages    = [{"role": "user", "content": prompt}],
                temperature = getattr(config, "GENERATOR_TEMPERATURE", 0.25),
                max_tokens  = MAX_TOKENS,
            )
            for token in stream:
                yield token
        except Exception as e:
            logger.error("Stream failed: %s", e)
            yield FALLBACK_ERROR_MESSAGE

    def _stream_chitchat(
        self,
        original_query:    str,
        detected_language: str,
        history_str:       str,
        stage_start:       float,
    ) -> Generator[str, None, None]:
        template = get_chitchat_prompt()
        prompt   = template.format(
            history_str       = history_str,
            original_query    = original_query,
            detected_language = detected_language,
        )
        try:
            stream = self.client.chat_stream(
                messages    = [{"role": "user", "content": prompt}],
                temperature = getattr(config, "GENERATOR_TEMPERATURE", 0.25),
                max_tokens  = CHITCHAT_MAX_TOKENS,
            )
            for token in stream:
                yield token
        except Exception as e:
            logger.error("Chitchat stream failed: %s", e)
            yield FALLBACK_ERROR_MESSAGE

    @staticmethod
    def _build_metadata_payload(sources: list[SourceInfo]) -> str:
        try:
            return f"__METADATA__:{json.dumps([s.model_dump() for s in sources])}"
        except Exception as e:
            logger.error("[Orchestrator] Metadata serialization failed: %s", e)
            return ""

    @staticmethod
    def _clean_history_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"[*_]{1,3}", "", text)
        text = re.sub(r"^-\s+", "", text, flags=re.MULTILINE)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _truncate_excerpt(text: str, max_chars: int = MAX_HISTORY_ANSWER_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "No previous context."

        recent = history[-MAX_HISTORY_TURNS:]
        kept_reversed: list[str] = []
        total_tokens = 0

        for t in reversed(recent):
            question       = (t.get("question", "") or "").strip()
            answer_raw     = t.get("answer", "") or ""
            answer_clean   = Orchestrator._clean_history_text(answer_raw)
            answer_excerpt = Orchestrator._truncate_excerpt(answer_clean)

            entry        = f"User: {question}\nAssistant: {answer_excerpt}"
            entry_tokens = int(len(entry.split()) * getattr(config, "TOKENS_PER_WORD", 1.3))

            if total_tokens + entry_tokens > MAX_HISTORY_TOTAL_TOKENS:
                break

            kept_reversed.append(entry)
            total_tokens += entry_tokens

        if not kept_reversed:
            return "No previous context."

        return "\n".join(reversed(kept_reversed))


def get_orchestrator(embedder: OllamaEmbedder) -> Orchestrator:
    return Orchestrator(embedder=embedder)