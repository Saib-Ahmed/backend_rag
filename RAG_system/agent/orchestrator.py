"""
agent/orchestrator.py

Orchestrator — final stage of the RAG pipeline.
Pipeline: QueryCleaner → Retriever → Assembler → Orchestrator → Response
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import asyncio
from typing import Generator, AsyncGenerator


import ollama

from agent.models import OrchestratorResult, SourceInfo, AssembledResult, CleanedQuery
from agent.query_cleaner import get_cleaner
from agent.retriever import get_retriever
from agent.assembler import get_assembler
from new_ingestion.embedder import OllamaEmbedder
from prompts.generator_prompt import get_generator_prompt
import config

logger = logging.getLogger("agent.orchestrator")

# ── Constants ──────────────────────────────────────────────────────────
MODEL             = config.GENERATOR_MODEL
MAX_HISTORY_TURNS = config.ORCHESTRATOR_MAX_HISTORY_TURNS
MAX_RETRIES       = config.ORCHESTRATOR_MAX_RETRIES
NUM_CTX           = config.NUM_CTX
MAX_TOKENS        = config.MAX_TOKENS

CLARIFICATION_MESSAGE = config.CLARIFICATION_MESSAGE

FALLBACK_ERROR_MESSAGE = config.FALLBACK_ERROR_MESSAGE


# ── Orchestrator ───────────────────────────────────────────────────────
class Orchestrator:

    def __init__(self, embedder: OllamaEmbedder, graph_rag=None):
        self.embedder  = embedder
        self.cleaner   = get_cleaner(model=config.CLEANER_MODEL)
        self.retriever = get_retriever(embedder=embedder, graph_rag=graph_rag)
        self.assembler = get_assembler()
        self.client    = ollama.Client(host=config.OLLAMA_BASE_URL)
        logger.info("Orchestrator initialized | generator=%s", MODEL)

    # ── Main pipeline entry ────────────────────────────────────────────
    def run(
        self,
        query:           str,
        history:         list[dict] = None,
        active_document: str = None,
        enable_graph_rag: bool = True,
    ) -> Generator[str, None, None]:

        total_start = time.perf_counter()
        history     = history or []
        times: dict[str, float] = {}

        try:
            # ── Stage 1: Clean ─────────────────────────────────────────
            t       = time.perf_counter()
            cleaned = self.cleaner.clean(query, active_document=active_document)
            times["cleaner"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Cleaned | %.3fs | scope=%s structure=%s specificity=%s",
                times["cleaner"], cleaned.target_scope,
                cleaned.answer_structure, cleaned.specificity,
            )

            # ── Stage 2: Retrieve ──────────────────────────────────────
            t      = time.perf_counter()
            chunks = self.retriever.retrieve(cleaned, enable_graph_rag=enable_graph_rag)
            times["retriever"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Retrieved %d chunks | %.3fs",
                len(chunks), times["retriever"],
            )

            # ── Stage 3: Assemble ──────────────────────────────────────
            t         = time.perf_counter()
            assembled = self.assembler.assemble(
                cleaned,
                chunks,
                relationships=self.retriever.last_relationships if hasattr(self.retriever, "last_relationships") else None
            )
            times["assembler"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Assembled | not_found=%s weak=%s "
                "tables=%s structure=%s sources=%d | %.3fs",
                assembled.not_found, assembled.has_weak_match,
                assembled.has_tables, assembled.answer_structure,
                assembled.sources_count, times["assembler"],
            )

            # no relevant chunks found — ask user to clarify
            if assembled.not_found:
                yield CLARIFICATION_MESSAGE
                return

            # ── Stage 4: Generate ──────────────────────────────────────
            history_str = self._format_history(history)
            t = time.perf_counter()
            
            yield from self._stream_generate(
                original_query    = query,
                improved_query    = cleaned.improved_query,
                detected_language = cleaned.detected_language,
                assembled         = assembled,
                history_str       = history_str
            )
            times["generator"] = round(time.perf_counter() - t, 3)

            # ── Metadata suffix — sources for frontend citation ────────
            if assembled.sources:
                yield self._build_metadata_payload(assembled.sources)

            total = round(time.perf_counter() - total_start, 3)
            logger.info(
                "[Orchestrator] Done | total=%.3fs | times=%s",
                total, times,
            )

        except Exception as e:
            logger.error(
                "[Orchestrator] Pipeline failed: %s\n%s",
                e, traceback.format_exc(),
            )
            yield FALLBACK_ERROR_MESSAGE

    # ── Async pipeline entry ───────────────────────────────────────────
    async def async_run(
        self,
        query:           str,
        history:         list[dict] = None,
        active_document: str = None,
        enable_graph_rag: bool = True,
    ) -> AsyncGenerator[str, None]:
        total_start = time.perf_counter()
        history     = history or []
        times: dict[str, float] = {}

        try:
            # ── Stage 1: Clean ─────────────────────────────────────────
            t       = time.perf_counter()
            cleaned = await asyncio.to_thread(self.cleaner.clean, query, active_document=active_document)
            times["cleaner"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Async Cleaned | %.3fs | scope=%s structure=%s specificity=%s",
                times["cleaner"], cleaned.target_scope,
                cleaned.answer_structure, cleaned.specificity,
            )

            # ── Stage 2: Retrieve ──────────────────────────────────────
            t      = time.perf_counter()
            chunks = await self.retriever.async_retrieve(cleaned, enable_graph_rag=enable_graph_rag)
            times["retriever"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Async Retrieved %d chunks | %.3fs",
                len(chunks), times["retriever"],
            )

            # ── Stage 3: Assemble ──────────────────────────────────────
            t         = time.perf_counter()
            assembled = await asyncio.to_thread(
                self.assembler.assemble,
                cleaned,
                chunks,
                relationships=self.retriever.last_relationships if hasattr(self.retriever, "last_relationships") else None
            )
            times["assembler"] = round(time.perf_counter() - t, 3)
            logger.info(
                "[Orchestrator] Async Assembled | not_found=%s weak=%s "
                "tables=%s structure=%s sources=%d | %.3fs",
                assembled.not_found, assembled.has_weak_match,
                assembled.has_tables, assembled.answer_structure,
                assembled.sources_count, times["assembler"],
            )

            # no relevant chunks found — ask user to clarify
            if assembled.not_found:
                yield CLARIFICATION_MESSAGE
                return

            # ── Stage 4: Generate ──────────────────────────────────────
            history_str = self._format_history(history)
            t = time.perf_counter()
            
            q = asyncio.Queue()
            loop = asyncio.get_event_loop()
            def run_stream():
                try:
                    for token in self._stream_generate(
                        original_query    = query,
                        improved_query    = cleaned.improved_query,
                        detected_language = cleaned.detected_language,
                        assembled         = assembled,
                        history_str       = history_str
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, token)
                except Exception as ex:
                    logger.error("Error in sync stream generate thread: %s", ex)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            await loop.run_in_executor(None, run_stream)
            while True:
                token = await q.get()
                if token is None:
                    break
                yield token
                
            times["generator"] = round(time.perf_counter() - t, 3)

            # ── Metadata suffix — sources for frontend citation ────────
            if assembled.sources:
                yield self._build_metadata_payload(assembled.sources)

            total = round(time.perf_counter() - total_start, 3)
            logger.info(
                "[Orchestrator] Async Done | total=%.3fs | times=%s",
                total, times,
            )

        except Exception as e:
            logger.error(
                "[Orchestrator] Async Pipeline failed: %s\n%s",
                e, traceback.format_exc(),
            )
            yield FALLBACK_ERROR_MESSAGE

    # ── Non-streaming run (returns full answer string + sources) ────────

    def run_sync(
        self,
        query:           str,
        history:         list[dict] = None,
        active_document: str = None,
        enable_graph_rag: bool = True,
    ) -> tuple[str, list[SourceInfo]]:
        """
        Synchronous entry point used by rag_engine.py wrapper.
        Returns (answer_str, sources_list).
        """
        answer_parts = []
        sources = []
        for token in self.run(query, history, active_document, enable_graph_rag=enable_graph_rag):
            if token.startswith("__METADATA__:"):
                try:
                    raw = json.loads(token[len("__METADATA__:"):])
                    sources = [SourceInfo(**s) for s in raw]
                except Exception:
                    pass
            else:
                answer_parts.append(token)
        return "".join(answer_parts), sources

    # ── Stream generator output ────────────────────────────────────────
    def _stream_generate(
        self,
        original_query:    str,
        improved_query:    str,
        detected_language: str,
        assembled:         AssembledResult,
        history_str:       str,
    ) -> Generator[str, None, None]:
        template = get_generator_prompt(assembled.answer_structure)
        prompt   = template.format(
            history_str       = history_str,
            original_query    = original_query,
            detected_language = detected_language,
            context_block     = assembled.context_block,
        )

        try:
            stream = self.client.chat(
                model      = MODEL,
                messages   = [{"role": "user", "content": prompt}],
                keep_alive = "5m",
                stream     = True,
                think      = False,
                options    = {
                    "temperature": config.GENERATOR_TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "num_ctx":     NUM_CTX,
                },
            )
            for chunk in stream:
                token = chunk.message.content
                if token:
                    yield token

        except Exception as e:
            logger.error("[Orchestrator] Stream failed: %s", e)
            yield FALLBACK_ERROR_MESSAGE

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _build_metadata_payload(sources: list[SourceInfo]) -> str:
        try:
            return f"__METADATA__:{json.dumps([s.model_dump() for s in sources])}"
        except Exception as e:
            logger.error("[Orchestrator] Metadata serialization failed: %s", e)
            return ""

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "No previous context."
        recent = history[-MAX_HISTORY_TURNS:]
        lines = []
        for t in recent:
            # Support both formats: {role, content} and {question, answer}
            if "role" in t:
                role = t.get("role", "")
                content = t.get("content", "")
                if role == "user":
                    lines.append(f"User: {content}")
                elif role == "assistant":
                    lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {t.get('question', '')}\nAssistant: {t.get('answer', '')}")
        return "\n".join(lines)


def get_orchestrator(embedder: OllamaEmbedder, graph_rag=None) -> Orchestrator:
    return Orchestrator(embedder=embedder, graph_rag=graph_rag)
