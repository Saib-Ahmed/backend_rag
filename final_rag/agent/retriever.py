"""
final_rag/agent/retriever.py

Hybrid retriever for the RAG pipeline.
Qwen3 Embedder (dense + BM25 sparse) + Qwen3 Reranker via Sentence Transformers CrossEncoder.
Features 4-strategy routing (Active, Compare, Single with Filter Relaxation, Broad with Diversity Cap),
explicit CUDA device pinning, Sigmoid probability normalization, and an 8-factor score adjustment matrix.
"""

from __future__ import annotations
import logging
import math
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import CrossEncoder

try:
    import final_rag.config as config
except ImportError:
    from .. import config

from .models import CleanedQuery, ComparisonArm, RetrievedChunk
from ..ingestion.embedder import OllamaEmbedder
from ..qdrant_storage.store import QdrantManager

logger = logging.getLogger("agent.retriever")


class Retriever:
    def __init__(self, embedder: OllamaEmbedder, db: QdrantManager):
        self.embedder = embedder
        self.db       = db

        device = getattr(config, "RERANKER_DEVICE", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_name = getattr(config, "RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        self.rerank_model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        logger.info(
            "Retriever initialized | embedder=OllamaEmbedder | reranker=%s | device=%s",
            model_name, device,
        )

    def retrieve(self, cleaned: CleanedQuery, domain: str = "") -> list[RetrievedChunk]:
        start = time.perf_counter()

        if cleaned.scope == "active" and cleaned.active_doc_filename:
            chunks = self._retrieve_active(cleaned, domain=domain)
        elif cleaned.scope == "active_compare" and (cleaned.active_doc_filenames or len(cleaned.comparison_arms) > 1):
            chunks = self._retrieve_active_compare(cleaned, domain=domain)
        elif cleaned.scope == "single":
            chunks = self._retrieve_single(cleaned, domain=domain)
        elif cleaned.scope == "few":
            chunks = self._retrieve_few(cleaned, domain=domain)
        else:
            chunks = self._retrieve_broad(cleaned, domain=domain)

        elapsed = round(time.perf_counter() - start, 3)
        logger.info(
            "Retrieval complete | scope=%s | chunks=%d | time=%.3fs",
            cleaned.scope, len(chunks), elapsed,
        )
        return chunks

    def _retrieve_active(self, cleaned: CleanedQuery, domain: str) -> list[RetrievedChunk]:
        queries = self._build_query_set(cleaned)
        active_filter = {"source_file": cleaned.active_doc_filename}

        candidates, hit_counts = self._embed_and_search(queries, active_filter, domain=domain)
        if not candidates:
            logger.warning("Active document search returned 0 chunks | %s", cleaned.active_doc_filename)
            return []

        reranked = self._rerank(cleaned.primary_query, candidates, pool_cap=len(candidates))
        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        return self._threshold_and_cap(boosted, top_k=getattr(config, "PER_QUERY_TOP_K", 20))

    def _retrieve_active_compare(self, cleaned: CleanedQuery, domain: str) -> list[RetrievedChunk]:
        filenames = cleaned.active_doc_filenames
        if not filenames and cleaned.comparison_arms:
            filenames = [arm.label for arm in cleaned.comparison_arms]

        all_chunks: list[RetrievedChunk] = []
        queries = self._build_query_set(cleaned)
        per_arm_top_k = max(2, getattr(config, "TOP_K_SEARCH", 30) // max(len(filenames), 1))

        for filename in filenames:
            arm_filter = {"source_file": filename}
            candidates, hit_counts = self._embed_and_search(queries, arm_filter, domain=domain)
            if candidates:
                reranked = self._rerank(cleaned.primary_query, candidates, pool_cap=len(candidates))
                boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
                passed   = self._threshold_and_cap(boosted, top_k=per_arm_top_k)
                for c in passed:
                    c.arm_label = filename
                all_chunks.extend(passed)

        return all_chunks

    def _retrieve_single(self, cleaned: CleanedQuery, domain: str) -> list[RetrievedChunk]:
        queries = self._build_query_set(cleaned)
        filter_dict = self._build_filter(cleaned.filter_hints)

        # Tier 1: strict filter
        candidates, hit_counts = self._embed_and_search(queries, filter_dict, domain=domain)

        # Tier 2: relaxed filter if too few candidates
        if len(candidates) < getattr(config, "MIN_RESULTS_THRESHOLD", 3) and filter_dict:
            relaxed = self._relax_filter(filter_dict)
            if relaxed != filter_dict:
                logger.info("Tier 1 yielded %d chunks -> relaxing filter (Tier 2)", len(candidates))
                candidates, hit_counts = self._embed_and_search(queries, relaxed, domain=domain)

        # Tier 3: no filter fallback
        if len(candidates) < getattr(config, "MIN_RESULTS_THRESHOLD", 3):
            logger.info("Filter yielded %d chunks -> falling back to unfiltered search (Tier 3)", len(candidates))
            candidates, hit_counts = self._embed_and_search(queries, None, domain=domain)

        reranked = self._rerank(cleaned.primary_query, candidates, pool_cap=len(candidates))
        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        return self._threshold_and_cap(boosted, top_k=getattr(config, "PER_QUERY_TOP_K", 20))

    def _retrieve_few(self, cleaned: CleanedQuery, domain: str) -> list[RetrievedChunk]:
        arms = cleaned.comparison_arms[:4]
        if not arms:
            return self._retrieve_broad(cleaned, domain=domain)

        per_arm_top_k = max(2, getattr(config, "TOP_K_SEARCH", 30) // len(arms))
        all_chunks: list[RetrievedChunk] = []

        for arm in arms:
            arm_query   = arm.query or cleaned.primary_query
            raw_sqs     = [sq.query if hasattr(sq, "query") else str(sq) for sq in cleaned.subqueries[:2]]
            queries     = [arm_query] + raw_sqs
            filter_dict = self._build_filter_from_arm(arm)

            candidates, hit_counts = self._embed_and_search(queries, filter_dict, domain=domain)
            if len(candidates) < getattr(config, "MIN_RESULTS_THRESHOLD", 3):
                candidates, hit_counts = self._embed_and_search(queries, None, domain=domain)

            reranked = self._rerank(arm_query, candidates, pool_cap=len(candidates))
            boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
            passed   = self._threshold_and_cap(boosted, top_k=per_arm_top_k)
            for c in passed:
                c.arm_label = arm.label
            all_chunks.extend(passed)

        return all_chunks

    def _retrieve_broad(self, cleaned: CleanedQuery, domain: str) -> list[RetrievedChunk]:
        queries = self._build_query_set(cleaned)
        candidates, hit_counts = self._embed_and_search(queries, None, domain=domain)

        reranked = self._rerank(
            cleaned.primary_query,
            candidates,
            pool_cap=getattr(config, "RERANK_POOL_CAP", 180),
        )
        boosted = self._apply_boosts(reranked, cleaned, hit_counts)
        capped  = self._diversity_cap(boosted)
        return self._threshold_and_cap(capped, top_k=getattr(config, "TOP_K_SEARCH", 30))

    def _build_query_set(self, cleaned: CleanedQuery) -> list[str]:
        max_sub = getattr(config, "MAX_SUBQUERIES", 6)
        raw_sqs = [sq.query if hasattr(sq, "query") else str(sq) for sq in cleaned.subqueries[:max_sub]]
        return [cleaned.primary_query] + raw_sqs

    def _embed_and_search(
        self,
        queries: list[str],
        filter_dict: dict | None,
        domain: str,
    ) -> tuple[list[dict], dict[str, int]]:
        all_results: list[dict] = []
        hit_counts: dict[str, int] = defaultdict(int)

        for q in queries:
            dense, sparse = self._embed(q)
            if dense is None:
                continue
            results = self._search(dense, sparse, filter_dict, domain=domain)
            for r in results:
                key = f"{r.get('source_file')}::{r.get('chunk_index')}"
                hit_counts[key] += 1
            all_results.extend(results)

        deduped = self._deduplicate(all_results)
        return deduped, dict(hit_counts)

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for r in results:
            key = f"{r.get('source_file')}::{r.get('chunk_index')}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        pool_cap: int = 180,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pool = candidates[:pool_cap]
        instruction = getattr(config, "RERANKER_INSTRUCTION", "")
        formatted_query = f"{instruction} Query: {query}" if instruction else query
        pairs = [[formatted_query, c["text"]] for c in pool]

        try:
            batch_size = getattr(config, "RERANKER_BATCH_SIZE", 16)
            scores = self.rerank_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.error("Reranking failed: %s | falling back to Qdrant scores", e)
            return [self._to_chunk(c, c.get("score", 0.0)) for c in pool]

        scored_chunks = []
        for c, score in zip(pool, scores):
            # Sigmoid normalization: 1 / (1 + exp(-score))
            score_float = float(score)
            prob_score = 1.0 / (1.0 + math.exp(-score_float))
            scored_chunks.append(self._to_chunk(c, prob_score))

        scored_chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return scored_chunks

    def _apply_boosts(
        self,
        chunks: list[RetrievedChunk],
        cleaned: CleanedQuery,
        hit_counts: dict[str, int],
    ) -> list[RetrievedChunk]:
        query_text = cleaned.primary_query.lower()
        hints = cleaned.filter_hints or {}
        doc_year = str(hints.get("doc_year", ""))
        keywords = [k.lower() for k in hints.get("keywords", [])]

        for chunk in chunks:
            boost = 0.0

            if chunk.is_table:
                boost += getattr(config, "TABLE_BOOST", 0.03)

            if doc_year and (chunk.doc_year == doc_year or doc_year in chunk.chunk_years):
                boost += getattr(config, "YEAR_BOOST", 0.04)

            if chunk.heading and chunk.heading.lower() in query_text:
                boost += getattr(config, "HEADING_BOOST", 0.04)

            for kw in keywords:
                if kw in chunk.text.lower():
                    boost += getattr(config, "KEYWORD_BOOST", 0.02)

            for ent in chunk.entities:
                if ent.lower() in query_text:
                    boost += getattr(config, "ENTITY_BOOST", 0.04)
                    break

            key = f"{chunk.source_file}::{chunk.chunk_index}"
            hit_count = hit_counts.get(key, 1)
            if hit_count > 1:
                boost += getattr(config, "HIT_COUNT_BOOST", 0.01) * (hit_count - 1)

            chunk.rerank_score += boost
            chunk.is_weak_match = chunk.rerank_score < getattr(config, "CONFIDENCE_THRESHOLD", 0.22)

        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks

    def _diversity_cap(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        doc_buckets: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            key = chunk.doc_id if chunk.doc_id else chunk.source_file
            doc_buckets[key].append(chunk)

        result    = []
        doc_count = 0
        max_docs   = getattr(config, "MAX_DOCS_BROAD", 7)
        max_chunks = getattr(config, "MAX_CHUNKS_PER_DOC", 8)

        for doc_key, doc_chunks in doc_buckets.items():
            if doc_count >= max_docs:
                break
            result.extend(doc_chunks[:max_chunks])
            doc_count += 1

        return result

    def _threshold_and_cap(self, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        thresh = getattr(config, "CONFIDENCE_THRESHOLD", 0.22)
        filtered = [c for c in chunks if c.rerank_score >= thresh]
        if not filtered:
            return chunks[:min(top_k, 3)]  # Return top fallback candidates if all below threshold
        return filtered[:top_k]

    def _build_filter(self, hints: dict) -> dict:
        f = {}
        if hints.get("doc_year"):
            f["doc_year__or__chunk_years"] = hints["doc_year"]

        fn_tokens = []
        if hints.get("filename_tokens"):
            for t in hints["filename_tokens"]:
                fn_tokens.extend(t.lower().split())

        keywords = [k.lower() for k in hints.get("keywords", [])]
        identity_terms = fn_tokens + keywords
        if identity_terms:
            f["entities__or__filename_tokens"] = identity_terms

        return f

    def _build_filter_from_arm(self, arm: ComparisonArm) -> dict:
        f = {}
        hints = arm.filter_hints or {}
        if hints.get("doc_year"):
            f["doc_year__or__chunk_years"] = hints["doc_year"]
        if hints.get("filename_tokens"):
            tokens = []
            for t in hints["filename_tokens"]:
                tokens.extend(t.lower().split())
            f["entities__or__filename_tokens"] = tokens
        return f

    def _relax_filter(self, filter_dict: dict) -> dict:
        relaxed = dict(filter_dict)
        if "entities__or__filename_tokens" in relaxed:
            del relaxed["entities__or__filename_tokens"]
            return relaxed
        return relaxed

    def _embed(self, query: str):
        try:
            dense, sparse = self.embedder.embed_query(query)
            return dense, sparse
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            return None, None

    def _search(self, dense, sparse, filter_dict: dict | None, domain: str) -> list[dict]:
        try:
            return self.db.search_hybrid(
                query_dense  = dense,
                query_sparse = sparse,
                domain       = domain,
                filter_dict  = filter_dict,
                top_k        = getattr(config, "TOP_K_SEARCH", 30),
            )
        except Exception as e:
            logger.error("Qdrant search failed: %s", e)
            return []

    def format_for_llm(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            label = chunk.arm_label or (chunk.heading or "General")
            header = f"--- Context {i} {chunk.source_tag} | {label} ---"
            parts.append(f"{header}\n{chunk.text}\n")

        return "\n".join(parts)

    def _to_chunk(self, result: dict, rerank_score: float) -> RetrievedChunk:
        source_file = result.get("source_file", "")
        page_label  = result.get("page_label", "")
        page_no     = result.get("page_no", 0)

        if page_label:
            source_tag = f"[{source_file}, Page {page_label}]"
        elif page_no > 0:
            source_tag = f"[{source_file}, Page {page_no}]"
        else:
            source_tag = f"[{source_file}]"

        return RetrievedChunk(
            text                  = result.get("text", ""),
            source_file           = source_file,
            page_no               = page_no,
            page_label            = page_label,
            chunk_index           = result.get("chunk_index", 0),
            heading               = result.get("heading", ""),
            is_table              = result.get("is_table", False),
            doc_year              = result.get("doc_year", ""),
            doc_lang              = result.get("doc_lang", ""),
            doc_id                = result.get("doc_id", ""),
            chunk_years           = result.get("chunk_years", []),
            entities              = result.get("entities", []),
            token_count           = result.get("token_count", 0),
            qdrant_score          = result.get("score", 0.0),
            rerank_score          = rerank_score,
            is_weak_match         = rerank_score < getattr(config, "CONFIDENCE_THRESHOLD", 0.22),
            source_tag            = source_tag,
            arm_label             = "",
        )


def get_retriever(embedder: OllamaEmbedder) -> Retriever:
    return Retriever(embedder=embedder, db=embedder.db)