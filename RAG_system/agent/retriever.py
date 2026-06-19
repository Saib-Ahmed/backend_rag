"""
agent/retriever.py
Hybrid retriever for RAG pipeline.
OllamaEmbedder (bge-m3 dense + BM25 sparse) + CrossEncoder Reranker.

Strategy branches on CleanedQuery signals:
    scope=single → filtered search + filter relaxation fallback
    scope=few    → per-arm filtered search (max 4 arms) + arm tagging
    scope=broad  → unfiltered diversity-capped search
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

import math
from collections import defaultdict
from functools import lru_cache
import torch
import ollama
from sentence_transformers import CrossEncoder

from agent.models import CleanedQuery, ComparisonArm, RetrievedChunk
from new_ingestion.embedder import OllamaEmbedder
from qdrant_storage.store import QdrantManager
import config

logger = logging.getLogger("agent.retriever")

# ── Per-type reranker instructions ─────────────────────────────────────
RERANKER_INSTRUCTIONS = {
    "direct": "Retrieve passages that directly answer the given query with specific facts, numbers, or definitions.",
    "compare": "Retrieve passages that compare, contrast, or provide parallel information about the entities mentioned in the query.",
    "synthesize": "Retrieve passages that provide broad coverage and diverse perspectives relevant to the given query.",
}

# ── Query embedding cache ──────────────────────────────────────────────
_EMBED_CACHE: dict[str, tuple[list[float], dict[int, float]]] = {}
_EMBED_CACHE_MAX = 128

def _cache_key(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()

class Retriever:
    def __init__(self, embedder: OllamaEmbedder, db: QdrantManager, graph_rag=None):
        self.embedder     = embedder
        self.db           = db
        self.graph_rag    = graph_rag
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rerank_model = CrossEncoder(config.RERANKER_MODEL, device=device, trust_remote_code=True)
        self.last_query_trace = {"stages": []}
        self.last_relationships = []
        logger.info(
            "Retriever initialized | embedder=OllamaEmbedder | reranker=%s on %s | graph_rag=%s",
            config.RERANKER_MODEL,
            device,
            graph_rag is not None,
        )

    # ── Public entry point ─────────────────────────────────────────────
    def retrieve(self, cleaned: CleanedQuery, enable_graph_rag: bool = True) -> list[RetrievedChunk]:
        scope = cleaned.target_scope  # "single" / "few" / "broad"

        logger.info(
            "Retrieval start | scope=%s | structure=%s | specificity=%s | graph_rag=%s",
            scope, cleaned.answer_structure, cleaned.specificity, enable_graph_rag,
        )

        self.last_query_trace = {"stages": []}
        self.last_relationships = []
        mode = "hybrid" if scope == "broad" else "local"

        if scope == "single":
            chunks = self._retrieve_single(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)
        elif scope == "few":
            chunks = self._retrieve_few(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)
        else:
            chunks = self._retrieve_broad(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)

        if not chunks:
            logger.warning("Retrieval returned no chunks | scope=%s", scope)
            return []

        logger.info(
            "Retrieval done | chunks=%d | sources=%d",
            len(chunks), len({c.source_file for c in chunks}),
        )
        return chunks

    # ── Async Public entry point ───────────────────────────────────────
    async def async_retrieve(self, cleaned: CleanedQuery, enable_graph_rag: bool = True) -> list[RetrievedChunk]:
        scope = cleaned.target_scope  # "single" / "few" / "broad"

        logger.info(
            "Async Retrieval start | scope=%s | structure=%s | specificity=%s | graph_rag=%s",
            scope, cleaned.answer_structure, cleaned.specificity, enable_graph_rag,
        )

        self.last_query_trace = {"stages": []}
        self.last_relationships = []
        mode = "hybrid" if scope == "broad" else "local"

        if scope == "single":
            chunks = await self._async_retrieve_single(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)
        elif scope == "few":
            chunks = await self._async_retrieve_few(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)
        else:
            chunks = await self._async_retrieve_broad(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)

        if not chunks:
            logger.warning("Async Retrieval returned no chunks | scope=%s", scope)
            return []

        logger.info(
            "Async Retrieval done | chunks=%d | sources=%d",
            len(chunks), len({c.source_file for c in chunks}),
        )
        return chunks

    async def _async_retrieve_single(self, cleaned: CleanedQuery, enable_graph_rag: bool = True, mode: str = "local") -> list[RetrievedChunk]:
        hints       = cleaned.filter_hints
        query_set   = self._build_query_set(cleaned)
        filter_dict = self._build_filter(hints)

        logger.info("Async Single scope | filter=%s", filter_dict)

        raw_pool, hit_counts = await self._async_embed_and_search(query_set, filter_dict, enable_graph_rag=enable_graph_rag, mode=mode)

        if len(raw_pool) < config.MIN_RESULTS_THRESHOLD and filter_dict:
            logger.warning("Strict filter returned %d results — relaxing filter asynchronously", len(raw_pool))
            relaxed = self._relax_filter(filter_dict)
            raw_pool, hit_counts = await self._async_embed_and_search(query_set, relaxed, enable_graph_rag=enable_graph_rag, mode=mode)

            if len(raw_pool) < config.MIN_RESULTS_THRESHOLD:
                logger.warning("Relaxed filter still thin — falling back to no filter asynchronously")
                raw_pool, hit_counts = await self._async_embed_and_search(query_set, None, enable_graph_rag=enable_graph_rag, mode=mode)

        self._capture_trace_stage("Qdrant Retrieval", raw_pool)

        deduped  = self._deduplicate(raw_pool)
        self._capture_trace_stage("Deduplication", deduped)

        reranked = await asyncio.to_thread(self._rerank, cleaned.improved_query, deduped, cleaned.answer_structure)
        self._capture_trace_stage("CrossEncoder Rerank", reranked)

        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        self._capture_trace_stage("Metadata Boost", boosted)

        capped = self._threshold_and_cap(boosted, config.PER_QUERY_TOP_K)

        capped = await self._async_expand_neighbors(capped)
        self._capture_trace_stage("Final Top-K + Neighbors", capped)

        return capped

    async def _async_retrieve_few(self, cleaned: CleanedQuery, enable_graph_rag: bool = True, mode: str = "local") -> list[RetrievedChunk]:
        arms      = cleaned.comparison_arms[:4]
        query_set = self._build_query_set(cleaned)

        if not arms:
            logger.warning("Few scope but no comparison_arms — falling back to single asynchronously")
            return await self._async_retrieve_single(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)

        async def process_arm(arm):
            filter_dict = self._build_filter_from_arm(arm)
            logger.info("Arm '%s' | filter=%s", arm.label, filter_dict)

            raw_pool, hit_counts = await self._async_embed_and_search(query_set, filter_dict, enable_graph_rag=enable_graph_rag, mode=mode)

            if len(raw_pool) < config.MIN_RESULTS_THRESHOLD and filter_dict:
                logger.warning("Arm '%s' thin — relaxing filter", arm.label)
                relaxed  = self._relax_filter(filter_dict)
                raw_pool, hit_counts = await self._async_embed_and_search(query_set, relaxed, enable_graph_rag=enable_graph_rag, mode=mode)

                if len(raw_pool) < config.MIN_RESULTS_THRESHOLD:
                    logger.warning("Arm '%s' still thin — falling back to no filter", arm.label)
                    raw_pool, hit_counts = await self._async_embed_and_search(query_set, None, enable_graph_rag=enable_graph_rag, mode=mode)

            deduped  = self._deduplicate(raw_pool)
            reranked = await asyncio.to_thread(self._rerank, cleaned.improved_query, deduped, cleaned.answer_structure)
            boosted  = self._apply_boosts(reranked, cleaned, hit_counts)

            slots = max(1, config.PER_QUERY_TOP_K // len(arms))
            arm_selected_chunks = []
            for chunk in boosted[:slots]:
                chunk.arm_label = arm.label
                arm_selected_chunks.append(chunk)

            logger.info("Arm '%s' | chunks=%d", arm.label, len(arm_selected_chunks))
            return arm_selected_chunks

        results = await asyncio.gather(*(process_arm(arm) for arm in arms))
        all_chunks = []
        for r in results:
            all_chunks.extend(r)

        all_chunks.sort(key=lambda c: (c.arm_label, -c.rerank_score))
        self._capture_trace_stage("Final Arm Selection", all_chunks)
        return all_chunks

    async def _async_retrieve_broad(self, cleaned: CleanedQuery, enable_graph_rag: bool = True, mode: str = "hybrid") -> list[RetrievedChunk]:
        query_set = self._build_query_set(cleaned)

        raw_pool, hit_counts = await self._async_embed_and_search(query_set, filter_dict=None, enable_graph_rag=enable_graph_rag, mode=mode)
        self._capture_trace_stage("Qdrant Retrieval", raw_pool)

        deduped  = self._deduplicate(raw_pool)
        self._capture_trace_stage("Deduplication", deduped)

        reranked = await asyncio.to_thread(self._rerank, cleaned.improved_query, deduped, cleaned.answer_structure)
        self._capture_trace_stage("CrossEncoder Rerank", reranked)

        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        self._capture_trace_stage("Metadata Boost", boosted)

        capped = self._diversity_cap(boosted)

        capped = await self._async_expand_neighbors(capped)
        self._capture_trace_stage("Diversity Cap + Neighbors", capped)

        logger.info(
            "Async Broad scope | specificity=%s | after_cap=%d",
            cleaned.specificity, len(capped),
        )
        return capped


    def _capture_trace_stage(self, stage_name: str, chunks_list: list) -> None:
        stage_data = []
        for c in chunks_list:
            if isinstance(c, dict):
                stage_data.append({
                    "id": str(c.get("chunk_id", c.get("chunk_index", 0))),
                    "weight": c.get("score", 0.0),
                    "text": c.get("text", ""),
                    "source": c.get("source_file", ""),
                    "is_weak": False,
                    "graph_score": c.get("graph_score", 0.0),
                    "qdrant_score": c.get("score", 0.0),
                    "chunk_index": c.get("chunk_index", 0)
                })
            else:
                stage_data.append({
                    "id": str(c.chunk_id if getattr(c, "chunk_id", None) else getattr(c, "chunk_index", 0)),
                    "weight": getattr(c, "rerank_score", getattr(c, "qdrant_score", 0.0)),
                    "text": getattr(c, "text", ""),
                    "source": getattr(c, "source_file", ""),
                    "is_weak": getattr(c, "is_weak_match", False),
                    "graph_score": getattr(c, "graph_score", 0.0),
                    "qdrant_score": getattr(c, "qdrant_score", 0.0),
                    "chunk_index": getattr(c, "chunk_index", 0)
                })
        self.last_query_trace["stages"].append({
            "name": stage_name,
            "chunks": stage_data
        })

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 1 — SINGLE
    # ══════════════════════════════════════════════════════════════════
    def _retrieve_single(self, cleaned: CleanedQuery, enable_graph_rag: bool = True, mode: str = "local") -> list[RetrievedChunk]:
        hints       = cleaned.filter_hints
        query_set   = self._build_query_set(cleaned)
        filter_dict = self._build_filter(hints)

        logger.info("Single scope | filter=%s", filter_dict)

        raw_pool, hit_counts = self._embed_and_search(query_set, filter_dict, enable_graph_rag=enable_graph_rag, mode=mode)

        if len(raw_pool) < config.MIN_RESULTS_THRESHOLD and filter_dict:
            logger.warning("Strict filter returned %d results — relaxing filter", len(raw_pool))
            relaxed = self._relax_filter(filter_dict)
            raw_pool, hit_counts = self._embed_and_search(query_set, relaxed, enable_graph_rag=enable_graph_rag, mode=mode)

            if len(raw_pool) < config.MIN_RESULTS_THRESHOLD:
                logger.warning("Relaxed filter still thin — falling back to no filter")
                raw_pool, hit_counts = self._embed_and_search(query_set, None, enable_graph_rag=enable_graph_rag, mode=mode)

        self._capture_trace_stage("Qdrant Retrieval", raw_pool)

        deduped  = self._deduplicate(raw_pool)
        self._capture_trace_stage("Deduplication", deduped)

        reranked = self._rerank(cleaned.improved_query, deduped, answer_structure=cleaned.answer_structure)
        self._capture_trace_stage("CrossEncoder Rerank", reranked)

        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        self._capture_trace_stage("Metadata Boost", boosted)

        capped = self._threshold_and_cap(boosted, config.PER_QUERY_TOP_K)

        # Chunk neighbor expansion: fetch adjacent chunks for top results
        capped = self._expand_neighbors(capped)
        self._capture_trace_stage("Final Top-K + Neighbors", capped)

        return capped

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 2 — FEW (comparison / multi-doc)
    # ══════════════════════════════════════════════════════════════════
    def _retrieve_few(self, cleaned: CleanedQuery, enable_graph_rag: bool = True, mode: str = "local") -> list[RetrievedChunk]:
        arms      = cleaned.comparison_arms[:4]
        query_set = self._build_query_set(cleaned)

        if not arms:
            logger.warning("Few scope but no comparison_arms — falling back to single")
            return self._retrieve_single(cleaned, enable_graph_rag=enable_graph_rag, mode=mode)

        all_chunks: list[RetrievedChunk] = []

        for arm in arms:
            filter_dict = self._build_filter_from_arm(arm)
            logger.info("Arm '%s' | filter=%s", arm.label, filter_dict)

            raw_pool, hit_counts = self._embed_and_search(query_set, filter_dict, enable_graph_rag=enable_graph_rag, mode=mode)

            if len(raw_pool) < config.MIN_RESULTS_THRESHOLD and filter_dict:
                logger.warning("Arm '%s' thin — relaxing filter", arm.label)
                relaxed  = self._relax_filter(filter_dict)
                raw_pool, hit_counts = self._embed_and_search(query_set, relaxed, enable_graph_rag=enable_graph_rag, mode=mode)

                if len(raw_pool) < config.MIN_RESULTS_THRESHOLD:
                    logger.warning("Arm '%s' still thin — falling back to no filter", arm.label)
                    raw_pool, hit_counts = self._embed_and_search(query_set, None, enable_graph_rag=enable_graph_rag, mode=mode)

            deduped  = self._deduplicate(raw_pool)
            reranked = self._rerank(cleaned.improved_query, deduped, answer_structure=cleaned.answer_structure)
            boosted  = self._apply_boosts(reranked, cleaned, hit_counts)

            slots = max(1, config.PER_QUERY_TOP_K // len(arms))
            for chunk in boosted[:slots]:
                chunk.arm_label = arm.label
                all_chunks.append(chunk)

            logger.info(
                "Arm '%s' | chunks=%d", arm.label, len(boosted[:slots])
            )

        all_chunks.sort(key=lambda c: (c.arm_label, -c.rerank_score))
        self._capture_trace_stage("Final Arm Selection", all_chunks)
        return all_chunks

    # ══════════════════════════════════════════════════════════════════
    # STRATEGY 3 — BROAD
    # ══════════════════════════════════════════════════════════════════
    def _retrieve_broad(self, cleaned: CleanedQuery, enable_graph_rag: bool = True, mode: str = "hybrid") -> list[RetrievedChunk]:
        query_set = self._build_query_set(cleaned)

        raw_pool, hit_counts = self._embed_and_search(query_set, filter_dict=None, enable_graph_rag=enable_graph_rag, mode=mode)
        self._capture_trace_stage("Qdrant Retrieval", raw_pool)

        deduped  = self._deduplicate(raw_pool)
        self._capture_trace_stage("Deduplication", deduped)

        reranked = self._rerank(cleaned.improved_query, deduped, answer_structure=cleaned.answer_structure)
        self._capture_trace_stage("CrossEncoder Rerank", reranked)

        boosted  = self._apply_boosts(reranked, cleaned, hit_counts)
        self._capture_trace_stage("Metadata Boost", boosted)

        capped = self._diversity_cap(boosted)

        # Chunk neighbor expansion for broad scope too
        capped = self._expand_neighbors(capped)
        self._capture_trace_stage("Diversity Cap + Neighbors", capped)

        logger.info(
            "Broad scope | specificity=%s | after_cap=%d",
            cleaned.specificity, len(capped),
        )
        return capped

    # ══════════════════════════════════════════════════════════════════
    # CORE — embed + search across all subqueries
    # ══════════════════════════════════════════════════════════════════
    def _build_query_set(self, cleaned: CleanedQuery) -> list[str]:
        queries = [cleaned.improved_query]
        if getattr(config, "ENABLE_MULTIQUERY", True):
            for sq in cleaned.subqueries:
                if sq.query not in queries:
                    queries.append(sq.query)
        return queries

    def _embed_and_search(
        self,
        queries:     list[str],
        filter_dict: dict | None,
        enable_graph_rag: bool = True,
        mode: str = "local",
    ) -> tuple[list[dict], dict[tuple, int]]:
        raw_pool:   list[dict]       = []
        hit_counts: dict[tuple, int] = defaultdict(int)

        # Batch embed all queries at once (cached individually)
        embeddings = []
        for query in queries:
            key = _cache_key(query)
            if key in _EMBED_CACHE:
                embeddings.append(_EMBED_CACHE[key])
            else:
                dense, sparse = self._embed(query)
                if dense is not None:
                    _EMBED_CACHE[key] = (dense, sparse)
                    # Evict oldest if cache too large
                    if len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
                        oldest_key = next(iter(_EMBED_CACHE))
                        del _EMBED_CACHE[oldest_key]
                    embeddings.append((dense, sparse))
                else:
                    embeddings.append(None)

        # Search with all embeddings
        for i, query in enumerate(queries):
            emb = embeddings[i]
            if emb is None:
                continue
            dense, sparse = emb

            results = self._search(dense, sparse, filter_dict)
            for r in results:
                key = (r.get("source_file", ""), r.get("chunk_index", 0))
                hit_counts[key] += 1
                raw_pool.append(r)

            logger.info(
                "Searched | query='%s...' | filter=%s | results=%d",
                query[:50], filter_dict, len(results),
            )

        if enable_graph_rag:
            raw_pool = self._merge_graph_candidates(raw_pool, queries[0], mode=mode)

        return raw_pool, hit_counts

    def _merge_graph_candidates(self, raw_pool: list[dict], query: str, mode: str = "local") -> list[dict]:
        if not self.graph_rag or not self.graph_rag.available:
            return raw_pool

        pool_by_key = {
            (r.get("source_file", ""), r.get("chunk_index", 0)): r
            for r in raw_pool
        }

        graph_candidates, relationships = self.graph_rag.retrieve_candidates(
            query,
            mode=mode,
            llm_invoke=self._graph_llm_invoke,
            embedding_fn=lambda q: self.embedder.embed_query(q)[0]
        )
        if relationships:
            for rel in relationships:
                if rel not in self.last_relationships:
                    self.last_relationships.append(rel)

        if not graph_candidates:
            return raw_pool

        added = 0
        for candidate in graph_candidates:
            matched_key = None
            for key, r in pool_by_key.items():
                if r.get("source_file") == candidate.source and str(r.get("chunk_id")) == str(candidate.chunk_id):
                    matched_key = key
                    break

            graph_boost = max(0.0, min(1.0, float(candidate.graph_score)))
            
            if matched_key:
                r = pool_by_key[matched_key]
                r["graph_score"] = max(r.get("graph_score", 0.0), graph_boost)
                r["graph_hops"] = int(candidate.graph_hops)
                r["graph_path"] = candidate.graph_path
                r["retrieval_channel"] = "hybrid_graph"
            else:
                r = self._fetch_chunk_by_source_id(candidate.source, candidate.chunk_id)
                if r:
                    r["graph_score"] = graph_boost
                    r["graph_hops"] = int(candidate.graph_hops)
                    r["graph_path"] = candidate.graph_path
                    r["retrieval_channel"] = "graph"
                    pool_by_key[(r.get("source_file", ""), r.get("chunk_index", 0))] = r
                    added += 1

        logger.info("[GraphRAG] Merged graph candidates: added=%d", added)
        return list(pool_by_key.values())

    def _graph_llm_invoke(self, prompt: str, timeout_sec: float = config.GRAPH_LLM_TIMEOUT_SEC, retries: int = 1) -> str:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        try:
            response = client.generate(
                model=config.GENERATOR_MODEL,
                prompt=prompt,
                think=False,
                options={"temperature": config.GRAPH_LLM_TEMPERATURE},
            )
            return response.get("response", "")
        except Exception as e:
            logger.error("[GraphRAG LLM] Generation failed: %s", e)
            return ""

    def _fetch_chunk_by_source_id(self, source: str, chunk_id: str) -> dict | None:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            client = self.db.get_client()
            points, _ = client.scroll(
                collection_name=self.db.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="source_file", match=MatchValue(value=str(source))),
                        FieldCondition(key="chunk_id", match=MatchValue(value=str(chunk_id))),
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                points, _ = client.scroll(
                    collection_name=self.db.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="source", match=MatchValue(value=str(source))),
                            FieldCondition(key="chunk_id", match=MatchValue(value=str(chunk_id))),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
            if not points:
                return None
            
            payload = points[0].payload or {}
            return {
                "score": 0.0,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", payload.get("source", "")),
                "page_no": payload.get("page_no", 0),
                "page_range": payload.get("page_range", [0, 0]),
                "section": payload.get("section", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "is_table": payload.get("is_table", False),
                "doc_year": payload.get("doc_year", ""),
                "chunk_year": payload.get("chunk_year", []),
                "token_count": payload.get("token_count", 0),
                "doc_id": payload.get("doc_id", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "filename_tokens": payload.get("filename_tokens", []),
                "has_tables": payload.get("has_tables", False),
                "summary": payload.get("summary", ""),
                "keywords": payload.get("keywords", []),
                "page_label": payload.get("page_label", ""),
            }
        except Exception as e:
            logger.error("Failed to fetch graph chunk %s:%s | %s", source, chunk_id, e)
            return None

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        seen: dict[tuple, dict] = {}
        for r in results:
            key = (r.get("source_file", ""), r.get("chunk_index", 0))
            if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
                seen[key] = r
        return list(seen.values())

    # ══════════════════════════════════════════════════════════════════
    # RERANK — CrossEncoder
    # ══════════════════════════════════════════════════════════════════
    def _rerank(
        self, improved_query: str, results: list[dict],
        answer_structure: str = "direct",
    ) -> list[RetrievedChunk]:
        if not results:
            return []

        # Use per-type reranker instruction
        instruction = RERANKER_INSTRUCTIONS.get(
            answer_structure,
            config.RERANKER_INSTRUCTION,
        )
        try:
            instruct_query = (
                f"Instruct: {instruction}\n"
                f"Query: {improved_query}"
            )
            pairs  = [[instruct_query, r.get("text", "")] for r in results]
            scores = self.rerank_model.predict(pairs)

            scored = []
            for raw_score, result in zip(scores, results):
                sigmoid_score = 1 / (1 + math.exp(-float(raw_score)))
                scored.append((sigmoid_score, result))

            scored.sort(key=lambda x: x[0], reverse=True)

            relevant = [(s, r) for s, r in scored if s >= config.CONFIDENCE_THRESHOLD]
            logger.info(
                "Rerank done | top=%.4f | relevant=%d | total=%d | instruction=%s",
                scored[0][0] if scored else 0.0,
                len(relevant), len(scored), answer_structure,
            )

            return [
                self._to_chunk(result, rerank_score=score)
                for score, result in scored
            ]

        except Exception as e:
            logger.error("Reranking failed | %s — falling back to Qdrant score", e)
            results.sort(key=lambda r: r.get("score", 0), reverse=True)
            return [
                self._to_chunk(r, rerank_score=r.get("score", 0.0))
                for r in results
            ]

    # ══════════════════════════════════════════════════════════════════
    # BOOSTS
    # ══════════════════════════════════════════════════════════════════
    def _apply_boosts(
        self,
        chunks:     list[RetrievedChunk],
        cleaned:    CleanedQuery,
        hit_counts: dict[tuple, int],
    ) -> list[RetrievedChunk]:
        hints        = cleaned.filter_hints
        
        raw_year = hints.get("doc_year")
        year = str(raw_year[0]).lower() if isinstance(raw_year, list) and raw_year else str(raw_year or "").lower()
        
        raw_sec = hints.get("section")
        section = str(raw_sec[0]).lower() if isinstance(raw_sec, list) and raw_sec else str(raw_sec or "").lower()
        keywords     = [k.lower() for k in hints.get("keywords", [])]
        fn_tokens    = [t.lower() for t in hints.get("filename_tokens", [])]

        for chunk in chunks:
            boost = 0.0
            key   = (chunk.source_file, chunk.chunk_index)

            if getattr(config, "ENABLE_METADATA_BOOSTING", True):
                if chunk.is_table:
                    boost += config.TABLE_BOOST

                if year and chunk.chunk_year:
                    if year in chunk.chunk_year:
                        boost += config.YEAR_BOOST

                if section and chunk.section:
                    if section in chunk.section.lower():
                        boost += config.SECTION_BOOST

                if keywords:
                    chunk_lower = chunk.text.lower()
                    if any(k in chunk_lower for k in keywords):
                        boost += config.KEYWORD_BOOST
                    else:
                        if cleaned.target_scope in ("single", "few"):
                            boost += config.ENTITY_PENALTY

                if fn_tokens:
                    source_lower = chunk.source_file.lower()
                    if any(t in source_lower for t in fn_tokens):
                        boost += config.FILENAME_TOKEN_BOOST

                hit_count = hit_counts.get(key, 1)
                if hit_count > 1:
                    boost += config.HIT_COUNT_BOOST * (hit_count - 1)

            if chunk.graph_score > 0:
                boost += getattr(config, 'GRAPH_SCORE_WEIGHT', 0.20) * chunk.graph_score

            chunk.rerank_score += boost

        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks

    # ── Diversity cap for broad scope ─────────────────────────────────
    def _diversity_cap(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        doc_buckets: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            key = chunk.doc_id if chunk.doc_id else chunk.source_file
            doc_buckets[key].append(chunk)

        # Sort documents by their best chunk score so we keep the highest-quality docs
        sorted_buckets = sorted(
            doc_buckets.items(),
            key=lambda x: max(c.rerank_score for c in x[1]),
            reverse=True,
        )

        result  = []
        doc_count = 0
        for doc_key, doc_chunks in sorted_buckets:
            if doc_count >= config.MAX_DOCS_BROAD:
                break
            result.extend(doc_chunks[:config.MAX_CHUNKS_PER_DOC])
            doc_count += 1

        return result

    # ── Threshold + cap ────────────────────────────────────────────────
    def _threshold_and_cap(
        self, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        filtered = [c for c in chunks if c.rerank_score >= config.CONFIDENCE_THRESHOLD]
        if not filtered:
            logger.warning(
                "All chunks below threshold %.2f", config.CONFIDENCE_THRESHOLD
            )
            return []
        return filtered[:top_k]

    # ── Neighbor Expansion ─────────────────────────────────────────────
    def _expand_neighbors(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        For each top chunk, try to fetch its immediate neighbors (index-1 and index+1) 
        from the same source file to expand the context window.
        """
        if not chunks:
            return chunks

        expanded = []
        client = self.db.get_client()
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Keep track of fetched keys to avoid duplicates
        seen_keys = {(c.source_file, c.chunk_index) for c in chunks}

        for chunk in list(chunks): # iterate over a copy
            expanded.append(chunk)

            # Define neighbors to fetch
            neighbors = [chunk.chunk_index - 1, chunk.chunk_index + 1]
            for n_idx in neighbors:
                if n_idx < 0:
                    continue
                n_key = (chunk.source_file, n_idx)
                if n_key in seen_keys:
                    continue

                try:
                    points, _ = client.scroll(
                        collection_name=self.db.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(key="source_file", match=MatchValue(value=chunk.source_file)),
                                FieldCondition(key="chunk_index", match=MatchValue(value=n_idx)),
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                    )

                    if points:
                        p = points[0].payload or {}
                        neighbor_chunk = RetrievedChunk(
                            text=p.get("text", ""),
                            source_file=p.get("source_file", p.get("source", "")),
                            page_no=p.get("page_no", 0),
                            page_label=p.get("page_label", ""),
                            chunk_index=p.get("chunk_index", 0),
                            chunk_id=p.get("chunk_id", ""),
                            section=p.get("section", ""),
                            is_table=p.get("is_table", False),
                            token_count=p.get("token_count", 0),
                            chunk_year=p.get("chunk_year", []),
                            doc_year=p.get("doc_year", ""),
                            doc_id=p.get("doc_id", ""),
                            filename_tokens=p.get("filename_tokens", []),
                            keywords=p.get("keywords", []),
                            rerank_score=chunk.rerank_score - 0.05, # slightly lower score than anchor
                            is_temporal_expanded=True,
                            source_tag=chunk.source_tag,
                            arm_label=chunk.arm_label,
                        )
                        expanded.append(neighbor_chunk)
                        seen_keys.add(n_key)
                except Exception as e:
                    logger.debug("Failed to fetch neighbor %s:%d | %s", chunk.source_file, n_idx, e)

        # Sort the final expanded list by score again
        expanded.sort(key=lambda c: c.rerank_score, reverse=True)
        return expanded

    # ── Filter builders ────────────────────────────────────────────────
    def _build_filter(self, hints: dict) -> dict:
        if not getattr(config, "ENABLE_METADATA_FILTERING", True):
            return {}
        f = {}
        if hints.get("doc_year"):
            f["doc_year"] = hints["doc_year"]
        if hints.get("filename_tokens"):
            tokens = []
            for t in hints["filename_tokens"]:
                tokens.extend(t.lower().split())  
            f["filename_tokens"] = tokens 
        if hints.get("section"):
            f["section"] = hints["section"]
        if hints.get("keywords"):
            f["keywords"] = hints["keywords"]
        if hints.get("source"):
            f["source"] = hints["source"]
        if hints.get("source_file"):
            f["source_file"] = hints["source_file"]
        return f

    def _build_filter_from_arm(self, arm: ComparisonArm) -> dict:
        if not getattr(config, "ENABLE_METADATA_FILTERING", True):
            return {}
        f = {}
        if arm.year:
            f["doc_year"] = arm.year
        if arm.filename_tokens:
            tokens = []
            for t in arm.filename_tokens:
                tokens.extend(t.lower().split())  
            f["filename_tokens"] = tokens 
        return f

    def _relax_filter(self, filter_dict: dict) -> dict:
        relaxed = dict(filter_dict)
        if "section" in relaxed:
            del relaxed["section"]
            return relaxed
        if "keywords" in relaxed:
            del relaxed["keywords"]
            return relaxed
        if "filename_tokens" in relaxed:
            del relaxed["filename_tokens"]
            return relaxed
        if "doc_year" in relaxed:
            del relaxed["doc_year"]
            return relaxed
        return {}

    # ── Embed + search helpers ─────────────────────────────────────────
    def _embed(self, query: str):
        try:
            dense, sparse = self.embedder.embed_query(query)
            return dense, sparse
        except Exception as e:
            logger.error("Query embedding failed | %s", e)
            return None, None

    def _search(
        self, dense, sparse, filter_dict: dict | None
    ) -> list[dict]:
        try:
            return self.db.search_hybrid(
                query_dense  = dense,
                query_sparse = sparse,
                filter_dict  = filter_dict,
                top_k        = config.TOP_K_SEARCH,
            )
        except Exception as e:
            logger.error("Qdrant search failed | %s", e)
            return []

    # ── Chunk formatter for LLM ────────────────────────────────────────
    def format_for_llm(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            if chunk.arm_label:
                label = chunk.arm_label
            elif chunk.is_weak_match:
                label = "[LOW CONFIDENCE]"
            else:
                label = chunk.section or "General"

            header = f"--- Context {i} {chunk.source_tag} | {label} ---"
            parts.append(f"{header}\n{chunk.text}\n")

        return "\n".join(parts)

    # ── Dict → RetrievedChunk ──────────────────────────────────────────
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
            chunk_id              = result.get("chunk_id", ""),
            section               = result.get("section", ""),
            is_table              = result.get("is_table", False),
            doc_year              = result.get("doc_year", ""),
            doc_id                = result.get("doc_id", ""),
            chunk_year            = result.get("chunk_year", []),
            token_count           = result.get("token_count", 0),
            qdrant_score          = result.get("score", 0.0),
            rerank_score          = rerank_score,
            is_weak_match         = rerank_score < config.CONFIDENCE_THRESHOLD,
            source_tag            = source_tag,
            arm_label             = "",
            summary               = result.get("summary", ""),
            graph_score           = result.get("graph_score", 0.0),
            graph_hops            = result.get("graph_hops", 0),
            graph_path            = result.get("graph_path", ""),
            retrieval_channel     = result.get("retrieval_channel", ""),
        )

    # ── Async Embed + search helpers ───────────────────────────────────
    async def _async_embed_and_search(
        self,
        queries:     list[str],
        filter_dict: dict | None,
        enable_graph_rag: bool = True,
        mode: str = "local",
    ) -> tuple[list[dict], dict[tuple, int]]:
        raw_pool:   list[dict]       = []
        hit_counts: dict[tuple, int] = defaultdict(int)

        async def embed_one(query):
            key = _cache_key(query)
            if key in _EMBED_CACHE:
                return _EMBED_CACHE[key]
            try:
                dense, sparse = await self.embedder.async_embed_query(query)
                _EMBED_CACHE[key] = (dense, sparse)
                if len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
                    oldest_key = next(iter(_EMBED_CACHE))
                    del _EMBED_CACHE[oldest_key]
                return dense, sparse
            except Exception as e:
                logger.error("Async query embedding failed for query '%s' | %s", query, e)
                return None

        embeddings = await asyncio.gather(*(embed_one(q) for q in queries))

        async def search_one(idx, query):
            emb = embeddings[idx]
            if emb is None:
                return []
            dense, sparse = emb
            try:
                return await self.db.async_search_hybrid(
                    query_dense  = dense,
                    query_sparse = sparse,
                    filter_dict  = filter_dict,
                    top_k        = config.TOP_K_SEARCH,
                )
            except Exception as e:
                logger.error("Async Qdrant search failed | %s", e)
                return []

        search_results = await asyncio.gather(*(search_one(i, q) for i, q in enumerate(queries)))

        for i, results in enumerate(search_results):
            for r in results:
                key = (r.get("source_file", ""), r.get("chunk_index", 0))
                hit_counts[key] += 1
                raw_pool.append(r)

            logger.info(
                "Async Searched | query='%s...' | filter=%s | results=%d",
                queries[i][:50], filter_dict, len(results),
            )

        if enable_graph_rag:
            raw_pool = await self._async_merge_graph_candidates(raw_pool, queries[0], mode=mode)

        return raw_pool, hit_counts

    async def _async_merge_graph_candidates(self, raw_pool: list[dict], query: str, mode: str = "local") -> list[dict]:
        if not self.graph_rag or not self.graph_rag.available:
            return raw_pool

        pool_by_key = {
            (r.get("source_file", ""), r.get("chunk_index", 0)): r
            for r in raw_pool
        }

        graph_candidates, relationships = await asyncio.to_thread(
            self.graph_rag.retrieve_candidates,
            query,
            mode=mode,
            llm_invoke=self._graph_llm_invoke,
            embedding_fn=lambda q: self.embedder.embed_query(q)[0]
        )
        if relationships:
            for rel in relationships:
                if rel not in self.last_relationships:
                    self.last_relationships.append(rel)

        if not graph_candidates:
            return raw_pool

        missing_candidates = []
        fetch_tasks = []

        for candidate in graph_candidates:
            matched_key = None
            for key, r in pool_by_key.items():
                if r.get("source_file") == candidate.source and str(r.get("chunk_id")) == str(candidate.chunk_id):
                    matched_key = key
                    break

            graph_boost = max(0.0, min(1.0, float(candidate.graph_score)))
            
            if matched_key:
                r = pool_by_key[matched_key]
                r["graph_score"] = max(r.get("graph_score", 0.0), graph_boost)
                r["graph_hops"] = int(candidate.graph_hops)
                r["graph_path"] = candidate.graph_path
                r["retrieval_channel"] = "hybrid_graph"
            else:
                missing_candidates.append(candidate)
                fetch_tasks.append(self._async_fetch_chunk_by_source_id(candidate.source, candidate.chunk_id))

        if fetch_tasks:
            fetched_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for candidate, r in zip(missing_candidates, fetched_results):
                if isinstance(r, Exception) or r is None:
                    continue
                graph_boost = max(0.0, min(1.0, float(candidate.graph_score)))
                r["graph_score"] = graph_boost
                r["graph_hops"] = int(candidate.graph_hops)
                r["graph_path"] = candidate.graph_path
                r["retrieval_channel"] = "graph"
                pool_by_key[(r.get("source_file", ""), r.get("chunk_index", 0))] = r

        logger.info("[GraphRAG] Async Merged graph candidates: missing fetched=%d", len(missing_candidates))
        return list(pool_by_key.values())

    async def _async_fetch_chunk_by_source_id(self, source: str, chunk_id: str) -> dict | None:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            client = self.db.async_client
            points, _ = await client.scroll(
                collection_name=self.db.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="source_file", match=MatchValue(value=str(source))),
                        FieldCondition(key="chunk_id", match=MatchValue(value=str(chunk_id))),
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                points, _ = await client.scroll(
                    collection_name=self.db.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="source", match=MatchValue(value=str(source))),
                            FieldCondition(key="chunk_id", match=MatchValue(value=str(chunk_id))),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
            if not points:
                return None
            
            payload = points[0].payload or {}
            return {
                "score": 0.0,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file", payload.get("source", "")),
                "page_no": payload.get("page_no", 0),
                "page_range": payload.get("page_range", [0, 0]),
                "section": payload.get("section", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "is_table": payload.get("is_table", False),
                "doc_year": payload.get("doc_year", ""),
                "chunk_year": payload.get("chunk_year", []),
                "token_count": payload.get("token_count", 0),
                "doc_id": payload.get("doc_id", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "filename_tokens": payload.get("filename_tokens", []),
                "has_tables": payload.get("has_tables", False),
                "summary": payload.get("summary", ""),
                "keywords": payload.get("keywords", []),
                "page_label": payload.get("page_label", ""),
            }
        except Exception as e:
            logger.error("Failed to async fetch graph chunk %s:%s | %s", source, chunk_id, e)
            return None

    async def _async_expand_neighbors(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        seen_keys = {(c.source_file, c.chunk_index) for c in chunks}
        
        tasks = []
        task_info = []
        
        for chunk in chunks:
            neighbors = [chunk.chunk_index - 1, chunk.chunk_index + 1]
            for n_idx in neighbors:
                if n_idx < 0:
                    continue
                n_key = (chunk.source_file, n_idx)
                if n_key in seen_keys:
                    continue
                seen_keys.add(n_key)
                
                tasks.append(
                    self.db.async_client.scroll(
                        collection_name=self.db.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(key="source_file", match=MatchValue(value=chunk.source_file)),
                                FieldCondition(key="chunk_index", match=MatchValue(value=n_idx)),
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
                task_info.append((chunk, n_idx))
                
        if not tasks:
            return chunks
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        expanded = list(chunks)
        for res, (chunk, n_idx) in zip(results, task_info):
            if isinstance(res, Exception):
                logger.debug("Failed async neighbor fetch | %s", res)
                continue
                
            points, _ = res
            if points:
                p = points[0].payload or {}
                neighbor_chunk = RetrievedChunk(
                    text=p.get("text", ""),
                    source_file=p.get("source_file", p.get("source", "")),
                    page_no=p.get("page_no", 0),
                    page_label=p.get("page_label", ""),
                    chunk_index=p.get("chunk_index", 0),
                    chunk_id=p.get("chunk_id", ""),
                    section=p.get("section", ""),
                    is_table=p.get("is_table", False),
                    token_count=p.get("token_count", 0),
                    chunk_year=p.get("chunk_year", []),
                    doc_year=p.get("doc_year", ""),
                    doc_id=p.get("doc_id", ""),
                    filename_tokens=p.get("filename_tokens", []),
                    keywords=p.get("keywords", []),
                    rerank_score=chunk.rerank_score - 0.05,
                    is_temporal_expanded=True,
                    source_tag=chunk.source_tag,
                    arm_label=chunk.arm_label,
                )
                expanded.append(neighbor_chunk)
                
        expanded.sort(key=lambda c: c.rerank_score, reverse=True)
        return expanded

def get_retriever(embedder: OllamaEmbedder, graph_rag=None) -> Retriever:
    return Retriever(embedder=embedder, db=embedder.db, graph_rag=graph_rag)

