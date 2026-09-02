"""
final_rag/agent/query_cleaner.py

Query Cleaner: Analyzes user query, detects intent (chitchat vs search), normalizes language (Hinglish/Hindi/English),
extracts search signals (scope, structure, filter hints, comparison arms), and scores subqueries using embedding similarity.
Fully supports active document scoping and active document comparison scoping.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
import httpx

try:
    import final_rag.config as config
except ImportError:
    from .. import config

from .models import CleanedQuery, ComparisonArm, Subquery
from .claude_client import ClaudeClient
from ..prompts.cleaner_prompt import CLEANER_SYSTEM_PROMPT, build_cleaner_prompt

logger = logging.getLogger("agent.query_cleaner")

SUBQUERY_WEIGHT_THRESHOLD = getattr(config, "SUBQUERY_WEIGHT_THRESHOLD", 0.55)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class QueryCleaner:
    def __init__(
        self,
        model:       str = getattr(config, "CLEANER_MODEL", "claude-sonnet-5"),
        embed_model: str = getattr(config, "EMBED_MODEL", "qwen3-embedding:4b"),
        ollama_url:  str = getattr(config, "OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"),
    ):
        self.model        = model
        self.embed_model  = embed_model
        self.ollama_url   = ollama_url
        self.llm_client   = ClaudeClient(model=model)
        self.http_client  = httpx.Client(timeout=30.0)

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                r = self.http_client.post(self.ollama_url, json={"model": self.embed_model, "prompt": text})
                r.raise_for_status()
                embeddings.append(r.json().get("embedding", []))
            except Exception:
                embeddings.append([])
        return embeddings

    def _score_subqueries(self, improved_query: str, subqueries: List[Dict]) -> List[Dict]:
        if not subqueries:
            return []
        texts      = [improved_query] + [sq.get("query", "") for sq in subqueries if sq.get("query")]
        embeddings = self._batch_embed(texts)
        if not embeddings or not embeddings[0]:
            return [{"query": sq.get("query", ""), "weight": 1.0} for sq in subqueries]

        anchor = embeddings[0]
        scored = []
        for i, sq in enumerate(subqueries):
            query_str = sq.get("query", "")
            if not query_str:
                continue
            emb = embeddings[i + 1] if i + 1 < len(embeddings) and embeddings[i + 1] else anchor
            weight = cosine_similarity(anchor, emb) if len(emb) == len(anchor) else 0.8
            scored.append({"query": query_str, "weight": round(weight, 4)})
        return sorted(scored, key=lambda x: x["weight"], reverse=True)

    def _parse_raw(self, raw: str) -> Dict[str, Any]:
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            elif raw.startswith("python"):
                raw = raw[6:].strip()
        return json.loads(raw)

    def clean_query(self, query: str) -> Dict[str, Any]:
        start       = time.perf_counter()
        user_prompt = build_cleaner_prompt(query)
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            try:
                raw_text = self.llm_client.generate(
                    system      = CLEANER_SYSTEM_PROMPT,
                    prompt      = user_prompt,
                    temperature = 0.0,
                )
                parsed = self._parse_raw(raw_text.strip())

                improved_query = parsed.get("improved_query", query)
                raw_subqueries = parsed.get("subqueries", [])
                all_subqueries = self._score_subqueries(improved_query, raw_subqueries)
                filtered = [
                    sq for sq in all_subqueries
                    if sq.get("weight", 0.0) >= SUBQUERY_WEIGHT_THRESHOLD
                ]

                result = {
                    "original_query":    query,
                    "improved_query":    improved_query,
                    "detected_language": parsed.get("detected_language", "english"),
                    "intent":            parsed.get("intent", "document_query"),
                    "target_scope":      parsed.get("target_scope", "broad"),
                    "answer_structure":  parsed.get("answer_structure", "direct"),
                    "specificity":       parsed.get("specificity", "low"),
                    "filter_hints":      parsed.get("filter_hints", {}),
                    "comparison_arms":   parsed.get("comparison_arms", []),
                    "all_subqueries":    all_subqueries,
                    "subqueries":        filtered,
                    "processing_time_sec": round(time.perf_counter() - start, 3),
                }
                return result

            except Exception as e:
                logger.warning("Query cleaning attempt %d/%d failed: %s", attempt, max_attempts, e)

        return {
            "original_query":      query,
            "improved_query":      query,
            "detected_language":   "english",
            "intent":              "document_query",
            "target_scope":        "broad",
            "answer_structure":    "direct",
            "specificity":         "low",
            "filter_hints":        {},
            "comparison_arms":     [],
            "all_subqueries":      [],
            "subqueries":          [],
            "processing_time_sec": round(time.perf_counter() - start, 3),
        }

    def clean(
        self,
        query: str,
        active_documents: Optional[List[str]] = None,
        active_doc_id: Optional[str] = None,
        active_doc_filename: Optional[str] = None,
        active_doc_ids: Optional[List[str]] = None,
        active_doc_filenames: Optional[List[str]] = None,
    ) -> CleanedQuery:
        result = self.clean_query(query)

        subqueries = [
            Subquery(query=sq["query"], weight=sq["weight"]) if isinstance(sq, dict) else Subquery(query=str(sq), weight=1.0)
            for sq in result.get("subqueries", [])
        ]
        all_subqueries = [
            Subquery(query=sq["query"], weight=sq["weight"]) if isinstance(sq, dict) else Subquery(query=str(sq), weight=1.0)
            for sq in result.get("all_subqueries", [])
        ]

        comparison_arms = [
            ComparisonArm(
                label           = arm.get("label", ""),
                query           = arm.get("label", ""),
                year            = arm.get("year"),
                filename_tokens = arm.get("filename_tokens", []),
                filter_hints    = {
                    "doc_year": arm.get("year"),
                    "filename_tokens": arm.get("filename_tokens", []),
                },
            )
            for arm in result.get("comparison_arms", [])[:4]
        ]

        # Resolve active documents list
        resolved_docs: List[str] = []
        if active_documents:
            resolved_docs.extend([str(d) for d in active_documents if d])
        if active_doc_filenames:
            resolved_docs.extend([str(d) for d in active_doc_filenames if d])
        if active_doc_filename and active_doc_filename not in resolved_docs:
            resolved_docs.append(str(active_doc_filename))

        scope = result.get("target_scope", "broad")
        if len(resolved_docs) == 1:
            scope = "active"
        elif len(resolved_docs) > 1:
            scope = "active_compare"

        return CleanedQuery(
            original_query      = result["original_query"],
            improved_query      = result["improved_query"],
            detected_language   = result.get("detected_language", "english"),
            intent              = result.get("intent", "document_query"),
            target_scope        = scope,
            answer_structure    = result.get("answer_structure", "direct"),
            specificity         = result.get("specificity", "low"),
            filter_hints        = result.get("filter_hints", {}),
            comparison_arms     = comparison_arms,
            active_documents    = resolved_docs,
            subqueries          = subqueries,
            all_subqueries      = all_subqueries,
            processing_time_sec = result.get("processing_time_sec", 0.0),
            active_doc_id       = active_doc_id,
            active_doc_ids      = active_doc_ids or [],
            extracted_keywords  = result.get("filter_hints", {}).get("keywords", []),
        )


def get_cleaner(model: str = getattr(config, "CLEANER_MODEL", "claude-sonnet-5")) -> QueryCleaner:
    return QueryCleaner(model=model)