"""
final_rag/qdrant_storage/store.py

Qdrant Vector Database Manager — Hybrid Dense (Qwen3) + Sparse (BM25) Architecture.
Supports scalar quantization, cross-field OR filtering, and full payload metadata indexing.
"""

from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    QuantizationSearchParams,
)

try:
    import final_rag.config as config
except ImportError:
    from .. import config

logger = logging.getLogger("qdrant_storage.store")

DENSE_WEIGHT  = getattr(config, "DENSE_WEIGHT", 0.85)
SPARSE_WEIGHT = getattr(config, "SPARSE_WEIGHT", 0.15)

_PAYLOAD_INDEXES = [
    ("source_file",      PayloadSchemaType.KEYWORD),
    ("page_no",          PayloadSchemaType.INTEGER),
    ("is_table",         PayloadSchemaType.BOOL),
    ("doc_year",         PayloadSchemaType.KEYWORD),
    ("chunk_years",      PayloadSchemaType.KEYWORD),
    ("entities",         PayloadSchemaType.KEYWORD),
    ("filename_tokens",  PayloadSchemaType.KEYWORD),
    ("doc_lang",         PayloadSchemaType.KEYWORD),
]


class QdrantManager:
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        default_collection: str = "",
        dimensions: int = 0,
    ):
        self.storage_path = Path(storage_path or getattr(config, "QDRANT_STORAGE_PATH", "./qdrant_db"))
        self.default_collection = default_collection or getattr(config, "QDRANT_COLLECTION_NAME", "rag_documents")
        self.dimensions = dimensions or getattr(config, "EMBED_DIMENSIONS", 1536)
        self._client: Optional[QdrantClient] = None
        self._initialized_collections: set[str] = set()

        os.makedirs(self.storage_path, exist_ok=True)
        self.ensure_collection()

    def get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=str(self.storage_path))
        return self._client

    def _collection_name(self, domain: str = "") -> str:
        if not domain:
            return self.default_collection
        collections_map = getattr(config, "QDRANT_COLLECTIONS", {})
        return collections_map.get(domain, domain if domain else self.default_collection)

    def recreate_collection(self, domain: str = "") -> None:
        name = self._collection_name(domain)
        client = self.get_client()
        existing = [c.name for c in client.get_collections().collections]
        if name in existing:
            client.delete_collection(collection_name=name)
            logger.info("Deleted existing Qdrant collection: %s", name)
        self._initialized_collections.discard(name)
        self.ensure_collection(domain)

    def ensure_collection(self, domain: str = "") -> None:
        name = self._collection_name(domain)
        if name in self._initialized_collections:
            return

        client = self.get_client()
        existing = [c.name for c in client.get_collections().collections]

        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=self.dimensions, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            logger.info("Created Qdrant collection: %s (dim=%d, quantized=int8)", name, self.dimensions)

        # Ensure payload schema indexes
        for field_name, schema_type in _PAYLOAD_INDEXES:
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass

        self._initialized_collections.add(name)

    def upsert(self, points: List[Any], domain: str = "") -> None:
        if not points:
            return
        collection_name = self._collection_name(domain)
        self.ensure_collection(domain)
        self.get_client().upsert(collection_name=collection_name, points=points)

    def delete_document(self, source_file: str, domain: str = "") -> None:
        collection_name = self._collection_name(domain)
        self.ensure_collection(domain)
        self.get_client().delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )

    def get_stats(self, domain: str = "") -> Dict[str, Any]:
        collection_name = self._collection_name(domain)
        try:
            client = self.get_client()
            info = client.get_collection(collection_name=collection_name)
            return {
                "num_chunks": info.points_count or 0,
                "status": "online",
            }
        except Exception as e:
            logger.warning(f"Failed to get collection stats for {collection_name}: {e}")
            return {
                "num_chunks": 0,
                "status": "online",
            }

    def _build_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filter_dict:
            return None

        must_conditions   = []
        should_conditions = []
        or_group_conditions = []
        remaining = {}

        for key, value in filter_dict.items():
            if "__or__" in key:
                field_a, field_b = key.split("__or__", 1)
                values = value if isinstance(value, list) else [value]
                group = [
                    FieldCondition(key=field_a, match=MatchValue(value=v)) for v in values
                ] + [
                    FieldCondition(key=field_b, match=MatchValue(value=v)) for v in values
                ]
                or_group_conditions.append(Filter(should=group))
            else:
                remaining[key] = value

        for key, value in remaining.items():
            if isinstance(value, list):
                should_conditions.extend([
                    FieldCondition(key=key, match=MatchValue(value=v)) for v in value
                ])
            else:
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        must_conditions.extend(or_group_conditions)

        if must_conditions and should_conditions:
            return Filter(must=must_conditions + [Filter(should=should_conditions)])
        if should_conditions:
            return Filter(should=should_conditions)
        if must_conditions:
            return Filter(must=must_conditions)
        return None

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    def search_hybrid(
        self,
        query_dense:  List[float],
        query_sparse: Dict[int, float],
        domain:       str = "",
        filter_dict:  Optional[Dict[str, Any]] = None,
        top_k:        int = 8,
    ) -> List[Dict[str, Any]]:
        collection_name = self._collection_name(domain)
        self.ensure_collection(domain)

        sorted_pairs = sorted(query_sparse.items())
        sparse_vec = SparseVector(
            indices=[k for k, _ in sorted_pairs],
            values=[v for _, v in sorted_pairs],
        )
        search_filter = self._build_filter(filter_dict)

        try:
            client = self.get_client()
            dense_mult  = getattr(config, "DENSE_OVERSAMPLE_MULTIPLIER", 5)
            sparse_mult = getattr(config, "SPARSE_OVERSAMPLE_MULTIPLIER", 2)

            dense_results = client.query_points(
                collection_name = collection_name,
                query           = query_dense,
                using           = "dense",
                limit           = top_k * dense_mult,
                with_payload    = True,
                query_filter    = search_filter,
                search_params   = SearchParams(
                    quantization=QuantizationSearchParams(rescore=True)
                ),
            )
            sparse_results = client.query_points(
                collection_name = collection_name,
                query           = sparse_vec,
                using           = "sparse",
                limit           = top_k * sparse_mult,
                with_payload    = True,
                query_filter    = search_filter,
            )

            dense_scores  = {str(r.id): r.score for r in dense_results.points}
            sparse_scores = {str(r.id): r.score for r in sparse_results.points}
            payload_map   = {str(r.id): r.payload for r in dense_results.points}
            for r in sparse_results.points:
                if str(r.id) not in payload_map:
                    payload_map[str(r.id)] = r.payload

            all_ids = list(payload_map.keys())
            if not all_ids:
                return []

            raw_dense   = [dense_scores.get(i, 0.0)  for i in all_ids]
            raw_sparse  = [sparse_scores.get(i, 0.0) for i in all_ids]
            norm_dense  = self._normalize(raw_dense)
            norm_sparse = self._normalize(raw_sparse)

            blended = [DENSE_WEIGHT * d + SPARSE_WEIGHT * s for d, s in zip(norm_dense, norm_sparse)]
            ranked  = sorted(zip(all_ids, blended), key=lambda x: x[1], reverse=True)[:top_k]

            hits = []
            for point_id, score in ranked:
                payload = payload_map.get(point_id, {})
                hits.append({
                    "score":           score,
                    "text":            payload.get("text", ""),
                    "source_file":     payload.get("source_file", ""),
                    "page_no":         payload.get("page_no", 0),
                    "page_range":      payload.get("page_range", (0, 0)),
                    "chunk_index":     payload.get("chunk_index", 0),
                    "is_table":        payload.get("is_table", False),
                    "doc_year":        payload.get("doc_year", ""),
                    "doc_lang":        payload.get("doc_lang", ""),
                    "chunk_years":     payload.get("chunk_years", []),
                    "heading":         payload.get("heading", ""),
                    "entities":        payload.get("entities", []),
                    "token_count":     payload.get("token_count", 0),
                    "doc_id":          payload.get("doc_id", ""),
                    "chunk_id":        payload.get("chunk_id", ""),
                    "filename_tokens": payload.get("filename_tokens", []),
                    "page_label":      payload.get("page_label", ""),
                })

            return hits

        except Exception as e:
            logger.error("Hybrid search failed on collection %s: %s", collection_name, e)
            return []