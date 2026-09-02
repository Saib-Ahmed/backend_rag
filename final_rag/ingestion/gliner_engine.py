"""
final_rag/ingestion/gliner_engine.py

GLiNER 2.5 entity extraction engine for chunk-level semantic enrichment.
Extracts deduped legal entities (court, petitioner, respondent, judge, lawyer, act, section, etc.)
for search scoring boosts.
"""

from __future__ import annotations

import logging
import time

try:
    from final_rag.config import (
        GLINER_MODEL_NAME,
        GLINER_LABELS,
        GLINER_THRESHOLD,
        GLINER_RETRY_COOLDOWN_SEC,
    )
except ImportError:
    from ..config import (
        GLINER_MODEL_NAME,
        GLINER_LABELS,
        GLINER_THRESHOLD,
        GLINER_RETRY_COOLDOWN_SEC,
    )

logger = logging.getLogger("ingestion.gliner_engine")

_model = None
_load_failed_at: float | None = None


def _load_model():
    """Loads GLiNER model checkpoint with automatic HF hub caching."""
    try:
        from gliner2 import AutoExtractor
        return AutoExtractor.from_pretrained(GLINER_MODEL_NAME)
    except ImportError:
        from gliner import GLiNER
        return GLiNER.from_pretrained(GLINER_MODEL_NAME)


def _get_model():
    """Lazy singleton with cooldown-based retry so failures don't penalize latency."""
    global _model, _load_failed_at

    if _model is not None:
        return _model

    if _load_failed_at is not None and (time.time() - _load_failed_at) < GLINER_RETRY_COOLDOWN_SEC:
        return None

    try:
        _model = _load_model()
        _load_failed_at = None
        logger.info("GLiNER model loaded | %s", GLINER_MODEL_NAME)
    except Exception as e:
        logger.warning("GLiNER unavailable (will retry after %ds): %s", GLINER_RETRY_COOLDOWN_SEC, e)
        _load_failed_at = time.time()
        _model = None

    return _model


def extract_gliner_entities(text: str) -> list[str]:
    """Returns a deduped list of entity strings found in chunk text."""
    model = _get_model()
    if model is None or not text.strip():
        return []
    try:
        if hasattr(model, "extract_entities"):
            result = model.extract_entities(text, GLINER_LABELS, threshold=GLINER_THRESHOLD)
            entities_by_label = result.get("entities", {}) if isinstance(result, dict) else {}

            seen: dict[str, str] = {}
            for spans in entities_by_label.values():
                for item in spans:
                    text_val = item.get("text") if isinstance(item, dict) else item
                    if not text_val:
                        continue
                    text_val = text_val.strip()
                    if not text_val:
                        continue
                    key = text_val.lower()
                    if key not in seen:
                        seen[key] = text_val
            return list(seen.values())
        elif hasattr(model, "predict_entities"):
            entities = model.predict_entities(text, GLINER_LABELS, threshold=GLINER_THRESHOLD)
            seen = {}
            for item in entities:
                text_val = item.get("text", "").strip()
                if text_val and text_val.lower() not in seen:
                    seen[text_val.lower()] = text_val
            return list(seen.values())
        return []
    except Exception as e:
        logger.warning("GLiNER inference failed on chunk: %s", e)
        return []
