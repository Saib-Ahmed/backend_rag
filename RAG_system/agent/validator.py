"""
agent/validator.py

Lightweight groundedness, numerical check, and citation formatting validator.
"""

from __future__ import annotations

import logging
import re

import config

logger = logging.getLogger("agent.validator")


class Validator:
    def __init__(
        self,
        min_overlap: float = getattr(config, "ANSWER_VALIDATION_MIN_OVERLAP", 0.05),
        max_tokens: int = getattr(config, "ANSWER_VALIDATION_MAX_TOKENS", 200),
        enable_validation: bool = getattr(config, "ENABLE_ANSWER_VALIDATION", True),
    ):
        self.min_overlap = min_overlap
        self.max_tokens = max_tokens
        self.enable_validation = enable_validation
        self.last_reason = "passed"

    def validate(self, answer: str, context: str) -> bool:
        """
        Runs the groundedness checks against context.
        """
        self.last_reason = "passed"
        if not self.enable_validation:
            self.last_reason = "disabled"
            return True

        normalized_answer = (answer or "").strip()
        if not normalized_answer or self._is_fallback_answer(normalized_answer):
            self.last_reason = "empty_or_fallback"
            return True

        overlap = self._validation_overlap(normalized_answer, context)
        if overlap < self.min_overlap:
            self.last_reason = f"low_overlap:{overlap:.3f}"
            return False

        if self._answer_uses_unseen_numbers(normalized_answer, context):
            self.last_reason = "unseen_numbers"
            return False

        if not self._answer_has_per_statement_citations(normalized_answer):
            self.last_reason = "citation_format"
            return False

        return True

    def compute_confidence(
        self,
        answer: str,
        context: str,
        retrieval_strength: float,
        validation_passed: bool,
    ) -> float:
        """
        Estimates answer confidence from retrieval and grounding signals.
        """
        overlap = self._validation_overlap(answer or "", context)
        confidence = (0.55 * retrieval_strength) + (0.45 * overlap)
        if not validation_passed:
            confidence *= 0.55
        return round(max(0.0, min(1.0, confidence)), 3)

    def _is_fallback_answer(self, answer: str) -> bool:
        fallback = getattr(config, "FALLBACK_ANSWER", "I cannot answer this based on the provided documents.").lower()
        ans = answer.lower()
        if fallback in ans:
            return True
        if "i cannot answer" in ans or "no relevant information" in ans or "couldn't find specific information" in ans:
            return True
        return False

    def _truncate_for_validation(self, text: str) -> str:
        tokens = (text or "").split()
        if len(tokens) <= self.max_tokens:
            return text or ""
        return " ".join(tokens[:self.max_tokens])

    def _validation_overlap(self, answer: str, context: str) -> float:
        answer_tokens = [
            tok for tok in self._tokenize_for_keyword(self._truncate_for_validation(answer))
            if len(tok) > 2
        ]
        context_tokens = set(
            tok for tok in self._tokenize_for_keyword(context)
            if len(tok) > 2
        )
        if not answer_tokens or not context_tokens:
            return 0.0

        matched = sum(1 for token in set(answer_tokens) if token in context_tokens)
        return matched / max(1, len(set(answer_tokens)))

    def _answer_uses_unseen_numbers(self, answer: str, context: str) -> bool:
        number_pattern = r"\b\d[\d,]*(?:\.\d+)?\b"
        answer_numbers = {n.replace(",", "") for n in re.findall(number_pattern, answer or "")}
        if not answer_numbers:
            return False
        answer_numbers = {n for n in answer_numbers if int(float(n)) > 3}
        if not answer_numbers:
            return False
        context_numbers = {n.replace(",", "") for n in re.findall(number_pattern, context or "")}
        return not answer_numbers.issubset(context_numbers)

    def _answer_has_per_statement_citations(self, answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return False

        citation_pattern = re.compile(
            r"\[Source:\s*[^,\]]+,\s*(?:Chunk|Page|page):?\s*[^\]]+\]"
            r"|"
            r"\[[^\]]+,\s*Page\s*\d+[^\]]*\]"
            r"|"
            r"\[[^\]]+,\s*page\s*\d+[^\]]*\]"
            r"|"
            r"\[[^\]]+,\s*Page\s*~?\d+[^\]]*\]"
        )
        if not citation_pattern.search(text):
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        substantive_lines = 0
        cited_lines = 0

        for line in lines:
            if line.startswith("#"):
                continue
            if len(line.split()) < 8:
                continue
            if line.endswith(":") and len(line.split()) <= 12:
                continue

            substantive_lines += 1
            if citation_pattern.search(line):
                cited_lines += 1

        if substantive_lines == 0:
            return False

        return (cited_lines / substantive_lines) >= 0.8

    def _tokenize_for_keyword(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"\w+", text.lower())
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "is", "are", "was", "were", "be",
            "been", "being", "it", "this", "that", "these", "those", "its",
            "from", "as", "up", "out", "if", "then", "than", "so", "do",
            "does", "did", "not", "no", "can", "will", "would", "could",
            "should", "may", "might", "has", "have", "had", "also", "each",
            "into", "over", "under", "between", "through", "about",
            "document", "section", "page", "table", "data", "information",
            "value", "values", "content", "text", "note",
        }
        return [t for t in tokens if t not in stop_words and len(t) >= 2]


def get_validator() -> Validator:
    return Validator()
