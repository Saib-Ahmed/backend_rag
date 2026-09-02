"""
agent/grounding_checker.py

Grounding / Faithfulness Checker
=================================
Verifies whether the LLM-generated answer is supported by the retrieved
document chunks (i.e. no hallucinations / out-of-chunk fabrications).

Provider strategy mirrors msme_extractor.py:
  Primary  → Gemini (gemini-2.0-flash)
  Fallback → NVIDIA GLM-5.2  (z-ai/glm-5.2)

If both providers fail the checker returns verdict="UNCHECKED" so the main
chat flow is NEVER broken.

Returned GroundingResult fields
--------------------------------
  verdict            : "GROUNDED" | "PARTIAL" | "UNGROUNDED" | "UNCHECKED"
  score              : float 0.0–1.0 (1.0 = fully grounded, 0.0 = hallucinated)
  unsupported_claims : list[str]  — specific claims that are NOT in the chunks
  reasoning          : str        — one-sentence explanation
  provider           : str        — which API was used ("gemini" | "kimi" | "none")
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent.grounding_checker")

# ── Constants ──────────────────────────────────────────────────────────────────
NVIDIA_GROUNDING_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_GROUNDING_MODEL = os.getenv("NVIDIA_GROUNDING_MODEL", "z-ai/glm-5.2")
GEMINI_GROUNDING_MODEL = os.getenv("GEMINI_GROUNDING_MODEL", "gemini-2.5-flash")
GROUNDING_TIMEOUT_SEC  = int(os.getenv("GROUNDING_TIMEOUT_SEC", "90"))
# GROUNDING_ENABLED is re-read per call (fix #9) — not cached at module level

# Maximum characters sent to the checker to stay well within context limits
MAX_ANSWER_CHARS  = 15000
MAX_CONTEXT_CHARS = 60000

# ── Regex: only strip file-citation markers, not legal references ──────────────
# Matches patterns like [filename.pdf, Page 3] or [DocName, Page 12]
# Does NOT strip [Section 2(n)], [Article 7], etc.
_FILE_CITATION_RE = re.compile(
    r'\[[^\]]*?(?:\.pdf|\.docx|\.doc|\.txt|page\s*\d+|pg\.?\s*\d+)[^\]]*?\]',
    re.IGNORECASE,
)


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class GroundingResult:
    verdict:            str          = "UNCHECKED"   # GROUNDED | PARTIAL | UNGROUNDED | UNCHECKED
    score:              float        = 0.0
    unsupported_claims: List[str]    = field(default_factory=list)
    claims:             List[dict]   = field(default_factory=list)
    reasoning:          str          = "Grounding check was not performed."
    provider:           str          = "none"

    def to_dict(self) -> dict:
        return {
            "verdict":            self.verdict,
            "score":              self.score,
            "unsupported_claims": self.unsupported_claims,
            "claims":             self.claims,
            "reasoning":          self.reasoning,
            "provider":           self.provider,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_valid_key(key: str) -> bool:
    """Same placeholder check used in msme_extractor.py."""
    if not key or not key.strip():
        return False
    key_lower = key.lower()
    for placeholder in ("placeholder", "your_gemini", "your_nvidia", "api_key_here", "key_here"):
        if placeholder in key_lower:
            return False
    return True


def _build_grounding_prompt(query: str, context_text: str, answer: str) -> str:
    """
    Multilingual-aware faithfulness evaluation prompt.
    The LLM must respond with a single JSON object — no markdown, no prose.
    """
    return f"""You are an expert multilingual faithfulness evaluator for a Retrieval-Augmented Generation (RAG) system.

TASK: 
1. Determine whether the GENERATED ANSWER is faithful to and fully supported by the PROVIDED CONTEXT.
2. Break down the GENERATED ANSWER statement-by-statement into individual claims and evaluate EACH statement against the context.

=== USER QUESTION ===
{query}

=== PROVIDED CONTEXT (retrieved document chunks) ===
{context_text}

=== GENERATED ANSWER ===
{answer}

=== CRITICAL MULTILINGUAL EVALUATION RULES ===
⚠️  IMPORTANT: The GENERATED ANSWER may be in a DIFFERENT LANGUAGE (e.g., Hindi, Urdu, Bengali) than the PROVIDED CONTEXT (which may be in English). This is EXPECTED and NORMAL.

When the answer and context are in different languages:
1. TRANSLATE the answer's claims mentally into English before comparing them to the context.
2. A claim is GROUNDED if its MEANING is directly stated or clearly implied by the context, regardless of which language is used.
3. Do NOT mark a claim as "unsupported" just because the exact wording does not appear in the context — check SEMANTIC equivalence across languages.
4. Example: If the context says "registration would have an effect prospectively" and the answer says "पंजीकरण का प्रभाव भविष्यात्मक होता है", these are IDENTICAL in meaning → GROUNDED.
5. Example: If the context says "Sub-section (n) of Section 2 defines a supplier" and the answer says "धारा 2(n) में 'सप्लायर' को परिभाषित किया गया", these are IDENTICAL in meaning → GROUNDED.
6. A claim is UNGROUNDED ONLY if its MEANING is genuinely absent from the context, contradicts the context, or is a fabrication with no textual basis whatsoever.
7. Ignore differences in phrasing, language, script, or formatting — focus purely on factual and semantic faithfulness.
8. If the answer says "Not found in uploaded documents." or similar refusal — it IS grounded (correct refusal).
9. Legal section references (e.g., "Section 2(n)", "Chapter V", "Section 18") are GROUNDED if those sections or their effects are discussed in the context.

=== VERDICT SCALE ===
- GROUNDED   : All claims are semantically supported by the context (score 0.8–1.0)
- PARTIAL    : Most claims are supported but 1–3 genuinely unsupported claims exist (score 0.4–0.79)
- UNGROUNDED : Multiple or significant claims have NO semantic basis in the context (score 0.0–0.39)

=== OUTPUT FORMAT ===
Respond with ONLY this JSON object (no markdown, no extra text):
{{
  "verdict": "GROUNDED" | "PARTIAL" | "UNGROUNDED",
  "score": <float 0.0 to 1.0>,
  "unsupported_claims": ["unsupported statement 1 in English", "unsupported statement 2 in English"],
  "reasoning": "<one sentence overall explanation>",
  "claims": [
    {{
      "statement": "<exact or clean statement sentence from the generated answer>",
      "verdict": "GROUNDED" | "PARTIAL" | "UNGROUNDED",
      "key_citation_line": "<exact or close text quote snippet from the context chunk supporting/refuting this statement>",
      "reasoning": "<one sentence rationale for this specific statement>"
    }}
  ]
}}"""


def _parse_grounding_response(raw: str) -> dict:
    """
    Robustly extract JSON from LLM response.
    Handles markdown code fences and stray text.
    """
    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Extract first {...} block
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1].strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse grounding JSON from response: {raw[:300]}")


# ── Provider implementations ───────────────────────────────────────────────────
def _check_with_gemini(prompt: str) -> dict:
    """Call Gemini (primary provider) with retry backoff."""
    from google import genai
    from google.genai import types
    import time

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not _is_valid_key(gemini_key):
        raise ValueError("GEMINI_API_KEY is not set or is a placeholder.")

    logger.info("[GroundingChecker] Calling Gemini (%s)...", GEMINI_GROUNDING_MODEL)
    client = genai.Client(api_key=gemini_key)

    max_retries = 3
    delay = 1.0
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_GROUNDING_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            raw_text = response.text
            result   = _parse_grounding_response(raw_text)
            logger.info("[GroundingChecker] Gemini responded | verdict=%s score=%s",
                        result.get("verdict"), result.get("score"))
            return result
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                logger.warning(
                    "[GroundingChecker] Gemini call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt, max_retries, e, delay
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error("[GroundingChecker] Gemini call failed permanently after %d attempts.", max_retries)

    raise last_err


def _check_with_nvidia_model(prompt: str, model_name: str) -> dict:
    """Call a specific NVIDIA integrate API model with retry backoff."""
    import time
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if not _is_valid_key(nvidia_key):
        raise ValueError("NVIDIA_API_KEY is not set or is a placeholder.")

    logger.info("[GroundingChecker] Calling NVIDIA model (%s)...", model_name)

    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":           model_name,
        "messages":        [{"role": "user", "content": prompt}],
        "max_tokens":      4096,
        "temperature":     0.0,
        "top_p":           1.0,
        "stream":          False,
        "response_format": {"type": "json_object"},
    }

    max_retries = 2
    delay = 1.0
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                NVIDIA_GROUNDING_URL,
                headers=headers,
                json=payload,
                timeout=GROUNDING_TIMEOUT_SEC,
            )
            response.raise_for_status()

            raw_text = response.json()["choices"][0]["message"]["content"]
            result   = _parse_grounding_response(raw_text)
            logger.info("[GroundingChecker] NVIDIA %s responded | verdict=%s score=%s",
                        model_name, result.get("verdict"), result.get("score"))
            return result
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                logger.warning(
                    "[GroundingChecker] NVIDIA model %s call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    model_name, attempt, max_retries, e, delay
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error("[GroundingChecker] NVIDIA model %s call failed permanently after %d attempts.", model_name, max_retries)

    raise last_err


# ── Main checker class ─────────────────────────────────────────────────────────
class GroundingChecker:
    """
    Faithfulness checker using Gemini → NVIDIA cloud models.

    Usage:
        checker = GroundingChecker()
        result  = checker.check(answer=answer, chunks=chunk_texts, query=query)
        # result is a GroundingResult dataclass
    """

    def check(
        self,
        answer:  str,
        chunks:  List[str],
        query:   str,
    ) -> GroundingResult:
        """
        Run the grounding check.

        Parameters
        ----------
        answer  : The full generated LLM answer.
        chunks  : List of raw chunk texts that were fed to the LLM as context.
        query   : The original user question.

        Returns
        -------
        GroundingResult — always returns (never raises); on failure verdict="UNCHECKED".
        """
        grounding_enabled = os.getenv("GROUNDING_ENABLED", "true").lower() == "true"
        if not grounding_enabled:
            logger.info("[GroundingChecker] Grounding disabled via GROUNDING_ENABLED=false")
            return GroundingResult(
                verdict="UNCHECKED",
                reasoning="Grounding check is disabled.",
                provider="none",
            )

        if not answer or not answer.strip():
            return GroundingResult(
                verdict="UNCHECKED",
                reasoning="No answer to verify.",
                provider="none",
            )

        if not chunks:
            return GroundingResult(
                verdict="UNCHECKED",
                reasoning="No retrieved chunks available for grounding.",
                provider="none",
            )

        context_text = "\n\n---\n\n".join(chunks)
        if len(context_text) > MAX_CONTEXT_CHARS:
            cutoff = context_text.rfind('\n', 0, MAX_CONTEXT_CHARS)
            context_text = context_text[:cutoff] if cutoff != -1 else context_text[:MAX_CONTEXT_CHARS]

        cleaned_answer = _FILE_CITATION_RE.sub(' ', answer)

        if len(cleaned_answer) > MAX_ANSWER_CHARS:
            cutoff = cleaned_answer.rfind('.', 0, MAX_ANSWER_CHARS)
            answer_text = cleaned_answer[:cutoff + 1] if cutoff != -1 else cleaned_answer[:MAX_ANSWER_CHARS]
        else:
            answer_text = cleaned_answer

        prompt = _build_grounding_prompt(query, context_text, answer_text)

        raw_result   = None
        provider_used = None

        providers = [
            ("gemini", lambda: _check_with_gemini(prompt)),
        ]

        for provider_name, checker_func in providers:
            try:
                raw_result = checker_func()
                provider_used = provider_name
                logger.info("✅ [GroundingChecker] %s grounding succeeded.", provider_name)
                break
            except Exception as e:
                logger.warning("⚠️  [GroundingChecker] %s failed: %s", provider_name, e)

        if not raw_result:
            logger.error("❌ [GroundingChecker] All grounding providers failed.")
            return GroundingResult(
                verdict="UNCHECKED",
                reasoning="All cloud grounding providers failed.",
                provider="none",
            )

        # ── Validate and build result ──────────────────────────────────────────
        try:
            verdict = raw_result.get("verdict", "UNCHECKED").upper()
            if verdict not in ("GROUNDED", "PARTIAL", "UNGROUNDED"):
                verdict = "UNCHECKED"

            score = float(raw_result.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]

            # Fix #4: enforce verdict/score consistency so LLM can't return GROUNDED + 0.3
            if verdict == "GROUNDED" and score < 0.8:
                score = max(score, 0.8)
            elif verdict == "PARTIAL" and not (0.4 <= score <= 0.79):
                score = 0.6
            elif verdict == "UNGROUNDED" and score > 0.39:
                score = min(score, 0.39)

            unsupported = raw_result.get("unsupported_claims", [])
            if not isinstance(unsupported, list):
                unsupported = [str(unsupported)]

            claims = raw_result.get("claims", [])
            if not isinstance(claims, list):
                claims = []

            # Fix #8: clamp reasoning to first sentence only
            raw_reasoning = str(raw_result.get("reasoning", "")).strip()
            first_sentence_end = raw_reasoning.find('.')
            if first_sentence_end != -1 and first_sentence_end < len(raw_reasoning) - 1:
                reasoning = raw_reasoning[:first_sentence_end + 1]
            else:
                reasoning = raw_reasoning or "No reasoning provided."

            return GroundingResult(
                verdict=verdict,
                score=score,
                unsupported_claims=unsupported,
                claims=claims,
                reasoning=reasoning,
                provider=provider_used,
            )

        except Exception as parse_err:
            logger.error("[GroundingChecker] Failed to parse result: %s | raw=%s", parse_err, raw_result)
            return GroundingResult(
                verdict="UNCHECKED",
                reasoning=f"Result parsing failed: {parse_err}",
                provider=provider_used or "none",
            )


# ── Thread-safe singleton factory (fix #6) ────────────────────────────────────
_checker_instance: GroundingChecker | None = None
_checker_lock = threading.Lock()


def get_grounding_checker() -> GroundingChecker:
    global _checker_instance
    if _checker_instance is None:
        with _checker_lock:
            if _checker_instance is None:  # double-checked locking
                _checker_instance = GroundingChecker()
                logger.info(
                    "[GroundingChecker] Initialized | gemini_model=%s nvidia_model=%s",
                    GEMINI_GROUNDING_MODEL, NVIDIA_GROUNDING_MODEL,
                )
    return _checker_instance
