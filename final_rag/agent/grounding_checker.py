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
from dataclasses import dataclass, field
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agent.grounding_checker")

# ── Constants ──────────────────────────────────────────────────────────────────
NVIDIA_GROUNDING_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_GROUNDING_MODEL = os.getenv("NVIDIA_GROUNDING_MODEL", "z-ai/glm-5.2")
GEMINI_GROUNDING_MODEL = os.getenv("GEMINI_GROUNDING_MODEL", "gemini-2.0-flash")
GROUNDING_TIMEOUT_SEC  = int(os.getenv("GROUNDING_TIMEOUT_SEC", "30"))
GROUNDING_ENABLED      = os.getenv("GROUNDING_ENABLED", "true").lower() == "true"

# Maximum characters sent to the checker to stay well within context limits
MAX_ANSWER_CHARS  = 3000
MAX_CONTEXT_CHARS = 8000


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class GroundingResult:
    verdict:            str          = "UNCHECKED"   # GROUNDED | PARTIAL | UNGROUNDED | UNCHECKED
    score:              float        = 0.0
    unsupported_claims: List[str]    = field(default_factory=list)
    reasoning:          str          = "Grounding check was not performed."
    provider:           str          = "none"

    def to_dict(self) -> dict:
        return {
            "verdict":            self.verdict,
            "score":              self.score,
            "unsupported_claims": self.unsupported_claims,
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
    Strict faithfulness evaluation prompt.
    The LLM must respond with a single JSON object — no markdown, no prose.
    """
    return f"""You are a strict faithfulness evaluator for a Retrieval-Augmented Generation (RAG) system.

TASK: Determine whether the GENERATED ANSWER is faithful to and fully supported by the PROVIDED CONTEXT.

=== USER QUESTION ===
{query}

=== PROVIDED CONTEXT (retrieved document chunks) ===
{context_text}

=== GENERATED ANSWER ===
{answer}

=== EVALUATION RULES ===
1. A claim is GROUNDED if it is directly stated or clearly implied by the context above.
2. A claim is UNGROUNDED if it is absent from the context, is a general assumption, or contradicts the context.
3. Ignore differences in phrasing or formatting — focus purely on factual faithfulness.
4. If the answer says "Not found in uploaded documents." or similar — it IS grounded (correct refusal).
5. Citation markers like [filename, Page X] are GROUNDED if the fact they cite appears in context.

=== VERDICT SCALE ===
- GROUNDED   : All claims are directly supported by the context (score 0.8–1.0)
- PARTIAL    : Most claims are supported but 1–3 minor unsupported claims exist (score 0.4–0.79)
- UNGROUNDED : Multiple or significant claims are NOT in the context (score 0.0–0.39)

=== OUTPUT ===
Respond with ONLY this JSON object (no markdown, no extra text):
{{
  "verdict": "GROUNDED" | "PARTIAL" | "UNGROUNDED",
  "score": <float 0.0 to 1.0>,
  "unsupported_claims": ["claim1", "claim2"],
  "reasoning": "<one sentence explanation>"
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
    """Call Gemini (primary provider)."""
    from google import genai
    from google.genai import types

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not _is_valid_key(gemini_key):
        raise ValueError("GEMINI_API_KEY is not set or is a placeholder.")

    logger.info("[GroundingChecker] Calling Gemini (%s)...", GEMINI_GROUNDING_MODEL)
    client = genai.Client(api_key=gemini_key)

    response = client.models.generate_content(
        model=GEMINI_GROUNDING_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    raw_text = response.text
    result   = _parse_grounding_response(raw_text)
    logger.info("[GroundingChecker] Gemini responded | verdict=%s score=%s",
                result.get("verdict"), result.get("score"))
    return result


def _check_with_nvidia_model(prompt: str, model_name: str) -> dict:
    """Call a specific NVIDIA integrate API model."""
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
        "max_tokens":      1024,
        "temperature":     0.0,
        "top_p":           1.0,
        "stream":          False,
        "response_format": {"type": "json_object"},
    }

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


# ── Main checker class ─────────────────────────────────────────────────────────
class GroundingChecker:
    """
    Faithfulness checker using Gemini → NVIDIA Multi-Model fallbacks.

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
        if not GROUNDING_ENABLED:
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

        # Truncate to stay within API context limits
        context_text = "\n\n---\n\n".join(chunks)
        context_text = context_text[:MAX_CONTEXT_CHARS]
        answer_text  = answer[:MAX_ANSWER_CHARS]

        prompt = _build_grounding_prompt(query, context_text, answer_text)

        # ── Try Gemini first, fall back to NVIDIA models list ──
        raw_result   = None
        provider_used = None

        try:
            raw_result    = _check_with_gemini(prompt)
            provider_used = "gemini"
            logger.info("✅ [GroundingChecker] Gemini grounding succeeded.")
        except Exception as gemini_err:
            logger.warning("⚠️  [GroundingChecker] Gemini failed: %s — trying NVIDIA NIM fallbacks...", gemini_err)
            
            nvidia_models = ["z-ai/glm-5.2", "minimaxai/minimax-m3", "nvidia/nemotron-3-ultra-550b-a55b"]
            for model_name in nvidia_models:
                try:
                    raw_result    = _check_with_nvidia_model(prompt, model_name)
                    provider_used = f"nvidia_{model_name.replace('/', '_')}"
                    logger.info("✅ [GroundingChecker] NVIDIA model %s succeeded.", model_name)
                    break
                except Exception as fallback_e:
                    logger.warning("⚠️  [GroundingChecker] NVIDIA model %s failed: %s", model_name, fallback_e)
            
            if not raw_result:
                logger.error("❌ [GroundingChecker] Both Gemini and all NVIDIA models failed.")
                return GroundingResult(
                    verdict="UNCHECKED",
                    reasoning=f"Both providers failed. Gemini: {gemini_err} | NVIDIA models failed.",
                    provider="none",
                )

        # ── Validate and build result ──────────────────────────────────────────
        try:
            verdict = raw_result.get("verdict", "UNCHECKED").upper()
            if verdict not in ("GROUNDED", "PARTIAL", "UNGROUNDED"):
                verdict = "UNCHECKED"

            score = float(raw_result.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]

            unsupported = raw_result.get("unsupported_claims", [])
            if not isinstance(unsupported, list):
                unsupported = [str(unsupported)]

            reasoning = str(raw_result.get("reasoning", "")).strip() or "No reasoning provided."

            return GroundingResult(
                verdict=verdict,
                score=score,
                unsupported_claims=unsupported,
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


# ── Singleton factory ──────────────────────────────────────────────────────────
_checker_instance: GroundingChecker | None = None


def get_grounding_checker() -> GroundingChecker:
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = GroundingChecker()
        logger.info("[GroundingChecker] Initialized | gemini_model=%s nvidia_model=%s enabled=%s",
                    GEMINI_GROUNDING_MODEL, NVIDIA_GROUNDING_MODEL, GROUNDING_ENABLED)
    return _checker_instance
