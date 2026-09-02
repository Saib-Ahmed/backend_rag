"""
final_rag/agent/claude_client.py

Claude API client (single model: claude-sonnet-5 or configured model).
Note: Sonnet 5 uses adaptive thinking by default and does not accept a
custom `temperature` parameter — the API rejects it with a 400 if sent.
So temperature is intentionally NOT included in the payload.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

import requests

try:
    import final_rag.config as config
except ImportError:
    from .. import config

logger = logging.getLogger("agent.claude_client")

ENDPOINT = "https://api.anthropic.com/v1/messages"


class ClaudeGenerationError(Exception):
    """Raised when the Claude API call fails (hard failure)."""


class ClaudeClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or getattr(config, "CLAUDE_MODEL", "claude-sonnet-5")
        self.api_key = api_key or getattr(config, "CLAUDE_API_KEY", "")
        if not self.api_key:
            logger.warning("[ClaudeClient] CLAUDE_API_KEY not set — calls will fail.")

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": getattr(config, "ANTHROPIC_VERSION", "2023-06-01"),
            "content-type": "application/json",
        }

    # ── Non-streaming: used by QueryCleaner ────────────────────────────

    def generate(
        self,
        prompt:      str,
        system:      str = "",
        temperature: float = 0.0,   # kept in signature for compat, NOT sent to API
        max_tokens:  int | None = None,
    ) -> str:
        """
        Single call to Claude.
        Raises ClaudeGenerationError on any hard failure.
        """
        if max_tokens is None:
            max_tokens = getattr(config, "CLAUDE_GENERATE_MAX_TOKENS", 3000)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                ENDPOINT,
                headers=self._headers(),
                json=payload,
                timeout=getattr(config, "CLAUDE_HTTP_TIMEOUT_SEC", 120),
            )
            if not resp.ok:
                logger.error("[ClaudeClient] Status: %s | Body: %s", resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()

            content_blocks = data.get("content", [])
            text = "".join(
                block.get("text", "") for block in content_blocks
                if block.get("type") == "text"
            ).strip()

            if not text:
                raise ValueError("Empty content in Claude response")

            logger.info("[ClaudeClient] generate() succeeded | model=%s", self.model)
            return text

        except Exception as e:
            logger.error("[ClaudeClient] generate() failed | model=%s | error=%s", self.model, e)
            raise ClaudeGenerationError(f"Claude generate() failed: {e}") from e

    # ── Streaming: used by Orchestrator ───────────────────────────────

    def chat_stream(
        self,
        messages:    list[dict],
        temperature: float = 1.0,   # kept in signature for compat, NOT sent to API
        max_tokens:  int | None = None,
        system:      str | None = None,
    ) -> Generator[str, None, None]:
        """
        Streams tokens from Claude.
        """
        if max_tokens is None:
            max_tokens = getattr(config, "CLAUDE_CHAT_STREAM_MAX_TOKENS", 9000)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True,
        }
        if system:
            payload["system"] = system

        first_token_yielded = False

        try:
            with requests.post(
                ENDPOINT,
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=getattr(config, "CLAUDE_HTTP_TIMEOUT_SEC", 120),
            ) as resp:
                if not resp.ok:
                    logger.error("[ClaudeClient] Status: %s | Body: %s", resp.status_code, resp.text)
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("[ClaudeClient] Skipping unparsable SSE line: %s", data_str[:200])
                        continue

                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        token = delta.get("text")
                        if token:
                            first_token_yielded = True
                            yield token

                    elif chunk.get("type") == "message_stop":
                        break

            logger.info("[ClaudeClient] chat_stream() completed | model=%s", self.model)

        except Exception as e:
            if first_token_yielded:
                logger.error("[ClaudeClient] chat_stream() failed MID-STREAM: %s", e)
            else:
                logger.error("[ClaudeClient] chat_stream() failed before first token: %s", e)
            raise ClaudeGenerationError(f"Claude chat_stream() failed: {e}") from e
