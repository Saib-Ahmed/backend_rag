"""
agent/nvidia_client.py
NVIDIA NIM API Client for Text Generation and Query Cleaning.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Generator

import requests
import final_rag.config as config

logger = logging.getLogger("agent.nvidia_client")

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


class NvidiaGenerationError(Exception):
    """Exception raised when NVIDIA generation fails."""
    pass


def _is_valid_key(key: str) -> bool:
    if not key or not key.strip():
        return False
    key_lower = key.lower()
    for placeholder in ("placeholder", "your_gemini", "your_nvidia", "api_key_here", "key_here"):
        if placeholder in key_lower:
            return False
    return True


class NvidiaClient:

    def __init__(self):
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        logger.info("NvidiaClient initialized")

    def _headers(self) -> dict:
        key = os.environ.get("NVIDIA_API_KEY", self.api_key)
        if not _is_valid_key(key):
            raise NvidiaGenerationError("NVIDIA_API_KEY is not set or is a placeholder.")
        return {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        system:      str,
        prompt:      str,
        model:       str,
        temperature: float = 0.0,
    ) -> str:
        """
        Performs a non-streaming chat completion request (primarily used by QueryCleaner).
        """
        payload = {
            "model":      model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens":  1024,
            "stream":      False
        }

        try:
            resp = requests.post(ENDPOINT, headers=self._headers(), json=payload, timeout=60)
            if not resp.ok:
                logger.error("[NvidiaClient] generate() HTTP %s | %s", resp.status_code, resp.text[:300])
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            logger.info("[NvidiaClient] generate() succeeded | model=%s", model)
            return text
        except Exception as e:
            logger.error("[NvidiaClient] generate() failed | model=%s | error=%s", model, e)
            raise NvidiaGenerationError(f"Nvidia generate() failed: {e}") from e

    def chat_stream(
        self,
        model:       str,
        messages:    list[dict],
        temperature: float = 0.25,
        max_tokens:  int | None = None,
    ) -> Generator[str, None, None]:
        """
        Streams tokens from NVIDIA NIM API (primarily used by Orchestrator).
        """
        if max_tokens is None:
            max_tokens = getattr(config, "MAX_TOKENS", 4096)

        payload = {
            "model":      model,
            "max_tokens": max_tokens,
            "messages":   messages,
            "temperature": temperature,
            "stream":     True,
        }

        first_token_yielded = False

        try:
            with requests.post(
                ENDPOINT, headers=self._headers(), json=payload, stream=True, timeout=120,
            ) as resp:
                if not resp.ok:
                    logger.error("[NvidiaClient] chat_stream() HTTP %s | %s", resp.status_code, resp.text[:300])
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    
                    raw_line = raw_line.strip()
                    if not raw_line.startswith("data:"):
                        continue
                        
                    data_str = raw_line[len("data:"):].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                        
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("[NvidiaClient] Skipping unparsable SSE line: %s", data_str[:200])
                        continue

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            first_token_yielded = True
                            yield token

            logger.info("[NvidiaClient] chat_stream() completed | model=%s", model)

        except Exception as e:
            if first_token_yielded:
                logger.error(
                    "[NvidiaClient] chat_stream() failed MID-STREAM | model=%s | error=%s",
                    model, e,
                )
            else:
                logger.error(
                    "[NvidiaClient] chat_stream() failed before first token | model=%s | error=%s",
                    model, e,
                )
            raise NvidiaGenerationError(f"Nvidia chat_stream() failed: {e}") from e
