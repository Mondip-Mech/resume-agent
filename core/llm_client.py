"""
core/llm_client.py
───────────────────
Wrapper around the NVIDIA NIM API (OpenAI-compatible, free tier).

NVIDIA NIM free tier: https://build.nvidia.com
  • 40 RPM per model
  • OpenAI-compatible endpoint at https://integrate.api.nvidia.com/v1

Handles:
  • Per-call rate-limiting delay
  • Retry with exponential backoff on rate-limit / server errors
  • Structured JSON extraction
  • System prompt via "system" role message
  • Token usage tracking
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from openai import APIStatusError, BadRequestError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# NVIDIA free-tier: 40 RPM → one call every 1.5 s is safe
_CALL_DELAY = 1.5   # seconds between calls


def _is_retryable(exc: BaseException) -> bool:
    """
    Only retry on rate-limit (429) or server-side errors (5xx).
    Never retry 400 Bad Request — these are permanent failures such as
    a DEGRADED model endpoint and will never recover on their own.
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, BadRequestError):
        return False          # 400 — fail immediately, no point retrying
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500   # 5xx only
    return False


class LLMClient:
    """
    Wrapper around NVIDIA NIM (OpenAI-compatible) with:
      • Rate-limit spacing between calls
      • Auto-retry on 429 / 5xx errors
      • Structured JSON output helper
    """

    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.1-8b-instruct",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
        )
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._last_call_time: float = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    # ─── Core call ────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
    ) -> str:
        """Make a single NIM API call. Returns the text response."""
        self._rate_limit_wait()
        result = self._call_with_retry(prompt, system, max_tokens or self.max_tokens)
        self._last_call_time = time.monotonic()
        return result

    def _rate_limit_wait(self):
        """Sleep just enough to stay within the model's RPM cap."""
        elapsed = time.monotonic() - self._last_call_time
        wait = _CALL_DELAY - elapsed
        if wait > 0:
            logger.debug(f"Rate-limit: sleeping {wait:.1f}s")
            time.sleep(wait)

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip control characters that can break JSON generation in the LLM."""
        # Keep printable ASCII + common Unicode (letters, digits, punctuation).
        # Remove C0/C1 control chars except tab (\x09), newline (\x0a), CR (\x0d).
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=10, max=60),
    )
    def _call_with_retry(self, prompt: str, system: str, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": self._sanitize(system)})
        messages.append({"role": "user", "content": self._sanitize(prompt)})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens or 0
            self._total_output_tokens += response.usage.completion_tokens or 0
            logger.debug(
                f"Tokens: in={response.usage.prompt_tokens} "
                f"out={response.usage.completion_tokens} | "
                f"total={self._total_input_tokens + self._total_output_tokens}"
            )

        return response.choices[0].message.content

    # ─── JSON extraction ──────────────────────────────────────

    def call_json(
        self,
        prompt: str,
        system: str = "",
        schema_hint: str = "",
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call NIM and extract a JSON object from the response.
        Retries once with a stricter cleanup prompt if parsing fails.
        """
        json_instruction = (
            "\n\nIMPORTANT: Your entire response must be a single valid JSON object. "
            "Do not include any text before or after the JSON. "
            "Do not wrap it in markdown code fences."
        )
        if schema_hint:
            json_instruction += f"\n\nExpected JSON shape:\n{schema_hint}"

        raw = self.call(prompt + json_instruction, system=system, max_tokens=max_tokens)

        try:
            return self._extract_json(raw)
        except ValueError:
            fallback_prompt = (
                "The following text should be valid JSON but may not parse correctly. "
                "Extract and return ONLY the JSON object, with no other text:\n\n" + raw
            )
            raw2 = self.call(fallback_prompt, system="You are a JSON extractor.")
            return self._extract_json(raw2)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Strip markdown fences and parse JSON from LLM output.
        Also attempts to repair truncated JSON caused by token-limit cutoffs.
        """
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Find first { ... } block and try to parse it
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 3. Response was truncated mid-JSON — try to close open brackets/strings
        if "{" in text:
            repaired = LLMClient._repair_truncated_json(text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from: {text[:200]}")

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Best-effort repair of a JSON object that was cut off mid-stream."""
        # Take everything from the first { onward
        start = text.find("{")
        if start == -1:
            return text
        text = text[start:]

        # Remove trailing partial string/value (last comma or incomplete token)
        text = re.sub(r',\s*$', '', text.rstrip())

        # Close any unclosed string (odd number of unescaped quotes after last key)
        # Simple heuristic: count open brackets/braces and close them
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False

        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    open_braces += 1
                elif ch == '}':
                    open_braces -= 1
                elif ch == '[':
                    open_brackets += 1
                elif ch == ']':
                    open_brackets -= 1

        # If we're mid-string, close it
        if in_string:
            text += '"'
        # Close open arrays then objects
        text += ']' * max(open_brackets, 0)
        text += '}' * max(open_braces, 0)
        return text

    # ─── Stats ────────────────────────────────────────────────

    @property
    def token_usage(self) -> Dict[str, int]:
        return {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "total": self._total_input_tokens + self._total_output_tokens,
        }
