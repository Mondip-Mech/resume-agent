"""
tests/test_llm_client.py
─────────────────────────
Unit tests for core/llm_client.py.

All tests bypass __init__ to avoid requiring a real NVIDIA_API_KEY.
No network calls are made.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMClient, _is_retryable

# ─── Helper to build a LLMClient without calling __init__ ────────────────────

def _make_client() -> LLMClient:
    """Instantiate LLMClient without touching OpenAI or env vars."""
    client = LLMClient.__new__(LLMClient)
    client.model_name = "meta/llama-3.1-8b-instruct"
    client.max_tokens = 512
    client.temperature = 0.2
    client._last_call_time = 0.0
    client._total_input_tokens = 0
    client._total_output_tokens = 0
    client.client = MagicMock()
    return client


# ─── _is_retryable ────────────────────────────────────────────────────────────

class TestIsRetryable:

    def test_rate_limit_error_is_retryable(self):
        from openai import RateLimitError
        exc = RateLimitError("rate limited", response=MagicMock(status_code=429), body={})
        assert _is_retryable(exc) is True

    def test_bad_request_400_is_not_retryable(self):
        from openai import BadRequestError
        exc = BadRequestError("DEGRADED", response=MagicMock(status_code=400), body={})
        assert _is_retryable(exc) is False

    def test_server_error_500_is_retryable(self):
        from openai import APIStatusError
        exc = APIStatusError("server error", response=MagicMock(status_code=500), body={})
        assert _is_retryable(exc) is True

    def test_api_status_400_is_not_retryable(self):
        from openai import APIStatusError
        exc = APIStatusError("bad request", response=MagicMock(status_code=400), body={})
        assert _is_retryable(exc) is False

    def test_api_status_503_is_retryable(self):
        from openai import APIStatusError
        exc = APIStatusError("service unavailable", response=MagicMock(status_code=503), body={})
        assert _is_retryable(exc) is True

    def test_generic_exception_is_not_retryable(self):
        assert _is_retryable(ValueError("oops")) is False


# ─── _sanitize ───────────────────────────────────────────────────────────────

class TestSanitize:

    def test_removes_null_bytes(self):
        assert "\x00" not in LLMClient._sanitize("hello\x00world")

    def test_removes_c0_control_chars(self):
        dirty = "".join(chr(i) for i in range(0, 32))  # \x00 through \x1f
        cleaned = LLMClient._sanitize(dirty)
        # Tab (\x09), newline (\x0a), CR (\x0d) should survive
        assert "\t" in cleaned
        assert "\n" in cleaned
        # Null byte and other controls should be gone
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned

    def test_preserves_normal_text(self):
        text = "Hello World\nLine 2\tTabbed"
        assert LLMClient._sanitize(text) == text

    def test_preserves_unicode(self):
        text = "Résumé: naïve café"
        assert LLMClient._sanitize(text) == text


# ─── _extract_json ────────────────────────────────────────────────────────────

class TestExtractJson:

    def test_plain_json(self):
        result = LLMClient._extract_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert LLMClient._extract_json(raw) == {"key": "value"}

    def test_strips_plain_fences(self):
        raw = "```\n{\"key\": 1}\n```"
        assert LLMClient._extract_json(raw) == {"key": 1}

    def test_extracts_json_from_prose(self):
        raw = 'Here is the result:\n{"score": 85, "passed": true}\nDone.'
        result = LLMClient._extract_json(raw)
        assert result["score"] == 85
        assert result["passed"] is True

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            LLMClient._extract_json("This has no JSON at all.")

    def test_nested_json(self):
        raw = '{"issues": [{"section": "exp", "severity": "critical"}]}'
        result = LLMClient._extract_json(raw)
        assert result["issues"][0]["severity"] == "critical"


# ─── _repair_truncated_json ───────────────────────────────────────────────────

class TestRepairTruncatedJson:

    def test_closes_open_brace(self):
        truncated = '{"key": "value"'
        repaired = LLMClient._repair_truncated_json(truncated)
        import json
        result = json.loads(repaired)
        assert result["key"] == "value"

    def test_closes_open_array(self):
        truncated = '{"items": ["a", "b"'
        repaired = LLMClient._repair_truncated_json(truncated)
        import json
        result = json.loads(repaired)
        assert "a" in result["items"]

    def test_handles_no_opening_brace(self):
        result = LLMClient._repair_truncated_json("no braces here")
        assert result == "no braces here"

    def test_complete_json_unchanged(self):
        complete = '{"key": "value"}'
        repaired = LLMClient._repair_truncated_json(complete)
        import json
        assert json.loads(repaired) == {"key": "value"}


# ─── call_json (with mocked call) ─────────────────────────────────────────────

class TestCallJson:

    def test_returns_parsed_dict_on_valid_response(self):
        client = _make_client()
        with patch.object(client, "call", return_value='{"gaps": [], "score": 72}'):
            result = client.call_json("some prompt")
        assert result == {"gaps": [], "score": 72}

    def test_retries_with_cleanup_on_bad_first_response(self):
        client = _make_client()
        # First call returns prose, second returns valid JSON
        with patch.object(client, "call", side_effect=[
            "Here is your answer: not JSON",
            '{"gaps": [], "score": 65}',
        ]):
            result = client.call_json("prompt")
        assert result["score"] == 65

    def test_raises_after_two_bad_responses(self):
        client = _make_client()
        with patch.object(client, "call", return_value="still no JSON here"):
            with pytest.raises(ValueError):
                client.call_json("prompt")


# ─── token_usage ─────────────────────────────────────────────────────────────

class TestTokenUsage:

    def test_usage_keys_exist(self):
        client = _make_client()
        usage = client.token_usage
        assert "input" in usage
        assert "output" in usage
        assert "total" in usage

    def test_total_equals_sum(self):
        client = _make_client()
        client._total_input_tokens = 150
        client._total_output_tokens = 300
        assert client.token_usage["total"] == 450
        assert client.token_usage["input"] == 150
        assert client.token_usage["output"] == 300
