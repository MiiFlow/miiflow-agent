"""Tests for Retry-After extraction on rate-limit errors.

Regression: provider clients used ``getattr(e.response.headers, "retry-after",
None)`` — an *attribute* lookup on the headers object, which always returned
None — so ``RateLimitError.retry_after`` was never populated and nothing could
do rate-limit-aware backoff. Extraction now lives in one place
(``retry_after_seconds``) and ``RateLimitError`` applies it automatically to
the wrapped SDK error.
"""

import httpx

from miiflow_agent.core.exceptions import RateLimitError, retry_after_seconds


class _FakeSDKError(Exception):
    """Shaped like an OpenAI/Anthropic/Groq SDK error: carries .response."""

    def __init__(self, headers=None):
        super().__init__("rate limited")
        if headers is not None:
            self.response = httpx.Response(
                429, headers=headers, request=httpx.Request("POST", "http://t")
            )


def test_extracts_delta_seconds():
    assert retry_after_seconds(_FakeSDKError({"retry-after": "7"})) == 7.0


def test_header_lookup_is_case_insensitive():
    assert retry_after_seconds(_FakeSDKError({"Retry-After": "1.5"})) == 1.5


def test_missing_header_returns_none():
    assert retry_after_seconds(_FakeSDKError({})) is None


def test_error_without_response_returns_none():
    assert retry_after_seconds(_FakeSDKError(headers=None)) is None
    assert retry_after_seconds(None) is None


def test_http_date_form_is_ignored_not_crashed():
    err = _FakeSDKError({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert retry_after_seconds(err) is None


def test_negative_value_is_rejected():
    assert retry_after_seconds(_FakeSDKError({"retry-after": "-3"})) is None


def test_rate_limit_error_auto_populates_from_original_error():
    sdk_err = _FakeSDKError({"retry-after": "12"})
    err = RateLimitError("limited", "openai", original_error=sdk_err)
    assert err.retry_after == 12.0


def test_explicit_retry_after_wins_over_header():
    sdk_err = _FakeSDKError({"retry-after": "12"})
    err = RateLimitError("limited", "openai", retry_after=3.0, original_error=sdk_err)
    assert err.retry_after == 3.0


def test_rate_limit_error_without_original_error_stays_none():
    assert RateLimitError("limited", "anthropic").retry_after is None
