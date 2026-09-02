"""Tests for transport retry on the streaming path.

The ReAct loop runs exclusively through ``LLMClient.astream_chat``, which
historically had no retry at all (tenacity only wrapped the providers'
non-streaming ``achat``), so a single transient 429/529 killed the whole run.
The retry added here applies only until the first chunk arrives — after that a
blind reopen would replay deltas the consumer already saw — and is targeted:
deterministic failures (auth, bad request) fail fast.
"""

import asyncio

import pytest
from unittest.mock import MagicMock

from miiflow_agent.core.client import LLMClient
from miiflow_agent.core.exceptions import (
    AuthenticationError,
    ModelError,
    ProviderError,
    RateLimitError,
    is_retryable_error,
    retry_delay_seconds,
)
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.streaming import StreamChunk


def _chunk(text, finish=None):
    return StreamChunk(
        content=text, delta=text, finish_reason=finish, usage=None, tool_calls=None
    )


def _make_llm_client(stream_fn):
    provider = MagicMock()
    provider.provider_name = "testprov"
    provider.model = "test-model"
    provider.astream_chat = stream_fn
    return LLMClient(client=provider)


def _messages():
    return [Message(role=MessageRole.USER, content="hi")]


@pytest.fixture
def no_sleep(monkeypatch):
    """Capture retry delays instead of actually sleeping."""
    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds, *args, **kwargs):
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


async def _collect(client):
    return [c async for c in client.astream_chat(_messages())]


async def test_retries_open_failure_then_succeeds(no_sleep):
    calls = {"n": 0}

    async def flaky_stream(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RateLimitError("slow down", "testprov")
        yield _chunk("hello")
        yield _chunk(" world", finish="stop")

    client = _make_llm_client(flaky_stream)
    chunks = await _collect(client)

    assert [c.delta for c in chunks] == ["hello", " world"]
    assert calls["n"] == 3
    assert len(no_sleep) == 2


async def test_no_retry_after_first_chunk(no_sleep):
    calls = {"n": 0}

    async def midstream_failure(messages, **kwargs):
        calls["n"] += 1
        yield _chunk("partial")
        raise RateLimitError("mid-stream blip", "testprov")

    client = _make_llm_client(midstream_failure)
    with pytest.raises(RateLimitError):
        await _collect(client)

    # One attempt only: content already reached the consumer, so a blind
    # reopen would duplicate it.
    assert calls["n"] == 1
    assert no_sleep == []


async def test_deterministic_errors_fail_fast(no_sleep):
    calls = {"n": 0}

    async def auth_failure(messages, **kwargs):
        calls["n"] += 1
        raise AuthenticationError("bad key", "testprov")
        yield  # pragma: no cover — makes this an async generator

    client = _make_llm_client(auth_failure)
    with pytest.raises(AuthenticationError):
        await _collect(client)

    assert calls["n"] == 1
    assert no_sleep == []


async def test_exhausted_retries_raise(no_sleep):
    calls = {"n": 0}

    async def always_limited(messages, **kwargs):
        calls["n"] += 1
        raise RateLimitError("slow down", "testprov")
        yield  # pragma: no cover

    client = _make_llm_client(always_limited)
    with pytest.raises(RateLimitError):
        await _collect(client)

    # 1 initial + _max_stream_retries (default 3)
    assert calls["n"] == 4
    assert len(no_sleep) == 3


async def test_server_retry_after_floors_the_delay(no_sleep):
    calls = {"n": 0}

    async def limited_with_hint(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("slow down", "testprov", retry_after=12.0)
        yield _chunk("ok", finish="stop")

    client = _make_llm_client(limited_with_hint)
    chunks = await _collect(client)

    assert [c.delta for c in chunks] == ["ok"]
    assert len(no_sleep) == 1
    assert no_sleep[0] >= 12.0


class TestIsRetryableError:
    def test_rate_limit_is_retryable(self):
        assert is_retryable_error(RateLimitError("x", "p")) is True

    def test_auth_is_not(self):
        assert is_retryable_error(AuthenticationError("x", "p")) is False

    def test_model_error_is_not(self):
        # ModelError wraps bad-request/context-length: deterministic.
        assert is_retryable_error(ModelError("prompt is too long", "m")) is False

    def test_provider_error_by_status(self):
        class _SDKError(Exception):
            def __init__(self, status):
                self.status_code = status

        overloaded = ProviderError("overloaded", "p", original_error=_SDKError(529))
        bad_request = ProviderError("bad", "p", original_error=_SDKError(400))
        assert is_retryable_error(overloaded) is True
        assert is_retryable_error(bad_request) is False

    def test_provider_error_wrapping_connection_failure_is_retryable(self):
        # Anthropic's catch-all wraps APIConnectionError into ProviderError
        # with no status code; the class name is the only transience signal.
        class APIConnectionError(Exception):
            pass

        err = ProviderError("boom", "p", original_error=APIConnectionError())
        assert is_retryable_error(err) is True

    def test_asyncio_timeout_is_retryable(self):
        assert is_retryable_error(asyncio.TimeoutError()) is True

    def test_httpx_transport_error_is_retryable(self):
        import httpx

        assert is_retryable_error(httpx.ConnectError("boom")) is True

    def test_unknown_error_is_not(self):
        assert is_retryable_error(ValueError("boom")) is False
        assert is_retryable_error(None) is False


class TestStreamInactivityTimeout:
    """A stream that opens and then stalls used to hang the run forever —
    the provider timeout guards stream-open only, and MaxTimeCondition
    cannot interrupt a hung await between steps."""

    async def test_midstream_stall_raises_timeout(self):
        from miiflow_agent.core.exceptions import TimeoutError as MiiflowTimeout

        async def stalling_stream(messages, **kwargs):
            yield _chunk("partial")
            await asyncio.Event().wait()  # stalls forever

        client = _make_llm_client(stalling_stream)
        client._stream_inactivity_timeout = 0.05

        with pytest.raises(MiiflowTimeout):
            await _collect(client)

    async def test_stall_before_first_chunk_is_retried(self, no_sleep):
        calls = {"n": 0}

        async def stall_once_then_succeed(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.Event().wait()  # stalls before any chunk
            yield _chunk("ok", finish="stop")

        client = _make_llm_client(stall_once_then_succeed)
        client._stream_inactivity_timeout = 0.05

        chunks = await _collect(client)
        assert [c.delta for c in chunks] == ["ok"]
        assert calls["n"] == 2

    async def test_zero_disables_the_guard(self):
        async def normal_stream(messages, **kwargs):
            yield _chunk("a")
            yield _chunk("b", finish="stop")

        client = _make_llm_client(normal_stream)
        client._stream_inactivity_timeout = 0

        chunks = await _collect(client)
        assert [c.delta for c in chunks] == ["a", "b"]


class TestStreamTimingDecomposition:
    """latency_ms alone cannot localize a slow step; POST_CALL now carries
    build/ttft/stream splits and the transport retry count."""

    async def test_post_call_carries_timing_fields(self):
        from miiflow_agent.core.callbacks import (
            CallbackEventType,
            get_global_registry,
        )

        async def slowish_stream(messages, **kwargs):
            await asyncio.sleep(0.02)  # simulated server think time (ttft)
            yield _chunk("hello")
            await asyncio.sleep(0.02)  # generation time (stream)
            yield _chunk(" world", finish="stop")

        client = _make_llm_client(slowish_stream)

        events = []
        get_global_registry().register(
            CallbackEventType.POST_CALL, lambda e: events.append(e)
        )
        try:
            await _collect(client)
            await asyncio.sleep(0)  # one tick for emit_sync scheduling safety
        finally:
            get_global_registry().clear(CallbackEventType.POST_CALL)

        assert len(events) == 1
        event = events[0]
        assert event.event_id and event.event_id.startswith("llm_")
        assert event.ttft_ms is not None and event.ttft_ms >= 15
        assert event.stream_ms is not None and event.stream_ms >= 15
        assert event.request_build_ms is not None
        assert event.transport_retries == 0

    async def test_retries_counted_and_ttft_excludes_backoff(self, no_sleep):
        from miiflow_agent.core.callbacks import (
            CallbackEventType,
            get_global_registry,
        )

        calls = {"n": 0}

        async def flaky(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RateLimitError("slow down", "testprov")
            yield _chunk("ok", finish="stop")

        client = _make_llm_client(flaky)
        events = []
        get_global_registry().register(
            CallbackEventType.POST_CALL, lambda e: events.append(e)
        )
        try:
            await _collect(client)
        finally:
            get_global_registry().clear(CallbackEventType.POST_CALL)

        assert len(events) == 1
        assert events[0].transport_retries == 1
        # TTFT is measured from the LAST open, so the first attempt and its
        # backoff cannot inflate it.
        assert events[0].ttft_ms < 1_000


class TestRetryDelaySeconds:
    def test_exponential_growth_with_cap(self):
        e = RuntimeError("x")
        d1 = retry_delay_seconds(e, 1)
        d2 = retry_delay_seconds(e, 2)
        d10 = retry_delay_seconds(e, 10)
        assert 1.0 <= d1 <= 1.25
        assert 2.0 <= d2 <= 2.5
        assert d10 <= 30 * 1.25  # capped

    def test_retry_after_is_a_floor(self):
        err = RateLimitError("x", "p", retry_after=20.0)
        assert retry_delay_seconds(err, 1) >= 20.0
