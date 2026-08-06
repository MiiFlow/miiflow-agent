"""Tests for rate-limit-adaptive parallel tool concurrency."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from miiflow_agent.core.exceptions import RateLimitError
from miiflow_agent.core.react.adaptive_concurrency import (
    AdaptiveConcurrencyLimiter,
    looks_like_rate_limit,
)
from miiflow_agent.core.react.tool_executor import AgentToolExecutor, ToolCall
from miiflow_agent.core.tools import ToolResult


class TestLooksLikeRateLimit:
    def test_typed_exception(self):
        assert looks_like_rate_limit(RateLimitError("x", "p")) is True

    def test_failed_result_with_429_text(self):
        result = ToolResult(
            name="t", input={}, output=None, success=False,
            error="Provider error: 429 Too Many Requests",
        )
        assert looks_like_rate_limit(result) is True

    def test_ordinary_failure_is_not(self):
        result = ToolResult(
            name="t", input={}, output=None, success=False, error="boom"
        )
        assert looks_like_rate_limit(result) is False

    def test_success_never_matches(self):
        result = ToolResult(
            name="t", input={}, output="429 rows returned", success=True
        )
        assert looks_like_rate_limit(result) is False


class TestAdaptiveConcurrencyLimiter:
    async def test_shrinks_on_rate_limit_with_cooldown(self):
        limiter = AdaptiveConcurrencyLimiter(8, shrink_cooldown_s=10.0)
        limiter.report_rate_limit()
        assert limiter.limit == 4
        # Burst within the cooldown counts as one event.
        limiter.report_rate_limit()
        limiter.report_rate_limit()
        assert limiter.limit == 4

    async def test_shrink_floors_at_min(self):
        limiter = AdaptiveConcurrencyLimiter(8, shrink_cooldown_s=0.0)
        for _ in range(10):
            limiter.report_rate_limit()
        assert limiter.limit == 1

    async def test_recovers_after_quiet_period(self):
        limiter = AdaptiveConcurrencyLimiter(
            4, shrink_cooldown_s=0.0, recovery_interval_s=0.01
        )
        limiter.report_rate_limit()
        assert limiter.limit == 2
        await asyncio.sleep(0.02)
        await limiter.acquire()  # recovery happens on acquire
        await limiter.release()
        assert limiter.limit == 3

    async def test_gate_enforces_current_limit(self):
        limiter = AdaptiveConcurrencyLimiter(2)
        active = {"now": 0, "peak": 0}

        async def worker():
            await limiter.acquire()
            try:
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
                await asyncio.sleep(0.01)
                active["now"] -= 1
            finally:
                await limiter.release()

        await asyncio.gather(*(worker() for _ in range(6)))
        assert active["peak"] == 2


class TestExecutorIntegration:
    def _executor(self):
        agent = MagicMock()
        registry = MagicMock()
        registry.tools = {}
        registry._tool_search_tool = None
        agent.tool_registry = registry
        agent.client = MagicMock()
        return AgentToolExecutor(agent)

    async def test_rate_limited_branch_shrinks_pool(self):
        executor = self._executor()

        async def fake_execute_tool(name, inputs, context=None):
            if name == "limited":
                return ToolResult(
                    name=name, input={}, output=None, success=False,
                    error="429 rate limit exceeded",
                )
            return ToolResult(name=name, input={}, output="ok", success=True)

        executor.execute_tool = fake_execute_tool
        batch = [
            ToolCall(tool_call_id="1", name="limited", inputs={}),
            ToolCall(tool_call_id="2", name="fine", inputs={}),
        ]
        results = await executor._execute_parallel(batch, context=None)

        assert len(results) == 2
        limiter = executor._parallel_limiter
        assert limiter.limit < limiter._max

    async def test_limiter_persists_across_batches(self):
        executor = self._executor()

        async def ok_tool(name, inputs, context=None):
            return ToolResult(name=name, input={}, output="ok", success=True)

        executor.execute_tool = ok_tool
        await executor._execute_parallel(
            [ToolCall(tool_call_id="1", name="a", inputs={})], context=None
        )
        first = executor._parallel_limiter
        await executor._execute_parallel(
            [ToolCall(tool_call_id="2", name="b", inputs={})], context=None
        )
        assert executor._parallel_limiter is first
