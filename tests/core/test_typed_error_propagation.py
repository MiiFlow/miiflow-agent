"""Tests for typed error propagation.

Agent.run() used to collapse every failure into MiiflowLLMError(MODEL_ERROR),
forcing callers to parse message strings to decide whether to retry, back
off, or re-auth. The orchestrator's crash result also carried nothing but an
error-shaped final_answer string — indistinguishable from a real answer
without string matching.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent.core.agent import Agent, AgentType
from miiflow_agent.core.client import LLMClient
from miiflow_agent.core.exceptions import (
    ErrorType,
    MiiflowLLMError,
    RateLimitError,
)
from miiflow_agent.core.react.enums import StopReason
from miiflow_agent.core.react.orchestrator import ReActOrchestrator


def _agent_with_failing_execute(error):
    provider = MagicMock()
    provider.provider_name = "testprov"
    provider.model = "test-model"
    agent = Agent(LLMClient(provider), agent_type=AgentType.SINGLE_HOP)
    agent._execute_with_context = AsyncMock(side_effect=error)
    return agent


class TestAgentRunErrorTypes:
    async def test_typed_error_keeps_its_type(self):
        agent = _agent_with_failing_execute(
            RateLimitError("slow down", "testprov", retry_after=7.0)
        )
        with pytest.raises(RateLimitError) as exc_info:
            await agent.run("hi")
        assert exc_info.value.retry_after == 7.0
        assert exc_info.value.error_type == ErrorType.RATE_LIMITED

    async def test_untyped_error_is_wrapped_with_original(self):
        agent = _agent_with_failing_execute(ValueError("boom"))
        with pytest.raises(MiiflowLLMError) as exc_info:
            await agent.run("hi")
        assert exc_info.value.error_type == ErrorType.MODEL_ERROR
        assert isinstance(exc_info.value.original_error, ValueError)
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestCrashResultMetadata:
    def test_error_result_is_machine_readable(self):
        from miiflow_agent.core.react.execution import ExecutionState

        orchestrator = ReActOrchestrator(
            tool_executor=MagicMock(),
            event_bus=None,
            safety_manager=None,
        )
        result = orchestrator._build_error_result(
            ExecutionState(),
            RateLimitError("too many requests", "anthropic"),
        )

        assert result.stop_reason == StopReason.FORCED_STOP
        error = result.metadata["error"]
        assert error["crashed"] is True
        assert error["exception"] == "RateLimitError"
        assert error["error_type"] == "rate_limited"
        assert "too many requests" in error["message"]

    def test_plain_exception_has_no_error_type(self):
        from miiflow_agent.core.react.execution import ExecutionState

        orchestrator = ReActOrchestrator(
            tool_executor=MagicMock(),
            event_bus=None,
            safety_manager=None,
        )
        result = orchestrator._build_error_result(
            ExecutionState(), RuntimeError("boom")
        )
        error = result.metadata["error"]
        assert error["crashed"] is True
        assert error["exception"] == "RuntimeError"
        assert error["error_type"] is None
