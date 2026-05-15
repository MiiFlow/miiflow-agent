"""Tests for new orchestration features borrowed from Claude Code patterns.

Tests cover:
- Context compression (P0)
- Recovery strategies (P0)
- Progress tracking (P1)
- Tool pool narrowing (P1)
- Targeted re-planning (P1)
"""

import asyncio
import time

import pytest

from miiflow_agent.core.context_compression import (
    CompressionStrategy,
    ContextCompressor,
    _estimate_tokens,
)
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.enums import ReActEventType, StopReason
from miiflow_agent.core.react.models import ReActResult, ReActStep
from miiflow_agent.core.react.progress import ProgressSnapshot, ProgressTracker, ToolActivity
from miiflow_agent.core.react.recovery import (
    RecoveryAction,
    RecoveryManager,
    RecoveryStrategy,
)
from miiflow_agent.core.tools.tool_filter import ToolFilter


# ==================== Context Compression Tests ====================


class TestContextCompressor:
    """Tests for ContextCompressor."""

    def test_estimate_tokens(self):
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello world"),
        ]
        tokens = _estimate_tokens(messages)
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_no_compression_when_under_threshold(self):
        compressor = ContextCompressor(
            max_context_tokens=100000,
            compression_threshold=0.75,
            strategy=CompressionStrategy.AUTO,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        result = await compressor.compress_if_needed(messages)
        assert not result.was_compressed
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_truncate_strategy(self):
        compressor = ContextCompressor(
            max_context_tokens=100,  # Very low to force compression
            compression_threshold=0.1,
            strategy=CompressionStrategy.TRUNCATE,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
        ] + [
            Message(role=MessageRole.USER, content=f"Message {i} " * 20)
            for i in range(20)
        ]
        result = await compressor.compress_if_needed(messages, preserve_recent=2)
        assert result.was_compressed
        assert result.compressed_count < result.original_count
        # Should have system + boundary marker + recent messages
        assert any("[Context compressed:" in (m.content or "") for m in result.messages)

    @pytest.mark.asyncio
    async def test_none_strategy_never_compresses(self):
        compressor = ContextCompressor(
            max_context_tokens=10,  # Very low
            compression_threshold=0.01,
            strategy=CompressionStrategy.NONE,
        )
        messages = [
            Message(role=MessageRole.USER, content="A" * 1000),
        ]
        result = await compressor.compress_if_needed(messages)
        assert not result.was_compressed

    @pytest.mark.asyncio
    async def test_summarize_requires_client(self):
        compressor = ContextCompressor(
            max_context_tokens=10,
            compression_threshold=0.01,
            strategy=CompressionStrategy.SUMMARIZE,
            client=None,
        )
        messages = [
            Message(role=MessageRole.USER, content="A" * 1000),
            Message(role=MessageRole.ASSISTANT, content="B" * 1000),
            Message(role=MessageRole.USER, content="C" * 1000),
            Message(role=MessageRole.ASSISTANT, content="D" * 1000),
            Message(role=MessageRole.USER, content="E" * 1000),
        ]
        with pytest.raises(ValueError, match="LLMClient required"):
            await compressor.compress_if_needed(messages, preserve_recent=2)

    @pytest.mark.asyncio
    async def test_auto_falls_back_to_truncate_without_client(self):
        compressor = ContextCompressor(
            max_context_tokens=100,
            compression_threshold=0.01,
            strategy=CompressionStrategy.AUTO,
            client=None,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content="Sys"),
        ] + [
            Message(role=MessageRole.USER, content=f"Msg {i} " * 20)
            for i in range(10)
        ]
        result = await compressor.compress_if_needed(messages, preserve_recent=2)
        assert result.was_compressed


# ==================== Recovery Strategy Tests ====================


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    @pytest.mark.asyncio
    async def test_first_error_returns_guidance(self):
        manager = RecoveryManager(max_recovery_attempts=3)
        action = await manager.attempt_recovery(
            error=Exception("tool failed"),
            context=None,
            tool_name="search_web",
        )
        assert action.should_continue
        assert action.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE
        assert "search_web" in action.guidance_message
        assert "tool failed" in action.guidance_message

    @pytest.mark.asyncio
    async def test_second_error_returns_compress(self):
        manager = RecoveryManager(max_recovery_attempts=3)
        # First attempt
        await manager.attempt_recovery(Exception("err1"), context=None)
        # Second attempt
        action = await manager.attempt_recovery(Exception("err2"), context=None)
        assert action.should_continue
        assert action.strategy_used == RecoveryStrategy.COMPRESS_AND_RETRY

    @pytest.mark.asyncio
    async def test_third_error_returns_simplify(self):
        manager = RecoveryManager(max_recovery_attempts=3)
        await manager.attempt_recovery(Exception("e1"), context=None, tool_name="bad_tool")
        await manager.attempt_recovery(Exception("e2"), context=None, tool_name="bad_tool")
        action = await manager.attempt_recovery(Exception("e3"), context=None, tool_name="bad_tool")
        assert action.should_continue
        assert action.strategy_used == RecoveryStrategy.SIMPLIFY_TOOLS
        # Tool should be excluded after 2+ failures
        assert "bad_tool" in action.excluded_tools

    @pytest.mark.asyncio
    async def test_exhausted_returns_stop(self):
        manager = RecoveryManager(max_recovery_attempts=2)
        await manager.attempt_recovery(Exception("e1"), context=None)
        await manager.attempt_recovery(Exception("e2"), context=None)
        action = await manager.attempt_recovery(Exception("e3"), context=None)
        assert not action.should_continue

    @pytest.mark.asyncio
    async def test_success_resets_counter(self):
        manager = RecoveryManager(max_recovery_attempts=2)
        await manager.attempt_recovery(Exception("e1"), context=None)
        manager.record_success()
        # After success, should start over
        action = await manager.attempt_recovery(Exception("e2"), context=None)
        assert action.should_continue
        assert action.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE

    def test_excluded_tools_property(self):
        manager = RecoveryManager()
        assert len(manager.excluded_tools) == 0

    def test_reset(self):
        manager = RecoveryManager()
        manager._attempt_count = 5
        manager.reset()
        assert manager._attempt_count == 0


# ==================== Progress Tracking Tests ====================


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_initial_snapshot(self):
        tracker = ProgressTracker(max_steps=10)
        snapshot = tracker.snapshot()
        assert snapshot.current_step == 0
        assert snapshot.total_steps_limit == 10
        assert snapshot.tool_calls_count == 0
        assert snapshot.total_cost == 0.0
        assert snapshot.progress_percentage == 0.0

    def test_record_step(self):
        tracker = ProgressTracker(max_steps=10)
        step = ReActStep(
            step_number=1,
            thought="thinking",
            action="search",
            action_input={"q": "test"},
            cost=0.01,
            tokens_used=100,
        )
        tracker.record_step(step)
        snapshot = tracker.snapshot()
        assert snapshot.current_step == 1
        assert snapshot.tool_calls_count == 1
        assert snapshot.total_cost == 0.01
        assert snapshot.progress_percentage == 10.0

    def test_record_tool_call(self):
        tracker = ProgressTracker(max_steps=25, max_activities=3)
        tracker.record_tool_call("search", "Searching for news", True, 1.5)
        tracker.record_tool_call("fetch", "Fetching page", True, 0.8)
        tracker.record_tool_call("analyze", "Analyzing data", False, 2.0)
        tracker.record_tool_call("summarize", "Summarizing", True, 0.5)

        snapshot = tracker.snapshot()
        assert snapshot.tool_calls_count == 4
        # Only last 3 activities retained
        assert len(snapshot.recent_activities) == 3
        assert snapshot.recent_activities[0].tool_name == "fetch"
        assert snapshot.recent_activities[-1].tool_name == "summarize"

    def test_record_tokens(self):
        tracker = ProgressTracker()
        tracker.record_tokens(input_tokens=500, output_tokens=200, cost=0.05)
        tracker.record_tokens(input_tokens=300, output_tokens=100, cost=0.03)
        snapshot = tracker.snapshot()
        assert snapshot.total_input_tokens == 800
        assert snapshot.total_output_tokens == 300
        assert abs(snapshot.total_cost - 0.08) < 0.001

    def test_snapshot_to_dict(self):
        tracker = ProgressTracker(max_steps=20)
        tracker.record_tool_call("search", "Searching", True, 1.0)
        snapshot = tracker.snapshot()
        d = snapshot.to_dict()
        assert "current_step" in d
        assert "progress_percentage" in d
        assert "recent_activities" in d
        assert len(d["recent_activities"]) == 1
        assert d["recent_activities"][0]["tool_name"] == "search"

    def test_progress_percentage_cap(self):
        tracker = ProgressTracker(max_steps=5)
        tracker._current_step = 10  # Over limit
        snapshot = tracker.snapshot()
        assert snapshot.progress_percentage == 100.0


class TestToolActivity:
    def test_creation(self):
        activity = ToolActivity(
            tool_name="search",
            description="Searching for Tesla",
            timestamp=time.time(),
            success=True,
            execution_time=1.5,
        )
        assert activity.tool_name == "search"
        assert activity.success


# ==================== Tool Filter Tests ====================


class TestToolFilter:
    """Tests for ToolFilter."""

    def test_no_filter_allows_all(self):
        f = ToolFilter()
        assert f.is_allowed("any_tool")
        assert f.is_allowed("another_tool")

    def test_allowlist(self):
        f = ToolFilter(allowed_tools=["search", "read"])
        assert f.is_allowed("search")
        assert f.is_allowed("read")
        assert not f.is_allowed("write")
        assert not f.is_allowed("delete")

    def test_denylist(self):
        f = ToolFilter(denied_tools=["delete", "bash"])
        assert f.is_allowed("search")
        assert f.is_allowed("read")
        assert not f.is_allowed("delete")
        assert not f.is_allowed("bash")

    def test_allowlist_and_denylist(self):
        f = ToolFilter(allowed_tools=["a", "b", "c"], denied_tools=["b"])
        assert f.is_allowed("a")
        assert not f.is_allowed("b")  # Denied overrides allowed
        assert f.is_allowed("c")
        assert not f.is_allowed("d")  # Not in allowlist

    def test_filter_schemas_openai_format(self):
        f = ToolFilter(allowed_tools=["search"])
        schemas = [
            {"type": "function", "function": {"name": "search", "description": "Search"}},
            {"type": "function", "function": {"name": "delete", "description": "Delete"}},
        ]
        filtered = f.filter_schemas(schemas)
        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "search"

    def test_filter_schemas_anthropic_format(self):
        f = ToolFilter(denied_tools=["dangerous"])
        schemas = [
            {"name": "safe_tool", "description": "Safe"},
            {"name": "dangerous", "description": "Dangerous"},
        ]
        filtered = f.filter_schemas(schemas)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "safe_tool"

    def test_filter_tool_names(self):
        f = ToolFilter(allowed_tools=["a", "b"])
        result = f.filter_tool_names(["a", "b", "c", "d"])
        assert result == ["a", "b"]

    def test_add_denied(self):
        f = ToolFilter()
        assert f.is_allowed("tool_x")
        f.add_denied("tool_x")
        assert not f.is_allowed("tool_x")

    def test_for_role_summarizer(self):
        f = ToolFilter.for_role("summarizer")
        assert not f.is_allowed("search")
        assert not f.is_allowed("anything")

    def test_for_role_researcher(self):
        tools = ["search_web", "fetch_page", "write_file", "delete_record", "read_doc"]
        f = ToolFilter.for_role("researcher", tools)
        assert f.is_allowed("search_web")
        assert f.is_allowed("fetch_page")
        assert f.is_allowed("read_doc")
        assert not f.is_allowed("write_file")
        assert not f.is_allowed("delete_record")

    def test_for_role_implementer(self):
        f = ToolFilter.for_role("implementer")
        assert f.is_allowed("anything")
        assert f.is_allowed("write")
        assert f.is_allowed("delete")

    def test_for_role_analyzer(self):
        tools = ["search", "analyze_data", "write_file", "execute_code", "read_doc"]
        f = ToolFilter.for_role("analyzer", tools)
        assert f.is_allowed("search")
        assert f.is_allowed("read_doc")
        assert not f.is_allowed("write_file")
        assert not f.is_allowed("execute_code")

    def test_repr(self):
        f = ToolFilter(allowed_tools=["a"], denied_tools=["b"])
        r = repr(f)
        assert "ToolFilter" in r
        assert "allowed" in r
        assert "denied" in r


# ==================== Targeted Re-planning Tests ====================


class TestNewEnumValues:
    """Test new enum values were added correctly."""

    def test_progress_event_type(self):
        assert ReActEventType.PROGRESS.value == "progress"

    def test_recovery_exhausted_stop_reason(self):
        assert StopReason.RECOVERY_EXHAUSTED.value == "recovery_exhausted"
