"""Tests for FailureKind routing in RecoveryManager.

SCHEMA / TRUNCATION failures must NOT count toward the per-tool exclusion
threshold or pull the recovery ladder forward — the orchestrator already
emits a structured tool_use_error, and the LLM should be given many
chances to self-correct against it. Only RUNTIME failures escalate.
"""

import pytest

from miiflow_agent.core.react.recovery import (
    FailureKind,
    RecoveryManager,
    RecoveryStrategy,
)


@pytest.mark.asyncio
async def test_schema_failure_does_not_count_toward_exclusion():
    manager = RecoveryManager(max_recovery_attempts=3)

    # Five SCHEMA failures on the same tool — pre-fix this would have hit
    # SIMPLIFY_TOOLS on attempt 3 and excluded `render_chart` for the rest
    # of the session.
    for _ in range(5):
        action = await manager.attempt_recovery(
            error=Exception("missing required: series"),
            context=None,
            tool_name="render_chart",
            failure_kind=FailureKind.SCHEMA,
        )
        assert action.should_continue
        assert action.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE
        # Recovery must NOT prepend a generic guidance message — the
        # orchestrator already wrote a tool_use_error tool_result.
        assert action.guidance_message is None

    assert "render_chart" not in manager.excluded_tools
    # The runtime attempt counter should be untouched, leaving the full
    # ladder available for genuine runtime errors.
    assert manager._attempt_count == 0


@pytest.mark.asyncio
async def test_truncation_failure_does_not_count_toward_exclusion():
    manager = RecoveryManager(max_recovery_attempts=3)

    for _ in range(4):
        action = await manager.attempt_recovery(
            error=Exception("truncated mid-stream"),
            context=None,
            tool_name="render_table",
            failure_kind=FailureKind.TRUNCATION,
        )
        assert action.should_continue
        assert action.guidance_message is None

    assert "render_table" not in manager.excluded_tools
    assert manager._attempt_count == 0


@pytest.mark.asyncio
async def test_runtime_failures_still_escalate_through_ladder():
    """Sanity check that the existing ladder still works for RUNTIME."""
    manager = RecoveryManager(max_recovery_attempts=3)

    # Default failure_kind is RUNTIME, mimicking pre-existing call sites.
    action1 = await manager.attempt_recovery(
        Exception("e1"), context=None, tool_name="bad_tool"
    )
    assert action1.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE

    action2 = await manager.attempt_recovery(
        Exception("e2"), context=None, tool_name="bad_tool"
    )
    assert action2.strategy_used == RecoveryStrategy.COMPRESS_AND_RETRY

    action3 = await manager.attempt_recovery(
        Exception("e3"), context=None, tool_name="bad_tool"
    )
    assert action3.strategy_used == RecoveryStrategy.SIMPLIFY_TOOLS
    assert "bad_tool" in (action3.excluded_tools or set())


@pytest.mark.asyncio
async def test_schema_failures_dont_consume_runtime_budget():
    """A storm of schema failures must not poison the runtime ladder.

    Concretely: if the model misuses a tool 5 times (schema), then a
    different tool genuinely throws, the runtime failure should still
    start at attempt 1 (RETRY_WITH_GUIDANCE), not advance straight to
    SIMPLIFY_TOOLS.
    """
    manager = RecoveryManager(max_recovery_attempts=3)

    for _ in range(5):
        await manager.attempt_recovery(
            Exception("schema mismatch"),
            context=None,
            tool_name="render_chart",
            failure_kind=FailureKind.SCHEMA,
        )

    runtime_action = await manager.attempt_recovery(
        Exception("oauth expired"),
        context=None,
        tool_name="google_ads_query",
        failure_kind=FailureKind.RUNTIME,
    )
    assert runtime_action.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE
    assert "render_chart" not in manager.excluded_tools
