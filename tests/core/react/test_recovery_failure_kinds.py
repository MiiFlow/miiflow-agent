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


# ── fatal provider errors ────────────────────────────────────────────────────

BILLING_400 = (
    "Step execution failed: Error code: 400 - {'type': 'error', 'error': "
    "{'type': 'invalid_request_error', 'message': 'Your credit balance is too "
    "low to access the Anthropic API. Please go to Plans & Billing to upgrade "
    "or purchase credits.'}}"
)


@pytest.mark.asyncio
async def test_billing_error_stops_the_ladder_immediately():
    """2026-08-11: an out-of-credit account produced five identical failing
    calls per agent in one second, because every ladder step retried the
    request. Billing/auth are about the account, not the request."""
    from miiflow_agent.core.react.recovery import is_fatal_provider_error

    assert is_fatal_provider_error(Exception(BILLING_400))
    manager = RecoveryManager(max_recovery_attempts=3)
    action = await manager.attempt_recovery(
        error=Exception(BILLING_400), context=None, tool_name=None
    )
    assert action.should_continue is False
    assert action.guidance_message is None
    # Nothing was consumed from the runtime ladder either.
    assert manager._attempt_count == 0


@pytest.mark.asyncio
async def test_auth_and_quota_wordings_are_fatal():
    from miiflow_agent.core.react.recovery import is_fatal_provider_error

    for text in (
        "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}",
        "Error code: 429 - You exceeded your current quota, please check your plan and billing details. (insufficient_quota)",
        "Error code: 403 - permission_error: This API key does not have access to model claude-opus-5",
    ):
        assert is_fatal_provider_error(Exception(text)), text


@pytest.mark.asyncio
async def test_ordinary_tool_and_transient_errors_are_not_fatal():
    from miiflow_agent.core.react.recovery import is_fatal_provider_error

    for text in (
        "account not found",
        "Error code: 529 - overloaded_error",
        "Tool reference 'list_all_ad_accounts' not found in available tools",
        "prompt is too long: 210000 tokens > 200000 maximum",
        "each tool_use must have a single result",
    ):
        assert not is_fatal_provider_error(Exception(text)), text


@pytest.mark.asyncio
async def test_tool_step_auth_wording_is_not_fatal():
    """A TOOL that answers 401 is one integration's expired credential; the
    ladder must keep running (guidance → SIMPLIFY_TOOLS), not halt the run."""
    manager = RecoveryManager(max_recovery_attempts=3)
    action = await manager.attempt_recovery(
        error=Exception("HTTP 401 error for tool 'shopify_orders': Unauthorized invalid api key"),
        context=None,
        tool_name="shopify_orders",
    )
    assert action.should_continue is True
