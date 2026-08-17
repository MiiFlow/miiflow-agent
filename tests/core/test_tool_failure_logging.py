"""Tool failures log at the severity that matches what they are.

A tool that raises an exception marked ``is_tool_validation_error = True`` is
telling the model "fix your call" (malformed GAQL, unknown field, …). That is
the tool's normal contract with the model — not a malfunction — so it must
NOT be logged with a traceback (``logger.exception``): the PostHog log bridge
turns every exception-bearing log record into an Error Tracking issue, and a
model typo would page on-call. Real failures keep the traceback.

Regression for PostHog issues 01a01101-c88f-7621-ad0c-41f3e6895473 /
01a010e1-d1f8-7af0-8ed5-2bed6461f1c8 (2026-08-17): every rejected GAQL
query became an alert after the TRY400 sweep.
"""

import logging

import pytest

from miiflow_agent.core.agent import RunContext
from miiflow_agent.core.tools import ToolRegistry, tool


class _QueryShapeError(Exception):
    is_tool_validation_error = True


@tool(description="Rejects the model's input")
async def rejecting(ctx: RunContext, query: str) -> dict:
    raise _QueryShapeError(f"Could not parse: {query}")


@tool(description="Genuinely broken")
async def broken(ctx: RunContext, query: str) -> dict:
    raise RuntimeError("connection reset")


@tool(description="Rejects without context")
def rejecting_plain(query: str) -> dict:
    raise _QueryShapeError(f"Could not parse: {query}")


class _BadInput(ValueError):
    """A declared rejection that ALSO happens to be a ValueError — it lands in
    FunctionTool's dedicated `except ValueError` branch, not the generic one."""

    is_tool_validation_error = True


@tool(description="Rejects via a ValueError subclass")
def rejecting_value_error(query: str) -> dict:
    raise _BadInput(f"Bad value: {query}")


@pytest.mark.asyncio
async def test_validation_error_logs_warning_without_traceback(caplog):
    registry = ToolRegistry()
    registry.register(rejecting)

    with caplog.at_level(logging.DEBUG):
        result = await registry.execute_safe_with_context(
            "rejecting", object(), query="SmetricsPlaceholder"
        )

    assert not result.success
    assert result.metadata["is_validation_error"] is True
    # The failure still reaches the model as the standard retry signal.
    assert "Could not parse: SmetricsPlaceholder" in result.error

    recs = [r for r in caplog.records if "rejecting" in r.getMessage()]
    assert recs, "the rejection must still be logged"
    assert all(r.levelno < logging.ERROR for r in recs), [
        (r.levelname, r.getMessage()) for r in recs
    ]
    assert all(r.exc_info is None for r in recs)


@pytest.mark.asyncio
async def test_real_failure_still_logs_exception(caplog):
    registry = ToolRegistry()
    registry.register(broken)

    with caplog.at_level(logging.DEBUG):
        result = await registry.execute_safe_with_context(
            "broken", object(), query="SELECT campaign.id FROM campaign"
        )

    assert not result.success
    assert result.metadata["is_validation_error"] is False

    error_recs = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "broken" in r.getMessage()
    ]
    assert error_recs
    assert any(r.exc_info for r in error_recs)


@pytest.mark.asyncio
async def test_validation_error_no_context_path_logs_warning(caplog):
    """``execute_safe`` (FunctionTool.execute → registry) takes the same rule."""
    registry = ToolRegistry()
    registry.register(rejecting_plain)

    with caplog.at_level(logging.DEBUG):
        result = await registry.execute_safe("rejecting_plain", query="nope")

    assert not result.success
    recs = [r for r in caplog.records if "rejecting_plain" in r.getMessage()]
    assert recs
    assert all(r.levelno < logging.ERROR for r in recs), [
        (r.levelname, r.getMessage()) for r in recs
    ]


@pytest.mark.asyncio
async def test_flag_is_stamped_on_every_function_tool_except_branch(caplog):
    """The logger and the recovery classifier read the SAME flag. A flagged
    ValueError subclass takes FunctionTool's `except ValueError` branch; if
    that branch logged WARNING but left metadata unstamped, two parallel
    rejections would trip the all-failed recovery ladder the flag exists to
    prevent."""
    registry = ToolRegistry()
    registry.register(rejecting_value_error)

    with caplog.at_level(logging.DEBUG):
        result = await registry.execute_safe("rejecting_value_error", query="nope")

    assert not result.success
    assert result.metadata["is_validation_error"] is True
    recs = [r for r in caplog.records if "rejecting_value_error" in r.getMessage()]
    assert recs
    assert all(r.levelno < logging.ERROR for r in recs), [
        (r.levelname, r.getMessage()) for r in recs
    ]
