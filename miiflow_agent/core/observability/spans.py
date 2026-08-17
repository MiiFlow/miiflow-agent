"""Agent-level spans, so a trace shows the AGENT TREE and not a flat call list.

OpenInference instruments the provider SDKs, so every LLM call already becomes
a span with prompt, completion, tokens and latency. What it cannot know is
which AGENT made the call: a four-way fan-out arrives as an undifferentiated
stream of LLM spans with no way to tell the root from its children.

That is the same blind spot `Message.metadata["llm_timeline"]` had before the
callback context was stamped — and the reason the 2026-08 dispatch-latency
investigation had to reconstruct the tree by hand from durations and token
totals. Wrapping each agent run in a span makes the LLM spans nest underneath
the agent that issued them, which is what turns a trace into an explanation.

Nesting relies on OpenTelemetry's contextvar propagation: `asyncio` copies the
current context when a task is created, so children dispatched inside an active
span inherit it through `asyncio.gather` without any explicit parent handoff.

## These spans must speak OpenInference, not just OTel

A span carrying only custom attributes is *valid* and *arrives* — and renders
as nothing. Arize/Phoenix key their entire UI off the OpenInference semantic
conventions: `openinference.span.kind` decides whether a span shows as an Agent,
an LLM, or an unclassified box, and the Input/Output panels read `input.value` /
`output.value`. Production 2026-08-04: agent spans landed in Arize AX with
`agent.handle`, `thread.id` and friends intact, and every panel an operator
actually reads was empty — kind `unknown`, Status `UNSET`, Cost `--`, no input,
no output. Nothing was lost in transit; the spans genuinely said nothing the
tool could interpret.

Class of error: **emitting a payload in a schema the consumer does not read**,
where the transport succeeds and so every check short of opening the UI passes.
The conventions here are not decoration, they ARE the payload.

The attribute names are inlined as literals rather than imported from
`openinference-semantic-conventions` because observability is an OPTIONAL extra
of this package — a hard import would make the agent unimportable wherever
tracing is not installed, which is the failure mode this module already guards
against everywhere else.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

TRACER_NAME = "miiflow.agent"

# OpenInference semantic conventions.
SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
SESSION_ID = "session.id"
USER_ID = "user.id"
TOOL_NAME = "tool.name"
TOOL_PARAMETERS = "tool.parameters"

# ── How large one attribute may be ───────────────────────────────────────────
# A whole conversation gets handed in as an input value, and an unbounded
# attribute is how a span grows past what a collector will accept — a rejection
# that costs the span SILENTLY, since an OTLP exporter does not raise. Generous
# enough to keep a real prompt readable.
#
# ONE owner for the number, consumed twice: `_truncate` below marks our own
# values as clipped, and `span_limits()` hands the same bound to the SDK so it
# also covers spans this package does not author (above all the OpenInference
# LLM spans, which carry the entire prompt in `input.value` AND again across
# `llm.input_messages.*` — roughly a megabyte per span on a 64k-token turn).
# Two independently-written caps would be the same "second copy" defect this
# module documents elsewhere, and the test that caught it is real: the SDK
# clipped our truncation MARKER off, so a partial value looked complete.
DEFAULT_ATTRIBUTE_VALUE_LIMIT = 32_000
_TRUNCATION_MARKER = "…[truncated {dropped} chars]"

# OUR variable, deliberately NOT the SDK's `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`.
# That name is already read by `SpanLimits.__init__`, and its semantics are
# literal: `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT=0` means "truncate to ZERO chars",
# not "no limit". Reusing the name with a `0 = disabled` convention meant that
# switching the cap off blanked EVERY attribute on every span — the exact
# "arrives and renders as nothing" failure this module exists to prevent, and it
# is unrecoverable from our side: with that env var set to 0 even
# `SpanLimits(max_attribute_length=None)` comes back 0, because None means
# "unset, read the environment".
#
# Class of error: **overloading a third party's configuration key with our own
# semantics.** The two readers never disagree loudly; the vendor's reading just
# silently wins. If a setting needs different meaning, it needs a different name.
ATTRIBUTE_LIMIT_ENV = "MIIFLOW_SPAN_ATTRIBUTE_LIMIT"


def attribute_value_limit() -> Optional[int]:
    """Chars allowed in one span attribute; None means no limit.

    Override with MIIFLOW_SPAN_ATTRIBUTE_LIMIT. 0 or negative disables the cap.
    Setting the SDK's own OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT still works and
    still wins wherever we do not pass an explicit value — but it is the
    vendor's knob with the vendor's meaning, so do not use it to disable.
    """
    raw = os.getenv(ATTRIBUTE_LIMIT_ENV)
    if raw is None:
        return DEFAULT_ATTRIBUTE_VALUE_LIMIT
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using %d",
            ATTRIBUTE_LIMIT_ENV,
            raw,
            DEFAULT_ATTRIBUTE_VALUE_LIMIT,
        )
        return DEFAULT_ATTRIBUTE_VALUE_LIMIT
    return None if parsed <= 0 else parsed


def span_limits():
    """SpanLimits carrying `attribute_value_limit()`, or None when unavailable."""
    limit = attribute_value_limit()
    if limit is None:
        return None
    try:
        from opentelemetry.sdk.trace import SpanLimits

        return SpanLimits(max_attribute_length=limit)
    except Exception as exc:  # noqa: BLE001 — older SDK without SpanLimits
        logger.debug("SpanLimits unavailable (%s); attributes stay unbounded", exc)
        return None


def _truncate(value: str) -> str:
    """Clip to the limit INCLUDING the marker, so the marker survives.

    Reserving room for it is the whole point: the SDK applies the same cap
    afterwards, so a value of `limit` chars plus a suffix comes back as exactly
    `limit` chars with the suffix gone — a partial value that reads as complete.
    """
    limit = attribute_value_limit()
    if limit is None or len(value) <= limit:
        return value
    marker = _TRUNCATION_MARKER.format(dropped=len(value) - limit)
    keep = max(0, limit - len(marker))
    return value[:keep] + marker


def _coerce(value: Any) -> Any:
    """Pass through what OTel accepts natively; stringify everything else.

    The previous blanket `str(value)` turned every number and bool into text, so
    a numeric attribute could not be filtered or aggregated on in the UI.
    """
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    return _truncate(str(value))


def set_span_attribute(span: Optional[Any], key: str, value: Any) -> None:
    """Set one attribute, tolerating a no-op span and any setter failure.

    Observability must never be able to fail the thing it observes.
    """
    if span is None or value is None:
        return
    try:
        span.set_attribute(key, _coerce(value))
    except Exception:  # noqa: BLE001 — tracing must not break the caller
        pass


def set_span_output(span: Optional[Any], value: Any) -> None:
    """Record what this agent produced, so the Output panel is not empty.

    Separate from `agent_span` because the output is only known once the body
    has run — for a streaming turn, after the generator is exhausted.
    """
    set_span_attribute(span, OUTPUT_VALUE, value)


async def traced_stream(
    name: str,
    source: Any,
    *,
    kind: str = "AGENT",
    input_value: Any = None,
    session_id: Any = None,
    user_id: Any = None,
    output_getter: Any = None,
    status_getter: Any = None,
    **attributes: Any,
) -> Any:
    """Wrap an async generator in an agent span WITHOUT straddling a `yield`.

    `agent_span` is a SYNCHRONOUS context manager. Used as::

        with agent_span(...):
            async for event in body():
                yield event

    inside an async generator, its `attach()` runs during the first `__anext__`
    and its `detach()` during a later one — and `contextvars` are per-TASK, not
    per-generator. If anything advances the generator from a different task than
    the one that started it (a pump task, a queue producer, an ASGI server that
    hands the response body to another task), the span is no longer current when
    the body runs, so OpenInference's LLM spans start a NEW ROOT TRACE. The
    result is exactly the 2026-08-04 production trace: one lone
    `agent.adlyse_ai_2` span, its eleven LLM calls nowhere in it. OTel says so
    out loud when it happens — `Failed to detach context: ... Token was created
    in a different Context` — which is the string to grep for.

    So this never makes the span current across a suspension. The span is
    started detached (`start_span`, not `start_as_current_span`), and its
    context is attached and detached INSIDE each `__anext__`, where the body —
    and therefore every LLM call — actually runs. Between yields no context is
    held, so nothing can be detached in the wrong task.

    `output_getter` is called once at the end for `output.value`; the answer is
    only known after the stream is exhausted, and it must be read even when the
    stream ends by exception or cancellation.

    `status_getter`, if given, is called once at the end (only when the stream
    did NOT end by exception) and returns an :class:`AgentOutcome` — or a
    plain outcome string — that decides the span's status. A turn that ends
    "gracefully" with a canned "I wasn't able to finish this run" answer is a
    failure to the reader, and Arize can only filter on span STATUS: without
    this every agent span was UNSET, successes and halts alike.
    """
    tracer = None
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(TRACER_NAME)
    except Exception:  # opentelemetry absent or misconfigured
        tracer = None

    if tracer is None:
        async for item in source:
            yield item
        return

    from opentelemetry import context as otel_context
    from opentelemetry import trace

    span = tracer.start_span(name)
    set_span_attribute(span, SPAN_KIND, kind)
    set_span_attribute(span, INPUT_VALUE, input_value)
    set_span_attribute(span, SESSION_ID, session_id)
    set_span_attribute(span, USER_ID, user_id)
    for key, value in attributes.items():
        set_span_attribute(span, key, value)

    span_context = trace.set_span_in_context(span)
    iterator = source.__aiter__()
    try:
        while True:
            token = otel_context.attach(span_context)
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                break
            finally:
                # Same call frame as the attach, so this can never run in a
                # different task than the one that attached.
                otel_context.detach(token)
            yield item
    except BaseException as exc:
        # `start_span` has no context manager to set this for us, unlike
        # `start_as_current_span` — so unlike `agent_span`, recording here is
        # not a duplicate.
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
        except Exception:  # noqa: BLE001
            pass
        raise
    else:
        _apply_outcome(span, status_getter)
    finally:
        if output_getter is not None:
            try:
                set_span_output(span, output_getter())
            except Exception:  # noqa: BLE001 — never fail a turn for telemetry
                pass
        span.end()


class AgentOutcome:
    """How an agent turn ended, for the span's status + `agent.outcome`.

    ``outcome`` is one of the OUTCOME_* constants (free-form is tolerated);
    ``ok`` decides the OTel status; ``description`` becomes the ERROR status
    message (bounded) so a failed turn's cause is readable in the trace list.
    """

    __slots__ = ("outcome", "ok", "description")

    def __init__(self, outcome: str, *, ok: bool = True, description: Any = None):
        self.outcome = outcome
        self.ok = ok
        self.description = description


OUTCOME_ANSWERED = "answered"
OUTCOME_HALTED = "halted"  # safety condition / repeated tool errors
OUTCOME_CLARIFICATION = "clarification"  # paused for the user
OUTCOME_APPROVAL = "approval"  # paused for a tool / plan approval
OUTCOME_ERROR = "error"
OUTCOME_ATTRIBUTE = "agent.outcome"


def _apply_outcome(span: Any, status_getter: Any) -> None:
    """Set status (+ `agent.outcome`) from `status_getter`; OK when absent."""
    outcome: Any = None
    if status_getter is not None:
        try:
            outcome = status_getter()
        except Exception:  # noqa: BLE001 — telemetry never fails the turn
            outcome = None
    if isinstance(outcome, str):
        outcome = AgentOutcome(outcome, ok=outcome not in (OUTCOME_HALTED, OUTCOME_ERROR))
    try:
        from opentelemetry.trace import Status, StatusCode

        if outcome is None:
            span.set_status(Status(StatusCode.OK))
            return
        set_span_attribute(span, OUTCOME_ATTRIBUTE, outcome.outcome)
        if outcome.ok:
            span.set_status(Status(StatusCode.OK))
        else:
            description = str(outcome.description or outcome.outcome)
            span.set_status(Status(StatusCode.ERROR, description[:512]))
    except Exception:  # noqa: BLE001
        return


@contextmanager
def agent_span(
    name: str,
    *,
    kind: str = "AGENT",
    input_value: Any = None,
    session_id: Any = None,
    user_id: Any = None,
    **attributes: Any,
) -> Iterator[Optional[Any]]:
    """Open a span for one agent run. A no-op when tracing is not configured.

    Failures in span SETUP degrade to a no-op; failures inside the wrapped body
    propagate untouched. The distinction is deliberate — a blanket try/except
    around the yield would swallow real errors from the code being traced and
    turn an observability feature into a source of silent failure, which is the
    exact class of bug this whole effort exists to surface.

    Nothing here catches the body's exception to mark the span errored:
    `start_as_current_span` defaults to `record_exception=True` and
    `set_status_on_exception=True`, so OTel already sets ERROR and attaches the
    exception event with a better description than we would write. Adding our
    own recorded it TWICE — caught by
    `test_a_raising_body_marks_the_span_errored`, which asserts the event count
    rather than merely that an exception event exists.

    `session_id` is what groups the turns of one conversation into a session in
    the Arize/Phoenix UI; pass the thread id. Without it every turn of a thread
    is an unrelated trace and the conversation cannot be replayed.
    """
    tracer = None
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(TRACER_NAME)
    except Exception:  # opentelemetry absent or misconfigured
        tracer = None

    if tracer is None:
        yield None
        return

    # With no configured provider OTel returns a non-recording span, so this
    # stays cheap when tracing is off rather than needing its own flag.
    with tracer.start_as_current_span(name) as span:
        set_span_attribute(span, SPAN_KIND, kind)
        set_span_attribute(span, INPUT_VALUE, input_value)
        set_span_attribute(span, SESSION_ID, session_id)
        set_span_attribute(span, USER_ID, user_id)
        for key, value in attributes.items():
            set_span_attribute(span, key, value)
        yield span
        # Reached only when the body did not raise (a raise propagates out of
        # the `with` and OTel marks ERROR itself). An explicit OK is what lets
        # Arize tell "ran fine" from "UNSET" — the default status — which is
        # also what a span that never finished properly reports.
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.OK))
        except Exception:  # noqa: BLE001
            pass


# Tool outputs are ALSO on the LLM span that follows (as a tool_result in the
# next prompt), so the tool span keeps a readable excerpt rather than a second
# full copy — enough to see what came back, small enough that a 200-call
# investigator run does not double its trace payload.
TOOL_OUTPUT_EXCERPT_CHARS = 8_000

# Exceptions a tool raises to PAUSE the run rather than fail it. Named, not
# imported, so this module stays free of react-package imports.
_PAUSE_EXCEPTION_NAMES = frozenset(
    {"ToolApprovalRequired", "PlanApprovalRequired", "GraphInterrupt"}
)


@contextmanager
def tool_span(tool_name: str, inputs: Any = None) -> Iterator[Optional[Any]]:
    """Open an OpenInference TOOL span around one tool execution.

    Before this existed the trace of a 15-minute investigator run was one AGENT
    span with LLM children and NOTHING in between: which tools ran, how long
    each took, which one failed and why were all invisible in Arize (0 TOOL
    spans in the first weeks of export). The LLM span that follows carries the
    tool result inside the next prompt, so this span holds the tool's identity,
    its input, timing, status and a bounded output excerpt.

    Status: OK for a successful result; ERROR (with the tool's error text) for
    a failed one; a raise that PAUSES the run (approval / interrupt) is OK with
    `tool.outcome=paused`, any other raise is ERROR. `record_tool_result` sets
    the result-derived half; the exception half is handled here.
    """
    tracer = None
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(TRACER_NAME)
    except Exception:  # opentelemetry absent or misconfigured
        tracer = None

    if tracer is None:
        yield None
        return

    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(
        f"tool.{tool_name}",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        set_span_attribute(span, SPAN_KIND, "TOOL")
        set_span_attribute(span, TOOL_NAME, tool_name)
        if inputs is not None:
            rendered = _render_json(inputs)
            set_span_attribute(span, INPUT_VALUE, rendered)
            set_span_attribute(span, TOOL_PARAMETERS, rendered)
        try:
            yield span
        except BaseException as exc:
            if type(exc).__name__ in _PAUSE_EXCEPTION_NAMES:
                set_span_attribute(span, "tool.outcome", "paused")
                span.set_status(Status(StatusCode.OK))
            else:
                set_span_attribute(span, "tool.outcome", "raised")
                try:
                    span.record_exception(exc)
                except Exception:  # noqa: BLE001
                    pass
                span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:512]))
            raise


def record_tool_result(span: Optional[Any], result: Any) -> None:
    """Stamp a ToolResult-shaped object onto its TOOL span (status + excerpt)."""
    if span is None or result is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        success = bool(getattr(result, "success", True)) and getattr(result, "error", None) is None
        error = getattr(result, "error", None)
        output = getattr(result, "output", None)
        metadata = getattr(result, "metadata", None) or {}
        if metadata.get("served_from_ledger"):
            set_span_attribute(span, "tool.served_from_ledger", True)
        if metadata.get("observation_ref"):
            set_span_attribute(span, "tool.observation_ref", metadata["observation_ref"])
        if output is not None:
            excerpt = _render_json(output)
            if len(excerpt) > TOOL_OUTPUT_EXCERPT_CHARS:
                excerpt = (
                    excerpt[:TOOL_OUTPUT_EXCERPT_CHARS]
                    + f"… [+{len(excerpt) - TOOL_OUTPUT_EXCERPT_CHARS} chars]"
                )
            set_span_attribute(span, OUTPUT_VALUE, excerpt)
        if success:
            set_span_attribute(span, "tool.outcome", "ok")
            span.set_status(Status(StatusCode.OK))
        else:
            set_span_attribute(span, "tool.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, str(error or "tool failed")[:512]))
    except Exception:  # noqa: BLE001 — telemetry never fails the tool
        return


def _render_json(value: Any) -> str:
    """Compact JSON for a span attribute; falls back to str() for odd types."""
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        return str(value)
