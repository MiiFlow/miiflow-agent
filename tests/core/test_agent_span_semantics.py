"""Agent spans must speak OpenInference, not just OTel.

Production 2026-08-04: agent spans arrived in Arize AX intact and every panel an
operator reads was empty — kind `unknown`, Status `UNSET`, Cost `--`, no input,
no output — because the span carried only custom `agent.*` attributes. Delivery
was never the problem; the payload was in a schema the consumer does not read.

These assert on the exact attribute names the UI keys off, since that is the
contract that broke. Nothing here touches the network.
"""

import pytest

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from miiflow_agent.core.observability.spans import (
    DEFAULT_ATTRIBUTE_VALUE_LIMIT,
    agent_span,
    attribute_value_limit,
    set_span_output,
    span_limits,
)


@pytest.fixture
def spans():
    """A private provider + in-memory exporter, returning finished spans by name.

    `trace.set_tracer_provider` only takes effect once per process, so this
    binds the tracer directly instead — running the suite in any order must not
    change what these tests measure.
    """
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider(span_limits=span_limits())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("miiflow.agent")

    original = trace.get_tracer
    trace.get_tracer = lambda *a, **k: tracer
    try:
        yield lambda: {s.name: s for s in exporter.get_finished_spans()}
    finally:
        trace.get_tracer = original


class TestOpenInferenceAttributes:
    def test_span_declares_itself_an_agent(self, spans):
        with agent_span("agent.root"):
            pass
        assert spans()["agent.root"].attributes["openinference.span.kind"] == "AGENT"

    def test_input_output_and_session_are_recorded(self, spans):
        with agent_span(
            "agent.root",
            input_value="how many orgs onboarded last week?",
            session_id="thread_abc",
            user_id="participant_xyz",
        ) as span:
            set_span_output(span, "8 organizations.")

        attrs = spans()["agent.root"].attributes
        assert attrs["input.value"] == "how many orgs onboarded last week?"
        assert attrs["output.value"] == "8 organizations."
        # session.id is what groups a thread's turns into one replayable
        # conversation; without it every turn is an unrelated trace.
        assert attrs["session.id"] == "thread_abc"
        assert attrs["user.id"] == "participant_xyz"

    def test_custom_attributes_still_travel(self, spans):
        """The OpenInference keys are the payload; `agent.*` remain the filters."""
        with agent_span("agent.root", **{"agent.handle": "adlyse_ai_2"}):
            pass
        assert spans()["agent.root"].attributes["agent.handle"] == "adlyse_ai_2"

    def test_numbers_keep_their_type(self, spans):
        """A blanket str() made numeric attributes unfilterable in the UI."""
        with agent_span("agent.root", **{"agent.depth": 2, "agent.root_flag": True}):
            pass
        attrs = spans()["agent.root"].attributes
        assert attrs["agent.depth"] == 2
        assert attrs["agent.root_flag"] is True

    def test_none_values_are_omitted_not_stringified(self, spans):
        with agent_span("agent.root", input_value=None, **{"agent.handle": None}):
            pass
        attrs = spans()["agent.root"].attributes
        assert "input.value" not in attrs
        assert "agent.handle" not in attrs


class TestFailuresAreVisible:
    def test_a_raising_body_marks_the_span_errored(self, spans):
        """UNSET on a span that raised is a lie the UI renders as success, and
        the error filter operators live in never shows the broken turn.

        OTel's own span context manager provides this
        (`set_status_on_exception` / `record_exception` both default True), so
        the assertion is on the event COUNT, not on presence: an added
        hand-rolled recorder passes a presence check while double-reporting
        every failure. It caught exactly that here.
        """
        with pytest.raises(RuntimeError):
            with agent_span("agent.root"):
                raise RuntimeError("provider 503")

        span = spans()["agent.root"]
        assert span.status.status_code.name == "ERROR"
        assert [e.name for e in span.events] == ["exception"]

    def test_the_exception_still_reaches_the_caller(self, spans):
        """Recording is not swallowing — tracing must not eat a real error."""
        with pytest.raises(ValueError, match="boom"):
            with agent_span("agent.root"):
                raise ValueError("boom")


class TestOversizedAttributesAreBoundedNotDropped:
    def test_a_huge_input_keeps_its_truncation_marker(self, spans):
        """The marker must fit INSIDE the cap, not hang off the end of it.

        The SDK applies the same limit after us, so a value of exactly `limit`
        chars plus a suffix came back as `limit` chars with the suffix gone —
        a partial value that reads as complete. Asserting `<= limit` (not
        `<= limit + slack`) is what pins that.
        """
        with agent_span(
            "agent.root", input_value="x" * (DEFAULT_ATTRIBUTE_VALUE_LIMIT * 4)
        ):
            pass
        value = spans()["agent.root"].attributes["input.value"]
        assert 0 < len(value) <= DEFAULT_ATTRIBUTE_VALUE_LIMIT
        assert "truncated" in value

    def test_disabling_the_cap_does_not_blank_every_attribute(self, monkeypatch):
        """Regression: the cap used to be keyed off the SDK's OWN env var,
        `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`, with a `0 = disabled` convention.
        The SDK reads that variable literally — 0 means truncate to ZERO chars —
        so switching the cap off blanked every attribute on every span, the
        exact "arrives and renders as nothing" failure this module prevents.
        Our knob has its own name now.
        """
        monkeypatch.setenv("MIIFLOW_SPAN_ATTRIBUTE_LIMIT", "0")
        monkeypatch.delenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", raising=False)
        assert attribute_value_limit() is None
        assert span_limits() is None

        exporter = InMemorySpanExporter()
        provider = trace_sdk.TracerProvider(span_limits=span_limits())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("miiflow.agent")
        monkeypatch.setattr(trace, "get_tracer", lambda *a, **k: tracer)

        with agent_span("agent.root", input_value="x" * 100_000):
            pass
        value = exporter.get_finished_spans()[0].attributes["input.value"]
        # Uncapped means the WHOLE value, not an empty string.
        assert len(value) == 100_000

    def test_span_limits_bound_spans_this_package_does_not_author(self):
        """The per-attribute cap in `agent_span` cannot reach OpenInference's
        LLM spans, which carry the whole prompt twice — SpanLimits is what
        bounds those, and an unbounded attribute is how a span grows past what
        a collector accepts (a rejection an OTLP exporter never raises on)."""
        limits = span_limits()
        assert limits is not None
        assert limits.max_attribute_length == 32_000


class TestNoTracerIsANoOp:
    def test_yields_none_and_runs_the_body(self, monkeypatch):
        def _explode(*_a, **_k):
            raise ImportError("opentelemetry not installed")

        monkeypatch.setattr(trace, "get_tracer", _explode)
        ran = []
        with agent_span("agent.root", input_value="x") as span:
            ran.append(span)
        assert ran == [None]
        # The output helper must tolerate the no-op span too.
        set_span_output(None, "anything")


class TestToolSpans:
    """Tool executions used to leave no span at all — a 15-minute investigator
    trace was one AGENT span with LLM children and nothing between them."""

    def test_success_records_name_input_output_and_ok(self, spans):
        from types import SimpleNamespace

        from opentelemetry.trace import StatusCode

        from miiflow_agent.core.observability.spans import record_tool_result, tool_span

        with tool_span("google_ads_query", {"customer_id": "123", "query": "SELECT 1"}) as span:
            record_tool_result(
                span,
                SimpleNamespace(
                    success=True, error=None, output={"rows": [1, 2]},
                    metadata={"observation_ref": "agent_obs_x"},
                ),
            )
        s = spans()["tool.google_ads_query"]
        assert s.attributes["openinference.span.kind"] == "TOOL"
        assert s.attributes["tool.name"] == "google_ads_query"
        assert '"customer_id":"123"' in s.attributes["input.value"]
        assert '"rows":[1,2]' in s.attributes["output.value"]
        assert s.attributes["tool.observation_ref"] == "agent_obs_x"
        assert s.attributes["tool.outcome"] == "ok"
        assert s.status.status_code == StatusCode.OK

    def test_failed_result_marks_error_with_tool_message(self, spans):
        from types import SimpleNamespace

        from opentelemetry.trace import StatusCode

        from miiflow_agent.core.observability.spans import record_tool_result, tool_span

        with tool_span("meta_ads_insights", {}) as span:
            record_tool_result(
                span, SimpleNamespace(success=False, error="account not found", output=None, metadata={})
            )
        s = spans()["tool.meta_ads_insights"]
        assert s.status.status_code == StatusCode.ERROR
        assert "account not found" in s.status.description
        assert s.attributes["tool.outcome"] == "error"

    def test_approval_pause_is_not_an_error(self, spans):
        from opentelemetry.trace import StatusCode

        from miiflow_agent.core.observability.spans import tool_span
        from miiflow_agent.core.react.exceptions import ToolApprovalRequired

        with pytest.raises(ToolApprovalRequired):
            with tool_span("update_campaign_budget", {"budget": 10}):
                raise ToolApprovalRequired("update_campaign_budget", {"budget": 10})
        s = spans()["tool.update_campaign_budget"]
        assert s.status.status_code == StatusCode.OK
        assert s.attributes["tool.outcome"] == "paused"

    def test_unexpected_raise_is_an_error(self, spans):
        from opentelemetry.trace import StatusCode

        from miiflow_agent.core.observability.spans import tool_span

        with pytest.raises(RuntimeError):
            with tool_span("boom", {}):
                raise RuntimeError("kaboom")
        s = spans()["tool.boom"]
        assert s.status.status_code == StatusCode.ERROR
        assert "kaboom" in s.status.description

    def test_large_output_is_excerpted(self, spans):
        from types import SimpleNamespace

        from miiflow_agent.core.observability.spans import (
            TOOL_OUTPUT_EXCERPT_CHARS,
            record_tool_result,
            tool_span,
        )

        with tool_span("big", {}) as span:
            record_tool_result(
                span, SimpleNamespace(success=True, error=None, output="x" * 50_000, metadata={})
            )
        out = spans()["tool.big"].attributes["output.value"]
        assert len(out) < TOOL_OUTPUT_EXCERPT_CHARS + 64
        assert "chars]" in out
