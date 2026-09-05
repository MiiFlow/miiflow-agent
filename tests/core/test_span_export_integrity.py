"""Spans must leave the pipeline in a shape the collector accepts.

Sept 2026, production: Arize AX refused every root Adlyse AI LLM span with
``InvalidArgument: llm tool function arguments must be valid json`` — the
SDK's 32K per-value cap had hard-cut a tool call's serialized input, and the
conversation replayed that call in every later request. Nothing in the
suite could see it: the value cap, the count cap and the collector's parser
were each exercised in isolation. These tests drive real spans through the
production ``span_limits()`` + processor pipeline and assert on what an
exporter would receive.
"""

import json

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from miiflow_agent.core.observability.auto_instrumentation import describe_spans_for_log
from miiflow_agent.core.observability.processors import (
    JSON_REPAIRED_ATTR,
    install_processors,
    is_json_typed_attribute,
    repair_json_attribute_value,
)
from miiflow_agent.core.observability.spans import (
    ATTRIBUTE_COUNT_LIMIT_ENV,
    DEFAULT_ATTRIBUTE_COUNT_LIMIT,
    attribute_count_limit,
    attribute_value_limit,
    span_limits,
)

ARGS_KEY = "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
HISTORY_ARGS_KEY = "llm.input_messages.3.message.tool_calls.0.tool_call.function.arguments"


@pytest.fixture
def production_pipeline():
    """Real SpanLimits (value + count caps) and the production processors."""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider(span_limits=span_limits())
    install_processors(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _llm_attrs(**extra):
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "claude-sonnet-5",
        "input.value": "hello",
        "output.value": "calling a tool",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "dispatch_assistant",
    }
    attrs.update(extra)
    return attrs


class TestTruncatedToolArgumentsAreRepaired:
    def test_oversized_arguments_arrive_as_valid_json(self, production_pipeline):
        tracer, exporter = production_pipeline
        limit = attribute_value_limit()
        big = json.dumps({"task": "z" * (limit + 5_000)})
        with tracer.start_as_current_span("messages.create", attributes=_llm_attrs(**{ARGS_KEY: big})):
            pass
        (span,) = exporter.get_finished_spans()
        repaired = json.loads(span.attributes[ARGS_KEY])  # must parse
        assert repaired["__truncated__"] is True
        assert repaired["head"].startswith('{"task": "zzz')
        assert len(span.attributes[ARGS_KEY]) <= limit
        assert span.attributes[JSON_REPAIRED_ATTR] == 1

    def test_replayed_history_arguments_are_repaired_too(self, production_pipeline):
        # The call that broke a span is replayed in every later request of
        # the thread; without this the first big tool call costs the rest of
        # the conversation.
        tracer, exporter = production_pipeline
        limit = attribute_value_limit()
        attrs = _llm_attrs(
            **{
                ARGS_KEY: json.dumps({"task": "small"}),
                "llm.input_messages.3.message.role": "assistant",
                HISTORY_ARGS_KEY: json.dumps({"content": "y" * (limit + 100)}),
            }
        )
        with tracer.start_as_current_span("messages.create", attributes=attrs):
            pass
        (span,) = exporter.get_finished_spans()
        assert json.loads(span.attributes[HISTORY_ARGS_KEY])["__truncated__"] is True
        assert span.attributes[ARGS_KEY] == json.dumps({"task": "small"})
        assert span.attributes[JSON_REPAIRED_ATTR] == 1

    def test_valid_arguments_are_left_byte_identical(self, production_pipeline):
        tracer, exporter = production_pipeline
        args = json.dumps({"task": "x" * 500, "nested": {"a": [1, 2, 3]}})
        with tracer.start_as_current_span("messages.create", attributes=_llm_attrs(**{ARGS_KEY: args})):
            pass
        (span,) = exporter.get_finished_spans()
        assert span.attributes[ARGS_KEY] == args
        assert JSON_REPAIRED_ATTR not in span.attributes

    def test_free_text_attributes_are_not_json_typed(self):
        # `message.content` and `output.value` are prose; a cut there is
        # ugly, not a rejection, and must not be wrapped in an envelope.
        assert not is_json_typed_attribute("llm.input_messages.0.message.content")
        assert not is_json_typed_attribute("output.value")
        assert is_json_typed_attribute(ARGS_KEY)
        assert is_json_typed_attribute("llm.tools.12.tool.json_schema")
        assert is_json_typed_attribute("llm.invocation_parameters")

    def test_non_llm_span_untouched(self, production_pipeline):
        tracer, exporter = production_pipeline
        attrs = _llm_attrs(**{ARGS_KEY: '{"cut'})
        attrs["openinference.span.kind"] = "AGENT"
        with tracer.start_as_current_span("agent.root", attributes=attrs):
            pass
        (span,) = exporter.get_finished_spans()
        assert span.attributes[ARGS_KEY] == '{"cut'

    def test_envelope_fits_even_when_escaping_inflates_the_head(self):
        # A head full of quotes doubles in size when JSON-escaped; the
        # repaired value must still fit the cap or the SDK would cut it again.
        value = '{"q": "' + '"' * 40_000
        out = repair_json_attribute_value(value, 32_000)
        assert len(out) <= 32_000
        assert json.loads(out)["__truncated__"] is True


class TestAttributeCountCap:
    def test_default_lifts_the_sdk_128_cap(self, production_pipeline):
        tracer, exporter = production_pipeline
        attrs = _llm_attrs()
        for i in range(200):
            attrs[f"llm.tools.{i}.tool.json_schema"] = json.dumps({"name": f"t{i}"})
        # Output block LAST — this is what the 128 cap used to drop.
        attrs["llm.token_count.prompt"] = 10
        with tracer.start_as_current_span("messages.create", attributes=attrs):
            pass
        (span,) = exporter.get_finished_spans()
        assert span.attributes["llm.tools.199.tool.json_schema"] == json.dumps({"name": "t199"})
        assert span.attributes["llm.token_count.prompt"] == 10
        assert len(span.attributes) > 128

    def test_env_override_and_defaults(self, monkeypatch):
        assert attribute_count_limit() == DEFAULT_ATTRIBUTE_COUNT_LIMIT
        assert span_limits().max_attributes == DEFAULT_ATTRIBUTE_COUNT_LIMIT
        monkeypatch.setenv(ATTRIBUTE_COUNT_LIMIT_ENV, "300")
        assert attribute_count_limit() == 300
        monkeypatch.setenv(ATTRIBUTE_COUNT_LIMIT_ENV, "nope")
        assert attribute_count_limit() == DEFAULT_ATTRIBUTE_COUNT_LIMIT
        monkeypatch.setenv(ATTRIBUTE_COUNT_LIMIT_ENV, "0")
        assert attribute_count_limit() == DEFAULT_ATTRIBUTE_COUNT_LIMIT


class TestRefusedBatchDigest:
    def test_digest_names_the_span_and_what_is_missing(self, production_pipeline):
        tracer, exporter = production_pipeline
        attrs = {"openinference.span.kind": "LLM", "input.value": "x" * 100}
        with tracer.start_as_current_span("messages.create", attributes=attrs):
            pass
        (digest,) = describe_spans_for_log(exporter.get_finished_spans())
        assert digest["name"] == "messages.create"
        assert digest["kind"] == "LLM"
        assert digest["n_attrs"] == 2
        assert digest["largest"][0] == ("input.value", 100)
        assert "llm.token_count.prompt" in digest["missing"]
        assert "output.value" in digest["missing"]

    def test_digest_never_raises_on_odd_input(self):
        assert describe_spans_for_log([object()])[0]["name"] == "?"
        assert describe_spans_for_log([]) == []
