"""The provider-level span processors: context stamping and secret redaction.

Both run on EVERY span the provider exports — including the LLM/embedding
spans the OpenInference instrumentors author — so they are tested against a
real in-memory ``TracerProvider`` with the processors installed in the same
order production uses (stamp, redact, then export).
"""

import json

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from miiflow_agent.core.callback_context import callback_context
from miiflow_agent.core.callbacks import CallbackContext
from miiflow_agent.core.observability.processors import (
    REDACTED,
    context_attributes,
    install_processors,
    redact_text,
)


@pytest.fixture
def pipeline():
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    installed = install_processors(provider)
    assert installed == ["ContextStampingProcessor", "SecretRedactingProcessor"]
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return tracer, exporter


class TestContextStamping:
    def test_stamps_org_and_usage_session_from_ambient_context(self, pipeline):
        tracer, exporter = pipeline
        ctx = CallbackContext(
            organization_id="org_1",
            thread_id="thread_1",
            metadata={
                "usage_session_id": "us_9",
                "usage_session_type": "suggestion_pipeline",
            },
        )
        with callback_context(ctx):
            with tracer.start_as_current_span("ChatCompletion"):
                pass
        (span,) = exporter.get_finished_spans()
        assert span.attributes["organization.id"] == "org_1"
        assert span.attributes["thread.id"] == "thread_1"
        assert span.attributes["usage_session.id"] == "us_9"
        assert span.attributes["usage_session.type"] == "suggestion_pipeline"

    def test_explicit_span_attribute_wins_over_context(self, pipeline):
        tracer, exporter = pipeline
        ctx = CallbackContext(organization_id="org_from_ctx")
        with callback_context(ctx):
            with tracer.start_as_current_span(
                "agent.root", attributes={"organization.id": "org_explicit"}
            ):
                pass
        (span,) = exporter.get_finished_spans()
        assert span.attributes["organization.id"] == "org_explicit"

    def test_no_context_stamps_nothing(self, pipeline):
        tracer, exporter = pipeline
        with tracer.start_as_current_span("CreateEmbeddings"):
            pass
        (span,) = exporter.get_finished_spans()
        assert "organization.id" not in span.attributes

    def test_context_attributes_skips_empty_values(self):
        ctx = CallbackContext(organization_id="", metadata={"usage_session_id": None})
        assert context_attributes(ctx) == {}
        assert context_attributes(None) == {}


MCP_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6InB1YmxpYyJ9.eyJhdWQiOlsiaHR0cHMiXX0.SIGNATURE_PART_1234567890"


class TestSecretRedaction:
    def test_authorization_token_in_invocation_parameters_is_redacted(self, pipeline):
        tracer, exporter = pipeline
        params = {
            "max_tokens": 500,
            "mcp_servers": [
                {
                    "type": "url",
                    "url": "https://mcp.triplewhale.com/v1/mcp",
                    "name": "Triple Whale",
                    "authorization_token": MCP_TOKEN,
                }
            ],
        }
        with tracer.start_as_current_span("beta.messages.create") as span:
            span.set_attribute("llm.invocation_parameters", json.dumps(params))
            span.set_attribute("input.value", json.dumps({"messages": [], **params}))
        (finished,) = exporter.get_finished_spans()
        inv = json.loads(finished.attributes["llm.invocation_parameters"])
        assert inv["mcp_servers"][0]["authorization_token"] == REDACTED
        assert inv["mcp_servers"][0]["url"] == "https://mcp.triplewhale.com/v1/mcp"
        assert inv["max_tokens"] == 500
        assert MCP_TOKEN not in finished.attributes["input.value"]

    def test_non_payload_attributes_are_left_alone(self, pipeline):
        tracer, exporter = pipeline
        with tracer.start_as_current_span("x") as span:
            span.set_attribute("agent.handle", "authorization_token")
            span.set_attribute("llm.model_name", "claude-sonnet-5")
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes["agent.handle"] == "authorization_token"

    def test_clean_json_is_byte_stable(self):
        text = '{"max_tokens":500,"tools":[{"name":"list_all_ad_accounts"}]}'
        assert redact_text(text) == text

    def test_bearer_and_jwt_in_plain_text(self):
        assert redact_text("Authorization: Bearer abc.def-ghi") == (
            "Authorization: Bearer " + REDACTED
        )
        assert MCP_TOKEN not in redact_text(f"connect with {MCP_TOKEN} please")

    def test_narrow_key_match_keeps_token_counts(self):
        text = json.dumps({"max_tokens": 5, "usage": {"input_tokens": 3}, "api_key": "sk-1"})
        out = json.loads(redact_text(text))
        assert out["max_tokens"] == 5
        assert out["usage"]["input_tokens"] == 3
        assert out["api_key"] == REDACTED


class TestRedactionEdgeCases:
    def test_regexes_still_run_after_structural_redaction(self):
        """A payload with an mcp_servers token AND a bearer inside a message
        string must lose both — the structural pass looks at keys only."""
        payload = json.dumps({
            "mcp_servers": [{"authorization_token": MCP_TOKEN}],
            "messages": [{"role": "user", "content": f"use Authorization: Bearer {MCP_TOKEN} please"}],
        })
        out = redact_text(payload)
        assert MCP_TOKEN not in out
        assert out.count(REDACTED) >= 2

    def test_truncated_json_still_scrubs_key_value_pairs(self):
        """The SDK truncates attribute values at set time, so a long payload can
        arrive as an unparseable JSON prefix; opaque tokens match no regex."""
        opaque = "opaque-token-0123456789abcdef0123456789abcdef"
        text = (
            '{"messages":[{"role":"user","content":"x"}],"mcp_servers":[{"url":"u",'
            '"authorization_token":"' + opaque + '"}],"tools":[{"name":"a"'
        )
        out = redact_text(text)
        assert opaque not in out
        assert '"authorization_token":"' + REDACTED + '"' in out
