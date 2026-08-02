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
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

TRACER_NAME = "miiflow.agent"


@contextmanager
def agent_span(name: str, **attributes: Any) -> Iterator[Optional[Any]]:
    """Open a span for one agent run. A no-op when tracing is not configured.

    Failures in span SETUP degrade to a no-op; failures inside the wrapped body
    propagate untouched. The distinction is deliberate — a blanket try/except
    around the yield would swallow real errors from the code being traced and
    turn an observability feature into a source of silent failure, which is the
    exact class of bug this whole effort exists to surface.
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
        try:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
        except Exception:
            pass
        yield span
