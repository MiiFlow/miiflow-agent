"""An agent span must not lose its children to the task that drives the stream.

Production 2026-08-04, trace 59f23413…3c93d480: a lone `agent.adlyse_ai_2` span,
1m42s long, and none of the eleven LLM calls it made anywhere in the trace.

Root cause: `agent_span` is a SYNCHRONOUS context manager and was used inside an
async generator (`_stream_response_inner`), so its `attach()` ran during the
first `__anext__` and its `detach()` during a later one. `contextvars` are
per-TASK, not per-generator — advance the generator from a different task and
the span is no longer current when the body runs, so OpenInference's LLM spans
start their own root trace. OTel announces it: "Failed to detach context: Token
was created in a different Context".

`traced_stream` attaches and detaches INSIDE each `__anext__`, so no context is
held across a suspension.

No network: the "LLM call" is a child span, since the question is purely about
context propagation.
"""
import asyncio

import pytest

from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from miiflow_agent.core.observability.spans import agent_span, traced_stream


@pytest.fixture
def harness(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("miiflow.agent")
    monkeypatch.setattr(trace, "get_tracer", lambda *a, **k: tracer)

    def traces():
        out = {}
        for s in exporter.get_finished_spans():
            out.setdefault(s.get_span_context().trace_id, set()).add(s.name)
        return list(out.values())

    return tracer, traces


async def _body(tracer, tag):
    """Yields, THEN calls the 'LLM' — i.e. on a later `__anext__`."""
    yield "first"
    with tracer.start_as_current_span(f"llm.{tag}"):
        pass
    yield "second"


async def _drive_across_tasks(agen):
    """Advance the generator from a SEPARATE task each time.

    This is the shape that breaks contextvar-based span propagation; an ASGI
    server, a pump task or a task-group consumer can all produce it.
    """
    while True:
        try:
            await asyncio.create_task(agen.__anext__())
        except StopAsyncIteration:
            return


def _nests(traces, tag):
    return any({f"agent.{tag}", f"llm.{tag}"} <= names for names in traces)


class TestTracedStreamKeepsItsChildren:
    @pytest.mark.parametrize("tag", ["case"])
    def test_children_nest_when_driven_from_another_task(self, harness, tag):
        tracer, traces = harness

        async def wrapped():
            async for ev in traced_stream(f"agent.{tag}", _body(tracer, tag)):
                yield ev

        asyncio.run(_drive_across_tasks(wrapped()))
        assert _nests(traces(), tag), "LLM span was orphaned into its own trace"

    def test_the_old_shape_is_what_broke(self, harness):
        """Probe: the same drive against `with agent_span(...)` must FAIL to nest.

        Without this, the test above could pass because the harness cannot
        observe orphaning at all, and would go quiet the moment the bug returned.
        """
        tracer, traces = harness
        tag = "legacy"

        async def wrapped():
            with agent_span(f"agent.{tag}"):
                async for ev in _body(tracer, tag):
                    yield ev

        asyncio.run(_drive_across_tasks(wrapped()))
        assert not _nests(traces(), tag), (
            "the sync-contextmanager shape nested anyway — this probe can no "
            "longer detect the regression it exists to catch"
        )

    def test_same_task_consumer_still_works(self, harness):
        tracer, traces = harness
        tag = "simple"

        async def go():
            async for _ in traced_stream(f"agent.{tag}", _body(tracer, tag)):
                pass

        asyncio.run(go())
        assert _nests(traces(), tag)


class TestTracedStreamSemantics:
    def test_attributes_output_and_passthrough(self, harness):
        tracer, traces = harness
        seen = []

        async def src():
            yield 1
            yield 2

        async def go():
            async for item in traced_stream(
                "agent.attrs",
                src(),
                input_value="ask",
                session_id="thread_1",
                output_getter=lambda: "answered",
                **{"agent.handle": "root"},
            ):
                seen.append(item)

        asyncio.run(go())
        assert seen == [1, 2], "items must pass through untouched"

    def test_output_is_recorded_even_when_the_stream_raises(self, harness):
        """A turn that dies mid-stream is the one an operator most needs to read."""
        tracer, traces = harness

        async def src():
            yield 1
            raise RuntimeError("provider 503")

        async def go():
            async for _ in traced_stream(
                "agent.boom", src(), output_getter=lambda: "partial answer"
            ):
                pass

        with pytest.raises(RuntimeError, match="503"):
            asyncio.run(go())

    def test_no_tracer_passes_items_through(self, harness, monkeypatch):
        def _explode(*_a, **_k):
            raise ImportError("opentelemetry not installed")

        monkeypatch.setattr(trace, "get_tracer", _explode)
        seen = []

        async def src():
            yield "a"

        async def go():
            async for item in traced_stream("agent.none", src()):
                seen.append(item)

        asyncio.run(go())
        assert seen == ["a"]
