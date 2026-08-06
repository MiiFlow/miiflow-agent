"""Tests for the file-backed observation sink and the ref-aware bound.

Standalone deployments used to lose everything past the truncation ceiling
("re-run the tool with a narrower scope"); the local sink spills oversized
outputs to disk and the truncation marker points at the stored copy.
"""

import pytest

from miiflow_agent.core.observation import (
    LLM_OBSERVATION_MAX_CHARS,
    ObservationRecord,
    bound_observation_for_llm,
)
from miiflow_agent.core.observation_local import (
    LocalFileObservationSink,
    make_read_observation_tool,
)


def _record(text, tool_name="big_dump"):
    return ObservationRecord(
        tool_name=tool_name, tool_call_id="t1", inputs={}, observation_text=text
    )


@pytest.fixture
def sink(tmp_path):
    return LocalFileObservationSink(
        directory=str(tmp_path), spill_threshold_chars=1_000
    )


class TestLocalFileObservationSink:
    async def test_small_output_is_not_spilled(self, sink):
        assert await sink.record(_record("small")) is None

    async def test_large_output_round_trips(self, sink):
        text = "R" * 5_000
        ref = await sink.record(_record(text))
        assert ref is not None and ref.startswith("local_obs_")

        stored = await sink.fetch(ref)
        assert stored is not None
        assert stored.observation_text == text
        assert stored.tool_name == "big_dump"

    async def test_fetch_rejects_foreign_and_malformed_refs(self, sink):
        assert await sink.fetch("../../etc/passwd") is None
        assert await sink.fetch("agent_obs_123") is None
        assert await sink.fetch("local_obs_" + "0" * 32) is None  # missing
        assert await sink.fetch("") is None

    async def test_llm_excerpt_bounds_and_names_the_ref(self, sink):
        text = "R" * 5_000
        ref = await sink.record(_record(text))
        excerpt = sink.llm_excerpt(text=text, tool_name="big_dump", ref=ref)

        assert len(excerpt) < len(text)
        assert f'read_observation(ref="{ref}")' in excerpt

    async def test_llm_excerpt_passthrough_without_ref(self, sink):
        # Spill failed → no ref → defer to the framework ceiling rather than
        # promising a ref that cannot resolve.
        text = "R" * 5_000
        assert sink.llm_excerpt(text=text, tool_name="x", ref=None) == text

    async def test_small_text_untouched(self, sink):
        assert sink.llm_excerpt(text="ok", tool_name="x", ref="local_obs_x") == "ok"


class TestRefAwareFallbackMarker:
    def test_marker_points_at_ref_when_available(self):
        class _SinkWithoutPolicy:  # no llm_excerpt
            pass

        text = "X" * (LLM_OBSERVATION_MAX_CHARS + 100)
        bounded = bound_observation_for_llm(
            _SinkWithoutPolicy(), text, tool_name="t", ref="agent_obs_42"
        )
        assert 'read_observation(ref="agent_obs_42")' in bounded
        assert "narrower scope" not in bounded

    def test_marker_says_rerun_when_no_ref(self):
        text = "X" * (LLM_OBSERVATION_MAX_CHARS + 100)
        bounded = bound_observation_for_llm(None, text, tool_name="t", ref=None)
        assert "narrower scope" in bounded
        assert "read_observation" not in bounded


class TestReadObservationTool:
    async def test_reads_back_through_the_sink(self, sink):
        text = "R" * 5_000
        ref = await sink.record(_record(text))

        ctx = type("Ctx", (), {"deps": {"observation_sink": sink}})()
        tool_fn = make_read_observation_tool()
        result = await tool_fn(ctx, ref=ref)

        assert result["observation"] == text
        assert result["tool_name"] == "big_dump"
        assert result["total_chars"] == len(text)
        assert "next_offset" not in result  # fits in one page

    async def test_large_observation_pages_with_offset(self, sink):
        from miiflow_agent.core.observation_local import (
            READ_OBSERVATION_PAGE_CHARS,
        )

        text = "R" * (READ_OBSERVATION_PAGE_CHARS + 500)
        ref = await sink.record(_record(text))
        ctx = type("Ctx", (), {"deps": {"observation_sink": sink}})()
        tool_fn = make_read_observation_tool()

        first = await tool_fn(ctx, ref=ref)
        assert len(first["observation"]) == READ_OBSERVATION_PAGE_CHARS
        assert first["next_offset"] == READ_OBSERVATION_PAGE_CHARS

        second = await tool_fn(ctx, ref=ref, offset=first["next_offset"])
        assert second["observation"] == "R" * 500
        assert "next_offset" not in second
        # The two pages reassemble the full stored text.
        assert first["observation"] + second["observation"] == text

    async def test_read_observation_results_are_never_respilled(self, sink):
        """The tool's own output must not mint a fresh ref/file — that was
        an unbounded loop of duplicate spills."""
        rec = ObservationRecord(
            tool_name="read_observation",
            tool_call_id="t9",
            inputs={},
            observation_text="R" * 10_000,  # over the spill threshold
        )
        assert await sink.record(rec) is None

    async def test_missing_sink_and_missing_ref(self, sink):
        tool_fn = make_read_observation_tool()

        no_sink_ctx = type("Ctx", (), {"deps": {}})()
        assert "error" in await tool_fn(no_sink_ctx, ref="local_obs_x")

        ctx = type("Ctx", (), {"deps": {"observation_sink": sink}})()
        assert "error" in await tool_fn(ctx, ref="local_obs_" + "f" * 32)

    def test_tool_is_registrable(self):
        from miiflow_agent.core.tools import ToolRegistry

        registry = ToolRegistry()
        registry.register(make_read_observation_tool())
        assert "read_observation" in registry.tools
