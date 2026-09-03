"""Tests for the canonical observation-store port (core/observation.py).

Pins the P1 contract:
  1. ``get_observation_sink`` resolves the deps-injected sink and degrades to
     None when absent / malformed (no sink ⇒ prior behavior everywhere).
  2. The orchestrator's ``_record_tool_observation`` seam awaits the sink
     inline, threads the returned ref out, swallows sink failures, and still
     reduces the call into the checkpoint ledger either way.
  3. ``EventFactory.observation`` carries the additive ``observation_ref`` /
     ``served_from_ledger`` fields without breaking legacy callers.
  4. The seam returns the BOUNDED observation alongside the ref, so an
     oversized tool result cannot reach ``context.messages`` intact. The store
     keeps the full copy; only the string travelling to the model is clamped.
"""

import asyncio
from types import SimpleNamespace

import pytest

from miiflow_agent.core.checkpoint import Checkpoint
from miiflow_agent.core.observation import (
    LLM_OBSERVATION_MAX_CHARS,
    ObservationRecord,
    StoredObservation,
    bound_observation_for_llm,
    get_observation_sink,
)
from miiflow_agent.core.react.events.bus import EventFactory
from miiflow_agent.core.react.enums import ReActEventType


class _RecordingSink:
    def __init__(self, ref="agent_obs_test123", raise_on_record=False):
        self.records = []
        self._ref = ref
        self._raise = raise_on_record

    async def record(self, rec: ObservationRecord):
        if self._raise:
            raise RuntimeError("sink exploded")
        self.records.append(rec)
        return self._ref

    async def fetch(self, ref):
        return StoredObservation(
            ref=ref,
            observation_text="full text",
            tool_name="t",
            success=True,
            created_at_ts=0.0,
        )


def _context(sink=None, checkpoint=None):
    deps = {}
    if sink is not None:
        deps["observation_sink"] = sink
    return SimpleNamespace(deps=deps, checkpoint=checkpoint)


def _state(step=3):
    return SimpleNamespace(current_step=step)


def _record(orchestrator_self, context, state, **kwargs):
    from miiflow_agent.core.react.orchestrator import ReActOrchestrator

    return ReActOrchestrator._record_tool_observation(
        orchestrator_self, context, state, **kwargs
    )


class TestGetObservationSink:
    def test_resolves_sink_from_deps(self):
        sink = _RecordingSink()
        assert get_observation_sink(_context(sink)) is sink

    def test_absent_sink_is_none(self):
        assert get_observation_sink(_context()) is None

    def test_context_without_deps_is_none(self):
        assert get_observation_sink(SimpleNamespace()) is None

    def test_malformed_sink_is_none(self):
        ctx = SimpleNamespace(deps={"observation_sink": object()})
        assert get_observation_sink(ctx) is None


class TestRecordToolObservationSeam:
    def _run(self, coro):
        return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)

    def test_records_and_returns_ref_and_merges_ledger(self):
        sink = _RecordingSink(ref="agent_obs_abc")
        cp = Checkpoint(thread_id="thread_x")
        ctx = _context(sink, checkpoint=cp)
        orch = SimpleNamespace()

        recorded = self._run(
            _record(
                orch,
                ctx,
                _state(step=7),
                tool_name="list_all_ad_accounts",
                inputs={"platform": "google_ads"},
                observation="3 accounts",
                success=True,
                tool_call_id="tc_1",
                raw_output={"accounts": []},
                execution_time_ms=42,
            )
        )

        assert recorded.ref == "agent_obs_abc"
        # The observation the model reads opens with its citation key; the
        # stored row keeps the same label as a field (asserted below).
        assert recorded.observation == "[ref:ad_accounts_1]\n3 accounts"
        assert len(sink.records) == 1
        rec = sink.records[0]
        assert rec.tool_name == "list_all_ad_accounts"
        assert rec.tool_call_id == "tc_1"
        assert rec.observation_text == "3 accounts"
        assert rec.reference_label == "ad_accounts_1"
        assert rec.raw_output == {"accounts": []}
        assert rec.step_number == 7
        # Ledger reduced regardless of sink
        assert len(cp.dispatch_ledger) == 1
        assert cp.dispatch_ledger[0].tool_name == "list_all_ad_accounts"

    def test_absent_sink_returns_none_but_still_merges_ledger(self):
        cp = Checkpoint(thread_id="thread_x")
        ctx = _context(sink=None, checkpoint=cp)

        recorded = self._run(
            _record(
                SimpleNamespace(),
                ctx,
                _state(),
                tool_name="google_ads_query",
                inputs={},
                observation="rows",
                success=True,
            )
        )

        assert recorded.ref is None
        assert recorded.observation == "[ref:ads_query_1]\nrows"
        assert len(cp.dispatch_ledger) == 1

    def test_sink_failure_is_swallowed(self):
        sink = _RecordingSink(raise_on_record=True)
        cp = Checkpoint(thread_id="thread_x")
        ctx = _context(sink, checkpoint=cp)

        recorded = self._run(
            _record(
                SimpleNamespace(),
                ctx,
                _state(),
                tool_name="t",
                inputs={},
                observation="x",
                success=False,
                error="boom",
            )
        )

        assert recorded.ref is None
        assert len(cp.dispatch_ledger) == 1

    def test_no_tool_name_is_noop(self):
        sink = _RecordingSink()
        cp = Checkpoint(thread_id="thread_x")
        ctx = _context(sink, checkpoint=cp)

        recorded = self._run(
            _record(
                SimpleNamespace(),
                ctx,
                _state(),
                tool_name=None,
                inputs={},
                observation="x",
                success=True,
            )
        )

        assert recorded.ref is None
        # No tool name means nothing to cite — no label, no prefix.
        assert recorded.observation == "x"
        assert sink.records == []
        assert cp.dispatch_ledger == []

    def test_oversized_observation_is_bounded_for_the_model_not_the_store(self):
        """The 2026-07-31 incident in one test.

        A 2.35 MB GAQL result must reach the sink whole (the row is the one
        full copy) while the string handed back for ``context.messages`` is
        clamped — the unbounded append is what produced a 1,110,727-token
        request against a 1,000,000-token ceiling.
        """
        sink = _RecordingSink(ref="agent_obs_big")
        cp = Checkpoint(thread_id="thread_x")
        ctx = _context(sink, checkpoint=cp)
        huge = "x" * 2_349_866

        recorded = self._run(
            _record(
                SimpleNamespace(),
                ctx,
                _state(step=8),
                tool_name="google_ads_query",
                inputs={"query": "SELECT ... FROM campaign_search_term_insight"},
                observation=huge,
                success=True,
                tool_call_id="tc_big",
            )
        )

        assert sink.records[0].observation_text == huge
        assert len(recorded.observation) < len(huge)
        assert recorded.observation.startswith("[ref:ads_query_1]\n" + "x" * 1000)
        assert "truncated" in recorded.observation

    def test_sink_excerpt_policy_wins_over_the_framework_default(self):
        class _PolicySink(_RecordingSink):
            def llm_excerpt(self, *, text, tool_name, ref):
                return f"[{tool_name}:{ref}] {text[:10]}"

        sink = _PolicySink(ref="agent_obs_p")
        ctx = _context(sink, checkpoint=Checkpoint(thread_id="thread_x"))

        recorded = self._run(
            _record(
                SimpleNamespace(),
                ctx,
                _state(),
                tool_name="google_ads_query",
                inputs={},
                observation="0123456789abcdef",
                success=True,
            )
        )

        # Labelling happens AFTER the sink's excerpt policy runs, so the
        # citation key survives whatever truncation the policy applies.
        assert recorded.observation == (
            "[ref:ads_query_1]\n[google_ads_query:agent_obs_p] 0123456789"
        )


class TestReferenceLabelling:
    """Every observation the model reads opens with the key it must cite.

    The prompt tells the model "tool results open with a reference tag like
    [ref:tool_name_N] ... append that exact tag", and citation_processor
    resolves those markers against the timeline. This path emitted no tag, so
    the model guessed labels, nothing resolved, and raw `[ref:google_ads_query_2]`
    shipped inside a client-facing audit.
    """

    def _run(self, coro):
        return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)

    def _record_one(self, sink, state, tool_name, observation):
        return self._run(
            _record(
                SimpleNamespace(),
                _context(sink, checkpoint=Checkpoint(thread_id="t")),
                state,
                tool_name=tool_name,
                inputs={},
                observation=observation,
                success=True,
            )
        )

    def test_labels_increment_per_tool_within_a_run(self):
        sink = _RecordingSink(ref="agent_obs_1")
        state = _state()
        first = self._record_one(sink, state, "meta_ads_insights", "a")
        second = self._record_one(sink, state, "meta_ads_insights", "b")
        other = self._record_one(sink, state, "google_ads_query", "c")

        assert first.observation.startswith("[ref:ads_insights_1]")
        assert second.observation.startswith("[ref:ads_insights_2]")
        assert other.observation.startswith("[ref:ads_query_1]")

    def test_artifact_receipts_are_not_labelled(self):
        """A `[VIZ:id]` marker is a render receipt, not a source — and
        `sse_converters` matches that marker ANCHORED, so a prefix in front of
        it would stop the visualization from being recognised at all."""
        sink = _RecordingSink(ref="agent_obs_1")
        state = _state()
        viz = self._record_one(sink, state, "render_table", "[VIZ:abc-123]")
        assert viz.observation == "[VIZ:abc-123]"
        assert sink.records[-1].reference_label is None

        # ...and the skipped call consumes no number, so keys stay dense over
        # the calls that actually produced data.
        real = self._record_one(sink, state, "render_table", "rows")
        assert real.observation.startswith("[ref:render_table_1]")

    def test_an_existing_label_is_not_doubled(self):
        sink = _RecordingSink(ref="agent_obs_1")
        recorded = self._record_one(
            sink, _state(), "google_ads_query", "[ref:already_9]\nrows"
        )
        assert recorded.observation == "[ref:already_9]\nrows"


class TestBoundObservationForLlm:
    def test_short_text_passes_through(self):
        assert bound_observation_for_llm(None, "hi", tool_name="t", ref=None) == "hi"

    def test_none_and_empty_are_safe(self):
        assert bound_observation_for_llm(None, None, tool_name="t", ref=None) == ""
        assert bound_observation_for_llm(None, "", tool_name="t", ref=None) == ""

    def test_framework_default_clamps_without_a_sink(self):
        text = "y" * (LLM_OBSERVATION_MAX_CHARS + 5_000)
        bounded = bound_observation_for_llm(None, text, tool_name="t", ref=None)
        assert len(bounded) < len(text)
        assert bounded.startswith("y" * 100)
        assert "5000 chars" in bounded

    def test_broken_sink_policy_degrades_to_the_framework_bound(self):
        class _Exploding:
            def llm_excerpt(self, **kwargs):
                raise RuntimeError("policy exploded")

        text = "z" * (LLM_OBSERVATION_MAX_CHARS * 2)
        bounded = bound_observation_for_llm(
            _Exploding(), text, tool_name="t", ref="r"
        )
        assert len(bounded) < len(text)
        assert bounded.count("z") == LLM_OBSERVATION_MAX_CHARS

    def test_non_string_sink_policy_degrades_to_the_framework_bound(self):
        class _Wrong:
            def llm_excerpt(self, **kwargs):
                return {"not": "a string"}

        bounded = bound_observation_for_llm(_Wrong(), "short", tool_name="t", ref="r")
        assert bounded == "short"


class TestObservationEventFields:
    def test_event_carries_ref_and_served_flag(self):
        event = EventFactory.observation(
            2,
            "obs",
            "tool_a",
            True,
            tool_call_id="tc_9",
            observation_ref="agent_obs_r1",
            served_from_ledger=True,
        )
        assert event.event_type == ReActEventType.OBSERVATION
        assert event.data["observation_ref"] == "agent_obs_r1"
        assert event.data["served_from_ledger"] is True

    def test_legacy_call_shape_still_works(self):
        event = EventFactory.observation(1, "obs", "tool_a", True)
        assert event.data["observation_ref"] is None
        assert event.data["served_from_ledger"] is False
