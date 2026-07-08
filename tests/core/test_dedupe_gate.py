"""Read-through dedupe gate (P2): serve contracts, TTLs, validity, single-flight.

The load-bearing guarantees:

  1. Only tools that DECLARED a serve contract are ever served; TTLs are
     per-class code-owned constants.
  2. Unsafe stored observations are never served: business-error payloads
     recorded as transport successes, and payloads carrying run-scoped
     ``_data_id`` render refs.
  3. Single-flight: concurrent identical calls coalesce onto one leader;
     waiters share the leader's result INCLUDING failures.
  4. Write/read identity symmetry: the ledger's ``inputs_hash`` (written via
     ``dedupe_identity``) matches what the gate computes on the read side —
     framework-injected keys like ``__description`` never break the match.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from miiflow_agent.core.checkpoint import Checkpoint, DispatchLedgerEntry
from miiflow_agent.core.observation import StoredObservation
from miiflow_agent.core.react.dedupe import (
    IDEMPOTENCY_TTLS_SECONDS,
    LedgerDedupeGate,
    dedupe_identity,
    observation_is_servable,
)


class _FakeSink:
    def __init__(self, store=None):
        self.store = store or {}
        self.fetch_calls = 0

    async def record(self, rec):
        return None

    async def fetch(self, ref):
        self.fetch_calls += 1
        return self.store.get(ref)


def _schema(idem="discovery", scope_dims=None):
    return SimpleNamespace(
        idempotency_class=idem, dedupe_scope_dims=list(scope_dims or [])
    )


def _stored(ref="agent_obs_1", text='{"accounts": [{"id": "123"}]}', success=True):
    return StoredObservation(
        ref=ref,
        observation_text=text,
        tool_name="list_all_ad_accounts",
        success=success,
        created_at_ts=time.time(),
    )


def _checkpoint_with_entry(
    tool_name="list_all_ad_accounts",
    inputs=None,
    ref="agent_obs_1",
    age_seconds=10.0,
    success=True,
):
    cp = Checkpoint(thread_id="thread_root")
    cp.merge_ledger(
        [
            DispatchLedgerEntry(
                kind="tool_call",
                tool_name=tool_name,
                inputs_hash=dedupe_identity(tool_name, inputs or {}),
                success=success,
                digest="…",
                observation_ref=ref,
                produced_at=time.time() - age_seconds,
                turn_index=1,
            )
        ]
    )
    return cp


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestIdentity:
    def test_framework_keys_excluded(self):
        a = dedupe_identity("t", {"platform": "google_ads"})
        b = dedupe_identity(
            "t", {"platform": "google_ads", "__description": "list accounts"}
        )
        assert a == b

    def test_scope_dims_fold_deps(self):
        base = dedupe_identity("t", {}, scope_dims=["assistant_id"], deps={"assistant_id": "a1"})
        other = dedupe_identity("t", {}, scope_dims=["assistant_id"], deps={"assistant_id": "a2"})
        assert base != other


class TestServability:
    def test_rejects_data_id_payloads(self):
        assert not observation_is_servable('{"results": [], "_data_id": "abc"}')

    def test_rejects_business_error_payloads(self):
        assert not observation_is_servable('{"error": "account not found"}')
        assert not observation_is_servable("{'error': 'nope'}")

    def test_accepts_normal_payloads(self):
        assert observation_is_servable('{"accounts": [], "total_count": 0}')


class TestTryServe:
    def test_serves_fresh_valid_entry(self):
        cp = _checkpoint_with_entry(inputs={})
        sink = _FakeSink({"agent_obs_1": _stored()})
        gate = LedgerDedupeGate(cp, sink)

        served = _run(gate.try_serve("list_all_ad_accounts", {}, _schema()))
        assert served is not None
        assert served["observation_ref"] == "agent_obs_1"
        assert "accounts" in served["output"]

    def test_no_contract_never_served(self):
        cp = _checkpoint_with_entry(inputs={})
        sink = _FakeSink({"agent_obs_1": _stored()})
        gate = LedgerDedupeGate(cp, sink)

        served = _run(gate.try_serve("list_all_ad_accounts", {}, _schema(idem="none")))
        assert served is None

    def test_expired_entry_not_served(self):
        ttl = IDEMPOTENCY_TTLS_SECONDS["discovery"]
        cp = _checkpoint_with_entry(inputs={}, age_seconds=ttl + 5)
        sink = _FakeSink({"agent_obs_1": _stored()})
        gate = LedgerDedupeGate(cp, sink)

        assert _run(gate.try_serve("list_all_ad_accounts", {}, _schema())) is None

    def test_failed_entry_not_served(self):
        cp = _checkpoint_with_entry(inputs={}, success=False)
        sink = _FakeSink({"agent_obs_1": _stored()})
        gate = LedgerDedupeGate(cp, sink)

        assert _run(gate.try_serve("list_all_ad_accounts", {}, _schema())) is None

    def test_invalid_stored_payload_not_served(self):
        cp = _checkpoint_with_entry(inputs={})
        sink = _FakeSink(
            {"agent_obs_1": _stored(text='{"error": "token expired"}')}
        )
        gate = LedgerDedupeGate(cp, sink)

        assert _run(gate.try_serve("list_all_ad_accounts", {}, _schema())) is None

    def test_dispatch_entries_never_served(self):
        cp = Checkpoint(thread_id="thread_root")
        cp.merge_ledger(
            [
                DispatchLedgerEntry(
                    kind="dispatch",
                    handle="google_ads_specialist",
                    task_hash="th1",
                    observation_ref="agent_obs_1",
                    produced_at=time.time(),
                )
            ]
        )
        sink = _FakeSink({"agent_obs_1": _stored()})
        gate = LedgerDedupeGate(cp, sink)
        # A tool_call lookup never matches a dispatch entry's dedupe key.
        assert _run(gate.try_serve("google_ads_specialist", {}, _schema())) is None

    def test_different_inputs_not_served(self):
        cp = _checkpoint_with_entry(inputs={"platform": "google_ads"})
        sink = _FakeSink({"agent_obs_1": _stored()})
        gate = LedgerDedupeGate(cp, sink)

        served = _run(
            gate.try_serve(
                "list_all_ad_accounts", {"platform": "meta_ads"}, _schema()
            )
        )
        assert served is None


class TestSingleFlight:
    def test_concurrent_identical_calls_coalesce(self):
        async def go():
            cp = Checkpoint(thread_id="thread_root")
            gate = LedgerDedupeGate(cp, _FakeSink())
            schema = _schema()
            executions = []

            async def call(i):
                key = gate.inflight_key("t", {"q": 1}, schema)
                assert key is not None
                leader = gate.claim(key)
                if leader is not None:
                    return await leader
                await asyncio.sleep(0.01)  # simulate execution latency
                result = f"result_{i}"
                executions.append(i)
                gate.resolve(key, result)
                return result

            results = await asyncio.gather(*[call(i) for i in range(5)])
            assert len(executions) == 1
            assert len(set(results)) == 1
            return True

        assert _run(go())

    def test_leader_failure_propagates_to_waiters(self):
        async def go():
            cp = Checkpoint(thread_id="thread_root")
            gate = LedgerDedupeGate(cp, _FakeSink())
            schema = _schema()
            key = gate.inflight_key("t", {"q": 1}, schema)
            assert gate.claim(key) is None  # leader
            waiter = gate.claim(key)
            assert waiter is not None
            gate.resolve_error(key, RuntimeError("boom"))
            with pytest.raises(RuntimeError):
                await waiter
            # Key deregistered — the next call becomes a fresh leader.
            assert gate.claim(key) is None
            return True

        assert _run(go())

    def test_no_contract_no_coalescing(self):
        cp = Checkpoint(thread_id="thread_root")
        gate = LedgerDedupeGate(cp, _FakeSink())
        assert gate.inflight_key("t", {}, _schema(idem="none")) is None


class TestLedgerV2:
    def test_legacy_entry_tolerated_and_digested(self):
        legacy = {
            "kind": "tool_call",
            "tool_name": "google_ads_query",
            "inputs_hash": "h1",
            "success": True,
            "observation": "x" * 5000,  # v1 fat payload
            "turn": 3,
        }
        entry = DispatchLedgerEntry.from_dict(legacy)
        assert len(entry.digest) <= 400
        assert entry.turn_index == 3
        assert entry.observation_ref is None

    def test_digest_cap_is_serializer_enforced(self):
        entry = DispatchLedgerEntry(kind="tool_call", tool_name="t", digest="y" * 2000)
        assert len(entry.digest) <= 400
        assert len(entry.to_dict()["digest"]) <= 400

    def test_prune_drops_stale_turns_and_caps_count(self):
        cp = Checkpoint(thread_id="t", turn_index=10)
        entries = [
            DispatchLedgerEntry(
                kind="tool_call", tool_name=f"t{i}", inputs_hash=str(i), turn_index=i
            )
            for i in range(1, 11)
        ]
        cp.merge_ledger(entries)
        cp.prune_ledger()
        kept_turns = {e.turn_index for e in cp.dispatch_ledger}
        assert min(kept_turns) >= 10 - Checkpoint.LEDGER_KEEP_TURNS
        # Count cap
        cp2 = Checkpoint(thread_id="t", turn_index=1)
        cp2.merge_ledger(
            [
                DispatchLedgerEntry(
                    kind="tool_call", tool_name="t", inputs_hash=str(i), turn_index=1
                )
                for i in range(Checkpoint.LEDGER_MAX_ENTRIES + 50)
            ]
        )
        cp2.prune_ledger()
        assert len(cp2.dispatch_ledger) == Checkpoint.LEDGER_MAX_ENTRIES


class TestWorklogDelta:
    def test_exclude_produced_by_filters_own_entries(self):
        cp = Checkpoint(thread_id="root", turn_index=1)
        cp.merge_ledger(
            [
                DispatchLedgerEntry(
                    kind="tool_call",
                    tool_name="list_all_ad_accounts",
                    inputs_hash="h1",
                    digest="accounts",
                    produced_by_path=["root"],
                    turn_index=1,
                ),
                DispatchLedgerEntry(
                    kind="tool_call",
                    tool_name="google_ads_query",
                    inputs_hash="h2",
                    digest="child's own rows",
                    produced_by_path=["child", "thread_c1"],
                    turn_index=1,
                ),
            ]
        )
        full = cp.render_worklog_block()
        assert "google_ads_query" in full and "list_all_ad_accounts" in full

        delta = cp.render_worklog_block(exclude_produced_by=["child", "thread_c1"])
        assert "list_all_ad_accounts" in delta
        assert "google_ads_query" not in delta

        # Excluding everything renders no block at all.
        only_child = Checkpoint(thread_id="root", turn_index=1)
        only_child.merge_ledger(
            [
                DispatchLedgerEntry(
                    kind="tool_call",
                    tool_name="t",
                    inputs_hash="h",
                    digest="d",
                    produced_by_path=["child", "thread_c1"],
                    turn_index=1,
                )
            ]
        )
        assert (
            only_child.render_worklog_block(
                exclude_produced_by=["child", "thread_c1"]
            )
            == ""
        )
