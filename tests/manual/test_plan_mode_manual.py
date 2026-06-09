#!/usr/bin/env python3
"""Manual end-to-end test for plan-mode under unified ReAct.

Exercises the Phase 2 primitives against a real LLM:

  * `enter_plan_mode` flips `RunState.permission_mode = "plan"` and the
    executor short-circuits non-read-only tools with a synthetic refusal.
  * A read-only tool still passes through plan mode.
  * `exit_plan_mode` restores `permission_mode = "default"` and emits
    PLAN_MODE_EXITED carrying the proposed plan text.
  * The model can self-route into plan mode and out without server-side
    orchestrator selection (the loop is always ReAct after the unified
    migration; this test exercises it end-to-end with plan-mode tools).

Usage:
    export OPENAI_API_KEY="sk-..."
    poetry run python tests/manual/test_plan_mode_manual.py

    export GOOGLE_API_KEY="..."
    poetry run python tests/manual/test_plan_mode_manual.py --provider gemini

    export ANTHROPIC_API_KEY="sk-ant-..."
    poetry run python tests/manual/test_plan_mode_manual.py --provider anthropic

    poetry run python tests/manual/test_plan_mode_manual.py --provider all
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from miiflow_agent import Agent, AgentType, LLMClient, RunContext
from miiflow_agent.core.config import AgentConfig
from miiflow_agent.core.react import ReActEventType
from miiflow_agent.core.tools import tool


PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "env_var": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "env_var": "ANTHROPIC_API_KEY",
    },
}


# Module-level state: tools mutate this so we can assert side-effects
# really were gated by plan mode. A simple counter beats a mock for a
# manual test against real LLMs.
_WRITE_LOG: List[dict] = []
_READ_LOG: List[dict] = []


@tool("list_project_files", "List files in the project. Read-only.")
def list_project_files() -> str:
    _READ_LOG.append({"call": "list_project_files"})
    return "Files: README.md, src/main.py, src/utils.py, tests/test_main.py"


@tool("read_file", "Read a file by path. Read-only.")
def read_file(path: str) -> str:
    _READ_LOG.append({"call": "read_file", "path": path})
    return f"// pretend contents of {path}\n# line 1\n# line 2\n"


@tool(
    "rewrite_file",
    "Rewrite a file with new contents. SIDE-EFFECTFUL — must not run "
    "during plan mode.",
)
def rewrite_file(path: str, contents: str) -> str:
    _WRITE_LOG.append({"call": "rewrite_file", "path": path, "contents": contents})
    return f"Wrote {len(contents)} chars to {path}"


# Mark the read-only tools so the plan-mode gate lets them through. The
# decorator builds a default ToolSchema with is_read_only=False; the
# manual flip is the same shape callers will use in production.
list_project_files._function_tool.definition.is_read_only = True
read_file._function_tool.definition.is_read_only = True
# rewrite_file deliberately stays is_read_only=False so the gate fires.


@dataclass
class CollectedEvents:
    """Minimal event capture: just the types we care about for plan-mode."""

    plan_entered: List[dict] = field(default_factory=list)
    plan_approval_needed: List[dict] = field(default_factory=list)
    plan_exited: List[dict] = field(default_factory=list)
    blocked: List[dict] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    all_event_types: List[str] = field(default_factory=list)

    def __call__(self, event):
        et = getattr(event, "event_type", None)
        if et is None:
            return
        et_value = et.value if hasattr(et, "value") else str(et)
        self.all_event_types.append(et_value)
        data = dict(getattr(event, "data", {}) or {})

        if et_value == "plan_mode_entered":
            self.plan_entered.append(data)
        elif et_value == "plan_approval_needed":
            self.plan_approval_needed.append(data)
        elif et_value == "plan_mode_exited":
            self.plan_exited.append(data)
        elif et_value == "tool_blocked_by_plan_mode":
            self.blocked.append(data)
        elif et_value == "action_planned":
            action = data.get("action") or ""
            if action:
                self.tool_calls.append(action)
        elif et_value == "final_answer":
            self.final_answer = data.get("answer", "")


def check_environment(provider_name: str) -> bool:
    env_var = PROVIDERS[provider_name]["env_var"]
    if not os.environ.get(env_var):
        print(f"SKIP: {env_var} not set for {provider_name}")
        return False
    print(f"Environment check passed: {env_var} is set")
    return True


def reset_logs() -> None:
    _WRITE_LOG.clear()
    _READ_LOG.clear()


def create_agent(provider_name: str) -> Agent:
    cfg = PROVIDERS[provider_name]
    client = LLMClient.create(
        cfg["provider"],
        model=cfg["model"],
        api_key=os.environ.get(cfg["env_var"]),
    )
    # AgentType is passed for telemetry labeling; the loop is the
    # unified ReAct regardless. `enable_plan_mode=True` registers the
    # plan-mode tools so the model can self-escalate.
    config = AgentConfig(
        client=client,
        agent_type=AgentType.REACT,
        max_iterations=10,
        system_prompt=(
            "You are a careful coding assistant.\n\n"
            "You have access to file-system tools. Some are read-only "
            "(list_project_files, read_file) and one is side-effectful "
            "(rewrite_file).\n\n"
            "When the user asks for a non-trivial change to multiple "
            "files, you MUST call `enter_plan_mode` first to investigate "
            "before any writes. While in plan mode, only read-only tools "
            "will execute — writes are auto-refused. Once you have a "
            "concrete plan, call `exit_plan_mode` with the plan as "
            "markdown, then proceed with the writes."
        ),
        tools=[
            list_project_files._function_tool,
            read_file._function_tool,
            rewrite_file._function_tool,
        ],
        enable_plan_mode=True,
    )
    return Agent(config=config)


async def run_with_collector(agent: Agent, query: str) -> CollectedEvents:
    collector = CollectedEvents()
    context = RunContext(deps=None, messages=[])
    async for event in agent.stream(query, context):
        collector(event)
    return collector


# ----- assertions -----


def expect(name: str, condition: bool, detail: str) -> List[str]:
    if condition:
        print(f"  PASS: {name}")
        return []
    print(f"  FAIL: {name} — {detail}")
    return [f"{name}: {detail}"]


async def test_plan_mode_happy_path(agent: Agent, provider_name: str) -> List[str]:
    """Multi-file refactor: model plans, then `exit_plan_mode` raises
    PlanApprovalRequired — the loop halts here. ``PLAN_MODE_EXITED``
    only fires after the Django resume path; an in-process run stops
    at ``PLAN_APPROVAL_NEEDED``."""
    print("\n" + "=" * 60)
    print(f"TEST [{provider_name}]: plan-mode happy path (loop pauses for approval)")
    print("=" * 60)
    reset_logs()

    query = (
        "I want to add a one-line copyright header to every Python file in "
        "src/. Plan first, then make the edits. Use the plan-mode tools."
    )
    collector = await run_with_collector(agent, query)

    issues: List[str] = []
    issues += expect(
        "model entered plan mode at least once",
        len(collector.plan_entered) >= 1,
        f"plan_mode_entered events: {len(collector.plan_entered)}",
    )
    issues += expect(
        "loop paused on plan_approval_needed (exit_plan_mode raised)",
        len(collector.plan_approval_needed) >= 1,
        f"plan_approval_needed events: {len(collector.plan_approval_needed)}",
    )
    if collector.plan_approval_needed:
        plan_text = collector.plan_approval_needed[-1].get("plan", "")
        issues += expect(
            "approval payload carried a non-empty plan",
            bool(plan_text and len(plan_text) > 20),
            f"plan length: {len(plan_text)}",
        )
    issues += expect(
        "no final_answer emitted — loop is paused, not completed",
        collector.final_answer is None,
        f"final_answer was: {collector.final_answer!r}",
    )
    return issues


async def test_plan_mode_safety_invariant(agent: Agent, provider_name: str) -> List[str]:
    """Zero writes execute during the plan window — between
    ``PLAN_MODE_ENTERED`` and ``PLAN_APPROVAL_NEEDED``. The loop pauses
    on the approval event without the model ever getting an approve
    signal, so the post-window phase doesn't exist in this in-process
    run (the Django resume path is what reopens it). Any rewrite_file
    call during the window must have been refused by the gate.
    """
    print("\n" + "=" * 60)
    print(f"TEST [{provider_name}]: plan-mode safety invariant (no writes in plan window)")
    print("=" * 60)
    reset_logs()

    query = (
        "I want to add a one-line copyright header to src/main.py. "
        "First call enter_plan_mode (reasoning: 'investigate the file before "
        "writing'). While in plan mode, read src/main.py with read_file. "
        "Then call exit_plan_mode with a markdown plan describing the "
        "change. (The loop will pause for user approval there.)"
    )

    # Track per-event ordering to detect writes inside the plan window.
    collector = CollectedEvents()
    write_indices: List[int] = []
    enter_idx: List[int] = []
    approval_idx: List[int] = []

    context = RunContext(deps=None, messages=[])
    idx = 0
    async for event in agent.stream(query, context):
        collector(event)
        et = getattr(event, "event_type", None)
        if et is None:
            idx += 1
            continue
        et_value = et.value if hasattr(et, "value") else str(et)
        if et_value == "plan_mode_entered":
            enter_idx.append(idx)
        elif et_value == "plan_approval_needed":
            approval_idx.append(idx)
        elif et_value == "observation":
            # An observation arrives AFTER the tool runs. If rewrite_file
            # successfully wrote, _WRITE_LOG just grew; record the index.
            if len(_WRITE_LOG) > len(write_indices):
                write_indices.append(idx)
        idx += 1
    # Keep `exit_idx` as an alias so the assertion code below stays valid.
    exit_idx = approval_idx

    issues: List[str] = []
    issues += expect(
        "enter_plan_mode fired",
        bool(enter_idx),
        f"enter events: {enter_idx}",
    )
    issues += expect(
        "plan_approval_needed fired (exit_plan_mode raised)",
        bool(approval_idx),
        f"approval events: {approval_idx}",
    )

    # Safety invariant: no write_indices fall inside any [enter, exit]
    # window. (If model entered and exited multiple times, check all
    # windows.)
    in_window = False
    inside_count = 0
    if enter_idx and exit_idx:
        # Build sorted enter/exit pairs (best-effort: zip in order).
        pairs = list(zip(sorted(enter_idx), sorted(exit_idx)))
        for w in write_indices:
            for e, x in pairs:
                if e < w < x:
                    inside_count += 1
                    break
    issues += expect(
        "no rewrite_file execution fell inside any plan-mode window",
        inside_count == 0,
        f"{inside_count} writes inside plan-mode window; writes at {write_indices}, "
        f"plan windows {list(zip(enter_idx, exit_idx))}",
    )

    print(
        f"  INFO: _WRITE_LOG={len(_WRITE_LOG)} _READ_LOG={len(_READ_LOG)} "
        f"blocked_events={len(collector.blocked)} enters={len(enter_idx)} exits={len(exit_idx)}"
    )
    return issues


# ----- runner -----


async def run_provider(provider_name: str) -> dict:
    print(f"\n{'#' * 60}\n# Provider: {provider_name}\n{'#' * 60}")
    if not check_environment(provider_name):
        return {"provider": provider_name, "skipped": True}

    agent = create_agent(provider_name)

    results: dict = {"provider": provider_name, "failures": []}
    for fn in (test_plan_mode_happy_path, test_plan_mode_safety_invariant):
        try:
            failures = await fn(agent, provider_name)
            results["failures"].extend(f"{fn.__name__}: {x}" for x in failures)
        except Exception as exc:  # noqa: BLE001
            results["failures"].append(f"{fn.__name__}: EXCEPTION {exc!r}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", "-p", default="openai")
    args = parser.parse_args()

    providers = list(PROVIDERS.keys()) if args.provider == "all" else [args.provider]
    if args.provider != "all" and args.provider not in PROVIDERS:
        print(f"Unknown provider: {args.provider}. Choices: {sorted(PROVIDERS)}")
        sys.exit(2)

    overall_failures: List[str] = []
    for p in providers:
        result = asyncio.run(run_provider(p))
        if result.get("skipped"):
            continue
        for f in result["failures"]:
            overall_failures.append(f"[{p}] {f}")

    print("\n" + "=" * 60)
    if overall_failures:
        print(f"FAILURES ({len(overall_failures)}):")
        for f in overall_failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
