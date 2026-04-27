#!/usr/bin/env python3
"""
End-to-end manual test for the ReAct orchestrator's SSE event contract.

Validates the post-rewrite invariants:

  * Per turn the model emits exactly one of: tool calls, or text answer.
    Never both. Never preamble narration.
  * In an answer turn, text deltas stream as FINAL_ANSWER_CHUNK events
    (not THINKING_CHUNK).
  * In an action turn, ACTION_PLANNED -> ACTION_EXECUTING -> OBSERVATION
    fire in order; no FINAL_ANSWER_CHUNK leaks before the answer turn.
  * The classifier was deleted: number of LLM round-trips per query
    equals (action turns + 1 answer turn) with no extra "is this
    thinking or answer" call.
  * THINKING_CHUNK is reserved for native extended-thinking deltas
    (Anthropic thinking blocks, OpenAI reasoning) — never for normal
    text deltas.

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    poetry run python tests/manual/test_react_manual.py

    # Google Gemini
    export GOOGLE_API_KEY="..."
    poetry run python tests/manual/test_react_manual.py --provider gemini

    # Anthropic Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    poetry run python tests/manual/test_react_manual.py --provider anthropic

    # All providers (skips any without API key)
    poetry run python tests/manual/test_react_manual.py --provider all
"""

import argparse
import asyncio
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from miiflow_agent import Agent, AgentType, LLMClient, RunContext
from miiflow_agent.core.callbacks import (
    CallbackEventType,
    register as register_callback,
    unregister as unregister_callback,
)
from miiflow_agent.core.react import ReActEventType
from miiflow_agent.core.tools import tool


# -----------------------------------------------------------------------------
# Provider configuration
# -----------------------------------------------------------------------------

PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "env_var": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "env_var": "ANTHROPIC_API_KEY",
    },
}


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


@tool("calculate", "Evaluate mathematical expressions")
def calculate(expression: str) -> str:
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "pi": math.pi, "e": math.e,
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


@tool("get_weather", "Get current weather for a location")
def get_weather(location: str) -> str:
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "paris": "Partly cloudy, 65°F (18°C)",
        "san francisco": "Foggy, 60°F (16°C)",
    }
    key = location.lower()
    if key in weather_data:
        return f"Weather in {location}: {weather_data[key]}"
    return f"Weather data not available for {location}"


# -----------------------------------------------------------------------------
# Event capture
# -----------------------------------------------------------------------------


@dataclass
class StepView:
    """All events captured for a single orchestrator step, in arrival order."""

    step_number: int
    events: List[tuple] = field(default_factory=list)  # list of (type_str, data)

    def types(self) -> List[str]:
        return [t for t, _ in self.events]

    def first_index(self, type_str: str) -> int:
        for i, (t, _) in enumerate(self.events):
            if t == type_str:
                return i
        return -1


class EventCollector:
    """Subscribes to the orchestrator's EventBus and groups events by step."""

    def __init__(self):
        self.all_events: List[tuple] = []  # (step_number, type_str, data)
        self.steps: dict[int, StepView] = {}
        self.final_answer_complete: Optional[str] = None
        self.streamed_answer_chunks: List[str] = []
        self.thinking_chunks_text: List[str] = []  # accumulated thinking_chunk deltas
        self.action_planned: List[str] = []  # tool names per ACTION_PLANNED
        self.observations: List[str] = []

    def __call__(self, event):
        type_obj = getattr(event, "event_type", None)
        if type_obj is None:
            return
        type_str = type_obj.value if hasattr(type_obj, "value") else str(type_obj)
        step_number = getattr(event, "step_number", 0) or 0
        data = dict(getattr(event, "data", {}) or {})

        self.all_events.append((step_number, type_str, data))
        view = self.steps.setdefault(step_number, StepView(step_number=step_number))
        view.events.append((type_str, data))

        if type_str == "final_answer_chunk":
            self.streamed_answer_chunks.append(data.get("delta", ""))
        elif type_str == "final_answer":
            self.final_answer_complete = data.get("answer", "")
        elif type_str == "thinking_chunk":
            self.thinking_chunks_text.append(data.get("delta", ""))
        elif type_str == "action_planned":
            self.action_planned.append(data.get("action") or "")
        elif type_str == "observation":
            obs = data.get("observation")
            self.observations.append("" if obs is None else str(obs))

    def streamed_answer(self) -> str:
        return "".join(self.streamed_answer_chunks)

    def step_count(self) -> int:
        return len(self.steps)

    def step_kinds(self) -> List[str]:
        """Classify each step as 'action' (has ACTION_PLANNED) or 'answer' (no
        action). Returns kinds in step-number order."""
        kinds = []
        for n in sorted(self.steps.keys()):
            types = self.steps[n].types()
            kinds.append("action" if "action_planned" in types else "answer")
        return kinds


class LLMCallCounter:
    """Counts every successful LLM round-trip via the POST_CALL callback.

    Used to verify the deleted classifier really is deleted: total LLM
    calls must equal the number of orchestrator steps (one call per
    step). Any extra calls would indicate a classifier-style sidecar.
    """

    def __init__(self):
        self.calls = 0
        self._cb = self._on_post_call

    def _on_post_call(self, event):
        self.calls += 1

    def __enter__(self):
        register_callback(CallbackEventType.POST_CALL, self._cb)
        return self

    def __exit__(self, *exc):
        unregister_callback(CallbackEventType.POST_CALL, self._cb)


# -----------------------------------------------------------------------------
# Test infrastructure
# -----------------------------------------------------------------------------


def check_environment(provider_name: str) -> bool:
    env_var = PROVIDERS[provider_name]["env_var"]
    if not os.environ.get(env_var):
        print(f"SKIP: {env_var} not set for {provider_name}")
        return False
    print(f"Environment check passed: {env_var} is set")
    return True


def create_agent(provider_name: str) -> Agent:
    config = PROVIDERS[provider_name]
    client = LLMClient.create(
        config["provider"],
        model=config["model"],
        api_key=os.environ.get(config["env_var"]),
    )
    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=8,
        system_prompt=(
            "You are a helpful assistant. Use tools when needed. "
            "Per turn, do exactly one: either call tools, or give the final answer. "
            "Never narrate that you are about to call a tool."
        ),
    )
    agent.add_tool(calculate)
    agent.add_tool(get_weather)
    return agent


async def run_with_collector(agent: Agent, query: str) -> tuple[EventCollector, int]:
    """Run a query, returning (collected events, observed LLM-call count)."""
    collector = EventCollector()
    context = RunContext(deps=None, messages=[])

    with LLMCallCounter() as counter:
        async for event in agent.stream(query, context):
            collector(event)

    return collector, counter.calls


# -----------------------------------------------------------------------------
# Assertions
# -----------------------------------------------------------------------------


def assert_per_turn_invariant(collector: EventCollector) -> List[str]:
    """Each step is either action (>=1 ACTION_PLANNED) or answer (0
    ACTION_PLANNED). FINAL_ANSWER_CHUNK is allowed only in the final
    answer step."""
    issues: List[str] = []
    kinds = collector.step_kinds()
    for n in sorted(collector.steps.keys()):
        view = collector.steps[n]
        types = Counter(view.types())
        action_count = types.get("action_planned", 0)
        chunk_count = types.get("final_answer_chunk", 0)
        kind = "action" if action_count >= 1 else "answer"

        if kind == "action" and chunk_count > 0:
            issues.append(
                f"step {n}: action turn ({action_count} action_planned) "
                f"emitted {chunk_count} final_answer_chunk events — "
                f"text leaked from preamble"
            )

    final_steps = [n for n, k in zip(sorted(collector.steps.keys()), kinds) if k == "answer"]
    if not final_steps:
        issues.append("no answer step observed (no FINAL_ANSWER would have been emitted)")
    elif final_steps[-1] != max(collector.steps.keys()):
        issues.append(
            f"answer step {final_steps[-1]} is not the last step "
            f"(last step is {max(collector.steps.keys())})"
        )
    return issues


def assert_action_event_ordering(collector: EventCollector) -> List[str]:
    """Within an action step, ACTION_PLANNED precedes ACTION_EXECUTING
    precedes OBSERVATION."""
    issues: List[str] = []
    for n, view in collector.steps.items():
        types = view.types()
        if "action_planned" not in types:
            continue
        ip = view.first_index("action_planned")
        ie = view.first_index("action_executing")
        io = view.first_index("observation")
        if ie != -1 and ie < ip:
            issues.append(f"step {n}: action_executing fired before action_planned")
        if io != -1 and ip != -1 and io < ip:
            issues.append(f"step {n}: observation fired before action_planned")
        if io != -1 and ie != -1 and io < ie:
            issues.append(f"step {n}: observation fired before action_executing")
    return issues


def assert_no_text_thinking_chunks(collector: EventCollector) -> List[str]:
    """THINKING_CHUNK should only fire from native extended-thinking
    deltas. With a model that lacks extended thinking enabled, total
    text inside thinking_chunk events should be zero (or near-zero).

    We can't perfectly distinguish native-thinking from text-as-thinking
    here, so this is a soft check: thinking text should not contain the
    final answer."""
    issues: List[str] = []
    if not collector.streamed_answer_chunks:
        return issues
    full_answer = collector.streamed_answer().strip()
    if not full_answer:
        return issues
    thinking_blob = "".join(collector.thinking_chunks_text)
    # If thinking_chunk text contains the final answer verbatim, that
    # means we're double-emitting (old behavior).
    if full_answer and full_answer in thinking_blob:
        issues.append(
            "the streamed final answer also appeared inside thinking_chunk "
            "events — text deltas should route to final_answer_chunk only"
        )
    return issues


def assert_final_answer_event(collector: EventCollector, expected_substrs: List[str]) -> List[str]:
    issues: List[str] = []
    if not collector.final_answer_complete:
        issues.append("FINAL_ANSWER event was never emitted")
        return issues
    answer_lower = collector.final_answer_complete.lower()
    for substr in expected_substrs:
        if substr.lower() not in answer_lower:
            issues.append(
                f"FINAL_ANSWER missing expected substring '{substr}'. "
                f"Got: {collector.final_answer_complete[:200]}"
            )
    return issues


def assert_streamed_answer_matches_complete(collector: EventCollector) -> List[str]:
    """The final FINAL_ANSWER event's payload should match (modulo
    whitespace) what was streamed via FINAL_ANSWER_CHUNKs in the same
    answer step."""
    issues: List[str] = []
    if not collector.final_answer_complete:
        return issues
    if not collector.streamed_answer_chunks:
        # No streamed chunks but a final answer — acceptable for empty
        # responses, though unusual.
        return issues
    streamed = collector.streamed_answer().strip()
    final = collector.final_answer_complete.strip()
    if streamed != final:
        # The chunks may include the answer plus some trailing
        # whitespace; tolerate prefix/suffix relationship.
        if streamed not in final and final not in streamed:
            issues.append(
                f"streamed answer chunks do not match FINAL_ANSWER. "
                f"streamed={streamed[:120]!r} final={final[:120]!r}"
            )
    return issues


def assert_llm_call_count(
    observed_calls: int, action_steps: int, answer_steps: int
) -> List[str]:
    """The number of LLM round-trips must equal action_steps +
    answer_steps. Any extra is a classifier-style sidecar call."""
    expected = action_steps + answer_steps
    if observed_calls != expected:
        return [
            f"expected {expected} LLM round-trips ({action_steps} action + "
            f"{answer_steps} answer), observed {observed_calls}"
        ]
    return []


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------


async def test_pure_answer_turn(agent: Agent) -> List[str]:
    """A query that needs no tools. Loop runs for exactly one answer step."""
    print("\n" + "=" * 60)
    print("TEST: pure answer turn (no tools)")
    print("Query: What does ROAS stand for in marketing? One sentence.")
    print("=" * 60)

    collector, llm_calls = await run_with_collector(
        agent, "What does ROAS stand for in marketing? Answer in one short sentence."
    )

    print(f"  steps observed:    {collector.step_count()}")
    print(f"  step kinds:        {collector.step_kinds()}")
    print(f"  LLM round-trips:   {llm_calls}")
    print(f"  action events:     {collector.action_planned}")
    print(f"  answer chunks:     {len(collector.streamed_answer_chunks)}")
    print(f"  thinking chunks:   {len(collector.thinking_chunks_text)}")
    print(f"  final answer:      {(collector.final_answer_complete or '')[:120]}")

    issues: List[str] = []
    if collector.action_planned:
        issues.append(f"unexpected tool calls: {collector.action_planned}")
    if collector.step_kinds() != ["answer"]:
        issues.append(f"expected exactly one answer step, got {collector.step_kinds()}")
    issues.extend(assert_per_turn_invariant(collector))
    issues.extend(assert_no_text_thinking_chunks(collector))
    issues.extend(
        assert_final_answer_event(collector, expected_substrs=["return on ad spend"])
    )
    issues.extend(assert_streamed_answer_matches_complete(collector))
    issues.extend(assert_llm_call_count(llm_calls, action_steps=0, answer_steps=1))
    if not collector.streamed_answer_chunks:
        issues.append("no FINAL_ANSWER_CHUNK events — answer did not stream live")
    return issues


async def test_action_then_answer(agent: Agent) -> List[str]:
    """A single tool call followed by a final answer. Two LLM round-trips."""
    print("\n" + "=" * 60)
    print("TEST: action-then-answer (1 tool call + 1 answer turn)")
    print("Query: What's the weather in Tokyo?")
    print("=" * 60)

    collector, llm_calls = await run_with_collector(agent, "What's the weather in Tokyo?")

    kinds = collector.step_kinds()
    action_steps = sum(1 for k in kinds if k == "action")
    answer_steps = sum(1 for k in kinds if k == "answer")

    print(f"  steps observed:    {collector.step_count()}")
    print(f"  step kinds:        {kinds}")
    print(f"  LLM round-trips:   {llm_calls}")
    print(f"  action events:     {collector.action_planned}")
    print(f"  observations:      {len(collector.observations)}")
    print(f"  answer chunks:     {len(collector.streamed_answer_chunks)}")
    print(f"  final answer:      {(collector.final_answer_complete or '')[:120]}")

    issues: List[str] = []
    if action_steps < 1:
        issues.append(
            f"expected >=1 action step (model should call get_weather), got 0"
        )
    if answer_steps < 1:
        issues.append("expected >=1 answer step, got 0")
    if "get_weather" not in collector.action_planned:
        issues.append(
            f"expected get_weather tool call, got {collector.action_planned}"
        )
    if len(collector.observations) < action_steps:
        issues.append(
            f"missing observations: got {len(collector.observations)} for "
            f"{action_steps} action step(s)"
        )
    issues.extend(assert_per_turn_invariant(collector))
    issues.extend(assert_action_event_ordering(collector))
    issues.extend(assert_no_text_thinking_chunks(collector))
    issues.extend(
        assert_final_answer_event(
            collector, expected_substrs=["tokyo"]
        )
    )
    issues.extend(assert_streamed_answer_matches_complete(collector))
    issues.extend(assert_llm_call_count(llm_calls, action_steps, answer_steps))
    return issues


async def test_multi_tool_loop(agent: Agent) -> List[str]:
    """Two tool calls then a final answer. Three LLM round-trips."""
    print("\n" + "=" * 60)
    print("TEST: multi-tool loop (2 tool calls + 1 answer turn)")
    print("Query: weather in Tokyo + sqrt(144)+8")
    print("=" * 60)

    collector, llm_calls = await run_with_collector(
        agent,
        "Two questions: (1) what's the weather in Tokyo and (2) what is sqrt(144) + 8? "
        "Use tools for both, then answer both in one final response.",
    )

    kinds = collector.step_kinds()
    action_steps = sum(1 for k in kinds if k == "action")
    answer_steps = sum(1 for k in kinds if k == "answer")

    print(f"  steps observed:    {collector.step_count()}")
    print(f"  step kinds:        {kinds}")
    print(f"  LLM round-trips:   {llm_calls}")
    print(f"  action events:     {collector.action_planned}")
    print(f"  observations:      {len(collector.observations)}")
    print(f"  answer chunks:     {len(collector.streamed_answer_chunks)}")
    print(f"  final answer:      {(collector.final_answer_complete or '')[:200]}")

    issues: List[str] = []
    if action_steps < 2:
        issues.append(
            f"expected >=2 action steps (weather + calculate), got {action_steps}"
        )
    if answer_steps < 1:
        issues.append("expected >=1 answer step, got 0")
    tools_used = set(collector.action_planned)
    if "get_weather" not in tools_used:
        issues.append(f"expected get_weather to be called; tools_used={tools_used}")
    if "calculate" not in tools_used:
        issues.append(f"expected calculate to be called; tools_used={tools_used}")
    issues.extend(assert_per_turn_invariant(collector))
    issues.extend(assert_action_event_ordering(collector))
    issues.extend(assert_no_text_thinking_chunks(collector))
    issues.extend(
        assert_final_answer_event(
            collector, expected_substrs=["tokyo", "20"]
        )
    )
    issues.extend(assert_streamed_answer_matches_complete(collector))
    issues.extend(assert_llm_call_count(llm_calls, action_steps, answer_steps))
    return issues


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


async def run_tests_for_provider(provider_name: str) -> tuple[int, int, list]:
    config = PROVIDERS[provider_name]
    print("\n" + "=" * 60)
    print(f"Provider: {provider_name.upper()} ({config['model']})")
    print("=" * 60)

    if not check_environment(provider_name):
        return 0, 0, []

    agent = create_agent(provider_name)

    cases = [
        ("pure answer turn", test_pure_answer_turn),
        ("action then answer", test_action_then_answer),
        ("multi-tool loop", test_multi_tool_loop),
    ]

    passed = 0
    failed = 0
    failures: List[tuple[str, str]] = []

    for name, fn in cases:
        try:
            issues = await fn(agent)
        except Exception as e:
            failed += 1
            failures.append((f"{provider_name}/{name}", f"exception: {e}"))
            print(f"\n  FAILED with exception: {e}")
            continue

        if issues:
            failed += 1
            failures.append((f"{provider_name}/{name}", "; ".join(issues)))
            print("  FAILED:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            passed += 1
            print("  PASSED")

    return passed, failed, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end manual test for the ReAct orchestrator's SSE event contract"
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "gemini", "anthropic", "all"],
        default="openai",
        help="LLM provider to test (default: openai)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("ReAct Orchestrator — SSE Event Contract Test")
    print("=" * 60)

    if args.provider == "all":
        providers_to_test = list(PROVIDERS.keys())
    else:
        providers_to_test = [args.provider]

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures: List[tuple[str, str]] = []

    for provider_name in providers_to_test:
        passed, failed, failures = await run_tests_for_provider(provider_name)
        if passed == 0 and failed == 0:
            total_skipped += 1
        else:
            total_passed += passed
            total_failed += failed
            all_failures.extend(failures)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Providers tested: {len(providers_to_test) - total_skipped}/{len(providers_to_test)}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    if all_failures:
        print("\nFailures:")
        for name, error in all_failures:
            print(f"  - {name}: {error}")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
