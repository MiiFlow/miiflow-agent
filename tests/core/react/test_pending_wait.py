"""A successful bounded wait is not a repeated-action failure."""
from miiflow_agent.core.react.models import ReActStep, ToolInvocation
from miiflow_agent.core.react.safety import SafetyManager


def wait_step(i, *, pending=True, error=None):
    inv = ToolInvocation(name="get_assistant_run", inputs={"thread_id": "run"},
                         observation="{'status': 'running', 'error': None}", error=error)
    inv.pending_wait = pending
    return ReActStep(step_number=i, thought="", action=inv.name,
                     action_input=inv.inputs, observation=inv.observation,
                     tool_invocations=[inv])


def test_pending_wait_can_outlive_three_polls_and_pattern_guard():
    manager = SafetyManager(max_steps=40)
    for count in (3, 6, 21):
        steps = [wait_step(i) for i in range(count)]
        assert manager.should_stop(steps, count) is None


def test_finished_reads_and_errors_still_stop():
    for pending, error in ((False, None), (True, "network failed")):
        steps = [wait_step(i, pending=pending, error=error) for i in range(3)]
        assert SafetyManager().should_stop(steps, 3) is not None


def test_waits_still_obey_total_step_budget():
    assert SafetyManager(max_steps=3).should_stop([wait_step(i) for i in range(3)], 3)


async def test_executor_and_orchestrator_preserve_wait_contract():
    """The production native tool path must carry the marker into safety checks."""
    from types import SimpleNamespace
    from miiflow_agent import RunContext, Message
    from miiflow_agent.core.react.orchestrator import ReActOrchestrator, ExecutionState
    from miiflow_agent.core.react.events import EventBus
    from miiflow_agent.core.react.tool_executor import AgentToolExecutor
    from miiflow_agent.core.tools.schemas import ToolResult

    class Executor:
        agent = SimpleNamespace(temperature=0, max_tokens=1024)
        output = {"status": "running", "wait_pending": True}
        async def stream_with_tools(self, messages, prebuilt_tools=None):
            yield SimpleNamespace(delta="", thinking_delta=None, usage=None, cost=0,
                finish_reason="tool_calls", tool_calls=[{"id": "call", "type": "function",
                "function": {"name": "get_assistant_run", "arguments": {"thread_id": "run"}}}])
        def get_tools_for_llm(self):
            return [{"name": "get_assistant_run"}]
        def has_tool(self, name):
            return True
        def get_tool_schema(self, name):
            return {"parameters": {"required": []}}
        def _get_tool_schema_obj(self, name):
            return SimpleNamespace(writes=False, metadata={"wait_statuses": ["queued", "running"]})
        async def _execute_tool_gated(self, name, inputs, context=None):
            return ToolResult(name=name, input=inputs, output=self.output)
        execute_tool = AgentToolExecutor.execute_tool

    orch = ReActOrchestrator.__new__(ReActOrchestrator)
    orch.tool_executor = Executor()
    orch.event_bus = EventBus()
    orch.context_compressor = None
    orch.safety_manager = SafetyManager(max_steps=25)
    context = RunContext(deps={}, messages=[Message.user("Wait for my report")])
    state = ExecutionState()
    for i in range(1, 8):
        state.current_step = i
        state.steps.append(await orch._execute_reasoning_step_native(context, state))
        assert state.steps[-1].all_invocations[0].pending_wait
        assert orch.safety_manager.should_stop(state.steps, i) is None
    orch.tool_executor.output = {"status": "completed", "wait_pending": False}
    state.current_step += 1
    state.steps.append(await orch._execute_reasoning_step_native(context, state))
    assert not state.steps[-1].all_invocations[0].pending_wait


def test_mixed_batch_does_not_exempt_repeated_writes():
    steps = [wait_step(i) for i in range(3)]
    for step in steps:
        step.tool_invocations.append(ToolInvocation(name="write", inputs={}, observation="ok"))
    assert SafetyManager().should_stop(steps, 3) is not None


def test_waits_still_obey_cost_and_time_limits():
    steps = [wait_step(1)]
    steps[0].cost = 2
    assert SafetyManager(max_budget=1).should_stop(steps, 1) is not None
    manager = SafetyManager(max_time_seconds=1)
    from miiflow_agent.core.react.safety import MaxTimeCondition
    for condition in manager.conditions:
        if isinstance(condition, MaxTimeCondition):
            condition.start_time = 0
    assert manager.should_stop(steps, 1) is not None
