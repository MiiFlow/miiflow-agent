"""Production regression: every image in a multi-call turn must remain usable."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.message import ImageBlock, MessageRole
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.tools.schemas import ToolResult
from miiflow_agent.visualization.types import LlmBlockInjection, MediaResult


def setup_run(outputs):
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="openai")
    agent.tool_registry = MagicMock()
    agent.tool_registry.list_tools.return_value = ["image_tool"]
    agent._tools = []
    orch = ReActFactory.create_orchestrator(
        agent=agent, max_steps=5, max_budget=None, max_time_seconds=None
    )
    results = [ToolResult(name="image_tool", input={}, output=o) for o in outputs]
    orch.tool_executor.execute_many = AsyncMock(return_value=results)
    orch.tool_executor.execute_tool = AsyncMock(side_effect=results)
    orch.tool_executor.has_tool = MagicMock(return_value=True)
    orch.tool_executor.get_tool_schema = MagicMock(return_value={})
    events = []
    orch.event_bus.subscribe(lambda e: events.append(e))
    return (
        orch,
        RunContext(deps={}, messages=[]),
        ExecutionState(current_step=1),
        events,
    )


async def batch(orch, ctx, state, count):
    calls = {
        i: {"id": f"call_{i}", "function": {"name": "image_tool", "arguments": {}}}
        for i in range(count)
    }
    step = ReActStep(step_number=state.current_step, thought="")
    await orch._handle_parallel_tool_batch(step, ctx, state, calls, "")
    return step


@pytest.mark.asyncio
async def test_all_seven_generated_images_are_registered_emitted_and_reusable():
    images = [
        MediaResult(
            id=f"frame-{i}",
            url=f"https://cdn.test/{i}.png",
            metadata={"media": {"width_px": 1536, "height_px": 864}},
        )
        for i in range(7)
    ]
    orch, ctx, state, events = setup_run(images)
    step = await batch(orch, ctx, state, 7)

    assert state.media_store == {image.id: image.url for image in images}
    assert ctx.deps["media_store"] is state.media_store
    assert len([e for e in events if e.event_type == ReActEventType.MEDIA]) == 7
    assert not step.has_failed_invocations
    messages = [m for m in ctx.messages if m.role == MessageRole.TOOL]
    assert [m.tool_call_id for m in messages] == [f"call_{i}" for i in range(7)]
    for image, message in zip(images, messages):
        assert f"media_ref:{image.id}" in message.content
        assert '"height_px": 864' in message.content
        # A following editing call resolves every advertised handle.
        assert (
            orch._resolve_media_refs({"image": f"media_ref:{image.id}"}, state)["image"]
            == image.url
        )


@pytest.mark.asyncio
async def test_mixed_batch_keeps_visual_blocks_on_their_own_tool_message():
    injection = LlmBlockInjection(
        blocks=[{"type": "image_url", "image_url": "https://cdn.test/logo.png"}],
        summary="Official logo",
    )
    collection = {
        "__media_collection__": True,
        "items": [
            MediaResult(id="a", url="https://cdn.test/a.png"),
            MediaResult(id="b", url="https://cdn.test/b.png"),
        ],
        "metadata": [{"label": "first"}, {"label": "second"}],
    }
    orch, ctx, state, events = setup_run([injection, collection, "ordinary result"])
    await batch(orch, ctx, state, 3)
    messages = [m for m in ctx.messages if m.role == MessageRole.TOOL]
    assert isinstance(messages[0].content[1], ImageBlock)
    assert messages[0].content[1].image_url == "https://cdn.test/logo.png"
    assert isinstance(messages[1].content, str)
    assert '"label": "first"' in messages[1].content
    assert isinstance(messages[2].content, str)
    assert not state.pending_llm_blocks
    assert set(state.media_store) == {"a", "b"}


@pytest.mark.asyncio
async def test_single_and_batch_media_observations_match():
    image = MediaResult(
        id="same",
        url="https://cdn.test/image.png",
        metadata={
            "generation": {
                "status": "needs_review",
                "logo_composition": {"applied": False},
            }
        },
    )
    single, ctx, state, _ = setup_run([image])
    step = ReActStep(step_number=1, thought="", action="image_tool", action_input={})
    await single._handle_tool_action(step, ctx, state, tool_call_id="single")
    multi, multi_ctx, multi_state, _ = setup_run([image, "other"])
    multi_step = await batch(multi, multi_ctx, multi_state, 2)
    assert step.observation == multi_step.tool_invocations[0].observation
    assert "needs_review" in step.observation
    assert "successfully" not in step.observation
    assert state.media_store == multi_state.media_store
