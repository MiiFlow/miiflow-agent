"""Tests for provider-executed (native) MCP blocks.

Production thread thread_RcXiwnKH97bGTGKDqcBnTm4P: an agent with the Klaviyo MCP
server attached asked "how many profiles do I have in klaviyo?". Anthropic ran
`get_profiles` server-side and returned an `mcp_tool_use` block (id
`mcptoolu_...`), which the client turned into an ordinary pending tool call. The
ReAct orchestrator dispatched it locally and failed three times with
"Tool 'get_profiles' not found", then gave up.

These tests pin the three properties that failure violated:
  1. an mcp_tool_use surfaces tagged `mcp_function`, so the orchestrator can
     tell it apart from work it owes,
  2. mcp_tool_result payloads surface as results, never as answer text,
  3. the pair round-trips to the API as mcp_* blocks, or not at all.
"""

from types import SimpleNamespace

import pytest

from miiflow_agent import RunContext
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.events import EventBus
from miiflow_agent.core.react.orchestrator import ExecutionState, ReActOrchestrator
from miiflow_agent.core.react.safety import SafetyManager
from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer
from miiflow_agent.providers.anthropic_client import AnthropicClient


def _mcp_tool_use_start(index: int = 0):
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(
            type="mcp_tool_use",
            id="mcptoolu_017CQkNV6chUEvYzibnddrGr",
            name="get_profiles",
            server_name="Klaviyo",
        ),
        usage=None,
    )


def _mcp_tool_result_start(index: int = 1, content=None, is_error: bool = False):
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(
            type="mcp_tool_result",
            tool_use_id="mcptoolu_017CQkNV6chUEvYzibnddrGr",
            is_error=is_error,
            content=content,
        ),
        usage=None,
    )


def _json_delta(partial_json: str, index: int = 0):
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(partial_json=partial_json),
        usage=None,
    )


def _text_delta(text: str, index: int = 0):
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(text=text),
        usage=None,
    )


def _stop():
    return SimpleNamespace(type="content_block_stop", usage=None)


@pytest.fixture
def normalizer():
    return AnthropicStreamNormalizer()


class TestNormalizer:
    def test_mcp_tool_use_is_tagged_mcp_function(self, normalizer):
        """The tag is what keeps the call out of the local dispatcher."""
        normalizer.normalize_chunk(_mcp_tool_use_start())
        normalizer.normalize_chunk(_json_delta('{"page_size": 1}'))
        chunk = normalizer.normalize_chunk(_stop())

        assert chunk.tool_calls is not None
        call = chunk.tool_calls[0]
        assert call["type"] == "mcp_function"
        assert call["id"] == "mcptoolu_017CQkNV6chUEvYzibnddrGr"
        assert call["function"]["name"] == "get_profiles"
        assert call["function"]["arguments"] == {"page_size": 1}
        assert call["server_name"] == "Klaviyo"

    def test_result_arriving_whole_on_start_is_captured(self, normalizer):
        normalizer.normalize_chunk(
            _mcp_tool_result_start(content=[{"type": "text", "text": '{"count": 4821}'}])
        )
        chunk = normalizer.normalize_chunk(_stop())

        assert chunk.mcp_tool_results is not None
        result = chunk.mcp_tool_results[0]
        assert result["tool_use_id"] == "mcptoolu_017CQkNV6chUEvYzibnddrGr"
        assert result["content"] == '{"count": 4821}'
        assert result["is_error"] is False

    def test_result_streamed_as_deltas_is_captured(self, normalizer):
        normalizer.normalize_chunk(_mcp_tool_result_start(content=None))
        normalizer.normalize_chunk(_text_delta('{"count": ', index=1))
        normalizer.normalize_chunk(_text_delta('4821}', index=1))
        chunk = normalizer.normalize_chunk(_stop())

        assert chunk.mcp_tool_results[0]["content"] == '{"count": 4821}'

    def test_result_text_never_reaches_the_answer_buffer(self, normalizer):
        """Raw tool payload rendered as assistant prose is the leak this prevents."""
        normalizer.normalize_chunk(_mcp_tool_result_start(content=None))
        chunk = normalizer.normalize_chunk(_text_delta('{"count": 4821}', index=1))

        assert chunk.delta == ""
        assert chunk.content == ""

    def test_error_result_preserves_the_error_flag(self, normalizer):
        normalizer.normalize_chunk(
            _mcp_tool_result_start(
                content=[{"type": "text", "text": "401 Unauthorized"}], is_error=True
            )
        )
        chunk = normalizer.normalize_chunk(_stop())

        assert chunk.mcp_tool_results[0]["is_error"] is True

    def test_ordinary_text_still_streams(self, normalizer):
        """The mcp_tool_result branch must not swallow normal assistant text."""
        start_text = SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text"),
            usage=None,
        )
        normalizer.normalize_chunk(start_text)
        chunk = normalizer.normalize_chunk(_text_delta("You have "))

        assert chunk.delta == "You have "


class TestMessageRoundTrip:
    """Conversion must not emit an mcp call as a plain tool_use block."""

    @pytest.fixture
    def client(self):
        return AnthropicClient(model="claude-sonnet-5", api_key="test-key", timeout=30.0)

    def _assistant(self, **metadata):
        return Message(
            role=MessageRole.ASSISTANT,
            content="You have 4,821 profiles.",
            tool_calls=[
                {
                    "id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                    "type": "mcp_function",
                    "function": {"name": "get_profiles", "arguments": {"page_size": 1}},
                    "server_name": "Klaviyo",
                }
            ],
            metadata=metadata,
        )

    def test_pair_round_trips_as_mcp_blocks(self, client):
        msg = self._assistant(
            mcp_tool_results=[
                {
                    "tool_use_id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                    "is_error": False,
                    "content": '{"count": 4821}',
                }
            ]
        )
        converted = client.convert_message_to_provider_format(msg)
        blocks = converted["content"]
        types = [b["type"] for b in blocks]

        assert "tool_use" not in types, "mcptoolu_ id in a tool_use block is rejected"
        assert types == ["text", "mcp_tool_use", "mcp_tool_result"]
        assert blocks[1]["server_name"] == "Klaviyo"
        assert blocks[2]["tool_use_id"] == blocks[1]["id"]

    def test_call_without_result_is_dropped_not_emitted_unpaired(self, client):
        """An unanswered mcp_tool_use would 400 every later turn in the thread."""
        converted = client.convert_message_to_provider_format(self._assistant())

        assert [b["type"] for b in converted["content"]] == ["text"]

    def test_call_without_server_name_is_dropped(self, client):
        msg = self._assistant(
            mcp_tool_results=[
                {
                    "tool_use_id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                    "is_error": False,
                    "content": "{}",
                }
            ]
        )
        msg.tool_calls[0].pop("server_name")
        converted = client.convert_message_to_provider_format(msg)

        assert [b["type"] for b in converted["content"]] == ["text"]

    def test_ordinary_tool_calls_are_unaffected(self, client):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "toolu_123",
                    "type": "function",
                    "function": {"name": "render_chart", "arguments": {"kind": "bar"}},
                }
            ],
        )
        converted = client.convert_message_to_provider_format(msg)

        assert [b["type"] for b in converted["content"]] == ["tool_use"]
        assert converted["content"][0]["id"] == "toolu_123"

    def test_client_tool_use_blocks_follow_every_mcp_pair(self, client):
        """thread_9Fm7LXCFbePThEBd1qnvv7Dg (2026-09-01): a replayed turn held
        local calls, then a GitHub call the provider ran, then more local
        calls. Emitted in that order, the inline `mcp_tool_result` sat between
        the earlier `tool_use` blocks and the next message's `tool_result`s,
        and the API 400'd on every `tool_use` before it. Client `tool_use`
        blocks must therefore be the tail of the message, after every
        provider-executed pair, whatever order the calls were made in."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Let me check.",
            tool_calls=[
                {
                    "id": "toolu_before",
                    "type": "function",
                    "function": {"name": "grep_files", "arguments": {"pattern": "a"}},
                },
                {
                    "id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                    "type": "mcp_function",
                    "function": {"name": "search_repositories", "arguments": {"q": "x"}},
                    "server_name": "GitHub",
                },
                {
                    "id": "toolu_after",
                    "type": "function",
                    "function": {"name": "grep_files", "arguments": {"pattern": "b"}},
                },
            ],
            metadata={
                "mcp_tool_results": [
                    {
                        "tool_use_id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                        "is_error": True,
                        "content": "Validation Failed",
                    }
                ]
            },
        )
        converted = client.convert_message_to_provider_format(msg)
        blocks = converted["content"]
        types = [b["type"] for b in blocks]

        assert types == ["text", "mcp_tool_use", "mcp_tool_result", "tool_use", "tool_use"]
        last_mcp_result = max(i for i, t in enumerate(types) if t == "mcp_tool_result")
        for i, t in enumerate(types):
            if t == "tool_use":
                assert i > last_mcp_result, f"tool_use at {i} precedes an mcp_tool_result"
        # Relative order among the client calls is preserved.
        assert [b["id"] for b in blocks if b["type"] == "tool_use"] == [
            "toolu_before",
            "toolu_after",
        ]


class TestOrchestratorDispatch:
    """The production failure itself: a provider-executed call must never be
    handed to the local tool executor, and must not turn an answer turn into an
    action turn."""

    def _fake_executor(self, dispatched):
        class FakeExecutor:
            agent = SimpleNamespace(temperature=0, max_tokens=1024)

            def _build_native_tool_schemas(self):
                return []

            async def stream_with_tools(self, messages, prebuilt_tools=None):
                # Anthropic streams the mcp_tool_use, then the result it
                # already produced server-side, then the answer text.
                yield SimpleNamespace(
                    delta="",
                    thinking_delta=None,
                    tool_calls=[
                        {
                            "id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                            "type": "mcp_function",
                            "function": {
                                "name": "get_profiles",
                                "arguments": {"page_size": 1},
                            },
                            "server_name": "Klaviyo",
                        }
                    ],
                    mcp_tool_results=None,
                    usage=None,
                    cost=0,
                    finish_reason=None,
                )
                yield SimpleNamespace(
                    delta="",
                    thinking_delta=None,
                    tool_calls=None,
                    mcp_tool_results=[
                        {
                            "tool_use_id": "mcptoolu_017CQkNV6chUEvYzibnddrGr",
                            "is_error": False,
                            "content": '{"count": 4821}',
                        }
                    ],
                    usage=None,
                    cost=0,
                    finish_reason=None,
                )
                yield SimpleNamespace(
                    delta="You have 4,821 profiles.",
                    thinking_delta=None,
                    tool_calls=None,
                    mcp_tool_results=None,
                    usage=None,
                    cost=0,
                    finish_reason="end_turn",
                )

            def has_tool(self, name):
                dispatched.append(name)
                return False

            def list_tools(self):
                return ["list_ads", "render_chart"]

            def get_tool_schema(self, name):
                return {"parameters": {"required": []}}

            async def execute_tool(self, name, inputs, context=None):
                raise AssertionError(f"local dispatch attempted for {name}")

        return FakeExecutor()

    async def _run_step(self, dispatched):
        orch = ReActOrchestrator.__new__(ReActOrchestrator)
        orch.tool_executor = self._fake_executor(dispatched)
        orch.event_bus = EventBus()
        orch.context_compressor = None
        orch.safety_manager = SafetyManager(max_steps=5)

        context = RunContext(deps={}, messages=[Message.user("how many profiles?")])
        state = ExecutionState()
        state.current_step = 1

        step = await ReActOrchestrator._execute_reasoning_step_native(
            orch, context, state
        )
        return orch, context, step

    @pytest.mark.asyncio
    async def test_provider_executed_call_never_reaches_local_dispatcher(self):
        dispatched: list = []
        _, _, step = await self._run_step(dispatched)

        assert dispatched == [], f"orchestrator tried to dispatch {dispatched}"
        assert step.error is None
        assert step.action is None

    @pytest.mark.asyncio
    async def test_turn_is_treated_as_an_answer_turn(self):
        """The text after the MCP result is the answer, not preamble narration."""
        _, _, step = await self._run_step([])

        assert step.answer == "You have 4,821 profiles."

    @pytest.mark.asyncio
    async def test_result_is_recorded_as_an_observation_event(self):
        orch, _, _ = await self._run_step([])

        observations = [
            e
            for e in orch.event_bus.event_buffer
            if e.event_type == ReActEventType.OBSERVATION
        ]
        assert len(observations) == 1
        assert observations[0].data["action"] == "get_profiles"
        assert observations[0].data["success"] is True
        assert observations[0].data["tool_call_id"] == "mcptoolu_017CQkNV6chUEvYzibnddrGr"

    @pytest.mark.asyncio
    async def test_call_is_announced_before_its_observation(self):
        """The action+observation pair is the contract every consumer builds on.

        Consumers create their tool row on ACTION_PLANNED and only *update* it
        on OBSERVATION (execution_timeline in streaming_service, both frontend
        chunk reducers). Emitting the observation alone made a native-MCP turn
        invisible: no reasoning panel live, empty execution_timeline on reload,
        and the turn mislabeled `single_hop` because `has_tool_events` stayed
        False (production thread thread_AE8cAXDlY8gqhCy4GKjlLoXp).
        """
        orch, _, _ = await self._run_step([])

        trail = [
            e
            for e in orch.event_bus.event_buffer
            if e.event_type
            in (ReActEventType.ACTION_PLANNED, ReActEventType.OBSERVATION)
        ]
        assert [e.event_type for e in trail] == [
            ReActEventType.ACTION_PLANNED,
            ReActEventType.OBSERVATION,
        ]

        planned = trail[0]
        assert planned.data["action"] == "get_profiles"
        # Same id on both halves so consumers correlate by id rather than by
        # name — a turn can run several calls of the same tool in one batch.
        assert planned.data["tool_call_id"] == trail[1].data["tool_call_id"]
        assert planned.data["action_input"] == {"page_size": 1}

    @pytest.mark.asyncio
    async def test_results_ride_the_assistant_message_for_replay(self):
        _, context, _ = await self._run_step([])

        assistant = [m for m in context.messages if m.role == MessageRole.ASSISTANT][-1]
        assert assistant.tool_calls[0]["type"] == "mcp_function"
        assert assistant.metadata["mcp_tool_results"][0]["content"] == '{"count": 4821}'
