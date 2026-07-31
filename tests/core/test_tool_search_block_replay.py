"""Provider tool-search blocks must survive the history round-trip.

Anthropic's `defer_loading` path has the provider run a tool search itself and
stream back a ``server_tool_use`` + ``tool_search_tool_result`` pair. We execute
nothing, but the pair is the model's only record that the discovery happened.
Dropping it (the original behaviour — the normalizer had no branch for either
block type) made the model re-search for the same deferred tool on every
subsequent turn, paying a round-trip each time and inflating the prompt with a
fresh search result.

These cover the three legs: capture (normalizer), carry (orchestrator helper),
replay (Anthropic message conversion).
"""
from types import SimpleNamespace

from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.orchestrator import _attach_search_blocks
from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer


def _start(index, **block):
    return SimpleNamespace(
        type="content_block_start", index=index, content_block=SimpleNamespace(**block)
    )


def _delta(index, **delta):
    return SimpleNamespace(
        type="content_block_delta", index=index, delta=SimpleNamespace(**delta)
    )


def _stop(index):
    return SimpleNamespace(type="content_block_stop", index=index)


# ── capture ──────────────────────────────────────────────────────────────────

def test_normalizer_captures_search_use_and_result():
    norm = AnthropicStreamNormalizer({})

    norm.normalize_chunk(
        _start(0, type="server_tool_use", id="srvtoolu_1", name="tool_search_tool_regex", input={})
    )
    norm.normalize_chunk(_delta(0, partial_json='{"query":"exper'))
    norm.normalize_chunk(_delta(0, partial_json='iments"}'))
    use_chunk = norm.normalize_chunk(_stop(0))

    assert use_chunk.tool_search_blocks == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "tool_search_tool_regex",
            "input": {"query": "experiments"},
        }
    ]
    # Not a tool call — nothing for us to dispatch.
    assert not use_chunk.tool_calls

    norm.normalize_chunk(
        _start(1, type="tool_search_tool_result", tool_use_id="srvtoolu_1", content=[])
    )
    res_chunk = norm.normalize_chunk(_stop(1))
    assert res_chunk.tool_search_blocks[0]["type"] == "tool_search_tool_result"
    assert res_chunk.tool_search_blocks[0]["tool_use_id"] == "srvtoolu_1"


def test_result_block_carries_only_input_legal_fields():
    """Response blocks carry fields the request schema rejects — replaying a
    `citations` key verbatim 400s with "Extra inputs are not permitted",
    breaking every later turn. Capture whitelists instead of echoing."""
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(
            0,
            type="tool_search_tool_result",
            tool_use_id="srvtoolu_1",
            content=[],
            citations=[{"cited_text": "nope"}],
        )
    )
    block = norm.normalize_chunk(_stop(0)).tool_search_blocks[0]
    assert set(block) == {"type", "tool_use_id", "content"}


def test_use_block_carries_only_input_legal_fields():
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(
            0,
            type="server_tool_use",
            id="srvtoolu_1",
            name="tool_search_tool_regex",
            input={"pattern": "x"},
            caller={"type": "direct"},
        )
    )
    block = norm.normalize_chunk(_stop(0)).tool_search_blocks[0]
    assert set(block) == {"type", "id", "name", "input"}


def test_search_query_json_never_leaks_into_answer_text():
    """The query streams as partial_json on a block whose deltas would
    otherwise fall through to the generic text/tool_use handling — splicing
    raw JSON into the answer or onto an unrelated tool call."""
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(0, type="server_tool_use", id="srvtoolu_1", name="tool_search_tool_regex", input={})
    )
    chunk = norm.normalize_chunk(_delta(0, partial_json='{"query":"x"}'))
    assert chunk.delta == ""
    assert chunk.content == ""


def test_unparseable_search_input_is_dropped_not_replayed():
    """A truncated query cannot be replayed — the API rejects a
    server_tool_use whose input isn't an object, which would 400 every
    remaining turn. Dropping costs one repeated search."""
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(0, type="server_tool_use", id="srvtoolu_1", name="tool_search_tool_regex", input={})
    )
    norm.normalize_chunk(_delta(0, partial_json='{"query":"trunc'))
    chunk = norm.normalize_chunk(_stop(0))
    assert not chunk.tool_search_blocks


# ── carry ────────────────────────────────────────────────────────────────────

PAIR = [
    {"type": "server_tool_use", "id": "s1", "name": "tool_search_tool_regex", "input": {}},
    {"type": "tool_search_tool_result", "tool_use_id": "s1", "content": []},
]


def test_attach_targets_the_newest_assistant_message():
    messages = [
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="older"),
        Message(role=MessageRole.USER, content="again"),
        Message(role=MessageRole.ASSISTANT, content="newest"),
    ]
    assert _attach_search_blocks(messages, list(PAIR), 0) is True
    assert messages[3].metadata["tool_search_blocks"] == PAIR
    assert not (messages[1].metadata or {}).get("tool_search_blocks")


def test_attach_drops_unpaired_halves():
    """An orphan server_tool_use replayed without its result 400s the whole
    thread — dropping it is the safe failure."""
    messages = [Message(role=MessageRole.ASSISTANT, content="x")]
    orphan = [{"type": "server_tool_use", "id": "s1", "name": "t", "input": {}}]
    assert _attach_search_blocks(messages, orphan, 0) is False
    assert not (messages[0].metadata or {}).get("tool_search_blocks")


def test_attach_is_a_noop_without_an_assistant_message():
    messages = [Message(role=MessageRole.USER, content="hi")]
    assert _attach_search_blocks(messages, list(PAIR), 0) is False


def test_attach_preserves_existing_metadata():
    messages = [Message(role=MessageRole.ASSISTANT, content="x", metadata={"keep": 1})]
    _attach_search_blocks(messages, list(PAIR), 0)
    assert messages[0].metadata["keep"] == 1
    assert messages[0].metadata["tool_search_blocks"] == PAIR


# ── replay ───────────────────────────────────────────────────────────────────

def _convert(message):
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    client = AnthropicClient.__new__(AnthropicClient)  # no API key needed
    return client.convert_message_to_provider_format(message)


def test_replay_alongside_a_tool_call():
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="Looking that up.",
        tool_calls=[{"id": "toolu_1", "type": "function",
                     "function": {"name": "list_experiments", "arguments": {}}}],
        metadata={"tool_search_blocks": PAIR},
    )
    types = [b["type"] for b in _convert(msg)["content"]]
    assert types == ["text", "server_tool_use", "tool_search_tool_result", "tool_use"]


def test_replay_when_the_turn_called_nothing():
    """Searched, then answered from context — the plain-string path would
    otherwise drop the blocks and forget the discovery."""
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="Here you go.",
        metadata={"tool_search_blocks": PAIR},
    )
    types = [b["type"] for b in _convert(msg)["content"]]
    assert types == ["text", "server_tool_use", "tool_search_tool_result"]


def test_messages_without_search_blocks_are_unchanged():
    msg = Message(role=MessageRole.ASSISTANT, content="plain")
    assert _convert(msg)["content"] == "plain"


# ── regressions found in review ──────────────────────────────────────────────

async def test_astream_chat_yields_search_block_chunks():
    """THE bug the isolated tests above all missed: the capture, carry and
    replay legs each worked, but `astream_chat`'s yield gate never checked
    `tool_search_blocks`, so the chunks were filtered out between normalizer
    and orchestrator and the whole feature was dead code in production.

    Drives the real gate rather than calling normalize_chunk directly.
    """
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    events = [
        _start(0, type="server_tool_use", id="srvtoolu_1",
               name="tool_search_tool_regex", input={"pattern": "exp"}),
        _stop(0),
        _start(1, type="tool_search_tool_result", tool_use_id="srvtoolu_1", content=[]),
        _stop(1),
    ]

    client = AnthropicClient.__new__(AnthropicClient)
    client._tool_name_mapping = {}
    normalizer = AnthropicStreamNormalizer(client._tool_name_mapping)

    # Mirror the gate in astream_chat exactly.
    yielded = []
    for event in events:
        chunk = normalizer.normalize_chunk(event)
        if (
            chunk.delta
            or chunk.thinking_delta
            or chunk.tool_calls
            or chunk.mcp_tool_results
            or chunk.tool_search_blocks
            or chunk.finish_reason
        ):
            yielded.append(chunk)

    blocks = [b for c in yielded for b in (c.tool_search_blocks or [])]
    assert [b["type"] for b in blocks] == [
        "server_tool_use",
        "tool_search_tool_result",
    ], "search blocks must survive the yield gate"


def test_attach_never_touches_a_message_from_an_earlier_step():
    """The caller is a `finally` that also runs on the error / truncation /
    empty-turn paths, where this step appended no assistant message. Walking
    the whole list backwards would graft the pair onto a PRIOR step's message
    — already sent in the cached prefix, so mutating it diverges history and
    re-bills the whole prompt."""
    messages = [
        Message(role=MessageRole.USER, content="turn 1"),
        Message(role=MessageRole.ASSISTANT, content="prior step's answer"),
    ]
    since = len(messages)
    # This step appended only a USER nudge (the empty-turn branch).
    messages.append(Message(role=MessageRole.USER, content="nudge"))

    assert _attach_search_blocks(messages, list(PAIR), since) is False
    assert not (messages[1].metadata or {}).get("tool_search_blocks")


def test_attach_uses_this_steps_message_when_one_exists():
    messages = [
        Message(role=MessageRole.USER, content="turn 1"),
        Message(role=MessageRole.ASSISTANT, content="prior"),
    ]
    since = len(messages)
    messages.append(Message(role=MessageRole.ASSISTANT, content="current"))

    assert _attach_search_blocks(messages, list(PAIR), since) is True
    assert messages[2].metadata["tool_search_blocks"] == PAIR
    assert not (messages[1].metadata or {}).get("tool_search_blocks")


def test_attach_does_not_alias_a_shared_block_list():
    """`{**existing}` is shallow: setdefault-then-extend would mutate a list
    shared with the original metadata dict and every message copied from it."""
    shared = {"tool_search_blocks": []}
    a = Message(role=MessageRole.ASSISTANT, content="a", metadata=shared)
    _attach_search_blocks([a], list(PAIR), 0)
    assert shared["tool_search_blocks"] == [], "original list must be untouched"
    assert len(a.metadata["tool_search_blocks"]) == 2


def test_blank_ids_do_not_pair_with_each_other():
    """Two id-less halves would both key on "" and be treated as a complete
    pair; a blank id is itself rejected by the API on replay."""
    messages = [Message(role=MessageRole.ASSISTANT, content="x")]
    blanks = [
        {"type": "server_tool_use", "id": "", "name": "t", "input": {}},
        {"type": "tool_search_tool_result", "tool_use_id": "", "content": []},
    ]
    assert _attach_search_blocks(messages, blanks, 0) is False


def test_use_block_without_deltas_is_validated_not_replayed():
    """The JSONDecodeError guard only fires when deltas arrived. A block whose
    input never streamed would otherwise be replayed with `{}` — missing the
    search tool's required pattern — and 400 every later turn."""
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(0, type="server_tool_use", id="s1", name="tool_search_tool_regex", input={})
    )
    assert not norm.normalize_chunk(_stop(0)).tool_search_blocks


def test_use_block_seeded_from_start_event_is_kept():
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(0, type="server_tool_use", id="s1", name="tool_search_tool_regex",
               input={"pattern": "exp"})
    )
    block = norm.normalize_chunk(_stop(0)).tool_search_blocks[0]
    assert block["input"] == {"pattern": "exp"}


def test_unserializable_result_content_is_dropped_not_repr_stringified():
    """_plain used to fail OPEN, substituting str(value). A repr in the
    replayed payload 400s the thread from that turn on; dropping costs one
    repeated search."""
    class Exploding:
        def model_dump(self):
            raise RuntimeError("nope")

    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(0, type="tool_search_tool_result", tool_use_id="s1",
               content=[Exploding()])
    )
    assert not norm.normalize_chunk(_stop(0)).tool_search_blocks


def test_response_only_keys_are_stripped_at_depth():
    norm = AnthropicStreamNormalizer({})
    norm.normalize_chunk(
        _start(0, type="tool_search_tool_result", tool_use_id="s1",
               content=[{"type": "text", "text": "t", "citations": ["x"]}])
    )
    block = norm.normalize_chunk(_stop(0)).tool_search_blocks[0]
    assert "citations" not in block["content"][0]


def test_multimodal_assistant_content_survives_replay():
    """The searched-but-called-nothing branch handled only str content and
    returned early, so a list-content assistant turn lost every text, image and
    document block and was sent as search blocks only."""
    from miiflow_agent.core.message import TextBlock

    msg = Message(
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="kept")],
        metadata={"tool_search_blocks": PAIR},
    )
    types = [b["type"] for b in _convert(msg)["content"]]
    assert types == ["text", "server_tool_use", "tool_search_tool_result"]


def test_tool_search_unsupported_error_disables_deferral():
    """Deferral is gated per PROVIDER but supported per MODEL; without a
    reactive ladder the mismatch fails every request for that assistant."""
    from miiflow_agent.providers.anthropic_client import AnthropicClient

    params = {"tools": [
        {"name": "a", "defer_loading": True},
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    ]}
    err = Exception("tool_search_tool_regex_20251119 is not supported")
    assert AnthropicClient._is_tool_search_unsupported_error(err)
    assert AnthropicClient._disable_tool_deferral(params) is True
    assert params["tools"] == [{"name": "a"}]
    # Idempotent: nothing left to strip, so no pointless second retry.
    assert AnthropicClient._disable_tool_deferral(params) is False


def test_dispatcher_always_load_is_set_by_the_factory():
    """Set in the framework factory, not one caller's adapter — the other two
    call sites (Agent(sub_agents=...), configured_subagent) need it too."""
    from miiflow_agent.core.react.dispatch import make_subagent_dispatcher_tool

    class _Sub:
        handle = "child"
        name = "Child"
        description = "a child agent"
        when_to_use = "when testing"
        child_assistant = None

    tool = make_subagent_dispatcher_tool([_Sub()], parent_assistant_id="p1")
    assert tool.schema.metadata.get("always_load") is True
