"""Volatile blocks and the cache anchor on the Anthropic wire.

A turn's request must be a byte prefix of the next turn's, or the next turn
cannot read the conversation from cache. Two mechanisms keep it that way:
per-request context rides in a volatile block AFTER the final breakpoint, and
the previous turn's user message carries a breakpoint of its own (the anchor).
"""

from miiflow_agent.core import wire_shape
from miiflow_agent.core.agent import _content_text, _ends_with_tool_answer
from miiflow_agent.core.message import Message, MessageRole, TextBlock
from miiflow_agent.providers.anthropic_client import AnthropicClient


def _client():
    return AnthropicClient(model="claude-sonnet-5", api_key="k", cache_ttl="1h")


def _request(client, messages):
    _, wire_messages = client._prepare_messages(messages)
    params = {
        "system": "system prompt",
        "tools": [{"name": "t", "input_schema": {"type": "object"}}],
        "messages": wire_messages,
    }
    client._apply_prompt_caching(params, ttl="1h", conversation_ttl="1h")
    return params


def _marked(block):
    return isinstance(block, dict) and "cache_control" in block


def _private_keys(params):
    keys = set()
    for msg in params["messages"]:
        keys |= {k for k in msg if k.startswith("_miiflow")}
        for block in msg["content"] if isinstance(msg["content"], list) else []:
            keys |= {k for k in block if k.startswith("_miiflow")}
    return keys


def test_volatile_block_goes_after_the_breakpoint_and_the_rest_replays_identically():
    client = _client()
    live = _request(
        client,
        [Message.user([TextBlock(text=" what now? "), TextBlock(text="[PAGE]", volatile=True)])],
    )
    replay = _request(client, [Message.user(" what now? ")])

    final = live["messages"][-1]["content"]
    assert final[0]["text"] == "what now?"  # stripped, exactly like the string path
    assert _marked(final[0])
    assert final[1] == {"type": "text", "text": "[PAGE]"}  # after the breakpoint
    assert replay["messages"][-1]["content"] == [final[0]]
    assert not _private_keys(live)


def test_volatile_block_on_a_tool_result_sits_after_the_tool_result():
    client = _client()
    params = _request(
        client,
        [
            Message.user("q"),
            Message.assistant("", tool_calls=[{"id": "c1", "function": {"name": "ask", "arguments": {}}}]),
            Message.tool(
                content=[TextBlock(text="Yes"), TextBlock(text="[FACTS]", volatile=True)],
                tool_call_id="c1",
            ),
        ],
    )
    final = params["messages"][-1]["content"]
    assert final[0] == {
        "type": "tool_result",
        "tool_use_id": "c1",
        "content": "Yes",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    assert final[1] == {"type": "text", "text": "[FACTS]"}


def test_anchor_gets_its_own_breakpoint_and_never_a_fifth():
    client = _client()
    params = _request(
        client,
        [
            Message.user("first", metadata={"cache_anchor": True}),
            Message.assistant("answer"),
            Message.user("second"),
        ],
    )
    msgs = params["messages"]
    assert _marked(msgs[0]["content"][0])
    assert _marked(msgs[-1]["content"][0])
    breakpoints = (
        sum(_marked(t) for t in params["tools"])
        + sum(_marked(b) for b in params["system"])
        + sum(_marked(b) for m in msgs for b in m["content"] if isinstance(m["content"], list))
    )
    assert breakpoints == 4
    assert not _private_keys(params)


def test_content_text_ignores_volatile_blocks():
    assert _content_text([TextBlock(text="hi"), TextBlock(text="[PAGE]", volatile=True)]) == "hi"


def test_a_trailing_tool_result_is_a_resume_turn():
    user = Message(role=MessageRole.USER, content="q")
    tool = Message.tool(content="answer", tool_call_id="c1")
    system = Message(role=MessageRole.SYSTEM, content="s")
    assert _ends_with_tool_answer([user, tool])
    assert _ends_with_tool_answer([user, tool, system])
    assert not _ends_with_tool_answer([tool, user])
    assert not _ends_with_tool_answer([])


def test_wire_shape_records_counts_and_per_message_hashes():
    client = _client()
    slot = wire_shape.open_slot()
    params = _request(client, [Message.user("a"), Message.assistant("b"), Message.user("c")])
    client._record_wire_shape(params)
    assert slot["tools_loaded"] == 1
    assert slot["tools_deferred"] == 0
    assert len(slot["msg_hashes"]) == 3
    # Hashes ignore cache markers: the same message hashes the same whether or
    # not this request put a breakpoint on it.
    unmarked = _request(client, [Message.user("a"), Message.assistant("b")])
    wire_shape.open_slot()
    client._record_wire_shape(unmarked)
    assert wire_shape._SLOT.get()["msg_hashes"][:2] == slot["msg_hashes"][:2]
