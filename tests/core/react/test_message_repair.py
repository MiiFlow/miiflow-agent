"""`repair_tool_pairing` must understand which calls a TOOL message answers.

Provider-executed MCP calls are answered inline on the assistant message
(`metadata["mcp_tool_results"]` → `mcp_tool_result` block), never by a TOOL
message. Treating them as pending made the repair "fix" a valid history by
synthesizing a local `tool_result` for an `mcptoolu_` id — a second protocol
error on top of the one that triggered the repair
(thread_9Fm7LXCFbePThEBd1qnvv7Dg, 2026-09-01).
"""

from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.message_repair import repair_tool_pairing

MCP_ID = "mcptoolu_01Eb3VDK8t2nDN22ghEH1sWq"


def _assistant(tool_calls, **metadata):
    return Message(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=tool_calls,
        metadata=metadata or None,
    )


def _mcp_call():
    return {
        "id": MCP_ID,
        "type": "mcp_function",
        "server_name": "GitHub",
        "function": {"name": "search_repositories", "arguments": {"q": "org:x"}},
    }


def _local_call(call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "grep_files", "arguments": {"pattern": "x"}},
    }


class TestMcpCallsAreNotPending:
    def test_an_mcp_call_with_no_tool_message_is_not_an_anomaly(self):
        history = [
            Message(role=MessageRole.USER, content="find it"),
            _assistant(
                [_mcp_call()],
                mcp_tool_results=[{"tool_use_id": MCP_ID, "is_error": False, "content": "{}"}],
            ),
            Message(role=MessageRole.USER, content="thanks"),
        ]

        repaired, anomalies = repair_tool_pairing(history)

        assert anomalies == []
        assert [m.role for m in repaired] == [m.role for m in history]
        assert not any(m.role == MessageRole.TOOL for m in repaired)

    def test_mixed_turn_only_needs_results_for_the_local_calls(self):
        history = [
            Message(role=MessageRole.USER, content="find it"),
            _assistant(
                [_local_call("toolu_a"), _mcp_call(), _local_call("toolu_b")],
                mcp_tool_results=[{"tool_use_id": MCP_ID, "is_error": True, "content": "nope"}],
            ),
            Message(role=MessageRole.TOOL, content="(no matches)", tool_call_id="toolu_a"),
            Message(role=MessageRole.TOOL, content="(no matches)", tool_call_id="toolu_b"),
            Message(role=MessageRole.USER, content="ok"),
        ]

        repaired, anomalies = repair_tool_pairing(history)

        assert anomalies == []
        assert len(repaired) == len(history)

    def test_an_unanswered_local_call_beside_an_mcp_call_is_still_repaired(self):
        history = [
            Message(role=MessageRole.USER, content="find it"),
            _assistant(
                [_mcp_call(), _local_call("toolu_a")],
                mcp_tool_results=[{"tool_use_id": MCP_ID, "is_error": False, "content": "{}"}],
            ),
            Message(role=MessageRole.USER, content="ok"),
        ]

        repaired, anomalies = repair_tool_pairing(history)

        assert anomalies == ["synthesized missing tool_result for toolu_a"]
        synthesized = [m for m in repaired if m.role == MessageRole.TOOL]
        assert [m.tool_call_id for m in synthesized] == ["toolu_a"]
