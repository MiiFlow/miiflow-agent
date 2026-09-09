"""Text-only histories are prepared on the event loop, not on a worker thread.

``_prepare_messages`` was always offloaded with ``asyncio.to_thread`` because
image blocks may download and resize. On a text-only history — every agent
step in the common case — that thread-pool round trip was pure pre-request
latency, so the offload is now gated on the history actually carrying a
non-text block.
"""

import pytest

from miiflow_agent.core.message import (
    DocumentBlock,
    ImageBlock,
    Message,
    MessageRole,
    TextBlock,
)
from miiflow_agent.providers.anthropic_client import AnthropicClient

pytestmark = pytest.mark.unit


def test_string_and_text_block_histories_stay_on_the_loop():
    messages = [
        Message(role=MessageRole.SYSTEM, content="be brief"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.USER, content=[TextBlock(text="hello"), TextBlock(text="again")]),
        Message(role=MessageRole.TOOL, content="result", tool_call_id="c1"),
    ]
    assert AnthropicClient._prepare_needs_thread(messages) is False


@pytest.mark.parametrize(
    "block",
    [ImageBlock(image_url="https://example.com/a.png"), DocumentBlock(document_url="https://example.com/a.pdf")],
)
def test_multimodal_block_anywhere_in_history_offloads(block):
    messages = [
        Message(role=MessageRole.USER, content="look"),
        Message(role=MessageRole.USER, content=[TextBlock(text="see"), block]),
        Message(role=MessageRole.ASSISTANT, content="ok"),
    ]
    assert AnthropicClient._prepare_needs_thread(messages) is True
