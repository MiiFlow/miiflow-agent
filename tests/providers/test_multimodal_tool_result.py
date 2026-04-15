"""Multimodal TOOL-message conversion across providers.

Covers the analyze_creative path: tool returns LlmBlockInjection → orchestrator
builds a TOOL-role Message with list content → provider converts it into the
right tool-result shape for each backend.
"""

import pytest
from unittest.mock import patch

from miiflow_agent.core.message import (
    ImageBlock,
    Message,
    MessageRole,
    TextBlock,
    VideoBlock,
)
from miiflow_agent.providers.anthropic_client import AnthropicClient
from miiflow_agent.providers.openai_client import OpenAIClient


@pytest.fixture
def anthropic_client():
    return AnthropicClient(model="claude-opus-4-6", api_key="test-key", timeout=30.0)


class TestAnthropicToolResultMultimodal:
    def test_list_content_becomes_nested_tool_result(self, anthropic_client):
        msg = Message(
            role=MessageRole.TOOL,
            content=[
                TextBlock(text="Injected 2 creatives"),
                ImageBlock(image_url="https://cdn.example.com/foo.jpg"),
                ImageBlock(image_url="https://i.ytimg.com/vi/abc123/hqdefault.jpg"),
            ],
            tool_call_id="tool_xyz",
        )
        out = anthropic_client.convert_message_to_provider_format(msg)
        assert out["role"] == "user"
        assert len(out["content"]) == 1
        tool_result = out["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_xyz"
        # Summary text + two image blocks
        sub = tool_result["content"]
        assert [b["type"] for b in sub] == ["text", "image", "image"]
        # URL source form (not base64) for external URLs
        assert sub[1]["source"]["type"] == "url"
        assert sub[1]["source"]["url"] == "https://cdn.example.com/foo.jpg"

    def test_data_uri_image_uses_base64_source(self, anthropic_client):
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        msg = Message(
            role=MessageRole.TOOL,
            content=[
                TextBlock(text="one creative"),
                ImageBlock(image_url=data_uri),
            ],
            tool_call_id="t1",
        )
        out = anthropic_client.convert_message_to_provider_format(msg)
        sub = out["content"][0]["content"]
        assert sub[1]["type"] == "image"
        assert sub[1]["source"]["type"] == "base64"
        assert sub[1]["source"]["media_type"] == "image/png"

    def test_video_block_degrades_to_text(self, anthropic_client):
        msg = Message(
            role=MessageRole.TOOL,
            content=[
                TextBlock(text="one video creative"),
                VideoBlock(video_url="https://video.example.com/x.mp4"),
            ],
            tool_call_id="t2",
        )
        out = anthropic_client.convert_message_to_provider_format(msg)
        sub = out["content"][0]["content"]
        # Text + text (video fallback) — no image/video block for Claude
        types = [b["type"] for b in sub]
        assert types == ["text", "text"]
        assert "cannot view videos" in sub[1]["text"]
        assert "https://video.example.com/x.mp4" in sub[1]["text"]

    def test_empty_list_content_gets_placeholder(self, anthropic_client):
        msg = Message(role=MessageRole.TOOL, content=[], tool_call_id="t3")
        out = anthropic_client.convert_message_to_provider_format(msg)
        sub = out["content"][0]["content"]
        assert sub == [{"type": "text", "text": "[empty result]"}]

    def test_whitespace_only_text_block_filtered(self, anthropic_client):
        msg = Message(
            role=MessageRole.TOOL,
            content=[TextBlock(text="   "), ImageBlock(image_url="https://x.com/a.jpg")],
            tool_call_id="t4",
        )
        out = anthropic_client.convert_message_to_provider_format(msg)
        sub = out["content"][0]["content"]
        assert [b["type"] for b in sub] == ["image"]

    def test_string_content_path_unchanged(self, anthropic_client):
        """Regression: existing string tool_result path must still work."""
        msg = Message(role=MessageRole.TOOL, content="plain text", tool_call_id="t5")
        out = anthropic_client.convert_message_to_provider_format(msg)
        assert out["content"][0]["content"] == "plain text"


class TestOpenAIToolResultFallback:
    """OpenAI's chat API requires string content for tool messages.
    Multimodal TOOL messages must collapse to text with URL references."""

    def test_list_content_collapses_to_text_with_url_notes(self):
        msg = Message(
            role=MessageRole.TOOL,
            content=[
                TextBlock(text="Injected 2 creatives"),
                ImageBlock(image_url="https://cdn.example.com/foo.jpg"),
                VideoBlock(video_url="https://video.example.com/x.mp4"),
            ],
            tool_call_id="t1",
        )
        out = OpenAIClient.convert_message_to_openai_format(msg)
        assert out["role"] == "tool"
        assert out["tool_call_id"] == "t1"
        assert isinstance(out["content"], str)
        # Original text preserved
        assert "Injected 2 creatives" in out["content"]
        # URLs surfaced as references
        assert "https://cdn.example.com/foo.jpg" in out["content"]
        assert "https://video.example.com/x.mp4" in out["content"]
        # Notes about API limitation
        assert "OpenAI" in out["content"]

    def test_empty_list_content_becomes_placeholder(self):
        msg = Message(role=MessageRole.TOOL, content=[], tool_call_id="t2")
        out = OpenAIClient.convert_message_to_openai_format(msg)
        assert out["content"] == "[empty result]"

    def test_string_content_path_unchanged(self):
        msg = Message(role=MessageRole.TOOL, content="plain", tool_call_id="t3")
        out = OpenAIClient.convert_message_to_openai_format(msg)
        assert out["content"] == "plain"
