"""Every provider must actually deliver a hosted attachment's CONTENT.

Five providers each hand-rolled document handling and three sent the model a
URL to the stored file, assuming the API would fetch it. None of them do for
text: Anthropic's `url` document source is PDF-only, OpenRouter's `file_data`
wants a base64 data URI, and OpenAI's Responses `input_file` fetches PDFs. In
each case the attachment was dropped with no fetch and no error (BUG-062).

These are parametrised across providers on purpose. The bug was a per-provider
drift, so the guard has to be a rule that holds for all of them rather than a
test that lives next to one client.
"""

import asyncio

import pytest
from unittest.mock import MagicMock, patch

from miiflow_agent.core import Message, MessageRole
from miiflow_agent.core.message import DocumentBlock
from miiflow_agent.providers.anthropic_client import AnthropicClient
from miiflow_agent.providers.gemini_client import GeminiClient
from miiflow_agent.providers.openai_client import OpenAIClient
from miiflow_agent.providers.openrouter_client import OpenRouterClient

HOSTED_TXT = "https://storage.example.com/org/notes.txt"
DOC_TEXT = "Q3 spend was 42000 and CAC fell to 18.10."


def _all_text(payload) -> str:
    """Collect every string in a provider payload, whatever its shape."""
    found = []

    def walk(node):
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)
    return "\n".join(found)


def _txt_block():
    return DocumentBlock(document_url=HOSTED_TXT, document_type="txt", filename="notes.txt")


def _message(block):
    return Message(role=MessageRole.USER, content=[block])


def _anthropic(block):
    client = AnthropicClient(model="claude-3-haiku-20240307", api_key="k", timeout=30.0)
    return client.convert_message_to_provider_format(_message(block))


def _openai_chat(block):
    return OpenAIClient.convert_message_to_openai_format(_message(block))


def _openai_responses(block):
    client = OpenAIClient(model="gpt-5", api_key="k")
    return client._convert_blocks_to_responses_content([block])


def _openrouter(block):
    # Only the DeepSeek, GLM and Grok families are allowed through OpenRouter.
    client = OpenRouterClient(model="deepseek/deepseek-v4-pro-0813", api_key="k")
    return client._convert_message_to_dict(_message(block))


def _gemini(block):
    client = GeminiClient(model="gemini-2.0-flash", api_key="k")
    return asyncio.run(client._convert_messages_to_gemini_format([_message(block)]))


CONVERTERS = {
    "anthropic": _anthropic,
    "openai_chat": _openai_chat,
    "openai_responses": _openai_responses,
    "openrouter": _openrouter,
    "gemini": _gemini,
}


@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_hosted_text_attachment_content_reaches_the_model(name):
    """The regression, stated once for every provider."""
    response = MagicMock()
    response.content = DOC_TEXT.encode("utf-8")
    response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=response) as mock_get:
        payload = CONVERTERS[name](_txt_block())

    text = _all_text(payload)
    assert DOC_TEXT in text, f"{name} did not deliver the document's content"
    assert "notes.txt" in text, f"{name} did not label the document"
    assert mock_get.called, f"{name} never fetched the hosted document"
    assert HOSTED_TXT not in text.replace(DOC_TEXT, ""), (
        f"{name} passed the storage URL through instead of the content"
    )


@pytest.mark.parametrize("name", sorted(CONVERTERS))
def test_unreachable_attachment_is_reported_not_dropped(name):
    """A fetch failure must be visible to the model, never silence."""
    with patch("httpx.get", side_effect=RuntimeError("connection refused")):
        payload = CONVERTERS[name](_txt_block())
    text = _all_text(payload)
    assert "notes.txt" in text and "connection refused" in text
