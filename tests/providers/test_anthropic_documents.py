"""What a DocumentBlock actually becomes on the wire.

The Anthropic `document` block accepts a `url` source for PDFs ONLY. A txt
document sent that way is dropped in silence, which is how an attached
.txt/.md/.json reached the model unread (BUG-062). The server maps all of
those to document_type "txt", so the boundary these tests pin is: a hosted
txt document must arrive as readable text, and a PDF must keep its url source.
"""

import pytest
from unittest.mock import MagicMock, patch

from miiflow_agent.core import Message, MessageRole
from miiflow_agent.core.message import DocumentBlock
from miiflow_agent.providers.anthropic_client import AnthropicClient

HOSTED_TXT = "https://storage.example.com/org/notes.txt"
DOC_TEXT = "Q3 spend was 42000 and CAC fell to 18.10."


@pytest.fixture
def client():
    return AnthropicClient(model="claude-3-haiku-20240307", api_key="test-key", timeout=30.0)


def _convert(client, block):
    return client.convert_message_to_provider_format(
        Message(role=MessageRole.USER, content=[block])
    )


def test_hosted_txt_arrives_as_readable_text(client):
    """The regression: the document's CONTENT must be in the payload."""
    response = MagicMock()
    response.content = DOC_TEXT.encode("utf-8")
    response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=response) as mock_get:
        result = _convert(
            client,
            DocumentBlock(document_url=HOSTED_TXT, document_type="txt", filename="notes.txt"),
        )

    mock_get.assert_called_once()
    blocks = result["content"]
    assert all(b["type"] != "document" for b in blocks), "hosted txt must not use a document block"
    text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
    assert DOC_TEXT in text
    assert "notes.txt" in text


def test_pdf_still_uses_the_url_source(client):
    """The url source is valid for PDFs, so that path must not change."""
    result = _convert(
        client, DocumentBlock(document_url="https://x.test/report.pdf", document_type="pdf")
    )
    docs = [b for b in result["content"] if b["type"] == "document"]
    assert len(docs) == 1
    assert docs[0]["source"] == {"type": "url", "url": "https://x.test/report.pdf"}


def test_inline_base64_txt_stays_a_document_block(client):
    """A data: URI carries the bytes, so Claude can take it as a document."""
    result = _convert(
        client,
        DocumentBlock(document_url="data:text/plain;base64,SGVsbG8=", document_type="txt"),
    )
    docs = [b for b in result["content"] if b["type"] == "document"]
    assert len(docs) == 1
    assert docs[0]["source"]["type"] == "base64"


def test_unreachable_txt_reports_instead_of_vanishing(client):
    """A fetch failure must say so in the payload, not drop the attachment."""
    with patch("httpx.get", side_effect=RuntimeError("connection refused")):
        result = _convert(
            client,
            DocumentBlock(document_url=HOSTED_TXT, document_type="txt", filename="notes.txt"),
        )
    text = "\n".join(b["text"] for b in result["content"] if b["type"] == "text")
    assert "notes.txt" in text and "connection refused" in text
