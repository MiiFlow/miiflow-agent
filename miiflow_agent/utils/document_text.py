"""Turning a DocumentBlock into text every provider can actually read.

This exists because five providers each hand-rolled it and three got it wrong,
all in the same direction: they handed the model a URL to the stored file and
assumed the API would go and fetch it. Anthropic's `url` document source is
PDF-only, OpenRouter's `file_data` wants a base64 data URI, and OpenAI's
Responses `input_file` documents PDF. In each case a text attachment was
dropped in transit with no fetch, no bytes and no error, so the user saw the
file on their message and an answer that had not read it (BUG-062).

The rule that holds everywhere: a provider-native document block is worth using
only when the bytes travel with it (inline base64) or the type is one the
provider actually fetches (PDF). Everything else is inlined as text by us.
"""

from typing import Optional

# Read cap. A single attachment should not be able to push a request past the
# model's context window; the tail is dropped with a visible marker rather than
# silently, so the model can say what it did not see.
MAX_DOCUMENT_CHARS = 400_000


def is_inline_data(document_url: str) -> bool:
    """True when the bytes are already in the URL, so no fetch is needed."""
    return document_url.startswith("data:")


def fetch_document_text(
    document_url: str,
    filename: Optional[str] = None,
    *,
    timeout: float = 30.0,
) -> str:
    """Download a hosted document and label it for the model.

    Never raises: a failure is reported in the returned text. An attachment that
    vanishes is the bug this module exists to prevent, and a visible error the
    model can relay beats an answer built on a file it was never given.
    """
    label = f" [{filename}]" if filename else ""
    try:
        import httpx

        response = httpx.get(document_url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        text = response.content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - reported to the model, never raised
        return f"[Error processing document{' ' + filename if filename else ''}: {exc}]"

    return f"[Document{label}]\n\n{_capped(text)}"


def document_to_text(
    document_url: str,
    document_type: str = "",
    filename: Optional[str] = None,
) -> str:
    """Render any document as text, extracting PDFs rather than decoding them.

    A PDF put through the plain-text path decodes to binary noise, so the two
    are dispatched here once instead of at each call site, which is where the
    providers drifted apart.
    """
    label = f" [{filename}]" if filename else ""
    if document_type == "pdf":
        try:
            from .pdf_extractor import extract_pdf_text_simple

            return f"[PDF Document{label}]\n\n{_capped(extract_pdf_text_simple(document_url))}"
        except Exception as exc:  # noqa: BLE001 - reported to the model, never raised
            return f"[Error processing document{' ' + filename if filename else ''}: {exc}]"
    return fetch_document_text(document_url, filename)


def _capped(text: str) -> str:
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    omitted = len(text) - MAX_DOCUMENT_CHARS
    return f"{text[:MAX_DOCUMENT_CHARS]}\n\n[truncated: {omitted} more characters]"
