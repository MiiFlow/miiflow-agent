"""Span processors installed on the tracing pipeline (Arize AX / Phoenix).

Two concerns that must hold for EVERY exported span — the OpenInference LLM and
embedding spans the SDK instrumentors author as much as the agent spans this
package authors — and therefore live on the ``TracerProvider`` rather than at
call sites:

``ContextStampingProcessor``
    Copies the ambient ``CallbackContext`` (org, thread, usage session) onto
    each span as it starts. Without it only root agent spans carried
    ``organization.id``; child agents, every LLM/embedding span, and the
    hundreds of standalone LLM calls made outside an agent (suggestion
    scoring, memory embeddings, KB indexing) landed in Arize with no owner and
    no use case, so nothing could be filtered or costed per org.

``SecretRedactingProcessor``
    Rewrites credential-bearing attributes before export. The instrumentors
    record the full request — ``llm.invocation_parameters`` and ``input.value``
    included — and a request that carries ``mcp_servers[].authorization_token``
    (Triple Whale et al.) or an API key therefore shipped that secret to a
    third party in plain text. Runs in ``on_end`` on the ``ReadableSpan``'s
    attribute mapping, which is the last hook before the exporter sees it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

try:  # optional dependency — the package must import without opentelemetry
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor
except Exception:  # pragma: no cover - exercised only without the SDK

    class _SpanProcessor:  # type: ignore[no-redef]
        """No-op stand-in so this module imports without the OTel SDK."""

        def on_start(self, span, parent_context=None):  # noqa: D401
            return None

        def on_end(self, span):
            return None

        def shutdown(self):
            return None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True


# ── Context stamping ─────────────────────────────────────────────────────

# OpenInference / Arize keys. `session.id` is what groups a conversation in
# the Arize sessions view; the rest are filters.
ORGANIZATION_ID = "organization.id"
THREAD_ID = "thread.id"
USAGE_SESSION_ID = "usage_session.id"
USAGE_SESSION_TYPE = "usage_session.type"
AGENT_NODE_RUN_ID = "agent_node_run.id"


def context_attributes(ctx: Any) -> Dict[str, str]:
    """Attributes derived from a ``CallbackContext``-shaped object.

    Only non-empty values are returned, so a stamping pass never overwrites a
    value a span author set explicitly with an empty one.
    """
    if ctx is None:
        return {}
    out: Dict[str, str] = {}
    org = getattr(ctx, "organization_id", None)
    if org:
        out[ORGANIZATION_ID] = str(org)
    thread = getattr(ctx, "thread_id", None)
    if thread:
        out[THREAD_ID] = str(thread)
    run_id = getattr(ctx, "agent_node_run_id", None)
    if run_id:
        out[AGENT_NODE_RUN_ID] = str(run_id)
    metadata = getattr(ctx, "metadata", None) or {}
    session_id = metadata.get("usage_session_id")
    if session_id:
        out[USAGE_SESSION_ID] = str(session_id)
    session_type = metadata.get("usage_session_type")
    if session_type:
        out[USAGE_SESSION_TYPE] = str(session_type)
    return out


class ContextStampingProcessor(_SpanProcessor):
    """Stamp the ambient CallbackContext onto every span at start.

    Reads the contextvar lazily so this module has no import-time dependency
    on the callback machinery, and never raises: telemetry must not be able
    to fail the call it observes.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            from ..callback_context import get_callback_context

            attrs = context_attributes(get_callback_context())
        except Exception:  # noqa: BLE001
            return
        if not attrs:
            return
        try:
            existing = getattr(span, "attributes", None) or {}
            for key, value in attrs.items():
                # Explicit values from the span author win (a root agent span
                # sets organization.id from the assistant, not the context).
                if existing.get(key) in (None, ""):
                    span.set_attribute(key, value)
        except Exception:  # noqa: BLE001
            return

    def on_end(self, span: Any) -> None:  # noqa: D401 - nothing to do
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ── Secret redaction ─────────────────────────────────────────────────────

REDACTED = "[REDACTED]"

# JSON keys whose VALUE is a credential wherever they appear (any nesting).
# Matched case-insensitively on the full key. Kept deliberately narrow — an
# over-broad rule (e.g. bare "token") would blank `max_tokens`, usage counts
# and tool-search results.
_SECRET_KEYS = frozenset(
    {
        "authorization_token",
        "authorization",
        "api_key",
        "apikey",
        "x-api-key",
        "access_token",
        "refresh_token",
        "client_secret",
        "secret",
        "secret_value",
        "password",
        "bearer_token",
        "private_key",
    }
)

# Attribute names that carry request payloads the instrumentors serialise.
# `input.value` is the whole kwargs dict for LLM spans; invocation_parameters
# is kwargs minus messages; the message flattenings can carry tool inputs.
_PAYLOAD_ATTRIBUTE_PREFIXES = (
    "llm.invocation_parameters",
    "input.value",
    "output.value",
    "llm.input_messages",
    "llm.output_messages",
    "llm.tools",
    "tool.parameters",
    "metadata",
)

# Bearer / raw-JWT patterns inside plain (non-JSON) strings.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
# `"authorization_token": "…"` inside text that is NOT parseable JSON — the
# SDK truncates attribute values at set time (32K), so a long `input.value`
# can be cut mid-document and arrive here as a JSON prefix. Opaque tokens
# (OAuth access tokens, org API keys) match none of the regexes above, so
# the key:value shape is scrubbed textually as the last line of defence.
_KEY_VALUE_RE = re.compile(
    r'("(?:' + "|".join(re.escape(k) for k in sorted(_SECRET_KEYS)) + r')"\s*:\s*")([^"\\]*(?:\\.[^"\\]*)*)(")',
    re.IGNORECASE,
)


def _is_secret_key(key: Any) -> bool:
    return isinstance(key, str) and key.lower() in _SECRET_KEYS


def redact_structure(value: Any) -> Any:
    """Return a copy of ``value`` with credential-bearing leaves replaced.

    Walks dicts/lists; leaves other types alone. Strings that are themselves
    JSON are NOT parsed here — that is the caller's job (see
    :func:`redact_text`) so a JSON payload nested as a string inside a dict
    field still gets one parse.
    """
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_secret_key(k) and v not in (None, "") else redact_structure(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_structure(v) for v in value]
    return value


def redact_text(text: str) -> str:
    """Redact a serialised attribute value.

    JSON is parsed, redacted structurally and re-serialised (compact
    separators — the instrumentors already emit compact JSON, so the round
    trip is byte-stable for payloads with nothing to redact). Anything that is
    not JSON gets the bearer/JWT regexes only.
    """
    if not text:
        return text
    out = text
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None:
            redacted = redact_structure(parsed)
            if redacted != parsed:
                out = json.dumps(redacted, separators=(",", ":"), ensure_ascii=False)
        else:
            # Truncated / malformed JSON: scrub the key:value shape textually.
            out = _KEY_VALUE_RE.sub(
                lambda m: m.group(1) + REDACTED + m.group(3) if m.group(2) else m.group(0),
                out,
            )
    # ALWAYS run the text patterns too — a structurally-redacted payload can
    # still carry `Authorization: Bearer …` or a JWT inside a message string
    # or tool result (the structural pass looks at keys, not string contents).
    out = _BEARER_RE.sub("Bearer " + REDACTED, out)
    out = _JWT_RE.sub(REDACTED, out)
    return out


def _is_payload_attribute(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in _PAYLOAD_ATTRIBUTE_PREFIXES)


def redact_attributes(attributes: Any) -> int:
    """Redact in place; returns the number of attributes rewritten.

    ``attributes`` is the SDK's ``BoundedAttributes`` (a mutable mapping) on
    a ``ReadableSpan``. Writing through it after ``on_end`` is the documented
    way for processors to scrub data before export.
    """
    if not attributes:
        return 0
    changed = 0
    try:
        keys = list(attributes.keys())
    except Exception:  # noqa: BLE001
        return 0
    for key in keys:
        if not isinstance(key, str) or not _is_payload_attribute(key):
            continue
        value = attributes.get(key)
        if not isinstance(value, str):
            continue
        redacted = redact_text(value)
        if redacted != value:
            try:
                attributes[key] = redacted
                changed += 1
            except Exception:  # noqa: BLE001
                # BoundedAttributes can be frozen on some SDK versions; fall
                # back to the underlying dict if it exposes one.
                inner = getattr(attributes, "_dict", None)
                if inner is not None:
                    inner[key] = redacted
                    changed += 1
    return changed


class SecretRedactingProcessor(_SpanProcessor):
    """Scrub credentials from payload attributes before export."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: Any) -> None:
        try:
            attributes = getattr(span, "_attributes", None)
            if attributes is None:
                attributes = getattr(span, "attributes", None)
            n = redact_attributes(attributes)
            if n:
                logger.debug("redacted %d attribute(s) on span %s", n, getattr(span, "name", "?"))
        except Exception:  # noqa: BLE001 — never fail export over redaction
            return

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ── Span size bounding ───────────────────────────────────────────────────

# Byte budget for the `llm.input_messages.*` family on one LLM span. The SDK
# caps each attribute VALUE (32K via span_limits()) and the attribute COUNT
# (128, the OTel default), but 128 attrs x 32K is still ~4MB for a single
# span that replays a long history — and an OTLP collector that refuses the
# payload drops it silently (spans simply absent — the exact prod symptom of
# root-agent LLM spans never reaching Arize). 256KB keeps the newest turns a
# trace reader actually looks at.
SPAN_MESSAGE_BYTE_LIMIT_ENV = "MIIFLOW_SPAN_MAX_MESSAGE_BYTES"
DEFAULT_SPAN_MESSAGE_BYTE_LIMIT = 262_144
_INPUT_MESSAGES_PREFIX = "llm.input_messages."
MESSAGES_TRUNCATED_ATTR = "miiflow.span.messages_truncated"


def message_byte_limit() -> int:
    import os

    raw = os.getenv(SPAN_MESSAGE_BYTE_LIMIT_ENV)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_SPAN_MESSAGE_BYTE_LIMIT


def _write_attribute(attributes: Any, key: str, value: Any) -> None:
    try:
        attributes[key] = value
    except Exception:  # noqa: BLE001 — frozen BoundedAttributes fallback
        inner = getattr(attributes, "_dict", None)
        if inner is not None:
            inner[key] = value


def _delete_attribute(attributes: Any, key: str) -> None:
    try:
        del attributes[key]
    except Exception:  # noqa: BLE001 — frozen BoundedAttributes fallback
        inner = getattr(attributes, "_dict", None)
        if inner is not None:
            inner.pop(key, None)


def bound_message_attributes(attributes: Any, byte_limit: int) -> int:
    """Drop the OLDEST whole messages until the family fits; returns count dropped.

    Message indices are chronological (0 = oldest), so dropping from the low
    end keeps the recent turns a trace reader actually looks at. Whole
    messages only — a message with half its attributes removed renders as
    garbage in the trace viewer. Byte cost is the serialized length of each
    attribute's value; keys are negligible next to 32K content values.
    """
    if not attributes:
        return 0
    try:
        keys = list(attributes.keys())
    except Exception:  # noqa: BLE001
        return 0
    by_message: Dict[int, list] = {}
    for key in keys:
        if not isinstance(key, str) or not key.startswith(_INPUT_MESSAGES_PREFIX):
            continue
        index_str = key[len(_INPUT_MESSAGES_PREFIX):].split(".", 1)[0]
        if index_str.isdigit():
            by_message.setdefault(int(index_str), []).append(key)
    if not by_message:
        return 0

    def _cost(message_keys: list) -> int:
        total = 0
        for key in message_keys:
            try:
                total += len(str(attributes.get(key, "")))
            except Exception:  # noqa: BLE001
                pass
        return total

    costs = {index: _cost(message_keys) for index, message_keys in by_message.items()}
    total_bytes = sum(costs.values())
    if total_bytes <= byte_limit:
        return 0
    dropped = 0
    for index in sorted(by_message):
        if total_bytes <= byte_limit:
            break
        for key in by_message[index]:
            _delete_attribute(attributes, key)
        total_bytes -= costs[index]
        dropped += 1
    if dropped:
        _write_attribute(attributes, MESSAGES_TRUNCATED_ATTR, dropped)
    return dropped


class SpanSizeBoundingProcessor(_SpanProcessor):
    """Bound the per-span message-attribute count before export.

    Runs AFTER SecretRedactingProcessor (redaction rewrites values; bounding
    deletes keys — order keeps both effective). LLM spans only: they are the
    only kind that replays the whole conversation into attributes.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: Any) -> None:
        try:
            attributes = getattr(span, "_attributes", None)
            if attributes is None:
                attributes = getattr(span, "attributes", None)
            if not attributes:
                return
            if attributes.get("openinference.span.kind") != "LLM":
                return
            n = bound_message_attributes(attributes, message_byte_limit())
            if n:
                logger.warning(
                    "[TRACE] bounded oversized LLM span %s: dropped %d oldest "
                    "input message(s) so the span stays exportable",
                    getattr(span, "name", "?"),
                    n,
                )
        except Exception:  # noqa: BLE001 — never fail export over bounding
            return

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def install_processors(tracer_provider: Any) -> Iterable[str]:
    """Attach the standard processors to ``tracer_provider``.

    Order matters: processors run in registration order, and the exporter's
    ``BatchSpanProcessor`` must be registered AFTER these so it sees the
    stamped, redacted, and size-bounded attributes. Returns the names
    installed (for logs).
    """
    installed = []
    for proc in (
        ContextStampingProcessor(),
        SecretRedactingProcessor(),
        SpanSizeBoundingProcessor(),
    ):
        try:
            tracer_provider.add_span_processor(proc)
            installed.append(type(proc).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to install %s: %s", type(proc).__name__, exc)
    return installed
