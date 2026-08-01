"""Local token estimation, corrected against real provider counts.

Design constraint: **no network round-trip on the estimation path.** Anthropic
and Gemini both expose exact ``count_tokens`` endpoints, and both are remote.
Putting one in front of the first LLM call adds latency to precisely the path
we are trying to make faster, and caching only helps from the second run on —
the cold path still pays. So estimation is local, always.

Accuracy comes from two places instead:

  1. **Content-aware ratios.** The old estimator used a flat 4 chars/token for
     everything. That is roughly right for English prose and materially wrong
     for the JSON that dominates a tool-heavy request: schemas and tool results
     tokenize closer to 3.0 chars/token because punctuation, braces and quoted
     keys fragment badly. A flat 4 therefore *under*-counts the largest tier,
     which is the dangerous direction (under-counting means compaction fires
     too late, and the first symptom is a provider 400).

  2. **Calibration against reality.** Every response reports the real prompt
     token count. We fold estimate-vs-actual into an EWMA correction factor per
     ``(provider, model)`` and scale later estimates by it. See
     ``calibration.py``. This costs nothing and grounds the estimate in the
     provider's own tokenizer within a few calls.

For OpenAI we can do better than ratios — ``tiktoken`` is already a hard
dependency and runs locally in microseconds, so we tokenize exactly and the
calibration factor simply converges to ~1.0.

The :class:`TokenCounter` protocol is deliberately shaped so a remote counter
could be dropped in later behind the same interface without touching callers.
We just don't ship one.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from ...message import ContentType, Message, MessageRole
from ..shape import RequestShape, TokenBreakdown
from .calibration import calibrator

logger = logging.getLogger(__name__)


# --- content-type ratios ------------------------------------------------------
#
# Chars per token, by what the text actually looks like. Derived by tokenizing
# representative payloads from this codebase (ad-platform tool schemas, Google
# Ads query results, prose system prompts) and dividing. They are starting
# points: the calibrator corrects whatever residual error remains, so the goal
# here is to be close enough that the factor stays inside its clamp, not to be
# exact.

#: Natural-language prose — system prompts, assistant text, user turns.
_CHARS_PER_TOKEN_PROSE = 3.9

#: JSON — tool schemas, tool call arguments, tool results. Denser in tokens
#: (i.e. fewer chars per token) because structural punctuation and quoted keys
#: fragment. This is the ratio the old flat-4 estimator got most wrong.
_CHARS_PER_TOKEN_JSON = 3.0

#: Per-message envelope: role, delimiters, and the provider's own framing.
#: Anthropic and OpenAI are both in this neighbourhood.
_MESSAGE_OVERHEAD_TOKENS = 4

#: Per-tool envelope on top of the serialized schema — the name/description
#: wrapper each provider adds around the raw JSON Schema.
_TOOL_OVERHEAD_TOKENS = 8

#: Flat costs for non-text blocks. Real image cost varies with resolution and
#: provider (Anthropic's high-res tier can reach ~4.8K for one image), but the
#: estimator only needs to be in the right order of magnitude — an image-heavy
#: turn will pull the calibration factor toward truth on the next call.
_IMAGE_TOKENS = 1_600
_DOCUMENT_TOKENS = 3_000
_VIDEO_TOKENS = 5_000


def _is_json_like(text: str) -> bool:
    """Cheap structural check: does this look like serialized JSON?

    Only the first and last non-space characters are inspected. A full parse
    would be correct but this runs on every message of every turn, and the
    penalty for a wrong answer is a ~20% ratio error on one message — well
    inside what calibration absorbs.
    """
    if not text:
        return False
    head = text[:1]
    if head not in "{[":
        # Tool results are often JSON with a short prose preamble; catch the
        # common "…: {" / "…: [" shape without scanning the whole string.
        prefix = text[:200]
        return "{" in prefix or "[" in prefix
    return True


def _chars_to_tokens(text: str, json_like: Optional[bool] = None) -> int:
    if not text:
        return 0
    if json_like is None:
        json_like = _is_json_like(text)
    ratio = _CHARS_PER_TOKEN_JSON if json_like else _CHARS_PER_TOKEN_PROSE
    return int(len(text) / ratio) + 1


class TokenCounter:
    """Protocol for anything that can size a :class:`RequestShape`.

    Implementations must be safe to call on the hot path — no I/O, no locks
    held across a call. A future remote counter would satisfy this by serving
    from a warm cache and falling back to a local estimate on a miss.
    """

    provider: Optional[str] = None

    def count_text(self, text: str, json_like: Optional[bool] = None) -> int:
        raise NotImplementedError

    def count_messages(self, messages: List[Message]) -> int:
        raise NotImplementedError

    def count_tools(self, tools: List[Dict[str, Any]]) -> int:
        raise NotImplementedError

    def count_system(
        self, system: Optional[Union[str, List[Dict[str, Any]]]]
    ) -> int:
        raise NotImplementedError

    def breakdown(self, shape: RequestShape) -> TokenBreakdown:
        raise NotImplementedError


class LocalTokenCounter(TokenCounter):
    """Ratio-based estimator with calibration. The default for every provider.

    Subclassed by :class:`TiktokenCounter` for OpenAI, which overrides only
    ``count_text``; every structural rule (message envelopes, tool wrappers,
    block costs) is shared.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider
        self.model = model

    # -- leaf counting ---------------------------------------------------

    def count_text(self, text: str, json_like: Optional[bool] = None) -> int:
        return _chars_to_tokens(text, json_like)

    def _count_content(
        self, content: Union[str, List[Any], None], json_like: Optional[bool] = None
    ) -> int:
        """Count a message body, which may be a string or a list of blocks."""
        if content is None:
            return 0
        if isinstance(content, str):
            return self.count_text(content, json_like)

        total = 0
        for block in content:
            total += self._count_block(block)
        return total

    def _count_block(self, block: Any) -> int:
        """Count one content block.

        Blocks arrive either as the dataclasses in ``core.message`` or as raw
        dicts (the Django adapter builds some of them by hand). Handle both by
        reading ``type`` off whichever shape it is.
        """
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")

        if block_type == ContentType.TEXT.value:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text", "")
            return self.count_text(text or "")

        if block_type == ContentType.IMAGE_URL.value or block_type == "image":
            return _IMAGE_TOKENS
        if block_type == ContentType.DOCUMENT.value:
            return _DOCUMENT_TOKENS
        if block_type == ContentType.VIDEO_URL.value:
            return _VIDEO_TOKENS

        # Unknown block type: fall back to sizing its serialized form rather
        # than returning zero. Silently counting an unrecognized block as free
        # is how a request sails past the threshold and 400s.
        try:
            return self.count_text(json.dumps(block, default=str), json_like=True)
        except Exception:  # noqa: BLE001 — never let sizing break a request
            return self.count_text(str(block), json_like=True)

    # -- tier counting ---------------------------------------------------

    def count_messages(
        self, messages: List[Message], include_system: bool = True
    ) -> int:
        total = 0
        for message in messages:
            if not include_system and message.role == MessageRole.SYSTEM:
                continue
            total += _MESSAGE_OVERHEAD_TOKENS

            # Tool results are JSON far more often than not, and they are the
            # tier the flat-4 estimator under-counted worst. Bias the ratio for
            # them rather than re-sniffing every payload.
            json_hint = True if message.role == MessageRole.TOOL else None
            total += self._count_content(message.content, json_hint)

            if message.tool_calls:
                # tool_calls never appear in `content`, so a counter that only
                # walks `content` misses every argument payload the assistant
                # emitted — on a parallel-dispatch turn that is most of the
                # message.
                try:
                    serialized = json.dumps(message.tool_calls, default=str)
                except Exception:  # noqa: BLE001
                    serialized = str(message.tool_calls)
                total += self.count_text(serialized, json_like=True)

            if message.name:
                total += self.count_text(message.name)

        return total

    def count_tools(self, tools: List[Dict[str, Any]]) -> int:
        if not tools:
            return 0
        total = 0
        for tool in tools:
            total += _TOOL_OVERHEAD_TOKENS
            try:
                serialized = json.dumps(tool, default=str)
            except Exception:  # noqa: BLE001
                serialized = str(tool)
            total += self.count_text(serialized, json_like=True)
        return total

    def count_system(
        self, system: Optional[Union[str, List[Dict[str, Any]]]]
    ) -> int:
        if not system:
            return 0
        if isinstance(system, str):
            return self.count_text(system)
        # Anthropic-style list of system blocks.
        total = 0
        for block in system:
            total += self._count_block(block)
        return total

    # -- the public entry point ------------------------------------------

    def breakdown(self, shape: RequestShape) -> TokenBreakdown:
        """Per-tier estimate for ``shape``, with the correction factor applied.

        The factor scales all three tiers uniformly. That is deliberate: the
        residual error it corrects is tokenizer-level (how this provider splits
        text), which applies equally to a schema and a paragraph. Scaling only
        one tier would distort the floor-vs-conversation ratio that policy
        reads.
        """
        raw_system, raw_tools, raw_messages = self._raw_tiers(shape)

        # The counter's own identity wins over the shape's. A counter picked
        # its tokenizer from the provider it was constructed for, so reading a
        # *different* provider's correction factor would apply e.g. Anthropic's
        # residual error on top of exact tiktoken output. The shape is only a
        # fallback for counters built without an explicit provider.
        provider = self.provider or shape.provider
        model = self.model or shape.model
        state = calibrator.state_for(provider, model)
        factor = state.factor

        return TokenBreakdown(
            system=int(raw_system * factor),
            tools=int(raw_tools * factor),
            messages=int(raw_messages * factor),
            calibration_factor=factor,
            calibrated=state.is_grounded,
        )

    def _raw_tiers(self, shape: RequestShape):
        """Uncorrected ``(system, tools, messages)`` counts.

        This package carries the system prompt as a SYSTEM-role entry inside
        ``messages``; the provider client splits it out at send time (see
        ``AnthropicClient._prepare_messages``). Callers that keep it separate
        pass ``shape.system`` instead.

        Attribute it to the system tier from wherever it actually lives, and
        never from both — double-counting the largest static block would
        inflate the floor and produce spurious ``FLOOR_EXCEEDED`` verdicts on
        requests compaction could in fact have rescued.
        """
        system = self.count_system(shape.system)
        system += self.count_messages(
            [m for m in shape.messages if m.role == MessageRole.SYSTEM]
        )
        tools = self.count_tools(shape.tools)
        messages = self.count_messages(shape.messages, include_system=False)
        return system, tools, messages

    def raw_total(self, shape: RequestShape) -> int:
        """Uncorrected total — what to feed back into the calibrator.

        Calibration compares its own *uncorrected* output against the
        provider's number. Passing a corrected estimate back would make the
        ratio converge to 1.0 no matter how wrong the underlying estimate is,
        silently disabling the loop.
        """
        return sum(self._raw_tiers(shape))


class TiktokenCounter(LocalTokenCounter):
    """Exact local tokenization for OpenAI models via ``tiktoken``.

    ``tiktoken`` is a hard dependency of this package and runs locally, so
    there is no reason to approximate on OpenAI. Only ``count_text`` changes;
    the structural overheads are inherited.

    The encoding is resolved once per counter. If resolution fails (an unknown
    model name, a stripped-down install), we fall back to the ratio estimator
    rather than raising — sizing must never be the thing that breaks a request.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        super().__init__(provider=provider, model=model)
        self._encoding = self._resolve_encoding(model)

    @staticmethod
    def _resolve_encoding(model: Optional[str]):
        try:
            import tiktoken
        except ImportError:
            logger.debug("[TOKENS] tiktoken unavailable; using ratio estimator")
            return None

        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:  # noqa: BLE001 — unknown model name is expected
                pass
        try:
            # Current OpenAI models (gpt-4o and later) all use o200k_base.
            return tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001
            logger.debug("[TOKENS] could not load o200k_base; using ratio estimator")
            return None

    def count_text(self, text: str, json_like: Optional[bool] = None) -> int:
        if not text:
            return 0
        if self._encoding is None:
            return super().count_text(text, json_like)
        try:
            return len(self._encoding.encode(text, disallowed_special=()))
        except Exception:  # noqa: BLE001
            return super().count_text(text, json_like)


#: Providers that route through an OpenAI-compatible tokenizer. Not every
#: OpenAI-compatible *API* uses the OpenAI *tokenizer* — Groq and Together
#: front Llama-family models with their own — so this list is narrow on
#: purpose. Anything not named here gets ratios plus calibration, which is
#: the safe default.
_TIKTOKEN_PROVIDERS = frozenset({"openai", "azure", "azure_openai"})


def get_counter(
    provider: Optional[str] = None, model: Optional[str] = None
) -> TokenCounter:
    """Return the counter for a provider/model pair.

    Cheap to call — counters hold no connections and only ``TiktokenCounter``
    does any setup work, which is itself cached inside ``tiktoken``.
    """
    normalized = (provider or "").strip().lower()
    if normalized in _TIKTOKEN_PROVIDERS:
        return TiktokenCounter(provider=normalized, model=model)
    return LocalTokenCounter(provider=normalized or None, model=model)
