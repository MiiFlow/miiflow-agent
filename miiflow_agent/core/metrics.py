"""Metrics and observability for LLM operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class MetricType(Enum):
    """Types of metrics collected."""

    TOKEN_USAGE = "token_usage"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    REQUEST_COUNT = "request_count"


@dataclass
class TokenCount:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Provider prompt-cache split of prompt_tokens (Anthropic: ~0.1x for
    # reads, ~1.25x for writes). Zero when the provider reports none —
    # observability for "is prompt caching actually hitting".
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Reasoning/thinking tokens billed as output but not present in the
    # visible completion (OpenAI `completion_tokens_details.reasoning_tokens`,
    # Gemini `usageMetadata.thoughtsTokenCount`). Already included in
    # `completion_tokens` — carried separately so cost attribution can tell
    # "the answer was long" from "the model thought hard".
    reasoning_tokens: int = 0

    @property
    def billed_prompt_tokens(self) -> int:
        """Total tokens the provider processed for the prompt.

        This is the number to reconcile a local estimate against. It is NOT
        always ``prompt_tokens``: on Anthropic, ``input_tokens`` reports only
        the *uncached remainder*, so on a cache hit it is near-zero while the
        real prompt was tens of thousands of tokens. Feeding that to the
        calibrator would make the estimator conclude it over-counts by 10x.

        Providers whose ``prompt_tokens`` is already inclusive report zero for
        the cache fields, so the sum is correct for them too — see
        ``AnthropicClient``, which folds the cache split into ``prompt_tokens``
        before constructing this, and reports the split for observability only.
        """
        return self.prompt_tokens or (
            self.cache_read_tokens + self.cache_write_tokens
        )

    @property
    def uncached_prompt_tokens(self) -> int:
        """Prompt tokens billed at the full input rate.

        Invariant (normalized at construction, not here): ``prompt_tokens``
        is INCLUSIVE of ``cache_read_tokens`` + ``cache_write_tokens`` for
        every provider. AnthropicClient folds the split in before
        constructing this (see ``billed_prompt_tokens``); OpenAI/Gemini
        report cache reads as a subset of ``prompt_tokens`` natively and
        never report cache writes. The ``max(0, ...)`` is a backstop for a
        future adapter that breaks the invariant — consumers that price
        tokens must re-check it themselves (undercharging is worse than
        degrading to cache-blind pricing).
        """
        return max(
            0,
            self.prompt_tokens - self.cache_read_tokens - self.cache_write_tokens,
        )

    def __add__(self, other: "TokenCount") -> "TokenCount":
        return TokenCount(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens
            + getattr(other, "cache_read_tokens", 0),
            cache_write_tokens=self.cache_write_tokens
            + getattr(other, "cache_write_tokens", 0),
            reasoning_tokens=self.reasoning_tokens
            + getattr(other, "reasoning_tokens", 0),
        )

    # -- provider wire-format adapters -----------------------------------
    #
    # These live on the dataclass rather than in each provider module so the
    # non-streaming client and the stream normalizer share one implementation.
    # `core` must not import from `providers`, and duplicating the detail-field
    # handling in both places is how the streaming path ended up reporting
    # different numbers from the non-streaming one.

    @classmethod
    def from_openai_usage(cls, usage_data: Any) -> "TokenCount":
        """Map either OpenAI usage shape onto a ``TokenCount``.

        Two field namings are in play and both reach this method:

        * **Chat Completions** — ``prompt_tokens`` / ``completion_tokens``,
          details under ``prompt_tokens_details`` / ``completion_tokens_details``.
        * **Responses API** — ``input_tokens`` / ``output_tokens``, details
          under ``input_tokens_details`` / ``output_tokens_details``.

        ``cached_tokens`` is a *subset* of the prompt total here (unlike
        Anthropic, where the cache fields sit alongside the uncached
        remainder), so it is recorded for observability and the prompt total
        needs no adjustment.

        Reads defensively: usage is absent on most streaming chunks, and the
        detail objects only appear on models that populate them.
        """
        if not usage_data:
            return cls()

        prompt = _as_int(
            _first_present(usage_data, "prompt_tokens", "input_tokens")
        )
        completion = _as_int(
            _first_present(usage_data, "completion_tokens", "output_tokens")
        )
        total = _as_int(getattr(usage_data, "total_tokens", 0)) or (
            prompt + completion
        )

        prompt_details = getattr(usage_data, "prompt_tokens_details", None) or getattr(
            usage_data, "input_tokens_details", None
        )
        completion_details = getattr(
            usage_data, "completion_tokens_details", None
        ) or getattr(usage_data, "output_tokens_details", None)

        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cache_read_tokens=(
                _as_int(getattr(prompt_details, "cached_tokens", 0))
                if prompt_details
                else 0
            ),
            cache_write_tokens=(
                _as_int(getattr(prompt_details, "cache_write_tokens", 0))
                if prompt_details
                else 0
            ),
            reasoning_tokens=(
                _as_int(getattr(completion_details, "reasoning_tokens", 0))
                if completion_details
                else 0
            ),
        )

    @classmethod
    def from_gemini_usage(cls, usage_meta: Any) -> "TokenCount":
        """Map Gemini's ``usageMetadata`` onto a ``TokenCount``.

        Accepts both the REST dict (``promptTokenCount``) and the protobuf SDK
        object (``prompt_token_count``) — the two paths differ only in case
        convention, so one reader covers both.

        Two fields matter beyond the headline counts:

        * ``cachedContentTokenCount`` — counted *inside* ``promptTokenCount``,
          so observability only.
        * ``thoughtsTokenCount`` — thinking tokens. Billed as output and
          included in ``totalTokenCount``, but NOT in ``candidatesTokenCount``.
          Omitting it makes ``prompt + candidates`` fail to reconcile against
          the total on every thinking model, which reads as a broken estimator
          rather than an unmodelled field.
        """
        if not usage_meta:
            return cls()

        def field_value(camel: str, snake: str) -> int:
            if isinstance(usage_meta, dict):
                return _as_int(usage_meta.get(camel, usage_meta.get(snake)))
            return _as_int(
                _first_present(usage_meta, snake, camel)
            )

        prompt = field_value("promptTokenCount", "prompt_token_count")
        prompt += field_value("toolUsePromptTokenCount", "tool_use_prompt_token_count")
        candidates = field_value("candidatesTokenCount", "candidates_token_count")
        thoughts = field_value("thoughtsTokenCount", "thoughts_token_count")
        cached = field_value("cachedContentTokenCount", "cached_content_token_count")
        total = field_value("totalTokenCount", "total_token_count") or (
            prompt + candidates + thoughts
        )

        return cls(
            prompt_tokens=prompt,
            # Thinking tokens are billed as output but excluded from
            # candidatesTokenCount; add them so completion usage doesn't
            # under-report on every thinking model.
            completion_tokens=candidates + thoughts,
            total_tokens=total,
            cache_read_tokens=cached,
            reasoning_tokens=thoughts,
        )


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _first_present(obj: Any, *names: str) -> Any:
    """First attribute in ``names`` that exists and is not None.

    Not ``getattr(obj, a, None) or getattr(obj, b, None)`` — a legitimate zero
    would fall through to the second name and read the wrong field.
    """
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return 0


@dataclass
class UsageData:
    """Usage metrics for a single request."""

    provider: str
    model: str
    operation: str
    tokens: TokenCount
    latency_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates LLM metrics."""

    def __init__(self):
        self._usage_history: List[UsageData] = []
        self._aggregated_metrics: Dict[str, Any] = {}

    def record_usage(self, usage: UsageData) -> None:
        """Record usage data for a request."""
        self._usage_history.append(usage)
        self._update_aggregates(usage)

    def _update_aggregates(self, usage: UsageData) -> None:
        """Update aggregated metrics."""
        key = f"{usage.provider}:{usage.model}"

        if key not in self._aggregated_metrics:
            self._aggregated_metrics[key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_tokens": TokenCount(),
                "total_latency_ms": 0.0,
                "error_count": 0,
            }

        metrics = self._aggregated_metrics[key]
        metrics["total_requests"] += 1

        if usage.success:
            metrics["successful_requests"] += 1
        else:
            metrics["error_count"] += 1

        metrics["total_tokens"] += usage.tokens
        metrics["total_latency_ms"] += usage.latency_ms

    def get_metrics(self, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated metrics, optionally filtered by provider/model."""
        if provider or model:
            filtered = {}
            for key, metrics in self._aggregated_metrics.items():
                p, m = key.split(":", 1)
                if (not provider or p == provider) and (not model or m == model):
                    filtered[key] = metrics
            return filtered

        return self._aggregated_metrics.copy()

    def get_usage_history(self, limit: Optional[int] = None) -> List[UsageData]:
        """Get usage history, optionally limited."""
        if limit:
            return self._usage_history[-limit:]
        return self._usage_history.copy()

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._usage_history.clear()
        self._aggregated_metrics.clear()


@dataclass
class LLMMetrics:
    """Metrics snapshot for LLM operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: TokenCount = field(default_factory=TokenCount)
    average_latency_ms: float = 0.0
    error_rate: float = 0.0

    @classmethod
    def from_collector(cls, collector: MetricsCollector) -> "LLMMetrics":
        """Create metrics snapshot from collector."""
        all_metrics = collector.get_metrics()

        total_requests = sum(m["total_requests"] for m in all_metrics.values())
        successful_requests = sum(m["successful_requests"] for m in all_metrics.values())
        failed_requests = sum(m["error_count"] for m in all_metrics.values())

        total_tokens = TokenCount()
        total_latency = 0.0

        for metrics in all_metrics.values():
            total_tokens += metrics["total_tokens"]
            total_latency += metrics["total_latency_ms"]

        return cls(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_tokens=total_tokens,
            average_latency_ms=total_latency / max(total_requests, 1),
            error_rate=failed_requests / max(total_requests, 1),
        )
