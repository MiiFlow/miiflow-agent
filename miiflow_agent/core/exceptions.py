"""Exception classes for Miiflow LLM."""

from enum import Enum
from typing import Any, Optional


class ErrorType(Enum):
    """Standardized error types across all providers."""
    
    AUTHENTICATION = "authentication_error"
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"
    MODEL_ERROR = "model_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout_error"
    PARSING_ERROR = "parsing_error"
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CONTENT_FILTERED = "content_filtered"
    TOKEN_LIMIT = "token_limit_exceeded"
    STREAMING_ERROR = "streaming_error"


class MiiflowLLMError(Exception):
    """Base exception for all Miiflow LLM errors."""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
        retry_after: Optional[float] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.provider = provider
        self.model = model
        self.original_error = original_error
        self.retry_after = retry_after
        super().__init__(message)


class ProviderError(MiiflowLLMError):
    """Error from LLM provider API."""
    
    def __init__(self, message: str, provider: str, **kwargs):
        super().__init__(message, ErrorType.MODEL_ERROR, provider=provider, **kwargs)


class AuthenticationError(MiiflowLLMError):
    """Authentication failed with provider."""
    
    def __init__(self, message: str, provider: str, **kwargs):
        super().__init__(message, ErrorType.AUTHENTICATION, provider=provider, **kwargs)


def retry_after_seconds(error) -> Optional[float]:
    """Extract a Retry-After delay in seconds from a provider SDK exception.

    Provider SDK errors (OpenAI, Anthropic, Groq, xAI, ...) carry the httpx
    response on ``error.response``; httpx header lookup is case-insensitive.
    The header must be read with ``headers.get("retry-after")`` — an earlier
    version used ``getattr(headers, "retry-after", None)``, which looks up an
    *attribute* of that name on the headers object and therefore always
    returned None, silently disabling rate-limit-aware backoff everywhere.

    HTTP-date values (the other legal Retry-After form) are ignored rather
    than parsed: the providers this package calls all send delta-seconds.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    value = get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


class RateLimitError(MiiflowLLMError):
    """Rate limit exceeded."""

    def __init__(self, message: str, provider: str, retry_after: Optional[float] = None, **kwargs):
        # Fall back to the wrapped SDK error's Retry-After header so every
        # raise site gets a populated delay without having to remember to
        # extract it (several never did).
        if retry_after is None:
            retry_after = retry_after_seconds(kwargs.get("original_error"))
        super().__init__(
            message,
            ErrorType.RATE_LIMITED,
            provider=provider,
            retry_after=retry_after,
            **kwargs
        )


class ModelError(MiiflowLLMError):
    """Model-specific error (context length, etc.)."""
    
    def __init__(self, message: str, model: str, **kwargs):
        super().__init__(message, ErrorType.MODEL_ERROR, model=model, **kwargs)


class TimeoutError(MiiflowLLMError):
    """Request timeout."""
    
    def __init__(self, message: str, timeout_duration: float, **kwargs):
        super().__init__(
            f"{message} (timeout: {timeout_duration}s)",
            ErrorType.TIMEOUT,
            **kwargs
        )


class ParsingError(MiiflowLLMError):
    """Error parsing structured output."""
    
    def __init__(self, message: str, raw_content: str, **kwargs):
        self.raw_content = raw_content
        super().__init__(message, ErrorType.PARSING_ERROR, **kwargs)


class ToolError(MiiflowLLMError):
    """Error executing tool."""
    
    def __init__(self, message: str, tool_name: str, **kwargs):
        self.tool_name = tool_name
        super().__init__(message, ErrorType.TOOL_ERROR, **kwargs)


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------

#: HTTP statuses worth retrying: transient overload/gateway failures plus
#: request-timeout and lock-conflict. Everything else 4xx is deterministic —
#: retrying a 401 or a 400 just replays the same failure with added latency.
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}

_RETRYABLE_ERROR_TYPES = {
    ErrorType.RATE_LIMITED,
    ErrorType.TIMEOUT,
    ErrorType.NETWORK_ERROR,
    ErrorType.SERVICE_UNAVAILABLE,
}

#: Types that are deterministic regardless of what they wrap. MODEL_ERROR is
#: deliberately NOT here: ProviderError (the catch-all wrapper every client
#: uses for unmapped SDK errors, including 5xx/overloaded/connection failures)
#: reports MODEL_ERROR too, so that type alone proves nothing — those fall
#: through to the status-code / class-name checks below.
_NON_RETRYABLE_ERROR_TYPES = {
    ErrorType.AUTHENTICATION,
    ErrorType.INVALID_REQUEST,
    ErrorType.VALIDATION_ERROR,
    ErrorType.CONFIGURATION_ERROR,
    ErrorType.QUOTA_EXCEEDED,
    ErrorType.CONTENT_FILTERED,
    ErrorType.TOKEN_LIMIT,
    ErrorType.PARSING_ERROR,
    ErrorType.TOOL_ERROR,
}


def _status_code_of(error) -> Optional[int]:
    for candidate in (error, getattr(error, "original_error", None)):
        if candidate is None:
            continue
        status = getattr(candidate, "status_code", None)
        if status is None:
            response = getattr(candidate, "response", None)
            status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def is_retryable_error(error) -> bool:
    """Whether ``error`` is plausibly transient and worth retrying.

    Used by the LLM transport retry layers. Deliberately answers False for
    anything deterministic (auth, bad request, context overflow, quota):
    retrying those burns seconds-to-minutes replaying a guaranteed failure,
    which is what the old untargeted ``@retry`` decorators did.
    """
    if error is None:
        return False

    if isinstance(error, MiiflowLLMError):
        if error.error_type in _RETRYABLE_ERROR_TYPES:
            return True
        if error.error_type in _NON_RETRYABLE_ERROR_TYPES:
            return False
        # MODEL_ERROR and other ambiguous wrappers: decide by the wrapped
        # SDK error's status, then by its class name (APIConnectionError,
        # ConnectTimeout, ... carry no status but are transient by nature).
        status = _status_code_of(error)
        if status is not None:
            return status in _RETRYABLE_STATUS_CODES
        original = getattr(error, "original_error", None)
        if original is not None:
            name = type(original).__name__
            return "Connection" in name or "Timeout" in name or "Overloaded" in name
        return False

    import asyncio as _asyncio

    if isinstance(error, (_asyncio.TimeoutError, ConnectionError, OSError)):
        return True

    try:
        import httpx

        if isinstance(error, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover - httpx ships with every SDK we use
        pass

    status = _status_code_of(error)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES

    # Unmapped SDK errors: connection/timeout classes are transient by
    # nature and every SDK names them as such (APIConnectionError,
    # ConnectTimeout, StreamTimeout, ...).
    name = type(error).__name__
    return "Connection" in name or "Timeout" in name


def retry_delay_seconds(
    error,
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 30.0,
) -> float:
    """Delay before retry ``attempt`` (1-based): capped exponential backoff
    with +0..25% jitter. A server-provided Retry-After acts as a floor — the
    server's number outranks the locally computed one.
    """
    import random

    delay = min(cap, base * (2 ** (attempt - 1)))
    delay *= 1 + random.random() * 0.25
    server_says = getattr(error, "retry_after", None)
    if server_says:
        try:
            delay = max(delay, float(server_says))
        except (TypeError, ValueError):
            pass
    return delay
