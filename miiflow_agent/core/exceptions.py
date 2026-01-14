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


class RateLimitError(MiiflowLLMError):
    """Rate limit exceeded."""
    
    def __init__(self, message: str, provider: str, retry_after: Optional[float] = None, **kwargs):
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
