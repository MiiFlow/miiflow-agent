"""Base model configuration types."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ParameterType(str, Enum):
    """Type of parameter for LLM configuration."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"


@dataclass
class ParameterConfig:
    """Configuration for an LLM parameter.

    This defines a parameter that can be passed to an LLM API, including
    its constraints and which models support it.
    """

    field_name: str  # API parameter name (e.g., "max_completion_tokens")
    display_name: str  # UI display name
    description: str = ""
    parameter_type: ParameterType = ParameterType.STRING
    required: bool = False
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[Union[float, Dict[str, float]]] = None  # Can be model-specific
    step: Optional[float] = None
    options: Optional[List[str]] = None
    supported_models: Optional[List[str]] = None  # Only these models support this param
    unsupported_models: Optional[List[str]] = None  # These models don't support this param


@dataclass
class ModelConfig:
    """Configuration for an LLM model.

    This defines a model's capabilities, parameter requirements, and pricing.
    """

    model_identifier: str  # API model name (e.g., "gpt-5.5")
    name: str  # Display name
    description: str = ""

    # Capabilities
    support_images: bool = False
    support_files: bool = False
    support_streaming: bool = True
    supports_json_mode: bool = False
    supports_tool_call: bool = False
    supports_structured_outputs: bool = False  # Native JSON schema support (e.g., Anthropic)
    reasoning: bool = False

    # Token limits
    maximum_context_tokens: int = 0
    maximum_output_tokens: int = 0

    # Parameter behavior
    token_param_name: str = "max_tokens"  # API param name for max tokens
    supports_temperature: bool = True

    # Pricing hints (per million tokens in USD)
    # These are approximate values - actual pricing may vary by provider/region
    input_cost_hint: float = 0.0
    output_cost_hint: float = 0.0
    # Cache pricing (per million tokens in USD). 0.0 = undeclared: consumers
    # bill cache tokens at input_cost_hint, i.e. the pre-cache-aware behavior.
    # Declared as explicit $/M (not multipliers of input_cost_hint) because
    # OpenAI publishes cached-input prices per model, not a uniform ratio.
    # NOTE: rates are per-model, but Anthropic's cache-WRITE rate is per
    # request TTL (1.25x input for the 5-minute default, 2x for 1h). This
    # codebase only ever requests the default TTL (see AnthropicClient.
    # _apply_prompt_caching, which never sets a "ttl" key); if a 1h TTL is
    # ever requested, this schema undercharges writes and needs a per-request
    # dimension instead.
    cache_read_cost_hint: float = 0.0
    cache_write_cost_hint: float = 0.0
