"""Core components for Miiflow LLM."""

from .client import LLMClient, ModelClient, ChatResponse, StreamChunk
from .message import Message, MessageRole, ContentBlock, TextBlock, ImageBlock
from .metrics import LLMMetrics, TokenCount, UsageData, MetricsCollector
from .streaming import (
    UnifiedStreamingClient,
    IncrementalParser,
    EnhancedStreamChunk
)
from .exceptions import (
    MiiflowLLMError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelError,
    TimeoutError,
    ParsingError,
    ToolError,
)
from .tools import (
    # Production-grade modular tools
    FunctionTool,
    ToolRegistry,
    ToolResult,
    FunctionType,
    ParameterType,
    PreparedCall,
    ToolPreparationError,
    ToolExecutionError,
    tool,
    detect_function_type,
    get_fun_schema
)
from .schema_normalizer import normalize_json_schema, SchemaMode
from .data_reference import put_render_data, get_render_data
from .checkpoint import (
    CHECKPOINT_VERSION,
    AgentFrame,
    Checkpoint,
    DispatchLedgerEntry,
    EstablishedFact,
    PendingApprovedAction,
    PendingInterrupt,
    ResumeCommand,
    stable_json_hash,
)
from .interrupt import (
    ClarificationDecision,
    GraphInterrupt,
    INTERRUPT_KIND_CLARIFICATION,
    INTERRUPT_KIND_PLAN_APPROVAL,
    INTERRUPT_KIND_TOOL_APPROVAL,
    MAX_CLARIFICATION_INTERRUPTS_PER_TURN,
    decide_clarification,
    mint_interrupt_id,
    partition_questions_by_facts,
    question_key,
    render_established_facts_block,
)
from .agent import (
    # Core agent architecture - Stateless framework
    Agent,
    RunContext,
    RunResult,
    AgentType,
)

# Observability exports (optional - requires explicit setup)
try:
    from .observability import (
        ObservabilityConfig,
        TraceContext,
        get_current_trace_context,
    )
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False


def setup_tracing(phoenix_endpoint: str = None, force: bool = False) -> bool:
    """
    Explicitly enable Phoenix tracing for observability.

    This function must be called explicitly to enable tracing.
    Tracing is NOT enabled automatically on import.

    Args:
        phoenix_endpoint: Optional Phoenix collector endpoint URL.
                         If not provided, uses PHOENIX_ENDPOINT env var.
        force: If True, enables tracing even if already initialized.

    Returns:
        True if tracing was enabled, False otherwise.

    Example:
        from miiflow_agent.core import setup_tracing

        # Enable with default endpoint from env
        setup_tracing()

        # Or specify endpoint explicitly
        setup_tracing(phoenix_endpoint="http://localhost:6006")
    """
    if not _OBSERVABILITY_AVAILABLE:
        return False

    from .observability.auto_instrumentation import enable_phoenix_tracing
    from .observability.config import ObservabilityConfig

    config = ObservabilityConfig.from_env()
    if phoenix_endpoint:
        config.phoenix_endpoint = phoenix_endpoint

    if config.phoenix_enabled or force:
        enable_phoenix_tracing(config.phoenix_endpoint)
        return True
    return False


__all__ = [
    "LLMClient",
    "ModelClient",
    "ChatResponse",
    "StreamChunk",
    "Message",
    "MessageRole",
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    "LLMMetrics",
    "TokenCount",
    "UsageData",
    "MetricsCollector",
    "UnifiedStreamingClient",
    "IncrementalParser",
    "EnhancedStreamChunk",
    "MiiflowLLMError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelError",
    "TimeoutError",
    "ParsingError",
    "ToolError",

    "FunctionTool",
    "ToolRegistry",
    "ToolResult",
    "FunctionType",
    "ParameterType",
    "PreparedCall",
    "ToolPreparationError",
    "ToolExecutionError",
    "tool",
    "detect_function_type",
    "get_fun_schema",

    # Core agent architecture - Stateless framework
    "Agent",
    "RunContext",
    "RunResult",
    "AgentType",

    # Schema normalization
    "normalize_json_schema",
    "SchemaMode",

    # Data-reference cache for render tools
    "put_render_data",
    "get_render_data",

    # Durable run checkpoint (pause/resume + multi-agent context spine)
    "Checkpoint",
    "CHECKPOINT_VERSION",
    "EstablishedFact",
    "PendingInterrupt",
    "PendingApprovedAction",
    "DispatchLedgerEntry",
    "AgentFrame",
    "ResumeCommand",
    "stable_json_hash",

    # Unified interrupt primitive + established-facts logic
    "GraphInterrupt",
    "INTERRUPT_KIND_CLARIFICATION",
    "INTERRUPT_KIND_TOOL_APPROVAL",
    "INTERRUPT_KIND_PLAN_APPROVAL",
    "MAX_CLARIFICATION_INTERRUPTS_PER_TURN",
    "mint_interrupt_id",
    "question_key",
    "partition_questions_by_facts",
    "decide_clarification",
    "ClarificationDecision",
    "render_established_facts_block",

    # Observability (optional)
    "ObservabilityConfig",
    "TraceContext",
    "get_current_trace_context",
    "setup_tracing",
]
