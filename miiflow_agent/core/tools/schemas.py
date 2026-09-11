"""Data classes for tool schemas and results."""

from typing import Any, Dict, List, Optional, Generic
from dataclasses import dataclass, field

from .types import ResultType, ToolType, ParameterType

# Internal variable types that are NOT JSON Schema types, mapped to the
# JSON Schema type (and format) that carries them on the wire. `VariableType`
# on the server side is wider than `ParameterType` here — a tool parameter
# declared `file`, `video`, `timestamp` or `enum` only ever reaches us nested
# inside `items` / `properties`, as a raw dict, because the top-level coercion
# in `tool_config_converter` falls back to STRING for names this enum lacks.
_CUSTOM_TYPE_TO_JSON_SCHEMA: Dict[str, Dict[str, str]] = {
    "media": {"type": "string", "format": "uri"},
    "file": {"type": "string", "format": "uri"},
    "video": {"type": "string", "format": "uri"},
    "text": {"type": "string"},
    "timestamp": {"type": "string", "format": "date-time"},
    "enum": {"type": "string"},
}

# Keywords whose values are themselves schemas (or maps/lists of schemas), so
# the custom-type mapping has to recurse through them.
_SCHEMA_VALUED_KEYWORDS = ("items", "additionalProperties", "not", "if", "then", "else")
_SCHEMA_MAP_KEYWORDS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_LIST_KEYWORDS = ("anyOf", "allOf", "oneOf", "prefixItems")


def map_custom_types_to_json_schema(node: Any) -> Any:
    """Rewrite internal variable types into valid JSON Schema, at any depth.

    This is the "handled upstream" half of the contract that
    `miiflow_agent/core/schema_normalizer.py` documents: that module owns
    provider-specific shaping and explicitly does NOT map custom types, so the
    mapping has to be complete by the time a schema leaves here.

    `to_json_schema_property` has always mapped `media` / `text` for the
    parameter's OWN type, but copied `items` and `properties` through verbatim.
    A nested `{"type": "file"}` therefore reached the model provider unchanged,
    and the Anthropic API rejects the whole request with
    `tools.<n>.custom.input_schema: JSON schema is invalid. It must match JSON
    Schema draft 2020-12` — which takes out EVERY tool call the agent makes,
    not just the offending one. The in-house Slack, Gmail, Outlook and Adlyse
    email tools all declare `files: {type: array, items: {type: file}}`, so any
    agent granted one of them could not call a single tool.
    """
    if isinstance(node, list):
        return [map_custom_types_to_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: Dict[str, Any] = dict(node)

    declared = out.get("type")
    if isinstance(declared, str) and declared in _CUSTOM_TYPE_TO_JSON_SCHEMA:
        mapped = _CUSTOM_TYPE_TO_JSON_SCHEMA[declared]
        out["type"] = mapped["type"]
        # Never overwrite a format the tool author set deliberately.
        if "format" in mapped and "format" not in out:
            out["format"] = mapped["format"]
    elif isinstance(declared, list):
        out["type"] = [
            _CUSTOM_TYPE_TO_JSON_SCHEMA[t]["type"]
            if isinstance(t, str) and t in _CUSTOM_TYPE_TO_JSON_SCHEMA
            else t
            for t in declared
        ]

    for keyword in _SCHEMA_VALUED_KEYWORDS:
        if isinstance(out.get(keyword), (dict, list)):
            out[keyword] = map_custom_types_to_json_schema(out[keyword])
    for keyword in _SCHEMA_MAP_KEYWORDS:
        nested = out.get(keyword)
        if isinstance(nested, dict):
            out[keyword] = {k: map_custom_types_to_json_schema(v) for k, v in nested.items()}
    for keyword in _SCHEMA_LIST_KEYWORDS:
        if isinstance(out.get(keyword), list):
            out[keyword] = [map_custom_types_to_json_schema(s) for s in out[keyword]]

    return out


@dataclass
class ParameterSchema:
    """Schema definition for tool parameters."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None
    # Support for nested schemas (arrays and objects)
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    additionalProperties: Optional[bool] = None
    # This parameter consumes symbolic media refs (``media_ref:<id>``)
    # directly: the orchestrator's pre-execution media resolution leaves it
    # untouched instead of rewriting the ref to a stored URL. Declare on
    # params of tools that track/re-emit refs rather than fetch the bytes.
    media_ref_passthrough: bool = False
    
    def to_json_schema_property(self) -> Dict[str, Any]:
        """Convert to JSON Schema property format."""
        json_type = self.type.value
        # Map custom types to valid JSON Schema types
        if self.type in (ParameterType.MEDIA, ParameterType.TEXT):
            json_type = "string"

        prop = {
            "type": json_type,
            "description": self.description,
        }

        # For MEDIA params, add format hint and URL instruction
        if self.type == ParameterType.MEDIA:
            prop["format"] = "uri"
            prop["description"] += (
                " Pass the image URL from the conversation, or use media_ref:<id>"
                " to reference a previously generated image (the id is shown in"
                " the generation result). Do NOT pass base64-encoded image data."
            )

        if self.default is not None:
            prop["default"] = self.default
        if self.enum is not None:
            prop["enum"] = self.enum
        if self.minimum is not None:
            prop["minimum"] = self.minimum
        if self.maximum is not None:
            prop["maximum"] = self.maximum
        if self.pattern is not None:
            prop["pattern"] = self.pattern

        # Add nested schema support for arrays and objects. These come straight
        # from a tool's declared parameters (and, for MCP-delivered tools, from
        # the cached remote schema), so they carry internal variable types that
        # are not JSON Schema types — normalize before handing them to a model
        # provider.
        if self.items is not None:
            prop["items"] = map_custom_types_to_json_schema(self.items)
        if self.properties is not None:
            prop["properties"] = {
                name: map_custom_types_to_json_schema(schema)
                for name, schema in self.properties.items()
            }
        if self.additionalProperties is not None:
            prop["additionalProperties"] = self.additionalProperties

        return prop


@dataclass
class ToolFailure(Generic[ResultType]):
    """Explicit failure returned by a tool body or result adapter.

    Tool output is deliberately opaque to the framework: ordinary mappings may
    contain fields such as ``status``, ``error`` or ``success`` as domain data.
    Returning this wrapper is the unambiguous way to say that the *invocation*
    failed while optionally preserving a structured payload for diagnostics.
    """

    error: str
    output: Optional[ResultType] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult(Generic[ResultType]):
    """Production-grade tool execution result.

    ``success`` and ``error`` are the execution envelope. ``output`` is opaque
    application data and is never inspected to infer whether execution failed.
    """
    name: str
    input: Dict[str, Any]
    output: Optional[ResultType] = None
    error: Optional[str] = None
    success: bool = True
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_failure(
        self, failure: ToolFailure[ResultType]
    ) -> "ToolResult[ResultType]":
        """Apply an explicit failure while retaining this result's identity."""

        self.output = failure.output
        self.error = failure.error
        self.success = False
        self.metadata.update(failure.metadata)
        if failure.error_type:
            self.metadata.setdefault("error_type", failure.error_type)
        return self
    
    @property
    def is_success(self) -> bool:
        return self.success and self.error is None


@dataclass
class ToolSchema:
    """Universal schema for tools (function and HTTP)."""
    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, ParameterSchema] = field(default_factory=dict)
    
    # HTTP-specific fields
    url: Optional[str] = None
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    
    # Tool-level default for requiring user approval before execution
    require_approval: bool = False

    # Opt-in marker: when True, the orchestrator may run this tool in
    # parallel with other parallelizable tools the model emits in the
    # same assistant turn (via the batch executor's asyncio.gather).
    # Defaults False — sequential is the safe default for any tool with
    # observable side effects, hidden ordering dependencies, or shared
    # mutable state. Approval-required tools also force serial execution
    # regardless of this flag (see executor docs).
    parallelizable: bool = False

    # Does this tool change state outside the agent's own context? Three-valued
    # ON PURPOSE:
    #   True  — declared write (mutates product/platform/external state)
    #   False — declared read
    #   None  — NOT YET CLASSIFIED
    #
    # None is not a synonym for False. A host that audits writes, or gates them,
    # needs to tell "someone decided this is a read" apart from "nobody has
    # said" — otherwise every new tool defaults into the safe-looking bucket and
    # the gap is invisible. Hosts are expected to fail their own coverage checks
    # on None rather than assume. It defaults to None (not False) so adding the
    # field cannot silently reclassify tools that predate it.
    writes: Optional[bool] = None

    # Plan-mode marker: when True, this tool may execute even while the
    # agent is inside `enter_plan_mode`. Read-only tools (search, list,
    # describe, ToolSearch itself) should set this; anything with side
    # effects (write/edit/send/post) MUST leave it False so the executor
    # synthesizes a "blocked — call exit_plan_mode first" tool result.
    #
    # Derived from `writes` by the @tool decorator when `writes` is declared —
    # the two encode the same fact and hand-maintaining both is how they drift.
    # Still settable directly for tools that construct a ToolSchema themselves.
    is_read_only: bool = False

    # ── Read-through dedupe serve contract ────────────────────────────────
    # Default: NOT servable. A tool opts in by declaring an idempotency
    # class; the ledger dedupe gate (core/react/dedupe.py) may then serve a
    # stored observation for an identical call instead of re-executing.
    # `is_read_only` is NOT sufficient — many reads are time-relative
    # (LAST_30_DAYS) or carry run-scoped artifacts (_data_id render refs)
    # that must never be served across contexts. TTLs are per-class
    # constants owned by code, not per-tool knobs:
    #   "none"             — never served (default).
    #   "discovery"        — structural facts (account lists, campaign
    #                        catalogs); hours-stable.
    #   "performance_read" — metrics reads; served within one turn only.
    idempotency_class: str = "none"
    # Extra deps keys folded into the dedupe key for scope-sensitive tools
    # (org scoping is implicit — the ledger lives on one org's thread and
    # serving is org-guarded at the observation store). E.g. ["assistant_id"]
    # for tools whose result depends on which agent asks.
    dedupe_scope_dims: List[str] = field(default_factory=list)

    # Add metadata for arbitrary data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make ToolSchema hashable by using immutable fields only."""
        return hash((self.name, self.description, self.tool_type, self.url, self.method, self.timeout))
    
    def __eq__(self, other):
        """Equality based on name and tool_type."""
        if not isinstance(other, ToolSchema):
            return False
        return self.name == other.name and self.tool_type == other.tool_type
    
    def to_universal_schema(self) -> Dict[str, Any]:
        """Convert to universal JSON Schema format."""
        properties = {}
        required = []

        for param_name, param in self.parameters.items():
            properties[param_name] = param.to_json_schema_property()
            if param.required:
                required.append(param_name)

        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

        # Add returns field for consistency with function tools
        if self.tool_type == ToolType.HTTP_API:
            schema["returns"] = {
                "type": "object",
                "description": "HTTP API response data"
            }

        # Carry metadata through so provider adapters (e.g. Anthropic's strict
        # mode) can read flags like `strict=True`. Omitted previously, which
        # silently dropped per-tool strict opt-ins on the way to the LLM.
        if self.metadata:
            schema["metadata"] = dict(self.metadata)

        return schema
    
    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format.

        Note: This is a fallback method. Provider clients should use their own
        convert_schema_to_provider_format() method for proper name sanitization
        and mapping support.
        """
        # Direct implementation to avoid import issues
        universal = self.to_universal_schema()

        provider = provider.lower()

        if provider in ["openai", "groq", "xai", "mistral", "ollama"]:
            # OpenAI format - name sanitization is handled by provider clients
            return {
                "type": "function",
                "function": universal
            }
        elif provider == "anthropic":
            # Anthropic format
            return {
                "name": universal["name"],
                "description": universal["description"],
                "input_schema": universal["parameters"]
            }
        elif provider in ["gemini", "google"]:
            # Gemini format
            return {
                "name": universal["name"],
                "description": universal["description"],
                "parameters": universal["parameters"]
            }
        else:
            # Default: return universal schema
            return universal


@dataclass
class PreparedCall(Generic[ResultType]):
    """Prepared tool call with validated context and inputs."""
    tool_name: str
    function: Any  # Callable - avoiding import issues
    context: Any   # ContextType - avoiding import issues
    validated_inputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        import time
        self.prepared_at = time.time()
