"""Anthropic provider implementation."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import anthropic
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError, is_retryable_error
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount, UsageData
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import AnthropicStreamNormalizer, extract_mcp_result_text
from ..core.streaming import StreamChunk
from ..models.anthropic import (
    EFFORT_LEVELS,
    supports_effort,
    supports_native_mcp,
    supports_structured_outputs,
    supports_temperature,
    supports_thinking,
    thinking_disable_param,
)
from ..utils.image import data_uri_to_base64_and_mimetype

if TYPE_CHECKING:
    from ..core.tools.mcp import NativeMCPServerConfig

logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKENS = 32768


class AnthropicClient(ModelClient):
    """Anthropic provider client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        # `effort` scales adaptive thinking on the models that think by default
        # (`output_config.effort`; see models.anthropic.supports_effort). It is a
        # per-client setting rather than a per-call kwarg because it arrives the
        # way `max_tokens` does — from the assistant's stored model_config, which
        # is spread into the client constructor — and an unhandled key there
        # would otherwise be absorbed silently by the base class.
        effort = kwargs.pop("effort", None)
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(
                f"effort must be one of {EFFORT_LEVELS}, got {effort!r}"
            )
        self.effort: Optional[str] = effort
        # `cache_ttl` extends the prompt-cache TTL of the tools + system
        # breakpoints (the org-stable prefix) beyond the 5-minute default;
        # the message breakpoint always stays default (agent rounds are
        # seconds apart, and the transcript prefix changes every turn anyway).
        # Same delivery path as `effort`: a model_config key spread into the
        # constructor. Motivation: 63% of system-agent turns arriving >5 min
        # after the previous one re-billed a ~40K-token prompt at the full
        # input rate (prod llm_timeline, Aug 2026); "1h" trades a 2x (vs
        # 1.25x) cache-write premium on the ~30K stable prefix for cache
        # reads on every returning turn and on new threads org-wide.
        cache_ttl = kwargs.pop("cache_ttl", None)
        if cache_ttl is not None and cache_ttl not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {cache_ttl!r}")
        self.cache_ttl: Optional[str] = cache_ttl
        super().__init__(model=model, api_key=api_key, **kwargs)
        from .sdk_client_cache import get_or_create_sdk_client

        # Reuse the SDK client (and its warm TLS connection pool) across
        # turns — see sdk_client_cache. AnthropicClient itself stays per-turn.
        self.client = get_or_create_sdk_client(
            "anthropic", api_key, lambda: anthropic.AsyncAnthropic(api_key=api_key)
        )
        self.provider_name = "anthropic"
        self._tool_name_mapping: Dict[str, str] = {}

        # Stream normalizer for unified streaming handling
        # Note: Pass instance-level mapping for tool name restoration
        self._stream_normalizer = AnthropicStreamNormalizer(self._tool_name_mapping)

    def _supports_structured_outputs(self) -> bool:
        """Check if the current model supports native structured outputs."""
        return supports_structured_outputs(self.model)

    def _apply_effort(self, request_params: Dict[str, Any]) -> None:
        """Set `output_config.effort` when configured and the model accepts it.

        Merged into an existing `output_config` (structured outputs use the
        same object for `format`) rather than replacing it. Silently skipped
        on models that would 400 on the parameter, so one config can be
        shared across model swaps.
        """
        if not self.effort or not supports_effort(self.model):
            return
        output_config = request_params.get("output_config")
        if not isinstance(output_config, dict):
            output_config = {}
        request_params["output_config"] = {**output_config, "effort": self.effort}

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Anthropic format with loosened constraints.

        Supports strict mode for models that support structured outputs:
        - If model supports structured outputs and schema has 'strict': True,
          use native strict mode without loosening schema
        - Otherwise, loosen schema for better compatibility
        """
        import re

        original_name = schema["name"]
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name)
        sanitized_name = re.sub(r"_+", "_", sanitized_name).strip("_")[:128]

        if sanitized_name != original_name:
            self._tool_name_mapping[sanitized_name] = original_name

        # Strict-mode tool schemas: emit `additionalProperties: false` so the
        # model cannot hallucinate extra keys (e.g. passing Google Ads
        # `customer_id`/`query` to a Meta Ads tool). Per-tool opt-in via
        # `schema["strict"]=True` or `schema["metadata"]["strict"]=True`.
        # Anthropic caps strict tools at 20 per request, so this has to stay
        # opt-in rather than default-on; use it for tools that have
        # cognitively-similar siblings whose parameter shapes the model
        # confuses.
        strict_flag = schema.get("strict", False) or schema.get("metadata", {}).get("strict", False)
        use_strict = strict_flag and self._supports_structured_outputs()

        tool_definition = {
            "name": sanitized_name,
            "description": schema["description"],
            "input_schema": (
                normalize_json_schema(schema["parameters"], SchemaMode.NATIVE_STRICT)
                if use_strict
                else normalize_json_schema(schema["parameters"], SchemaMode.LOOSE)
            ),
        }

        # Add strict flag if using strict mode
        if use_strict:
            tool_definition["strict"] = True

        return tool_definition

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        """Convert Message to Anthropic format."""
        from ..core.message import DocumentBlock, ImageBlock, TextBlock, VideoBlock

        # Handle tool result messages (for sending tool outputs back)
        # Anthropic expects "user" role for tool results, not "tool"
        if message.tool_call_id and message.role in (MessageRole.USER, MessageRole.TOOL):
            # This is a tool result message - Anthropic requires "user" role
            anthropic_message = {"role": "user"}

            # Structured (multimodal) tool result: content is a list of blocks.
            # Anthropic's tool_result.content accepts a list of text + image blocks,
            # which is how we pipe pixels from tools like analyze_creative into
            # the next LLM turn without needing a separate user message.
            if isinstance(message.content, list):
                sub_content: List[Dict[str, Any]] = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if block.text and block.text.strip():
                            sub_content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageBlock):
                        if block.image_url.startswith("data:"):
                            base64_content, media_type = data_uri_to_base64_and_mimetype(
                                block.image_url
                            )
                            sub_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_content,
                                    },
                                }
                            )
                        else:
                            sub_content.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": block.image_url},
                                }
                            )
                    elif isinstance(block, VideoBlock):
                        # Claude cannot view videos — degrade to a text reference
                        # so the rest of the tool_result still delivers.
                        sub_content.append(
                            {
                                "type": "text",
                                "text": (
                                    f"[Video: {block.video_url} — Claude cannot view "
                                    f"videos; switch to a Gemini-powered agent for video analysis.]"
                                ),
                            }
                        )
                if not sub_content:
                    sub_content = [{"type": "text", "text": "[empty result]"}]
                anthropic_message["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": sub_content,
                    }
                ]
                return anthropic_message

            # Ensure tool result content is not empty or whitespace-only
            tool_content = (
                message.content if isinstance(message.content, str) else str(message.content)
            )
            if not tool_content or not tool_content.strip():
                tool_content = "[empty result]"

            anthropic_message["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": tool_content,
                }
            ]
            return anthropic_message

        anthropic_message = {"role": message.role.value}

        # Provider-run tool-search blocks (server_tool_use +
        # tool_search_tool_result) recorded by the orchestrator. Replaying them
        # is what stops the model re-searching for the same deferred tool every
        # turn. Emitted verbatim and adjacent — the API validates the pair.
        search_blocks = []
        if message.role == MessageRole.ASSISTANT:
            search_blocks = list((message.metadata or {}).get("tool_search_blocks") or [])

        # Handle assistant messages with tool calls
        if message.tool_calls and message.role == MessageRole.ASSISTANT:
            content_list = []

            # Add text content if present and non-whitespace
            if message.content and message.content.strip():
                content_list.append({"type": "text", "text": message.content})

            content_list.extend(search_blocks)

            # Results of provider-executed MCP calls, keyed by the id of the
            # mcp_tool_use they answer. Replayed inline below so each
            # mcp_tool_use block keeps its pair.
            mcp_results_by_id = {
                r.get("tool_use_id"): r
                for r in (message.metadata or {}).get("mcp_tool_results", [])
            }

            # Add tool use blocks
            for tool_call in message.tool_calls:
                import json

                args = tool_call.get("function", {}).get("arguments", {})
                # Normalize: OpenAI-style string arguments → dict for Anthropic API
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                # Provider-executed MCP calls round-trip as the block types the
                # API issued them as. Emitting them as plain `tool_use` is a
                # protocol error twice over: the id is an `mcptoolu_` id the
                # API will not accept in a tool_use block, and there is no
                # local tool_result message to pair with it (nothing ran on our
                # side), so the request 400s on an unanswered tool_use.
                if tool_call.get("type") == "mcp_function":
                    call_id = tool_call.get("id", "")
                    server_name = tool_call.get("server_name")
                    result = mcp_results_by_id.get(call_id)

                    # Replay the pair only when it can be reconstructed
                    # completely AND this endpoint accepts the blocks at all.
                    # A half-formed mcp_tool_use — no id, no server_name, or no
                    # matching result — is rejected by the API, which would 400
                    # every remaining turn of the thread; so is any mcp_tool_use
                    # on Bedrock, which refuses the MCP beta outright and whose
                    # threads can inherit these calls when a model is switched
                    # mid-conversation. Dropping it instead costs the model the
                    # raw payload from a turn it already summarized in its own
                    # text; that degrades grounding, it does not break the
                    # conversation.
                    if not (
                        call_id
                        and server_name
                        and result
                        and self._supports_native_mcp()
                    ):
                        logger.debug(
                            "Dropping unreplayable native-MCP call from history "
                            "(id=%s server_name=%s has_result=%s native_mcp=%s)",
                            call_id,
                            server_name,
                            result is not None,
                            self._supports_native_mcp(),
                        )
                        continue

                    content_list.append(
                        {
                            "type": "mcp_tool_use",
                            "id": call_id,
                            "name": tool_call.get("function", {}).get("name", ""),
                            "input": args,
                            "server_name": server_name,
                        }
                    )
                    content_list.append(
                        {
                            "type": "mcp_tool_result",
                            "tool_use_id": call_id,
                            "is_error": bool(result.get("is_error", False)),
                            "content": [
                                {"type": "text", "text": result.get("content", "") or ""}
                            ],
                        }
                    )
                    continue

                content_list.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": args,
                    }
                )

            anthropic_message["content"] = content_list
            return anthropic_message

        # An assistant turn that searched but called nothing (the model looked
        # up a tool, then answered from context). Without this the blocks are
        # dropped by the plain-string path below and the discovery is forgotten
        # exactly as if we had never recorded it.
        #
        # Only the str shape is handled here; a list content (multi-modal
        # assistant turn built by _build_message) falls through to the block
        # loop below and has the search blocks appended there instead. Handling
        # str-only *and* returning early would silently drop every text, image
        # and document block on such a message.
        if search_blocks and isinstance(message.content, str):
            content_list = []
            if message.content.strip():
                content_list.append({"type": "text", "text": message.content})
            content_list.extend(search_blocks)
            anthropic_message["content"] = content_list
            return anthropic_message

        # Handle regular messages
        if isinstance(message.content, str):
            # Anthropic requires non-empty, non-whitespace content
            # Ensure we always have content, or use a placeholder
            content = message.content.strip() if message.content else ""

            if not content:
                # Empty content - use a minimal placeholder that's not whitespace
                # Anthropic rejects whitespace-only content
                anthropic_message["content"] = [{"type": "text", "text": "[no content]"}]
            else:
                anthropic_message["content"] = content
        else:
            content_list = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    # Only add text blocks with non-whitespace content
                    if block.text and block.text.strip():
                        content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.image_url.startswith("data:"):
                        base64_content, media_type = data_uri_to_base64_and_mimetype(
                            block.image_url
                        )
                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_content,
                                },
                            }
                        )
                    else:
                        # Anthropic API doesn't support URL image sources for most models.
                        # Download the image and convert to base64.
                        try:
                            from ..utils.image import url_to_base64_and_mimetype

                            base64_content, media_type = url_to_base64_and_mimetype(
                                block.image_url, resize=True
                            )
                            content_list.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_content,
                                    },
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to download image from URL, falling back to URL source: {e}"
                            )
                            content_list.append(
                                {"type": "image", "source": {"type": "url", "url": block.image_url}}
                            )
                elif isinstance(block, VideoBlock):
                    # Claude cannot view videos; emit a text reference so the model
                    # stays coherent instead of erroring on an unsupported block type.
                    content_list.append(
                        {
                            "type": "text",
                            "text": (
                                f"[Video at {block.video_url} — Claude cannot view videos; "
                                f"switch to a Gemini-powered agent to analyze video creatives.]"
                            ),
                        }
                    )
                elif isinstance(block, DocumentBlock):
                    # Claude document blocks only support PDF and plain text
                    if block.document_type in ("pdf", "txt"):
                        if block.document_url.startswith("data:"):
                            base64_content, media_type = data_uri_to_base64_and_mimetype(
                                block.document_url
                            )
                            content_list.append(
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_content,
                                    },
                                }
                            )
                        else:
                            content_list.append(
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "url",
                                        "url": block.document_url,
                                    },
                                }
                            )
                    else:
                        # For non-PDF/txt documents (csv, xlsx, xls, etc.),
                        # download and send as text content
                        try:
                            import httpx

                            resp = httpx.get(block.document_url, timeout=30, follow_redirects=True)
                            resp.raise_for_status()
                            text = resp.content.decode("utf-8", errors="replace")
                            filename_info = f" [{block.filename}]" if block.filename else ""
                            content_list.append(
                                {"type": "text", "text": f"[Document{filename_info}]\n\n{text}"}
                            )
                        except Exception as e:
                            filename_info = f" {block.filename}" if block.filename else ""
                            content_list.append(
                                {
                                    "type": "text",
                                    "text": f"[Error processing document{filename_info}: {str(e)}]",
                                }
                            )

            # A multi-modal assistant turn that also searched: append the pair
            # after the real blocks rather than replacing them.
            content_list.extend(search_blocks)

            # Ensure content_list is not empty (after filtering whitespace-only blocks)
            if not content_list:
                content_list = [{"type": "text", "text": "[no content]"}]

            anthropic_message["content"] = content_list

        return anthropic_message

    def _prepare_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Prepare messages for Anthropic format (system separate)."""
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                anthropic_messages.append(self.convert_message_to_provider_format(msg))

        return system_content, anthropic_messages

    def _supports_native_mcp(self) -> bool:
        """Check if the current model supports native MCP via beta API."""
        return supports_native_mcp(self.model)

    @staticmethod
    def _count_optional_properties(schema: Any) -> int:
        """Count optional (non-required) properties recursively in a JSON schema.

        Anthropic's strict-mode grammar compiler caps the *total* number of
        optional properties across all strict tool schemas in a request at
        24. We count every property whose name isn't in its enclosing
        object's `required` array — at the top level *and* nested inside
        properties, items, or allOf/anyOf/oneOf branches — so the demotion
        logic can stay safely under the cap.
        """
        if not isinstance(schema, dict):
            return 0
        count = 0
        if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
            required = set(schema.get("required") or [])
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name not in required:
                    count += 1
                count += AnthropicClient._count_optional_properties(prop_schema)
        if isinstance(schema.get("items"), dict):
            count += AnthropicClient._count_optional_properties(schema["items"])
        for combinator in ("allOf", "anyOf", "oneOf"):
            branches = schema.get(combinator)
            if isinstance(branches, list):
                for branch in branches:
                    count += AnthropicClient._count_optional_properties(branch)
        return count

    @staticmethod
    def _count_union_type_params(schema: Any) -> int:
        """Count union-typed properties recursively in a JSON schema.

        Anthropic's strict-mode grammar compiler caps union-typed parameters
        across all strict schemas in a request at 16, and documents them as
        *the* most expensive feature ("union types ... create exponential
        compilation cost"). A property is union-typed when it uses
        `anyOf`/`oneOf`, or declares `type` as a list of >1 concrete type
        (e.g. `["string", "null"]` — the shape an Optional/nullable field
        usually compiles to). We count these the same way as optional
        properties so the cap can keep the request under the 16 limit, which
        ``_count_optional_properties`` does not measure.
        """
        if not isinstance(schema, dict):
            return 0
        count = 0
        type_val = schema.get("type")
        # A multi-entry `type` list is a union — including the common nullable
        # form `["string", "null"]`, which Anthropic's docs explicitly count
        # toward the 16-union-param limit.
        if isinstance(type_val, list) and len(type_val) > 1:
            count += 1
        if isinstance(schema.get("anyOf"), list) or isinstance(schema.get("oneOf"), list):
            count += 1
        if isinstance(schema.get("properties"), dict):
            for prop_schema in schema["properties"].values():
                count += AnthropicClient._count_union_type_params(prop_schema)
        if isinstance(schema.get("items"), dict):
            count += AnthropicClient._count_union_type_params(schema["items"])
        for combinator in ("allOf", "anyOf", "oneOf"):
            branches = schema.get(combinator)
            if isinstance(branches, list):
                for branch in branches:
                    count += AnthropicClient._count_union_type_params(branch)
        return count

    @staticmethod
    def _demote_strict_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of `tool` with the strict flag removed and schema loosened."""
        demoted = dict(tool)
        demoted.pop("strict", None)
        schema = demoted.get("input_schema")
        # Loosen the schema so additionalProperties:false isn't interpreted as
        # a hard rejection rule by the API. Keep the rest of the schema intact.
        if isinstance(schema, dict):
            schema = dict(schema)
            if schema.get("additionalProperties") is False:
                schema["additionalProperties"] = True
            demoted["input_schema"] = schema
        return demoted

    @staticmethod
    def _prune_stale_tool_references(
        request_params: Dict[str, Any],
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
    ) -> int:
        """Drop replayed `tool_reference`s that name tools absent from this request.

        Anthropic validates every `tool_reference` inside a replayed
        `tool_search_tool_result` against the request's CURRENT `tools` array
        and rejects the whole request otherwise:

            400 Tool reference 'list_all_ad_accounts' not found in available tools

        The search blocks are recorded on the assistant message the turn they
        happen and replayed verbatim on every later turn (see
        `convert_message_to_provider_format`), while the tool list is rebuilt
        per turn — so any drift between the two (a tool hidden by a
        visibility filter, dropped by a cap, or a bundle that stopped loading)
        surfaces as a 400 on a request that has nothing to do with that tool.
        In production (2026-08-10..15) that killed 18 pipeline specialist runs
        12–20 minutes in, each ending "I wasn't able to finish this run".

        A stale reference is not information the model needs — the tool is not
        callable — so it is pruned here rather than failing the turn. When a
        result block would be left with no references, BOTH halves of the pair
        (`server_tool_use` + `tool_search_tool_result`, matched by id) are
        removed: the API rejects either half alone. Blocks are copied, never
        mutated in place — the replayed dicts are the ones stored on the
        message metadata / checkpoint.

        Returns the number of references dropped (for the log line).

        ``mcp_servers`` are the connector servers on THIS request. Their tools
        are not in ``tools`` (an ``mcp_toolset`` stands in for the whole
        server), yet a deferred connector tool the model discovered through
        search is replayed as a ``tool_reference`` like any other — so the
        names a present server is known to serve count as available. The API
        names a connector tool in a reference as ``"<server name>_<tool>"``,
        verbatim and unsanitised (observed live: server ``"Deep Wiki-X"`` →
        ``"Deep Wiki-X_read_wiki_structure"``), while ``mcp_tool_use`` carries
        the bare tool name — both spellings are accepted. A name no present
        server claims is pruned; the worst case is one extra search, whereas
        keeping a reference the API cannot resolve is the 400 this method
        exists to prevent.
        """
        tools = request_params.get("tools") or []
        available = {
            t.get("name") for t in tools if isinstance(t, dict) and t.get("name")
        }
        for server in mcp_servers or []:
            server_name = getattr(server, "name", "") or ""
            for source in (
                getattr(server, "known_tools", None),
                getattr(server, "allowed_tools", None),
            ):
                for tool_name in source or []:
                    if isinstance(tool_name, str):
                        available.add(tool_name)
                        available.add(f"{server_name}_{tool_name}")
        messages = request_params.get("messages") or []
        dropped = 0
        for msg_index, msg in enumerate(messages):
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            new_content: List[Any] = []
            remove_ids: set = set()
            changed = False
            for block in content:
                if not (
                    isinstance(block, dict)
                    and block.get("type") == "tool_search_tool_result"
                ):
                    new_content.append(block)
                    continue
                inner = block.get("content")
                refs = inner.get("tool_references") if isinstance(inner, dict) else None
                if not isinstance(refs, list):
                    new_content.append(block)
                    continue
                kept = [
                    r
                    for r in refs
                    if not (
                        isinstance(r, dict)
                        and r.get("type") == "tool_reference"
                        and r.get("tool_name") not in available
                    )
                ]
                if len(kept) == len(refs):
                    new_content.append(block)
                    continue
                dropped += len(refs) - len(kept)
                changed = True
                if kept:
                    new_content.append(
                        {**block, "content": {**inner, "tool_references": kept}}
                    )
                else:
                    remove_ids.add(block.get("tool_use_id"))
            if not changed:
                continue
            if remove_ids:
                new_content = [
                    b
                    for b in new_content
                    if not (
                        isinstance(b, dict)
                        and (
                            (b.get("type") == "server_tool_use" and b.get("id") in remove_ids)
                            or (
                                b.get("type") == "tool_search_tool_result"
                                and b.get("tool_use_id") in remove_ids
                            )
                        )
                    )
                ]
            if not new_content:
                # An assistant turn that consisted only of a search pair.
                new_content = [{"type": "text", "text": "[no content]"}]
            messages[msg_index] = {**msg, "content": new_content}
        if dropped:
            logger.warning(
                "[TOOL_SEARCH] pruned %d stale tool_reference(s) from replayed "
                "search results (tools not in this request's tool list)",
                dropped,
            )
        return dropped

    @staticmethod
    def _apply_strict_tool_cap(request_params: Dict[str, Any]) -> None:
        """Demote excess strict tools to non-strict to fit Anthropic's per-request caps.

        Anthropic compiles strict-mode tool schemas into a constrained-decoding
        grammar (regular non-strict tools are *not* compiled) and enforces three
        hard caps on the *combined total across all strict schemas in one
        request*: (1) ≤20 tools with `strict: true`, (2) ≤24 optional
        (non-required) properties, and (3) ≤16 union-typed parameters
        (`anyOf`/`oneOf` or `["x","null"]`) — union types being the most
        expensive feature ("exponential compilation cost"). Exceeding any of
        them, or producing a grammar that's simply too large, yields a 400
        ("Schemas contains too many optional parameters" / "Schema is too
        complex for compilation"). These caps are easy to blow when an agent
        auto-loads tool bundles (e.g. the always-on ad_platform set) on top of
        its configured tools — the combined set, not any single tool, is what
        overflows. We demote the tail (the last N strict tools) so the prefix —
        typically the most important / always-loaded tools — keeps strict
        semantics, and emit a WARNING so the operator can reduce the tool set or
        opt specific tools out at definition time. The reactive
        `_demote_all_strict_tools` retry in astream_chat/chat is the backstop
        for the residual "grammar too large" case these counts don't capture.
        """
        tool_cap = 20
        optional_param_cap = 24
        union_param_cap = 16
        tools = request_params.get("tools")
        if not tools:
            return
        strict_indices = [i for i, t in enumerate(tools) if isinstance(t, dict) and t.get("strict")]
        if not strict_indices:
            return

        new_tools = list(tools)
        demoted_names: List[str] = []
        original_strict_count = len(strict_indices)

        # Phase 1: enforce the 20-tool cap. Demote tail strict tools first.
        while len(strict_indices) > tool_cap:
            idx = strict_indices.pop()
            new_tools[idx] = AnthropicClient._demote_strict_tool(new_tools[idx])
            demoted_names.append(new_tools[idx].get("name", f"<idx={idx}>"))

        # Phase 2: enforce the 24-optional-param cap by walking the strict
        # schemas that survived phase 1 and demoting from the tail until the
        # running total fits. Recomputing the total each iteration keeps the
        # logic simple at the cost of O(N²) on the strict subset, which is
        # fine for N ≤ 20.
        def total_optional_params() -> int:
            return sum(
                AnthropicClient._count_optional_properties(
                    new_tools[i].get("input_schema", {}) if isinstance(new_tools[i], dict) else {}
                )
                for i in strict_indices
            )

        original_optional_total = total_optional_params()
        while strict_indices and total_optional_params() > optional_param_cap:
            idx = strict_indices.pop()
            new_tools[idx] = AnthropicClient._demote_strict_tool(new_tools[idx])
            demoted_names.append(new_tools[idx].get("name", f"<idx={idx}>"))

        # Phase 3: enforce the 16-union-type-param cap the same way. Union types
        # are the most expensive grammar feature, so this is often the limit a
        # request actually trips even when tool count and optional params look
        # fine.
        def total_union_params() -> int:
            return sum(
                AnthropicClient._count_union_type_params(
                    new_tools[i].get("input_schema", {}) if isinstance(new_tools[i], dict) else {}
                )
                for i in strict_indices
            )

        original_union_total = total_union_params()
        while strict_indices and total_union_params() > union_param_cap:
            idx = strict_indices.pop()
            new_tools[idx] = AnthropicClient._demote_strict_tool(new_tools[idx])
            demoted_names.append(new_tools[idx].get("name", f"<idx={idx}>"))

        if not demoted_names:
            return

        request_params["tools"] = new_tools
        logger.warning(
            "[STRICT_TOOL_CAP] strict tools=%d optional_params=%d union_params=%d "
            "exceeded Anthropic caps (tools=%d, optional_params=%d, union_params=%d) "
            "— demoted %d tool(s) to non-strict: %s. To restore strictness, opt "
            "specific tools out at definition (strict=False) or reduce the tool set "
            "passed to this agent.",
            original_strict_count,
            original_optional_total,
            original_union_total,
            tool_cap,
            optional_param_cap,
            union_param_cap,
            len(demoted_names),
            demoted_names,
        )

    @classmethod
    def _demote_all_strict_tools(cls, request_params: Dict[str, Any]) -> bool:
        """Demote every strict tool in the request to non-strict. Returns True if any changed.

        This is the documented, guaranteed remedy for ``400 "Schema is too
        complex for compilation"``: grammar compilation runs *only* for strict
        tools (and structured outputs), so removing `strict` from all tools
        means there is no grammar left to compile, regardless of which
        combination of tool count / optional params / union types blew the
        budget. It's the reactive backstop for the residual case
        ``_apply_strict_tool_cap`` can't pre-empt (e.g. a grammar that's simply
        too large rather than over a counted limit). Strict's only benefit is
        disambiguating cognitively-similar tools; trading that for a working
        request is the right call on a hard rejection. Originals are not
        mutated — modified tools are replaced with copies, preserving
        ``cache_control`` and every other wrapper field.
        """
        tools = request_params.get("tools")
        if not tools:
            return False
        new_tools = list(tools)
        changed = False
        for i, tool in enumerate(new_tools):
            if isinstance(tool, dict) and tool.get("strict"):
                new_tools[i] = cls._demote_strict_tool(tool)
                changed = True
        if changed:
            request_params["tools"] = new_tools
        return changed

    @staticmethod
    def _is_schema_too_complex_error(error: Exception) -> bool:
        """True if `error` is Anthropic's tool-schema grammar-compilation 400.

        Anthropic returns this rejection under at least two distinct wordings,
        both meaning the strict-tool grammar can't be compiled cheaply and both
        remedied by demoting strict tools:
        - "Schema is too complex for compilation" — a single schema or the
          combined optional/union counts exceed a hard cap.
        - "The compiled grammar is too large, which would cause performance
          issues. Simplify your tool schemas or reduce the number of strict
          tools." — the residual "grammar simply too large" case with no fixed
          count, which the proactive `_apply_strict_tool_cap` can't pre-empt.
        Match either so the `_demote_all_strict_tools` backstop fires for both.
        """
        text = str(getattr(error, "message", None) or error).lower()
        return (
            "too complex for compilation" in text
            or "schema is too complex" in text
            or "compiled grammar is too large" in text
            or "reduce the number of strict tools" in text
        )

    @staticmethod
    def _is_tool_search_unsupported_error(error: Exception) -> bool:
        """Whether a 400 says this model/endpoint can't do server-side tool search.

        `defer_loading` + the `tool_search_tool_regex_*` server tool are gated on
        the PROVIDER being Anthropic, not on the model supporting them — an
        Anthropic-named proxy, or a model predating the feature, therefore 400s
        on every request with no way to recover. Detect it so the caller can
        retry once with the full tool list, the way the strict-tool ladder does.
        """
        text = str(getattr(error, "message", None) or error).lower()
        return "tool_search" in text or "defer_loading" in text

    #: Beta header for Anthropic's MCP connector. The 2025-04-04 revision put
    #: tool selection on the server entry (`tool_configuration`) and loaded
    #: EVERY tool of every server into context with no way to defer; the
    #: 2025-11-20 revision moves selection into a `mcp_toolset` tools entry,
    #: which is also where `defer_loading` for connector tools lives.
    MCP_CONNECTOR_BETA = "mcp-client-2025-11-20"

    @staticmethod
    def _is_tool_search_tool(tool: Any) -> bool:
        return isinstance(tool, dict) and str(tool.get("type", "")).startswith(
            "tool_search_tool"
        )

    @classmethod
    def _apply_native_mcp(
        cls, request_params: Dict[str, Any], mcp_servers: List["NativeMCPServerConfig"]
    ) -> None:
        """Wire the MCP connector into `request_params` in place.

        Two halves the API validates together: an `mcp_servers[]` entry per
        server (connection details) and exactly one `mcp_toolset` per server
        in `tools` (which of its tools the model sees, and whether they load
        up front). The toolsets are DEFERRED whenever the request already
        carries a `tool_search_tool_*` entry — i.e. whenever the caller has
        chosen native deferral for the local tools — so connector tools get
        the same treatment as the ~60 local ones and are discovered through
        the same search tool. Without a search tool there is nothing to
        discover them with, so they load eagerly, as before.

        Toolsets are inserted BEFORE the search tool rather than appended: the
        search tool must stay last so `_apply_prompt_caching`'s final-tool
        cache breakpoint lands on it (a deferred entry cannot carry
        `cache_control`). With no search tool present the toolsets go last
        and are not deferred, and a non-deferred toolset accepts the marker.
        """
        request_params["mcp_servers"] = [s.to_anthropic_format() for s in mcp_servers]

        tools = list(request_params.get("tools") or [])
        search_index = next(
            (i for i, t in enumerate(tools) if cls._is_tool_search_tool(t)), None
        )
        defer = search_index is not None
        toolsets = [s.to_anthropic_toolset(defer_loading=defer) for s in mcp_servers]
        insert_at = search_index if search_index is not None else len(tools)
        tools[insert_at:insert_at] = toolsets
        request_params["tools"] = tools

        betas = request_params.get("betas")
        betas = list(betas) if isinstance(betas, list) else []
        if cls.MCP_CONNECTOR_BETA not in betas:
            betas.append(cls.MCP_CONNECTOR_BETA)
        request_params["betas"] = betas

        logger.debug(
            "Native MCP: %d server(s) %s, toolsets deferred=%s",
            len(mcp_servers),
            [s.name for s in mcp_servers],
            defer,
        )

    @classmethod
    def _disable_tool_deferral(cls, request_params: Dict[str, Any]) -> bool:
        """Strip deferral + the server search tool; load every tool instead.

        Covers both places deferral is expressed: the per-tool
        ``defer_loading`` flag on local schemas and the ``default_config`` /
        ``configs`` of an ``mcp_toolset``. Leaving the toolset flags in place
        after dropping the search tool would leave those tools unreachable.

        Returns True if anything changed (i.e. a retry is worth attempting).
        """
        tools = request_params.get("tools")
        if not tools:
            return False
        rebuilt, changed = [], False
        for tool in tools:
            if not isinstance(tool, dict):
                rebuilt.append(tool)
                continue
            if cls._is_tool_search_tool(tool):
                changed = True  # drop the server tool entirely
                continue
            if "defer_loading" in tool:
                tool = {k: v for k, v in tool.items() if k != "defer_loading"}
                changed = True
            if tool.get("type") == "mcp_toolset":
                stripped, toolset_changed = cls._strip_toolset_deferral(tool)
                if toolset_changed:
                    tool = stripped
                    changed = True
            rebuilt.append(tool)
        if changed:
            request_params["tools"] = rebuilt
        return changed

    @staticmethod
    def _strip_toolset_deferral(toolset: Dict[str, Any]) -> tuple:
        """Copy of an `mcp_toolset` with every `defer_loading` removed."""
        changed = False
        out = dict(toolset)
        default_config = out.get("default_config")
        if isinstance(default_config, dict) and "defer_loading" in default_config:
            default_config = {k: v for k, v in default_config.items() if k != "defer_loading"}
            changed = True
            if default_config:
                out["default_config"] = default_config
            else:
                out.pop("default_config", None)
        configs = out.get("configs")
        if isinstance(configs, dict):
            new_configs: Dict[str, Any] = {}
            configs_changed = False
            for name, cfg in configs.items():
                if isinstance(cfg, dict) and "defer_loading" in cfg:
                    cfg = {k: v for k, v in cfg.items() if k != "defer_loading"}
                    configs_changed = True
                    if not cfg:
                        continue  # nothing left to say about this tool
                new_configs[name] = cfg
            if configs_changed:
                changed = True
                if new_configs:
                    out["configs"] = new_configs
                else:
                    out.pop("configs", None)
        return out, changed

    @staticmethod
    def _stamp_stream_span_cache_attrs(stream: Any, usage: Any) -> bool:
        """Backfill prompt-cache token attrs on the OpenInference stream span.

        openinference-instrumentation-anthropic 1.1.0 emits
        ``llm.token_count.prompt_details.cache_read/cache_write`` for
        NON-streaming responses only; its streaming extractor folds cache
        tokens into the prompt total and drops the split
        (``_stream._MessageExtractor``). Every chat call here streams, so
        Arize showed no Anthropic cache telemetry at all — an export gap
        that once got "prod runs no prompt caching" written up as a finding
        when ``llm_timeline`` showed caching working fine.

        Reaches into the wrapper's ``_with_span`` deliberately; every access
        is guarded so an instrumentor upgrade degrades to missing attrs,
        never a broken stream. Returns True once attrs were written so the
        caller can stop re-checking.
        """
        try:
            with_span = getattr(stream, "_with_span", None)
            if with_span is None:
                return False
            attrs: Dict[str, Any] = {}
            read = getattr(usage, "cache_read_tokens", 0) or 0
            write = getattr(usage, "cache_write_tokens", 0) or 0
            if read:
                attrs["llm.token_count.prompt_details.cache_read"] = read
            if write:
                attrs["llm.token_count.prompt_details.cache_write"] = write
            if not attrs:
                return False
            with_span.set_attributes(attrs)
            return True
        except Exception:  # noqa: BLE001 — telemetry must never break a stream
            return False

    async def _drain_abandoned_stream(self, stream: Any, stream_normalizer: Any) -> None:
        """Drain an abandoned instrumented stream so its LLM span exports.

        Shared rescue body for the GeneratorExit and CancelledError handlers
        in ``astream_chat``: the OpenInference wrapper only ends its span
        when the iterator completes, so drain the few remaining events
        (usually just ``message_stop``) inside a hard bound, stamping cache
        attrs if usage arrives during the drain. Best-effort by contract —
        any failure (including a second cancellation landing mid-drain) is
        swallowed; the caller re-raises the original abandonment either way.
        """
        try:

            async def _drain_remaining() -> None:
                stamped = False
                async for ev in stream:
                    if stamped:
                        continue
                    drained_chunk = stream_normalizer.normalize_chunk(ev)
                    if drained_chunk.usage:
                        stamped = self._stamp_stream_span_cache_attrs(
                            stream, drained_chunk.usage
                        )

            await asyncio.wait_for(_drain_remaining(), timeout=2.0)
        except BaseException:  # noqa: BLE001 — already closing
            pass

    # Block types that accept a cache_control marker. Notably absent:
    # thinking / redacted_thinking — marking those is a 400.
    _CACHEABLE_BLOCK_TYPES = frozenset({"text", "image", "tool_use", "tool_result", "document"})

    @classmethod
    def _apply_prompt_caching(
        cls, request_params: Dict[str, Any], *, ttl: Optional[str] = None
    ) -> None:
        """Add Anthropic prompt-cache breakpoints to `request_params` in place.

        Anthropic renders the prompt as ``tools → system → messages`` and a
        cache_control marker covers everything *before* it, so each
        breakpoint caches its own prefix tier:

        - last tool          → tool definitions
        - last system block  → tools + system prompt
        - last message block → tools + system + conversation so far

        The message breakpoint is what makes multi-turn agent loops cheap:
        every round re-sends the whole transcript, and without it each round
        re-bills the entire history at the full input rate. With it, round N
        reads round N-1's prefix from cache and only the newest turn is
        uncached. The tool marker is kept even though the system marker
        subsumes its prefix — tools and system are separate cache tiers, so
        a system-prompt change still leaves the tools tier hittable.

        Anthropic allows 4 breakpoints per request; we place at most 3.
        TTL: within-turn agent rounds are seconds apart, so the message
        breakpoint always stays at the 5-minute default. The tools/system
        tiers honour ``ttl`` (call sites pass ``self.cache_ttl``) — prod
        data showed the dominant cache misses were BETWEEN user turns (63%
        of system-agent turns >5 min idle re-billed the whole ~40K-token
        prompt), which "1h" on the stable prefix converts to reads at a
        one-time 2x write premium.

        Known limit: a breakpoint only looks back 20 content blocks for the
        previous cache entry, so a single turn that adds more than 20 blocks
        (e.g. a very wide parallel tool batch) re-writes the prefix once
        instead of reading it. The tools/system tiers still hit.
        """
        # The prefix tiers' marker. Anthropic's default TTL is 5m; only a
        # non-default value is sent on the wire so unconfigured clients keep
        # byte-identical requests.
        if ttl and ttl != "5m":
            prefix_cache_control = {"type": "ephemeral", "ttl": ttl}
        else:
            prefix_cache_control = {"type": "ephemeral"}

        tools = request_params.get("tools")
        if tools:
            new_tools = list(tools)
            last = new_tools[-1]
            if isinstance(last, dict):
                last_copy = dict(last)
                last_copy["cache_control"] = dict(prefix_cache_control)
                new_tools[-1] = last_copy
                request_params["tools"] = new_tools

        system = request_params.get("system")
        if isinstance(system, str) and system:
            request_params["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": dict(prefix_cache_control),
                }
            ]
        elif isinstance(system, list) and system:
            new_system = list(system)
            last = new_system[-1]
            if isinstance(last, dict):
                last_copy = dict(last)
                last_copy["cache_control"] = dict(prefix_cache_control)
                new_system[-1] = last_copy
                request_params["system"] = new_system

        cls._mark_final_message_block(request_params)

    @classmethod
    def _mark_final_message_block(cls, request_params: Dict[str, Any]) -> None:
        """Mark the last cacheable content block of the final message."""
        messages = request_params.get("messages")
        if not messages:
            return
        last_msg = messages[-1]
        if not isinstance(last_msg, dict):
            return

        content = last_msg.get("content")
        if isinstance(content, str):
            if not content:
                return
            new_content: List[Any] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            idx = next(
                (
                    i
                    for i in range(len(content) - 1, -1, -1)
                    if isinstance(content[i], dict)
                    and content[i].get("type") in cls._CACHEABLE_BLOCK_TYPES
                ),
                None,
            )
            if idx is None:
                return
            new_content = list(content)
            block_copy = dict(new_content[idx])
            block_copy["cache_control"] = {"type": "ephemeral"}
            new_content[idx] = block_copy
        else:
            return

        new_messages = list(messages)
        new_msg = dict(last_msg)
        new_msg["content"] = new_content
        new_messages[-1] = new_msg
        request_params["messages"] = new_messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to Anthropic.

        Supports:
        1. Native MCP (for all Claude models via beta API):
           Uses mcp_servers parameter for server-side tool execution
        2. Native structured outputs (for supported models like Claude Sonnet 4.5):
           Uses output_format parameter with guaranteed schema compliance
        3. Tool-based workaround (for older models):
           Uses a synthetic tool to force JSON output
        """
        try:

            # _prepare_messages may download images via blocking httpx and run
            # Pillow resize — both CPU/I/O bound. Offload to a worker thread so
            # we never stall the event loop under ASGI servers.
            system_content, anthropic_messages = await asyncio.to_thread(
                self._prepare_messages, messages
            )

            # Extract thinking parameters from kwargs (won't be passed to API directly)
            thinking_enabled = kwargs.pop("thinking_enabled", False)
            budget_tokens = kwargs.pop("budget_tokens", 32000)
            # Provider-neutral opt-OUT of thinking (other providers ignore the
            # kwarg). Needed for short deterministic completions on models that
            # think by default — see `thinking_disable_param`.
            thinking_disabled = kwargs.pop("thinking_disabled", False)

            # Check for native MCP
            use_native_mcp = mcp_servers and len(mcp_servers) > 0 and self._supports_native_mcp()

            # Handle JSON schema
            json_tool_name = None
            use_native_structured_output = json_schema and self._supports_structured_outputs()

            if json_schema:
                if use_native_structured_output:
                    # Use native structured output API (beta feature)
                    # Prepare schema by ensuring additionalProperties is set
                    prepared_schema = normalize_json_schema(json_schema, SchemaMode.NATIVE_STRICT)

                    request_params = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        "betas": ["structured-outputs-2025-11-13"],
                        "output_format": {
                            "type": "json_schema",
                            "schema": prepared_schema,
                        },
                        **kwargs,
                    }

                    request_params["max_tokens"] = (
                        max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
                    )

                    logger.debug(f"Using native structured output API for model {self.model}")
                else:
                    # Fall back to tool-based approach for older models
                    json_tool_name = "json_tool"
                    json_tool = {
                        "name": json_tool_name,
                        "description": "Respond with structured JSON matching the specified schema",
                        "input_schema": json_schema,
                    }

                    if tools:
                        tools = list(tools) + [json_tool]
                    else:
                        tools = [json_tool]

                    kwargs["tool_choice"] = {"type": "tool", "name": json_tool_name}

                    request_params = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        **kwargs,
                    }

                    request_params["max_tokens"] = (
                        max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
                    )

                    logger.debug(f"Using tool-based JSON output for model {self.model}")
            else:
                # Regular request without JSON schema
                request_params = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    "temperature": temperature,
                    **kwargs,
                }

                request_params["max_tokens"] = (
                    max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
                )

            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools
            self._apply_effort(request_params)

            # Enable extended thinking if requested and model supports it
            if thinking_enabled and supports_thinking(self.model):
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                # Extended thinking requires temperature=1
                request_params["temperature"] = 1
                logger.debug(
                    f"Extended thinking enabled for {self.model} "
                    f"with budget_tokens={budget_tokens}"
                )
                logger.debug(
                    f"Anthropic tools parameter:\n{json.dumps(tools, indent=2, default=str)}"
                )
            elif thinking_disabled and not thinking_enabled:
                disable = thinking_disable_param(self.model)
                if disable is not None:
                    request_params["thinking"] = disable

            # Handle native MCP servers
            if use_native_mcp:
                self._apply_native_mcp(request_params, mcp_servers)

            # Determine which client to use
            # Some models (e.g. Opus 4.7) reject `temperature` as a deprecated
            # parameter; drop it before hitting the API.
            if not supports_temperature(self.model):
                request_params.pop("temperature", None)

            self._apply_strict_tool_cap(request_params)
            self._prune_stale_tool_references(
                request_params, mcp_servers if use_native_mcp else None
            )
            self._apply_prompt_caching(request_params, ttl=self.cache_ttl)

            # Use beta client for structured outputs or native MCP
            use_beta_client = use_native_structured_output or use_native_mcp
            create_fn = (
                self.client.beta.messages.create
                if use_beta_client
                else self.client.messages.create
            )
            try:
                response = await asyncio.wait_for(
                    create_fn(**request_params), timeout=self.timeout
                )
            except anthropic.BadRequestError as e:
                # See astream_chat: retry once with simplified tool schemas when
                # Anthropic rejects the combined schemas as too complex to compile.
                if self._is_schema_too_complex_error(
                    e
                ) and self._demote_all_strict_tools(request_params):
                    logger.warning(
                        "[SCHEMA_TOO_COMPLEX] Anthropic rejected tool schemas as too "
                        "complex to compile (model=%s, tools=%d); retrying with "
                        "simplified schemas. If this recurs, reduce the tool set "
                        "for this agent.",
                        self.model,
                        len(request_params.get("tools") or []),
                    )
                    response = await asyncio.wait_for(
                        create_fn(**request_params), timeout=self.timeout
                    )
                else:
                    raise

            # Extract content and tool calls from response
            content = ""
            tool_calls = []
            mcp_tool_results = []  # Track MCP tool results for metadata

            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
                    elif hasattr(block, "type") and block.type == "tool_use":
                        if json_tool_name and block.name == json_tool_name:
                            # Extract JSON from tool response (fallback mode)
                            content = json.dumps(block.input)
                        else:
                            # Convert Anthropic tool_use to OpenAI-compatible format
                            # Restore original tool name if it was sanitized
                            tool_name = block.name
                            original_name = self._tool_name_mapping.get(tool_name, tool_name)

                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {"name": original_name, "arguments": block.input},
                                }
                            )
                    elif hasattr(block, "type") and block.type == "mcp_tool_use":
                        # Native MCP tool call (server-side execution)
                        # These are handled by Anthropic's API directly
                        tool_calls.append(
                            {
                                "id": getattr(block, "id", ""),
                                "type": "mcp_function",
                                "function": {
                                    "name": getattr(block, "name", ""),
                                    "arguments": getattr(block, "input", {}),
                                },
                                "server_name": getattr(block, "server_name", None),
                            }
                        )
                        logger.debug(f"MCP tool use: {getattr(block, 'name', 'unknown')}")
                    elif hasattr(block, "type") and block.type == "mcp_tool_result":
                        # Native MCP tool result (already executed by the API).
                        # This is an observation, not prose — it must NOT be
                        # concatenated into `content`, or raw tool payloads end
                        # up rendered as the assistant's answer. Mirrors the
                        # streaming path in AnthropicStreamNormalizer.
                        is_error = getattr(block, "is_error", False)
                        result_content = extract_mcp_result_text(
                            getattr(block, "content", "")
                        )

                        mcp_tool_results.append(
                            {
                                "tool_use_id": getattr(block, "tool_use_id", ""),
                                "is_error": is_error,
                                "content": result_content,
                            }
                        )

                        if is_error:
                            logger.warning(f"MCP tool error: {result_content}")
                        else:
                            logger.debug(f"MCP tool result received")

            if tool_calls:
                logger.debug(f"Returning {len(tool_calls)} tool calls to orchestrator")

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )

            # When prompt caching is active, Anthropic splits input tokens into
            # three buckets: `input_tokens` (uncached, full price),
            # `cache_creation_input_tokens` (just-cached, ~1.25x), and
            # `cache_read_input_tokens` (cache hit, ~0.1x). Sum them so
            # `prompt_tokens` continues to mean "all input tokens this turn"
            # — otherwise downstream rate-limit and billing telemetry silently
            # under-reports as soon as cache hits start landing.
            #
            # The SDK returns int or None for these fields. Be strict about
            # typing — a plain `or 0` would let truthy non-numeric values
            # (e.g. MagicMock attributes in tests) sneak in and corrupt math.
            def _as_int(val: Any) -> int:
                return val if isinstance(val, int) else 0

            cache_creation_tokens = _as_int(
                getattr(response.usage, "cache_creation_input_tokens", 0)
            )
            cache_read_tokens = _as_int(getattr(response.usage, "cache_read_input_tokens", 0))
            input_tokens_total = (
                response.usage.input_tokens + cache_creation_tokens + cache_read_tokens
            )

            usage = TokenCount(
                prompt_tokens=input_tokens_total,
                completion_tokens=response.usage.output_tokens,
                total_tokens=input_tokens_total + response.usage.output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_creation_tokens,
            )

            if cache_creation_tokens or cache_read_tokens:
                logger.debug(
                    "Anthropic prompt cache: read=%d, written=%d, uncached=%d",
                    cache_read_tokens,
                    cache_creation_tokens,
                    response.usage.input_tokens,
                )

            # Build metadata
            metadata = {"response_id": response.id}
            if cache_creation_tokens or cache_read_tokens:
                metadata["cache_read_input_tokens"] = cache_read_tokens
                metadata["cache_creation_input_tokens"] = cache_creation_tokens
            if mcp_tool_results:
                metadata["mcp_tool_results"] = mcp_tool_results
            if use_native_mcp:
                metadata["native_mcp"] = True

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.stop_reason,
                metadata=metadata,
            )

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except anthropic.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"Anthropic API error: {str(e)}", self.provider_name, original_error=e
            )

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to Anthropic.

        Supports:
        1. Native MCP (for all Claude models via beta API):
           Uses mcp_servers parameter for server-side tool execution
        2. Native structured outputs (for supported models like Claude Sonnet 4.5):
           Uses output_format parameter with guaranteed schema compliance (streaming supported!)
        3. Tool-based workaround (for older models):
           Uses a synthetic tool to force JSON output

        Emits a structured warning on completion if the call took longer
        than the legacy 120s default, so we can measure whether the bump
        to 300s is masking a real perf regression.
        """
        import json
        import logging
        import time

        logger = logging.getLogger(__name__)

        _slow_threshold_s = 120.0
        _astream_started_at = time.monotonic()

        try:
            # Offload sync image-download + Pillow resize work off the event loop.
            system_content, anthropic_messages = await asyncio.to_thread(
                self._prepare_messages, messages
            )

            # Extract thinking parameters from kwargs (won't be passed to API directly)
            thinking_enabled = kwargs.pop("thinking_enabled", False)
            budget_tokens = kwargs.pop("budget_tokens", 32000)
            # Provider-neutral opt-OUT of thinking (other providers ignore the
            # kwarg). Needed for short deterministic completions on models that
            # think by default — see `thinking_disable_param`.
            thinking_disabled = kwargs.pop("thinking_disabled", False)

            logger.debug(f"Streaming request to Anthropic with {len(anthropic_messages)} messages:")
            for idx, msg in enumerate(anthropic_messages):
                logger.debug(
                    f"  Message {idx}: role={msg.get('role')}, content_type={type(msg.get('content'))}, content_length={len(str(msg.get('content')))}"
                )
                logger.debug(
                    f"    Content preview: {json.dumps(msg.get('content'), default=str)[:200]}"
                )

            # Check for native MCP
            use_native_mcp = mcp_servers and len(mcp_servers) > 0 and self._supports_native_mcp()

            # Handle JSON schema
            json_tool_name = None
            use_native_structured_output = json_schema and self._supports_structured_outputs()

            if json_schema:
                if use_native_structured_output:
                    # Use native structured output API (beta feature) - streaming supported!
                    # Prepare schema by ensuring additionalProperties is set
                    prepared_schema = normalize_json_schema(json_schema, SchemaMode.NATIVE_STRICT)

                    request_params = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        "stream": True,
                        "betas": ["structured-outputs-2025-11-13"],
                        "output_format": {
                            "type": "json_schema",
                            "schema": prepared_schema,
                        },
                        **kwargs,
                    }

                    request_params["max_tokens"] = (
                        max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
                    )

                    logger.debug(
                        f"Using native structured output API with streaming for model {self.model}"
                    )
                else:
                    # Fall back to tool-based approach for older models
                    json_tool_name = "json_tool"
                    json_tool = {
                        "name": json_tool_name,
                        "description": "Respond with structured JSON matching the specified schema",
                        "input_schema": json_schema,
                    }

                    if tools:
                        tools = list(tools) + [json_tool]
                    else:
                        tools = [json_tool]

                    kwargs["tool_choice"] = {"type": "tool", "name": json_tool_name}

                    request_params = {
                        "model": self.model,
                        "messages": anthropic_messages,
                        "temperature": temperature,
                        "stream": True,
                        **kwargs,
                    }

                    request_params["max_tokens"] = (
                        max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
                    )

                    logger.debug(
                        f"Using tool-based JSON output with streaming for model {self.model}"
                    )
            else:
                # Regular streaming without JSON schema
                request_params = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    "temperature": temperature,
                    "stream": True,
                    **kwargs,
                }

                request_params["max_tokens"] = (
                    max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
                )

            if system_content:
                request_params["system"] = system_content
            if tools:
                request_params["tools"] = tools
            self._apply_effort(request_params)

            # Enable extended thinking if requested and model supports it
            if thinking_enabled and supports_thinking(self.model):
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
                # Extended thinking requires temperature=1
                request_params["temperature"] = 1
                logger.debug(
                    f"Extended thinking enabled for {self.model} "
                    f"with budget_tokens={budget_tokens}"
                )
            elif thinking_disabled and not thinking_enabled:
                disable = thinking_disable_param(self.model)
                if disable is not None:
                    request_params["thinking"] = disable

            # Handle native MCP servers
            if use_native_mcp:
                self._apply_native_mcp(request_params, mcp_servers)

            # Some models (e.g. Opus 4.7) reject `temperature` as a deprecated
            # parameter; drop it before hitting the API.
            if not supports_temperature(self.model):
                request_params.pop("temperature", None)

            self._apply_strict_tool_cap(request_params)
            self._prune_stale_tool_references(
                request_params, mcp_servers if use_native_mcp else None
            )
            self._apply_prompt_caching(request_params, ttl=self.cache_ttl)

            # Determine which client to use
            # Use beta client for structured outputs or native MCP
            use_beta_client = use_native_structured_output or use_native_mcp
            create_fn = (
                self.client.beta.messages.create
                if use_beta_client
                else self.client.messages.create
            )
            try:
                stream = await asyncio.wait_for(
                    create_fn(**request_params), timeout=self.timeout
                )
            except anthropic.BadRequestError as e:
                # Anthropic compiles a constraint grammar from the combined tool
                # schemas; past a complexity budget it rejects the request with
                # "Schema is too complex for compilation." Retry once with the
                # tool schemas simplified rather than hard-failing the turn.
                if self._is_schema_too_complex_error(
                    e
                ) and self._demote_all_strict_tools(request_params):
                    logger.warning(
                        "[SCHEMA_TOO_COMPLEX] Anthropic rejected tool schemas as too "
                        "complex to compile (model=%s, tools=%d); retrying with "
                        "simplified schemas. If this recurs, reduce the tool set "
                        "for this agent.",
                        self.model,
                        len(request_params.get("tools") or []),
                    )
                    stream = await asyncio.wait_for(
                        create_fn(**request_params), timeout=self.timeout
                    )
                elif self._is_tool_search_unsupported_error(
                    e
                ) and self._disable_tool_deferral(request_params):
                    # Deferral is enabled per PROVIDER, but supported per MODEL.
                    # Without this the mismatch fails every request for that
                    # assistant, with no ladder to recover — pre-loading the
                    # full tool list costs prompt tokens but keeps it working.
                    logger.warning(
                        "[TOOL_SEARCH_UNSUPPORTED] Anthropic rejected deferred "
                        "tool loading (model=%s); retrying with all tools loaded. "
                        "This model does not support server-side tool search — "
                        "raise the tool_search threshold for it.",
                        self.model,
                    )
                    stream = await asyncio.wait_for(
                        create_fn(**request_params), timeout=self.timeout
                    )
                else:
                    raise

            # Create a NEW normalizer instance for each streaming session
            # This prevents race conditions when multiple streams run in parallel
            # (e.g., multi-agent parallel subagent execution)
            # Previously, we used self._stream_normalizer.reset_state() but shared
            # state causes corruption when streams interleave.
            stream_normalizer = AnthropicStreamNormalizer(self._tool_name_mapping)

            _events_seen = 0
            _cache_attrs_stamped = False
            try:
                async for event in stream:
                    _events_seen += 1
                    # Normalize Anthropic events to StreamChunk
                    normalized_chunk = stream_normalizer.normalize_chunk(event)

                    # Backfill the cache-token split onto the instrumentor's
                    # LLM span while it is still open (the streaming
                    # extractor drops it — see _stamp_stream_span_cache_attrs).
                    if not _cache_attrs_stamped and normalized_chunk.usage:
                        _cache_attrs_stamped = self._stamp_stream_span_cache_attrs(
                            stream, normalized_chunk.usage
                        )

                    # Only yield if there's actual content or metadata to send
                    if (
                        normalized_chunk.delta
                        or normalized_chunk.thinking_delta
                        or normalized_chunk.tool_calls
                        or normalized_chunk.mcp_tool_results
                        # Search blocks arrive on content_block_stop, where every
                        # other field above is empty — omit this and the chunk is
                        # filtered out here and the whole capture→replay chain is
                        # dead code. Any future provider-payload field added to
                        # StreamChunk needs a clause here for the same reason.
                        or normalized_chunk.tool_search_blocks
                        or normalized_chunk.finish_reason
                    ):
                        yield normalized_chunk

                    # Deliberately NO `break` on `message_stop`. The SDK closes the
                    # SSE stream right after that event, so exhausting the iterator
                    # costs nothing — and it is REQUIRED for observability: the
                    # OpenInference wrapper around this stream only ends its LLM
                    # span when the `async for` runs to completion. Breaking out
                    # early abandons that async generator (GeneratorExit, not an
                    # Exception, so its except-branch never runs either), the span
                    # is never ended and never exported. That is exactly why, for
                    # weeks, no streaming Anthropic call appeared under any agent
                    # span in Arize while non-streaming and 4xx calls did.
            except GeneratorExit:
                # A CONSUMER abandoned this generator mid-stream (an upstream
                # generator in the chain being closed has the same effect as
                # the in-loop `break` the comment above forbids): the
                # OpenInference wrapper's iterator never completes, its
                # `except Exception` cannot catch GeneratorExit, and the LLM
                # span is silently dropped — tokens/cost recorded by the
                # POST_CALL callback, nothing in Arize. Production symptom:
                # web-root turns whose usage rows exist but whose traces hold
                # zero LLM spans. Drain the few remaining events (usually
                # just message_stop — usage has already arrived by the time
                # anyone abandons a finished answer) inside a hard bound so
                # the wrapper can end its span, then keep closing. PEP 525
                # allows awaiting during aclose; yielding is what it forbids.
                logger.warning(
                    "[TRACE] anthropic stream abandoned by consumer after "
                    "%d events (model=%s); draining briefly so the LLM span "
                    "still exports",
                    _events_seen,
                    self.model,
                )
                if getattr(stream, "_with_span", None) is None:
                    # No instrumentor wrapper — nothing to rescue; don't
                    # spend teardown time draining a bare SDK stream.
                    raise
                await self._drain_abandoned_stream(stream, stream_normalizer)
                raise
            except asyncio.CancelledError:
                # Task cancellation (user stop, upstream wait_for timeout,
                # client disconnect tearing the task down) abandons the
                # wrapper the same way GeneratorExit does — BaseException,
                # so neither the wrapper's `except Exception` nor our
                # handlers below see it, and the span silently vanishes
                # with NO witness log. Same rescue, then re-raise so
                # cancellation semantics are preserved.
                logger.warning(
                    "[TRACE] anthropic stream cancelled after %d events "
                    "(model=%s); draining briefly so the LLM span still "
                    "exports",
                    _events_seen,
                    self.model,
                )
                if getattr(stream, "_with_span", None) is not None:
                    await self._drain_abandoned_stream(stream, stream_normalizer)
                raise

            # Slow-call observability (rec #5 review finding): emit a
            # structured warning when a successful stream exceeded the
            # legacy 120s default. Tells us whether the 120→300s bump is
            # masking real perf issues (high rate of >120s calls) vs.
            # just removing noisy false-positive timeouts (rare >120s).
            _elapsed = time.monotonic() - _astream_started_at
            if _elapsed > _slow_threshold_s:
                logger.warning(
                    "anthropic_stream_slow",
                    extra={
                        "model": self.model,
                        "elapsed_s": round(_elapsed, 1),
                        "threshold_s": _slow_threshold_s,
                        "configured_timeout_s": self.timeout,
                    },
                )
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except anthropic.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"Anthropic streaming error: {str(e)}", self.provider_name, original_error=e
            )
