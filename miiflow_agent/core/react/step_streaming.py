"""One reasoning step: the streaming LLM call and everything it accumulates.

The loop's hottest and most intricate method: opens the provider stream via
the tool executor, accumulates text/thinking/tool-call deltas (including the
optimistic answer streaming + ANSWER_RETRACTED protocol), classifies the
turn (tool calls vs final answer vs truncation), and produces the ReActStep
the loop routes on. Moved verbatim from ReActOrchestrator (which keeps a
thin delegate); ``self._orch`` is the orchestrator.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List

from ..message import Message, MessageRole
from .events import EventFactory
from .models import ReActStep
from .orchestrator import (
    _LazyTrace,
    _attach_search_blocks,
    _extract_partial_results,
    _format_missing_params_error,
    _optimistic_answer_streaming_enabled,
    _preparse_tool_args_string,
    _preview,
    _summarize_messages_for_trace,
    _summarize_tool_call,
)

if TYPE_CHECKING:
    from ..agent import RunContext
    from .execution import ExecutionState
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


class StepStreamer:
    """Runs one streaming reasoning step and classifies its outcome."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    async def execute_reasoning_step_native(
        self, context: RunContext, state: "ExecutionState"
    ) -> ReActStep:
        """Execute a single reasoning step with native tool calling.

        The model either calls tools (continue loop) or responds with
        plain text (final answer). No explicit thinking mechanism needed.
        """
        step = ReActStep(step_number=state.current_step, thought="")
        step_start_time = time.time()
        # Initialized before the try so the except handler can retract
        # optimistically streamed deltas no matter where the failure hit.
        pending_answer_deltas: List[str] = []
        # Set when a provider-executed (native MCP) block retracted streamed
        # answer text. Unlike a local tool call — which makes the whole step an
        # ACTION step, so the narration correctly rides along with the tool_use
        # — an MCP-only step is still an ANSWER step, and `buffer` holds the
        # narration and the real answer concatenated. Answering from the buffer
        # is how a fabricated "you opened 14 pull requests: #1367, ..." written
        # BEFORE GitHub was queried ended up persisted as part of the answer,
        # with the model apologising for it in the same breath.
        native_mcp_retracted = False
        # Same reason: the finally reads these, and a BaseException (task
        # cancellation on the step_started await) skips `except Exception`
        # entirely — an unbound local would raise from the finally and replace
        # the CancelledError, turning a graceful timeout into an internal error.
        provider_search_blocks: List[Dict[str, Any]] = []
        # Watermark: messages at or after this index are the ones THIS step
        # appended. Search blocks may only attach within that window — see
        # _attach_search_blocks.
        messages_before_step = len(context.messages)

        try:
            # Publish step start event
            await self._orch.event_bus.publish(EventFactory.step_started(state.current_step))

            # Single-phase: Stream LLM response WITH tools enabled
            buffer = ""
            tokens_used = 0
            # Slots whose ACTION_STREAMING announcement already went out.
            announced_tool_slots: set = set()
            cost = 0.0
            # Accumulate tool calls during streaming
            accumulated_tool_calls = {}  # index -> {id, function: {name, arguments}}
            # Native-MCP calls the provider executed itself (Anthropic/OpenAI
            # connect to the MCP server directly). Kept SEPARATE from
            # accumulated_tool_calls: they are already-finished transcript
            # events, not work this process owes, so they must not make the
            # turn an action turn nor reach the local dispatcher.
            provider_executed_calls: Dict[str, Dict[str, Any]] = {}  # id -> call
            provider_tool_results: List[Dict[str, Any]] = []
            finish_reason = None

            logger.debug(f"Step {state.current_step} - Calling LLM with tools enabled")

            # Build the provider-format tool schemas ONCE per step: the trace
            # log below and stream_with_tools share the same list (building
            # ~50 schemas twice was pure setup latency).
            try:
                step_tools = self._orch.tool_executor._build_native_tool_schemas()
            except Exception:
                # Let stream_with_tools rebuild and surface the real error.
                step_tools = None

            # Trace: dump everything going INTO the LLM so we can audit the
            # exact messages, tool schemas, and generation settings on each
            # turn. Kept at INFO level (single prefix [LLM_TURN]) so it's easy
            # to grep and filter. Message/tool summaries are lazily rendered
            # so deployments logging above INFO pay nothing.
            def _render_trace_tool_names() -> str:
                if step_tools is None:
                    return "<unavailable: schema build failed>"
                names = []
                for _t in step_tools:
                    # Providers vary: OpenAI nests under function.name, others use top-level name.
                    _fn = _t.get("function") if isinstance(_t, dict) else None
                    names.append(
                        (_fn or {}).get("name")
                        if _fn
                        else (_t.get("name") if isinstance(_t, dict) else str(_t))
                    )
                return str(names)

            logger.info(
                "[LLM_TURN] step=%d OUT messages=%d tools=%d temp=%s max_tokens=%s\n%s\n  tool_names=%s",
                state.current_step,
                len(context.messages),
                len(step_tools or []),
                getattr(self._orch.tool_executor.agent, "temperature", None),
                getattr(self._orch.tool_executor.agent, "max_tokens", None),
                _LazyTrace(lambda: _summarize_messages_for_trace(context.messages)),
                _LazyTrace(_render_trace_tool_names),
            )

            async for chunk in self._orch.tool_executor.stream_with_tools(
                messages=context.messages, prebuilt_tools=step_tools
            ):
                # Native extended thinking (Anthropic thinking blocks, OpenAI
                # reasoning tokens) arrive on a dedicated channel separate
                # from response text. Pass through to the UI's thinking panel;
                # never enters the answer buffer.
                if getattr(chunk, "thinking_delta", None):
                    await self._orch.event_bus.publish(
                        EventFactory.thinking_chunk(
                            state.current_step, chunk.thinking_delta, buffer
                        )
                    )

                # Text deltas are not final until the provider turn is classified.
                # Frontier models often emit narration before tool_use blocks; if
                # a tool appears later the streamed text must not leak into the
                # final answer. Publish deltas optimistically as
                # FINAL_ANSWER_CHUNK and retract them (ANSWER_RETRACTED) the
                # moment a tool_use block starts; pending_answer_deltas records
                # what was streamed so it can be retracted/demoted. With the
                # kill switch off, deltas are buffered and replayed after the
                # stream closes (legacy behavior).
                if chunk.delta:
                    buffer += chunk.delta
                    if accumulated_tool_calls:
                        # Action turn already declared; treat further text as
                        # preamble narration. Emit as thinking_chunk.
                        await self._orch.event_bus.publish(
                            EventFactory.thinking_chunk(
                                state.current_step, chunk.delta, buffer
                            )
                        )
                    else:
                        pending_answer_deltas.append(chunk.delta)
                        if _optimistic_answer_streaming_enabled():
                            await self._orch.event_bus.publish(
                                EventFactory.final_answer_chunk(
                                    state.current_step,
                                    chunk.delta,
                                    "".join(pending_answer_deltas),
                                )
                            )

                # Split off calls the provider ran server-side before anything
                # else looks at this chunk's tool calls. They carry provider
                # ids (Anthropic `mcptoolu_*`) and have no pending work, so
                # they are recorded as observations after the stream, never
                # dispatched. Re-keying by id means the finalized block
                # (emitted at content_block_stop with parsed args) replaces the
                # placeholder emitted at content_block_start.
                if chunk.tool_calls:
                    for provider_call in chunk.tool_calls:
                        if provider_call.get("type") != "mcp_function":
                            continue
                        call_id = provider_call.get("id") or ""
                        is_new_call = call_id not in provider_executed_calls
                        provider_executed_calls[call_id] = provider_call

                        # Text streamed BEFORE this block was narration, not the
                        # answer — exactly as for a local tool call below. The
                        # retraction used to fire only on the local path, so a
                        # step whose only tool calls were MCP ones published its
                        # preamble as the final answer and left it on screen:
                        # a "here are your 14 pull requests" table written from
                        # priors, sent before GitHub had been queried at all.
                        # Retract per mcp_tool_use block start rather than once
                        # per step, so a turn that interleaves
                        # text → call → result → text retracts only the text
                        # that preceded a call. Deliberately NOT added to
                        # `accumulated_tool_calls`: the provider resumes
                        # generating after the result, and that trailing text
                        # IS the final answer.
                        if is_new_call and pending_answer_deltas:
                            preamble = "".join(pending_answer_deltas)
                            pending_answer_deltas = []
                            native_mcp_retracted = True
                            if _optimistic_answer_streaming_enabled():
                                await self._orch.event_bus.publish(
                                    EventFactory.answer_retracted(
                                        state.current_step,
                                        preamble,
                                        "native_mcp_tool_call",
                                    )
                                )
                            await self._orch.event_bus.publish(
                                EventFactory.thinking_chunk(
                                    state.current_step, preamble, buffer
                                )
                            )
                if getattr(chunk, "mcp_tool_results", None):
                    provider_tool_results.extend(chunk.mcp_tool_results)
                if getattr(chunk, "tool_search_blocks", None):
                    provider_search_blocks.extend(chunk.tool_search_blocks)

                # Accumulate tool calls if present in chunk
                # All providers now normalize to dict format via stream normalizers
                local_tool_calls = [
                    tc
                    for tc in (chunk.tool_calls or [])
                    if tc.get("type") != "mcp_function"
                ]
                if local_tool_calls:
                    if buffer and not accumulated_tool_calls:
                        # Frontier models often narrate before a tool call
                        # ("I'll do X"). Logged at INFO for observability;
                        # not actionable on its own.
                        logger.info(
                            "Step %d - Tool-call preamble: model emitted text "
                            "before tool call. text_preview=%s",
                            state.current_step,
                            _preview(buffer, 200),
                        )
                        if pending_answer_deltas:
                            preamble = "".join(pending_answer_deltas)
                            pending_answer_deltas = []
                            if _optimistic_answer_streaming_enabled():
                                # The preamble already went out live as
                                # FINAL_ANSWER_CHUNKs — tell consumers to clear
                                # their answer buffer before re-emitting it on
                                # the thinking channel.
                                await self._orch.event_bus.publish(
                                    EventFactory.answer_retracted(
                                        state.current_step, preamble, "tool_call"
                                    )
                                )
                            await self._orch.event_bus.publish(
                                EventFactory.thinking_chunk(
                                    state.current_step, preamble, buffer
                                )
                            )
                    for tool_call_dict in local_tool_calls:
                        # All tool calls are now dicts thanks to provider normalizers.
                        # Resolve the index for this tool call chunk so parallel
                        # tool_use blocks accumulate as distinct entries.
                        #
                        # Resolution order:
                        # 1. Use the provider-supplied `index` field when present
                        #    (Anthropic streams it on every content_block_start;
                        #    OpenAI's delta also carries it).
                        # 2. Else, key off the chunk's `id` — first time we see
                        #    an id, allocate the next slot; subsequent chunks
                        #    with the same id reuse it.
                        # 3. Else, fall back to slot 0 (legacy single-tool path
                        #    when no index/id signal is available).
                        provider_idx = tool_call_dict.get("index")
                        chunk_id = tool_call_dict.get("id")
                        if isinstance(provider_idx, int):
                            idx = provider_idx
                        elif chunk_id is not None:
                            # Find existing slot by id, else allocate next.
                            idx = None
                            for slot, slot_data in accumulated_tool_calls.items():
                                if slot_data.get("id") == chunk_id:
                                    idx = slot
                                    break
                            if idx is None:
                                idx = len(accumulated_tool_calls)
                        else:
                            # No index, no id — keep merging into slot 0.
                            # Either we haven't seen the id yet (first chunk
                            # of the only tool call) or this is a single-tool
                            # turn that streamed across multiple chunks.
                            idx = 0 if accumulated_tool_calls else 0

                        # Initialize on first chunk, merge on subsequent chunks
                        if idx not in accumulated_tool_calls:
                            # First chunk: initialize structure
                            accumulated_tool_calls[idx] = {
                                "id": None,
                                "type": "function",
                                "function": {
                                    "name": None,
                                    "arguments": None,  # Will be set based on provider format
                                },
                            }

                        # Update ID if present in this chunk
                        if tool_call_dict.get("id") is not None:
                            accumulated_tool_calls[idx]["id"] = tool_call_dict.get("id")

                        # Update type if present
                        if tool_call_dict.get("type") is not None:
                            accumulated_tool_calls[idx]["type"] = tool_call_dict.get(
                                "type"
                            )

                        # Preserve function_call_metadata (e.g. Gemini thought_signature)
                        if tool_call_dict.get("function_call_metadata"):
                            accumulated_tool_calls[idx]["function_call_metadata"] = (
                                tool_call_dict["function_call_metadata"]
                            )

                        # Update function name if present
                        function_data = tool_call_dict.get("function", {})
                        if function_data.get("name") is not None:
                            accumulated_tool_calls[idx]["function"]["name"] = (
                                function_data.get("name")
                            )
                            # Announce the tool the moment its block STARTS:
                            # argument generation can run tens of seconds (e.g.
                            # dispatch task briefs) with no other event, and
                            # ACTION_PLANNED only fires after args complete —
                            # without this the user stares at a blank bubble
                            # for the whole argument stream.
                            if idx not in announced_tool_slots:
                                announced_tool_slots.add(idx)
                                await self._orch.event_bus.publish(
                                    EventFactory.action_streaming(
                                        state.current_step,
                                        function_data.get("name"),
                                        tool_call_id=accumulated_tool_calls[idx].get(
                                            "id"
                                        ),
                                    )
                                )

                        # Handle arguments based on format:
                        # - OpenAI: sends progressively longer strings in each chunk
                        # - Anthropic: sends complete dict in final chunk
                        new_args = function_data.get("arguments")
                        if new_args is not None:
                            current_args = accumulated_tool_calls[idx]["function"][
                                "arguments"
                            ]

                            if isinstance(new_args, str):
                                # OpenAI format: string that grows with each chunk
                                # Provider already accumulates, so just use the latest value
                                accumulated_tool_calls[idx]["function"][
                                    "arguments"
                                ] = new_args
                            elif isinstance(new_args, dict):
                                # Anthropic format: dict (usually sent complete in one chunk)
                                if current_args is None or not isinstance(
                                    current_args, dict
                                ):
                                    accumulated_tool_calls[idx]["function"][
                                        "arguments"
                                    ] = new_args
                                else:
                                    # Merge dicts if both exist (defensive)
                                    current_args.update(new_args)
                            else:
                                # Unexpected format, log and store as-is
                                logger.warning(
                                    f"Unexpected arguments type in chunk: {type(new_args)}"
                                )
                                accumulated_tool_calls[idx]["function"][
                                    "arguments"
                                ] = new_args

                        logger.debug(
                            f"Tool call accumulated: {accumulated_tool_calls[idx]}"
                        )

                # Accumulate metrics
                output_tokens = None
                input_tokens = None
                if chunk.usage:
                    tokens_used = chunk.usage.total_tokens
                    output_tokens = getattr(chunk.usage, "completion_tokens", None)
                    input_tokens = getattr(chunk.usage, "prompt_tokens", None)
                    # Ground the context engine's estimate in what the provider
                    # actually counted. This is the only place with both halves
                    # of the comparison, and it costs nothing — the number
                    # already came back on the response.
                    self._orch._reconcile_context_usage(chunk.usage, state)
                if hasattr(chunk, "cost"):
                    cost += chunk.cost
                if getattr(chunk, "finish_reason", None):
                    finish_reason = chunk.finish_reason

            step.tokens_used = tokens_used
            step.cost = cost

            # Native-MCP calls finished server-side during the stream. Record
            # them on the observation trail now so the timeline, citations and
            # read_observation see the same history they would for a locally
            # dispatched tool, and carry the results on the assistant message
            # so the next request can replay the blocks.
            provider_mcp_metadata: Dict[str, Any] = {}
            if provider_executed_calls:
                provider_mcp_metadata = await self._orch._record_provider_executed_calls(
                    context=context,
                    state=state,
                    calls=provider_executed_calls,
                    results=provider_tool_results,
                )

            assistant_content = buffer.strip()
            if native_mcp_retracted and not accumulated_tool_calls:
                # Answer from what survived the retraction, not from the raw
                # buffer. `pending_answer_deltas` is emptied by each retraction
                # and refills from the deltas the provider streamed after the
                # mcp_tool_result, so it is exactly the post-call answer. An
                # empty result means the turn was narration only — which the
                # empty-turn branch below turns into another turn rather than a
                # blank final answer.
                assistant_content = "".join(pending_answer_deltas).strip()
            if not accumulated_tool_calls and finish_reason != "length":
                # Stream closed with no tool calls: the buffered deltas ARE the
                # final answer. With optimistic streaming they already went out
                # live; otherwise replay them now (legacy behavior).
                if not _optimistic_answer_streaming_enabled():
                    replayed = ""
                    for delta in pending_answer_deltas:
                        replayed += delta
                        await self._orch.event_bus.publish(
                            EventFactory.final_answer_chunk(
                                state.current_step, delta, replayed
                            )
                        )
            elif not accumulated_tool_calls and pending_answer_deltas:
                # max_tokens truncation: the text is not a final answer. Retract
                # what was optimistically streamed, then demote it to thinking.
                if _optimistic_answer_streaming_enabled():
                    await self._orch.event_bus.publish(
                        EventFactory.answer_retracted(
                            state.current_step,
                            "".join(pending_answer_deltas),
                            "max_tokens",
                        )
                    )
                await self._orch.event_bus.publish(
                    EventFactory.thinking_chunk(
                        state.current_step, "".join(pending_answer_deltas), buffer
                    )
                )

            agent_max_tokens = getattr(self._orch.tool_executor.agent, "max_tokens", None)

            # Trace: dump the LLM response we got back, including any accumulated
            # tool calls. Pairs with the OUT line emitted before stream_with_tools.
            logger.info(
                "[LLM_TURN] step=%d IN finish_reason=%s in_tokens=%s out_tokens=%s/max=%s total=%s cost=%s text=%s tool_calls=[%s]",
                state.current_step,
                finish_reason,
                input_tokens,
                output_tokens,
                agent_max_tokens,
                tokens_used,
                cost,
                _preview(assistant_content, 240),
                ", ".join(
                    _summarize_tool_call(tc) for tc in accumulated_tool_calls.values()
                ),
            )

            # Dedicated warning + event when the model hit max_tokens. Carries
            # enough context for postmortem (tool, json length, buffer prefix)
            # without log scraping.
            if finish_reason == "length":
                tool_names = [
                    (tc.get("function", {}) or {}).get("name") or "<unnamed>"
                    for tc in accumulated_tool_calls.values()
                ]
                # Pull truncation details from the first tool call that has them.
                # The stream_normalizer attaches _truncation_error when JSON parse
                # fails at content_block_stop; an _truncation_error-less tool call
                # means the model emitted a complete tool_use block but max_tokens
                # cut off subsequent content (or the buffer was empty).
                trunc_meta = (
                    next(
                        (
                            tc.get("_truncation_error")
                            for tc in accumulated_tool_calls.values()
                            if tc.get("_truncation_error")
                        ),
                        None,
                    )
                    or {}
                )
                accumulated_json_length = trunc_meta.get("accumulated_length")
                raw_prefix = trunc_meta.get("raw_prefix")

                logger.warning(
                    "[LLM_TRUNCATED] step=%d in_tokens=%s out_tokens=%s/max=%s "
                    "tool_calls=%s json_len=%s raw_prefix=%r "
                    "— bump max_tokens or narrow scope.",
                    state.current_step,
                    input_tokens,
                    output_tokens,
                    agent_max_tokens,
                    tool_names,
                    accumulated_json_length,
                    (raw_prefix or "")[:200],
                )

                # Stash on step.metadata so a serialized step trace contains
                # everything needed to diagnose without re-running.
                step.metadata["truncation"] = {
                    "finish_reason": finish_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "max_tokens": agent_max_tokens,
                    "tool_names": tool_names,
                    "accumulated_json_length": accumulated_json_length,
                    "raw_prefix": raw_prefix,
                }

                try:
                    await self._orch.event_bus.publish(
                        EventFactory.llm_truncated(
                            step_number=state.current_step,
                            finish_reason=finish_reason,
                            output_tokens=output_tokens,
                            max_tokens=agent_max_tokens,
                            input_tokens=input_tokens,
                            tool_names=tool_names,
                            accumulated_json_length=accumulated_json_length,
                            raw_prefix=raw_prefix,
                        )
                    )
                except Exception as _evt_err:
                    logger.debug("Failed to publish llm_truncated event: %s", _evt_err)

            # === TURN ROUTING ===
            # Tool calls present = action turn, execute and loop.
            # finish_reason == "length" = response truncated, nudge model.
            # Otherwise = answer turn, accumulated text is the final answer.

            if accumulated_tool_calls and len(accumulated_tool_calls) > 1:
                # Parallel batch path — 2+ tool_use blocks emitted by the model
                # in this assistant turn. Routes through the batch executor
                # which applies the all-or-nothing parallelism rule (parallel
                # only when every tool is parallelizable=True and not
                # require_approval; otherwise serial in input order).
                tool_names = [
                    accumulated_tool_calls[k].get("function", {}).get("name", "?")
                    for k in sorted(accumulated_tool_calls.keys())
                ]
                logger.info(
                    "[ORCH] step=%d PARALLEL_BATCH tools=%s",
                    state.current_step,
                    tool_names,
                )
                await self._orch._handle_parallel_tool_batch(
                    step=step,
                    context=context,
                    state=state,
                    accumulated_tool_calls=accumulated_tool_calls,
                    assistant_content=assistant_content,
                )
            elif accumulated_tool_calls:
                # Take first tool call (ReAct is single-action per step)
                tool_call_data = accumulated_tool_calls.get(0)
                if not tool_call_data:
                    tool_call_data = list(accumulated_tool_calls.values())[0]

                # Extract tool name, arguments, and ID from accumulated data
                step.action = tool_call_data["function"]["name"]
                tool_args = tool_call_data["function"]["arguments"]
                tool_call_id = tool_call_data["id"]

                # Pre-parse string args *before* the truncation check so a
                # JSONDecodeError gets routed through the truncation handler
                # (mirrors Anthropic, where stream_normalizer attaches
                # _truncation_error at content_block_stop).
                _preparse_tool_args_string(
                    tool_call_data, state.current_step, step.action
                )
                tool_args = tool_call_data["function"]["arguments"]

                # Detect tool_use truncation. Two signals:
                # - stream_normalizer attached _truncation_error (JSON didn't parse)
                # - finish_reason="length" with a tool call in flight
                # Either way, the args are incomplete — running schema validation
                # against them surfaces a misleading "missing required params" error.
                # Skip validation and feed the truncation back to the LLM directly.
                truncation_error = tool_call_data.get("_truncation_error")
                if truncation_error or finish_reason == "length":
                    logger.warning(
                        "Step %d - Tool '%s' truncated mid-stream "
                        "(finish_reason=%s, json_parse_failed=%s). Asking model to retry.",
                        state.current_step,
                        step.action,
                        finish_reason,
                        bool(truncation_error),
                    )
                    retry_msg = (
                        f"<tool_use_error>\n"
                        f"The tool call to '{step.action}' was truncated mid-stream by the "
                        f"model's max_tokens limit before the input JSON was complete. The "
                        f"arguments are unusable. Retry with a smaller scope (e.g. fewer rows, "
                        f"narrower date range, or split into multiple smaller calls). If a "
                        f"tool needs to emit a large data array inline, prefer summarising "
                        f"or paginating.\n"
                        f"</tool_use_error>"
                    )
                    step.error = retry_msg
                    step.observation = retry_msg
                    step.metadata["failure_kind"] = "truncation"
                    # Backfill truncation debug fields for the case where the
                    # parse failed without finish_reason=="length" (rare but
                    # possible) — the [LLM_TRUNCATED] block above only fires
                    # on finish_reason=="length".
                    if "truncation" not in step.metadata and truncation_error:
                        step.metadata["truncation"] = {
                            "finish_reason": finish_reason,
                            "tool_names": [step.action],
                            "accumulated_json_length": truncation_error.get(
                                "accumulated_length"
                            ),
                            "raw_prefix": truncation_error.get("raw_prefix"),
                            "parse_error": truncation_error.get("message"),
                        }
                    if tool_call_id:
                        tool_calls_list = list(accumulated_tool_calls.values())
                        context.messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=tool_calls_list,
                            )
                        )
                        context.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=step.observation,
                                tool_call_id=tool_call_id,
                            )
                        )
                    else:
                        if assistant_content:
                            context.messages.append(
                                Message(
                                    role=MessageRole.ASSISTANT,
                                    content=assistant_content,
                                )
                            )
                        context.messages.append(
                            Message(role=MessageRole.USER, content=step.observation)
                        )
                    # Skip the rest of this step's tool processing on truncation —
                    # outer loop will iterate, model retries with the feedback above.
                    # The finally: block at the end of this method publishes
                    # step_complete and sets execution_time.
                    return step

                # Parse arguments based on format:
                # - OpenAI: string (JSON) that needs parsing
                # - Anthropic: already a dict
                if tool_args is None:
                    logger.warning(
                        f"Step {state.current_step} - Tool '{step.action}' has None arguments "
                        "(streaming may be incomplete)"
                    )
                    step.action_input = {}
                elif isinstance(tool_args, str):
                    import json

                    # Handle empty string case
                    if not tool_args or tool_args.strip() == "":
                        logger.warning(
                            f"Step {state.current_step} - Tool '{step.action}' has empty arguments string"
                        )
                        step.action_input = {}
                    else:
                        try:
                            step.action_input = json.loads(tool_args)
                        except json.JSONDecodeError as e:
                            logger.exception(
                                f"Step {state.current_step} - Failed to parse tool arguments as JSON. "
                                f"Error: {e}. Arguments preview: {tool_args[:200]}..."
                            )
                            step.error = (
                                f"Malformed tool arguments: Invalid JSON format"
                            )
                            step.action_input = {}
                elif isinstance(tool_args, dict):
                    # Already parsed (Anthropic format)
                    step.action_input = tool_args
                else:
                    logger.error(
                        f"Step {state.current_step} - Unexpected tool_args type: {type(tool_args)}, "
                        f"value: {tool_args}"
                    )
                    step.error = f"Malformed tool call: arguments type is {type(tool_args).__name__}"
                    step.action_input = {}

                # Extract __description from action_input (LLM-generated human-readable description).
                # Format: short verb-led imperative phrase, e.g. "Search the web for Tesla news"
                # — used as a UI label and as the question text for tool approval consent.
                tool_description = None
                if isinstance(step.action_input, dict):
                    tool_description = step.action_input.pop("__description", None)

                # Validate tool name is not None or empty
                if not step.action:
                    logger.warning(
                        f"Step {state.current_step} - Malformed tool call: function name is None or empty"
                    )
                    step.error = "Malformed tool call: function name is missing"
                    # Drop tool_calls: we can't emit a valid tool_use block
                    # without a name. Append a user nudge so the conversation
                    # ends on a user turn (Anthropic rejects trailing
                    # assistant messages on models that don't support prefill).
                    response_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        tool_calls=None,
                    )
                    context.messages.append(response_message)
                    step.observation = (
                        "Previous tool call was malformed (missing function name). "
                        "Retry with a valid tool call or provide a final answer."
                    )
                    context.messages.append(
                        Message(role=MessageRole.USER, content=step.observation)
                    )
                # Validate required parameters are present
                elif step.action_input is not None:
                    # Get tool schema to check required parameters
                    tool_schema = self._orch.tool_executor.get_tool_schema(step.action)
                    required_params = tool_schema.get("parameters", {}).get(
                        "required", []
                    )

                    # Check if any required parameters are missing
                    if required_params:
                        missing_params = [
                            param
                            for param in required_params
                            if param not in step.action_input
                        ]

                        if missing_params:
                            logger.error(
                                f"Step {state.current_step} - Tool '{step.action}' missing required parameters: "
                                f"{missing_params}. Provided parameters: {list(step.action_input.keys())}"
                            )
                            step.error = _format_missing_params_error(
                                tool_name=step.action,
                                missing_params=missing_params,
                                provided_params=list(step.action_input.keys()),
                                tool_schema=tool_schema,
                            )
                            step.observation = step.error
                            step.metadata["failure_kind"] = "schema"
                            # Preserve tool_calls + append a synthetic tool_result
                            # so the LLM sees the error as a normal tool failure
                            # and the conversation ends on a user turn. Fall back
                            # to dropping tool_calls if we have no id to pair
                            # with (Anthropic rejects tool_use blocks without ids).
                            if tool_call_id:
                                tool_calls_list = list(accumulated_tool_calls.values())
                                context.messages.append(
                                    Message(
                                        role=MessageRole.ASSISTANT,
                                        content=assistant_content,
                                        tool_calls=tool_calls_list,
                                    )
                                )
                                context.messages.append(
                                    Message(
                                        role=MessageRole.TOOL,
                                        content=step.observation,
                                        tool_call_id=tool_call_id,
                                    )
                                )
                            else:
                                context.messages.append(
                                    Message(
                                        role=MessageRole.ASSISTANT,
                                        content=assistant_content,
                                        tool_calls=None,
                                    )
                                )
                                context.messages.append(
                                    Message(
                                        role=MessageRole.USER, content=step.observation
                                    )
                                )
                        else:
                            # All required parameters present, execute tool
                            # Add assistant message with both text and tool calls to context
                            tool_calls_list = list(accumulated_tool_calls.values())

                            response_message = Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=tool_calls_list,
                            )
                            context.messages.append(response_message)

                            # Execute the tool
                            await self._orch._handle_tool_action(
                                step,
                                context,
                                state,
                                tool_call_id=tool_call_id,
                                tool_description=tool_description,
                            )
                    else:
                        # No required parameters, safe to execute
                        # Add assistant message with both text and tool calls to context
                        tool_calls_list = list(accumulated_tool_calls.values())

                        response_message = Message(
                            role=MessageRole.ASSISTANT,
                            content=assistant_content,
                            tool_calls=tool_calls_list,
                        )
                        context.messages.append(response_message)

                        # Execute the tool
                        await self._orch._handle_tool_action(
                            step,
                            context,
                            state,
                            tool_call_id=tool_call_id,
                            tool_description=tool_description,
                        )
                else:
                    # action_input is None (shouldn't happen, but defensive)
                    logger.error(
                        f"Step {state.current_step} - Tool '{step.action}' has None action_input"
                    )
                    step.error = "Internal error: action_input is None"
                    step.observation = step.error
                    if tool_call_id:
                        tool_calls_list = list(accumulated_tool_calls.values())
                        context.messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=tool_calls_list,
                            )
                        )
                        context.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=step.observation,
                                tool_call_id=tool_call_id,
                            )
                        )
                    else:
                        context.messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=assistant_content,
                                tool_calls=None,
                            )
                        )
                        context.messages.append(
                            Message(role=MessageRole.USER, content=step.observation)
                        )

            # No tool calls — check if this is truly a final answer or an incomplete response
            elif finish_reason == "length":
                # Response was truncated by max_tokens — LLM was likely about to call a tool
                logger.warning(
                    f"Step {state.current_step} - Response truncated (finish_reason='length'). "
                    f"Continuing loop to give LLM another turn. Preview: {assistant_content[:200]}..."
                )
                # Append the partial response plus a user nudge so the
                # conversation ends on a user turn (Opus 4.7 and similar
                # models reject trailing assistant messages as prefill).
                # Skip the assistant append entirely when there's no content
                # to preserve, so we don't emit an empty message.
                if assistant_content:
                    context.messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=assistant_content,
                            tool_calls=list(provider_executed_calls.values()) or None,
                            metadata=provider_mcp_metadata,
                        )
                    )
                # Two different situations share finish_reason="length": a
                # long visible answer that got cut, and — on models that think
                # by default — a turn whose ENTIRE budget went to thinking, so
                # nothing visible was produced. "Continue from where you left
                # off" is wrong for the second (there is nothing to continue)
                # and invites another all-thinking turn; ask for the answer
                # directly instead.
                if assistant_content:
                    nudge = (
                        "Previous response was truncated by max_tokens. "
                        "Continue from where you left off, call the tool you intended, "
                        "or provide a final answer."
                    )
                else:
                    nudge = (
                        "Your previous turn hit max_tokens before producing any "
                        "visible output. Do not deliberate further — call the tool "
                        "you intended or write the final answer now, concisely."
                    )
                context.messages.append(Message(role=MessageRole.USER, content=nudge))
                # Don't set step.answer — loop will continue

            elif not (assistant_content or "").strip():
                # Empty turn: no tool calls, no truncation, and nothing said.
                # This is NOT an answer. Setting step.answer = "" would make
                # is_final_step true (it tests `is not None`), break the loop,
                # and hand _build_result a falsy final_answer — which then
                # emits the generic "I wasn't able to produce a complete
                # answer" fallback with no stop condition and no failure
                # metadata, so neither the user nor a dispatching parent
                # learns anything. It also short-circuits EmptyResponseCondition,
                # the safety condition that exists for exactly this: the loop
                # ended before the condition was ever consulted.
                #
                # Nudge and continue instead. EmptyResponseCondition and
                # MaxStepsCondition bound the retries, and stopping through a
                # safety condition means the run reports a real cause.
                #
                # There is nothing of the model's turn to replay here: the
                # buffer is empty by definition of this branch, and any
                # extended-thinking blocks arrived on `thinking_delta` without
                # the signature Anthropic requires to send them back. So the
                # nudge instead NAMES what the run already retrieved. Without
                # that the model re-derives its plan from the raw transcript
                # every retry, which is why empty turns cluster rather than
                # occurring singly.
                logger.warning(
                    "Step %d - LLM returned an empty response (no tool calls, "
                    "finish_reason=%s); nudging and continuing",
                    state.current_step,
                    finish_reason,
                )
                nudge = (
                    "Your last response was empty. Provide your answer from the "
                    "data you already have, or call a tool. If you cannot "
                    "complete the request, say what blocked you and what you "
                    "did retrieve."
                )
                completed = _extract_partial_results(state.steps, excerpt=0)
                if completed:
                    done = "\n".join(
                        f"- {e.get('description') or e['tool']}" for e in completed
                    )
                    nudge += (
                        "\n\nYou have already completed these calls; their results "
                        f"are in this conversation and do not need re-running:\n{done}"
                    )
                context.messages.append(
                    Message(role=MessageRole.USER, content=nudge)
                )
                # Don't set step.answer — loop will continue

            else:
                # Answer turn: no tool calls, no truncation. Buffered text
                # deltas were replayed as final_answer_chunk events above;
                # set step.answer so is_final_step returns True and the
                # main loop publishes the closing final_answer event.
                logger.debug(
                    f"Step {state.current_step} - No tool calls; treating as final answer"
                )
                context.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        tool_calls=list(provider_executed_calls.values()) or None,
                        metadata=provider_mcp_metadata,
                    )
                )
                step.answer = assistant_content

        except Exception as e:
            # The stream died mid-answer: any optimistically streamed deltas
            # were never classified, and the recovery ladder may retry this
            # turn and stream a fresh answer on top of them. Retract so every
            # consumer clears the orphaned fragment. No-op on tool-call paths
            # (the tool-call retraction already cleared the list) and on
            # approval/plan interrupts (raised during tool execution, after
            # the clear).
            if pending_answer_deltas and _optimistic_answer_streaming_enabled():
                try:
                    await self._orch.event_bus.publish(
                        EventFactory.answer_retracted(
                            state.current_step,
                            "".join(pending_answer_deltas),
                            "stream_error",
                        )
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("failed to retract answer on stream error")
            self._orch._handle_step_error(step, e, state)

        finally:
            # Carry the provider's tool-search blocks on whichever assistant
            # message this step appended. Done here, at the one point every
            # branch converges, rather than at the dozen Message(...) sites —
            # threading a new field through each is how the next branch silently
            # drops it. Bounded to messages this step appended: this finally also
            # runs on the error/truncation/empty-turn paths, where attaching to
            # an earlier message would mutate already-sent history.
            if provider_search_blocks:
                _attach_search_blocks(
                    context.messages, provider_search_blocks, messages_before_step
                )

            # TRANSFER: a dispatch_assistant call handed this turn to a
            # sub-agent. The child's answer already streamed to the user as
            # real FINAL_ANSWER_CHUNKs, so there is nothing left for this agent
            # to say — mark the step final and let the main loop's existing
            # `if step.is_final_step` branch end the run. That branch also
            # publishes the closing FINAL_ANSWER; the SSE layer drops its
            # content because the chunks already delivered it.
            #
            # Consumed here rather than inside each tool path so the
            # single-call and parallel-batch paths share ONE implementation —
            # and `pop` so a stale flag can never leak into a later step.
            pending_transfer = None
            if isinstance(getattr(context, "deps", None), dict):
                pending_transfer = context.deps.pop("pending_transfer", None)
            if pending_transfer and not step.error:
                # `is_final_step` is derived (`answer is not None`, models.py:95)
                # and has no setter — assigning it raises AttributeError, which
                # is caught upstream and silently costs the whole answer.
                # Setting `answer` IS how a step is marked final.
                step.answer = pending_transfer.get("answer") or ""
                step.metadata["transferred_to"] = pending_transfer.get("handle")
                step.metadata["answered_by_assistant_id"] = pending_transfer.get(
                    "child_assistant_id"
                )
                logger.info(
                    "[ORCH] step=%d TRANSFER handle=%s — turn handed to sub-agent, "
                    "ending run without a synthesis pass (answer=%s)",
                    state.current_step,
                    pending_transfer.get("handle"),
                    _preview(step.answer, 200),
                )

            step.execution_time = time.time() - step_start_time
            await self._orch.event_bus.publish(
                EventFactory.step_complete(state.current_step, step)
            )

        # Add observation to context if present (from tool execution)
        # NOTE: This is actually added in _handle_tool_action now with proper tool_call_id
        return step
