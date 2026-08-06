"""Executing the model's tool calls and turning results into observations.

The two action paths of the loop — the single-tool step and the parallel
batch — plus the shared execute seam that provisions per-step run state.
Owns result processing (visualization/media/artifact/clarification
detection, citation refs, observation recording via the recording
collaborator), the transcript pairing invariant (one assistant message with
tool_calls, then one TOOL message per call), and the approval-pause
placeholder protocol for batches. Methods were moved verbatim from
ReActOrchestrator (which keeps thin delegates); ``self._orch`` is the
orchestrator.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..message import Message, MessageRole
from .enums import ReActEventType
from .events import EventFactory
from .exceptions import PlanApprovalRequired, ToolApprovalRequired
from .models import ReActStep, ToolInvocation
from .react_events import ReActEvent
from .tool_executor import ToolCall
from .orchestrator import (
    _format_missing_params_error,
    _observation_with_citation_ref,
    _preparse_tool_args_string,
    _preview,
    _sanitize_error_message,
    visualization_observation,
)

if TYPE_CHECKING:
    from ..agent import RunContext
    from ..tools import ToolResult
    from .execution import ExecutionState
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


class ToolActionHandler:
    """The single-call and parallel-batch action paths of the loop."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    async def execute_tool(
        self, step: ReActStep, context: RunContext, state: "ExecutionState" = None
    ):
        """Execute tool with proper context injection."""
        # Tool name should already be resolved by _handle_tool_action
        # Just verify it exists (fuzzy matching was already done if needed)
        if not self._orch.tool_executor.has_tool(step.action):
            available_tools = self._orch.tool_executor.list_tools()
            step.error = f"Tool '{step.action}' not found. Available: {available_tools}"
            raise Exception(step.error)

        if step.action_input is None:
            step.action_input = {}

        # Ensure action_input is a dictionary
        if not isinstance(step.action_input, dict):
            # For single-parameter tools, infer the parameter name
            tool_schema = self._orch.tool_executor.get_tool_schema(step.action)
            params = tool_schema.get("parameters", {}).get("properties", {})
            if len(params) == 1:
                param_name = next(iter(params.keys()))
                step.action_input = {param_name: step.action_input}
            else:
                raise Exception(
                    f"Tool '{step.action}' expects dict input but got: {step.action_input}"
                )

        # Resolve media_ref:<id> references in tool inputs to actual URLs.
        # Some tools intentionally accept symbolic refs because they need to
        # resolve them against media_store themselves, for example to recover
        # the backing FileAsset before saving into a workspace.
        if isinstance(step.action_input, dict) and state:
            step.action_input = self._orch._resolve_media_refs(
                step.action_input,
                state,
                tool_name=step.action,
            )

        # Expose media_store + event_bus + step_number to tools so they can:
        #   - resolve media_ref IDs without re-implementing resolution logic
        #     (e.g. analyze_creative)
        #   - publish events back to the parent's stream during execution
        #     (e.g. dispatch_assistant streaming a sub-assistant's progress)
        # Safe in-place assignment because the orchestrator owns ctx lifecycle
        # for the duration of a ReAct run.
        if context is not None:
            if state is not None:
                context.run_state.media_store = state.media_store
                context.run_state.step_number = state.current_step
            context.run_state.event_bus = self._orch.event_bus

            # Legacy dual-write to ctx.deps (see RunState docstring). Remove
            # once every reader of these keys has been switched to
            # ctx.run_state.*.
            if isinstance(getattr(context, "deps", None), dict):
                if state is not None:
                    context.deps["media_store"] = state.media_store
                    context.deps["step_number"] = state.current_step
                context.deps["event_bus"] = self._orch.event_bus
                # This is the single-tool path, so the model emitted exactly
                # one call. Must be written explicitly, not left to default:
                # ctx.deps outlives the step, so a stale batch_size from an
                # earlier parallel batch would make dispatch_assistant refuse a
                # perfectly valid lone transfer.
                context.deps["batch_size"] = 1

        # Determine if tool needs context injection
        needs_context = self._orch.tool_executor.tool_needs_context(step.action)

        # Execute tool with or without context based on tool's requirements
        return await self._orch.tool_executor.execute_tool(
            step.action, step.action_input, context=context if needs_context else None
        )

    def handle_step_error(
        self, step: ReActStep, error: Exception, state: "ExecutionState"
    ):
        """Handle step execution errors."""
        step.error = f"Step execution failed: {str(error)}"
        step.observation = f"An error occurred: {str(error)}"
        logger.error(f"Step {state.current_step} failed: {error}", exc_info=True)

    async def handle_tool_action(
        self,
        step: ReActStep,
        context: RunContext,
        state: "ExecutionState",
        tool_call_id: Optional[str] = None,
        tool_description: Optional[str] = None,
    ):
        """Handle tool action execution."""
        # step.action and step.action_input are already set from parsed data

        # Resolve tool name BEFORE emitting events to ensure consistency
        # (fuzzy matching may correct LLM hallucinations)
        if not self._orch.tool_executor.has_tool(step.action):
            corrected_name = self._orch._find_similar_tool(step.action)
            if corrected_name:
                logger.warning(
                    f"Tool '{step.action}' not found, auto-correcting to '{corrected_name}'"
                )
                step.action = corrected_name

        # Publish action events with tool_description for human-readable display
        await self._orch.event_bus.publish(
            EventFactory.action_planned(
                state.current_step,
                step.action,
                step.action_input,
                tool_description,
                tool_call_id=tool_call_id,
            )
        )

        await self._orch.event_bus.publish(
            EventFactory.action_executing(
                state.current_step,
                step.action,
                step.action_input,
                tool_description,
                tool_call_id=tool_call_id,
            )
        )

        # Execute tool
        try:
            result = await self._orch._execute_tool(step, context, state)

            if result.success:
                # Check if this is a visualization result BEFORE stringification
                # This is critical because str(VisualizationResult) returns [VIZ:uuid]
                # which loses the actual chart data
                from miiflow_agent.visualization import (
                    is_visualization_result,
                    extract_visualization_data,
                )
                from miiflow_agent.visualization.types import (
                    is_media_result,
                    extract_media_data,
                    is_media_collection,
                    extract_media_collection,
                    extract_collection_metadata,
                    is_llm_block_injection,
                    extract_llm_blocks,
                )
                from miiflow_agent.artifacts import (
                    extract_artifact_data,
                    format_artifact_observation,
                    is_artifact_result,
                )

                if is_llm_block_injection(result.output):
                    # Tool wants the LLM to actually see pixels on the next turn.
                    # Queue the raw block dicts; the TOOL-message construction below
                    # will materialize them as multimodal content.
                    inj = extract_llm_blocks(result.output) or {}
                    blocks = inj.get("blocks") or []
                    summary = inj.get("summary") or (
                        f"Injected {len(blocks)} content block(s) for visual analysis."
                    )
                    step.observation = summary
                    state.pending_llm_blocks = list(blocks)
                    logger.info(
                        f"Step {state.current_step} - Queued {len(blocks)} LLM blocks for next turn"
                    )
                elif is_media_collection(result.output):
                    media_items = extract_media_collection(result.output) or []
                    metadata_items = extract_collection_metadata(result.output) or []
                    observation_lines = []
                    for idx, media_data in enumerate(media_items):
                        await self._orch.event_bus.publish(
                            EventFactory.media(
                                state.current_step, media_data, step.action
                            )
                        )
                        media_id = media_data["id"]
                        media_url = media_data.get("url", "")
                        if media_url and not media_url.startswith("data:"):
                            state.media_store[media_id] = media_url
                        # Correlate metadata entries by index (tools return parallel lists)
                        meta = (
                            metadata_items[idx] if idx < len(metadata_items) else None
                        )
                        if meta is not None:
                            try:
                                meta_json = json.dumps(meta, default=str)
                            except Exception:
                                meta_json = str(meta)
                            observation_lines.append(f"[MEDIA:{media_id}] {meta_json}")
                        else:
                            observation_lines.append(
                                f"[MEDIA:{media_id}] media_type={media_data.get('media_type')} "
                                f"url={media_url}"
                            )
                    step.observation = (
                        f"Returned {len(media_items)} media item(s). "
                        "Reference any of them with media_ref:<id>.\n"
                        + "\n".join(observation_lines)
                    )
                    logger.info(
                        f"Step {state.current_step} - Emitted {len(media_items)} media events (collection)"
                    )
                elif is_media_result(result.output):
                    media_data = extract_media_data(result.output)
                    await self._orch.event_bus.publish(
                        EventFactory.media(state.current_step, media_data, step.action)
                    )
                    media_id = media_data["id"]
                    media_url = media_data.get("url", "")

                    # Store media URL in execution state so subsequent tool calls
                    # (e.g. image editing) can resolve media_ref:<id> to actual URL.
                    # Only store actual URLs, not data URIs (which can be MBs of base64).
                    # System tools already persist to S3 before reaching here, so
                    # media_url should be an S3 URL for normal image gen flows.
                    if media_url and not media_url.startswith("data:"):
                        state.media_store[media_id] = media_url

                    # Always include media_ref so LLM can reference this image
                    # in subsequent tool calls (e.g. edit_gpt_image_1)
                    if media_url and not media_url.startswith("data:"):
                        step.observation = (
                            f"[MEDIA:{media_id}] Image generated successfully. "
                            f"To edit this image, use media_ref:{media_id} as the image parameter. "
                            f"Image URL: {media_url}"
                        )
                    else:
                        step.observation = (
                            f"[MEDIA:{media_id}] Image generated successfully. "
                            f"To edit this image, use media_ref:{media_id} as the image parameter."
                        )
                    logger.info(
                        f"Step {state.current_step} - Emitted media event: "
                        f"id={media_id}, type={media_data.get('media_type')}"
                    )
                elif is_visualization_result(result.output):
                    viz_data = extract_visualization_data(result.output)
                    if viz_data:
                        # Emit visualization event with full data BEFORE stringification
                        await self._orch.event_bus.publish(
                            EventFactory.visualization(
                                state.current_step, viz_data, step.action
                            )
                        )
                        # Store marker for observation (what gets sent to LLM
                        # context). auth_prompt spells out that nothing ran —
                        # see visualization_observation.
                        step.observation = visualization_observation(viz_data)
                        logger.info(
                            f"Step {state.current_step} - Emitted visualization event: "
                            f"type={viz_data.get('type')}, id={viz_data.get('id')}"
                        )
                    else:
                        # Extraction failed, fall back to string
                        step.observation = _observation_with_citation_ref(result.output)
                elif is_artifact_result(result.output):
                    artifact_data = extract_artifact_data(result.output)
                    if artifact_data:
                        await self._orch.event_bus.publish(
                            EventFactory.artifact(
                                state.current_step, artifact_data, step.action
                            )
                        )
                        step.observation = format_artifact_observation(artifact_data)
                        logger.info(
                            f"Step {state.current_step} - Emitted artifact event: "
                            f"kind={artifact_data.get('kind')}, id={artifact_data.get('id')}"
                        )
                    else:
                        step.observation = _observation_with_citation_ref(result.output)
                else:
                    step.observation = _observation_with_citation_ref(result.output)

                if await self._orch._handle_tool_approval_marker_result(
                    context,
                    state,
                    result,
                    parent_tool_call_id=tool_call_id,
                ):
                    step.observation = (
                        "Tool execution paused - waiting for user approval."
                    )

                # Check if this is a clarification request
                from ..tools.clarification import (
                    is_clarification_result,
                    extract_clarification_data,
                )

                if not state.needs_clarification and is_clarification_result(result):
                    clarification = extract_clarification_data(result)
                    if clarification:
                        # Phase 1: deterministic established-facts short-circuit (R4).
                        # Check the asked questions against facts already resolved this
                        # run (threaded in via deps["established_facts"] by the adapter —
                        # absent ⇒ behaviour identical to before). Questions whose stable
                        # key is already answered NEVER re-pause; if every question is
                        # resolved we skip the pause entirely and hand the model the
                        # known answers as the observation.
                        from ..checkpoint import EstablishedFact
                        from ..interrupt import decide_clarification

                        facts_by_key = {}
                        deps = getattr(context, "deps", None)
                        if isinstance(deps, dict):
                            for fd in deps.get("established_facts") or []:
                                try:
                                    f = EstablishedFact.from_dict(fd)
                                    facts_by_key[f.key] = f
                                except Exception:
                                    continue

                        question_dicts = [q.to_dict() for q in clarification.questions]
                        clarification_round = 0
                        if isinstance(deps, dict):
                            clarification_round = int(
                                deps.get("clarification_round", 0) or 0
                            )
                        decision = decide_clarification(
                            question_dicts,
                            facts_by_key,
                            interrupt_count=clarification_round,
                        )

                        if not decision.should_pause:
                            # Everything was already answered — do NOT pause; the model
                            # proceeds deterministically with the known answers.
                            step.observation = (
                                decision.resolved_observation or step.observation
                            )
                            logger.info(
                                "Clarification short-circuited: all question(s) already settled"
                            )
                        else:
                            state.needs_clarification = True
                            clarification_data = clarification.to_dict()
                            clarification_data["questions"] = decision.pause_questions
                            clarification_data["tool_call_id"] = tool_call_id
                            raw_clarification_output = (
                                result.output if isinstance(result.output, dict) else {}
                            )
                            for meta_key in (
                                "handle",
                                "child_assistant_id",
                                "subagent_id",
                                "status",
                                "subagent_path",
                            ):
                                if meta_key in raw_clarification_output:
                                    clarification_data[meta_key] = (
                                        raw_clarification_output[meta_key]
                                    )
                            subagent_path = raw_clarification_output.get(
                                "subagent_path"
                            )
                            raised_by_path = ["root"] + list(subagent_path or [])
                            interrupt = await self._orch._record_interrupt(
                                context,
                                state,
                                kind="clarification",
                                payload={
                                    "questions": decision.pause_questions,
                                    "context": clarification.context,
                                    **{
                                        k: v
                                        for k, v in clarification_data.items()
                                        if k
                                        in (
                                            "handle",
                                            "child_assistant_id",
                                            "subagent_id",
                                            "status",
                                            "subagent_path",
                                        )
                                    },
                                },
                                tool_call_id=tool_call_id,
                                raised_by_path=raised_by_path,
                            )
                            clarification_data["interrupt_id"] = interrupt.interrupt_id
                            clarification_data["raised_by_path"] = (
                                interrupt.raised_by_path
                            )
                            state.clarification_data = clarification_data
                            logger.info(
                                f"Clarification requested: {len(decision.pause_questions)} question(s)"
                            )

                            # Emit clarification event
                            await self._orch.event_bus.publish(
                                ReActEvent(
                                    event_type=ReActEventType.CLARIFICATION_NEEDED,
                                    step_number=state.current_step,
                                    data={
                                        "step": state.current_step,
                                        "questions": decision.pause_questions,
                                        "context": clarification.context,
                                        "tool_call_id": tool_call_id,
                                        "interrupt_id": interrupt.interrupt_id,
                                        "raised_by_path": interrupt.raised_by_path,
                                    },
                                )
                            )
            else:
                # Sanitize error message for LLM consumption
                sanitized_error = _sanitize_error_message(result.error)
                step.error = result.error  # Keep full error for debugging
                step.observation = f"Tool execution failed: {sanitized_error}"

            # Update metrics
            step.cost += getattr(result, "cost", 0.0)
            step.execution_time += result.execution_time

            # Persist canonical observation + reduce into the ledger, then
            # publish the observation event carrying the storage ref. A
            # result served from the dedupe gate reuses its existing ref —
            # no new row, no ledger freshness refresh (TTL measures the
            # DATA's age, not the last time someone asked).
            _result_meta = getattr(result, "metadata", None) or {}
            _served = bool(_result_meta.get("served_from_ledger"))
            if _served:
                observation_ref = _result_meta.get("observation_ref")
            else:
                _result_error = getattr(result, "error", None)
                recorded = await self._orch._record_tool_observation(
                    context,
                    state,
                    tool_name=step.action,
                    inputs=step.action_input,
                    observation=step.observation,
                    success=getattr(result, "success", True),
                    tool_call_id=tool_call_id,
                    raw_output=getattr(result, "output", None),
                    error=str(_result_error) if _result_error else None,
                    execution_time_ms=int(
                        (getattr(result, "execution_time", 0) or 0) * 1000
                    ),
                )
                observation_ref = recorded.ref
                # Everything downstream (the event, the TOOL message appended
                # below, step.observation on the transcript) reads this — one
                # assignment bounds them all.
                step.observation = recorded.observation
            # This path has no ToolInvocation to hold the ref (`all_invocations`
            # synthesizes one from the legacy fields), so stash it where that
            # accessor can find it. Same reason as the batch path above: a
            # force-stopped run must be able to name what it retrieved.
            if observation_ref:
                step.metadata["observation_ref"] = observation_ref
            await self._orch.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    step.observation,
                    step.action,
                    result.success,
                    tool_call_id=tool_call_id,
                    observation_ref=observation_ref,
                    served_from_ledger=_served,
                )
            )

            # Add tool result to context (required for native tool calling).
            # If the tool returned LlmBlockInjection, the orchestrator stashed
            # raw block dicts on state.pending_llm_blocks — attach them as
            # multimodal TOOL-message content so the next LLM turn sees the
            # actual pixels rather than a URL string.
            if tool_call_id:
                if state.pending_llm_blocks:
                    from ..message import ImageBlock, TextBlock, VideoBlock

                    content_blocks: List[Any] = [TextBlock(text=step.observation or "")]
                    for b in state.pending_llm_blocks:
                        btype = b.get("type")
                        if btype == "text":
                            content_blocks.append(TextBlock(text=b.get("text", "")))
                        elif btype == "image_url":
                            content_blocks.append(
                                ImageBlock(
                                    image_url=b.get("image_url", ""),
                                    detail=b.get("detail", "auto"),
                                )
                            )
                        elif btype == "video_url":
                            content_blocks.append(
                                VideoBlock(
                                    video_url=b.get("video_url", ""),
                                    mime_type=b.get("mime_type"),
                                )
                            )
                    observation_message = Message(
                        role=MessageRole.TOOL,
                        content=content_blocks,
                        tool_call_id=tool_call_id,
                    )
                    state.pending_llm_blocks = []
                else:
                    observation_message = Message(
                        role=MessageRole.TOOL,
                        content=step.observation,
                        tool_call_id=tool_call_id,
                    )
                context.messages.append(observation_message)
                logger.debug(
                    f"Step {state.current_step} - Added tool result to context with ID: {tool_call_id}"
                )

        except ToolApprovalRequired as e:
            # Tool requires user approval before execution
            # NOTE: __description was already popped from action_input at line 523
            # and stored in `tool_description`, so use that instead of re-popping
            state.needs_clarification = True  # Reuse existing pause mechanism
            state.clarification_data = {
                "type": "tool_approval",
                "tool_name": e.tool_name,
                "tool_inputs": e.tool_inputs or {},
                "tool_description": tool_description or "",
                "tool_schema": self._orch.tool_executor.get_tool_schema(e.tool_name),
                "tool_call_id": tool_call_id,
                "reason": e.reason,
            }
            interrupt = await self._orch._record_interrupt(
                context,
                state,
                kind="tool_approval",
                payload={
                    "tool_name": e.tool_name,
                    "tool_inputs": e.tool_inputs or {},
                    "tool_description": tool_description or "",
                    "tool_schema": self._orch.tool_executor.get_tool_schema(e.tool_name),
                    "reason": e.reason,
                },
                tool_call_id=tool_call_id,
            )
            state.clarification_data["interrupt_id"] = interrupt.interrupt_id
            state.clarification_data["raised_by_path"] = interrupt.raised_by_path

            # Emit approval event for SSE
            await self._orch.event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                    step_number=state.current_step,
                    data=state.clarification_data,
                )
            )

            logger.info(f"Tool '{e.tool_name}' requires approval - pausing execution")

            # CRITICAL: Must add a tool result to context, otherwise Anthropic API
            # rejects with "tool_use ids were found without tool_result blocks"
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content="Tool execution paused - waiting for user approval.",
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)

        except PlanApprovalRequired as e:
            # `exit_plan_mode` raised — pause the loop while the user
            # decides whether to approve the proposed plan. Reuses the
            # same `state.needs_clarification` pause mechanism the
            # tool-approval path uses; the streaming service
            # distinguishes them by `clarification_data["type"]`.
            state.needs_clarification = True
            state.clarification_data = {
                "type": "plan_approval",
                "plan": e.plan_text,
                "tool_call_id": tool_call_id or e.tool_call_id,
            }
            interrupt = await self._orch._record_interrupt(
                context,
                state,
                kind="plan_approval",
                payload={"plan": e.plan_text},
                tool_call_id=tool_call_id or e.tool_call_id,
            )
            state.clarification_data["interrupt_id"] = interrupt.interrupt_id
            state.clarification_data["raised_by_path"] = interrupt.raised_by_path

            await self._orch.event_bus.publish(
                ReActEvent(
                    event_type=ReActEventType.PLAN_APPROVAL_NEEDED,
                    step_number=state.current_step,
                    data=state.clarification_data,
                )
            )

            logger.info(
                "exit_plan_mode raised — pausing for plan approval "
                f"(plan length: {len(e.plan_text)} chars)"
            )

            # Same Anthropic tool_use/tool_result pairing invariant as
            # the ToolApprovalRequired path: a tool_use without a
            # matching tool_result block makes the next API call 4xx.
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content="Plan submitted — waiting for user approval.",
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)

        except Exception as e:
            # Sanitize error message for LLM consumption
            sanitized_error = _sanitize_error_message(str(e))
            step.error = (
                f"Tool execution error: {str(e)}"  # Keep full error for debugging
            )
            step.observation = f"Tool '{step.action}' failed: {sanitized_error}"
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            await self._orch.event_bus.publish(
                EventFactory.observation(
                    state.current_step,
                    step.observation,
                    step.action,
                    False,
                    tool_call_id=tool_call_id,
                )
            )

            # Add tool result to context even on exception (required for native tool calling)
            # Without this, Anthropic API will reject subsequent calls with:
            # "tool_use ids were found without tool_result blocks"
            if tool_call_id:
                observation_message = Message(
                    role=MessageRole.TOOL,
                    content=step.observation,
                    tool_call_id=tool_call_id,
                )
                context.messages.append(observation_message)
                logger.debug(
                    f"Step {state.current_step} - Added error tool result to context with ID: {tool_call_id}"
                )

    async def handle_parallel_tool_batch(
        self,
        step: ReActStep,
        context: RunContext,
        state: "ExecutionState",
        accumulated_tool_calls: Dict[int, Dict[str, Any]],
        assistant_content: str,
    ) -> None:
        """Execute 2+ tool_use blocks the model emitted in one assistant turn.

        Parses each tool call's args, validates schema, appends ONE
        ASSISTANT message preserving Anthropic's tool_use/tool_result
        pairing invariant, dispatches the batch through
        ``executor.execute_many`` (which applies the all-or-nothing
        parallelism rule), and appends N TOOL messages with each
        observation paired by ``tool_call_id``.

        Per-invocation results are stored on ``step.tool_invocations``.
        For back-compat with single-action consumers, ``step.action``,
        ``step.action_input``, and ``step.observation`` mirror the FIRST
        invocation's fields. Per-call errors land on
        ``step.tool_invocations[i].error``; ``step.error`` is set only
        for step-level failures (e.g. malformed transcript), not for
        individual tool failures (those still surface as observations
        the LLM can react to).

        Rich result types (visualizations, artifacts, media collections,
        LLM block injections) get FULL observation processing when the
        executor's serial fallback fires (mixed-parallelizability or
        approval-required batches). When the batch runs in true parallel
        mode (every tool is ``parallelizable=True``), only visualization
        markers + simple stringification are handled — by construction
        parallelizable tools shouldn't return media/artifact/llm-block
        results, so this is a documented v1 trade-off rather than a
        correctness gap.
        """
        from ..tools import ToolResult

        # ── Phase 1: parse, validate, and build invocations ─────────────
        ordered_keys = sorted(accumulated_tool_calls.keys())
        invocations: List[ToolInvocation] = []
        tool_call_dicts: List[Dict[str, Any]] = []  # for the ASSISTANT message
        pre_exec_errors: Dict[str, str] = {}  # tool_call_id -> error string

        for key in ordered_keys:
            tool_call_data = accumulated_tool_calls[key]
            name = tool_call_data.get("function", {}).get("name")
            raw_args = tool_call_data.get("function", {}).get("arguments")
            tool_call_id = (
                tool_call_data.get("id") or f"call_{state.current_step}_{key}"
            )

            # Parse args (OpenAI sends string; Anthropic sends dict)
            _preparse_tool_args_string(tool_call_data, state.current_step, name)
            raw_args = tool_call_data.get("function", {}).get("arguments")

            inputs: Optional[Dict[str, Any]]
            if raw_args is None:
                inputs = {}
            elif isinstance(raw_args, str):
                if not raw_args or raw_args.strip() == "":
                    inputs = {}
                else:
                    try:
                        inputs = json.loads(raw_args)
                    except json.JSONDecodeError:
                        inputs = {}
                        pre_exec_errors[tool_call_id] = (
                            f"Malformed tool arguments for '{name}': invalid JSON"
                        )
            elif isinstance(raw_args, dict):
                inputs = raw_args
            else:
                inputs = {}
                pre_exec_errors[tool_call_id] = (
                    f"Malformed tool arguments for '{name}': "
                    f"unexpected type {type(raw_args).__name__}"
                )

            # Extract __description for UI labels
            description = None
            if isinstance(inputs, dict):
                description = inputs.pop("__description", None)

            # Truncation / malformed name guards (mirror single-tool path)
            if tool_call_data.get("_truncation_error"):
                pre_exec_errors[tool_call_id] = (
                    f"Tool call to '{name}' was truncated mid-stream. "
                    "Retry with a narrower scope or split into smaller calls."
                )
            if not name:
                pre_exec_errors[tool_call_id] = (
                    "Malformed tool call: function name is missing."
                )

            # Schema validation — required-params check (mirrors single-tool path)
            if name and tool_call_id not in pre_exec_errors and inputs is not None:
                try:
                    tool_schema = self._orch.tool_executor.get_tool_schema(name)
                    required_params = (
                        tool_schema.get("parameters", {}).get("required") or []
                    )
                    if required_params:
                        missing = [p for p in required_params if p not in inputs]
                        if missing:
                            pre_exec_errors[tool_call_id] = (
                                _format_missing_params_error(
                                    tool_name=name,
                                    missing_params=missing,
                                    provided_params=list(inputs.keys()),
                                    tool_schema=tool_schema,
                                )
                            )
                except Exception as exc:
                    # Schema lookup failure is non-fatal; let the
                    # executor surface the real error.
                    logger.debug(
                        "Step %d - Schema lookup failed for '%s': %s",
                        state.current_step,
                        name,
                        exc,
                    )

            invocations.append(
                ToolInvocation(
                    tool_call_id=tool_call_id,
                    name=name,
                    inputs=inputs or {},
                    description=description,
                )
            )
            tool_call_dicts.append(tool_call_data)

        # ── Phase 2: append the assistant message with ALL tool_calls ──
        # This satisfies the Anthropic invariant that every tool_use block
        # in an assistant message must be followed (across messages) by a
        # matching tool_result keyed on the same tool_call_id.
        context.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                tool_calls=tool_call_dicts,
            )
        )

        # ── Phase 3: execute the batch ─────────────────────────────────
        # Tools that pre-failed (schema/truncation/missing-name) get a
        # synthetic error observation and don't get sent to the executor.
        # Successfully-parsed tools go through execute_many, which decides
        # parallel vs serial based on the all-or-nothing rule.
        runnable: List[ToolCall] = []
        runnable_indices: List[int] = []
        for i, inv in enumerate(invocations):
            if inv.tool_call_id in pre_exec_errors or not inv.name:
                continue
            if isinstance(inv.inputs, dict):
                inv.inputs = self._orch._resolve_media_refs(
                    inv.inputs,
                    state,
                    tool_name=inv.name,
                )
            runnable.append(
                ToolCall(
                    tool_call_id=inv.tool_call_id, name=inv.name, inputs=inv.inputs
                )
            )
            runnable_indices.append(i)

        # Publish action_planned events for visibility (one per invocation).
        for inv in invocations:
            if inv.name:
                try:
                    await self._orch.event_bus.publish(
                        EventFactory.action_planned(
                            state.current_step,
                            inv.name,
                            inv.inputs,
                            inv.description,
                            tool_call_id=inv.tool_call_id,
                        )
                    )
                except Exception as evt_err:
                    logger.debug("Failed to publish action_planned: %s", evt_err)

        results: List[Optional[ToolResult]] = [None] * len(invocations)
        # Set when an approval-required tool in the batch trips the approval
        # gate. ``execute_many`` forces a batch serial when any tool is
        # ``require_approval=True`` and raises ``ToolApprovalRequired`` at that
        # tool. We catch it here (the single-tool path catches it too, but the
        # batch path previously let it escape to the generic step-error handler,
        # which routed an unsatisfiable approval through the recovery ladder AND
        # left the just-appended tool_use blocks without matching tool_results —
        # the next LLM call then 400s with "tool_use ids were found without
        # tool_result blocks"). Pausing here keeps the transcript valid.
        approval_pause: Optional[ToolApprovalRequired] = None
        if runnable:
            # Mirror the per-step ctx.run_state injection that _execute_tool
            # does on the single-tool path. Without this, tools running in a
            # parallel batch (notably dispatch_assistant) see stale or absent
            # step_number / media_store. Each parallel branch then gets its
            # OWN Context copy via _execute_parallel's create_task, so this
            # write is visible to all branches at the moment of fan-out.
            if context is not None:
                context.run_state.media_store = state.media_store
                context.run_state.step_number = state.current_step
                context.run_state.event_bus = self._orch.event_bus
                # Legacy dual-write — remove once readers move to run_state.
                if isinstance(getattr(context, "deps", None), dict):
                    context.deps["media_store"] = state.media_store
                    context.deps["step_number"] = state.current_step
                    context.deps["event_bus"] = self._orch.event_bus
                    # How many tool calls the model emitted in this one
                    # assistant message. dispatch_assistant reads it to refuse
                    # a transfer that isn't the lone call in the turn — handing
                    # the turn away while other tools are still running would
                    # strand their results.
                    context.deps["batch_size"] = len(invocations)

            # Provide context only if any tool needs it (single-tool path
            # checks per-tool; here we pass it always — execute_tool
            # internally handles the per-tool needs_context check).
            try:
                batch_results = await self._orch.tool_executor.execute_many(
                    runnable, context=context
                )
                for runnable_pos, idx in enumerate(runnable_indices):
                    results[idx] = batch_results[runnable_pos]
            except ToolApprovalRequired as e:
                approval_pause = e
                logger.info(
                    "Parallel batch contains approval-required tool '%s' - "
                    "pausing execution for user approval",
                    e.tool_name,
                )

        # ── Phase 4: process results into observations ─────────────────
        # Per-invocation observation handling: basic stringification +
        # visualization marker detection. More exotic result types
        # (media/artifact/llm_block_injection) fall through to str() in
        # batch mode — see method docstring for the v1 trade-off.
        from miiflow_agent.visualization import (
            extract_visualization_data,
            is_visualization_result,
        )

        if approval_pause is not None:
            # An approval-required tool tripped the gate. Pause the run (mirrors
            # the single-tool ToolApprovalRequired handler) and give EVERY
            # invocation a placeholder observation so Phase 5 can pair each
            # tool_use block with a tool_result. ``execute_many`` raised before
            # returning, so no tool in the batch actually ran.
            approval_inv = next(
                (iv for iv in invocations if iv.name == approval_pause.tool_name),
                None,
            )
            state.needs_clarification = True
            state.clarification_data = {
                "type": "tool_approval",
                "tool_name": approval_pause.tool_name,
                "tool_inputs": approval_pause.tool_inputs
                or (approval_inv.inputs if approval_inv else {}),
                "tool_description": (approval_inv.description if approval_inv else "")
                or "",
                "tool_schema": self._orch.tool_executor.get_tool_schema(
                    approval_pause.tool_name
                ),
                "tool_call_id": approval_inv.tool_call_id if approval_inv else None,
                "reason": approval_pause.reason,
            }
            interrupt = await self._orch._record_interrupt(
                context,
                state,
                kind="tool_approval",
                payload={
                    "tool_name": approval_pause.tool_name,
                    "tool_inputs": approval_pause.tool_inputs
                    or (approval_inv.inputs if approval_inv else {}),
                    "tool_description": (
                        approval_inv.description if approval_inv else ""
                    )
                    or "",
                    "tool_schema": self._orch.tool_executor.get_tool_schema(
                        approval_pause.tool_name
                    ),
                    "reason": approval_pause.reason,
                },
                tool_call_id=approval_inv.tool_call_id if approval_inv else None,
            )
            state.clarification_data["interrupt_id"] = interrupt.interrupt_id
            state.clarification_data["raised_by_path"] = interrupt.raised_by_path
            try:
                await self._orch.event_bus.publish(
                    ReActEvent(
                        event_type=ReActEventType.TOOL_APPROVAL_NEEDED,
                        step_number=state.current_step,
                        data=state.clarification_data,
                    )
                )
            except Exception as evt_err:
                logger.debug("Failed to publish approval event: %s", evt_err)
            for inv in invocations:
                if inv.name == approval_pause.tool_name:
                    inv.observation = (
                        "Tool execution paused - waiting for user approval."
                    )
                else:
                    inv.observation = (
                        f"Not executed - batch paused pending approval of "
                        f"'{approval_pause.tool_name}'."
                    )
            # Skip normal result processing; fall through to Phase 5 which
            # appends one TOOL message per invocation (pairing invariant).
            invocations_to_process: List[ToolInvocation] = []
        else:
            invocations_to_process = invocations

        for i, inv in enumerate(invocations_to_process):
            # Pre-execution error case
            if inv.tool_call_id in pre_exec_errors:
                err = pre_exec_errors[inv.tool_call_id]
                inv.error = err
                inv.observation = err
                continue

            result = results[i]
            if result is None:
                inv.error = "Tool execution returned no result"
                inv.observation = inv.error
                continue

            if not result.success:
                sanitized = _sanitize_error_message(result.error or "Unknown error")
                inv.error = result.error
                inv.observation = f"Tool execution failed: {sanitized}"
                # Propagate the validation marker the registry stamped from the
                # raised exception's `is_tool_validation_error` attribute. Used
                # below to classify an all-failed step as schema-kind so the
                # recovery_manager skips the runtime ladder.
                inv.is_validation_error = bool(
                    (result.metadata or {}).get("is_validation_error")
                )
            else:
                # Visualization → emit event + use [VIZ:id] marker
                if is_visualization_result(result.output):
                    viz_data = extract_visualization_data(result.output)
                    if viz_data:
                        try:
                            await self._orch.event_bus.publish(
                                EventFactory.visualization(
                                    state.current_step, viz_data, inv.name
                                )
                            )
                        except Exception as evt_err:
                            logger.debug(
                                "Failed to publish visualization event: %s", evt_err
                            )
                        inv.observation = visualization_observation(viz_data)
                    else:
                        inv.observation = _observation_with_citation_ref(result.output)
                else:
                    inv.observation = _observation_with_citation_ref(result.output)

            if await self._orch._handle_tool_approval_marker_result(
                context,
                state,
                result,
                parent_tool_call_id=inv.tool_call_id,
            ):
                inv.observation = "Tool execution paused - waiting for user approval."

            # Per-invocation canonical record + observation event. Served
            # results reuse their existing ref (no new row, no TTL refresh).
            _result_meta = getattr(result, "metadata", None) or {}
            _served = bool(_result_meta.get("served_from_ledger"))
            if _served:
                observation_ref = _result_meta.get("observation_ref")
            else:
                recorded = await self._orch._record_tool_observation(
                    context,
                    state,
                    tool_name=inv.name,
                    inputs=inv.inputs,
                    observation=inv.observation,
                    success=inv.error is None,
                    tool_call_id=inv.tool_call_id,
                    raw_output=getattr(result, "output", None),
                    error=str(inv.error) if inv.error else None,
                    execution_time_ms=int((getattr(result, "execution_time", 0) or 0) * 1000),
                )
                observation_ref = recorded.ref
                # Phase 5 appends `inv.observation` as this call's TOOL
                # message — write the bounded form back so N parallel calls
                # can't each paste an unbounded result into one request.
                inv.observation = recorded.observation
            # Keep the ref on the invocation, not just on the event: a run
            # that force-stops still has to hand its caller a pointer to what
            # it retrieved, and the event stream is not readable from there.
            inv.observation_ref = observation_ref
            try:
                await self._orch.event_bus.publish(
                    EventFactory.observation(
                        state.current_step,
                        inv.observation,
                        inv.name,
                        inv.error is None,
                        tool_call_id=inv.tool_call_id,
                        observation_ref=observation_ref,
                        served_from_ledger=_served,
                    )
                )
            except Exception as evt_err:
                logger.debug("Failed to publish observation event: %s", evt_err)

        # ── Phase 5: append N TOOL messages, paired by tool_call_id ────
        for inv in invocations:
            context.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=inv.observation or "",
                    tool_call_id=inv.tool_call_id,
                )
            )

        # ── Phase 6: write to step (canonical + back-compat) ───────────
        step.tool_invocations = invocations
        # Back-compat: mirror the first invocation's fields on legacy
        # singular attributes so consumers that haven't migrated still
        # observe sensible values.
        first = invocations[0] if invocations else None
        if first is not None:
            step.action = first.name
            step.action_input = first.inputs
            step.observation = first.observation
            # step.error is set ONLY if every invocation failed — this
            # signals "this whole step failed" to recovery_manager.
            # Per-invocation failures stay on each invocation.error and
            # don't trigger step-level recovery (the LLM sees the errors
            # as tool observations and can react).
            if all(inv.error is not None for inv in invocations):
                step.error = (
                    f"All {len(invocations)} parallel tool calls failed. "
                    f"First error: {first.error}"
                )
                # If every failure was a deterministic input-shape rejection
                # (e.g. GAQL preflight), classify as schema so recovery_manager
                # short-circuits: the per-invocation tool observations already
                # carry the corrective hint the LLM needs to fix its next call.
                # Without this, two parallel preflight failures would burn the
                # 3-attempt recovery ladder and force a fallback answer.
                if all(
                    getattr(inv, "is_validation_error", False) for inv in invocations
                ):
                    step.metadata["failure_kind"] = "schema"
                else:
                    step.metadata["failure_kind"] = "all_failed"
