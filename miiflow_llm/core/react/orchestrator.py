"""Focused ReAct orchestrator with clean separation of concerns."""

import time
import logging
from typing import Optional, Dict, Any

from ..agent import RunContext

# Import observability components
try:
    from opentelemetry import trace
    from ..observability.context import set_trace_context
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
from ..message import Message, MessageRole
from .data import ReActStep, ReActResult, StopReason, REACT_SYSTEM_PROMPT
from .parser import ReActParser, ReActParsingError
from .safety import SafetyManager
from .events import EventBus, EventFactory
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


class ReActOrchestrator:
    """ReAct orchestrator with clean separation of concerns."""

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        parser: ReActParser
    ):
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.parser = parser

    async def execute(self, query: str, context: RunContext) -> ReActResult:
        """Execute ReAct reasoning with observability."""
        execution_state = ExecutionState()

        # Get tracer for observability using standard OpenTelemetry
        tracer = None
        if OBSERVABILITY_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            if context.trace_context:
                set_trace_context(context.trace_context)

        if tracer:
            with tracer.start_as_current_span(
                "react.execute",
                attributes={
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "max_steps": execution_state.max_steps
                }
            ) as span:
                return await self._execute_with_observability(query, context, execution_state, span)
        else:
            return await self._execute_without_observability(query, context, execution_state)

    async def _execute_with_observability(
        self, query: str, context: RunContext, execution_state, span
    ) -> ReActResult:
        """Execute ReAct with observability tracing."""
        try:
            self._setup_context(query, context)
            span.add_event("react_execution_start", {"query": query})

            while execution_state.is_running:
                execution_state.current_step += 1

                # Create span for each ReAct step
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(
                    f"react.step_{execution_state.current_step}",
                    attributes={
                        "step_number": execution_state.current_step,
                        "max_steps": execution_state.max_steps
                    }
                ) as step_span:
                    if await self._should_stop(execution_state):
                        span.add_event("react_early_stop", {
                            "reason": "max_steps_reached",
                            "steps_completed": execution_state.current_step - 1
                        })
                        break

                    step = await self._execute_reasoning_step(context, execution_state)
                    execution_state.steps.append(step)

                    # Add step metadata to span
                    if step.thought:
                        step_span.set_attribute("thought", step.thought[:100] + "..." if len(step.thought) > 100 else step.thought)
                    if step.action:
                        step_span.set_attribute("action", step.action)
                    step_span.set_attribute("is_final", step.is_final_step)
                    step_span.set_attribute("is_error", step.is_error_step)

                    span.add_event("react_step_complete", {
                        "step": execution_state.current_step,
                        "action": step.action if step.action else None,
                        "is_final": step.is_final_step,
                        "is_error": step.is_error_step
                    })

                    if step.is_final_step:
                        execution_state.final_answer = step.answer
                        await self._publish_final_answer_event(step, execution_state)
                        span.add_event("react_final_answer", {
                            "steps_completed": execution_state.current_step,
                            "answer_length": len(step.answer) if step.answer else 0
                        })
                        break

                    if step.is_error_step:
                        span.add_event("react_error_step", {
                            "step": execution_state.current_step,
                            "error": step.observation
                        })
                        step_span.set_attribute("error", True)
                        step_span.set_attribute("error_message", step.observation)
                        break

            result = self._build_result(execution_state)

            # Add final result metadata
            span.set_attribute("total_steps", len(execution_state.steps))
            span.set_attribute("success", execution_state.final_answer is not None)
            if result.stop_reason:
                span.set_attribute("stop_reason", result.stop_reason.value)

            span.add_event("react_execution_complete", {
                "total_steps": len(execution_state.steps),
                "success": execution_state.final_answer is not None,
                "stop_reason": result.stop_reason.value if result.stop_reason else None
            })

            return result

        except Exception as e:
            span.add_event("react_execution_error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "steps_completed": len(execution_state.steps)
            })
            span.set_attribute("error", True)
            span.set_attribute("error_message", str(e))
            raise

    async def _execute_without_observability(
        self, query: str, context: RunContext, execution_state
    ) -> ReActResult:
        """Execute ReAct without observability (fallback)."""
        try:
            self._setup_context(query, context)
            while execution_state.is_running:
                execution_state.current_step += 1
                if await self._should_stop(execution_state):
                    break
                step = await self._execute_reasoning_step(context, execution_state)
                execution_state.steps.append(step)

                if step.is_final_step:
                    execution_state.final_answer = step.answer
                    await self._publish_final_answer_event(step, execution_state)
                    break

                if step.is_error_step:
                    break

            return self._build_result(execution_state)

        except Exception as e:
            logger.error(f"ReAct execution failed: {e}", exc_info=True)
            return self._build_error_result(execution_state, e)

    def _setup_context(self, query: str, context: RunContext):
        """Setup context with system prompt and user query."""
        tools_info = self.tool_executor.build_tools_description()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools=tools_info)

        messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]
        messages.extend(context.messages)
        messages.append(Message(role=MessageRole.USER, content=query))
        context.messages = messages

    async def _should_stop(self, state: 'ExecutionState') -> bool:
        """Check safety conditions."""
        stop_condition = self.safety_manager.should_stop(state.steps, state.current_step)
        if stop_condition:
            event = EventFactory.stop_condition(
                state.current_step,
                stop_condition.get_stop_reason().value,
                stop_condition.get_description()
            )
            await self.event_bus.publish(event)
            return True
        return False

    async def _execute_reasoning_step(self, context: RunContext, state: 'ExecutionState') -> ReActStep:
        """Execute a single reasoning step - focused responsibility."""
        step = ReActStep(
            step_number=state.current_step,
            thought=""
        )

        step_start_time = time.time()

        try:
            # Publish step start event
            await self.event_bus.publish(EventFactory.step_started(state.current_step))

            # Get LLM response
            response = await self._get_llm_response(context)
            context.messages.append(response.message)

            # Parse response
            parsed = await self._parse_response(response.message.content, state.current_step)
            self._update_step_from_parsed(step, parsed, response)

            # Publish thought event
            await self.event_bus.publish(EventFactory.thought(state.current_step, step.thought))

            # Handle action or final answer
            if parsed.action_type == "tool_call":
                await self._handle_tool_action(step, context, state)
            elif parsed.action_type == "final_answer":
                step.answer = parsed.answer

        except Exception as e:
            self._handle_step_error(step, e, state)

        finally:
            step.execution_time = time.time() - step_start_time
            await self.event_bus.publish(EventFactory.step_complete(state.current_step, step))

        # Add observation to context
        if step.observation:
            context.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=f"Observation: {step.observation}"
            ))

        return step

    async def _get_llm_response(self, context: RunContext):
        """Get LLM response without tools (pure ReAct pattern)."""
        return await self.tool_executor.execute_without_tools(
            messages=context.messages
        )

    async def _parse_response(self, response_content: str, step_number: int):
        """Parse LLM response with error handling."""
        try:
            return self.parser.parse_response(response_content, step_number)
        except ReActParsingError as e:
            logger.warning(f"Step {step_number} parsing failed: {e}")
            raise

    def _update_step_from_parsed(self, step: ReActStep, parsed, response):
        """Update step with parsed data and response metadata."""
        step.thought = parsed.thought
        step.cost = getattr(response, 'cost', 0.0)
        step.tokens_used = getattr(response, 'tokens', 0)

        # Set action data if this is a tool call
        if parsed.action_type == "tool_call":
            step.action = parsed.action
            step.action_input = parsed.action_input

    async def _handle_tool_action(self, step: ReActStep, context: RunContext, state: 'ExecutionState'):
        """Handle tool action execution."""
        # step.action and step.action_input are already set from parsed data

        # Publish action events
        await self.event_bus.publish(EventFactory.action_planned(
            state.current_step, step.action, step.action_input
        ))

        await self.event_bus.publish(EventFactory.action_executing(
            state.current_step, step.action, step.action_input
        ))

        # Execute tool
        try:
            result = await self._execute_tool(step, context)

            if result.success:
                step.observation = str(result.output)
            else:
                step.error = result.error
                step.observation = f"Tool execution failed: {result.error}"

            # Update metrics
            step.cost += getattr(result, 'cost', 0.0)
            step.execution_time += result.execution_time

            # Publish observation event
            await self.event_bus.publish(EventFactory.observation(
                state.current_step, step.observation, step.action, result.success
            ))

        except Exception as e:
            step.error = f"Tool execution error: {str(e)}"
            step.observation = f"Tool '{step.action}' failed: {str(e)}"
            logger.error(f"Tool execution failed: {e}", exc_info=True)

            await self.event_bus.publish(EventFactory.observation(
                state.current_step, step.observation, step.action, False
            ))

    async def _execute_tool(self, step: ReActStep, context: RunContext):
        """Execute tool with proper context injection."""
        # Check if tool exists first
        if not self.tool_executor.has_tool(step.action):
            available_tools = self.tool_executor.list_tools()
            step.error = f"Tool '{step.action}' not found. Available: {available_tools}"
            raise Exception(step.error)

        if step.action_input is None:
            step.action_input = {}

        # Ensure action_input is a dictionary
        if not isinstance(step.action_input, dict):
            # For single-parameter tools, infer the parameter name
            tool_schema = self.tool_executor.get_tool_schema(step.action)
            params = tool_schema.get('parameters', {}).get('properties', {})
            if len(params) == 1:
                param_name = next(iter(params.keys()))
                step.action_input = {param_name: step.action_input}
            else:
                raise Exception(f"Tool '{step.action}' expects dict input but got: {step.action_input}")

        # Execute tool
        return await self.tool_executor.execute_tool(
            step.action, step.action_input, context
        )

    def _handle_step_error(self, step: ReActStep, error: Exception, state: 'ExecutionState'):
        """Handle step execution errors."""
        if isinstance(error, ReActParsingError):
            step.error = f"Parsing failed: {str(error)}"
            step.observation = "Could not parse the reasoning step. Please try again with valid JSON format."
        else:
            step.error = f"Step execution failed: {str(error)}"
            step.observation = f"An error occurred: {str(error)}"

        logger.error(f"Step {state.current_step} failed: {error}", exc_info=True)

    async def _publish_final_answer_event(self, step: ReActStep, state: 'ExecutionState'):
        """Publish final answer event."""
        await self.event_bus.publish(EventFactory.final_answer(
            state.current_step, step.answer
        ))

    def _build_result(self, state: 'ExecutionState') -> ReActResult:
        """Build successful result."""
        # Determine stop reason
        if state.final_answer:
            stop_reason = StopReason.ANSWER_COMPLETE
        else:
            stop_reason = StopReason.FORCED_STOP
            state.final_answer = self._generate_fallback_answer(state.steps)

        # Calculate totals
        total_time = time.time() - state.start_time
        total_cost = sum(step.cost for step in state.steps)
        total_tokens = sum(step.tokens_used for step in state.steps)

        return ReActResult(
            steps=state.steps,
            final_answer=state.final_answer,
            stop_reason=stop_reason,
            total_cost=total_cost,
            total_execution_time=total_time,
            total_tokens=total_tokens
        )

    def _build_error_result(self, state: 'ExecutionState', error: Exception) -> ReActResult:
        """Build error result."""
        return ReActResult(
            steps=state.steps,
            final_answer=f"Error occurred during execution: {str(error)}",
            stop_reason=StopReason.FORCED_STOP
        )

    def _generate_fallback_answer(self, steps) -> str:
        """Generate fallback answer when no explicit answer is provided."""
        if not steps:
            return "No reasoning steps were completed."

        last_step = steps[-1]
        if last_step.observation:
            return f"Based on the available information: {last_step.observation}"
        elif last_step.thought:
            return f"My reasoning: {last_step.thought}"
        else:
            return "Unable to provide a complete answer due to execution issues."


    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "agent_type": "react_orchestrator"
        }


class ExecutionState:
    """Simple state container for execution tracking."""

    def __init__(self, max_steps: int = 10):
        self.current_step = 0
        self.max_steps = max_steps
        self.steps = []
        self.start_time = time.time()
        self.is_running = True
        self.final_answer = None
