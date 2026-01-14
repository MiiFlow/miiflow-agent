"""Simple factory for ReAct components."""

from typing import Optional, Literal

from .events import EventBus, EventFormat
from .orchestrator import ReActOrchestrator
from .parser import ReActParser
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor


class ReActFactory:
    """Simple factory for creating ReAct orchestrators."""

    @staticmethod
    def create_orchestrator(
        agent,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        event_format: EventFormat = "react",
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> ReActOrchestrator:
        """Create ReAct orchestrator with clean dependency injection.

        Args:
            agent: The agent instance
            max_steps: Maximum number of reasoning steps
            max_budget: Optional budget limit
            max_time_seconds: Optional time limit in seconds
            event_format: Event format - "react" for legacy, "agui" for AG-UI protocol
            thread_id: Thread ID (required for agui format)
            message_id: Message ID (required for agui format)

        Returns:
            ReActOrchestrator instance
        """
        return ReActOrchestrator(
            tool_executor=AgentToolExecutor(agent),
            event_bus=EventBus(
                event_format=event_format,
                thread_id=thread_id,
                message_id=message_id,
            ),
            safety_manager=SafetyManager(
                max_steps=max_steps, max_budget=max_budget, max_time_seconds=max_time_seconds
            ),
            parser=ReActParser(),
        )
