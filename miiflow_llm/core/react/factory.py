"""Simple factory for ReAct components."""

from typing import Optional

from .events import EventBus
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
        use_native_tools: bool = False,
    ) -> ReActOrchestrator:
        """Create ReAct orchestrator with clean dependency injection.

        Args:
            agent: The agent instance
            max_steps: Maximum number of reasoning steps
            max_budget: Optional budget limit
            max_time_seconds: Optional time limit in seconds
            use_native_tools: If True, use native provider tool calling instead of XML parsing

        Returns:
            ReActOrchestrator instance
        """
        return ReActOrchestrator(
            tool_executor=AgentToolExecutor(agent),
            event_bus=EventBus(),
            safety_manager=SafetyManager(
                max_steps=max_steps, max_budget=max_budget, max_time_seconds=max_time_seconds
            ),
            parser=ReActParser(),
            use_native_tools=use_native_tools,
        )
