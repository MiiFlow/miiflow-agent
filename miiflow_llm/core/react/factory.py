"""Simple factory for ReAct components."""

from typing import Optional
from .orchestrator import ReActOrchestrator
from .safety import SafetyManager
from .parser import ReActParser
from .events import EventBus
from .tool_executor import AgentToolExecutor


class ReActFactory:
    """Simple factory for creating ReAct orchestrators."""
    
    @staticmethod
    def create_orchestrator(
        agent,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None
    ) -> ReActOrchestrator:
        """Create ReAct orchestrator with clean dependency injection."""
        return ReActOrchestrator(
            tool_executor=AgentToolExecutor(agent),
            event_bus=EventBus(),
            safety_manager=SafetyManager(
                max_steps=max_steps,
                max_budget=max_budget,
                max_time_seconds=max_time_seconds
            ),
            parser=ReActParser()
        )
