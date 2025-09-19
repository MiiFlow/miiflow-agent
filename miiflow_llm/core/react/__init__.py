"""ReAct (Reasoning + Acting) - Architecture

Usage:
    from miiflow_llm.core.react import ReActOrchestrator, ReActFactory
    
    orchestrator = ReActFactory.create_orchestrator(agent, max_steps=10)
    result = await orchestrator.execute("Find today's top news", context)
"""

# New clean architecture - no legacy imports
from .orchestrator import ReActOrchestrator
from .factory import ReActFactory
from .events import EventBus, EventFactory

# Core data structures (still needed)
from .data import ReActStep, ReActResult, ReActEvent, ReActEventType
from .parser import ReActParser, ReActParsingError
from .safety import StopCondition, StopReason, SafetyManager

__all__ = [
    # Main interfaces
    "ReActOrchestrator",  
    "ReActFactory",       
    
    # Clean event system
    "EventBus",           
    "EventFactory",       
    # Core data structures
    "ReActStep", 
    "ReActResult",
    "ReActEvent",
    "ReActEventType",
    
    "ReActParser",
    "ReActParsingError", 
    "StopCondition",
    "StopReason",
    "SafetyManager",
]

__version__ = "0.2.0"  # New clean architecture version
