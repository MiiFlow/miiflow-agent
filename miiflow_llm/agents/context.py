"""Minimal context system for agent execution."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """Context container for agent execution."""
    
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set metadata value."""
        self.metadata[key] = value
