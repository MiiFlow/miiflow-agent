"""Minimal context system for agent execution."""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """Simple context container for agent execution metadata only."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set metadata value."""
        self.metadata[key] = value
