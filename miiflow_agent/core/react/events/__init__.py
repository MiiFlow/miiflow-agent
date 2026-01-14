"""Clean event system for ReAct - single publish method.

Supports both legacy ReAct events and AG-UI protocol events.
"""

from .bus import EventBus, EventFactory, EventFormat
from .formatters import StructuredLogger, JSONFormatter, SSEFormatter, CompactFormatter, EventProcessor

# Optional AG-UI support
try:
    from .agui_factory import AGUIEventFactory, AGUI_AVAILABLE
except ImportError:
    AGUIEventFactory = None
    AGUI_AVAILABLE = False

__all__ = [
    # Core event system
    "EventBus",
    "EventFactory",
    "EventFormat",
    # Formatters
    "StructuredLogger",
    "JSONFormatter",
    "SSEFormatter",
    "CompactFormatter",
    "EventProcessor",
    # Optional AG-UI
    "AGUIEventFactory",
    "AGUI_AVAILABLE",
]
