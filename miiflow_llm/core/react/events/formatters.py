"""Professional event formatters for production systems."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator
from enum import Enum

from ..data import ReActEvent, ReActEventType


class LogLevel(Enum):
    """Log levels for different event types."""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"


class EventFormatter(ABC):
    """Base formatter interface."""
    
    @abstractmethod
    def format(self, event: ReActEvent) -> str:
        """Format event to string."""
        pass


class StructuredLogger:
    """Production-grade structured logging for ReAct events."""
    
    def __init__(self, logger_name: str = "react.execution"):
        self.logger = logging.getLogger(logger_name)
        self._event_levels = {
            ReActEventType.STEP_START: LogLevel.DEBUG,
            ReActEventType.THOUGHT: LogLevel.DEBUG,
            ReActEventType.ACTION_PLANNED: LogLevel.INFO,
            ReActEventType.ACTION_EXECUTING: LogLevel.DEBUG,
            ReActEventType.OBSERVATION: LogLevel.INFO,
            ReActEventType.STEP_COMPLETE: LogLevel.DEBUG,
            ReActEventType.FINAL_ANSWER: LogLevel.INFO,
            ReActEventType.ERROR: LogLevel.ERROR,
            ReActEventType.STOP_CONDITION: LogLevel.WARNING
        }
    
    def log_event(self, event: ReActEvent):
        """Log event with structured data."""
        level = self._event_levels.get(event.event_type, LogLevel.INFO)
        
        extra_data = {
            'event_type': event.event_type.value,
            'step_number': event.step_number,
            'session_id': event.session_id,
            'trace_id': event.trace_id,
            **event.data
        }
        
        message = self._get_message(event)
        
        getattr(self.logger, level.value)(message, extra=extra_data)
    
    def _get_message(self, event: ReActEvent) -> str:
        """Get concise, professional message for event."""
        if event.event_type == ReActEventType.ACTION_PLANNED:
            action = event.data.get("action", "unknown")
            return f"Planned tool execution: {action}"
        
        elif event.event_type == ReActEventType.OBSERVATION:
            action = event.data.get("action", "unknown")
            success = event.data.get("success", True)
            status = "completed" if success else "failed"
            return f"Tool execution {status}: {action}"
        
        elif event.event_type == ReActEventType.FINAL_ANSWER:
            return "ReAct execution completed"
        
        elif event.event_type == ReActEventType.ERROR:
            error_type = event.data.get("error_type", "UnknownError")
            return f"Execution error: {error_type}"
        
        elif event.event_type == ReActEventType.STOP_CONDITION:
            reason = event.data.get("stop_reason", "unknown")
            return f"Execution stopped: {reason}"
        
        else:
            return f"Step {event.step_number}: {event.event_type.value}"


class JSONFormatter(EventFormatter):
    """Structured JSON output for logging systems."""
    
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata
    
    def format(self, event: ReActEvent) -> str:
        """Format as clean JSON."""
        data = {
            "event_type": event.event_type.value,
            "step": event.step_number,
            "data": event.data
        }
        
        if self.include_metadata:
            data.update({
                "timestamp": event.timestamp,
                "session_id": event.session_id,
                "trace_id": event.trace_id
            })
        
        return json.dumps(data, separators=(',', ':'))


class SSEFormatter(EventFormatter):
    """Server-Sent Events for web streaming."""
    
    def format(self, event: ReActEvent) -> str:
        """Format as SSE data."""
        json_formatter = JSONFormatter(include_metadata=True)
        data = json_formatter.format(event)
        return f"data: {data}\n\n"


class CompactFormatter(EventFormatter):
    """Minimal text format for debugging."""
    
    def format(self, event: ReActEvent) -> str:
        """Format as compact text."""
        if event.event_type == ReActEventType.ACTION_PLANNED:
            action = event.data.get("action", "?")
            return f"[{event.step_number}] -> {action}"
        
        elif event.event_type == ReActEventType.OBSERVATION:
            action = event.data.get("action", "?")
            success = "✓" if event.data.get("success", True) else "✗"
            return f"[{event.step_number}] {success} {action}"
        
        elif event.event_type == ReActEventType.FINAL_ANSWER:
            return f"[{event.step_number}] DONE"
        
        else:
            return f"[{event.step_number}] {event.event_type.value}"


class EventProcessor:
    """Process events through multiple formatters."""
    
    def __init__(self):
        self.formatters: Dict[str, EventFormatter] = {}
        self.handlers: Dict[str, callable] = {}
    
    def add_formatter(self, name: str, formatter: EventFormatter, handler: Optional[callable] = None):
        """Add formatter with optional custom handler."""
        self.formatters[name] = formatter
        if handler:
            self.handlers[name] = handler
    
    def process(self, event: ReActEvent):
        """Process event through all formatters."""
        for name, formatter in self.formatters.items():
            try:
                formatted = formatter.format(event)
                if name in self.handlers:
                    self.handlers[name](formatted)
                else:
                    print(formatted)  # Default handler
            except Exception as e:
                logging.error(f"Formatter {name} failed: {e}")
