"""Thread-safe shared state for multi-agent coordination."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SharedAgentState:
    """Thread-safe shared state for multi-agent coordination.

    Following Google ADK pattern: each agent writes to unique keys to prevent
    race conditions. The state acts as a shared whiteboard for inter-agent
    communication.

    Usage:
        state = SharedAgentState()

        # Agent 1 writes its results
        await state.write("researcher_results", {...})

        # Agent 2 writes its results
        await state.write("analyzer_results", {...})

        # Lead agent reads all results
        snapshot = state.snapshot()
    """

    _state: Dict[str, Any] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _write_log: List[Dict[str, Any]] = field(default_factory=list)

    async def write(self, key: str, value: Any, agent_id: Optional[str] = None) -> None:
        """Write a value to shared state.

        Args:
            key: Unique key for this value (should be unique per agent)
            value: Value to store
            agent_id: Optional agent identifier for logging
        """
        async with self._lock:
            self._state[key] = value
            self._write_log.append({
                "key": key,
                "agent_id": agent_id,
                "value_type": type(value).__name__,
            })
            logger.debug(f"SharedState: Agent '{agent_id}' wrote to key '{key}'")

    async def read(self, key: str, default: Any = None) -> Any:
        """Read a value from shared state.

        Args:
            key: Key to read
            default: Default value if key not found

        Returns:
            Value at key, or default if not found
        """
        async with self._lock:
            return self._state.get(key, default)

    async def read_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Read multiple values from shared state atomically.

        Args:
            keys: List of keys to read

        Returns:
            Dictionary of key -> value for requested keys
        """
        async with self._lock:
            return {key: self._state.get(key) for key in keys}

    def snapshot(self) -> Dict[str, Any]:
        """Get an immutable snapshot of current state.

        This is NOT async-safe - use for synthesis phase after all agents complete.

        Returns:
            Copy of current state dictionary
        """
        return dict(self._state)

    async def snapshot_async(self) -> Dict[str, Any]:
        """Get an async-safe immutable snapshot of current state.

        Returns:
            Copy of current state dictionary
        """
        async with self._lock:
            return dict(self._state)

    def get_write_log(self) -> List[Dict[str, Any]]:
        """Get log of all writes for debugging.

        Returns:
            List of write operations with key, agent_id, and value_type
        """
        return list(self._write_log)

    async def clear(self) -> None:
        """Clear all state. Use between runs."""
        async with self._lock:
            self._state.clear()
            self._write_log.clear()

    @property
    def keys(self) -> List[str]:
        """Get all current keys."""
        return list(self._state.keys())

    def __len__(self) -> int:
        """Return number of entries in state."""
        return len(self._state)
