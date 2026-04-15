"""
Type definitions for the visualization system.

This module provides generic dataclasses for defining visualization results.
The SDK does NOT define specific visualization types - that's an application concern.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class VisualizationConfig:
    """
    Generic configuration options for visualizations.

    This is a flexible container for configuration settings that
    gets passed through to the frontend. The SDK does not prescribe
    specific config options - applications define what configs their
    frontend supports.

    Usage:
        config = VisualizationConfig(
            height=400,
            colors=["#3B82F6", "#10B981"],
            custom_option="value",
        )

    Note: All kwargs are stored and serialized. Common options include:
        - height: int - Height in pixels
        - width: str - Width (e.g., "100%", "500px")
        - colors: List[str] - Custom color palette
        - And any visualization-specific options defined by the application
    """

    def __init__(self, **kwargs: Any):
        """Initialize with arbitrary configuration options."""
        self._config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        # Convert snake_case to camelCase for JavaScript convention
        result = {}
        for key, value in self._config.items():
            if value is not None:
                # Convert snake_case to camelCase
                parts = key.split("_")
                camel_key = parts[0] + "".join(p.capitalize() for p in parts[1:])
                result[camel_key] = value
        return result

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to config values."""
        if name.startswith("_"):
            raise AttributeError(name)
        return self._config.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute setting of config values."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._config[name] = value


@dataclass
class VisualizationResult:
    """
    Result object returned by visualization tools.

    This is the marker object that tells the streaming handler to emit
    a visualization event to the frontend. The SDK provides this as a
    generic container - specific visualization types are defined by
    the application.

    Attributes:
        type: The visualization type (application-defined, e.g., "chart", "table")
        data: The visualization data (structure depends on type)
        title: Optional title for the visualization
        description: Optional description text
        config: Optional configuration settings
        id: Unique identifier for the visualization (auto-generated if not provided)

    Example:
        # Create a visualization (type and data structure defined by application)
        result = VisualizationResult(
            type="chart",
            data={
                "chartType": "line",
                "series": [
                    {"name": "Revenue", "data": [{"x": "Jan", "y": 100}]}
                ],
            },
            title="Monthly Revenue",
        )

        # The result is serialized and detected by the streaming handler
        # which emits a visualization SSE event to the frontend
    """
    type: str  # Application-defined type (e.g., "chart", "table", "custom_viz")
    data: Dict[str, Any]
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[VisualizationConfig] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        The __visualization__ key is a marker that tells the streaming handler
        this is a visualization result that should be emitted as a separate event.
        """
        return {
            "__visualization__": True,  # Marker for streaming handler detection
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "data": self.data,
            "config": self.config.to_dict() if self.config else {},
        }

    def __str__(self) -> str:
        """String representation for tool observation output."""
        # Return marker in text stream for positioning
        # The streaming handler will replace this with the actual visualization
        return f"[VIZ:{self.id}]"

    def __repr__(self) -> str:
        return f"VisualizationResult(type={self.type!r}, id={self.id!r}, title={self.title!r})"


@dataclass
class MediaResult:
    """Result containing media (image/video/audio) that should render inline."""
    url: str  # URL or data URI of the media
    media_type: str = "image"  # "image", "video", "audio"
    alt_text: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "__media__": True,
            "id": self.id,
            "url": self.url,
            "media_type": self.media_type,
            "alt_text": self.alt_text,
        }

    def __str__(self) -> str:
        """Return marker for observation text (what gets sent to LLM context)."""
        return f"[MEDIA:{self.id}]"

    def __repr__(self) -> str:
        return f"MediaResult(media_type={self.media_type!r}, id={self.id!r}, alt_text={self.alt_text!r})"


def is_media_result(value: Any) -> bool:
    """Check if a value is a MediaResult."""
    if isinstance(value, MediaResult):
        return True
    if isinstance(value, dict):
        return value.get("__media__") is True
    return False


def extract_media_data(value: Any) -> Optional[Dict[str, Any]]:
    """Extract media data from a MediaResult or its dict representation."""
    if isinstance(value, MediaResult):
        return value.to_dict()
    if isinstance(value, dict) and value.get("__media__"):
        return value
    return None


def is_media_collection(value: Any) -> bool:
    """
    Check if a value is a collection of MediaResults.

    Recognized shapes:
    - A list/tuple where every element satisfies is_media_result()
    - A dict with a "creatives" key whose value is a list of MediaResults
    - A dict with a "__media_collection__" marker whose "items" is a list of MediaResults
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        return all(is_media_result(v) for v in value)
    if isinstance(value, dict):
        if value.get("__media_collection__") is True:
            items = value.get("items")
            return isinstance(items, (list, tuple)) and all(is_media_result(v) for v in items)
        creatives = value.get("creatives")
        if isinstance(creatives, (list, tuple)) and creatives:
            return all(is_media_result(v) for v in creatives)
    return False


def extract_media_collection(value: Any) -> Optional[list]:
    """
    Extract a list of media dicts from a media collection.

    Returns a list of dicts (each with __media__ marker) or None if not a collection.
    """
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            data = extract_media_data(v)
            if data:
                out.append(data)
        return out if out else None
    if isinstance(value, dict):
        if value.get("__media_collection__") is True:
            items = value.get("items") or []
        elif isinstance(value.get("creatives"), (list, tuple)):
            items = value.get("creatives")
        else:
            return None
        out = []
        for v in items:
            data = extract_media_data(v)
            if data:
                out.append(data)
        return out if out else None
    return None


def extract_collection_metadata(value: Any) -> Optional[list]:
    """
    Extract accompanying per-item metadata from a media collection dict, if present.

    Applications may return {"creatives": [...], "metadata": [...]} so the LLM sees
    descriptive context alongside [MEDIA:id] markers. This returns the metadata list
    verbatim, or None if the shape isn't recognized.
    """
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, list):
            return metadata
    return None


@dataclass
class LlmBlockInjection:
    """
    Tool return value that must be materialized as multimodal content blocks
    on the LLM's NEXT turn — not just summarized as a tool observation string.

    Used by tools like analyze_creative that need the LLM to actually see
    pixels/frames instead of just reading URLs. The orchestrator detects
    this shape, emits a short text summary into the trace observation, and
    queues the blocks so the next provider prompt includes them as
    ImageBlock/VideoBlock/TextBlock content.

    Fields:
        blocks: Pre-serialized content blocks (each has a "type" key matching
            miiflow_agent.core.message block types: "text", "image_url",
            "video_url", "document"). The orchestrator rehydrates them into
            real block instances when building the next provider prompt.
        summary: Short text shown in the trace observation line. Something
            the LLM can ground subsequent reasoning in (e.g. "Injected 6
            creatives for visual analysis"). Keep under ~200 chars.
    """
    blocks: List[Dict[str, Any]]
    summary: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "__llm_blocks__": True,
            "id": self.id,
            "blocks": self.blocks,
            "summary": self.summary,
        }

    def __str__(self) -> str:
        return f"[LLM_BLOCKS:{self.id}]"

    def __repr__(self) -> str:
        return f"LlmBlockInjection(id={self.id!r}, blocks={len(self.blocks)})"


def is_llm_block_injection(value: Any) -> bool:
    """Check if a value is an LlmBlockInjection."""
    if isinstance(value, LlmBlockInjection):
        return True
    if isinstance(value, dict):
        return value.get("__llm_blocks__") is True
    return False


def extract_llm_blocks(value: Any) -> Optional[Dict[str, Any]]:
    """Extract the block-injection payload. Returns {"blocks": [...], "summary": "..."} or None."""
    if isinstance(value, LlmBlockInjection):
        return value.to_dict()
    if isinstance(value, dict) and value.get("__llm_blocks__"):
        return value
    return None


def is_visualization_result(value: Any) -> bool:
    """
    Check if a value is a VisualizationResult.

    This function is used by the orchestrator to detect visualization results
    BEFORE they are stringified. It handles:
    - VisualizationResult instances directly
    - Dict representations with __visualization__ marker
    - Objects with to_dict() method that return __visualization__ marker

    Args:
        value: The value to check (could be VisualizationResult, dict, or any object)

    Returns:
        True if this is a visualization result
    """
    # Direct VisualizationResult instance
    if isinstance(value, VisualizationResult):
        return True

    # Dict with __visualization__ marker
    if isinstance(value, dict):
        return value.get("__visualization__") is True

    # Object with to_dict() method
    if hasattr(value, "to_dict"):
        try:
            dict_repr = value.to_dict()
            return isinstance(dict_repr, dict) and dict_repr.get("__visualization__") is True
        except Exception:
            return False

    return False


def extract_visualization_data(value: Any) -> Optional[Dict[str, Any]]:
    """
    Extract visualization data from a VisualizationResult or its dict representation.

    This function is used by the orchestrator to extract the full visualization data
    BEFORE the result is stringified. The extracted data can then be emitted as a
    separate VISUALIZATION event to the streaming service.

    Args:
        value: A VisualizationResult object or its dict representation

    Returns:
        Dict with visualization data ready for the streaming service, or None if
        the value is not a visualization result.
    """
    # Direct VisualizationResult instance
    if isinstance(value, VisualizationResult):
        return value.to_dict()

    # Object with to_dict() method (handles subclasses or similar objects)
    if hasattr(value, "to_dict"):
        try:
            dict_repr = value.to_dict()
            if isinstance(dict_repr, dict) and dict_repr.get("__visualization__"):
                return dict_repr
        except Exception:
            return None

    # Dict with __visualization__ marker (already converted)
    if isinstance(value, dict) and value.get("__visualization__"):
        return value

    return None
