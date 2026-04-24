"""
Type definitions for the artifact system.

An artifact is an opaque file (PDF, HTML doc, etc.) produced by a tool and
persisted for download + side-panel viewing. Unlike MediaResult or
VisualizationResult (which are inline display payloads), artifacts have
real size, are uploaded to object storage by the streaming layer, and are
listed at the thread level.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid


@dataclass
class ArtifactResult:
    """
    Result returned by tools that produce downloadable artifacts.

    The tool body stays thin — it returns an ArtifactResult with the raw
    source (e.g. an HTML document). The streaming pipeline on the server
    side detects the __artifact__ marker, renders/stores the file, creates
    a DB row, and emits an SSE chunk to the frontend. The LLM observation
    gets replaced with a short "[ARTIFACT:{id}]" marker to preserve tokens.

    Attributes:
        kind: Artifact kind ("pdf", "html", ...). Drives server-side
            rendering and frontend viewer selection.
        title: Human-readable title shown in the inline card + side panel.
        source_html: Canonical source document. For kind="pdf" the server
            renders this HTML to PDF. For kind="html" it is served directly
            (sandboxed). Capped at 512 KB by the tool schema.
        description: Optional short description shown below the title.
        metadata: Per-kind extras (e.g. {"page_size": "A4"}).
        id: Unique identifier (auto-generated if not provided).
    """

    kind: str
    title: str
    source_html: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "__artifact__": True,
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "description": self.description,
            "source_html": self.source_html,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return f"[ARTIFACT:{self.id}]"

    def __repr__(self) -> str:
        return f"ArtifactResult(kind={self.kind!r}, id={self.id!r}, title={self.title!r})"


def format_artifact_observation(artifact_data: Dict[str, Any]) -> str:
    """Build the tool_result observation string for a produced artifact.

    Replaces the raw ArtifactResult (which contains the full HTML source) with
    a compact marker. The guidance here is explicit about parameter handling
    on revision — an earlier version said "you already have the data you need
    in this conversation," which the model interpreted as permission to omit
    required parameters and call the tool with empty args. Never again.

    Kept here (instead of in each orchestrator) so server-side replay paths
    and the orchestrator produce identical strings.
    """
    art_id = artifact_data.get("id", "unknown")
    kind = (artifact_data.get("kind") or "file").upper()
    title = artifact_data.get("title") or ""
    title_suffix = f" titled {title!r}" if title else ""
    return (
        f"[ARTIFACT:{art_id}] {kind} artifact created{title_suffix}. "
        f"If the user asks for changes, call the artifact tool again and "
        f"supply ALL required parameters (kind, title, html, etc.) with the "
        f"updated content. Parameters are not carried over between calls — "
        f"you must re-supply the full updated html source, not an empty or "
        f"partial object."
    )


def is_artifact_result(value: Any) -> bool:
    if isinstance(value, ArtifactResult):
        return True
    if isinstance(value, dict):
        return value.get("__artifact__") is True
    if hasattr(value, "to_dict"):
        try:
            dict_repr = value.to_dict()
            return isinstance(dict_repr, dict) and dict_repr.get("__artifact__") is True
        except Exception:
            return False
    return False


def extract_artifact_data(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, ArtifactResult):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        try:
            dict_repr = value.to_dict()
            if isinstance(dict_repr, dict) and dict_repr.get("__artifact__"):
                return dict_repr
        except Exception:
            return None
    if isinstance(value, dict) and value.get("__artifact__"):
        return value
    return None
