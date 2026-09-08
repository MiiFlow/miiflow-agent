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


#: A dict tool result may carry a list of artifact dicts under this key. The
#: orchestrator pops it, publishes one artifact event per entry, and appends
#: one `[ARTIFACT:...]` line per entry to the ordinary observation — so a
#: tool can hand the person a file AND keep telling the model what it did
#: (a report tool's `document_id`, `builder_path`, ...).
ATTACHED_ARTIFACTS_KEY = "__artifacts__"


def pop_attached_artifacts(output: Any) -> "tuple[Any, list[Dict[str, Any]]]":
    """``(output_without_the_key, [artifact dicts])`` for a dict result that
    carries :data:`ATTACHED_ARTIFACTS_KEY`; ``(output, [])`` for anything else.

    Never mutates ``output``: the same dict may already be recorded elsewhere.
    Entries that are not artifact results are dropped, not raised on — a
    malformed side-channel must not turn a working tool call into a failure.
    """
    if not isinstance(output, dict) or not isinstance(output.get(ATTACHED_ARTIFACTS_KEY), list):
        return output, []
    attached = [
        extract_artifact_data(entry)
        for entry in output[ATTACHED_ARTIFACTS_KEY]
        if is_artifact_result(entry)
    ]
    rest = {key: value for key, value in output.items() if key != ATTACHED_ARTIFACTS_KEY}
    return rest, [entry for entry in attached if entry]


def is_file_backed_artifact(artifact_data: Dict[str, Any]) -> bool:
    """An artifact whose bytes already exist (``metadata.file_asset_id``),
    as opposed to one the server renders from ``source_html``. That one fact
    decides everything downstream: no render, no revision via edit_artifact,
    and a different observation."""
    metadata = artifact_data.get("metadata") if isinstance(artifact_data, dict) else None
    return bool(isinstance(metadata, dict) and metadata.get("file_asset_id"))


def format_artifact_observation(artifact_data: Dict[str, Any]) -> str:
    """Build the tool_result observation string for a produced artifact.

    Replaces the raw ArtifactResult (which contains the full HTML source) with
    a compact marker. The revision guidance points at get_artifact/edit_artifact
    — actions the model can actually take — rather than telling it to "re-supply
    the full html", which it cannot do once the source has been compacted out of
    context. An earlier version said "you already have the data you need in this
    conversation," which the model read as permission to call with empty args;
    the version after that told it to re-supply from memory, which it couldn't,
    so it echoed the compaction placeholder and rendered blank PDFs. The fix is
    to make the current source retrievable (get_artifact) instead of relying on
    the model to reproduce it.

    Kept here (instead of in each orchestrator) so server-side replay paths
    and the orchestrator produce identical strings.
    """
    art_id = artifact_data.get("id", "unknown")
    kind = (artifact_data.get("kind") or "file").upper()
    title = artifact_data.get("title") or ""
    title_suffix = f" titled {title!r}" if title else ""
    if is_file_backed_artifact(artifact_data):
        # A rendered file, delivered as a card. The model must not paste,
        # quote or invent a link for it (the one it would reach for is a
        # presigned bucket URL), and cannot revise it through edit_artifact:
        # the source is whatever produced the file.
        return (
            f"[ARTIFACT:{art_id}] {kind} file{title_suffix} is attached to your "
            "reply as a downloadable file card; the person already has it. Refer "
            "to it by name (\"the PDF above\"). Do NOT paste, quote, or invent a "
            "link for it. It is a rendered file, not an editable document: "
            "get_artifact and edit_artifact do not apply — to change it, change "
            "its source and produce it again."
        )
    return (
        f"[ARTIFACT:{art_id}] {kind} artifact created{title_suffix}. "
        f'To revise it, first call get_artifact("{art_id}") to read its current '
        f'HTML, then call edit_artifact("{art_id}", html=<complete updated HTML>). '
        f"Do NOT rebuild the HTML from memory and do NOT pass a placeholder or "
        f"partial body — always send the full document."
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
