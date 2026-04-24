"""
Artifact module for miiflow-agent.

Provides the ArtifactResult marker dataclass that tools return when they
produce a downloadable file (PDF, HTML, etc.) for the user. The streaming
handler detects the __artifact__ marker and persists + emits the artifact
as a dedicated SSE event, mirroring the MediaResult / VisualizationResult
pattern.
"""

from .types import (
    ArtifactResult,
    extract_artifact_data,
    format_artifact_observation,
    is_artifact_result,
)

__all__ = [
    "ArtifactResult",
    "extract_artifact_data",
    "format_artifact_observation",
    "is_artifact_result",
]
