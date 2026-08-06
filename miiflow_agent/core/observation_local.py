"""File-backed reference implementation of the ObservationSink port.

Standalone deployments (no adapter sink wired) previously had exactly one
behavior for an oversized tool output: truncate it and tell the model to
re-run the tool with a narrower scope — the omitted data was gone. This sink
gives them the same spill-and-page-back behavior an adapter store provides,
patterned after kimi-code's tool-result budget (large outputs land in a local
file with a pointer the model can follow):

    from miiflow_agent.core.observation_local import (
        LocalFileObservationSink, make_read_observation_tool,
    )

    deps["observation_sink"] = LocalFileObservationSink()
    registry.register(make_read_observation_tool())

Only outputs at or above ``spill_threshold_chars`` are persisted — a row per
tool call is an adapter-store concern; locally the point is not losing the
tail of the occasional multi-hundred-KB result.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from .observation import ObservationRecord, StoredObservation, get_observation_sink

logger = logging.getLogger(__name__)

#: Outputs at least this large are spilled to disk. ~12k tokens: big enough
#: that normal analysis results stay purely in-context, small enough that the
#: model's bounded excerpt plus a ref covers the pathological dumps.
DEFAULT_SPILL_THRESHOLD_CHARS = 50_000

_REF_PATTERN = re.compile(r"local_obs_[0-9a-f]{32}")

_TRUNCATION_MARKER = (
    '\n…[truncated {omitted} chars to fit the context window. The full output '
    'is stored — call read_observation(ref="{ref}") to fetch it.]'
)


class LocalFileObservationSink:
    """Spills oversized tool outputs to local files, served back by ref.

    Honors the ObservationSink contract: ``record``/``fetch`` never raise into
    the run loop, and ``llm_excerpt`` keeps the context bound and the store in
    agreement (what the marker promises, ``fetch`` can serve).
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        spill_threshold_chars: int = DEFAULT_SPILL_THRESHOLD_CHARS,
    ):
        base = (
            directory
            or os.getenv("MIIFLOW_OBSERVATION_DIR")
            or os.path.join(
                tempfile.gettempdir(), f"miiflow-observations-{os.getpid()}"
            )
        )
        self._dir = Path(base)
        self._spill_threshold = max(1, int(spill_threshold_chars))
        self._dir_ready = False

    def _ensure_dir(self) -> None:
        if not self._dir_ready:
            self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._dir_ready = True

    def _path_for(self, ref: str) -> Path:
        return self._dir / f"{ref}.json"

    async def record(self, rec: ObservationRecord) -> Optional[str]:
        text = rec.observation_text or ""
        # Never re-spill read_observation's own output: it is already a page
        # of a stored observation, and spilling it would mint a duplicate
        # file with a fresh ref on every read — an unbounded loop of copies
        # whose markers promise a tail that is never served.
        if rec.tool_name == "read_observation":
            return None
        if len(text) < self._spill_threshold:
            return None
        try:
            ref = f"local_obs_{uuid.uuid4().hex}"
            payload = json.dumps(
                {
                    "tool_name": rec.tool_name,
                    "success": rec.success,
                    "created_at_ts": time.time(),
                    "observation_text": text,
                }
            )
            self._ensure_dir()
            # Offloaded: a multi-MB write must not stall the event loop.
            await asyncio.to_thread(
                self._path_for(ref).write_text, payload, "utf-8"
            )
            return ref
        except Exception:  # noqa: BLE001 — the contract: never fail the run
            logger.debug("local observation spill failed", exc_info=True)
            return None

    async def fetch(self, ref: str) -> Optional[StoredObservation]:
        # The ref format doubles as the path-traversal guard: anything that
        # is not exactly a ref we minted maps to no file.
        if not ref or not _REF_PATTERN.fullmatch(ref):
            return None
        path = self._path_for(ref)
        try:
            raw = await asyncio.to_thread(path.read_text, "utf-8")
            data = json.loads(raw)
            return StoredObservation(
                ref=ref,
                observation_text=data.get("observation_text", ""),
                tool_name=data.get("tool_name", ""),
                success=bool(data.get("success", True)),
                created_at_ts=float(data.get("created_at_ts", 0.0)),
            )
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001
            logger.debug("local observation fetch failed", exc_info=True)
            return None

    def llm_excerpt(
        self, text: str, tool_name: Optional[str], ref: Optional[str]
    ) -> str:
        """Bound what enters the context; the marker points at the ref.

        The bound equals the spill threshold, so exactly the outputs that got
        a ref are the ones that get truncated — the marker never promises a
        ref that does not exist.
        """
        if not text or len(text) <= self._spill_threshold:
            return text
        if ref is None:
            # Spill failed (or record was skipped): fall back to the
            # framework's no-ref wording by returning the text unchanged and
            # letting bound_observation_for_llm apply its own ceiling.
            return text
        omitted = len(text) - self._spill_threshold
        return text[: self._spill_threshold] + _TRUNCATION_MARKER.format(
            omitted=omitted, ref=ref
        )


#: Page size for read_observation. One page must come back under every
#: excerpt bound on the return path — the tool's own result flows through
#: the same recording seam as any other observation, and a page larger than
#: the bound would be re-truncated into a marker promising a tail that can
#: never be served (the loop this constant exists to break).
READ_OBSERVATION_PAGE_CHARS = 40_000


def make_read_observation_tool():
    """Build the framework ``read_observation`` tool for local deployments.

    Adapter deployments usually ship their own (org-guarded) version; this
    one is sink-generic — it resolves whatever sink the run carries and
    fetches by ref, so it works with any ObservationSink implementation.
    Serves the stored text in pages (``offset``): a stored output larger
    than the excerpt bound cannot be returned whole without being
    re-truncated by the very bound that spilled it.
    """
    from .tools import ParameterSchema, tool
    from .tools.types import ParameterType

    @tool(
        name="read_observation",
        always_load=True,
        parallelizable=True,
        search_keywords=[
            "read", "observation", "full", "output", "truncated", "ref",
        ],
        description=(
            "Fetch the stored output of an earlier tool execution by its "
            "ref, paged. Use when a tool result in context was truncated "
            "with a marker naming a ref and the omitted content matters. "
            "Pass offset to continue reading (the result names next_offset "
            "when more remains). Prefer the excerpt already in context when "
            "it answers the question."
        ),
        parameters={
            "ref": ParameterSchema(
                name="ref",
                type=ParameterType.STRING,
                description="The observation ref from a truncation marker.",
                required=True,
            ),
            "offset": ParameterSchema(
                name="offset",
                type=ParameterType.INTEGER,
                description=(
                    "Character offset to read from (default 0). Use the "
                    "next_offset from a previous read_observation result."
                ),
                required=False,
            ),
        },
    )
    async def read_observation(ctx, ref: str, offset: int = 0) -> dict:
        """Fetch a page of a stored observation's text by ref."""
        sink = get_observation_sink(ctx)
        if sink is None:
            return {"error": "Observation store is not available in this run."}
        stored = await sink.fetch(ref)
        if stored is None:
            return {"error": f"No stored observation found for ref '{ref}'."}

        text = stored.observation_text or ""
        try:
            start = max(0, int(offset))
        except (TypeError, ValueError):
            start = 0
        page = text[start : start + READ_OBSERVATION_PAGE_CHARS]
        next_offset = start + len(page)
        result = {
            "ref": stored.ref,
            "tool_name": stored.tool_name,
            "success": stored.success,
            "total_chars": len(text),
            "offset": start,
            "observation": page,
        }
        if next_offset < len(text):
            result["next_offset"] = next_offset
            result["note"] = (
                f"Partial: chars {start}-{next_offset} of {len(text)}. Call "
                f'read_observation(ref="{stored.ref}", offset={next_offset}) '
                "for the next page."
            )
        return result

    return read_observation
