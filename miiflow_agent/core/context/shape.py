"""The shape of a request, as the context engine sees it.

The defect this type exists to fix: ``ContextCompressor.compress_if_needed()``
took only ``messages``. It never saw the system prompt or the tool schemas, so
it was blind to the two largest and *least* compressible parts of the request.

On a tool-heavy assistant the schemas alone can be ~30K tokens — larger than
most conversations. A compressor that cannot see them has two failure modes,
and we hit both:

  * **Late compaction.** Messages look comfortably under the threshold while
    the real request is already over it, so the first sign of trouble is a
    context-overflow 400 from the provider.
  * **Thrash.** Once over the line, every pass shrinks messages by a healthy
    margin and the request stays over anyway, because the incompressible floor
    (system + tools) alone exceeds the threshold. So the next turn compacts
    again. Forever.

``RequestShape`` carries all three tiers, which makes the floor measurable —
and a measurable floor is what lets the engine tell "compact harder" apart
from "compaction cannot help here, stop trying" (see ``ContextEngine``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..message import Message


@dataclass
class RequestShape:
    """Everything that will go on the wire for one LLM call.

    ``tools`` holds the schemas *as they will actually be sent* — already
    filtered, already provider-formatted. Passing the unfiltered registry here
    would over-count, and passing the universal (pre-conversion) schemas would
    under-count on providers whose format is more verbose. Take whatever
    ``get_filtered_schemas()`` returned for this call.
    """

    messages: List[Message] = field(default_factory=list)
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None

    def with_messages(self, messages: List[Message]) -> "RequestShape":
        """Copy carrying a different message list.

        Compaction rewrites messages and leaves the floor untouched, so this
        keeps the two concerns from getting tangled: the engine never mutates
        the caller's shape in place.
        """
        return RequestShape(
            messages=messages,
            system=self.system,
            tools=self.tools,
            provider=self.provider,
            model=self.model,
        )


@dataclass
class TokenBreakdown:
    """Per-tier token estimate for one :class:`RequestShape`.

    Kept as separate tiers rather than a single total because the *ratio*
    between them is what drives policy. A 100K-token request that is 90K
    conversation is a compaction problem; the same request that is 90K tool
    schemas is a tool-surface problem, and compacting it is wasted work.
    """

    system: int = 0
    tools: int = 0
    messages: int = 0

    #: Correction factor applied to the raw local estimate, and whether it was
    #: learned from a real provider count or is still the 1.0 default.
    calibration_factor: float = 1.0
    calibrated: bool = False

    @property
    def floor(self) -> int:
        """Tokens compaction cannot remove: system prompt + tool schemas.

        Compaction only ever touches ``messages``. When ``floor`` alone is at
        or above the threshold, no amount of compaction will bring the request
        under it — that is the anti-thrash signal.
        """
        return self.system + self.tools

    @property
    def total(self) -> int:
        return self.system + self.tools + self.messages

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form, for the breakdown event and observability."""
        return {
            "system": self.system,
            "tools": self.tools,
            "messages": self.messages,
            "floor": self.floor,
            "total": self.total,
            "calibration_factor": round(self.calibration_factor, 4),
            "calibrated": self.calibrated,
        }
