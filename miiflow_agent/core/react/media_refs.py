"""Resolution of symbolic media references in tool inputs.

When image-generation tools produce media, the URLs land in the run's
``media_store`` keyed by id, and the model refers to them symbolically.
Before a tool executes, its inputs are rewritten so the tool receives real
URLs — unless the parameter *declares* it consumes the symbolic ref itself
(``ParameterSchema.media_ref_passthrough=True``), e.g. tools that re-emit or
track refs rather than fetching the bytes.

Handled reference shapes, in order:
  1. ``media_ref:<id>`` — the explicit, documented form.
  2. Hallucinated sandbox paths (``/mnt/data/<uuid>.png`` — common with
     GPT-family models) whose embedded UUID matches a stored media id.
  3. A lone stored media + a file-path-looking value → assume it means that
     media.

Extracted from the orchestrator: this is pure input rewriting with no loop
state beyond the media store, and the passthrough declaration belongs on the
tool schema — not in a table of application tool names inside the framework.
"""

import logging
import re
from typing import Dict, Iterable, Optional, Set

logger = logging.getLogger(__name__)

_MEDIA_REF_PATTERN = re.compile(r"^media_ref:(.+)$")
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def declared_passthrough_params(schema) -> Set[str]:
    """Parameter names a tool schema marks ``media_ref_passthrough=True``."""
    params = getattr(schema, "parameters", None) or {}
    if not isinstance(params, dict):
        return set()
    return {
        name
        for name, param in params.items()
        if getattr(param, "media_ref_passthrough", False)
    }


def resolve_media_refs(
    inputs: Dict,
    media_store: Dict[str, str],
    passthrough_params: Optional[Iterable[str]] = None,
) -> Dict:
    """Rewrite symbolic media references in ``inputs`` to stored URLs.

    ``passthrough_params`` are left untouched — those parameters consume the
    symbolic ref directly. Returns a new dict; never mutates ``inputs``.
    """
    if not media_store:
        return inputs

    passthrough = set(passthrough_params or ())
    resolved: Dict = {}

    for key, value in inputs.items():
        if key in passthrough:
            resolved[key] = value
            continue

        if isinstance(value, str):
            stripped = value.strip()

            # 1. Explicit media_ref:<id>
            match = _MEDIA_REF_PATTERN.match(stripped)
            if match:
                media_id = match.group(1)
                stored_url = media_store.get(media_id)
                if stored_url:
                    resolved[key] = stored_url
                    logger.info(f"Resolved media_ref:{media_id} to stored URL")
                    continue
                else:
                    logger.warning(f"media_ref:{media_id} not found in media store")

            # 2. Non-URL string (hallucinated path like /mnt/data/..., or bare
            #    filename): try to find a UUID matching a stored media ID.
            if not stripped.startswith(("http://", "https://", "data:")):
                uuid_matches = _UUID_PATTERN.findall(stripped)
                resolved_from_uuid = False
                for uuid_str in uuid_matches:
                    stored_url = media_store.get(uuid_str)
                    if stored_url:
                        resolved[key] = stored_url
                        logger.info(
                            f"Resolved hallucinated path '{stripped}' to stored "
                            f"URL via media ID {uuid_str}"
                        )
                        resolved_from_uuid = True
                        break
                if resolved_from_uuid:
                    continue

                # 3. If only one media exists and the value looks like a file
                #    path, assume it refers to the most recent generated image.
                if len(media_store) == 1 and (
                    stripped.startswith("/") or stripped.startswith("file://")
                ):
                    only_url = next(iter(media_store.values()))
                    resolved[key] = only_url
                    logger.info(
                        f"Resolved file path '{stripped}' to only available media URL"
                    )
                    continue

            resolved[key] = value
        else:
            resolved[key] = value

    return resolved
