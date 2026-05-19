"""file_read / file_edit / file_write — the opt-in coding toolkit.

See ``__init__.py`` for the invariants these tools enforce and the
shared state model on ``ctx.run_state.read_files``. This module is the
concrete implementation; consumers register the kit via
``register_coding_tools(agent.tool_registry)``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from ..function import FunctionTool
from ..schemas import ParameterSchema, ToolSchema
from ..types import ParameterType, ToolType


logger = logging.getLogger(__name__)


FILE_READ_TOOL_NAME = "file_read"
FILE_EDIT_TOOL_NAME = "file_edit"
FILE_WRITE_TOOL_NAME = "file_write"


# Upper bound on the bytes a single file_read returns. Larger files must
# be paged with offset/limit. Mirrors the spirit of Claude Code's Read
# size threshold — the model gets told to retry with a window rather
# than silently truncating.
_MAX_READ_BYTES = 512 * 1024  # 512 KiB

# Default number of lines returned when ``limit`` isn't supplied. Keeps
# token budgets bounded for casual reads; the model passes an explicit
# limit when it needs more.
_DEFAULT_READ_LINES = 2000


def _err(kind: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Structured tool observation for recoverable failures.

    Returned (not raised) so the model sees the error mode and can pick
    a different action. Raising would surface as a generic tool failure
    and lose the structured ``kind`` the model can branch on.
    """
    return {"status": "error", "error_kind": kind, "error": message, **extra}


def _require_absolute(path: str) -> Optional[Dict[str, Any]]:
    """Reject relative paths up front. Returns an error dict on reject,
    None when the path is acceptable."""
    if not os.path.isabs(path):
        return _err(
            "relative_path",
            (
                f"Path '{path}' is relative. Pass an absolute path — "
                "tools don't resolve against a session cwd because that "
                "would race under parallel calls."
            ),
        )
    return None


def _get_read_log(ctx: Any) -> Optional[Dict[str, float]]:
    """Return the ``read_files`` dict on ctx.run_state, or None if the
    caller wired a context without one. None is treated as "tracking
    disabled" — the invariants fall back to permissive mode rather than
    crashing tests that pass a bare dict as ctx."""
    run_state = getattr(ctx, "run_state", None)
    if run_state is None:
        return None
    return getattr(run_state, "read_files", None)


async def _file_read(
    ctx: Any,
    path: str,
    offset: int = 0,
    limit: int = _DEFAULT_READ_LINES,
) -> Dict[str, Any]:
    """Read a file by absolute path, return numbered lines, record mtime.

    Recording the mtime is the load-bearing side effect: ``file_edit``
    and ``file_write`` consult it to detect on-disk drift and refuse
    edits to files that changed after the read.
    """
    if (rel_err := _require_absolute(path)) is not None:
        return rel_err

    if not os.path.exists(path):
        return _err("not_found", f"File '{path}' does not exist.")

    if os.path.isdir(path):
        return _err(
            "is_directory",
            f"Path '{path}' is a directory, not a file. Use a directory listing tool.",
        )

    try:
        stat = os.stat(path)
    except OSError as exc:
        return _err("stat_failed", f"Could not stat '{path}': {exc}")

    if stat.st_size > _MAX_READ_BYTES and limit == _DEFAULT_READ_LINES:
        # The model didn't ask for a window but the file is big — surface
        # the size so it can retry with offset/limit instead of silently
        # truncating mid-content.
        return _err(
            "file_too_large",
            (
                f"File is {stat.st_size} bytes (>{_MAX_READ_BYTES} byte limit). "
                "Retry with explicit offset/limit to read a specific line range."
            ),
            size_bytes=stat.st_size,
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except UnicodeDecodeError:
        return _err(
            "binary_file",
            f"File '{path}' is not valid UTF-8. The coding kit handles text files only.",
        )
    except OSError as exc:
        return _err("read_failed", f"Could not read '{path}': {exc}")

    total_lines = len(lines)
    if offset < 0:
        return _err("invalid_offset", f"offset must be >= 0, got {offset}.")
    if limit <= 0:
        return _err("invalid_limit", f"limit must be > 0, got {limit}.")

    window = lines[offset : offset + limit]
    # Numbered output mirrors what Claude Code's Read returns — gives
    # the model stable line anchors for follow-up edits.
    numbered = "".join(
        f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(window)
    )

    # Record mtime AFTER reading — if the file is mutated between read
    # and stat the worst case is a false rejection on the next edit,
    # which is the safe direction.
    read_log = _get_read_log(ctx)
    if read_log is not None:
        read_log[path] = stat.st_mtime

    return {
        "status": "ok",
        "path": path,
        "content": numbered,
        "offset": offset,
        "lines_returned": len(window),
        "total_lines": total_lines,
        "truncated": offset + len(window) < total_lines,
    }


def _check_read_invariants(
    ctx: Any, path: str, *, require_existing_read: bool
) -> Optional[Dict[str, Any]]:
    """Verify read-before-edit + mtime-match for ``path``.

    Returns an error dict on violation, None on success. When
    ``require_existing_read`` is False (file_write to a new file) we
    skip the read-log check entirely — Claude Code's Write follows the
    same rule: only existing files need a prior Read.
    """
    read_log = _get_read_log(ctx)
    if read_log is None:
        # No tracking wired (bare-dict ctx in tests, or run_state not
        # initialized). Treat as permissive — the invariants can't be
        # enforced without state, and crashing here would make the kit
        # unusable in test harnesses that don't construct a RunState.
        return None

    file_exists = os.path.exists(path)

    if not file_exists:
        if require_existing_read:
            return _err(
                "not_found",
                f"File '{path}' does not exist. Cannot edit a missing file.",
            )
        # Write to a new path is fine.
        return None

    recorded_mtime = read_log.get(path)
    if recorded_mtime is None:
        return _err(
            "not_read",
            (
                f"File '{path}' has not been read this run. Call "
                "file_read first, then re-issue the edit. This guards "
                "against editing files the model can't see."
            ),
        )

    try:
        current_mtime = os.stat(path).st_mtime
    except OSError as exc:
        return _err("stat_failed", f"Could not stat '{path}': {exc}")

    if current_mtime != recorded_mtime:
        return _err(
            "stale_read",
            (
                f"File '{path}' has changed on disk since it was read "
                f"(mtime {recorded_mtime} → {current_mtime}). Re-read "
                "the file to get the current content before editing."
            ),
        )
    return None


async def _file_edit(
    ctx: Any,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """Replace ``old_string`` with ``new_string`` in ``path``.

    Exact string match. Refuses ambiguous matches (>1 occurrence) unless
    ``replace_all=True`` — the model has to supply more surrounding
    context to pin down one occurrence, which is the only way to make
    edits idempotent under repeated runs.
    """
    if (rel_err := _require_absolute(path)) is not None:
        return rel_err

    if old_string == new_string:
        return _err(
            "noop",
            "old_string and new_string are identical — nothing to do.",
        )

    if (inv_err := _check_read_invariants(ctx, path, require_existing_read=True)) is not None:
        return inv_err

    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except UnicodeDecodeError:
        return _err(
            "binary_file",
            f"File '{path}' is not valid UTF-8.",
        )
    except OSError as exc:
        return _err("read_failed", f"Could not read '{path}': {exc}")

    occurrences = content.count(old_string)
    if occurrences == 0:
        return _err(
            "not_found_in_file",
            (
                "old_string was not found in the file. Whitespace and "
                "indentation must match exactly — re-read the file and "
                "copy the exact bytes you want to replace."
            ),
        )
    if occurrences > 1 and not replace_all:
        return _err(
            "ambiguous_match",
            (
                f"old_string matches {occurrences} locations in '{path}'. "
                "Either supply more surrounding context to pin down one "
                "occurrence, or set replace_all=True to replace every match."
            ),
            occurrences=occurrences,
        )

    new_content = content.replace(old_string, new_string)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_content)
    except OSError as exc:
        return _err("write_failed", f"Could not write '{path}': {exc}")

    # Refresh the recorded mtime so a follow-up edit in the same run
    # passes the mtime-match check.
    read_log = _get_read_log(ctx)
    if read_log is not None:
        try:
            read_log[path] = os.stat(path).st_mtime
        except OSError:
            # Mtime refresh failure isn't worth failing the edit over —
            # the worst case is a forced re-read on the next edit.
            read_log.pop(path, None)

    return {
        "status": "ok",
        "path": path,
        "replacements": occurrences if replace_all else 1,
        "bytes_written": len(new_content.encode("utf-8")),
    }


async def _file_write(
    ctx: Any, path: str, content: str
) -> Dict[str, Any]:
    """Create or overwrite ``path`` with ``content``.

    Overwriting an existing file requires a prior file_read this run
    (same invariant as file_edit). New files are fine without a prior
    Read since there's nothing to lose.
    """
    if (rel_err := _require_absolute(path)) is not None:
        return rel_err

    file_existed = os.path.exists(path)
    if file_existed and os.path.isdir(path):
        return _err(
            "is_directory",
            f"Path '{path}' is a directory.",
        )

    if (inv_err := _check_read_invariants(ctx, path, require_existing_read=False)) is not None:
        return inv_err

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        return _err(
            "parent_missing",
            (
                f"Parent directory '{parent}' does not exist. Create it "
                "first — file_write does not auto-mkdir to avoid "
                "accidentally provisioning paths far from the intended "
                "tree."
            ),
        )

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as exc:
        return _err("write_failed", f"Could not write '{path}': {exc}")

    read_log = _get_read_log(ctx)
    if read_log is not None:
        try:
            read_log[path] = os.stat(path).st_mtime
        except OSError:
            read_log.pop(path, None)

    return {
        "status": "ok",
        "path": path,
        "created": not file_existed,
        "bytes_written": len(content.encode("utf-8")),
    }


_READ_DESCRIPTION = """Read a UTF-8 text file by absolute path.

Returns numbered lines (``"%6d\\t<line>"``) for stable anchor points in
follow-up edits, plus the file's total line count and a truncation flag.

Use ``offset`` (0-based) and ``limit`` to page through large files. The
default window is the first 2000 lines; files over 512 KiB without an
explicit window return a ``file_too_large`` error so you re-issue with
a smaller range.

Side effect: records the file's mtime in run state so a later
``file_edit`` or ``file_write`` on the same path passes the
read-before-edit invariant. You must read a file before editing or
overwriting it."""


_EDIT_DESCRIPTION = """Replace an exact string in a file.

``old_string`` is matched literally — no regex, no fuzzy matching. The
match must be unique unless you set ``replace_all=True``; ambiguous
matches return an ``ambiguous_match`` error so you can supply more
surrounding context.

Preconditions enforced:
  - The file must have been read this run via ``file_read``.
  - The file's on-disk mtime must match what was recorded at read time;
    if the file changed, you'll get a ``stale_read`` error and must
    re-read before editing.

Whitespace and indentation must match the file exactly. If your edit
fails with ``not_found_in_file``, re-read the file and copy the bytes
you want to replace literally — model-paraphrased indentation is the
most common cause."""


_WRITE_DESCRIPTION = """Create a new file or overwrite an existing one.

Writes ``content`` to ``path`` as UTF-8. For overwrites, the same
read-before-edit invariant as ``file_edit`` applies: you must
``file_read`` the existing file first, and its mtime must still match
when the write runs. New files don't require a prior read.

Does NOT create parent directories — a missing parent returns
``parent_missing`` so unintended path provisioning has to be explicit.

Use ``file_edit`` for partial changes; reserve ``file_write`` for new
files or full rewrites."""


def _build_file_read_tool() -> FunctionTool:
    schema = ToolSchema(
        name=FILE_READ_TOOL_NAME,
        description=_READ_DESCRIPTION,
        tool_type=ToolType.FUNCTION,
        is_read_only=True,
        # Multiple reads in one batch are safe — they only mutate the
        # per-path mtime entry in read_files, and last-write-wins on a
        # dict key is fine because both readers stat the same file.
        parallelizable=True,
        parameters={
            "path": ParameterSchema(
                name="path",
                type=ParameterType.STRING,
                description="Absolute path to the file to read.",
                required=True,
            ),
            "offset": ParameterSchema(
                name="offset",
                type=ParameterType.INTEGER,
                description=(
                    "0-based line offset to start reading from. Use with "
                    "``limit`` to page through large files."
                ),
                required=False,
                default=0,
            ),
            "limit": ParameterSchema(
                name="limit",
                type=ParameterType.INTEGER,
                description=(
                    "Max lines to return. Defaults to 2000. Reduce for "
                    "narrow windows, increase when you need more context."
                ),
                required=False,
                default=_DEFAULT_READ_LINES,
            ),
        },
    )
    _file_read._tool_schema = schema  # type: ignore[attr-defined]
    return FunctionTool(_file_read)


def _build_file_edit_tool() -> FunctionTool:
    schema = ToolSchema(
        name=FILE_EDIT_TOOL_NAME,
        description=_EDIT_DESCRIPTION,
        tool_type=ToolType.FUNCTION,
        is_read_only=False,
        # Edits mutate shared filesystem state — never safe to gather.
        # Two parallel edits to the same file would both stat the same
        # mtime and both pass the check, then race on write.
        parallelizable=False,
        parameters={
            "path": ParameterSchema(
                name="path",
                type=ParameterType.STRING,
                description="Absolute path to the file to edit.",
                required=True,
            ),
            "old_string": ParameterSchema(
                name="old_string",
                type=ParameterType.STRING,
                description=(
                    "Exact bytes to replace. Must occur exactly once "
                    "unless ``replace_all=True``."
                ),
                required=True,
            ),
            "new_string": ParameterSchema(
                name="new_string",
                type=ParameterType.STRING,
                description="Replacement bytes.",
                required=True,
            ),
            "replace_all": ParameterSchema(
                name="replace_all",
                type=ParameterType.BOOLEAN,
                description=(
                    "Replace every occurrence instead of requiring "
                    "uniqueness. Defaults to False — prefer narrowing "
                    "``old_string`` with surrounding context over flipping "
                    "this flag, since blanket replaces are easier to get "
                    "wrong."
                ),
                required=False,
                default=False,
            ),
        },
    )
    _file_edit._tool_schema = schema  # type: ignore[attr-defined]
    return FunctionTool(_file_edit)


def _build_file_write_tool() -> FunctionTool:
    schema = ToolSchema(
        name=FILE_WRITE_TOOL_NAME,
        description=_WRITE_DESCRIPTION,
        tool_type=ToolType.FUNCTION,
        is_read_only=False,
        parallelizable=False,
        parameters={
            "path": ParameterSchema(
                name="path",
                type=ParameterType.STRING,
                description="Absolute path to the file to create or overwrite.",
                required=True,
            ),
            "content": ParameterSchema(
                name="content",
                type=ParameterType.STRING,
                description="Full file contents, written as UTF-8.",
                required=True,
            ),
        },
    )
    _file_write._tool_schema = schema  # type: ignore[attr-defined]
    return FunctionTool(_file_write)


def build_coding_tools() -> List[FunctionTool]:
    """Return freshly-built (file_read, file_edit, file_write) instances.

    Caller is responsible for registering them with a ``ToolRegistry``.
    Use ``register_coding_tools(registry)`` for the common case.
    """
    return [_build_file_read_tool(), _build_file_edit_tool(), _build_file_write_tool()]


def register_coding_tools(tool_registry) -> List[FunctionTool]:
    """Register the coding kit on a ``ToolRegistry``.

    Idempotent: silently skips tools already present (matches the
    plan-mode registration pattern). Returns the list of FunctionTool
    instances that were registered (or already existed) so callers can
    mirror them onto ``Agent._tools``.
    """
    tools: List[FunctionTool] = []
    existing = (
        set(tool_registry.tools.keys()) if hasattr(tool_registry, "tools") else set()
    )
    for tool in build_coding_tools():
        if tool.name in existing:
            tools.append(tool_registry.tools[tool.name])
            continue
        tool_registry.register(tool)
        tools.append(tool)
    return tools


__all__ = [
    "FILE_EDIT_TOOL_NAME",
    "FILE_READ_TOOL_NAME",
    "FILE_WRITE_TOOL_NAME",
    "build_coding_tools",
    "register_coding_tools",
]
