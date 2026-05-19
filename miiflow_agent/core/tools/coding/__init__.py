"""Opt-in coding toolkit: file_read / file_edit / file_write.

Three FunctionTools that bake in the read-before-edit + exact-match +
uniqueness invariants that make coding-agent file operations correct
under concurrent or repeated edits. Modeled on Claude Code's Read/Edit/
Write tools, but the SDK ships them OFF by default — register
explicitly with ``register_coding_tools(agent.tool_registry)`` or
``agent.add_tools(build_coding_tools())``.

The three tools share state through ``ctx.run_state.read_files`` (added
in ``core.agent.RunState``): a ``Dict[str, float]`` of path → mtime at
read time. ``file_edit`` and ``file_write`` consult this map to enforce:

  1. **Read-before-edit**: an Edit on a file the model hasn't Read this
     run is refused with a structured error. Same for overwriting an
     existing file with Write.
  2. **On-disk-mtime-match**: if the file changed since the recorded
     read (someone else edited it, or the model edited it via Bash),
     the Edit is refused — the model must re-Read first.
  3. **Exact match + uniqueness**: Edit's ``old_string`` must occur
     exactly once unless ``replace_all=True``; otherwise refused so the
     model can supply more context.

Errors are returned as structured ``{"status": "error", ...}`` dicts so
the model can observe the failure mode and recover, not raised as tool
execution errors.
"""
from .file_ops import (
    FILE_EDIT_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    build_coding_tools,
    register_coding_tools,
)

__all__ = [
    "FILE_EDIT_TOOL_NAME",
    "FILE_READ_TOOL_NAME",
    "FILE_WRITE_TOOL_NAME",
    "build_coding_tools",
    "register_coding_tools",
]
