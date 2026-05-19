"""Invariant tests for the opt-in coding kit (file_read/edit/write).

Focus is on the *invariants*: read-before-edit, on-disk-mtime-match,
exact-match + uniqueness on Edit, parent-dir-exists on Write. Happy
paths get one test each; the rest verify each failure mode returns a
structured error dict with the right ``error_kind`` so the model can
branch on the failure shape.
"""
from __future__ import annotations

import os
import time

import pytest

from miiflow_agent.core.agent import RunContext
from miiflow_agent.core.tools.coding.file_ops import (
    _file_edit,
    _file_read,
    _file_write,
    build_coding_tools,
    register_coding_tools,
)


@pytest.fixture
def ctx():
    """Fresh RunContext with an empty read_files map on run_state."""
    return RunContext(deps={}, messages=[])


@pytest.fixture
def sample_file(tmp_path):
    """A small UTF-8 file with three distinct lines for exact-match tests."""
    path = tmp_path / "sample.py"
    path.write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_read_records_mtime(ctx, sample_file):
    result = await _file_read(ctx, sample_file)
    assert result["status"] == "ok"
    assert result["total_lines"] == 5
    assert sample_file in ctx.run_state.read_files
    assert ctx.run_state.read_files[sample_file] == os.stat(sample_file).st_mtime


@pytest.mark.asyncio
async def test_file_read_returns_numbered_lines(ctx, sample_file):
    result = await _file_read(ctx, sample_file)
    # Line numbers start at 1 and are tab-separated from content.
    assert "     1\tdef alpha():\n" in result["content"]
    assert "     4\tdef beta():\n" in result["content"]


@pytest.mark.asyncio
async def test_file_read_rejects_relative_path(ctx):
    result = await _file_read(ctx, "relative/path.py")
    assert result["status"] == "error"
    assert result["error_kind"] == "relative_path"


@pytest.mark.asyncio
async def test_file_read_missing_file(ctx, tmp_path):
    result = await _file_read(ctx, str(tmp_path / "nope.txt"))
    assert result["error_kind"] == "not_found"


@pytest.mark.asyncio
async def test_file_read_directory(ctx, tmp_path):
    result = await _file_read(ctx, str(tmp_path))
    assert result["error_kind"] == "is_directory"


@pytest.mark.asyncio
async def test_file_read_window_with_offset_limit(ctx, tmp_path):
    path = tmp_path / "ten_lines.txt"
    path.write_text("\n".join(f"line{i}" for i in range(10)) + "\n", encoding="utf-8")
    result = await _file_read(ctx, str(path), offset=3, limit=2)
    assert result["status"] == "ok"
    assert result["lines_returned"] == 2
    assert result["truncated"] is True
    assert "     4\tline3\n" in result["content"]
    assert "     5\tline4\n" in result["content"]
    assert "line5" not in result["content"]


# ---------------------------------------------------------------------------
# file_edit invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_edit_succeeds_after_read(ctx, sample_file):
    await _file_read(ctx, sample_file)
    result = await _file_edit(
        ctx, sample_file, old_string="return 1", new_string="return 42"
    )
    assert result["status"] == "ok"
    assert result["replacements"] == 1
    assert "return 42" in open(sample_file).read()
    # mtime should be refreshed so a follow-up edit passes the freshness check.
    assert ctx.run_state.read_files[sample_file] == os.stat(sample_file).st_mtime


@pytest.mark.asyncio
async def test_file_edit_without_prior_read_is_refused(ctx, sample_file):
    result = await _file_edit(
        ctx, sample_file, old_string="return 1", new_string="return 42"
    )
    assert result["status"] == "error"
    assert result["error_kind"] == "not_read"
    # File should be unchanged.
    assert "return 1" in open(sample_file).read()


@pytest.mark.asyncio
async def test_file_edit_stale_read_after_disk_mutation(ctx, sample_file):
    await _file_read(ctx, sample_file)
    # Mutate on disk to invalidate the recorded mtime. Sleep is necessary
    # on filesystems where mtime resolution is whole seconds (macOS HFS+),
    # otherwise the new write may end up with the same mtime as the read.
    time.sleep(1.1)
    with open(sample_file, "w") as fh:
        fh.write("# clobbered\n")
    result = await _file_edit(
        ctx, sample_file, old_string="clobbered", new_string="rewritten"
    )
    assert result["error_kind"] == "stale_read"


@pytest.mark.asyncio
async def test_file_edit_ambiguous_match_refused(ctx, tmp_path):
    path = tmp_path / "dup.py"
    path.write_text("x = 1\nx = 1\n", encoding="utf-8")
    await _file_read(ctx, str(path))
    result = await _file_edit(ctx, str(path), old_string="x = 1", new_string="x = 2")
    assert result["error_kind"] == "ambiguous_match"
    assert result["occurrences"] == 2
    # File untouched.
    assert open(path).read() == "x = 1\nx = 1\n"


@pytest.mark.asyncio
async def test_file_edit_replace_all_overrides_uniqueness(ctx, tmp_path):
    path = tmp_path / "dup.py"
    path.write_text("x = 1\nx = 1\n", encoding="utf-8")
    await _file_read(ctx, str(path))
    result = await _file_edit(
        ctx,
        str(path),
        old_string="x = 1",
        new_string="x = 2",
        replace_all=True,
    )
    assert result["status"] == "ok"
    assert result["replacements"] == 2
    assert open(path).read() == "x = 2\nx = 2\n"


@pytest.mark.asyncio
async def test_file_edit_string_not_in_file(ctx, sample_file):
    await _file_read(ctx, sample_file)
    result = await _file_edit(
        ctx, sample_file, old_string="nonexistent", new_string="anything"
    )
    assert result["error_kind"] == "not_found_in_file"


@pytest.mark.asyncio
async def test_file_edit_noop_when_strings_match(ctx, sample_file):
    await _file_read(ctx, sample_file)
    result = await _file_edit(
        ctx, sample_file, old_string="return 1", new_string="return 1"
    )
    assert result["error_kind"] == "noop"


# ---------------------------------------------------------------------------
# file_write invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_new_file_does_not_require_read(ctx, tmp_path):
    path = str(tmp_path / "new.txt")
    result = await _file_write(ctx, path, content="hello\n")
    assert result["status"] == "ok"
    assert result["created"] is True
    assert open(path).read() == "hello\n"
    # mtime recorded so a follow-up edit on this file works without re-reading.
    assert path in ctx.run_state.read_files


@pytest.mark.asyncio
async def test_file_write_overwrite_requires_prior_read(ctx, sample_file):
    result = await _file_write(ctx, sample_file, content="overwritten\n")
    assert result["error_kind"] == "not_read"
    # File should be unchanged.
    assert "def alpha" in open(sample_file).read()


@pytest.mark.asyncio
async def test_file_write_overwrite_after_read_succeeds(ctx, sample_file):
    await _file_read(ctx, sample_file)
    result = await _file_write(ctx, sample_file, content="overwritten\n")
    assert result["status"] == "ok"
    assert result["created"] is False
    assert open(sample_file).read() == "overwritten\n"


@pytest.mark.asyncio
async def test_file_write_missing_parent_refused(ctx, tmp_path):
    path = str(tmp_path / "does" / "not" / "exist.txt")
    result = await _file_write(ctx, path, content="x")
    assert result["error_kind"] == "parent_missing"
    assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def test_build_coding_tools_returns_three_named_tools():
    tools = build_coding_tools()
    names = sorted(t.name for t in tools)
    assert names == ["file_edit", "file_read", "file_write"]


def test_register_coding_tools_idempotent():
    from miiflow_agent.core.tools import ToolRegistry

    registry = ToolRegistry()
    first = register_coding_tools(registry)
    second = register_coding_tools(registry)
    assert [t.name for t in first] == [t.name for t in second]
    assert len(registry.tools) == 3


def test_coding_tools_carry_correct_flags():
    by_name = {t.name: t for t in build_coding_tools()}
    assert by_name["file_read"].schema.is_read_only is True
    assert by_name["file_read"].schema.parallelizable is True
    assert by_name["file_edit"].schema.is_read_only is False
    assert by_name["file_edit"].schema.parallelizable is False
    assert by_name["file_write"].schema.is_read_only is False
    assert by_name["file_write"].schema.parallelizable is False
