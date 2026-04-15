"""Unit tests for LlmBlockInjection sentinel.

Covers: detection / extraction / isolation from MediaResult and
VisualizationResult / serialization round-trip.
"""

import pytest

from miiflow_agent.visualization.types import (
    LlmBlockInjection,
    MediaResult,
    VisualizationResult,
    extract_llm_blocks,
    extract_media_data,
    is_llm_block_injection,
    is_media_collection,
    is_media_result,
    is_visualization_result,
)


class TestLlmBlockInjectionDetection:
    def test_instance_detected(self):
        inj = LlmBlockInjection(blocks=[{"type": "text", "text": "hi"}], summary="x")
        assert is_llm_block_injection(inj) is True

    def test_dict_with_marker_detected(self):
        inj = LlmBlockInjection(blocks=[], summary="")
        assert is_llm_block_injection(inj.to_dict()) is True

    def test_plain_dict_without_marker_rejected(self):
        assert is_llm_block_injection({"blocks": [], "summary": ""}) is False

    def test_none_rejected(self):
        assert is_llm_block_injection(None) is False

    def test_string_rejected(self):
        assert is_llm_block_injection("hello") is False

    def test_list_rejected(self):
        assert is_llm_block_injection([]) is False
        assert is_llm_block_injection([{"type": "text"}]) is False


class TestLlmBlockInjectionIsolation:
    """Sentinel must not collide with other result types."""

    def test_not_media_result(self):
        inj = LlmBlockInjection(blocks=[{"type": "image_url", "image_url": "http://x"}], summary="")
        assert is_media_result(inj) is False

    def test_not_media_collection(self):
        inj = LlmBlockInjection(blocks=[{"type": "image_url", "image_url": "http://x"}], summary="")
        assert is_media_collection(inj) is False

    def test_not_visualization_result(self):
        inj = LlmBlockInjection(blocks=[], summary="")
        assert is_visualization_result(inj) is False

    def test_media_result_not_llm_block(self):
        m = MediaResult(url="http://x", media_type="image", alt_text="a")
        assert is_llm_block_injection(m) is False
        assert is_llm_block_injection(m.to_dict()) is False


class TestLlmBlockInjectionExtraction:
    def test_extract_from_instance(self):
        inj = LlmBlockInjection(
            blocks=[{"type": "text", "text": "a"}, {"type": "image_url", "image_url": "u"}],
            summary="s",
        )
        data = extract_llm_blocks(inj)
        assert data is not None
        assert data["summary"] == "s"
        assert len(data["blocks"]) == 2

    def test_extract_from_dict(self):
        original = LlmBlockInjection(blocks=[{"type": "text", "text": "a"}], summary="s")
        data = extract_llm_blocks(original.to_dict())
        assert data is not None
        assert data["summary"] == "s"

    def test_extract_from_non_sentinel_returns_none(self):
        assert extract_llm_blocks({"blocks": [], "summary": ""}) is None
        assert extract_llm_blocks(None) is None
        assert extract_llm_blocks("hi") is None


class TestLlmBlockInjectionSerialization:
    def test_roundtrip(self):
        inj = LlmBlockInjection(
            blocks=[
                {"type": "image_url", "image_url": "http://a", "detail": "high"},
                {"type": "video_url", "video_url": "http://b", "mime_type": "video/mp4"},
            ],
            summary="two blocks",
        )
        d = inj.to_dict()
        assert d["__llm_blocks__"] is True
        assert d["summary"] == "two blocks"
        assert d["blocks"] == inj.blocks
        assert "id" in d and d["id"] == inj.id

    def test_str_marker_format(self):
        inj = LlmBlockInjection(blocks=[], summary="")
        s = str(inj)
        assert s.startswith("[LLM_BLOCKS:")
        assert s.endswith("]")
        assert inj.id in s

    def test_unique_ids(self):
        a = LlmBlockInjection(blocks=[], summary="")
        b = LlmBlockInjection(blocks=[], summary="")
        assert a.id != b.id
