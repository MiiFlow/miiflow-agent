"""The tools-array fingerprint and its name companion.

The fingerprint is the drift *detector* (order-sensitive, shallow); the names
are the drift *diagnosis* — consumers persist them sparsely so two turns whose
hashes differ can be diffed without re-running anything.
"""

import pytest

from miiflow_agent.core.client import _tools_fingerprint, _tools_fingerprint_and_names

pytestmark = pytest.mark.unit


TOOLS = [
    {"name": "google_ads_query", "description": "Run GAQL.", "input_schema": {}},
    {"name": "meta_ads_insights", "description": "Read Meta insights.", "input_schema": {}},
    {"type": "web_search_20250305"},  # provider-native entry with no name
]


class TestFingerprintAndNames:
    def test_names_are_wire_order(self):
        _, names = _tools_fingerprint_and_names(TOOLS)
        assert names == ["google_ads_query", "meta_ads_insights", "web_search_20250305"]

    def test_wrapper_matches_pair_hash(self):
        digest, _ = _tools_fingerprint_and_names(TOOLS)
        assert _tools_fingerprint(TOOLS) == digest
        assert len(digest) == 12

    def test_reorder_changes_hash_not_name_set(self):
        digest_a, names_a = _tools_fingerprint_and_names(TOOLS)
        digest_b, names_b = _tools_fingerprint_and_names(list(reversed(TOOLS)))
        assert digest_a != digest_b  # tools tier is byte-order-sensitive
        assert sorted(names_a) == sorted(names_b)

    def test_empty_and_none_are_none(self):
        assert _tools_fingerprint_and_names(None) == (None, None)
        assert _tools_fingerprint_and_names([]) == (None, None)

    def test_openai_function_shape(self):
        tools = [{"type": "function", "function": {"name": "f1", "description": "d"}}]
        digest, names = _tools_fingerprint_and_names(tools)
        assert names == ["f1"]
        assert digest is not None


class TestDeferredToolsAreNotCacheBytes:
    """The Anthropic API strips defer_loading tools from the prompt, so churn
    confined to them must NOT move the digest — prod showed hash "drift" with
    an identical name set and a full cache hit, which was this."""

    LOADED = {"name": "core_tool", "description": "Always loaded.", "input_schema": {}}
    DEFERRED = [
        {"name": "mcp_a", "description": "d", "input_schema": {}, "defer_loading": True},
        {"name": "mcp_b", "description": "d", "input_schema": {}, "defer_loading": True},
    ]

    def test_deferred_reorder_keeps_hash(self):
        digest_a, _ = _tools_fingerprint_and_names([self.LOADED, *self.DEFERRED])
        digest_b, _ = _tools_fingerprint_and_names(
            [self.LOADED, *reversed(self.DEFERRED)]
        )
        assert digest_a == digest_b

    def test_deferred_description_edit_keeps_hash(self):
        # The keyword suffix appended to deferred descriptions must not read
        # as cache drift.
        edited = dict(self.DEFERRED[0], description="d [keywords: x, y]")
        digest_a, _ = _tools_fingerprint_and_names([self.LOADED, self.DEFERRED[0]])
        digest_b, _ = _tools_fingerprint_and_names([self.LOADED, edited])
        assert digest_a == digest_b

    def test_deferred_tools_still_listed_in_names(self):
        _, names = _tools_fingerprint_and_names([self.LOADED, *self.DEFERRED])
        assert names == ["core_tool", "mcp_a", "mcp_b"]

    def test_defer_flip_changes_hash(self):
        # A tool moving between deferred and loaded DOES change the cached
        # prefix, so it must change the digest.
        as_deferred = dict(self.LOADED, defer_loading=True)
        digest_a, _ = _tools_fingerprint_and_names([self.LOADED])
        digest_b, _ = _tools_fingerprint_and_names([as_deferred])
        assert digest_a != digest_b

    def test_loaded_churn_still_detected(self):
        digest_a, _ = _tools_fingerprint_and_names([self.LOADED, *self.DEFERRED])
        renamed = dict(self.LOADED, name="other_tool")
        digest_b, _ = _tools_fingerprint_and_names([renamed, *self.DEFERRED])
        assert digest_a != digest_b
