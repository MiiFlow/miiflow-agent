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
