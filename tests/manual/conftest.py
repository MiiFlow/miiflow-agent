"""Configuration for manual tests.

These tests are meant to be run manually with real API keys, not as part of CI/CD.
They are skipped when collected by pytest.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip all items in the manual folder unless explicitly requested."""
    skip_manual = pytest.mark.skip(
        reason="Manual tests require API keys and are not run in CI/CD. "
        "Run directly with: python tests/manual/test_*.py"
    )
    for item in items:
        if "manual" in str(item.fspath):
            item.add_marker(skip_manual)
