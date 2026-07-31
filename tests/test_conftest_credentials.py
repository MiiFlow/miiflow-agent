"""Cover the test-harness credential plumbing in ``tests/conftest.py``.

This code decides whether an integration test runs, skips, or fails, so it
needs cover of its own — a skip helper that swallowed real failures would
turn the suite green while the code under test was broken.

Background: a revoked ``sk-proj-…`` key exported from a shell profile outranked
the repo's current key in ``server/.env``, so ``test_autonomous_tool_calling``
401'd on every local run. The presence check (`has_api_key`) cannot tell a live
key from a revoked one, which is why validity is handled at the call site.
"""

import os

import pytest

from tests.conftest import (
    _ENV_FILES,
    _PROVIDER_KEY_VARS,
    _is_auth_error,
    _load_local_env,
    has_api_key,
    skip_on_provider_auth_error,
)


class TestAuthErrorDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-proj-…'}}",
            "invalid_api_key",
            "invalid x-api-key",
        ],
    )
    def test_recognizes_provider_auth_failures(self, message):
        assert _is_auth_error(RuntimeError(message)) is True

    def test_recognizes_by_exception_type_name(self):
        class AuthenticationError(Exception):
            pass

        assert _is_auth_error(AuthenticationError("no detail")) is True

    def test_finds_auth_error_through_the_cause_chain(self):
        """agent.run() re-raises as MiiflowLLMError, so the 401 is nested."""
        try:
            try:
                raise RuntimeError("Error code: 401 - incorrect api key")
            except RuntimeError as inner:
                raise ValueError("Agent failed after 1 retries") from inner
        except ValueError as exc:
            assert _is_auth_error(exc) is True

    def test_does_not_flag_a_cyclic_chain(self):
        exc = RuntimeError("boom")
        exc.__cause__ = exc  # pathological, but must terminate
        assert _is_auth_error(exc) is False

    @pytest.mark.parametrize(
        "message",
        [
            "rate limit exceeded",
            "Error code: 500 - internal server error",
            "connection reset by peer",
            "AssertionError: LLM did not autonomously call either tool",
        ],
    )
    def test_ignores_everything_that_is_not_auth(self, message):
        assert _is_auth_error(RuntimeError(message)) is False


class TestSkipOnProviderAuthError:
    def test_auth_failure_becomes_a_skip(self):
        with pytest.raises(pytest.skip.Exception) as excinfo:
            with skip_on_provider_auth_error("openai", "OPENAI_API_KEY"):
                raise RuntimeError("Error code: 401 - Incorrect API key provided")
        # The message has to be actionable: which key, and where to fix it.
        assert "OPENAI_API_KEY" in str(excinfo.value)
        assert "server/.env" in str(excinfo.value)

    def test_real_failures_still_propagate(self):
        """The load-bearing property: this must not become a blanket
        try/except that hides a genuinely broken agent."""
        with pytest.raises(AssertionError, match="did not call the tool"):
            with skip_on_provider_auth_error("openai", "OPENAI_API_KEY"):
                raise AssertionError("did not call the tool")

    def test_success_passes_through_untouched(self):
        with skip_on_provider_auth_error("openai", "OPENAI_API_KEY"):
            value = 1 + 1
        assert value == 2


class TestLocalEnvLoading:
    @pytest.fixture(autouse=True)
    def _neutral_opt_out(self, monkeypatch):
        """Control the opt-out explicitly rather than inheriting it.

        Without this, running the suite with MIIFLOW_TEST_ENV_OVERRIDE=0 in the
        environment made every loading test fail — the tests were asserting on
        behaviour they had not pinned down.
        """
        monkeypatch.delenv("MIIFLOW_TEST_ENV_OVERRIDE", raising=False)

    def test_only_provider_keys_are_imported(self, monkeypatch, tmp_path):
        """A blanket load would drag the server's whole environment in here."""
        env = tmp_path / ".env"
        env.write_text(
            "OPENAI_API_KEY=sk-from-file\n"
            "MIIFLOW_OPTIMISTIC_ANSWER_STREAMING=0\n"
            "DATABASE_URL=postgres://nope\n"
        )
        monkeypatch.setattr("tests.conftest._ENV_FILES", (env,))
        monkeypatch.delenv("MIIFLOW_OPTIMISTIC_ANSWER_STREAMING", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-from-shell")

        _load_local_env()

        assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
        assert "MIIFLOW_OPTIMISTIC_ANSWER_STREAMING" not in os.environ
        assert "DATABASE_URL" not in os.environ

    def test_file_beats_a_stale_shell_export(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-current\n")
        monkeypatch.setattr("tests.conftest._ENV_FILES", (env,))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-revoked")

        _load_local_env()

        assert os.environ["OPENAI_API_KEY"] == "sk-current"

    def test_later_files_win(self, monkeypatch, tmp_path):
        """Package-local .env overrides the server's."""
        first = tmp_path / "server.env"
        first.write_text("OPENAI_API_KEY=sk-server\n")
        second = tmp_path / "package.env"
        second.write_text("OPENAI_API_KEY=sk-package\n")
        monkeypatch.setattr("tests.conftest._ENV_FILES", (first, second))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        _load_local_env()

        assert os.environ["OPENAI_API_KEY"] == "sk-package"

    def test_opt_out_keeps_the_shell_authoritative(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-from-file\n")
        monkeypatch.setattr("tests.conftest._ENV_FILES", (env,))
        monkeypatch.setenv("MIIFLOW_TEST_ENV_OVERRIDE", "0")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-deliberate")

        _load_local_env()

        assert os.environ["OPENAI_API_KEY"] == "sk-deliberate"

    def test_missing_files_are_not_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tests.conftest._ENV_FILES", (tmp_path / "absent",))
        _load_local_env()  # CI has no .env on disk; must be a silent no-op

    def test_env_file_list_points_at_the_repo_secrets(self):
        assert any(p.parts[-2:] == ("server", ".env") for p in _ENV_FILES)

    def test_placeholder_keys_are_still_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "your-key-here")
        assert has_api_key("OPENAI_API_KEY") is False
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-realish")
        assert has_api_key("OPENAI_API_KEY") is True

    def test_openai_is_among_the_imported_keys(self):
        assert "OPENAI_API_KEY" in _PROVIDER_KEY_VARS
