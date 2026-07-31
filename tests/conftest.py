"""Test configuration and fixtures."""

import os
import pytest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, List, Dict, Any

from miiflow_agent.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


# ── Local credentials for the integration tests ──────────────────────────────
#
# The integration tests read provider keys from the environment. Locally the
# monorepo's canonical secrets live in ``server/.env``, so load them here —
# otherwise a fresh checkout skips every real-API test even though the repo is
# fully configured.
#
# Overriding is deliberate: a stale key exported from a shell profile outranks
# the repo's current one and turns these tests RED (a revoked sk-proj-… key
# produced a 401 on every local run of this suite), which is worse than not
# running them at all. For a repo test run, the repo's own file is the
# authority. Files are applied least-specific first, so a package-local
# ``.env`` wins over the server's.
#
# Only the provider keys below are taken. A blanket ``load_dotenv(override=True)``
# would drag the server's entire environment into this process, so the day
# someone adds e.g. MIIFLOW_OPTIMISTIC_ANSWER_STREAMING=0 to server/.env for
# local dev, the unit tests would silently start exercising a different code
# path. Narrow need, narrow mechanism.
#
# In CI there is no ``.env`` on disk, so this is a no-op and the environment
# stays authoritative. Set MIIFLOW_TEST_ENV_OVERRIDE=0 to keep an explicitly
# exported key when you mean to test with a specific one.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_ENV_FILES = (_REPO_ROOT / "server" / ".env", _PACKAGE_ROOT / ".env")
_PROVIDER_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
)


def _load_local_env() -> None:
    if os.getenv("MIIFLOW_TEST_ENV_OVERRIDE") == "0":
        return
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover — dotenv is a declared dependency
        return
    for path in _ENV_FILES:
        if not path.is_file():
            continue
        values = dotenv_values(path)
        for var in _PROVIDER_KEY_VARS:
            value = values.get(var)
            if value:
                os.environ[var] = value


_load_local_env()


# Provider auth failures are an environment precondition, not a defect in the
# code under test. The presence checks below (`has_api_key`) can only see that
# SOME key is set — they cannot tell a live key from a revoked one, so a
# rotated credential surfaces as a failing assertion in whatever suite happens
# to run next. Tests that reach a real provider route auth failures here and
# skip with an actionable message instead. ONLY auth signatures skip;
# everything else re-raises untouched.
_AUTH_ERROR_MARKERS = (
    "authenticationerror",
    "invalid_api_key",
    "incorrect api key",
    "invalid x-api-key",
    "error code: 401",
    "status': 401",
    "permissiondeniederror",
)


def _is_auth_error(exc: BaseException) -> bool:
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        haystack = f"{type(current).__name__} {current}".lower()
        if any(marker in haystack for marker in _AUTH_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def skip_on_provider_auth_error(provider: str, env_var: str):
    """Turn a provider credential failure into a skip, not a red test."""
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 — re-raised unless it's auth
        if not _is_auth_error(exc):
            raise
        pytest.skip(
            f"{provider} rejected the credentials in {env_var} "
            f"(revoked or wrong account). Update it in server/.env, or unset "
            f"a stale export in your shell. Original: {str(exc)[:200]}"
        )


def is_ci_environment() -> bool:
    """Check if running in a CI/CD environment."""
    ci_env_vars = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "JENKINS_URL",
        "BUILDKITE",
        "DRONE",
        "TEAMCITY_VERSION",
        "TF_BUILD",  # Azure DevOps
    ]
    return any(os.getenv(var) for var in ci_env_vars)


def has_api_key(env_var: str) -> bool:
    """Check if an API key environment variable is set and valid."""
    value = os.getenv(env_var)
    if not value:
        return False
    # Check for placeholder values
    if value.startswith("your-") or value.startswith("sk-test") or value == "test":
        return False
    return True


def skip_in_ci_without_api_key(api_key_env: str):
    """Skip test in CI environment if API key is not available."""
    return pytest.mark.skipif(
        is_ci_environment() and not has_api_key(api_key_env),
        reason=f"Skipping in CI: {api_key_env} not configured"
    )


# Register custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_api_key(env_var): mark test as requiring an API key"
    )


@pytest.fixture
def sample_messages() -> List[Message]:
    """Sample messages for testing."""
    return [
        Message.system("You are a helpful assistant."),
        Message.user("Hello, how are you?")
    ]


@pytest.fixture
def sample_usage() -> TokenCount:
    """Sample token usage for testing."""
    return TokenCount(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! I'm doing well, thank you for asking."
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    return mock_response


@pytest.fixture
def mock_openai_stream_chunks():
    """Mock OpenAI streaming response chunks."""
    chunks = []
    
    # First chunk with delta content
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello!"
    chunk1.choices[0].finish_reason = None
    chunks.append(chunk1)
    
    # Second chunk with more content
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " How are you?"
    chunk2.choices[0].finish_reason = None
    chunks.append(chunk2)
    
    # Final chunk with finish_reason
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = None
    chunk3.choices[0].finish_reason = "stop"
    chunk3.usage = MagicMock()
    chunk3.usage.prompt_tokens = 10
    chunk3.usage.completion_tokens = 20
    chunk3.usage.total_tokens = 30
    chunks.append(chunk3)
    
    return chunks


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic chat completion response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Hello! I'm doing well, thank you for asking."
    mock_response.role = "assistant"
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    return mock_response


@pytest.fixture
def mock_anthropic_stream_chunks():
    """Mock Anthropic streaming response chunks."""
    chunks = []
    
    # Content block start
    chunk1 = MagicMock()
    chunk1.type = "content_block_start"
    chunk1.index = 0
    chunk1.content_block.type = "text"
    chunks.append(chunk1)

    # Content block delta
    chunk2 = MagicMock()
    chunk2.type = "content_block_delta"
    chunk2.index = 0
    chunk2.delta.text = "Hello! How are you?"
    chunks.append(chunk2)
    
    # Message stop
    chunk3 = MagicMock()
    chunk3.type = "message_stop"
    chunks.append(chunk3)
    
    return chunks


@pytest.fixture
def expected_chat_response(sample_usage) -> ChatResponse:
    """Expected chat response format."""
    return ChatResponse(
        message=Message.assistant("Hello! I'm doing well, thank you for asking."),
        usage=sample_usage,
        model="gpt-4",
        provider="openai",
        finish_reason="stop"
    )


@pytest.fixture
def expected_stream_chunks() -> List[StreamChunk]:
    """Expected stream chunks format."""
    return [
        StreamChunk(content="Hello!", delta="Hello!"),
        StreamChunk(content="Hello! How are you?", delta=" How are you?"),
        StreamChunk(
            content="Hello! How are you?", 
            delta="", 
            finish_reason="stop",
            usage=TokenCount(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
    ]


class MockAsyncClient:
    """Mock async client for testing providers."""
    
    def __init__(self, responses=None, stream_chunks=None):
        self.responses = responses or []
        self.stream_chunks = stream_chunks or []
        self.call_count = 0
    
    async def create(self, **kwargs):
        """Mock create method."""
        self.call_count += 1
        if self.responses:
            return self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        return MagicMock()
    
    async def stream(self, **kwargs):
        """Mock streaming method."""
        for chunk in self.stream_chunks:
            yield chunk


@pytest.fixture
def mock_async_client():
    """Mock async client fixture."""
    return MockAsyncClient()