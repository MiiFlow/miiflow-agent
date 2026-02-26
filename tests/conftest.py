"""Test configuration and fixtures."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, List, Dict, Any

from miiflow_agent.core import Message, MessageRole, TokenCount, StreamChunk, ChatResponse


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