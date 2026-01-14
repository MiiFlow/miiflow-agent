"""Provider configuration fixtures for tests."""

import os
from dataclasses import dataclass
from typing import List, Optional
import pytest


@dataclass
class ProviderConfig:
    """Configuration for a provider in tests."""
    
    name: str
    model: str
    api_key_env: str
    supports_vision: bool = False
    supports_json: bool = True
    supports_streaming: bool = True
    supports_tools: bool = True
    
    def has_api_key(self) -> bool:
        """Check if API key is configured."""
        key = os.getenv(self.api_key_env)
        return bool(key and not key.startswith('your-'))
    
    def get_api_key(self) -> Optional[str]:
        """Get API key if available."""
        if self.has_api_key():
            return os.getenv(self.api_key_env)
        return None
    
    def skip_if_no_key(self):
        """Skip test if API key not available."""
        if not self.has_api_key():
            pytest.skip(f"{self.api_key_env} not configured")


# Provider configurations
ALL_PROVIDERS = [
    ProviderConfig(
        name="openai",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        supports_vision=True,
        supports_json=True,
        supports_streaming=True,
        supports_tools=True
    ),
    ProviderConfig(
        name="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        supports_vision=True,
        supports_json=True,
        supports_streaming=True,
        supports_tools=True
    ),
    ProviderConfig(
        name="gemini",
        model="gemini-2.0-flash-exp",
        api_key_env="GOOGLE_API_KEY",
        supports_vision=True,
        supports_json=True,
        supports_streaming=True,
        supports_tools=True
    ),
    ProviderConfig(
        name="groq",
        model="llama-3.1-8b-instant",
        api_key_env="GROQ_API_KEY",
        supports_vision=False,
        supports_json=False,
        supports_streaming=True,
        supports_tools=True
    ),
    ProviderConfig(
        name="mistral",
        model="mistral-small-latest",
        api_key_env="MISTRAL_API_KEY",
        supports_vision=False,
        supports_json=True,
        supports_streaming=True,
        supports_tools=True
    ),
]


def get_all_providers() -> List[ProviderConfig]:
    """Get all provider configurations."""
    return ALL_PROVIDERS


def get_json_providers() -> List[ProviderConfig]:
    """Get providers that support JSON schema."""
    return [p for p in ALL_PROVIDERS if p.supports_json]


def get_streaming_providers() -> List[ProviderConfig]:
    """Get providers that support streaming."""
    return [p for p in ALL_PROVIDERS if p.supports_streaming]


def get_vision_providers() -> List[ProviderConfig]:
    """Get providers that support vision/multimodal."""
    return [p for p in ALL_PROVIDERS if p.supports_vision]


def get_tool_providers() -> List[ProviderConfig]:
    """Get providers that support tool/function calling."""
    return [p for p in ALL_PROVIDERS if p.supports_tools]


# Pytest fixtures
@pytest.fixture(params=ALL_PROVIDERS, ids=lambda p: p.name)
def provider_config(request) -> ProviderConfig:
    """Fixture that provides each provider configuration."""
    return request.param


@pytest.fixture(params=get_json_providers(), ids=lambda p: p.name)
def json_provider_config(request) -> ProviderConfig:
    """Fixture for JSON-capable providers."""
    return request.param


@pytest.fixture(params=get_vision_providers(), ids=lambda p: p.name)
def vision_provider_config(request) -> ProviderConfig:
    """Fixture for vision-capable providers."""
    return request.param


@pytest.fixture(params=get_tool_providers(), ids=lambda p: p.name)
def tool_provider_config(request) -> ProviderConfig:
    """Fixture for tool-capable providers."""
    return request.param
