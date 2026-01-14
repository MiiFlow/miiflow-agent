"""Provider registry for auto-discovery and configuration."""

from typing import Dict, Optional, Type, Any
from dataclasses import dataclass


@dataclass
class ProviderMetadata:
    """Provider configuration metadata."""
    name: str
    client_class: Type
    normalizer_class: Optional[Type] = None
    requires_api_key: bool = True
    special_auth: Optional[Dict[str, Any]] = None


class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, ProviderMetadata] = {}

    def register(
        self,
        name: str,
        client_class: Type,
        normalizer_class: Optional[Type] = None,
        requires_api_key: bool = True,
        special_auth: Optional[Dict[str, Any]] = None
    ):
        metadata = ProviderMetadata(
            name=name,
            client_class=client_class,
            normalizer_class=normalizer_class,
            requires_api_key=requires_api_key,
            special_auth=special_auth
        )
        self._providers[name.lower()] = metadata

    def get(self, name: str) -> Optional[ProviderMetadata]:
        return self._providers.get(name.lower())

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def create_client(self, provider: str, model: str, api_key: Optional[str] = None, **kwargs):
        metadata = self.get(provider)
        if not metadata:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {', '.join(self.list_providers())}"
            )

        if metadata.special_auth:
            required_creds = metadata.special_auth.get('required_fields', [])
            missing = [k for k in required_creds if k not in kwargs]
            if missing:
                raise ValueError(f"{provider} requires: {', '.join(missing)}")
            return metadata.client_class(model=model, **kwargs)

        if metadata.requires_api_key and not api_key and provider != "ollama":
            raise ValueError(f"{provider} requires api_key")

        return metadata.client_class(model=model, api_key=api_key, **kwargs)


_registry = ProviderRegistry()


def register_provider(
    name: str,
    normalizer_class: Optional[Type] = None,
    requires_api_key: bool = True,
    special_auth: Optional[Dict[str, Any]] = None
):
    def decorator(client_class: Type):
        _registry.register(
            name=name,
            client_class=client_class,
            normalizer_class=normalizer_class,
            requires_api_key=requires_api_key,
            special_auth=special_auth
        )
        client_class._provider_name = name
        client_class._normalizer_class = normalizer_class
        return client_class
    return decorator


def get_registry() -> ProviderRegistry:
    return _registry
