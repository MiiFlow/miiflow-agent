"""Per-event-loop cache of provider SDK clients.

SDK clients (``anthropic.AsyncAnthropic``, ``openai.AsyncOpenAI``, ...) hold
an httpx connection pool. Constructing a fresh one per agent turn throws that
pool away, so every run pays a new TLS handshake to the provider on its first
LLM call — directly on time-to-first-token. Reusing the SDK client keeps the
connection warm across turns.

Only the SDK client is cached: the ModelClient / LLMClient wrappers around it
carry per-run mutable state (tool registries, name mappings, normalizers) and
must stay per-turn.

The cache is keyed per event loop because httpx async transports are bound to
the loop they first run on — handing a cached client to a different loop
(e.g. a Celery task's fresh ``asyncio.run`` loop) raises. Entries die with
their loop via the WeakKeyDictionary. Keyed per api_key so organizations with
their own credentials never share a client. Callers in a context with no
running loop get an uncached instance.

Eviction: each per-loop cache is an LRU capped at ``_MAX_CLIENTS_PER_LOOP`` so
key rotations in long-lived multi-tenant workers cannot accumulate unbounded
idle connection pools. An evicted client may still serve an in-flight stream,
so it is closed on its loop after ``_EVICTED_CLOSE_DELAY_SECONDS`` (longer
than any run's stream) rather than immediately.
"""

import asyncio
import logging
import weakref
from collections import OrderedDict
from typing import Any, Callable, Tuple

logger = logging.getLogger(__name__)

_MAX_CLIENTS_PER_LOOP = 32
_EVICTED_CLOSE_DELAY_SECONDS = 15 * 60

_cache: "weakref.WeakKeyDictionary[Any, OrderedDict[Tuple[str, str], Any]]" = (
    weakref.WeakKeyDictionary()
)


def _schedule_delayed_close(loop, client: Any) -> None:
    """Close an evicted client on its loop after a grace period."""

    def _spawn_close() -> None:
        aclose = getattr(client, "aclose", None) or getattr(client, "close", None)
        if aclose is None:
            return

        async def _close() -> None:
            try:
                result = aclose()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.debug("evicted SDK client close failed", exc_info=True)

        loop.create_task(_close())

    try:
        loop.call_later(_EVICTED_CLOSE_DELAY_SECONDS, _spawn_close)
    except Exception:  # noqa: BLE001
        # Loop already closing — its transports die with it anyway.
        logger.debug("could not schedule evicted SDK client close", exc_info=True)


def get_or_create_sdk_client(
    provider: str, api_key: str, factory: Callable[[], Any]
) -> Any:
    """Return the cached SDK client for (provider, api_key) on the running
    loop, creating it via ``factory`` on first use."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return factory()

    per_loop = _cache.get(loop)
    if per_loop is None:
        per_loop = OrderedDict()
        _cache[loop] = per_loop
    key = (provider, api_key or "")
    client = per_loop.get(key)
    if client is None:
        client = factory()
        per_loop[key] = client
        while len(per_loop) > _MAX_CLIENTS_PER_LOOP:
            _evicted_key, evicted = per_loop.popitem(last=False)
            _schedule_delayed_close(loop, evicted)
    else:
        per_loop.move_to_end(key)
    return client
