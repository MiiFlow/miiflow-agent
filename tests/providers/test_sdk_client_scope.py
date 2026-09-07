import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock

import pytest

from miiflow_agent.providers.sdk_client_cache import get_or_create_sdk_client, sdk_client_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_scoped_client_closes_before_loop_exit_even_on_failure(fail):
    closed_on = []

    class Client:
        async def close(self):
            closed_on.append(asyncio.get_running_loop())

    with pytest.raises(ValueError) if fail else nullcontext():
        async with sdk_client_scope():
            client = await asyncio.to_thread(get_or_create_sdk_client, "test", "key", Client)
            assert get_or_create_sdk_client("test", "key", Client) is client
            if fail:
                raise ValueError("provider failed")
    assert closed_on == [asyncio.get_running_loop()]


@pytest.mark.asyncio
async def test_nested_scopes_do_not_close_outer_or_shared_clients():
    shared = get_or_create_sdk_client("test", "nested", AsyncMock)
    async with sdk_client_scope():
        outer = get_or_create_sdk_client("test", "nested", AsyncMock)
        async with sdk_client_scope():
            inner = get_or_create_sdk_client("test", "nested", AsyncMock)
            assert inner is not outer and outer is not shared
        inner.aclose.assert_awaited_once()
        outer.aclose.assert_not_awaited()
        assert get_or_create_sdk_client("test", "nested", AsyncMock) is outer
    outer.aclose.assert_awaited_once()
    shared.aclose.assert_not_awaited()
    assert get_or_create_sdk_client("test", "nested", AsyncMock) is shared
