"""Tests for run-scoped callback registries.

The global registry is process-wide: under concurrent ASGI requests, one
request's PRE_TOOL_USE approval gate registered there is visible to every
other request in the process (isolated_scope only protects *nesting*).
scoped_callbacks() gives a run its own ContextVar-scoped registry that
chains to the global for emissions but keeps registrations local.
"""

import asyncio

import pytest

from miiflow_agent.core.callbacks import (
    CallbackEvent,
    CallbackEventType,
    CallbackRegistry,
    get_active_registry,
    get_global_registry,
    scoped_callbacks,
    scoped_callbacks_stream,
)


def _event():
    return CallbackEvent(event_type=CallbackEventType.POST_CALL)


class TestScopedCallbacks:
    def test_active_registry_defaults_to_global(self):
        assert get_active_registry() is get_global_registry()

    async def test_scoped_registration_is_invisible_outside(self):
        seen = []

        with scoped_callbacks() as cbs:
            cbs.register(CallbackEventType.POST_CALL, lambda e: seen.append("in"))
            await get_active_registry().emit(_event())

        # Back outside the scope: the run-local callback no longer fires.
        await get_active_registry().emit(_event())
        assert seen == ["in"]

    async def test_global_listeners_still_fire_inside_scope(self):
        seen = []

        def global_listener(event):
            seen.append("global")

        get_global_registry().register(CallbackEventType.POST_CALL, global_listener)
        try:
            with scoped_callbacks() as cbs:
                cbs.register(
                    CallbackEventType.POST_CALL, lambda e: seen.append("scoped")
                )
                await get_active_registry().emit(_event())
        finally:
            get_global_registry().unregister(
                CallbackEventType.POST_CALL, global_listener
            )

        # Parent (global) listeners run first, then the scoped one.
        assert seen == ["global", "scoped"]

    async def test_scoped_clear_cannot_wipe_global(self):
        """clear() inside a scope SHADOWS inherited listeners for that scope
        (replace-the-policy semantics) but never mutates the global registry:
        outside the scope everything is intact."""

        def global_listener(event):
            pass

        get_global_registry().register(CallbackEventType.POST_CALL, global_listener)
        try:
            with scoped_callbacks() as cbs:
                cbs.clear()
                # Shadowed inside the scope...
                assert cbs.get_callbacks(CallbackEventType.POST_CALL) == []
            # ...but the global registry itself was never touched.
            assert global_listener in get_global_registry().get_callbacks(
                CallbackEventType.POST_CALL
            )
        finally:
            get_global_registry().unregister(
                CallbackEventType.POST_CALL, global_listener
            )

    async def test_concurrent_scopes_do_not_leak(self):
        """Two concurrent 'requests' each register their own gate; neither
        sees the other's — the exact ASGI cross-request hazard."""
        observed = {"a": [], "b": []}

        async def request(name):
            with scoped_callbacks() as cbs:
                cbs.register(
                    CallbackEventType.PRE_TOOL_USE,
                    lambda e, n=name: observed[n].append(n),
                )
                await asyncio.sleep(0.01)  # interleave the two scopes
                await get_active_registry().emit(
                    CallbackEvent(event_type=CallbackEventType.PRE_TOOL_USE)
                )

        # Each task gets its own Context copy, hence its own scope.
        await asyncio.gather(request("a"), request("b"))

        assert observed["a"] == ["a"]
        assert observed["b"] == ["b"]

    async def test_scope_visible_in_forked_tasks(self):
        """A task created inside the scope inherits the Context snapshot and
        therefore the scoped registry — matching how parallel tool branches
        are spawned."""
        seen = []

        with scoped_callbacks() as cbs:
            cbs.register(CallbackEventType.POST_CALL, lambda e: seen.append(1))

            async def branch():
                await get_active_registry().emit(_event())

            await asyncio.create_task(branch())

        assert seen == [1]

    async def test_custom_registry_can_be_provided(self):
        custom = CallbackRegistry(parent=get_global_registry())
        with scoped_callbacks(custom) as cbs:
            assert cbs is custom
            assert get_active_registry() is custom


class TestNestedScopes:
    """A dispatched child enters its own scope inside the parent turn's
    scope. The child must inherit the parent's listeners (token tracking
    bills child LLM calls to the parent's turn) while being able to REPLACE
    inherited policies via clear() (the parent's approval gate must not fire
    for the child's tools)."""

    async def test_child_scope_chains_to_parent_scope(self):
        seen = []

        with scoped_callbacks() as parent:
            parent.register(
                CallbackEventType.POST_CALL, lambda e: seen.append("parent")
            )
            with scoped_callbacks() as child:
                child.register(
                    CallbackEventType.POST_CALL, lambda e: seen.append("child")
                )
                await get_active_registry().emit(_event())

        # Parent-scoped listener fired for the child's event, then the
        # child's own.
        assert seen == ["parent", "child"]

    async def test_shadow_clear_replaces_inherited_policy(self):
        pre = CallbackEventType.PRE_TOOL_USE
        post = CallbackEventType.POST_CALL
        seen = []

        with scoped_callbacks() as parent:
            parent.register(pre, lambda e: seen.append("parent-gate"))
            parent.register(post, lambda e: seen.append("parent-tokens"))

            with scoped_callbacks() as child:
                # The child replaces the approval gate...
                child.clear(pre)
                child.register(pre, lambda e: seen.append("child-gate"))

                await get_active_registry().emit(CallbackEvent(event_type=pre))
                await get_active_registry().emit(CallbackEvent(event_type=post))

            # ...and back in the parent scope, the parent's gate is intact.
            seen.append("--exit--")
            await get_active_registry().emit(CallbackEvent(event_type=pre))

        # Inside the child: only the child's gate fired for PRE_TOOL_USE,
        # but the parent's token tracking still fired for POST_CALL.
        assert seen == [
            "child-gate",
            "parent-tokens",
            "--exit--",
            "parent-gate",
        ]

    async def test_clear_all_shadows_everything(self):
        seen = []

        with scoped_callbacks() as parent:
            parent.register(
                CallbackEventType.POST_CALL, lambda e: seen.append("parent")
            )
            with scoped_callbacks() as child:
                child.clear()
                await get_active_registry().emit(_event())
        assert seen == []

    async def test_global_clear_semantics_unchanged(self):
        """clear() on the parentless global registry behaves exactly as
        before — no shadow bookkeeping involved."""
        def listener(event):
            pass

        registry = get_global_registry()
        registry.register(CallbackEventType.POST_CALL, listener)
        registry.clear(CallbackEventType.POST_CALL)
        assert registry.get_callbacks(CallbackEventType.POST_CALL) == []
        assert registry._shadowed == set()


class TestScopedCallbacksStream:
    """The async-generator-safe form of scoped_callbacks: the registry is
    activated inside each advancement (traced_stream mechanics), so a stream
    driven from different tasks can never reset a token across contexts."""

    async def test_scope_active_inside_body_inactive_between_yields(self):
        observed = {}

        async def body():
            observed["inside"] = get_active_registry()
            yield "a"
            observed["inside_again"] = get_active_registry()
            yield "b"

        events = []
        async for event in scoped_callbacks_stream(body()):
            # Between yields the consumer sees the previous (global) registry.
            assert get_active_registry() is get_global_registry()
            events.append(event)

        assert events == ["a", "b"]
        assert observed["inside"] is not get_global_registry()
        assert observed["inside"] is observed["inside_again"]
        assert observed["inside"]._parent is get_global_registry()

    async def test_caller_owned_registry_is_activated(self):
        turn_registry = CallbackRegistry(parent=get_global_registry())
        seen = []
        turn_registry.register(
            CallbackEventType.POST_CALL, lambda e: seen.append("turn")
        )

        async def body():
            await get_active_registry().emit(
                CallbackEvent(event_type=CallbackEventType.POST_CALL)
            )
            yield "done"

        events = [
            e async for e in scoped_callbacks_stream(body(), registry=turn_registry)
        ]
        assert events == ["done"]
        assert seen == ["turn"]

    async def test_cross_task_driving_is_safe(self):
        """Each __anext__ driven by a fresh task — the exact shape that broke
        the with-around-yield form (token reset in a different Context)."""
        observed = []

        async def body():
            for i in range(3):
                observed.append(get_active_registry() is not get_global_registry())
                yield i

        wrapper = scoped_callbacks_stream(body())
        iterator = wrapper.__aiter__()
        events = []
        while True:
            try:
                # A new task per advancement = a new Context per advancement.
                events.append(await asyncio.create_task(iterator.__anext__()))
            except StopAsyncIteration:
                break

        assert events == [0, 1, 2]
        assert observed == [True, True, True]

    async def test_early_close_closes_inner_generator(self):
        closed = {"inner": False}

        async def body():
            try:
                yield 1
                yield 2
            finally:
                closed["inner"] = True

        wrapper = scoped_callbacks_stream(body())
        async for event in wrapper:
            break  # abandon early
        await wrapper.aclose()
        assert closed["inner"] is True


    async def test_teardown_emissions_stay_in_scope(self):
        """An abandoned stream's inner finally may emit real events (an
        interrupted LLM call's POST_CALL with partial usage). The scope must
        be active during the deterministic close so run-scoped listeners
        (billing) still hear them."""
        seen = []
        turn_registry = CallbackRegistry(parent=get_global_registry())
        turn_registry.register(
            CallbackEventType.POST_CALL, lambda e: seen.append("turn")
        )

        async def body():
            try:
                yield "first"
                yield "second"
            finally:
                # Mirrors LLMClient.astream_chat's finally-block emission.
                await get_active_registry().emit(_event())

        wrapper = scoped_callbacks_stream(body(), registry=turn_registry)
        async for _ in wrapper:
            break  # consumer disconnects mid-stream
        await wrapper.aclose()

        assert seen == ["turn"]
