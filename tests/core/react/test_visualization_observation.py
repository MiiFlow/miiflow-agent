"""What the model is told after a tool returned a visualization.

A `[VIZ:id]` marker is a handle to something the USER can see. For a chart the
observation is that handle plus the one rule for holding it: the visual ships
only where the marker is embedded, and an unembedded render is dropped. That
sentence is what makes revising safe — before it, a strategist that rendered
the same KPI card three times (refining labels) and embedded only the last saw
the two abandoned drafts appended under its answer as duplicates
(`thread_rQCbwiTgYvPjYSFZTl1DrFbd`, 2026-08-17). For an `auth_prompt` the bare
marker is a lie of omission: the tool returned no data because a provider is
not connected, and a marker that reads as "visualization generated" invites the
model to keep going as though it had results.

The two result paths in the orchestrator had drifted — the single-tool path
explained the auth case, the batch path emitted the bare marker — so the same
blocked tool did or didn't explain itself depending on whether the model
happened to call it alongside another one. These tests pin them together.
"""

from miiflow_agent.core.react.orchestrator import (
    VISUALIZATION_MARKER_CONTRACT,
    visualization_observation,
)


CHART = {"__visualization__": True, "id": "viz-1", "type": "bar_chart", "data": {}}
AUTH_PROMPT = {
    "__visualization__": True,
    "id": "mcp-auth-srv_1",
    "type": "auth_prompt",
    "data": {"providerName": "GitHub", "mcpServerId": "srv_1"},
}


class TestVisualizationObservation:
    def test_a_chart_is_its_marker_plus_the_embed_rule(self):
        observation = visualization_observation(CHART)
        assert observation.startswith("[VIZ:viz-1] ")
        assert observation == f"[VIZ:viz-1] {VISUALIZATION_MARKER_CONTRACT}"

    def test_the_embed_rule_says_unembedded_renders_are_dropped(self):
        # The model must learn what becomes of a render it does NOT embed,
        # or it has no safe way to revise a visual.
        assert "not embedded is dropped" in VISUALIZATION_MARKER_CONTRACT
        assert "embed only the final marker" in VISUALIZATION_MARKER_CONTRACT

    def test_an_auth_prompt_does_not_get_the_embed_rule(self):
        # The host shows the connect card regardless of embedding, so telling
        # the model an unembedded auth card is dropped would be false.
        assert VISUALIZATION_MARKER_CONTRACT not in visualization_observation(AUTH_PROMPT)

    def test_an_auth_prompt_says_no_data_was_returned(self):
        observation = visualization_observation(AUTH_PROMPT)
        assert observation.startswith("[VIZ:mcp-auth-srv_1]")
        assert "No data was returned" in observation
        assert "GitHub" in observation

    def test_an_auth_prompt_never_reads_as_success(self):
        # The specific regression: the batch path's bare marker was later
        # rendered as "Visualization generated successfully."
        assert "success" not in visualization_observation(AUTH_PROMPT).lower()

    def test_it_tells_the_model_to_stop_retrying_that_provider(self):
        # Without this the model burns its remaining iterations re-calling a
        # tool whose credential cannot appear mid-run.
        observation = visualization_observation(AUTH_PROMPT)
        assert "Do not retry" in observation

    def test_a_nameless_provider_does_not_crash_the_turn(self):
        observation = visualization_observation(
            {"__visualization__": True, "id": "x", "type": "auth_prompt", "data": {}}
        )
        assert "the provider" in observation

    def test_a_missing_id_does_not_crash_the_turn(self):
        # The replay path in enhanced_response_generator hands this helper a
        # stored output dict, which is not guaranteed to carry an id.
        assert visualization_observation({"type": "bar_chart"}).startswith("[VIZ:unknown] ")
