"""One reading of a stored ``effort`` on every provider client.

The same stored value must behave the same whichever client an assistant's
model needs, so a model swap never turns a working assistant into one that
fails every turn (see ``miiflow_agent.models.effort``).
"""

import logging

import pytest

from miiflow_agent.models.effort import known_effort_names, normalize_effort, warn_effort_not_sent
from miiflow_agent.providers.anthropic_client import AnthropicClient
from miiflow_agent.providers.gemini_client import GeminiClient
from miiflow_agent.providers.openai_client import OpenAIClient
from miiflow_agent.providers.xai_client import XAIClient

CLIENTS = [
    pytest.param(AnthropicClient, "claude-sonnet-5.5", id="anthropic"),
    pytest.param(OpenAIClient, "gpt-6.1-sol", id="openai"),
    pytest.param(GeminiClient, "gemini-3.1-pro", id="gemini"),
    pytest.param(XAIClient, "grok-4.6", id="xai"),
]


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("   ", None), ("Medium", "medium"), (" HIGH ", "high"), ("auto", "auto")],
)
def test_normalize_effort(value, expected):
    assert normalize_effort(value) == expected


@pytest.mark.parametrize("client_cls, model", CLIENTS)
@pytest.mark.parametrize("stored", ["Medium", " medium ", "MEDIUM"])
def test_every_client_normalises_a_stored_effort(client_cls, model, stored):
    assert client_cls(model=model, api_key="test-key", effort=stored).effort == "medium"


@pytest.mark.parametrize("client_cls, model", CLIENTS)
@pytest.mark.parametrize("stored", ["auto", "extreme", "Max-ish"])
def test_no_client_refuses_an_unknown_stored_effort(client_cls, model, stored):
    # Before, Anthropic and OpenAI raised ValueError here, so the same value a
    # Gemini assistant ran with failed every turn after a swap to Claude/GPT.
    client = client_cls(model=model, api_key="test-key", effort=stored)
    assert client.effort == normalize_effort(stored)


def test_known_names_cover_every_provider_scale():
    from miiflow_agent.models.anthropic import EFFORT_LEVELS
    from miiflow_agent.models.google import THINKING_LEVELS
    from miiflow_agent.models.openai import REASONING_EFFORT_VALUES
    from miiflow_agent.models.xai import reasoning_effort_levels

    names = known_effort_names()
    for scale in (EFFORT_LEVELS, THINKING_LEVELS, REASONING_EFFORT_VALUES, reasoning_effort_levels("grok-4.3")):
        assert set(scale) <= names


def test_not_sending_warns_once_per_provider_model_and_level(caplog):
    warn_effort_not_sent.cache_clear()
    with caplog.at_level(logging.WARNING, logger="miiflow_agent.models.effort"):
        for _ in range(3):
            warn_effort_not_sent("anthropic", "claude-haiku-4.5", "auto", ("low", "high"))
        warn_effort_not_sent("openai", "claude-haiku-4.5", "auto", ("low", "high"))
    assert caplog.text.count("effort='auto'") == 2
    assert "no provider knows that level" in caplog.text


def test_levels_may_be_a_list(caplog):
    # A list is unhashable; logging must never fail the turn it reports on.
    warn_effort_not_sent.cache_clear()
    with caplog.at_level(logging.WARNING, logger="miiflow_agent.models.effort"):
        warn_effort_not_sent("anthropic", "claude-sonnet-4-6", "xhigh", ["low", "medium", "high", "max"])
    assert "effort='xhigh'" in caplog.text


@pytest.mark.parametrize(
    "client_cls, model",
    [pytest.param(OpenAIClient, "gpt-4.1", id="openai"), pytest.param(XAIClient, "grok-3-mini", id="xai")],
)
def test_a_model_with_no_effort_setting_skips_it_silently(client_cls, model, caplog):
    warn_effort_not_sent.cache_clear()
    client = client_cls(model=model, api_key="test-key", effort="medium")
    with caplog.at_level(logging.WARNING, logger="miiflow_agent.models.effort"):
        client._with_configured_effort({})
    assert caplog.text == ""
