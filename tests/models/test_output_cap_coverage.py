"""Which provider output-cap maps are guarded, and which are known-divergent.

`assert_output_cap_map_matches_catalog` is applied to OpenAI (both of its
maps), Anthropic, Google, DeepSeek and xAI — xAI's own test was moved onto the
shared helper so that claim is exact rather than approximate. The one provider
in this package it is NOT applied to is GROQ, because Groq's map already
disagrees with its own catalog and fixing that needs Groq's published limits
rather than a guess. Note the Groq exclusion covers VALUES only: the pin below
still requires every Groq model to HAVE a key, so a newly added one cannot
silently inherit the default.

This module exists so that exclusion is a recorded fact with a test that fails
when it changes, rather than an unstated assumption: the November 23, 2026
audit first claimed "every provider is now guarded", which was false, and Groq
is why. Note also that the token parameter's NAME differs by provider — Groq
uses `max_tokens`, DeepSeek `max_completion_tokens` — so a provider cannot be
added to the guarded set by copying a sibling's field name.
"""

import pytest

from tests.utils.assertions import assert_output_cap_map_matches_catalog

from miiflow_agent.models.groq import GROQ_MODELS, GROQ_PARAMETERS
from miiflow_agent.models.deepseek import DEEPSEEK_MODELS, DEEPSEEK_PARAMETERS


# Groq's `max_tokens` map disagrees with its own catalog on 6 of its 8 keyed
# models, and it disagrees in the OVER-ADVERTISING direction — the one that
# produces provider 400s rather than short answers. Every Groq entry declares
# `maximum_output_tokens=4096`, but the map hands four of them 8192 and two of
# them 32768. Two of those numbers are visibly copied from the MODEL NAME
# rather than from the model (`llama3-70b-8192` -> 8192,
# `mixtral-8x7b-32768` -> 32768), which is the likeliest origin of the whole
# divergence.
#
# WHICH SIDE IS WRONG IS NOT KNOWN and that is exactly why this is pinned
# rather than fixed here: it needs Groq's published per-model output limits,
# and Groq is outside the provider set the audit that found this was auditing
# (OpenAI, Anthropic, Google, Bedrock). A blind "fix" either truncates real
# capacity (if 4096 is the stale number) or keeps shipping 400s (if the map
# is). Resolve it against Groq's docs in a change of its own, then delete this
# test and add Groq to the guarded set above — at which point the presence
# check inside it becomes redundant, because the shared helper does both.
_GROQ_KNOWN_DIVERGENCES = {
    "llama3-70b-8192": (4096, 8192),
    "llama-3.3-70b-versatile": (4096, 32768),
    "llama-guard-3-8b": (4096, 8192),
    "llama-3.1-8b-instant": (4096, 8192),
    "mixtral-8x7b-32768": (4096, 32768),
    "gemma2-9b-it": (4096, 8192),
}


def test_groq_output_cap_map_still_diverges_exactly_as_recorded():
    """Pins the known Groq divergence so it cannot grow or be forgotten.

    This test PASSES while the bug is present, which is unusual and
    deliberate: it is a record, not an approval. It fails if a Groq entry
    starts or stops diverging, which is the moment someone should either
    finish the fix or update this record.
    """
    limits = next(p for p in GROQ_PARAMETERS if p.field_name == "max_tokens").max_value

    # Every Groq model must at least HAVE a key, checked separately from the
    # values. Without this, the divergence comparison below — which can only
    # look at models already in the map — would quietly ignore a NEW Groq model
    # added without one, and that model is the case this whole module is about:
    # it would inherit the 8192 default against whatever it really declares.
    # So Groq is unguarded on VALUES only, not on presence.
    missing = sorted(set(GROQ_MODELS) - set(limits))
    assert not missing, f"Groq models absent from the max_tokens map: {missing}"

    observed = {
        name: (config.maximum_output_tokens, limits[name])
        for name, config in GROQ_MODELS.items()
        if limits[name] != config.maximum_output_tokens
    }
    assert observed == _GROQ_KNOWN_DIVERGENCES

    # And the shared guard really would catch it, so the exclusion above is
    # the only reason Groq passes CI — not a hole in the helper.
    with pytest.raises(AssertionError):
        assert_output_cap_map_matches_catalog(GROQ_MODELS, GROQ_PARAMETERS, "max_tokens")


def test_deepseek_output_cap_map_matches_its_catalog():
    """DeepSeek was unguarded but not divergent, so it simply joins the set.

    Its `max_completion_tokens` map already agrees with its catalog, which is
    why guarding it is a one-line addition rather than a fix. Recorded as its
    own test because the audit that added the shared helper initially listed
    DeepSeek as "no per-model map" — it has one, on `max_completion_tokens`
    rather than `max_tokens`, which is why reading the parameter name off a
    sibling provider is not safe.
    """
    assert_output_cap_map_matches_catalog(
        DEEPSEEK_MODELS, DEEPSEEK_PARAMETERS, "max_completion_tokens"
    )
