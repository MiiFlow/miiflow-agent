"""Tests for provider usage wire-format adapters.

These fields feed two consumers: cost attribution, and the calibration loop
that grounds local token estimates. A silently-dropped field shows up as
mis-billing in one and a permanently-wrong estimator in the other, so the
mapping is worth pinning down precisely.
"""

from types import SimpleNamespace

import pytest

from miiflow_agent.core.metrics import TokenCount

pytestmark = pytest.mark.unit


class TestOpenAIUsage:
    def test_chat_completions_shape(self):
        usage = TokenCount.from_openai_usage(
            SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=768),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=150),
            )
        )
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 200
        assert usage.cache_read_tokens == 768
        assert usage.reasoning_tokens == 150

    def test_responses_api_shape(self):
        usage = TokenCount.from_openai_usage(
            SimpleNamespace(
                input_tokens=500,
                output_tokens=80,
                total_tokens=580,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=256, cache_write_tokens=128
                ),
                output_tokens_details=SimpleNamespace(reasoning_tokens=40),
            )
        )
        assert usage.prompt_tokens == 500
        assert usage.completion_tokens == 80
        assert usage.cache_read_tokens == 256
        assert usage.cache_write_tokens == 128
        assert usage.reasoning_tokens == 40

    def test_missing_details_are_zero_not_error(self):
        usage = TokenCount.from_openai_usage(
            SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        )
        assert usage.cache_read_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_none_usage(self):
        assert TokenCount.from_openai_usage(None) == TokenCount()

    def test_total_derived_when_absent(self):
        usage = TokenCount.from_openai_usage(
            SimpleNamespace(prompt_tokens=100, completion_tokens=25, total_tokens=0)
        )
        assert usage.total_tokens == 125

    def test_zero_prompt_tokens_does_not_fall_through_to_input_tokens(self):
        """`getattr(o, 'a', None) or getattr(o, 'b', None)` would read the
        wrong field when the first is a legitimate zero."""
        usage = TokenCount.from_openai_usage(
            SimpleNamespace(prompt_tokens=0, input_tokens=999, completion_tokens=5)
        )
        assert usage.prompt_tokens == 0

    def test_cached_is_a_subset_not_an_addend(self):
        """OpenAI counts cached_tokens inside prompt_tokens, so the prompt
        total must not be adjusted upward."""
        usage = TokenCount.from_openai_usage(
            SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=0,
                total_tokens=1000,
                prompt_tokens_details=SimpleNamespace(cached_tokens=900),
            )
        )
        assert usage.prompt_tokens == 1000


class TestGeminiUsage:
    def test_rest_dict_shape(self):
        usage = TokenCount.from_gemini_usage(
            {
                "promptTokenCount": 2000,
                "candidatesTokenCount": 300,
                "thoughtsTokenCount": 500,
                "cachedContentTokenCount": 1500,
                "totalTokenCount": 2800,
            }
        )
        assert usage.prompt_tokens == 2000
        assert usage.reasoning_tokens == 500
        assert usage.cache_read_tokens == 1500

    def test_thinking_tokens_added_to_completion(self):
        """thoughtsTokenCount is billed as output but excluded from
        candidatesTokenCount — omitting it makes prompt+completion fail to
        reconcile against the total on every thinking model."""
        usage = TokenCount.from_gemini_usage(
            {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "thoughtsTokenCount": 200,
                "totalTokenCount": 350,
            }
        )
        assert usage.completion_tokens == 250
        assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens

    def test_protobuf_snake_case_shape(self):
        usage = TokenCount.from_gemini_usage(
            SimpleNamespace(
                prompt_token_count=800,
                candidates_token_count=100,
                thoughts_token_count=60,
                cached_content_token_count=400,
                total_token_count=960,
            )
        )
        assert usage.prompt_tokens == 800
        assert usage.reasoning_tokens == 60
        assert usage.cache_read_tokens == 400
        assert usage.completion_tokens == 160

    def test_tool_use_prompt_tokens_folded_into_prompt(self):
        usage = TokenCount.from_gemini_usage(
            {"promptTokenCount": 100, "toolUsePromptTokenCount": 40}
        )
        assert usage.prompt_tokens == 140

    def test_total_derived_when_absent(self):
        usage = TokenCount.from_gemini_usage(
            {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5}
        )
        assert usage.total_tokens == 125

    def test_none_usage(self):
        assert TokenCount.from_gemini_usage(None) == TokenCount()


class TestBilledPromptTokens:
    def test_uses_prompt_tokens_when_inclusive(self):
        usage = TokenCount(prompt_tokens=5000, cache_read_tokens=4000)
        assert usage.billed_prompt_tokens == 5000

    def test_falls_back_to_cache_split(self):
        """Anthropic's input_tokens is the uncached remainder only. If a
        client ever reports the split without folding it in, calibration must
        still see the real prompt size rather than near-zero."""
        usage = TokenCount(
            prompt_tokens=0, cache_read_tokens=30_000, cache_write_tokens=1_000
        )
        assert usage.billed_prompt_tokens == 31_000


class TestUncachedPromptTokens:
    def test_subtracts_cache_split_from_inclusive_prompt(self):
        usage = TokenCount(
            prompt_tokens=10_000, cache_read_tokens=8_000, cache_write_tokens=1_000
        )
        assert usage.uncached_prompt_tokens == 1_000

    def test_no_split_means_everything_uncached(self):
        assert TokenCount(prompt_tokens=5_000).uncached_prompt_tokens == 5_000

    def test_clamps_when_invariant_broken(self):
        """An adapter reporting an EXCLUSIVE prompt count (Anthropic's raw
        input_tokens) would make the subtraction negative; clamp to zero so a
        consumer that misses its own invariant check degrades rather than
        producing a negative token bucket."""
        usage = TokenCount(prompt_tokens=100, cache_read_tokens=30_000)
        assert usage.uncached_prompt_tokens == 0


class TestAddition:
    def test_reasoning_tokens_accumulate(self):
        a = TokenCount(prompt_tokens=1, completion_tokens=2, reasoning_tokens=10)
        b = TokenCount(prompt_tokens=3, completion_tokens=4, reasoning_tokens=20)
        assert (a + b).reasoning_tokens == 30

    def test_addition_tolerates_legacy_tokencount(self):
        """`__add__` reads the new fields via getattr so a TokenCount
        unpickled from an older process doesn't blow up the sum."""

        class Legacy:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        result = TokenCount(prompt_tokens=1, completion_tokens=1) + Legacy()
        assert result.reasoning_tokens == 0
        assert result.prompt_tokens == 2
