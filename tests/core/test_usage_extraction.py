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

    def test_addition_keeps_no_split_as_zero(self):
        assert (TokenCount(prompt_tokens=1) + TokenCount(prompt_tokens=2)).request_prompt_tokens == 0
        mixed = TokenCount(prompt_tokens=600, request_prompt_tokens=150) + TokenCount(prompt_tokens=200)
        assert mixed.sent_prompt_tokens == 350
        unknown = TokenCount(prompt_tokens=900, sampling_iterations=3)
        split = TokenCount(prompt_tokens=600, request_prompt_tokens=150, sampling_iterations=2)
        assert (unknown + split).sent_size_unknown


class TestAnthropicIterations:
    """Server-side tools (tool search, web search, the MCP connector) run as
    extra sampling passes inside ONE Anthropic call, and the top-level usage
    is summed across them. The first `message` pass is the request we sent."""

    @staticmethod
    def _pass(input_tokens, cache_read=0, cache_write=0, kind="message"):
        return SimpleNamespace(
            type=kind,
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        )

    @staticmethod
    def _delta(passes, input_tokens=0, cache_read=0, output_tokens=0):
        return SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=0,
                iterations=passes,
            ),
        )

    def test_first_message_pass_is_the_request(self):
        usage = SimpleNamespace(
            iterations=[self._pass(10, 140_000, 50), self._pass(900, 140_050)]
        )
        assert TokenCount.anthropic_iteration_split(usage) == (140_060, 2)

    def test_dict_shape_and_non_message_passes(self):
        usage = {
            "iterations": [
                {"type": "compaction", "input_tokens": 999_999},
                {"type": "message", "input_tokens": 7, "cache_read_input_tokens": 3},
            ]
        }
        assert TokenCount.anthropic_iteration_split(usage) == (10, 1)

    def test_absent_iterations(self):
        assert TokenCount.anthropic_iteration_split(SimpleNamespace(input_tokens=5)) == (0, 0)
        assert TokenCount.anthropic_iteration_split(SimpleNamespace(iterations=None)) == (0, 0)

    def test_sent_prompt_falls_back_to_billed(self):
        assert TokenCount(prompt_tokens=500).sent_prompt_tokens == 500
        split = TokenCount(prompt_tokens=611_398, request_prompt_tokens=150_000)
        assert split.sent_prompt_tokens == 150_000
        assert split.billed_prompt_tokens == 611_398

    def test_stream_delta_carries_the_request_size(self):
        from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer

        delta = self._delta(
            [self._pass(4, 150_000), self._pass(4, 150_500), self._pass(4, 151_000)],
            input_tokens=12,
            cache_read=451_500,
            output_tokens=40,
        )
        usage = AnthropicStreamNormalizer()._extract_usage(delta)
        assert usage.billed_prompt_tokens == 451_512
        assert usage.sent_prompt_tokens == 150_004
        assert usage.sampling_iterations == 3

    def test_tool_search_split_comes_from_message_start(self):
        """What the API actually sends for tool search (probe, 2026-10-10):
        `iterations` is None, message_start carries the request as sent (765)
        and message_delta the sum over both passes (1,697)."""
        from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer

        normalizer = AnthropicStreamNormalizer()
        start = SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=765, cache_read_input_tokens=0, cache_creation_input_tokens=0, iterations=None,
            )),
        )
        normalizer._extract_usage(start)
        normalizer._handle_content_block_start(SimpleNamespace(
            type="content_block_start", index=1,
            content_block=SimpleNamespace(type="server_tool_use", id="srvtoolu_1", name="tool_search_tool_regex", input={}),
        ))
        usage = normalizer._extract_usage(self._delta(None, input_tokens=1_697, output_tokens=176))
        assert usage.billed_prompt_tokens == 1_697
        assert usage.sent_prompt_tokens == 765
        assert usage.sampling_iterations == 2

    def test_growth_without_a_server_tool_is_not_called_a_split(self):
        from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer

        normalizer = AnthropicStreamNormalizer()
        normalizer._extract_usage(SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=765, cache_read_input_tokens=0, cache_creation_input_tokens=0,
            )),
        ))
        usage = normalizer._extract_usage(self._delta(None, input_tokens=900, output_tokens=5))
        assert usage.request_prompt_tokens == 0
        assert usage.sent_prompt_tokens == 900

    def test_plain_call_has_no_split(self):
        from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer

        normalizer = AnthropicStreamNormalizer()
        normalizer._extract_usage(SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=4, cache_read_input_tokens=20_000, cache_creation_input_tokens=1_000,
            )),
        ))
        delta = SimpleNamespace(type="message_delta", usage=SimpleNamespace(
            input_tokens=4, cache_read_input_tokens=20_000, cache_creation_input_tokens=1_000, output_tokens=9,
        ))
        usage = normalizer._extract_usage(delta)
        assert usage.request_prompt_tokens == 0
        assert usage.sent_prompt_tokens == 21_004

    def test_single_pass_reports_no_split(self):
        from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer

        delta = self._delta([self._pass(100)], input_tokens=100, output_tokens=5)
        usage = AnthropicStreamNormalizer()._extract_usage(delta)
        assert usage.request_prompt_tokens == 0
        assert usage.sent_prompt_tokens == 100
