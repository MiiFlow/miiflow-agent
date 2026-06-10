"""Tests for AnthropicStreamNormalizer usage extraction with prompt caching.

With prompt caching active, Anthropic splits input tokens into
`input_tokens` (uncached), `cache_creation_input_tokens` and
`cache_read_input_tokens`. The normalizer must sum all three into
`prompt_tokens` — mirroring AnthropicClient.achat — or streaming calls
under-report input as soon as cache hits land. It must also remember the
input-side total from message_start, because message_delta events may
carry only `output_tokens`.
"""

from types import SimpleNamespace

from miiflow_agent.core.stream_normalizer import AnthropicStreamNormalizer


def _message_start(input_tokens=0, cache_creation=0, cache_read=0):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                output_tokens=1,
            )
        ),
    )


def _message_delta(usage):
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="end_turn"),
        usage=usage,
    )


class TestAnthropicStreamUsage:
    def test_cache_tokens_summed_from_message_delta(self):
        normalizer = AnthropicStreamNormalizer()
        chunk = normalizer.normalize_chunk(
            _message_delta(
                SimpleNamespace(
                    input_tokens=100,
                    cache_creation_input_tokens=2000,
                    cache_read_input_tokens=8000,
                    output_tokens=50,
                )
            )
        )

        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 10100
        assert chunk.usage.completion_tokens == 50
        assert chunk.usage.total_tokens == 10150

    def test_input_total_remembered_from_message_start(self):
        # message_delta often carries only output_tokens; the input-side
        # total (including cache buckets) arrives on message_start.
        normalizer = AnthropicStreamNormalizer()
        normalizer.normalize_chunk(
            _message_start(input_tokens=100, cache_creation=2000, cache_read=8000)
        )
        chunk = normalizer.normalize_chunk(
            _message_delta(SimpleNamespace(output_tokens=50))
        )

        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 10100
        assert chunk.usage.completion_tokens == 50

    def test_uncached_stream_unchanged(self):
        normalizer = AnthropicStreamNormalizer()
        chunk = normalizer.normalize_chunk(
            _message_delta(SimpleNamespace(input_tokens=42, output_tokens=7))
        )

        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 42
        assert chunk.usage.completion_tokens == 7

    def test_none_fields_are_safe(self):
        # Bedrock can return a usage object whose fields are None.
        normalizer = AnthropicStreamNormalizer()
        chunk = normalizer.normalize_chunk(
            _message_delta(
                SimpleNamespace(
                    input_tokens=None,
                    cache_creation_input_tokens=None,
                    cache_read_input_tokens=None,
                    output_tokens=None,
                )
            )
        )

        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 0
        assert chunk.usage.completion_tokens == 0

    def test_state_reset_clears_input_total(self):
        normalizer = AnthropicStreamNormalizer()
        normalizer.normalize_chunk(_message_start(input_tokens=500))
        normalizer.reset_state()
        chunk = normalizer.normalize_chunk(
            _message_delta(SimpleNamespace(output_tokens=5))
        )

        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 0
