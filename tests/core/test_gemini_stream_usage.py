"""Gemini streams repeat usage running totals on every chunk.

Measured 2026-11-07: a 2,232-token prompt streamed in 7 chunks carried
promptTokenCount 2232 on all seven and a growing candidatesTokenCount. The
client adds chunk usages up for billing, so the prompt was billed once per
chunk (15,624 tokens). The normalizer now reports what each chunk added.
"""

from miiflow_agent.core.metrics import TokenCount
from miiflow_agent.core.stream_normalizer import GeminiStreamNormalizer


def _chunk(text, prompt, candidates, thoughts=0, finish=None):
    candidate = {"content": {"parts": [{"text": text}]}}
    if finish:
        candidate["finishReason"] = finish
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": candidates, "totalTokenCount": prompt + candidates + thoughts}
    if thoughts:
        usage["thoughtsTokenCount"] = thoughts
    return {"candidates": [candidate], "usageMetadata": usage}


def _stream(normalizer, chunks):
    normalizer.reset_state()
    return [normalizer.normalize_chunk(c) for c in chunks]


def test_running_totals_become_increments_that_sum_to_the_real_usage():
    normalizer = GeminiStreamNormalizer()
    chunks = [
        _chunk("1", 2232, 2),
        _chunk(" 2", 2232, 20),
        _chunk(" 3", 2232, 44, thoughts=10),
        _chunk("", 2232, 109, thoughts=10, finish="STOP"),
        _chunk("", 2232, 109, thoughts=10),  # trailing repeat of the final totals
    ]
    out = _stream(normalizer, chunks)
    total = TokenCount()
    for chunk in out:
        if chunk.usage:
            total = total + chunk.usage
    assert total.prompt_tokens == 2232
    assert total.completion_tokens == 109 + 10  # thinking is billed as output
    assert total.reasoning_tokens == 10
    # Only the first chunk carries the prompt; a repeat of the same totals adds nothing.
    assert [c.usage.prompt_tokens if c.usage else None for c in out] == [2232, 0, 0, 0, None]


def test_each_stream_starts_from_zero():
    normalizer = GeminiStreamNormalizer()
    _stream(normalizer, [_chunk("a", 100, 5)])
    second = _stream(normalizer, [_chunk("b", 300, 7)])
    assert (second[0].usage.prompt_tokens, second[0].usage.completion_tokens) == (300, 7)


def test_a_field_left_out_of_one_chunk_is_not_counted_again():
    normalizer = GeminiStreamNormalizer()
    out = _stream(normalizer, [_chunk("a", 100, 5, thoughts=10), _chunk("b", 100, 8), _chunk("", 100, 8, thoughts=10, finish="STOP")])
    total = TokenCount()
    for chunk in out:
        if chunk.usage:
            total = total + chunk.usage
    assert (total.prompt_tokens, total.reasoning_tokens) == (100, 10)
