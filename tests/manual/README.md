# Manual Tests

These standalone test scripts mirror the Jupyter notebooks in `examples/notebooks/` to verify that the notebooks will work correctly.

## Purpose

If these manual tests pass with your local environment variables, the corresponding notebooks should also work.

## Supported Providers

The tests support three major LLM providers:

| Provider | Model | Environment Variable |
|----------|-------|---------------------|
| OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| Google Gemini | gemini-2.0-flash | `GOOGLE_API_KEY` |
| Anthropic Claude | claude-3-5-haiku-latest | `ANTHROPIC_API_KEY` |

## Running Tests

### Single Provider (default: OpenAI)

```bash
cd packages/miiflow-agent

# OpenAI (default)
export OPENAI_API_KEY="sk-..."
poetry run python tests/manual/test_react_manual.py

# Google Gemini
export GOOGLE_API_KEY="..."
poetry run python tests/manual/test_react_manual.py --provider gemini

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
poetry run python tests/manual/test_react_manual.py --provider anthropic
```

### All Providers

Test all providers at once (skips any without API keys set):

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run ReAct tests with all providers
poetry run python tests/manual/test_react_manual.py --provider all

# Run Plan & Execute tests with all providers
poetry run python tests/manual/test_plan_execute_manual.py --provider all
```

### Command Line Options

```
--provider, -p    Provider to test: openai, gemini, anthropic, or all
                  (default: openai)
```

## Test Scripts

| Script | Mirrors | Tests |
|--------|---------|-------|
| `test_react_manual.py` | `examples/notebooks/react_tutorial.ipynb` | Simple lookup, multi-step reasoning, stock comparison, calculations, streaming, error handling |
| `test_plan_execute_manual.py` | `examples/notebooks/plan_execute_tutorial.ipynb` | Basic planning, streaming events, comprehensive report, single stock deep dive |

## Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed

## Troubleshooting

### yfinance Issues

The tests use real stock data from Yahoo Finance via `yfinance`. If you see errors like:
```
WARNING: Tool test returned: Unable to fetch data...
```

This usually means:
- Rate limiting from Yahoo Finance
- Network connectivity issues
- Yahoo Finance API changes

Try running the tests again after a few minutes.

### API Key Issues

If you see:
```
SKIP: OPENAI_API_KEY not set for openai
```

Make sure your API key is exported in the current shell session:
```bash
export OPENAI_API_KEY="sk-..."
```

### Provider-Specific Issues

**Gemini**: Requires a Google AI API key from https://aistudio.google.com/

**Anthropic**: Requires an Anthropic API key from https://console.anthropic.com/
