# Contributing

Thanks for your interest in miiflow-agent. We welcome contributions.

## Getting Started

1. Fork the repo
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/miiflow-agent.git
   cd miiflow-agent
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[all]"
   pip install -e ".[dev]"
   ```

4. Create a branch:
   ```bash
   git checkout -b feature/your-feature
   ```

## Development

### Run Tests

```bash
pytest tests/
```

### Code Style

We use:
- `black` for formatting
- `isort` for imports
- `flake8` for linting
- `mypy` for type checking

Run all checks:
```bash
black miiflow_agent/
isort miiflow_agent/
flake8 miiflow_agent/
mypy miiflow_agent/
```

Or setup pre-commit:
```bash
pre-commit install
```

### Project Structure

```
miiflow_agent/
├── core/           # Core abstractions
│   ├── client.py   # LLM client
│   ├── agent.py    # Agent implementation
│   ├── message.py  # Message types
│   ├── tools/      # Tool system
│   └── react/      # ReAct orchestration
├── providers/      # Provider adapters
│   ├── openai.py
│   ├── anthropic.py
│   └── ...
└── utils/          # Utilities
```

## What to Contribute

### Good First Issues
- Fix typos in docs
- Add examples
- Improve error messages
- Write tests

### Bigger Contributions
- Add new providers
- Improve ReAct agent
- Add observability features
- Performance optimizations

## Pull Requests

1. Write clear commits:
   ```
   Add support for X provider

   - Implement XProvider adapter
   - Add tests for streaming
   - Update docs
   ```

2. Add tests for new code:
   ```python
   def test_new_feature():
       # Your test
       assert result == expected
   ```

3. Update docs if needed

4. Keep PRs focused - one feature per PR

5. Make sure tests pass:
   ```bash
   pytest tests/
   ```

## Adding a Provider

1. Create `miiflow_agent/providers/newprovider.py`:
   ```python
   from miiflow_agent.core.client import BaseLLMClient
   from miiflow_agent.core.message import Message
   from miiflow_agent.core.streaming import StreamChunk

   class NewProviderClient(BaseLLMClient):
       async def _achat(self, messages):
           # Implementation
           pass

       async def _astream_chat(self, messages):
           # Implementation
           pass
   ```

2. Register in `miiflow_agent/__init__.py`:
   ```python
   "newprovider": "miiflow_agent.providers.newprovider.NewProviderClient"
   ```

3. Add tests in `tests/providers/test_newprovider.py`

4. Update `docs/providers.md`

## Testing

### Unit Tests
```bash
pytest tests/core/
```

### Provider Tests
```bash
pytest tests/providers/
```

### Integration Tests
```bash
pytest tests/test_llm_client.py
```

### With Coverage
```bash
pytest --cov=miiflow_agent tests/
```

## Documentation

Docs are in `docs/`. Use clear examples:

```markdown
# Good
message = Message.user("Hello")

# Bad
Construct a message object using the Message class factory method
```

## Code Review

We look for:
- Clear code
- Good tests
- Updated docs
- Type hints
- Error handling

## Questions?

- Open an issue
- Ask in your PR
- Check existing issues

## License

By contributing, you agree your code is licensed under MIT.
