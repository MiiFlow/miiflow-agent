# Observability

Track LLM calls and agent execution with [Phoenix](https://phoenix.arize.com/).

## Quick Setup

```bash
pip install "miiflow-agent[observability]"
```

```python
from miiflow_agent.core.observability import setup_phoenix_for_miiflow

setup_phoenix_for_miiflow()

# Use normally - all calls are traced
from miiflow_agent import LLMClient, Message
client = LLMClient.create("openai", model="gpt-4o-mini")
response = await client.achat([Message.user("Hello")])

# View traces at http://localhost:6006
```

## Configuration

### Environment Variables

```bash
export PHOENIX_ENABLED=true
export PHOENIX_ENDPOINT=http://localhost:6006
export TRACE_SAMPLE_RATE=1.0  # 0.0-1.0, lower for high volume
```

### Programmatic Setup

```python
from miiflow_agent.core.observability import setup_phoenix_for_miiflow

# With custom endpoint
result = setup_phoenix_for_miiflow(
    endpoint="https://phoenix.yourcompany.com"
)

# Check status
if result["phoenix_enabled"]:
    print(f"Phoenix ready: {result.get('phoenix_session', {}).get('url')}")
```

## What Gets Traced

- LLM requests: model, tokens, latency, content
- Agent execution: step-by-step reasoning
- Tool calls: inputs and outputs
- Streaming: real-time chunks

## Phoenix Dashboard

Open http://localhost:6006 to view:

**Traces Tab:**
- Request/response for each LLM call
- Token counts and latency
- Agent reasoning steps
- Tool executions

**Timeline View:**
- See when each step happened
- Identify slow operations
- Track token usage over time

**Search:**
- Filter by provider, model, or time range
- Search trace content
- Find specific agent runs

## Example

```python
from miiflow_agent import LLMClient, Agent, Message
from miiflow_agent.core.tools import tool
import asyncio

@tool("calculate", "Do math")
def calculate(expr: str) -> str:
    return str(eval(expr))

async def main():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    agent = Agent(client=client)
    agent.add_tool(calculate)

    result = await agent.run("What is 25 * 4?")
    print(result.data)

asyncio.run(main())
# Check Phoenix dashboard for full trace
```

## Troubleshooting

### Phoenix Not Starting

**Check installation:**
```bash
pip install "miiflow-agent[observability]"
# Verify Phoenix installed
python -c "import phoenix; print('Phoenix OK')"
```

**Manual startup:**
```python
from miiflow_agent.core.observability.auto_instrumentation import setup_phoenix_session

session = setup_phoenix_session()
if session:
    print(f"Phoenix at: {session.url}")
else:
    print("Failed - check dependencies")
```

### No Traces Appearing

**1. Check Phoenix is running:**
- Visit http://localhost:6006
- Should see Phoenix UI

**2. Verify instrumentation:**
```python
from miiflow_agent.core.observability.auto_instrumentation import check_instrumentation_status

status = check_instrumentation_status()
for provider, info in status.items():
    print(f"{provider}: {info}")
```

**3. Check dependencies:**
```bash
# Install OpenInference instrumentations
pip install openinference-instrumentation-openai
pip install openinference-instrumentation-anthropic
```

### Common Errors

**"OpenInference instrumentation not available"**
```bash
pip install openinference-instrumentation-openai openinference-instrumentation-anthropic
```

**"Phoenix session setup failed"**
```bash
pip install arize-phoenix
```

**Traces delayed or missing**
- Check `TRACE_SAMPLE_RATE` (default 1.0 = 100%)
- Verify Phoenix endpoint is accessible
- Check firewall/network settings



## Agent Evaluation

### Quick Start

Automatically evaluate agent responses:

```python
from miiflow_agent.core.observability.evaluation import create_evaluated_agent

# Wrap agent with evaluation
evaluated_agent = create_evaluated_agent(agent)

# Run normally - evaluation happens automatically
result = await evaluated_agent.run("What is the capital of France?")

# Access evaluation results
evaluation = result.metadata["evaluation"]
print(f"Metrics: {evaluation['metrics']}")
# Output: {'relevance': 0.95, 'helpfulness': True, 'response_time': True, 'safety': True}
```

### Default Metrics

**Relevance (0.0-1.0):** Keyword overlap between query and response
**Helpfulness (bool):** Response is substantial and useful
**Response Time (bool):** Generated within acceptable time
**Safety (bool):** No unsafe or inappropriate content

### Custom Metrics

Add your own evaluation logic:

```python
from miiflow_agent.core.observability.evaluation import AgentEvaluator, EvaluationMetric

evaluator = AgentEvaluator()

# Add custom metric
def check_conciseness(response: str, context: dict) -> bool:
    """Response should be under 100 words."""
    return len(response.split()) <= 100

evaluator.add_metric(EvaluationMetric(
    name="conciseness",
    description="Response is under 100 words",
    evaluator=check_conciseness
))

# Add accuracy metric with ground truth
ground_truth = {
    "What is the capital of France?": "Paris",
    "What's 2 + 2?": "4"
}

def check_accuracy(response: str, context: dict) -> bool:
    query = context.get("user_query", "")
    expected = ground_truth.get(query)
    return expected and expected.lower() in response.lower()

evaluator.add_metric(EvaluationMetric(
    name="accuracy",
    description="Response contains correct answer",
    evaluator=check_accuracy
))

# Use custom evaluator
from miiflow_agent.core.observability.evaluation import EvaluatedAgent
evaluated_agent = EvaluatedAgent(agent, evaluator)

result = await evaluated_agent.run("What is the capital of France?")
print(result.metadata["evaluation"]["metrics"])
# Output: {'relevance': 0.95, 'helpfulness': True, 'response_time': True,
#          'safety': True, 'conciseness': True, 'accuracy': True}
```

### Evaluation Summary

Track performance across multiple queries:

```python
# Run multiple evaluations
for query in ["Query 1", "Query 2", "Query 3"]:
    result = await evaluated_agent.run(query)

# Get aggregate statistics
summary = evaluated_agent.get_evaluation_summary()
print(f"Total evaluations: {summary['total_evaluations']}")

# Per-metric stats
for metric_name, stats in summary["metric_summaries"].items():
    if stats["type"] == "boolean":
        print(f"{metric_name}: {stats['success_rate']:.1%} success rate")
    elif stats["type"] == "numeric":
        print(f"{metric_name}: {stats['mean']:.2f} average")
```

See [examples/agent_evaluation_example.py](../examples/agent_evaluation_example.py) for more patterns.

