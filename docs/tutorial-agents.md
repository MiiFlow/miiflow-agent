# Tutorial: Building a ReAct Agent

Let's build an agent that does research and calculations.

## What We'll Build

An agent that can:
- Search Wikipedia for information
- Perform calculations
- Combine multiple tools to answer questions

Takes 20 minutes.

## Step 1: Create the Client

Start with a basic LLM client:

```python
from miiflow_agent import LLMClient
import asyncio

client = LLMClient.create("openai", model="gpt-4o-mini")
```

Nothing special yet. Just a normal client.

## Step 2: Create Empty Agent

Add an agent with no tools:

```python
from miiflow_agent import Agent
from miiflow_agent.core import AgentType

agent = Agent(
    client=client,
    agent_type=AgentType.REACT,
    system_prompt="You are a helpful research assistant."
)

# Try it without tools
result = asyncio.run(agent.run("What is 25 * 4?", deps={}))
print(result.data)
```

Run this. The agent will answer but can't use any tools yet.

## Step 3: Add Calculator Tool

Give it a calculator:

```python
from miiflow_agent.core.tools import tool
import math

@tool("calculate", "Evaluate a math expression")
def calculate(expression: str) -> str:
    """
    Calculate a math expression.

    Args:
        expression: Math like "25 * 4" or "sqrt(144)"
    """
    # Safe eval with only math functions
    allowed = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "abs": abs,
        "pow": pow,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

agent.add_tool(calculate)
```

Now test:

```python
result = asyncio.run(agent.run("What is 25 * 4 + 100?", deps={}))
print(result.data)
```

Watch what happens:
1. Agent sees the question
2. Thinks: "I need to calculate this"
3. Calls: `calculate("25 * 4 + 100")`
4. Gets: "Result: 200"
5. Answers: "The result is 200"

## Step 4: Add Search Tool

Add Wikipedia search:

```python
import httpx

@tool("search_wikipedia", "Search Wikipedia for information")
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia.

    Args:
        query: What to search for
    """
    response = httpx.get(
        "https://en.wikipedia.org/api/rest_v1/page/summary/" + query
    )

    if response.status_code == 404:
        return f"No Wikipedia page found for '{query}'"

    response.raise_for_status()
    data = response.json()

    return data.get("extract", "No summary available")

agent.add_tool(search_wikipedia)
```

Now you have two tools. Test both:

```python
result = asyncio.run(agent.run(
    "Who invented Python and when were they born?",
    deps={}
))
print(result.data)
```

The agent will:
1. Search Wikipedia for "Python programming language"
2. Find Guido van Rossum
3. Search Wikipedia for "Guido van Rossum"
4. Extract birth year
5. Answer the question

## Step 5: Combine Tools

Ask something requiring both tools:

```python
result = asyncio.run(agent.run(
    "How many years ago was the creator of Python born? (Current year is 2024)",
    deps={}
))
```

The agent will:
1. Search for Python creator → "Guido van Rossum"
2. Search for his info → "Born 1956"
3. Calculate → `calculate("2024 - 1956")` → "68"
4. Answer → "68 years ago"

**This is the power of agents** - they combine tools to solve complex tasks.

## Step 6: See What It's Doing

Check the execution steps:

```python
result = asyncio.run(agent.run(
    "What is the square root of 144 plus 10?",
    deps={}
))

# Look at what it did
for i, step in enumerate(result.metadata["react_steps"], 1):
    print(f"\nStep {i}:")
    print(f"  Thought: {step.get('thought', 'N/A')}")

    if step.get("action"):
        print(f"  Action: {step['action']}")
        print(f"  Input: {step.get('action_input', {})}")

    if step.get("observation"):
        print(f"  Result: {step['observation']}")

print(f"\nFinal answer: {result.data}")
```

Output:
```
Step 1:
  Thought: I need to calculate sqrt(144) first
  Action: calculate
  Input: {'expression': 'sqrt(144)'}
  Result: Result: 12.0

Step 2:
  Thought: Now add 10 to that result
  Action: calculate
  Input: {'expression': '12 + 10'}
  Result: Result: 22.0

Final answer: The result is 22.
```

## Step 7: Handle Failures

Tools can fail. The agent handles it:

```python
result = asyncio.run(agent.run(
    "Search for 'asdfasdfasdf' on Wikipedia",
    deps={}
))
print(result.data)
```

Output: "I couldn't find a Wikipedia page for that. Could you try a different search term?"

The agent sees the error and responds appropriately.

## Step 8: Better System Prompt

Guide the agent's behavior:

```python
agent = Agent(
    client=client,
    agent_type=AgentType.REACT,
    system_prompt="""You are a helpful research assistant.

When asked questions:
1. Use search_wikipedia for factual information
2. Use calculate for any math
3. Always cite your sources
4. If you can't find something, say so clearly

Be concise and accurate."""
)
```

This shapes how the agent thinks and responds.

## Step 9: Add More Tools

Build a more capable agent:

```python
@tool("get_current_time", "Get current date and time")
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool("convert_currency", "Convert between currencies")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    # Simplified - use real API in production
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 110.0}

    if from_currency not in rates or to_currency not in rates:
        return "Currency not supported"

    usd_amount = amount / rates[from_currency]
    result = usd_amount * rates[to_currency]

    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

agent.add_tool(get_current_time)
agent.add_tool(convert_currency)
```

Now you have a multi-tool agent:

```python
result = asyncio.run(agent.run(
    "What time is it, and how much is 100 USD in EUR?",
    deps={}
))
```

## Complete Example

```python
import asyncio
import httpx
import math
from miiflow_agent import LLMClient, Agent
from miiflow_agent.core import AgentType
from miiflow_agent.core.tools import tool

@tool("calculate", "Evaluate math expressions")
def calculate(expression: str) -> str:
    allowed = {
        "sqrt": math.sqrt,
        "pi": math.pi,
        "abs": abs,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool("search_wikipedia", "Search Wikipedia")
def search_wikipedia(query: str) -> str:
    response = httpx.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    )
    if response.status_code == 404:
        return f"Not found: {query}"
    data = response.json()
    return data.get("extract", "No summary")

async def main():
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client=client,
        agent_type=AgentType.REACT,
        system_prompt="You are a research assistant. Use tools to answer accurately."
    )

    agent.add_tool(calculate)
    agent.add_tool(search_wikipedia)

    # Ask a complex question
    result = await agent.run(
        "Who created Python, and how old would they be in 2024?",
        deps={}
    )

    print(result.data)

    # Show token usage
    print(f"\nTokens used: {result.usage.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding the XML Format

Behind the scenes, the agent uses XML (not JSON) for tool calling:

```xml
<thought>I need to search for Python's creator</thought>
<action>
  <tool>search_wikipedia</tool>
  <input>
    <query>Python programming language</query>
  </input>
</action>
```

You don't write this. The agent generates it automatically.

**Why XML?**
- More natural for LLMs
- Better in long conversations
- We can stream the final answer

## Tips for Success

1. **Clear tool descriptions** - The agent reads these to decide what to call
2. **Validate inputs** - Tools should check their parameters
3. **Return useful info** - Give the agent data it can use
4. **Test tools alone first** - Make sure they work before giving to agent
5. **Start simple** - Add one tool at a time

## Common Issues

**Agent doesn't call tools:**
- Check tool descriptions are clear
- Make sure system prompt encourages tool use
- Verify tools are actually added

**Agent calls wrong tool:**
- Tool descriptions might be ambiguous
- Add examples in tool docstrings
- Use more specific names

**Agent loops forever:**
- Usually means tool returned unhelpful data
- Add better error messages
- Check tool actually solves the problem

## Next Steps

- [Tools Tutorial](tutorial-tools.md) - Build better tools
- [API Reference](api.md) - All agent options
- [Examples](../examples/) - More agent patterns
