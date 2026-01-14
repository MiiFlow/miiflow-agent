# Tutorial: Building Tools for Agents

Let's build a weather agent step by step.

## What We'll Build

An agent that can:
- Check weather for any city
- Give clothing recommendations
- Handle errors gracefully

Takes 15 minutes.

## Step 1: Basic Tool

Start simple. A tool that returns fake weather:

```python
from miiflow_agent.core.tools import tool

@tool("get_weather", "Get current weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"
```

That's it. The `@tool` decorator makes your function available to agents.

**What happens:**
- Function name becomes tool name
- Docstring becomes tool description
- Type hints define parameters
- Return type shows what comes back

## Step 2: Add to Agent

Now hook it up:

```python
from miiflow_agent import LLMClient, Agent
from miiflow_agent.core import AgentType
import asyncio

client = LLMClient.create("openai", model="gpt-4o-mini")
agent = Agent(client=client, agent_type=AgentType.REACT)

agent.add_tool(get_weather)

result = asyncio.run(agent.run("What's the weather in Paris?", deps={}))
print(result.data)
```

Run it. The agent will:
1. See your question
2. Decide to call `get_weather`
3. Pass `city="Paris"`
4. Get "sunny, 22°C"
5. Answer you

## Step 3: Make It Real

Replace fake data with real API:

```python
import httpx

@tool("get_weather", "Get current weather for a city")
def get_weather(city: str) -> str:
    response = httpx.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": "YOUR_KEY", "units": "metric"}
    )

    if response.status_code == 404:
        return f"City '{city}' not found"

    response.raise_for_status()
    data = response.json()

    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]

    return f"{city}: {temp}°C, {description}"
```

Same interface. Agent doesn't know you changed anything.

## Step 4: Add Another Tool

Clothing recommendations based on temperature:

```python
@tool("recommend_clothing", "Suggest what to wear based on temperature")
def recommend_clothing(temperature: float) -> str:
    if temperature < 10:
        return "Wear a heavy coat, it's cold"
    elif temperature < 20:
        return "Light jacket recommended"
    else:
        return "T-shirt weather, no jacket needed"
```

Add both tools:

```python
agent.add_tool(get_weather)
agent.add_tool(recommend_clothing)

result = asyncio.run(agent.run(
    "What should I wear in Paris today?",
    deps={}
))
```

The agent will:
1. Call `get_weather("Paris")` → "22°C, sunny"
2. Extract temperature: 22
3. Call `recommend_clothing(22)` → "T-shirt weather"
4. Combine info: "In Paris it's 22°C and sunny. T-shirt weather, no jacket needed."

## Step 5: Handle Errors

Add validation:

```python
@tool("get_weather", "Get current weather for a city")
def get_weather(city: str) -> str:
    if not city or len(city) < 2:
        raise ValueError("City name must be at least 2 characters")

    try:
        response = httpx.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": "YOUR_KEY", "units": "metric"},
            timeout=5.0
        )

        if response.status_code == 404:
            raise ValueError(f"City '{city}' not found")

        response.raise_for_status()
        data = response.json()

        return f"{city}: {data['main']['temp']}°C, {data['weather'][0]['description']}"

    except httpx.TimeoutError:
        raise ValueError("Weather service timeout, try again")
    except httpx.HTTPError as e:
        raise ValueError(f"Weather service error: {e}")
```

When errors happen, the agent sees them and can respond appropriately.

## Step 6: Use Dependencies

Pass database connections or config:

```python
@tool("save_weather", "Save weather to database")
def save_weather(city: str, weather: str, deps: dict) -> str:
    db = deps["database"]
    db.execute(
        "INSERT INTO weather (city, data) VALUES (?, ?)",
        (city, weather)
    )
    return f"Saved weather for {city}"

# Provide deps when running
result = asyncio.run(agent.run(
    "Check and save weather for London",
    deps={"database": my_db_connection}
))
```

## Complete Example

```python
import httpx
import asyncio
from miiflow_agent import LLMClient, Agent
from miiflow_agent.core import AgentType
from miiflow_agent.core.tools import tool

@tool("get_weather", "Get current weather for a city")
def get_weather(city: str) -> str:
    response = httpx.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": "YOUR_KEY", "units": "metric"}
    )
    response.raise_for_status()
    data = response.json()
    return f"{city}: {data['main']['temp']}°C, {data['weather'][0]['description']}"

@tool("recommend_clothing", "Suggest clothing for temperature")
def recommend_clothing(temperature: float) -> str:
    if temperature < 10:
        return "Heavy coat needed"
    elif temperature < 20:
        return "Light jacket recommended"
    else:
        return "T-shirt weather"

async def main():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    agent = Agent(
        client=client,
        agent_type=AgentType.REACT,
        system_prompt="You help with weather and clothing advice."
    )

    agent.add_tool(get_weather)
    agent.add_tool(recommend_clothing)

    result = await agent.run(
        "What should I wear in Tokyo today?",
        deps={}
    )

    print(result.data)

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

1. **Tools are just functions** - Decorate and done
2. **Type hints matter** - They define the schema
3. **Raise exceptions** - Don't return error strings
4. **One tool, one job** - Keep them focused
5. **Test independently** - Make sure they work before giving to agent

## Next Steps

- [API Reference](api.md) - All tool options
- [Agent Tutorial](tutorial-agents.md) - Build complex agents
- [Examples](../examples/) - More tool patterns
