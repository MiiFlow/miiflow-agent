"""Context Injection example (Pydantic AI style).

This example demonstrates dependency injection for tools:
- Defining typed context with dataclasses
- Injecting context into tools via RunContext
- Using deps_type for type-safe context access
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from miiflow_agent import LLMClient, Agent, RunContext, tool


# Define your application context
@dataclass
class UserContext:
    """Context containing user information and preferences."""

    user_id: str
    username: str
    role: str = "user"
    preferences: dict = field(default_factory=dict)


@dataclass
class DatabaseContext:
    """Context with database connection info."""

    db_name: str
    connection_string: str = "localhost:5432"
    tables: List[str] = field(default_factory=list)


@dataclass
class AppContext:
    """Combined application context."""

    user: UserContext
    db: DatabaseContext
    session_id: str = "session_001"


# Tools that use context injection
@tool("get_user_profile", "Get the current user's profile information")
def get_user_profile(ctx: RunContext[UserContext]) -> str:
    """Get profile info from the injected context.

    The ctx parameter is automatically injected by the agent.
    """
    user = ctx.deps
    return f"""
    User Profile:
    - ID: {user.user_id}
    - Username: {user.username}
    - Role: {user.role}
    - Preferences: {user.preferences}
    """


@tool("get_user_preferences", "Get the user's saved preferences")
def get_user_preferences(ctx: RunContext[UserContext], category: Optional[str] = None) -> str:
    """Get user preferences, optionally filtered by category."""
    prefs = ctx.deps.preferences
    if category and category in prefs:
        return f"Preference for {category}: {prefs[category]}"
    return f"All preferences: {prefs}"


@tool("check_permission", "Check if user has a specific permission")
def check_permission(ctx: RunContext[UserContext], permission: str) -> str:
    """Check if the current user has a permission based on their role."""
    role_permissions = {
        "admin": ["read", "write", "delete", "admin"],
        "user": ["read", "write"],
        "guest": ["read"],
    }

    user_perms = role_permissions.get(ctx.deps.role, [])
    has_perm = permission in user_perms
    return f"User {ctx.deps.username} {'has' if has_perm else 'does not have'} '{permission}' permission"


@tool("query_database", "Query the database for information")
def query_database(ctx: RunContext[AppContext], table: str, query: str) -> str:
    """Query the database using the injected connection context."""
    db = ctx.deps.db
    if table not in db.tables:
        return f"Error: Table '{table}' not found. Available tables: {db.tables}"

    return f"""
    Query executed on {db.db_name}:
    - Table: {table}
    - Query: {query}
    - Session: {ctx.deps.session_id}
    - Result: [Simulated results for '{query}']
    """


async def user_context_example():
    """Example with user context injection."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent with typed context
    agent = Agent(client, deps_type=UserContext)

    # Add context-aware tools
    agent.add_tool(get_user_profile)
    agent.add_tool(get_user_preferences)
    agent.add_tool(check_permission)

    # Create context
    user = UserContext(
        user_id="user_123",
        username="alice",
        role="admin",
        preferences={"theme": "dark", "language": "en", "notifications": True},
    )

    # Run with injected context
    print("Query: What is my profile and do I have admin permission?")
    result = await agent.run(
        "What is my profile and do I have admin permission?",
        deps=user,
    )
    print(f"Answer: {result.data}\n")


async def app_context_example():
    """Example with combined application context."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent with complex context type
    agent = Agent(client, deps_type=AppContext)

    agent.add_tool(query_database)

    # Create nested context
    app_ctx = AppContext(
        user=UserContext(user_id="u1", username="bob", role="user"),
        db=DatabaseContext(
            db_name="production_db",
            connection_string="postgres://localhost:5432",
            tables=["users", "orders", "products"],
        ),
        session_id="sess_abc123",
    )

    print("Query: Find all orders from the orders table")
    result = await agent.run(
        "Find all orders from the orders table",
        deps=app_ctx,
    )
    print(f"Answer: {result.data}\n")


async def context_without_injection():
    """Example showing tools without context injection still work."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Tool without context
    @tool("add_numbers", "Add two numbers together")
    def add_numbers(a: int, b: int) -> int:
        """Simple tool that doesn't need context."""
        return a + b

    agent = Agent(client)
    agent.add_tool(add_numbers)

    print("Query: What is 15 + 27?")
    result = await agent.run("What is 15 + 27?")
    print(f"Answer: {result.data}\n")


async def mixed_tools_example():
    """Example mixing context-injected and regular tools."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    @tool("calculate", "Calculate a math expression")
    def calculate(expression: str) -> str:
        """Tool without context."""
        try:
            return str(eval(expression))
        except:
            return "Error evaluating expression"

    agent = Agent(client, deps_type=UserContext)

    # Add both types of tools
    agent.add_tool(calculate)  # No context
    agent.add_tool(get_user_profile)  # With context

    user = UserContext(user_id="u1", username="carol", role="user")

    print("Query: What's my username and what's 100 / 4?")
    result = await agent.run(
        "What's my username and what's 100 / 4?",
        deps=user,
    )
    print(f"Answer: {result.data}\n")


if __name__ == "__main__":
    print("=== User Context Injection ===")
    asyncio.run(user_context_example())

    print("=== Application Context Injection ===")
    asyncio.run(app_context_example())

    print("=== Tools Without Context ===")
    asyncio.run(context_without_injection())

    print("=== Mixed Tools Example ===")
    asyncio.run(mixed_tools_example())
