"""Example subagent templates.

These ``DynamicSubAgentConfig`` entries used to ship as auto-registered
defaults on ``SubAgentRegistry``, but they referenced tool names
(``Glob``/``Grep``/``Read``/``Edit``/``Write``/``Bash``/``WebSearch``/
``WebFetch``) the SDK itself doesn't ship — so the configs looked real
but couldn't actually dispatch. They live here now as templates to
copy into your own setup code.

Adapt each ``tools`` list to the tool names your application registers.
The opt-in coding kit (``miiflow_agent.core.tools.coding``) ships three
file-ops tools — ``file_read``, ``file_edit``, ``file_write`` — that
cover most of the "explorer" and "implementer" templates below. Web
search and shell are not shipped; bring your own via MCP, HTTPTool, or
a custom FunctionTool.

End-to-end wiring example::

    from miiflow_agent import Agent, AgentType, LLMClient
    from miiflow_agent.core.tools.coding import register_coding_tools, build_coding_tools
    from miiflow_agent.core.react import (
        DynamicSubAgentConfig,
        SubAgentRegistry,
        make_registry_dispatcher_tool,
    )
    from examples.subagents import EXPLORER_CONFIG, IMPLEMENTER_CONFIG

    client = LLMClient.create("anthropic", model="claude-sonnet-4-6")
    registry = SubAgentRegistry()
    registry.register(EXPLORER_CONFIG)
    registry.register(IMPLEMENTER_CONFIG)

    coding_tools = build_coding_tools()
    coding_by_name = {t.name: t for t in coding_tools}

    def child_factory(config):
        child_tools = (
            [coding_by_name[n] for n in config.tools if n in coding_by_name]
            if config.tools else coding_tools
        )
        return Agent(
            client=client,
            agent_type=AgentType.REACT,
            system_prompt=config.system_prompt,
            tools=child_tools,
            max_iterations=config.max_steps,
        )

    parent_agent = Agent(client=client, agent_type=AgentType.REACT)
    register_coding_tools(parent_agent.tool_registry)
    parent_agent.add_tool(make_registry_dispatcher_tool(
        registry,
        child_agent_factory=child_factory,
        parent_assistant_id="parent",
    ))
"""

from miiflow_agent.core.react.subagent_registry import DynamicSubAgentConfig


EXPLORER_PROMPT = """You are a codebase exploration specialist.

Your role is to efficiently find and understand code:
- Use file_read to examine file contents
- Use any additional search tools the parent has registered

Be thorough but focused. Report what you find clearly and concisely.
Do not make changes to any files — observation only.
If a search returns unexpected results, note what you expected vs what you found."""


RESEARCHER_PROMPT = """You are a research specialist focused on gathering information.

Your role is to find accurate, up-to-date information using the web
search and fetch tools the parent has registered. Cross-reference
multiple sources for accuracy.

Cite your sources and note the date of information when relevant.
Focus on authoritative sources and official documentation.
If sources conflict, note the disagreement — don't silently pick one."""


IMPLEMENTER_PROMPT = """You are a code implementation specialist.

Your role is to write clean, correct code:
- file_read existing code to understand conventions before writing
- Make focused, minimal changes — don't refactor surrounding code via
  file_edit; use file_write only for new files or full rewrites
- Follow project patterns and style guides

Write code that is:
- Well-structured and readable
- Consistent with existing codebase
- Properly handling edge cases

After making changes, verify they work. Be accurate about what
succeeded and what didn't. Do not perform destructive actions (delete
files, drop tables) without explicit instruction."""


REVIEWER_PROMPT = """You are a code review specialist.

Your role is to analyze code for:
- Bugs and logic errors
- Security vulnerabilities
- Performance issues
- Code style and best practices

Be thorough but constructive. Explain issues clearly and suggest
improvements. Do not make changes — only report findings.
Distinguish between confirmed bugs and potential concerns. Be specific
about severity."""


PLANNER_PROMPT = """You are a planning and architecture specialist.

Your role is to:
- Break down complex tasks into manageable steps
- Identify dependencies between tasks
- Consider trade-offs between approaches
- Plan for error handling and edge cases

Provide clear, actionable plans with specific steps."""


EXPLORER_CONFIG = DynamicSubAgentConfig(
    name="explorer",
    description=(
        "Explore the codebase to find relevant files, understand "
        "patterns, and trace code paths. Use for file discovery, code "
        "navigation, and understanding project structure."
    ),
    system_prompt=EXPLORER_PROMPT,
    tools=["file_read"],
    model="haiku",
    max_steps=5,
    timeout_seconds=60.0,
    priority=10,
)


RESEARCHER_CONFIG = DynamicSubAgentConfig(
    name="researcher",
    description=(
        "Search the web and gather information on topics. Use for "
        "documentation lookups, finding solutions, and researching best "
        "practices. Requires the parent to register web-capable tools."
    ),
    system_prompt=RESEARCHER_PROMPT,
    # Tool names listed here as a hint to the consumer — they map to
    # whatever HTTP/MCP web-search tool the application brings.
    tools=None,
    model="haiku",
    max_steps=5,
    timeout_seconds=60.0,
    priority=5,
)


IMPLEMENTER_CONFIG = DynamicSubAgentConfig(
    name="implementer",
    description=(
        "Write and modify code. Use for implementing features, fixing "
        "bugs, and making code changes."
    ),
    system_prompt=IMPLEMENTER_PROMPT,
    tools=["file_read", "file_edit", "file_write"],
    model="sonnet",
    max_steps=15,
    timeout_seconds=180.0,
    priority=8,
)


REVIEWER_CONFIG = DynamicSubAgentConfig(
    name="reviewer",
    description=(
        "Review code for bugs, security issues, and best practices. "
        "Use for code quality analysis and finding issues."
    ),
    system_prompt=REVIEWER_PROMPT,
    tools=["file_read"],
    model="sonnet",
    max_steps=10,
    timeout_seconds=360.0,
    priority=7,
)


PLANNER_CONFIG = DynamicSubAgentConfig(
    name="planner",
    description=(
        "Plan implementation approaches and break down complex tasks. "
        "Use for architecture decisions and multi-step planning."
    ),
    system_prompt=PLANNER_PROMPT,
    tools=["file_read"],
    model="sonnet",
    max_steps=8,
    timeout_seconds=90.0,
    can_spawn_subagents=True,
    priority=9,
)


ALL_DEFAULTS = [
    EXPLORER_CONFIG,
    RESEARCHER_CONFIG,
    IMPLEMENTER_CONFIG,
    REVIEWER_CONFIG,
    PLANNER_CONFIG,
]
