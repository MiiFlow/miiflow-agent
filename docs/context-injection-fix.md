# Context Injection Fix

## Problem

After fixing the XML-based ReAct tool execution bug, context injection broke for tools that need it (like CRM tools):

```
ERROR: Tool 'Multiply' failed: CRMToolSet.multiply() missing 1 required positional argument: 'ctx'
```

## Root Cause

The initial fix passed `context=None` to ALL tools:

```python
# orchestrator.py:276 - BROKEN
return await self.tool_executor.execute_tool(step.action, step.action_input, context=None)
```

This worked for simple tools (like `add_tool(a, b)`) but broke CRM tools that expect context as the first parameter (like `multiply(ctx, a, b)`).

## Solution

Detect which tools need context injection and only pass context when needed:

### 1. Added `tool_needs_context()` method to `AgentToolExecutor`

**File**: `packages/miiflow-llm/miiflow_llm/core/react/tool_executor.py`

```python
def tool_needs_context(self, tool_name: str) -> bool:
    """Check if a tool requires context injection."""
    tool = self._tool_registry.tools.get(tool_name)
    if not tool:
        return False
    # Check if tool has context_injection attribute and if pattern is not 'none'
    if hasattr(tool, 'context_injection'):
        pattern = tool.context_injection.get('pattern', 'none')
        return pattern in ('first_param', 'keyword')
    return False
```

### 2. Updated orchestrator to check context requirements

**File**: `packages/miiflow-llm/miiflow_llm/core/react/orchestrator.py`

```python
async def _execute_tool(self, step: ReActStep, context: RunContext):
    """Execute tool with proper context injection."""
    # ... validation code ...

    # Determine if tool needs context injection
    needs_context = self.tool_executor.tool_needs_context(step.action)

    # Execute tool with or without context based on tool's requirements
    return await self.tool_executor.execute_tool(
        step.action,
        step.action_input,
        context=context if needs_context else None
    )
```

## How It Works

### Context Patterns Detected

The `FunctionTool` class already analyzes function signatures to detect context injection patterns:

1. **`first_param` pattern**: Context as first parameter (Pydantic AI style)
   ```python
   def multiply(ctx, a: float, b: float):  # ctx is first param
       ...
   ```

2. **`keyword` pattern**: Context as keyword parameter
   ```python
   def multiply(a: float, b: float, context=None):  # context keyword param
       ...
   ```

3. **`none` pattern**: No context parameter
   ```python
   def add(a: float, b: float):  # no context
       ...
   ```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ ReAct Orchestrator                                          │
│                                                             │
│  _execute_tool(step, context)                              │
│    │                                                        │
│    ├─► tool_executor.tool_needs_context("add")             │
│    │   └─► Returns False (no ctx param)                    │
│    │                                                        │
│    └─► tool_executor.execute_tool("add", {...}, None)      │
│        └─► registry.execute_safe("add", a=15, b=27)        │
│            └─► add(15, 27) ✅                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ ReAct Orchestrator                                          │
│                                                             │
│  _execute_tool(step, context)                              │
│    │                                                        │
│    ├─► tool_executor.tool_needs_context("Multiply")        │
│    │   └─► Returns True (ctx as first param)               │
│    │                                                        │
│    └─► tool_executor.execute_tool("Multiply", {...}, ctx)  │
│        └─► registry.execute_safe_with_context(...)         │
│            └─► multiply(ctx, a=15, b=27) ✅                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Test Coverage

### 1. Context Detection Test (`test_context_injection.py`)

Tests that the system correctly identifies which tools need context:

```python
✓ Tool 'add_simple' needs context: False
✓ Tool 'add_with_context' needs context: True
```

### 2. Tool Execution Test

Tests that tools execute correctly with appropriate context handling:

```python
✅ Simple tool works without context
✅ Context tool works with context injection
✅ Context was injected: org_id = org_12345
```

### 3. End-to-End Tests

- **`test_react_streaming.py`**: Simple tools (no context) ✅
- **CRM Tools**: Context-injected tools (e.g., `Multiply`, `Find Accounts`) ✅
- **XML Integration**: All XML parsing and streaming ✅

## Results

✅ **Test tools work** (no context parameter)
```bash
poetry run python examples/test_react_streaming.py
# ✅ 389 thinking chunks streamed
# ✅ All tool executions successful
```

✅ **CRM tools work** (context injection)
```python
# CRM tools like Multiply now work correctly with ctx parameter
@tool(name="Multiply", description="Multiply two numbers")
def multiply(ctx, a: float, b: float) -> Dict[str, Any]:
    # ctx is injected automatically
    result = a * b
    return {"success": True, "result": result}
```

✅ **Context injection is secure**
- Organization context passed via `ctx.deps["organization_id"]`
- LLM never sees organization_id parameter
- Prevents prompt injection attacks

## Migration Notes

No migration needed! The fix is backward compatible:

- Tools without context continue to work
- Tools with context continue to receive it
- No changes required to existing tool definitions
- Detection is automatic based on function signature

## Related Files

- `packages/miiflow-llm/miiflow_llm/core/react/orchestrator.py` - Orchestrator context detection
- `packages/miiflow-llm/miiflow_llm/core/react/tool_executor.py` - Tool executor with `tool_needs_context()`
- `packages/miiflow-llm/miiflow_llm/core/tools/function/context_patterns.py` - Context pattern detection
- `packages/miiflow-llm/test_context_injection.py` - Test suite for context injection
- `server/assistant/tools/crm_tools.py` - CRM tools using context injection
