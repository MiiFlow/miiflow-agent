# Parallel Execution Architecture for miiflow-llm

## Summary

This document describes the new parallel execution capabilities added to miiflow-llm, enabling up to **90% reduction in execution time** for parallelizable tasks (based on Anthropic's multi-agent research findings).

## Changes Implemented

### New Agent Types

| Type | Description | Use Case |
|------|-------------|----------|
| `PARALLEL_PLAN` | Executes independent subtasks in parallel waves | Tasks with independent subtasks that can run concurrently |
| `MULTI_AGENT` | Spawns specialized subagents working in parallel | Complex queries requiring multiple perspectives |

### New Files

```
miiflow_llm/core/react/
├── shared_state.py              # Thread-safe shared state for multi-agent coordination
├── parallel_plan_orchestrator.py # Wave-based parallel subtask execution
└── multi_agent_orchestrator.py   # Multi-subagent parallel orchestration
```

### Modified Files

| File | Changes |
|------|---------|
| `core/react/enums.py` | Added `ParallelPlanEventType`, `MultiAgentEventType` |
| `core/react/models.py` | Added `ExecutionWave`, `SubAgentConfig`, `SubAgentResult`, `SubAgentPlan`, `MultiAgentResult` |
| `core/react/react_events.py` | Added `ParallelPlanEvent`, `MultiAgentEvent` |
| `core/agent.py` | Added new AgentType values and streaming methods |

---

## Architecture

### Parallel Plan (Wave-Based Execution)

```
Planning Phase
    ↓
Dependency Graph Analysis
    ↓
┌──────────────────────────────────────────┐
│  Wave 0: No dependencies                 │
│  ┌─────┐  ┌─────┐  ┌─────┐              │
│  │ ST1 │  │ ST2 │  │ ST3 │  (parallel)  │
│  └──┬──┘  └──┬──┘  └──┬──┘              │
│     └────────┴────────┘                  │
│              ↓                           │
│  Wave 1: Depends on Wave 0              │
│         ┌─────┐                          │
│         │ ST4 │                          │
│         └─────┘                          │
└──────────────────────────────────────────┘
    ↓
Synthesis Phase
```

**Key Features:**
- Topological sort groups subtasks into parallel waves
- Respects dependency ordering between subtasks
- Falls back to sequential if dependencies require it
- Configurable `max_parallel_subtasks` per wave (default: 5)

### Multi-Agent (Parallel Subagents)

```
User Query
    ↓
Lead Agent (orchestrator)
    ├── Analyze & Plan subagent allocation
    ├── Spawn SubAgents (parallel)
    │   ┌─────────────────────────────────────┐
    │   │  ┌─────────┐  ┌─────────┐  ┌─────────┐
    │   │  │Researcher│  │Analyzer │  │ Coder   │
    │   │  └────┬────┘  └────┬────┘  └────┬────┘
    │   │       └────────────┴────────────┘
    │   │                    ↓
    │   │            Shared State (results)
    │   └─────────────────────────────────────┘
    └── Synthesize Results
    ↓
Final Answer
```

**Key Features:**
- Lead agent plans subagent allocation based on query
- 1-5 subagents based on query complexity
- Thread-safe shared state with unique output keys
- Synchronous coordination (waits for all to complete)
- All subagents share the same tool registry

---

## Usage

```python
from miiflow_llm import Agent, AgentType, LLMClient

# Parallel Plan - for tasks with parallelizable subtasks
agent = Agent(
    client=LLMClient.create("anthropic", "claude-sonnet-4-20250514"),
    agent_type=AgentType.PARALLEL_PLAN,
    tools=[search_tool, analyze_tool, summarize_tool],
)
result = await agent.run("Compare Python web frameworks: Django, FastAPI, and Flask")

# Multi-Agent - for complex multi-faceted queries
agent = Agent(
    client=LLMClient.create("openai", "gpt-4o"),
    agent_type=AgentType.MULTI_AGENT,
    tools=[web_search, code_analysis, security_scan],
)
result = await agent.run("Analyze this codebase for security vulnerabilities and performance issues")
```

### Streaming Events

Both patterns emit real-time events for UI feedback:

```python
async for event in agent.stream(query, context, agent_type=AgentType.PARALLEL_PLAN):
    if event.event_type == ParallelPlanEventType.WAVE_START:
        print(f"Starting wave {event.data['wave_number']} with {event.data['parallel_count']} tasks")
    elif event.event_type == ParallelPlanEventType.PARALLEL_SUBTASK_COMPLETE:
        print(f"Subtask {event.data['subtask_id']} completed")
```

---

## Event Types

### ParallelPlanEventType

| Event | Description |
|-------|-------------|
| `WAVE_START` | Starting a new parallel execution wave |
| `WAVE_COMPLETE` | Wave finished executing |
| `PARALLEL_SUBTASK_START` | Individual subtask in wave started |
| `PARALLEL_SUBTASK_COMPLETE` | Individual subtask in wave completed |

### MultiAgentEventType

| Event | Description |
|-------|-------------|
| `PLANNING_START` | Lead agent analyzing query |
| `PLANNING_THINKING_CHUNK` | Streaming planning reasoning |
| `PLANNING_COMPLETE` | Subagent allocation planned |
| `EXECUTION_START` | Starting parallel subagent execution |
| `SUBAGENT_START` | Individual subagent started |
| `SUBAGENT_COMPLETE` | Subagent finished successfully |
| `SUBAGENT_FAILED` | Subagent failed |
| `SYNTHESIS_START` | Starting result synthesis |
| `FINAL_ANSWER` | Final synthesized answer |

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hierarchy | Flat only | Single level - no nested subagents for simplicity |
| Coordination | Synchronous | Wait for all to complete before synthesis |
| Tools | Shared | All subagents access the same tool registry |
| State | Thread-safe | Unique output keys prevent race conditions |

---

## Follow-up Plans

### Phase 1: Testing & Validation (Next)
- [ ] Add unit tests for `ParallelPlanOrchestrator`
- [ ] Add unit tests for `MultiAgentOrchestrator`
- [ ] Add unit tests for `SharedAgentState`
- [ ] Integration tests with mock LLM responses
- [ ] Performance benchmarks comparing sequential vs parallel

### Phase 2: Optimizations
- [ ] Early termination when sufficient results gathered
- [ ] Effort scaling based on query complexity (1-5 subagents)
- [ ] Token budget management across subagents
- [ ] Retry logic for failed subagents

### Phase 3: Advanced Features
- [ ] Dynamic subagent spawning (agent can request more)
- [ ] Inter-agent communication (agents can message each other)
- [ ] Specialized tool sets per subagent role
- [ ] Hierarchical nesting (subagents spawning sub-subagents)

### Phase 4: UI Integration
- [ ] Frontend components for wave visualization
- [ ] Subagent progress indicators
- [ ] Parallel execution timeline view
- [ ] Cost/token tracking per subagent

---

## References

- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) - 90% latency reduction with parallel subagents
- [Google ADK Multi-Agent Patterns](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/) - 8 orchestration patterns
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/) - Supervisor and hierarchical patterns

---

## Test Results

All **267 existing tests passed** after implementation. The changes are backward-compatible.

```
================ 267 passed, 9 skipped, 52 deselected in 32.83s ================
```
