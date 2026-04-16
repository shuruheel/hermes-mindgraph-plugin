# hermes-mindgraph-plugin

MindGraph semantic graph memory provider for [Hermes Agent](https://github.com/shuruheel/hermes-agent). One install gives your agent persistent, structured memory across sessions — no separate MCP server needed.

Implements the Hermes **MemoryProvider** plugin interface for deep lifecycle integration.

## What You Get

**5 tools** for persistent semantic memory:

| Tool | Purpose |
|------|---------|
| `mindgraph_remember` | Store knowledge — entities, observations, claims, preferences, notes |
| `mindgraph_retrieve` | Query the graph — hybrid search (FTS + semantic), structured queries |
| `mindgraph_commit` | Track agentic state — goals, decisions, plans, tasks, risks, questions |
| `mindgraph_ingest` | Bulk content ingestion (articles, transcripts, code) |
| `mindgraph_synthesize` | Project-scoped cross-document synthesis — mine signals, spawn Article-generation jobs |

**Hybrid retrieval** (FTS + semantic) via `/retrieve/context` — natural language queries work out of the box. Falls back to FTS-only when no embedding provider is configured.

**MemoryProvider lifecycle hooks** for automatic memory management:

| Hook | Purpose |
|------|---------|
| `initialize` | Opens a MindGraph session, pre-fetches context (goals, decisions, policies, weak claims) |
| `system_prompt_block` | Injects behavioral contract + session context into the system prompt |
| `prefetch` | Hybrid retrieval each turn for topic-relevant knowledge |
| `sync_turn` | Accumulates conversation messages for post-session ingestion |
| `on_session_end` | Closes the session with full transcript ingestion for 5-layer knowledge extraction |
| `on_pre_compress` | Preserves key context before context window compression |
| `on_memory_write` | Mirrors built-in Hermes memory writes into MindGraph |
| `shutdown` | Cleans up open sessions on process exit |

## Install

```bash
pip install hermes-mindgraph-plugin
```

## Configure

Set your MindGraph API key (get one at [mindgraph.cloud](https://mindgraph.cloud)):

```bash
# In ~/.hermes/.env
MINDGRAPH_API_KEY=your_api_key_here
```

The plugin is auto-discovered via the `hermes_agent.plugins` entry point. No further configuration needed — the memory provider registers automatically when Hermes starts.

## Tools

### `mindgraph_remember` — store knowledge

| Action | What it does |
|--------|-------------|
| `entity` | Create/find entities (person, org, concept, place, event, etc.) with dedup |
| `observation` | Factual observations — pass `entity_uid` to link to an entity |
| `claim` | Epistemic claims with evidence and calibrated confidence (0.0–1.0) |
| `preference` | User preferences and corrections |
| `note` | General notes, reflections, insights |

### `mindgraph_retrieve` — query the graph

| Mode | What it does |
|------|-------------|
| `context` | **Default.** Hybrid FTS + semantic search. Natural language works. |
| `search` | Keyword-only FTS. Faster for exact name lookups. |
| `recent` | Recently updated nodes, filterable by `node_type`. |
| `goals` | Active goals sorted by salience. |
| `questions` | Open questions and hypotheses. |
| `decisions` | Open decisions needing resolution. |
| `neighborhood` | Nodes connected to a specific node (by UID). |
| `weak_claims` | Claims with low confidence. |
| `contradictions` | Contradictory claims in the graph. |

### `mindgraph_commit` — track agentic state

| Action | Layer |
|--------|-------|
| `goal`, `project`, `milestone` | Intent (with dedup + update via uid) |
| `open_decision`, `add_option`, `add_constraint`, `resolve_decision` | Intent |
| `assess_risk`, `add_affordance` | Action |
| `create_plan`, `create_task`, `add_step`, `update_status`, `get_plan` | Agent |
| `start`, `complete`, `fail` | Agent (execution) |
| `create_policy` | Agent (governance) |
| `question`, `hypothesis`, `anomaly` | Epistemic |

### `mindgraph_ingest` — bulk content

Ingests long-form content (articles, transcripts, code). Under 500 chars: sync. Over 500 chars: async with job tracking.

## Architecture

The plugin uses MindGraph's [5-layer cognitive architecture](https://mindgraph.cloud):

- **Reality layer** — entities, observations, facts about the world
- **Epistemic layer** — claims, evidence, questions, hypotheses, concepts
- **Intent layer** — goals, decisions, projects, milestones
- **Action layer** — risks, affordances, capabilities
- **Agent layer** — plans, tasks, execution tracking, governance policies

Context is proactively injected each turn via `prefetch()` using hybrid retrieval (FTS + semantic via RRF fusion) — the agent gets relevant knowledge without explicit tool calls. Session lifecycle is fully automatic.

## Requirements

- Python >= 3.10
- `mindgraph-sdk >= 0.2.0`
- A MindGraph API key
- Hermes Agent with MemoryProvider support

## License

MIT
