# hermes-mindgraph-plugin

MindGraph semantic graph memory provider for [Hermes Agent](https://github.com/shuruheel/hermes-agent). One install gives your agent persistent, structured memory across sessions — no separate MCP server needed.

Implements the Hermes **MemoryProvider** plugin interface for deep lifecycle integration.

## What You Get

**11 cognitive tools** across MindGraph's 5-layer architecture:

| Tool | Layer | Purpose |
|------|-------|---------|
| `mindgraph_session` | Memory | Open/close sessions |
| `mindgraph_journal` | Memory | Low-friction capture (observations, notes, preferences) |
| `mindgraph_capture` | Reality | Entities, observations, concepts with deduplication |
| `mindgraph_argue` | Epistemic | Structured claims with evidence + confidence |
| `mindgraph_inquire` | Epistemic | Questions, hypotheses, anomalies |
| `mindgraph_commit` | Intent | Goals, projects, milestones |
| `mindgraph_decide` | Intent | Decisions with options and resolution |
| `mindgraph_action` | Action | Risks and affordances |
| `mindgraph_plan` | Agent | Plans, tasks, execution, governance policies |
| `mindgraph_retrieve` | Query | Hybrid semantic search, goals, questions, context |
| `mindgraph_ingest` | Memory | Long-form content ingestion |

**MemoryProvider lifecycle hooks** for automatic memory management:

| Hook | Purpose |
|------|---------|
| `initialize` | Opens a MindGraph session, pre-fetches context (goals, decisions, policies, weak claims) |
| `system_prompt_block` | Injects behavioral contract + session context into the system prompt |
| `prefetch` | Score-gated semantic retrieval each turn for topic-relevant knowledge |
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

## Manual Install

Alternatively, copy the `hermes_mindgraph_plugin` package to `~/.hermes/plugins/memory/mindgraph/` with a `plugin.yaml`:

```yaml
name: mindgraph
version: 0.4.0
description: Semantic graph memory provider
```

## Architecture

The plugin uses MindGraph's [5-layer cognitive architecture](https://mindgraph.cloud):

- **Reality layer** — entities, observations, facts about the world
- **Epistemic layer** — claims, evidence, questions, hypotheses, concepts
- **Intent layer** — goals, decisions, projects, milestones
- **Action layer** — risks, affordances, capabilities
- **Agent layer** — plans, tasks, execution tracking, governance policies

Context is proactively injected each turn via `prefetch()` using score-gated semantic retrieval (threshold: 0.55) — the agent gets relevant knowledge without explicit tool calls. Tools are available for deeper queries and active knowledge capture.

## Requirements

- Python >= 3.10
- `mindgraph-sdk >= 0.2.0`
- A MindGraph API key
- Hermes Agent with MemoryProvider support

## License

MIT
