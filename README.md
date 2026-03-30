# hermes-mindgraph-plugin

MindGraph semantic graph memory plugin for [Hermes Agent](https://github.com/shuruheel/hermes-agent). One install gives your agent persistent, structured memory across sessions — no separate MCP server needed.

## What You Get

**11 cognitive tools** registered into the `mindgraph` toolset:

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

**3 lifecycle hooks** for automatic memory management:

- **on_session_start** — Opens a MindGraph session, pre-fetches context (goals, decisions, policies, weak claims)
- **pre_llm_call** — Injects session context on first turn; score-gated semantic retrieval every turn
- **on_session_end** — Closes the session with transcript ingestion for post-session knowledge extraction

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

The plugin is auto-discovered via the `hermes_agent.plugins` entry point. No further configuration needed — tools and hooks register automatically when Hermes starts.

## Manual Install

Alternatively, copy the `hermes_mindgraph_plugin` package to `~/.hermes/plugins/mindgraph/` with a `plugin.yaml`:

```yaml
name: mindgraph
version: 0.2.0
description: MindGraph semantic graph memory
```

## Architecture

The plugin uses MindGraph's [5-layer cognitive architecture](https://mindgraph.cloud):

- **Reality layer** — entities, observations, facts about the world
- **Epistemic layer** — claims, evidence, questions, hypotheses, concepts
- **Intent layer** — goals, decisions, projects, milestones
- **Action layer** — risks, affordances, capabilities
- **Agent layer** — plans, tasks, execution tracking, governance policies

Context is proactively injected each turn using score-gated semantic retrieval (threshold: 0.55) — the agent gets relevant knowledge without explicit tool calls. Tools are available for deeper queries and active knowledge capture.

## Requirements

- Python >= 3.10
- `mindgraph-sdk >= 0.1.4`
- A MindGraph API key
- Hermes Agent with plugin support

## License

MIT
