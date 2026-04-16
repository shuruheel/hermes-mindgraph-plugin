# hermes-mindgraph-plugin

[![PyPI](https://img.shields.io/pypi/v/hermes-mindgraph-plugin)](https://pypi.org/project/hermes-mindgraph-plugin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

MindGraph semantic graph memory plugin for [Hermes Agent](https://github.com/shuruheel/hermes-agent). One install gives any Hermes-powered agent persistent, structured memory across sessions.

## What You Get

**5 tools** for persistent semantic memory:

| Tool | Purpose |
|------|---------|
| `mindgraph_remember` | Store knowledge — entities, observations, claims, preferences, notes |
| `mindgraph_retrieve` | Query the graph — hybrid FTS + semantic, structured queries, traversal |
| `mindgraph_commit` | Track agentic state — goals, decisions, plans, tasks, risks, questions |
| `mindgraph_ingest` | Bulk content ingestion (articles, transcripts, code) |
| `mindgraph_synthesize` | Project-scoped cross-document synthesis — mine signals, spawn Article-generation jobs |

**A bundled behavioral-contract skill** (`skill.md`) installed to `~/.hermes/skills/mindgraph/SKILL.md` on first load. The agent sees it like any other Hermes skill and uses it to decide when to store, when to retrieve, and when to synthesize — without being prompted.

**Session lifecycle hooks** for automatic bookkeeping: on `on_session_start` the plugin opens a MindGraph session node; on `on_session_end` it closes it. (These Hermes hooks are declared but not yet invoked by the agent core — the tools themselves work today regardless.)

## Install

```bash
pip install hermes-mindgraph-plugin
```

The plugin registers itself via the `hermes_agent.plugins` entry point — Hermes discovers it on next startup, no config changes needed.

## Configure

Set your MindGraph API key in `~/.hermes/.env` (get one at [mindgraph.cloud](https://mindgraph.cloud)):

```bash
MINDGRAPH_API_KEY=mg_your_key_here
```

If `MINDGRAPH_API_KEY` isn't set, Hermes disables the plugin cleanly (per the manifest's `requires_env` gate) rather than crashing.

Optional:

```bash
MINDGRAPH_BASE_URL=https://api.mindgraph.cloud   # override for self-hosted deployments
MINDGRAPH_AGENT_ID=hermes                         # stamped onto every write for provenance
```

## How It's Wired

```
pip install hermes-mindgraph-plugin
        │
        ▼
Hermes startup discovers the plugin via `hermes_agent.plugins` entry point
        │
        ▼
Plugin's `register(ctx)` runs once
        │
        ├── ctx.register_tool(name=..., toolset="mindgraph", schema=..., handler=...)  × 5
        ├── ctx.register_hook("on_session_start", …)
        ├── ctx.register_hook("on_session_end", …)
        └── copies skill.md → ~/.hermes/skills/mindgraph/SKILL.md
```

Run `/plugins` inside Hermes to confirm:

```
Plugins (1):
  ✓ mindgraph v0.9.0 (5 tools, 2 hooks)
```

## Tool details

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
| `document_index` | List all ingested documents (orient before searching). |
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

Ingests long-form content (articles, transcripts, code). Under 500 chars: sync. Over 500 chars: async with job tracking via the returned `job_id`.

### `mindgraph_synthesize` — project-scoped synthesis

| Action | What it does |
|--------|-------------|
| `signals` | Mine cross-document structural signals (entity bridges, claim hubs, theory gaps, concept clusters, analogy candidates, dialectical pairs). No LLM. |
| `run` | Spawn a background synthesis job that turns top idea clusters into Article nodes. Returns a `job_id`. |
| `job_status` | Poll a synthesis job. |

Create a project via `mindgraph_commit(action='project')`, link documents to it with the `PartOfProject` edge, then run `mindgraph_synthesize`.

## Architecture

The plugin uses MindGraph's [5-layer cognitive architecture](https://mindgraph.cloud):

- **Reality layer** — entities, observations, facts about the world
- **Epistemic layer** — claims, evidence, questions, hypotheses, concepts
- **Intent layer** — goals, decisions, projects, milestones
- **Action layer** — risks, affordances, capabilities
- **Agent layer** — plans, tasks, execution tracking, governance policies

The bundled skill (`skill.md`) teaches the agent a decision tree for routing knowledge across these layers, plus anti-patterns to avoid. It's installed once on first plugin load and never overwritten if you edit it.

## Requirements

- Python ≥ 3.10
- `mindgraph-sdk ≥ 0.4.0`
- A MindGraph API key
- Hermes Agent with the plugin system (any recent version)

## License

MIT
