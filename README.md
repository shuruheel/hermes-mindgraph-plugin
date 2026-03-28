# hermes-mindgraph-plugin

Semantic graph memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Gives any Hermes-powered agent persistent memory across sessions using [MindGraph](https://mindgraph.cloud).

## What it does

This plugin hooks into Hermes Agent's lifecycle to provide:

- **Session context** — Active goals, open decisions, governance policies, weak claims, and user profile are injected into the agent's context at the start of each conversation.
- **Per-turn retrieval** — Score-gated semantic search finds topic-relevant knowledge from the graph and injects it ephemerally (never persisted to conversation history, no prompt cache breakage).
- **Session lifecycle** — Automatically opens and closes MindGraph sessions aligned with Hermes conversations.

## Install

```bash
pip install hermes-mindgraph-plugin
```

The plugin is auto-discovered via the `hermes_agent.plugins` entry point. No configuration beyond the API key is needed.

## Setup

1. Get a MindGraph API key at [mindgraph.cloud](https://mindgraph.cloud)
2. Add it to your Hermes environment:

```bash
echo "MINDGRAPH_API_KEY=mg_your_key_here" >> ~/.hermes/.env
```

3. Verify the plugin is loaded:

```bash
hermes plugins list
```

## How it works

The plugin registers three [lifecycle hooks](https://hermes.nousresearch.com/docs/user-guide/features/plugins):

| Hook | When | What |
|------|------|------|
| `on_session_start` | New conversation begins | Opens a MindGraph session, pre-fetches goals/decisions/policies |
| `pre_llm_call` | Before each agent turn | Returns session context (first turn) + semantic retrieval (every turn) |
| `on_session_end` | Conversation ends | Closes the MindGraph session |

Context is injected **ephemerally** into the system prompt — it's never persisted to the session database and doesn't break Anthropic's prompt cache prefix.

All MindGraph API calls are wrapped in try/except. A MindGraph failure never breaks the conversation.

## Manual install (alternative)

If you prefer not to use pip, copy the plugin directly:

```bash
mkdir -p ~/.hermes/plugins/mindgraph
# Copy plugin.yaml and __init__.py into that directory
```

See the `plugin.yaml` in this repo for the manifest format.

## Development

```bash
git clone https://github.com/shuruheel/hermes-mindgraph-plugin
cd hermes-mindgraph-plugin
pip install -e ".[dev]"
pytest
```

## Requirements

- Hermes Agent ≥ v0.5.0 (plugin lifecycle hooks)
- Python ≥ 3.10
- `mindgraph-sdk` ≥ 0.1.4
- `MINDGRAPH_API_KEY` environment variable

## License

MIT
