# MindGraph memory plugin installed

Activate it as your memory provider:

```
hermes memory setup hermes-mindgraph-plugin
```

That will:

1. Install `mindgraph-sdk` into the Hermes environment.
2. Prompt for your `MINDGRAPH_API_KEY` (get one at https://mindgraph.cloud).
3. Write `memory.provider: hermes-mindgraph-plugin` to `~/.hermes/config.yaml`.

Next conversation, Hermes will:

- Open a MindGraph session node on the first turn.
- Bake your active goals, policies, open questions, and open decisions into the system prompt.
- Prefetch topic-relevant graph context each turn.
- Expose 5 tools to the model: `mindgraph_remember`, `mindgraph_retrieve`, `mindgraph_commit`, `mindgraph_ingest`, `mindgraph_synthesize`.
- Ingest the full transcript on session end for 5-layer extraction (Reality, Epistemic, Intent, Action, Memory).

Docs: https://github.com/shuruheel/hermes-mindgraph-plugin
