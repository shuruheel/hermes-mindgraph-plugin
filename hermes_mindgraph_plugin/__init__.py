"""MindGraph memory provider plugin for Hermes Agent.

Installs as a Hermes memory-provider plugin via::

    hermes plugins install shuruheel/hermes-mindgraph-plugin
    hermes memory setup hermes-mindgraph-plugin

Once activated (``memory.provider: hermes-mindgraph-plugin`` in
``~/.hermes/config.yaml``), the MemoryManager calls this provider's
lifecycle methods on every conversation: opening a MindGraph session,
baking active goals/policies/questions into the system prompt,
prefetching topic-relevant context each turn, exposing 5 graph-memory
tools to the model, and ingesting the full transcript on session end
for 5-layer extraction.

See ``provider.py`` for the full ``MindGraphMemoryProvider`` implementation
and ``tools.py`` for the underlying SDK integration.
"""

from __future__ import annotations

__version__ = "0.10.0"

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Hermes plugin entry point — called once at discovery.

    Registers a single MindGraphMemoryProvider on the plugin context.
    Tools, prefetch, and session lifecycle are all routed through that
    provider once the MemoryManager activates it.
    """
    from hermes_mindgraph_plugin.provider import MindGraphMemoryProvider

    ctx.register_memory_provider(MindGraphMemoryProvider())
    logger.info("MindGraph memory provider v%s registered", __version__)
