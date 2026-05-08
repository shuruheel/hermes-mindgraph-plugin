"""MindGraph memory provider plugin for Hermes Agent.

Installs via::

    hermes plugins install shuruheel/mindgraph
    hermes memory setup   # interactive picker — select mindgraph

Once activated (``memory.provider: mindgraph`` in
``~/.hermes/config.yaml``), the MemoryManager calls this provider's
lifecycle methods on every conversation: opening a MindGraph session,
baking active goals/policies/questions into the system prompt, prefetching
topic-relevant context each turn, exposing 5 graph-memory tools, and
ingesting the full transcript on session end for 5-layer extraction.

See ``provider.py`` for the full ``MindGraphMemoryProvider`` implementation
and ``tools.py`` for the SDK integration.
"""

from __future__ import annotations

import logging

__version__ = "0.10.0"


# ---------------------------------------------------------------------------
# Submodule bootstrap
# ---------------------------------------------------------------------------
#
# Hermes's user-plugin loader (``plugins/memory/__init__.py``:_load_provider_
# from_dir) synthesizes a package name like ``_hermes_user_memory.<dirname>``
# for user-installed plugins, but:
#
#   1. It does not register the ``_hermes_user_memory`` parent namespace in
#      ``sys.modules``, so relative imports that traverse through it fail
#      with ``No module named '_hermes_user_memory'``.
#
#   2. It pre-execs sibling submodules (``tools.py``, ``provider.py``) in
#      filesystem-glob order BEFORE ``__init__.py`` runs. If ``provider.py``
#      is processed first, its ``from . import tools as _tools`` fires
#      while ``tools`` hasn't been registered yet, leaving a stub module
#      in ``sys.modules`` with no exports.
#
# Both issues break relative imports in multi-file user-installed plugins.
# We fix both at ``__init__.py`` load time: register a stub parent namespace,
# then explicitly (re)load submodules in dependency order (tools → provider).
# Bundled plugins at ``plugins/memory/<name>/`` don't hit either issue because
# the loader pre-registers ``plugins`` and ``plugins.memory``.

def _bootstrap_submodules() -> None:
    import importlib.util
    import sys
    import types
    from pathlib import Path

    pkg_name = __name__
    pkg_dir = Path(__file__).parent

    # 1. Register stub parent namespace(s) so relative imports can traverse.
    if "." in pkg_name:
        parent = pkg_name.rsplit(".", 1)[0]
        if parent and parent not in sys.modules:
            stub = types.ModuleType(parent)
            stub.__path__ = []  # mark as package
            sys.modules[parent] = stub

    # 2. Load sibling submodules in dependency order, skipping anything
    #    that was already loaded cleanly. A module counts as "already loaded"
    #    when it exposes the sentinel attribute we look for.
    _load_order = (
        ("tools", "check_requirements"),
        ("provider", "MindGraphMemoryProvider"),
    )
    for sub_name, required_attr in _load_order:
        sub_file = pkg_dir / f"{sub_name}.py"
        if not sub_file.exists():
            continue
        full_name = f"{pkg_name}.{sub_name}"
        existing = sys.modules.get(full_name)
        if existing is not None and hasattr(existing, required_attr):
            continue  # already loaded cleanly — don't reset module state
        spec = importlib.util.spec_from_file_location(full_name, str(sub_file))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)


_bootstrap_submodules()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# register(ctx) — called once by Hermes when discovering the plugin
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Hand a MindGraphMemoryProvider to the plugin context.

    Tools, prefetch, and session lifecycle all route through that provider
    once the MemoryManager activates it (triggered by ``memory.provider``
    in ``~/.hermes/config.yaml``).
    """
    from .provider import MindGraphMemoryProvider

    ctx.register_memory_provider(MindGraphMemoryProvider())
    logger.info("MindGraph memory provider v%s registered", __version__)
