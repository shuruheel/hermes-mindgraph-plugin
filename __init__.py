"""Hermes directory-plugin shim for ``hermes plugins install``.

When installed via ``hermes plugins install shuruheel/hermes-mindgraph-plugin``,
the plugin loader imports this ``__init__.py`` from the cloned repo root.

When installed via ``pip install hermes-mindgraph-plugin``, the entry-point
system loads ``hermes_mindgraph_plugin/`` directly and this file is never used.
"""

import importlib.util
import sys
from pathlib import Path


def register(ctx):
    """Load the real plugin package and delegate registration."""
    _pkg_init = Path(__file__).parent / "hermes_mindgraph_plugin" / "__init__.py"

    if not _pkg_init.exists():
        return  # Broken install — skip silently

    # Load the real package into sys.modules so its internal imports work.
    _name = "hermes_mindgraph_plugin"
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            _name, str(_pkg_init),
            submodule_search_locations=[str(_pkg_init.parent)],
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)

    sys.modules[_name].register(ctx)
