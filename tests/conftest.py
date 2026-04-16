"""Test config — loads the flat-layout plugin under a stable import name.

Hermes clones this plugin into ``~/.hermes/plugins/mindgraph/`` and loads the
repo-root ``__init__.py`` directly — the repo root IS the package.
The repo directory name contains hyphens, so it can't be imported via a normal
``import`` statement. This conftest uses ``importlib.util`` to load the
package under the name ``hermes_mindgraph_plugin`` so tests can ``import``
it the same way regardless of where the repo lives on disk.

It also injects a minimal stub of ``agent.memory_provider`` so the provider
module imports cleanly outside a Hermes runtime.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from abc import ABC
from pathlib import Path


# ---------------------------------------------------------------------------
# Stub agent.memory_provider when Hermes isn't on the path
# ---------------------------------------------------------------------------

def _ensure_memory_provider_stub() -> None:
    if "agent.memory_provider" in sys.modules:
        return

    class _StubMemoryProvider(ABC):
        """Structural stand-in for the real MemoryProvider ABC."""

        @property
        def name(self) -> str:
            raise NotImplementedError

        def is_available(self) -> bool:
            raise NotImplementedError

        def initialize(self, session_id: str, **kwargs) -> None:
            pass

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            pass

        def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
            pass

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, tool_name: str, args, **kwargs) -> str:
            return "{}"

        def shutdown(self) -> None:
            pass

        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
            pass

        def on_session_end(self, messages) -> None:
            pass

        def on_pre_compress(self, messages) -> str:
            return ""

        def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
            pass

        def get_config_schema(self):
            return []

        def save_config(self, values, hermes_home) -> None:
            pass

        def on_memory_write(self, action: str, target: str, content: str) -> None:
            pass

    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []
    mp_mod = types.ModuleType("agent.memory_provider")
    mp_mod.MemoryProvider = _StubMemoryProvider
    sys.modules.setdefault("agent", agent_pkg)
    sys.modules["agent.memory_provider"] = mp_mod


# ---------------------------------------------------------------------------
# Load the flat-layout repo under the name ``hermes_mindgraph_plugin``
# ---------------------------------------------------------------------------

def _load_flat_plugin_as_package() -> None:
    if "hermes_mindgraph_plugin" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parent.parent
    init_path = repo_root / "__init__.py"
    assert init_path.exists(), f"Expected plugin __init__.py at {init_path}"

    # Pre-register submodules so relative imports in __init__.py / provider.py
    # resolve correctly regardless of where the repo lives on disk.
    pkg_name = "hermes_mindgraph_plugin"
    spec = importlib.util.spec_from_file_location(
        pkg_name, str(init_path),
        submodule_search_locations=[str(repo_root)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod

    for sub in ("tools", "provider"):
        sub_file = repo_root / f"{sub}.py"
        if not sub_file.exists():
            continue
        full = f"{pkg_name}.{sub}"
        if full in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(full, str(sub_file))
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[full] = sub_mod
        sub_spec.loader.exec_module(sub_mod)

    spec.loader.exec_module(mod)


_ensure_memory_provider_stub()
_load_flat_plugin_as_package()
