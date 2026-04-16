"""Test config — stubs Hermes runtime modules so the plugin imports outside Hermes.

When Hermes loads the plugin, ``agent.memory_provider`` is available. In this
repo's tests we are NOT running inside Hermes, so we inject a minimal stub
of that module into ``sys.modules`` before the plugin is imported.
"""

from __future__ import annotations

import sys
import types
from abc import ABC


def _ensure_memory_provider_stub() -> None:
    """Install a minimal ``agent.memory_provider`` shim if Hermes isn't present."""
    if "agent.memory_provider" in sys.modules:
        return

    class _StubMemoryProvider(ABC):
        """Structural stand-in for the real MemoryProvider ABC.

        Mirrors the method surface the real base class declares abstract so
        subclasses can be instantiated in-process for testing. The real class
        is always substituted at runtime when Hermes loads the plugin.
        """

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
    agent_pkg.__path__ = []  # mark as package
    mp_mod = types.ModuleType("agent.memory_provider")
    mp_mod.MemoryProvider = _StubMemoryProvider
    sys.modules.setdefault("agent", agent_pkg)
    sys.modules["agent.memory_provider"] = mp_mod


_ensure_memory_provider_stub()
