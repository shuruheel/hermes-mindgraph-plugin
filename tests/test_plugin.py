"""Tests for hermes-mindgraph-plugin — MemoryProvider interface contract.

These tests verify:

1. ``register(ctx)`` invokes ``ctx.register_memory_provider`` exactly once.
2. The provider implements every method the MemoryManager calls.
3. Tool schemas match OpenAI function-calling shape.
4. Tool dispatch routes by name and returns JSON strings.
5. ``plugin.yaml`` is internally consistent with the code.

Backend calls are either mocked or short-circuited via missing credentials;
no tests hit the real MindGraph API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# register(ctx)
# ---------------------------------------------------------------------------


class TestRegister:
    """register(ctx) hands a single MemoryProvider instance to Hermes."""

    def test_registers_memory_provider(self):
        import hermes_mindgraph_plugin as plugin

        ctx = MagicMock()
        plugin.register(ctx)

        ctx.register_memory_provider.assert_called_once()
        (provider,), _ = ctx.register_memory_provider.call_args
        assert provider.name == "mindgraph"

    def test_register_does_not_touch_tool_or_hook_api(self):
        """Old pip-plugin registration paths are gone — nothing should call them."""
        import hermes_mindgraph_plugin as plugin

        ctx = MagicMock()
        plugin.register(ctx)

        ctx.register_tool.assert_not_called()
        ctx.register_hook.assert_not_called()


# ---------------------------------------------------------------------------
# MindGraphMemoryProvider — interface contract
# ---------------------------------------------------------------------------


class TestProviderContract:
    """The provider implements every MemoryManager-called method."""

    def _provider(self):
        from hermes_mindgraph_plugin.provider import MindGraphMemoryProvider

        return MindGraphMemoryProvider()

    def test_name(self):
        assert self._provider().name == "mindgraph"

    def test_is_available_requires_api_key(self):
        p = self._provider()
        with patch.dict(os.environ, {"MINDGRAPH_API_KEY": ""}, clear=False):
            # Explicitly clear the env var; patch.dict won't remove it unless we tell it to
            os.environ.pop("MINDGRAPH_API_KEY", None)
            assert p.is_available() is False

    def test_get_config_schema_shape(self):
        schema = self._provider().get_config_schema()
        assert isinstance(schema, list)
        keys = {field["key"] for field in schema}
        assert "api_key" in keys
        # api_key entry is secret and required
        api_key_entry = next(f for f in schema if f["key"] == "api_key")
        assert api_key_entry.get("secret") is True
        assert api_key_entry.get("required") is True
        assert api_key_entry.get("env_var") == "MINDGRAPH_API_KEY"

    def test_get_tool_schemas_returns_five_tools(self):
        schemas = self._provider().get_tool_schemas()
        assert len(schemas) == 5
        names = {s["name"] for s in schemas}
        assert names == {
            "mindgraph_remember",
            "mindgraph_retrieve",
            "mindgraph_commit",
            "mindgraph_ingest",
            "mindgraph_synthesize",
        }

    def test_tool_schemas_have_openai_function_shape(self):
        for s in self._provider().get_tool_schemas():
            assert "name" in s
            assert "description" in s
            assert "parameters" in s
            assert s["parameters"].get("type") == "object"

    def test_lifecycle_methods_are_callable(self):
        """Every MemoryManager-invoked method must be present and not raise."""
        p = self._provider()
        # These call the MindGraph SDK but should swallow errors when the
        # client isn't configured (no MINDGRAPH_API_KEY set in test env).
        os.environ.pop("MINDGRAPH_API_KEY", None)

        p.initialize("test-session-123")
        _ = p.system_prompt_block()
        p.queue_prefetch("hello world")
        _ = p.prefetch("hello world")
        p.sync_turn("u", "a")
        p.on_session_end([{"role": "user", "content": "hi"}])
        p.shutdown()

    def test_non_primary_context_skips_session_open(self):
        """Subagents and cron runs should not open graph sessions."""
        from hermes_mindgraph_plugin import tools

        p = self._provider()
        with patch.object(tools, "auto_open_session") as mock_open:
            p.initialize("s1", agent_context="subagent")
            mock_open.assert_not_called()

    def test_primary_context_opens_session(self):
        from hermes_mindgraph_plugin import tools

        p = self._provider()
        with patch.object(tools, "auto_open_session", return_value="sess-uid") as mock_open:
            p.initialize("abcdef1234", agent_context="primary")
            mock_open.assert_called_once()
            assert "abcdef12" in mock_open.call_args.kwargs.get("label", "")


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    """handle_tool_call routes to the right handler and always returns JSON."""

    def _provider(self):
        from hermes_mindgraph_plugin.provider import MindGraphMemoryProvider

        return MindGraphMemoryProvider()

    def test_unknown_tool_returns_error_json(self):
        result = self._provider().handle_tool_call("not_a_real_tool", {})
        data = json.loads(result)
        assert data["success"] is False
        assert "Unknown tool" in data["error"]

    def test_remember_missing_label_returns_error_json(self):
        result = self._provider().handle_tool_call("mindgraph_remember", {})
        data = json.loads(result)
        assert data["success"] is False

    def test_commit_unknown_action_returns_error_json(self):
        result = self._provider().handle_tool_call(
            "mindgraph_commit", {"action": "definitely_not_real", "label": "x"}
        )
        data = json.loads(result)
        assert data["success"] is False

    def test_ingest_empty_content_returns_error_json(self):
        result = self._provider().handle_tool_call("mindgraph_ingest", {"content": ""})
        data = json.loads(result)
        assert data["success"] is False
        assert "required" in data["error"].lower()

    def test_synthesize_missing_project_uid_returns_error_json(self):
        result = self._provider().handle_tool_call(
            "mindgraph_synthesize", {"action": "signals"}
        )
        data = json.loads(result)
        assert data["success"] is False
        assert "project_uid" in data["error"]


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------


class TestManifest:
    """plugin.yaml is internally consistent with the code."""

    def _manifest(self):
        import yaml
        import hermes_mindgraph_plugin as plugin

        # The flat-layout plugin's __file__ is the repo-root __init__.py,
        # so the manifest lives next to it, not one level up.
        path = Path(plugin.__file__).resolve().parent / "plugin.yaml"
        return yaml.safe_load(path.read_text())

    def test_version_matches_package(self):
        import hermes_mindgraph_plugin as plugin

        assert str(self._manifest()["version"]) == plugin.__version__

    def test_manifest_version_one(self):
        assert self._manifest().get("manifest_version") == 1

    def test_requires_env_declares_api_key(self):
        manifest = self._manifest()
        env = manifest.get("requires_env") or []
        names = {e["name"] if isinstance(e, dict) else e for e in env}
        assert "MINDGRAPH_API_KEY" in names

    def test_pip_dependencies_include_sdk(self):
        deps = self._manifest().get("pip_dependencies") or []
        assert any(dep.startswith("mindgraph-sdk") for dep in deps)
