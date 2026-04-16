"""Tests for hermes-mindgraph-plugin (Hermes plugin interface).

These tests verify the plugin wiring against the Hermes
``PluginContext`` API — ``register_tool`` and ``register_hook``. They do
not hit the MindGraph backend; every outbound SDK call is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# register(ctx)
# ---------------------------------------------------------------------------


class TestRegister:
    """The plugin entry point wires tools + hooks onto the PluginContext."""

    def test_registers_all_tools(self):
        import hermes_mindgraph_plugin as plugin
        from hermes_mindgraph_plugin.tools import TOOLS

        ctx = MagicMock()
        plugin.register(ctx)

        assert ctx.register_tool.call_count == len(TOOLS)
        registered_names = {
            call.kwargs["name"] for call in ctx.register_tool.call_args_list
        }
        assert registered_names == {t["name"] for t in TOOLS}

    def test_registers_tool_with_hermes_signature(self):
        """Each register_tool call must use keyword args Hermes understands."""
        import hermes_mindgraph_plugin as plugin

        ctx = MagicMock()
        plugin.register(ctx)

        for call in ctx.register_tool.call_args_list:
            assert "name" in call.kwargs
            assert "toolset" in call.kwargs
            assert "schema" in call.kwargs
            assert "handler" in call.kwargs
            assert callable(call.kwargs["handler"])
            assert isinstance(call.kwargs["schema"], dict)

    def test_registers_session_lifecycle_hooks(self):
        import hermes_mindgraph_plugin as plugin

        ctx = MagicMock()
        plugin.register(ctx)

        hook_names = {call.args[0] for call in ctx.register_hook.call_args_list}
        assert "on_session_start" in hook_names
        assert "on_session_end" in hook_names

    def test_registered_tool_schemas_have_required_fields(self):
        """Tool schemas must match OpenAI function-calling shape."""
        import hermes_mindgraph_plugin as plugin

        ctx = MagicMock()
        plugin.register(ctx)

        for call in ctx.register_tool.call_args_list:
            schema = call.kwargs["schema"]
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert schema["parameters"].get("type") == "object"


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class TestToolHandlers:
    """Handlers always return a JSON string, even on error — never raise."""

    def _handler(self, name: str):
        from hermes_mindgraph_plugin.tools import TOOLS

        for t in TOOLS:
            if t["name"] == name:
                return t["handler"]
        raise AssertionError(f"Tool {name} not found")

    def test_remember_returns_json_string_on_missing_label(self):
        handler = self._handler("mindgraph_remember")
        result = handler({})  # No label
        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data

    def test_commit_returns_json_string_on_unknown_action(self):
        handler = self._handler("mindgraph_commit")
        result = handler({"action": "definitely_not_a_real_action"})
        data = json.loads(result)
        assert data["success"] is False

    def test_ingest_returns_json_string_on_empty_content(self):
        handler = self._handler("mindgraph_ingest")
        result = handler({"content": ""})
        data = json.loads(result)
        assert data["success"] is False
        assert "required" in data["error"].lower()

    def test_synthesize_returns_json_string_on_missing_project_uid(self):
        handler = self._handler("mindgraph_synthesize")
        result = handler({"action": "signals"})  # No project_uid
        data = json.loads(result)
        assert data["success"] is False
        assert "project_uid" in data["error"]

    def test_synthesize_unknown_action_rejected(self):
        handler = self._handler("mindgraph_synthesize")
        result = handler({"action": "foo", "project_uid": "p1"})
        data = json.loads(result)
        assert data["success"] is False


# ---------------------------------------------------------------------------
# Session lifecycle hooks
# ---------------------------------------------------------------------------


class TestSessionHooks:
    """Hooks are registered callables that never raise."""

    def test_on_session_start_calls_auto_open(self):
        import hermes_mindgraph_plugin as plugin

        with patch(
            "hermes_mindgraph_plugin.tools.auto_open_session"
        ) as mock_open:
            mock_open.return_value = "sess_123"
            plugin._on_session_start(session_id="abcdef1234", platform="cli")
            mock_open.assert_called_once()
            assert "abcdef12" in mock_open.call_args.kwargs["label"]

    def test_on_session_end_calls_auto_close(self):
        import hermes_mindgraph_plugin as plugin

        with patch(
            "hermes_mindgraph_plugin.tools.auto_close_session"
        ) as mock_close:
            plugin._on_session_end(session_id="abcdef1234", platform="cli")
            mock_close.assert_called_once()

    def test_hooks_swallow_exceptions(self):
        """A broken MindGraph call must not propagate out of a hook."""
        import hermes_mindgraph_plugin as plugin

        with patch(
            "hermes_mindgraph_plugin.tools.auto_open_session",
            side_effect=RuntimeError("backend down"),
        ):
            # Should not raise
            plugin._on_session_start(session_id="x", platform="cli")

        with patch(
            "hermes_mindgraph_plugin.tools.auto_close_session",
            side_effect=RuntimeError("backend down"),
        ):
            plugin._on_session_end(session_id="x", platform="cli")


# ---------------------------------------------------------------------------
# Skill installation
# ---------------------------------------------------------------------------


class TestSkillInstall:
    def test_install_skill_copies_file(self, tmp_path):
        """When hermes_cli isn't available, fall back to ~/.hermes via Path.home()."""
        import hermes_mindgraph_plugin as plugin

        with patch("hermes_mindgraph_plugin.Path.home", return_value=tmp_path):
            plugin._install_skill()

        dest = tmp_path / ".hermes" / "skills" / "mindgraph" / "SKILL.md"
        assert dest.exists()
        content = dest.read_text()
        assert "MindGraph" in content
        assert "mindgraph_remember" in content

    def test_install_skill_does_not_overwrite_user_edits(self, tmp_path):
        import hermes_mindgraph_plugin as plugin

        dest_dir = tmp_path / ".hermes" / "skills" / "mindgraph"
        dest_dir.mkdir(parents=True)
        dest = dest_dir / "SKILL.md"
        dest.write_text("USER CUSTOM CONTENT")

        with patch("hermes_mindgraph_plugin.Path.home", return_value=tmp_path):
            plugin._install_skill()

        assert dest.read_text() == "USER CUSTOM CONTENT"


# ---------------------------------------------------------------------------
# Manifest-level sanity
# ---------------------------------------------------------------------------


class TestManifest:
    def test_plugin_yaml_matches_registered_tools(self):
        import yaml
        from pathlib import Path

        import hermes_mindgraph_plugin as plugin

        manifest_path = (
            Path(plugin.__file__).resolve().parent.parent / "plugin.yaml"
        )
        manifest = yaml.safe_load(manifest_path.read_text())

        ctx = MagicMock()
        plugin.register(ctx)
        registered = {c.kwargs["name"] for c in ctx.register_tool.call_args_list}

        assert set(manifest["provides_tools"]) == registered

    def test_version_in_sync(self):
        import yaml
        from pathlib import Path

        import hermes_mindgraph_plugin as plugin

        manifest_path = (
            Path(plugin.__file__).resolve().parent.parent / "plugin.yaml"
        )
        manifest = yaml.safe_load(manifest_path.read_text())
        assert str(manifest["version"]) == plugin.__version__
