"""Tests for hermes-mindgraph-plugin."""

from unittest.mock import MagicMock, patch

import pytest

import hermes_mindgraph_plugin as plugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_plugin_state():
    """Reset module-level state between tests."""
    plugin._session_context_cache = None
    plugin._session_started = False
    plugin._accumulated_messages = []
    plugin._current_session_id = None
    yield
    plugin._session_context_cache = None
    plugin._session_started = False
    plugin._accumulated_messages = []
    plugin._current_session_id = None


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:

    def test_registers_three_hooks(self):
        ctx = MagicMock()
        plugin.register(ctx)
        assert ctx.register_hook.call_count == 3
        hook_names = {call.args[0] for call in ctx.register_hook.call_args_list}
        assert hook_names == {"on_session_start", "pre_llm_call", "on_session_end"}

    def test_registers_eleven_tools(self):
        ctx = MagicMock()
        plugin.register(ctx)
        assert ctx.register_tool.call_count == 11
        tool_names = {call.kwargs["name"] for call in ctx.register_tool.call_args_list}
        expected = {
            "mindgraph_session", "mindgraph_journal", "mindgraph_argue",
            "mindgraph_commit", "mindgraph_retrieve", "mindgraph_ingest",
            "mindgraph_capture", "mindgraph_inquire", "mindgraph_action",
            "mindgraph_decide", "mindgraph_plan",
        }
        assert tool_names == expected

    def test_all_tools_have_mindgraph_toolset(self):
        ctx = MagicMock()
        plugin.register(ctx)
        for call in ctx.register_tool.call_args_list:
            toolset = call.kwargs["toolset"]
            assert toolset == "mindgraph"

    def test_all_tools_require_api_key(self):
        ctx = MagicMock()
        plugin.register(ctx)
        for call in ctx.register_tool.call_args_list:
            requires_env = call.kwargs.get("requires_env")
            assert requires_env == ["MINDGRAPH_API_KEY"]

    def test_callbacks_are_callable(self):
        ctx = MagicMock()
        plugin.register(ctx)
        for call in ctx.register_hook.call_args_list:
            assert callable(call.args[1])


# ---------------------------------------------------------------------------
# on_session_start
# ---------------------------------------------------------------------------

class TestOnSessionStart:

    @patch("hermes_mindgraph_plugin._is_available", return_value=False)
    def test_noop_when_unavailable(self, _mock):
        plugin._on_session_start(session_id="abc123")
        assert not plugin._session_started
        assert plugin._session_context_cache is None

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_opens_session_and_prefetches(self, _avail):
        mock_open = MagicMock(return_value="sess-uid-123")
        mock_ctx = MagicMock(return_value="## Goals\n- Ship it")

        with patch("hermes_mindgraph_plugin.tools.auto_open_session", mock_open), \
             patch("hermes_mindgraph_plugin.tools.retrieve_session_context", mock_ctx):
            plugin._on_session_start(session_id="abcdef99", model="test", platform="cli")

        assert plugin._session_started is True
        assert plugin._session_context_cache == "## Goals\n- Ship it"
        mock_open.assert_called_once_with(label="hermes-abcdef99")

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_cron_reads_context_but_skips_session_open(self, _avail):
        """Cron sessions should fetch context (read) but not open a session (write)."""
        mock_open = MagicMock(return_value="session-uid")
        mock_ctx = MagicMock(return_value="## Goals\n- Ship it")

        with patch("hermes_mindgraph_plugin.tools.auto_open_session", mock_open), \
             patch("hermes_mindgraph_plugin.tools.retrieve_session_context", mock_ctx):
            plugin._on_session_start(session_id="cron-job-1", model="test", platform="cron")

        assert not plugin._session_started  # No session opened
        mock_open.assert_not_called()       # auto_open_session NOT called
        mock_ctx.assert_called_once()       # But context WAS fetched
        assert plugin._session_context_cache == "## Goals\n- Ship it"
        assert plugin._is_cron_session is True

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_session_start_failure_is_nonfatal(self, _avail):
        mock_open = MagicMock(side_effect=RuntimeError("connection refused"))

        with patch("hermes_mindgraph_plugin.tools.auto_open_session", mock_open), \
             patch("hermes_mindgraph_plugin.tools.retrieve_session_context", MagicMock(return_value=None)):
            # Should not raise
            plugin._on_session_start(session_id="test")

        assert not plugin._session_started


# ---------------------------------------------------------------------------
# pre_llm_call
# ---------------------------------------------------------------------------

class TestPreLlmCall:

    @patch("hermes_mindgraph_plugin._is_available", return_value=False)
    def test_returns_none_when_unavailable(self, _mock):
        result = plugin._pre_llm_call(
            session_id="x", user_message="hello", is_first_turn=True,
        )
        assert result is None

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_returns_session_context_on_first_turn(self, _avail):
        plugin._session_context_cache = "## Active Goals\n- Goal 1"
        result = plugin._pre_llm_call(
            session_id="x", user_message="hi",
            conversation_history=[], is_first_turn=True,
        )
        assert result is not None
        assert "Active Goals" in result["context"]

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_no_session_context_on_subsequent_turn(self, _avail):
        plugin._session_context_cache = "## Active Goals\n- Goal 1"
        result = plugin._pre_llm_call(
            session_id="x", user_message="ok",
            conversation_history=[{"role": "user"}], is_first_turn=False,
        )
        # Short message + not first turn = no retrieval, no session context
        assert result is None

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_returns_turn_context_from_retrieval(self, _avail):
        mock_retrieve = MagicMock(return_value="## Relevant: plugin hooks")

        with patch("hermes_mindgraph_plugin.tools.proactive_graph_retrieve", mock_retrieve):
            result = plugin._pre_llm_call(
                session_id="x",
                user_message="Tell me about the plugin lifecycle hooks",
                conversation_history=[{"role": "user"}],
                is_first_turn=False,
            )

        assert result is not None
        assert "plugin hooks" in result["context"]
        mock_retrieve.assert_called_once()

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_combines_session_and_turn_context(self, _avail):
        plugin._session_context_cache = "## Goals\n- Ship plugin"
        mock_retrieve = MagicMock(return_value="## Related: memory systems")

        with patch("hermes_mindgraph_plugin.tools.proactive_graph_retrieve", mock_retrieve):
            result = plugin._pre_llm_call(
                session_id="x",
                user_message="How does memory work in Hermes?",
                conversation_history=[],
                is_first_turn=True,
            )

        assert result is not None
        ctx = result["context"]
        assert "## Goals" in ctx
        assert "## Related" in ctx

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_retrieval_failure_is_nonfatal(self, _avail):
        with patch("hermes_mindgraph_plugin.tools.proactive_graph_retrieve",
                    MagicMock(side_effect=Exception("timeout"))):
            result = plugin._pre_llm_call(
                session_id="x",
                user_message="Tell me about something complex",
                is_first_turn=False,
            )
        # Should not raise, returns None
        assert result is None

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_preserves_cache_across_calls(self, _avail):
        """Session context cache is not cleared after first use."""
        plugin._session_context_cache = "## Goals\n- Persist"

        plugin._pre_llm_call(
            session_id="x", user_message="hi",
            conversation_history=[], is_first_turn=True,
        )
        # Cache should still be there
        assert plugin._session_context_cache == "## Goals\n- Persist"


# ---------------------------------------------------------------------------
# on_session_end
# ---------------------------------------------------------------------------

class TestOnSessionEnd:

    def test_noop_when_not_started(self):
        plugin._session_started = False
        # Should not raise or call anything
        plugin._on_session_end(session_id="test", completed=True)
        assert not plugin._session_started

    def test_normal_completion_defers_close(self):
        """Normal completed sessions defer close to next session_start (or atexit)."""
        plugin._session_started = True
        plugin._session_context_cache = "some cached context"

        plugin._on_session_end(
            session_id="abcdef99", completed=True, interrupted=False,
        )

        # Session should still be "started" — close is deferred
        assert plugin._session_started is True

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_interrupted_session_closes_immediately(self, _avail):
        plugin._session_started = True
        plugin._session_context_cache = "some cached context"
        mock_close = MagicMock()

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            plugin._on_session_end(
                session_id="abcdef99", completed=True, interrupted=True,
            )

        assert not plugin._session_started
        assert plugin._session_context_cache is None
        mock_close.assert_called_once()

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_close_failure_is_nonfatal(self, _avail):
        plugin._session_started = True
        mock_close = MagicMock(side_effect=RuntimeError("api down"))

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            # Should not raise
            plugin._on_session_end(session_id="test", completed=True, interrupted=True)

        # State should be cleaned up even on failure
        assert not plugin._session_started


# ---------------------------------------------------------------------------
# Message accumulation
# ---------------------------------------------------------------------------

class TestAccumulateMessages:

    def test_accumulates_user_and_assistant_only(self):
        history = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "Thanks"},
        ]
        plugin._accumulate_messages(history)
        assert len(plugin._accumulated_messages) == 3
        assert plugin._accumulated_messages[0]["role"] == "user"
        assert plugin._accumulated_messages[1]["role"] == "assistant"
        assert plugin._accumulated_messages[2]["role"] == "user"

    def test_grows_incrementally(self):
        history1 = [{"role": "user", "content": "Hello"}]
        plugin._accumulate_messages(history1)
        assert len(plugin._accumulated_messages) == 1

        history2 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        plugin._accumulate_messages(history2)
        assert len(plugin._accumulated_messages) == 2


# ---------------------------------------------------------------------------
# Tools module
# ---------------------------------------------------------------------------

class TestCaptureTypedRouting:
    """Test that mindgraph_capture routes to typed SDK methods."""

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_person_routes_to_find_or_create_person(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        client.find_or_create_person.return_value = {"uid": "p1", "label": "Alice"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Alice", capture_type="entity", entity_type="person"))
        assert result["success"] is True
        client.find_or_create_person.assert_called_once_with("Alice", props=None)
        client.find_or_create_entity.assert_not_called()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_organization_routes_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        client.find_or_create_organization.return_value = {"uid": "o1", "label": "Acme"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Acme", capture_type="entity", entity_type="organization"))
        assert result["success"] is True
        client.find_or_create_organization.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_nation_routes_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        client.find_or_create_nation.return_value = {"uid": "n1", "label": "Japan"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Japan", capture_type="entity", entity_type="nation"))
        assert result["success"] is True
        client.find_or_create_nation.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_concept_routes_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        client.find_or_create_concept.return_value = {"uid": "c1", "label": "Entropy"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Entropy", capture_type="entity", entity_type="concept"))
        assert result["success"] is True
        client.find_or_create_concept.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_work_falls_back_to_generic(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        client.find_or_create_entity.return_value = {"uid": "w1", "label": "Hamlet"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Hamlet", capture_type="entity", entity_type="work"))
        assert result["success"] is True
        client.find_or_create_entity.assert_called_once_with("Hamlet", props={"entity_type": "work"})

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_properties_passed_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        client.find_or_create_person.return_value = {"uid": "p2", "label": "Bob"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture(
            "Bob", capture_type="entity", entity_type="person",
            properties={"role": "engineer", "company": "Acme"},
        ))
        assert result["success"] is True
        client.find_or_create_person.assert_called_once_with(
            "Bob", props={"role": "engineer", "company": "Acme"},
        )

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_all_typed_methods_covered(self, mock_get):
        """Ensure all 6 typed entity types route to their specific method."""
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        import json
        client = MagicMock()
        mock_get.return_value = client

        for etype, method in [
            ("person", "find_or_create_person"),
            ("organization", "find_or_create_organization"),
            ("nation", "find_or_create_nation"),
            ("event", "find_or_create_event"),
            ("place", "find_or_create_place"),
            ("concept", "find_or_create_concept"),
        ]:
            getattr(client, method).return_value = {"uid": "x", "label": "test"}
            mindgraph_capture("test", capture_type="entity", entity_type=etype)
            getattr(client, method).assert_called()


class TestToolsModule:

    def test_all_tools_have_required_keys(self):
        from hermes_mindgraph_plugin.tools import TOOLS
        required_keys = {"name", "toolset", "schema", "handler", "check_fn", "requires_env", "emoji"}
        for tool in TOOLS:
            assert required_keys.issubset(tool.keys()), f"Tool {tool.get('name')} missing keys"

    def test_all_handlers_are_callable(self):
        from hermes_mindgraph_plugin.tools import TOOLS
        for tool in TOOLS:
            assert callable(tool["handler"]), f"Tool {tool['name']} handler not callable"

    def test_all_schemas_have_name_and_parameters(self):
        from hermes_mindgraph_plugin.tools import TOOLS
        for tool in TOOLS:
            schema = tool["schema"]
            assert "name" in schema, f"Tool {tool['name']} schema missing 'name'"
            assert "parameters" in schema, f"Tool {tool['name']} schema missing 'parameters'"
            assert schema["name"] == tool["name"], f"Tool {tool['name']} schema name mismatch"
