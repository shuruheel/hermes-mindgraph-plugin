"""Tests for hermes-mindgraph-plugin."""

import sys
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
    yield
    plugin._session_context_cache = None
    plugin._session_started = False


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

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(
                auto_open_session=mock_open,
                retrieve_session_context=mock_ctx,
            ),
        }):
            plugin._on_session_start(session_id="abcdef99", model="test", platform="cli")

        assert plugin._session_started is True
        assert plugin._session_context_cache == "## Goals\n- Ship it"
        mock_open.assert_called_once_with(label="hermes-abcdef99")

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_skips_session_for_cron_platform(self, _avail):
        """Cron sessions should not open MindGraph sessions or prefetch context."""
        mock_open = MagicMock(return_value="sess-uid-123")

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(
                auto_open_session=mock_open,
                retrieve_session_context=MagicMock(return_value="context"),
            ),
        }):
            plugin._on_session_start(session_id="cron-job-1", model="test", platform="cron")

        assert not plugin._session_started
        assert plugin._session_context_cache is None
        mock_open.assert_not_called()

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_session_start_failure_is_nonfatal(self, _avail):
        mock_open = MagicMock(side_effect=RuntimeError("connection refused"))

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(
                auto_open_session=mock_open,
                retrieve_session_context=MagicMock(return_value=None),
            ),
        }):
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

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(
                proactive_graph_retrieve=mock_retrieve,
            ),
        }):
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

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(
                proactive_graph_retrieve=mock_retrieve,
            ),
        }):
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
        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(
                proactive_graph_retrieve=MagicMock(side_effect=Exception("timeout")),
            ),
        }):
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

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_closes_session(self, _avail):
        plugin._session_started = True
        plugin._session_context_cache = "some cached context"
        mock_close = MagicMock()

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(auto_close_session=mock_close),
        }):
            plugin._on_session_end(
                session_id="abcdef99", completed=True, interrupted=False,
            )

        assert not plugin._session_started
        assert plugin._session_context_cache is None
        mock_close.assert_called_once()

    @patch("hermes_mindgraph_plugin._is_available", return_value=True)
    def test_close_failure_is_nonfatal(self, _avail):
        plugin._session_started = True
        mock_close = MagicMock(side_effect=RuntimeError("api down"))

        with patch.dict(sys.modules, {
            "tools.mindgraph_tool": MagicMock(auto_close_session=mock_close),
        }):
            # Should not raise
            plugin._on_session_end(session_id="test", completed=True)
