"""Tests for hermes-mindgraph-plugin (MemoryProvider interface)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_mindgraph_plugin import MindGraphMemoryProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    """Create a fresh MindGraphMemoryProvider instance for each test."""
    return MindGraphMemoryProvider()


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:

    def test_registers_memory_provider(self):
        import hermes_mindgraph_plugin as plugin
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_memory_provider.assert_called_once()
        arg = ctx.register_memory_provider.call_args[0][0]
        assert isinstance(arg, MindGraphMemoryProvider)

    def test_provider_name(self, provider):
        assert provider.name == "mindgraph"


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------

class TestIsAvailable:

    def test_available_when_key_set(self, provider):
        with patch.dict("os.environ", {"MINDGRAPH_API_KEY": "test-key"}):
            assert provider.is_available() is True

    def test_unavailable_when_key_missing(self, provider):
        with patch.dict("os.environ", {}, clear=True):
            assert provider.is_available() is False

    def test_unavailable_when_key_empty(self, provider):
        with patch.dict("os.environ", {"MINDGRAPH_API_KEY": ""}):
            assert provider.is_available() is False


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------

class TestInitialize:

    @patch("hermes_mindgraph_plugin.tools.retrieve_session_context")
    @patch("hermes_mindgraph_plugin.tools.auto_open_session")
    def test_opens_session_and_prefetches(self, mock_open, mock_ctx, provider):
        mock_open.return_value = "sess-uid-123"
        mock_ctx.return_value = "## Goals\n- Ship it"

        provider.initialize("abcdef99", platform="cli", hermes_home="/tmp")

        assert provider._session_started is True
        assert provider._session_context_cache == "## Goals\n- Ship it"
        mock_open.assert_called_once_with(label="hermes-abcdef99")
        assert provider._session_id == "abcdef99"
        assert provider._hermes_home == "/tmp"

    @patch("hermes_mindgraph_plugin.tools.retrieve_session_context")
    @patch("hermes_mindgraph_plugin.tools.auto_open_session")
    def test_cron_reads_context_but_skips_session_open(self, mock_open, mock_ctx, provider):
        mock_open.return_value = "session-uid"
        mock_ctx.return_value = "## Goals\n- Ship it"

        provider.initialize("cron-job-1", platform="cron")

        assert not provider._session_started
        mock_open.assert_not_called()
        mock_ctx.assert_called_once()
        assert provider._session_context_cache == "## Goals\n- Ship it"
        assert provider._is_cron_session is True

    @patch("hermes_mindgraph_plugin.tools.retrieve_session_context")
    @patch("hermes_mindgraph_plugin.tools.auto_open_session")
    def test_session_start_failure_is_nonfatal(self, mock_open, mock_ctx, provider):
        mock_open.side_effect = RuntimeError("connection refused")
        mock_ctx.return_value = None

        # Should not raise
        provider.initialize("test")
        assert not provider._session_started

    @patch("hermes_mindgraph_plugin.tools.retrieve_session_context")
    @patch("hermes_mindgraph_plugin.tools.auto_open_session")
    def test_closes_previous_session_on_new_id(self, mock_open, mock_ctx, provider):
        """When initialize is called with a new session_id, close the previous."""
        mock_open.return_value = "uid-1"
        mock_ctx.return_value = None

        provider.initialize("session-1", platform="cli")
        assert provider._session_started is True

        # Now re-initialize with a different session
        mock_open.return_value = "uid-2"
        with patch("hermes_mindgraph_plugin.tools.auto_close_session") as mock_close:
            provider.initialize("session-2", platform="cli")

        mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# system_prompt_block()
# ---------------------------------------------------------------------------

class TestSystemPromptBlock:

    def test_returns_cached_context(self, provider):
        provider._session_context_cache = "## Active Goals\n- Goal 1"
        result = provider.system_prompt_block()
        assert "Active Goals" in result

    def test_returns_empty_when_no_cache(self, provider):
        provider._session_context_cache = None
        result = provider.system_prompt_block()
        assert result == ""


# ---------------------------------------------------------------------------
# prefetch()
# ---------------------------------------------------------------------------

class TestPrefetch:

    @patch("hermes_mindgraph_plugin.MindGraphMemoryProvider.is_available", return_value=False)
    def test_returns_empty_when_unavailable(self, _mock, provider):
        result = provider.prefetch("hello world")
        assert result == ""

    @patch("hermes_mindgraph_plugin.MindGraphMemoryProvider.is_available", return_value=True)
    def test_returns_retrieval_context(self, _avail, provider):
        mock_retrieve = MagicMock(return_value="## Relevant: plugin hooks")
        with patch("hermes_mindgraph_plugin.tools.proactive_graph_retrieve", mock_retrieve):
            result = provider.prefetch("Tell me about the plugin lifecycle hooks")

        assert "plugin hooks" in result
        mock_retrieve.assert_called_once()

    @patch("hermes_mindgraph_plugin.MindGraphMemoryProvider.is_available", return_value=True)
    def test_returns_empty_on_no_results(self, _avail, provider):
        mock_retrieve = MagicMock(return_value=None)
        with patch("hermes_mindgraph_plugin.tools.proactive_graph_retrieve", mock_retrieve):
            result = provider.prefetch("hi")
        assert result == ""

    @patch("hermes_mindgraph_plugin.MindGraphMemoryProvider.is_available", return_value=True)
    def test_retrieval_failure_is_nonfatal(self, _avail, provider):
        with patch(
            "hermes_mindgraph_plugin.tools.proactive_graph_retrieve",
            MagicMock(side_effect=Exception("timeout")),
        ):
            result = provider.prefetch("Tell me about something complex")
        assert result == ""


# ---------------------------------------------------------------------------
# get_tool_schemas()
# ---------------------------------------------------------------------------

class TestGetToolSchemas:

    def test_returns_eleven_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 11

    def test_all_schemas_have_name_and_parameters(self, provider):
        for schema in provider.get_tool_schemas():
            assert "name" in schema
            assert "parameters" in schema

    def test_schema_names_match_expected(self, provider):
        names = {s["name"] for s in provider.get_tool_schemas()}
        expected = {
            "mindgraph_session", "mindgraph_journal", "mindgraph_argue",
            "mindgraph_commit", "mindgraph_retrieve", "mindgraph_ingest",
            "mindgraph_capture", "mindgraph_inquire", "mindgraph_action",
            "mindgraph_decide", "mindgraph_plan",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# handle_tool_call()
# ---------------------------------------------------------------------------

class TestHandleToolCall:

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_dispatches_journal(self, mock_get, provider):
        client = MagicMock()
        client.journal.return_value = {"uid": "j1", "label": "test entry"}
        mock_get.return_value = client

        result = json.loads(provider.handle_tool_call(
            "mindgraph_journal", {"entry": "test note", "entry_type": "note"},
        ))
        assert result["success"] is True
        client.journal.assert_called_once()

    def test_unknown_tool_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("nonexistent_tool", {}))
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_dispatches_retrieve(self, mock_get, provider):
        client = MagicMock()
        # _fts_search prefers client.search() — mock enriched FTS response
        client.search.return_value = {
            "results": [{"uid": "n1", "label": "test", "node_type": "Concept"}],
            "edges": [],
            "chunks": [],
        }
        mock_get.return_value = client

        result = json.loads(provider.handle_tool_call(
            "mindgraph_retrieve", {"query": "test", "mode": "search"},
        ))
        assert result["success"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["results"][0]["uid"] == "n1"
        # Verify search was called (not hybrid_search or _request)
        client.search.assert_called_once()


# ---------------------------------------------------------------------------
# get_config_schema()
# ---------------------------------------------------------------------------

class TestGetConfigSchema:

    def test_returns_api_key_config(self, provider):
        schema = provider.get_config_schema()
        assert len(schema) == 1
        assert schema[0]["key"] == "api_key"
        assert schema[0]["secret"] is True
        assert schema[0]["env_var"] == "MINDGRAPH_API_KEY"
        assert schema[0]["required"] is True


# ---------------------------------------------------------------------------
# sync_turn()
# ---------------------------------------------------------------------------

class TestSyncTurn:

    def test_accumulates_messages(self, provider):
        provider.sync_turn("Hello", "Hi there!")
        assert len(provider._accumulated_messages) == 2
        assert provider._accumulated_messages[0] == {"role": "user", "content": "Hello"}
        assert provider._accumulated_messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_grows_incrementally(self, provider):
        provider.sync_turn("Hello", "Hi")
        provider.sync_turn("Thanks", "Welcome")
        assert len(provider._accumulated_messages) == 4

    def test_skips_empty_content(self, provider):
        provider.sync_turn("", "response")
        assert len(provider._accumulated_messages) == 1
        assert provider._accumulated_messages[0]["role"] == "assistant"

    def test_noop_in_cron_mode(self, provider):
        provider._is_cron_session = True
        provider.sync_turn("Hello", "Hi")
        assert len(provider._accumulated_messages) == 0


# ---------------------------------------------------------------------------
# on_session_end()
# ---------------------------------------------------------------------------

class TestOnSessionEnd:

    def test_noop_when_not_started(self, provider):
        provider._session_started = False
        # Should not raise or call anything
        provider.on_session_end([])
        assert not provider._session_started

    def test_closes_with_provided_messages(self, provider):
        provider._session_started = True
        provider._session_id = "abcdef99"
        mock_close = MagicMock()

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
            provider.on_session_end(messages)

        assert not provider._session_started
        mock_close.assert_called_once()
        call_kwargs = mock_close.call_args[1]
        assert call_kwargs["transcript_messages"] == messages

    def test_falls_back_to_accumulated(self, provider):
        provider._session_started = True
        provider._session_id = "test1234"
        provider._accumulated_messages = [{"role": "user", "content": "hi"}]
        mock_close = MagicMock()

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            provider.on_session_end([])  # empty messages

        mock_close.assert_called_once()
        call_kwargs = mock_close.call_args[1]
        assert call_kwargs["transcript_messages"] == [{"role": "user", "content": "hi"}]

    def test_close_failure_is_nonfatal(self, provider):
        provider._session_started = True
        provider._session_id = "test"
        mock_close = MagicMock(side_effect=RuntimeError("api down"))

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            # Should not raise
            provider.on_session_end([{"role": "user", "content": "hi"}])

        # State should be cleaned up even on failure
        assert not provider._session_started


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_closes_open_session(self, provider):
        provider._session_started = True
        provider._session_id = "test"
        mock_close = MagicMock()

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            provider.shutdown()

        mock_close.assert_called_once()
        assert not provider._session_started

    def test_noop_when_not_started(self, provider):
        provider._session_started = False
        # Should not raise
        provider.shutdown()
        assert not provider._session_started

    def test_uses_accumulated_messages(self, provider):
        provider._session_started = True
        provider._session_id = "test"
        provider._accumulated_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        mock_close = MagicMock()

        with patch("hermes_mindgraph_plugin.tools.auto_close_session", mock_close):
            provider.shutdown()

        call_kwargs = mock_close.call_args[1]
        assert len(call_kwargs["transcript_messages"]) == 2


# ---------------------------------------------------------------------------
# on_memory_write()
# ---------------------------------------------------------------------------

class TestOnMemoryWrite:

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_mirrors_preference_write(self, mock_get, provider):
        client = MagicMock()
        client.journal.return_value = {"uid": "j1"}
        mock_get.return_value = client

        with patch.dict("os.environ", {"MINDGRAPH_API_KEY": "test-key"}):
            provider.on_memory_write("preference", "theme", "dark mode preferred")

        client.journal.assert_called_once()

    def test_noop_when_unavailable(self, provider):
        with patch.dict("os.environ", {}, clear=True):
            # Should not raise
            provider.on_memory_write("preference", "theme", "dark mode")

    def test_noop_in_cron_mode(self, provider):
        provider._is_cron_session = True
        with patch.dict("os.environ", {"MINDGRAPH_API_KEY": "test-key"}):
            # Should not raise or call anything
            provider.on_memory_write("preference", "theme", "dark mode")


# ---------------------------------------------------------------------------
# Capture typed routing (preserved from original tests)
# ---------------------------------------------------------------------------

class TestCaptureTypedRouting:
    """Test that mindgraph_capture routes to typed SDK methods."""

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_person_routes_to_find_or_create_person(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
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
        client = MagicMock()
        client.find_or_create_organization.return_value = {"uid": "o1", "label": "Acme"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Acme", capture_type="entity", entity_type="organization"))
        assert result["success"] is True
        client.find_or_create_organization.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_nation_routes_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        client = MagicMock()
        client.find_or_create_nation.return_value = {"uid": "n1", "label": "Japan"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Japan", capture_type="entity", entity_type="nation"))
        assert result["success"] is True
        client.find_or_create_nation.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_concept_routes_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        client = MagicMock()
        client.find_or_create_concept.return_value = {"uid": "c1", "label": "Entropy"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Entropy", capture_type="entity", entity_type="concept"))
        assert result["success"] is True
        client.find_or_create_concept.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_work_falls_back_to_generic(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        client = MagicMock()
        client.find_or_create_entity.return_value = {"uid": "w1", "label": "Hamlet"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture("Hamlet", capture_type="entity", entity_type="work"))
        assert result["success"] is True
        client.find_or_create_entity.assert_called_once_with("Hamlet", props={"entity_type": "work"})

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_properties_passed_to_typed_method(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
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


# ---------------------------------------------------------------------------
# Observation-entity linking
# ---------------------------------------------------------------------------

class TestObservationEntityLinking:
    """Test that observations can be linked to entities via entity_uid."""

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_observation_creates_edge_when_entity_uid_provided(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        client = MagicMock()
        client.capture.return_value = {"uid": "obs-1", "label": "Alice is a researcher"}
        client.add_edge.return_value = {"uid": "edge-1"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture(
            "Alice is a researcher",
            capture_type="observation",
            entity_uid="person-alice-uid",
        ))
        assert result["success"] is True
        assert "linked to" in result["data"]["message"]
        client.add_edge.assert_called_once_with(
            source_uid="person-alice-uid",
            target_uid="obs-1",
            edge_type="HAS_OBSERVATION",
        )

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_observation_no_edge_without_entity_uid(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        client = MagicMock()
        client.capture.return_value = {"uid": "obs-2", "label": "test obs"}
        mock_get.return_value = client

        result = json.loads(mindgraph_capture(
            "test obs", capture_type="observation",
        ))
        assert result["success"] is True
        client.add_edge.assert_not_called()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_observation_link_failure_is_nonfatal(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_capture
        client = MagicMock()
        client.capture.return_value = {"uid": "obs-3", "label": "test"}
        client.add_edge.side_effect = Exception("edge creation failed")
        mock_get.return_value = client

        result = json.loads(mindgraph_capture(
            "test", capture_type="observation", entity_uid="entity-1",
        ))
        # Should succeed even though edge creation failed
        assert result["success"] is True
        assert "linked to" not in result["data"]["message"]


# ---------------------------------------------------------------------------
# Commit dedup + update
# ---------------------------------------------------------------------------

class TestCommitDedup:
    """Test that mindgraph_commit deduplicates and supports updates."""

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_dedup_finds_existing_goal_by_label(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = [
            {"uid": "goal-1", "label": "Ship v2.0", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(label="Ship v2.0", commit_type="goal"))
        assert result["success"] is True
        assert result["data"]["deduplicated"] is True
        assert result["data"]["uid"] == "goal-1"
        # Should NOT have called commit() to create
        client.commit.assert_not_called()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_dedup_updates_status_on_existing(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = [
            {"uid": "goal-1", "label": "Ship v2.0", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(
            label="Ship v2.0", commit_type="goal", status="completed",
        ))
        assert result["success"] is True
        assert result["data"]["deduplicated"] is True
        # Should have called update_node to change status
        client.update_node.assert_called_once_with("goal-1", status="completed")

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_dedup_is_case_insensitive(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = [
            {"uid": "goal-2", "label": "ship v2.0", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(label="Ship V2.0"))
        assert result["success"] is True
        assert result["data"]["deduplicated"] is True

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_creates_new_when_no_match(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = []  # No matches
        client.commit.return_value = {"uid": "new-1", "label": "New Goal"}
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(label="New Goal"))
        assert result["success"] is True
        assert "deduplicated" not in result.get("data", {})
        client.commit.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_update_by_uid(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.update_node.return_value = {"uid": "goal-1", "label": "Ship v2.0"}
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(
            uid="goal-1", status="completed", description="Done!",
        ))
        assert result["success"] is True
        client.update_node.assert_called_once_with(
            "goal-1", status="completed", description="Done!",
        )
        # Should NOT search or create
        client.search.assert_not_called()
        client.commit.assert_not_called()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_dedup_search_failure_falls_through_to_create(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.side_effect = Exception("search failed")
        client.commit.return_value = {"uid": "new-2", "label": "My Goal"}
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(label="My Goal"))
        assert result["success"] is True
        client.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Fuzzy dedup
# ---------------------------------------------------------------------------

class TestFuzzyDedup:
    """Test fuzzy label matching in mindgraph_commit dedup."""

    def test_normalize_label(self):
        from hermes_mindgraph_plugin.tools import _normalize_label
        assert _normalize_label("Ship v2.0") == "ship version 2 0"
        assert _normalize_label("Ship version 2.0") == "ship version 2 0"
        assert _normalize_label("  Hello   World  ") == "hello world"
        assert _normalize_label("ver.3 release") == "version 3 release"

    def test_label_similarity_identical(self):
        from hermes_mindgraph_plugin.tools import _label_similarity
        assert _label_similarity("Ship v2", "Ship v2") == 1.0

    def test_label_similarity_abbreviation(self):
        from hermes_mindgraph_plugin.tools import _label_similarity
        score = _label_similarity("Ship v2", "Ship version 2")
        assert score > 0.85  # Should be a strong match after normalization

    def test_label_similarity_dissimilar(self):
        from hermes_mindgraph_plugin.tools import _label_similarity
        score = _label_similarity("Ship v2", "Hire new engineer")
        assert score < 0.5

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_fuzzy_dedup_catches_abbreviation(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        # Existing node uses full form
        client.search.return_value = [
            {"uid": "goal-1", "label": "Ship version 2.0", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        # Commit uses abbreviated form
        result = json.loads(mindgraph_commit(label="Ship v2.0", commit_type="goal"))
        assert result["success"] is True
        assert result["data"]["deduplicated"] is True
        assert result["data"]["uid"] == "goal-1"
        assert "fuzzy" in result["data"]["message"]
        client.commit.assert_not_called()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_fuzzy_dedup_rejects_low_similarity(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = [
            {"uid": "goal-1", "label": "Ship alpha release", "props": {"status": "active"}},
        ]
        client.commit.return_value = {"uid": "new-1", "label": "Ship v2.0"}
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(label="Ship v2.0", commit_type="goal"))
        assert result["success"] is True
        # Should create new — not dedup against a dissimilar label
        assert "deduplicated" not in result.get("data", {})
        client.commit.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_fuzzy_prefers_exact_over_fuzzy(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        # Both an exact match and a fuzzy match exist
        client.search.return_value = [
            {"uid": "goal-fuzzy", "label": "Ship version 2.0", "props": {"status": "active"}},
            {"uid": "goal-exact", "label": "Ship v2.0", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(label="Ship v2.0", commit_type="goal"))
        assert result["success"] is True
        assert result["data"]["deduplicated"] is True
        # Should pick the exact match
        assert result["data"]["uid"] == "goal-exact"
        assert "exact" in result["data"]["message"]

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_fuzzy_disabled_at_threshold_1(self, mock_get):
        """Setting threshold to 1.0 effectively disables fuzzy matching."""
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = [
            {"uid": "goal-1", "label": "Ship version 2.0", "props": {"status": "active"}},
        ]
        client.commit.return_value = {"uid": "new-1", "label": "Ship v2.0"}
        mock_get.return_value = client

        with patch("hermes_mindgraph_plugin.tools.DEDUP_FUZZY_THRESHOLD", 1.0):
            result = json.loads(mindgraph_commit(label="Ship v2.0", commit_type="goal"))

        # No exact match and fuzzy disabled → create new
        assert result["success"] is True
        assert "deduplicated" not in result.get("data", {})
        client.commit.assert_called_once()

    @patch("hermes_mindgraph_plugin.tools._get_client")
    def test_fuzzy_dedup_updates_status(self, mock_get):
        from hermes_mindgraph_plugin.tools import mindgraph_commit
        client = MagicMock()
        client.search.return_value = [
            {"uid": "goal-1", "label": "Ship version 2.0", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        result = json.loads(mindgraph_commit(
            label="Ship v2.0", commit_type="goal", status="completed",
        ))
        assert result["success"] is True
        assert result["data"]["deduplicated"] is True
        client.update_node.assert_called_once_with("goal-1", status="completed")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Test configurable settings via environment variables."""

    def test_default_config_values(self):
        import hermes_mindgraph_plugin.tools as tools
        # These should have sensible defaults
        assert tools.PRE_COMPRESS_LIMIT == 4000
        assert tools.PROACTIVE_K == 5
        assert tools.PROACTIVE_RETRIEVAL_ENABLED is True

    def test_env_helpers(self):
        from hermes_mindgraph_plugin.tools import _env_float, _env_int, _env_bool
        with patch.dict("os.environ", {"TEST_FLOAT": "0.8"}):
            assert _env_float("TEST_FLOAT", 0.5) == 0.8
        with patch.dict("os.environ", {"TEST_INT": "20"}):
            assert _env_int("TEST_INT", 5) == 20
        with patch.dict("os.environ", {"TEST_BOOL": "false"}):
            assert _env_bool("TEST_BOOL", True) is False

    def test_env_helpers_invalid_values(self):
        from hermes_mindgraph_plugin.tools import _env_float, _env_int
        with patch.dict("os.environ", {"TEST_BAD": "notanumber"}):
            assert _env_float("TEST_BAD", 0.5) == 0.5
            assert _env_int("TEST_BAD", 10) == 10

    def test_commit_schema_has_uid_param(self):
        from hermes_mindgraph_plugin.tools import MINDGRAPH_COMMIT_SCHEMA
        props = MINDGRAPH_COMMIT_SCHEMA["parameters"]["properties"]
        assert "uid" in props
        assert props["uid"]["type"] == "string"

    def test_capture_schema_has_entity_uid_param(self):
        from hermes_mindgraph_plugin.tools import MINDGRAPH_CAPTURE_SCHEMA
        props = MINDGRAPH_CAPTURE_SCHEMA["parameters"]["properties"]
        assert "entity_uid" in props
        assert props["entity_uid"]["type"] == "string"


# ---------------------------------------------------------------------------
# Structured project/task retrieval
# ---------------------------------------------------------------------------

class TestStructuredRetrieval:
    """Test that session context uses get_nodes instead of search for projects/tasks."""

    @patch("hermes_mindgraph_plugin.tools._get_client")
    @patch("hermes_mindgraph_plugin.tools.check_requirements", return_value=True)
    def test_projects_use_get_nodes(self, _req, mock_get):
        from hermes_mindgraph_plugin.tools import retrieve_session_context
        client = MagicMock()
        client.get_goals.return_value = []
        client.get_open_decisions.return_value = []
        client.get_open_questions.return_value = []
        client.get_weak_claims.return_value = []
        client.search.return_value = []  # policies still use search
        client.get_nodes.return_value = [
            {"label": "Project Alpha", "props": {"status": "active"}},
        ]
        mock_get.return_value = client

        result = retrieve_session_context()
        assert result is not None
        # get_nodes should have been called for projects and tasks
        calls = client.get_nodes.call_args_list
        node_types = [c.kwargs.get("node_type") or c[1].get("node_type", "") for c in calls]
        assert "Project" in node_types
        assert "Task" in node_types
        assert "Project Alpha" in result


# ---------------------------------------------------------------------------
# Prefetch toggle
# ---------------------------------------------------------------------------

class TestPrefetchToggle:
    """Test that proactive retrieval can be disabled."""

    @patch("hermes_mindgraph_plugin.MindGraphMemoryProvider.is_available", return_value=True)
    def test_prefetch_disabled_returns_empty(self, _avail, provider):
        with patch("hermes_mindgraph_plugin.tools.PROACTIVE_RETRIEVAL_ENABLED", False):
            result = provider.prefetch("Tell me about something")
        assert result == ""


# ---------------------------------------------------------------------------
# Tools module internals
# ---------------------------------------------------------------------------

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
