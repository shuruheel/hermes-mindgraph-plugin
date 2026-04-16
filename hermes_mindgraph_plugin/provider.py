"""MindGraphMemoryProvider — MemoryProvider adapter for Hermes Agent.

Implements the MemoryProvider ABC defined in ``agent/memory_provider.py`` by
delegating to the existing MindGraph integration code in ``tools.py``.

Lifecycle mapping:

    initialize(session_id)       → auto_open_session(label="hermes-<id>")
    system_prompt_block()        → retrieve_session_context() + behavioral contract
    queue_prefetch(query)        → background proactive_graph_retrieve()
    prefetch(query)              → return cached prefetch result
    sync_turn(user, assistant)   → buffer turn for on_session_end ingestion
    get_tool_schemas()           → 5 MindGraph tool schemas
    handle_tool_call(name, args) → dispatch to mindgraph_* handlers
    on_session_end(messages)     → auto_close_session(..., transcript=messages)
    shutdown()                   → flush + auto_close_session()
    get_config_schema()          → fields for ``hermes memory setup``

The heavy lifting (SDK calls, dedup, retrieval formatting, metrics) lives in
``tools.py`` — this file is a thin adapter and stays that way.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List

try:
    from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover — only hit outside a Hermes runtime
    # Allows imports during local tooling (mypy, partial pytest runs) without
    # Hermes on the path. Hermes will always provide the real base class when
    # it loads this plugin.
    from abc import ABC

    class MemoryProvider(ABC):  # type: ignore[no-redef]
        pass

from . import tools as _tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Behavioral contract header — injected into system_prompt_block()
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_HEADER = (
    "# MindGraph — Semantic Graph Memory\n\n"
    "You have a persistent semantic graph that stores structured knowledge "
    "across sessions. MindGraph is not a tool you use when asked — it is how "
    "you think. Every conversation produces knowledge. Your job is to route "
    "that knowledge to the right place in the graph so your future selves "
    "can act on it.\n\n"
    "Tools: **mindgraph_remember** (store knowledge), **mindgraph_retrieve** "
    "(query knowledge), **mindgraph_commit** (track goals/decisions/plans/"
    "risks/questions), **mindgraph_ingest** (bulk content), "
    "**mindgraph_synthesize** (project-scoped cross-document synthesis).\n\n"
    "## Persist triggers — store without being asked\n"
    "- **Person discussed substantively** → remember(entity, person) + at "
    "least one observation. A Person node without observations is a failure "
    "state.\n"
    "- **Organization mentioned with context** → remember(entity, organization) "
    "+ observation.\n"
    "- **User states a preference or corrects you** → remember(preference).\n"
    "- **Analytical conclusion reached** → remember(claim) with evidence and "
    "calibrated confidence. If you formed a view, the graph needs it.\n"
    "- **User expresses intent** → commit(goal/project/milestone). User's own "
    "intent only — never commit content-derived goals.\n"
    "- **Decision point identified** → commit(open_decision) + add options.\n"
    "- **Risk or opportunity surfaced** → commit(assess_risk/add_affordance).\n"
    "- **Open question worth tracking** → commit(question).\n\n"
    "## Retrieve before acting\n"
    "Before drafting communication about a person, making recommendations, "
    "or responding to 'remember when…' — call mindgraph_retrieve first. "
    "Draft from retrieved context, not session memory. Session memory dies; "
    "the graph persists.\n\n"
    "## Judgment\n"
    "- Entity nodes = stable identity. Observations = everything you learn "
    "ABOUT them. Use descriptive keywords in observation labels for "
    "findability.\n"
    "- Claims should be falsifiable. Confidence 0.7 is a good default; "
    "don't inflate without strong evidence.\n"
    "- Topic-relevant context is auto-injected each turn — use retrieve for "
    "deeper or specific queries."
)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch_tool_call(tool_name: str, args: Dict[str, Any]) -> str:
    """Route a tool call to the right mindgraph_* handler. Returns JSON string."""
    if tool_name == "mindgraph_remember":
        return _tools.mindgraph_remember(
            label=args.get("label", ""),
            action=args.get("action", "note"),
            entity_type=args.get("entity_type", "concept"),
            properties=args.get("properties"),
            entity_uid=args.get("entity_uid", ""),
            evidence=args.get("evidence", ""),
            warrant=args.get("warrant", ""),
            confidence=args.get("confidence", 0.7),
        )
    if tool_name == "mindgraph_retrieve":
        return _tools.mindgraph_retrieve(
            query=args.get("query", ""),
            mode=args.get("mode", "context"),
            limit=args.get("limit", 27),
            include_chunks=args.get("include_chunks", False),
            include_graph=args.get("include_graph", True),
            node_type=args.get("node_type", ""),
        )
    if tool_name == "mindgraph_commit":
        return _tools.mindgraph_commit(
            action=args.get("action", "goal"),
            label=args.get("label", ""),
            uid=args.get("uid", ""),
            status=args.get("status", ""),
            description=args.get("description", ""),
            summary=args.get("summary", ""),
            option_label=args.get("option_label", ""),
            chosen_option_uid=args.get("chosen_option_uid", ""),
            plan_uid=args.get("plan_uid", ""),
            task_uid=args.get("task_uid", ""),
            execution_uid=args.get("execution_uid", ""),
        )
    if tool_name == "mindgraph_ingest":
        return _tools.mindgraph_ingest(
            content=args.get("content", ""),
            source=args.get("source", ""),
            content_type=args.get("content_type", "text"),
        )
    if tool_name == "mindgraph_synthesize":
        return _tools.mindgraph_synthesize(
            action=args.get("action", "signals"),
            project_uid=args.get("project_uid", ""),
            signals=args.get("signals", ""),
            target_types=args.get("target_types", ""),
            job_id=args.get("job_id", ""),
        )
    return json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"})


# ---------------------------------------------------------------------------
# MindGraphMemoryProvider
# ---------------------------------------------------------------------------

class MindGraphMemoryProvider(MemoryProvider):
    """MindGraph semantic graph memory for Hermes Agent.

    Provides:
      - Session lifecycle: auto-open on first turn, auto-close + transcript
        ingestion on session end.
      - System prompt context: goals, policies, open questions, decisions,
        weak claims (baked in once per system prompt build).
      - Per-turn prefetch: hybrid FTS + semantic retrieval fired in a
        background thread at turn-end, consumed at next turn-start.
      - 5 tools: mindgraph_remember, _retrieve, _commit, _ingest, _synthesize.
    """

    def __init__(self):
        self._session_id: str = ""
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        # Hermes-provided metadata from initialize() kwargs.
        self._user_id: str = ""
        self._agent_context: str = "primary"

    # ── Identity ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "mindgraph"

    def is_available(self) -> bool:
        """True when MINDGRAPH_API_KEY is set and the SDK is importable."""
        if not _tools.check_requirements():
            return False
        try:
            import mindgraph  # noqa: F401
        except ImportError:
            return False
        return True

    # ── Config schema (drives `hermes memory setup`) ───────────────────────

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "MindGraph API key",
                "secret": True,
                "required": True,
                "env_var": "MINDGRAPH_API_KEY",
                "url": "https://mindgraph.cloud",
            },
            {
                "key": "base_url",
                "description": "API endpoint (override for self-hosted deployments)",
                "default": "https://api.mindgraph.cloud",
                "env_var": "MINDGRAPH_BASE_URL",
            },
            {
                "key": "agent_id",
                "description": "Agent identifier stamped onto every write for provenance",
                "default": "hermes",
                "env_var": "MINDGRAPH_AGENT_ID",
            },
        ]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Open a MindGraph session for this conversation.

        ``auto_open_session`` is idempotent via ``_active_session_uid`` — safe
        to call on every AIAgent init, including per-message gateway restarts.
        """
        self._session_id = session_id or ""
        self._user_id = str(kwargs.get("user_id") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary")

        # Skip session open for non-primary agent contexts (subagents, cron)
        # to avoid polluting the graph with internal orchestration sessions.
        if self._agent_context != "primary":
            return

        label = f"hermes-{session_id[:8]}" if session_id else "hermes-session"
        try:
            _tools.auto_open_session(label=label)
        except Exception as e:
            logger.debug("MindGraph session open failed (non-fatal): %s", e)

    def system_prompt_block(self) -> str:
        """Behavioral contract + current graph state (goals, policies, etc).

        The contract is static; the current context (active goals, open
        decisions, weak claims) is pulled live from the graph so the
        system prompt reflects the user's current agenda.
        """
        parts = [_SYSTEM_PROMPT_HEADER]
        try:
            ctx = _tools.retrieve_session_context()
            if ctx:
                # retrieve_session_context returns its own markdown header +
                # context block. We strip the duplicate preamble and keep
                # only the "## Current Context" section.
                marker = "## Current Context"
                if marker in ctx:
                    parts.append("## Current Context\n\n" + ctx.split(marker, 1)[1].lstrip())
                else:
                    parts.append(ctx)
        except Exception as e:
            logger.debug("MindGraph session context retrieval failed: %s", e)
        return "\n\n".join(parts)

    # ── Prefetch (background per-turn retrieval) ───────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached result from the previous queue_prefetch() call.

        Wait briefly for an in-flight prefetch so we don't miss context on
        the first turn (when no prior queue_prefetch has fired yet).
        """
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result or ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire proactive retrieval in a background thread for the next turn."""
        if not query or not query.strip():
            return

        def _run():
            try:
                ctx = _tools.proactive_graph_retrieve(query)
            except Exception as e:
                logger.debug("MindGraph prefetch failed (non-fatal): %s", e)
                return
            if ctx:
                with self._prefetch_lock:
                    self._prefetch_result = ctx

        # Wait for any still-running prefetch so threads don't accumulate.
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mindgraph-prefetch"
        )
        self._prefetch_thread.start()

    # ── Turn sync ──────────────────────────────────────────────────────────

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Per-turn persistence is handled by on_session_end via transcript.

        MindGraph's 5-layer extraction pipeline produces better results on a
        full transcript than on turn-by-turn inserts, so we defer persistence
        to session end. Explicit ``mindgraph_remember`` / ``mindgraph_commit``
        tool calls during the conversation still capture turn-level facts
        immediately.
        """

    # ── Tools ──────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            _tools.MINDGRAPH_REMEMBER_SCHEMA,
            _tools.MINDGRAPH_RETRIEVE_SCHEMA,
            _tools.MINDGRAPH_COMMIT_SCHEMA,
            _tools.MINDGRAPH_INGEST_SCHEMA,
            _tools.MINDGRAPH_SYNTHESIZE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        return _dispatch_tool_call(tool_name, args)

    # ── Shutdown + session end ─────────────────────────────────────────────

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Close the MindGraph session and ingest the full transcript.

        ``auto_close_session`` runs ``ingest_session`` when ``transcript_messages``
        is supplied, triggering 5-layer extraction (Reality, Epistemic, Intent,
        Action, Memory) on the full conversation. This is the primary durable-
        persistence path — individual turn syncs are deferred to this point.
        """
        if self._agent_context != "primary":
            return
        summary = ""
        if self._session_id:
            summary = f"Hermes session {self._session_id[:8]}"
        try:
            _tools.auto_close_session(
                summary=summary,
                transcript_messages=messages,
            )
        except Exception as e:
            logger.debug("MindGraph session close failed (non-fatal): %s", e)

    def shutdown(self) -> None:
        """Drain the prefetch thread and close any still-open session."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        try:
            _tools.auto_close_session()
        except Exception as e:
            logger.debug("MindGraph shutdown close failed (non-fatal): %s", e)
