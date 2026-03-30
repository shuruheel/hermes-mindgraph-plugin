"""MindGraph semantic graph memory plugin for Hermes Agent.

Bridges MindGraph (https://mindgraph.cloud) into Hermes Agent's plugin
lifecycle hooks, giving any Hermes-powered agent persistent semantic memory
across sessions.

Install:
    pip install hermes-mindgraph-plugin

Configure:
    Set MINDGRAPH_API_KEY in ~/.hermes/.env (get one at https://mindgraph.cloud)

The plugin is auto-discovered via the ``hermes_agent.plugins`` entry point.
It can also be installed manually by copying this package to
``~/.hermes/plugins/mindgraph/`` with an accompanying ``plugin.yaml``.

What you get:
    11 MindGraph tools — session, journal, argue, commit, retrieve, ingest,
    capture, inquire, action, decide, plan — registered into the "mindgraph"
    toolset so the agent can read/write the semantic graph.

    3 lifecycle hooks — on_session_start, pre_llm_call, on_session_end —
    for automatic session management, context injection, and transcript
    ingestion.

Hooks registered:
    on_session_start  — Opens a MindGraph session, pre-fetches context.
    pre_llm_call      — Injects session context (goals, decisions, policies)
                        on first turn; score-gated semantic retrieval every turn.
    on_session_end    — Closes the MindGraph session.
"""

__version__ = "0.2.2"

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State — per-process, reset on session_start
# ---------------------------------------------------------------------------

_session_context_cache: Optional[str] = None
_session_started: bool = False
_accumulated_messages: list = []
_current_session_id: Optional[str] = None
_atexit_registered: bool = False
_is_cron_session: bool = False  # Read-only mode: context injection, no session/ingestion


def _is_available() -> bool:
    """Check if MindGraph is configured and ready."""
    try:
        from hermes_mindgraph_plugin.tools import check_requirements
        return check_requirements()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

def _on_session_start(
    *, session_id: str = "", model: str = "", platform: str = "", **kw
):
    """Open a MindGraph session and pre-fetch session context.

    If a previous session is still open (different session_id), close it
    first with the accumulated transcript for post-session ingestion.
    """
    global _session_context_cache, _session_started, _accumulated_messages, _current_session_id, _is_cron_session

    _is_cron = platform == "cron"
    _is_cron_session = _is_cron

    # Close the *previous* session if one exists with a different ID.
    # This is the real session-close point: on_session_end fires after
    # every run_conversation() call (every message), but on_session_start
    # fires only when a new session begins — so we close here.
    # (Skip for cron — cron never opens sessions.)
    if not _is_cron and _session_started and _current_session_id and _current_session_id != session_id:
        _close_mindgraph_session(_current_session_id)

    _session_context_cache = None
    _session_started = False
    _accumulated_messages = []
    _current_session_id = session_id

    if not _is_available():
        return

    # Open session — skip for cron (no session node pollution in graph)
    if not _is_cron:
        try:
            from hermes_mindgraph_plugin.tools import auto_open_session

            label = f"hermes-{session_id[:8]}" if session_id else "hermes-session"
            sid = auto_open_session(label=label)
            if sid:
                logger.info("MindGraph session opened: %s", sid)
                _session_started = True
        except Exception as exc:
            logger.debug("MindGraph session open failed (non-fatal): %s", exc)
    else:
        logger.info("MindGraph cron mode: read-only (no session open, context will be fetched)")

    # Pre-fetch session context (goals, decisions, policies, weak claims)
    # so it's ready for the first pre_llm_call without blocking.
    # Cron jobs GET this too — they need goals, policies, and recent
    # activity context to make informed decisions.
    try:
        from hermes_mindgraph_plugin.tools import retrieve_session_context

        _session_context_cache = retrieve_session_context()
    except Exception as exc:
        logger.debug("MindGraph session context prefetch failed: %s", exc)


def _pre_llm_call(
    *,
    session_id: str = "",
    user_message: str = "",
    conversation_history: list = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    **kw,
) -> dict | None:
    """Return context to inject into the agent's ephemeral system prompt.

    First turn: cached session context (goals, decisions, policies, weak claims).
    Every turn:  score-gated semantic retrieval for topic relevance.

    Returns ``{"context": "..."}`` or ``None``.
    """
    if not _is_available():
        return None

    # --- Accumulate messages for post-session ingestion (skip for cron) ---
    if conversation_history and not _is_cron_session:
        _accumulate_messages(conversation_history)

    parts: list[str] = []

    # --- Session context (first turn only) ---
    if is_first_turn and _session_context_cache:
        parts.append(_session_context_cache)
        # Don't clear the cache — the gateway creates a fresh AIAgent per
        # message, so is_first_turn may be True on every gateway turn for
        # continuing sessions.  The cache stays valid for the process
        # lifetime and is reset on the next on_session_start.

    # --- Per-turn semantic retrieval ---
    if user_message:
        try:
            from hermes_mindgraph_plugin.tools import proactive_graph_retrieve

            turn_ctx = proactive_graph_retrieve(user_message)
            if turn_ctx:
                parts.append(turn_ctx)
        except Exception as exc:
            logger.debug("MindGraph turn retrieval failed (non-fatal): %s", exc)

    if parts:
        return {"context": "\n\n".join(parts)}
    return None


def _close_mindgraph_session(session_id: str):
    """Close the MindGraph session with accumulated transcript.

    Shared by on_session_start (closing previous session) and
    on_session_end (final cleanup on process exit).
    """
    global _session_context_cache, _session_started, _accumulated_messages, _current_session_id

    if not _session_started:
        return

    try:
        from hermes_mindgraph_plugin.tools import auto_close_session

        summary = (
            f"Session {session_id[:8] if session_id else 'unknown'} (completed)"
        )

        transcript = _accumulated_messages if _accumulated_messages else None
        if transcript:
            logger.info(
                "MindGraph session close: passing %d messages for ingestion",
                len(transcript),
            )

        auto_close_session(
            summary=summary,
            transcript_messages=transcript,
        )
        logger.info("MindGraph session closed: %s", session_id[:8] if session_id else "unknown")
    except Exception as exc:
        logger.debug("MindGraph session close failed (non-fatal): %s", exc)
    finally:
        _session_started = False
        _session_context_cache = None
        _accumulated_messages = []
        _current_session_id = None


def _accumulate_messages(conversation_history: list):
    """Snapshot user/assistant messages from conversation_history.

    Called each turn with the full message list.  We keep only user and
    assistant messages (by role) and deduplicate by checking whether the
    last accumulated message matches the last message in the incoming list
    — this avoids re-storing the entire history every turn.
    """
    global _accumulated_messages

    # Find new messages: conversation_history grows each turn, so we only
    # need messages beyond what we already have.
    start = len(_accumulated_messages)

    # Filter to user/assistant only from the full history
    relevant = [
        m for m in conversation_history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]

    if len(relevant) > start:
        _accumulated_messages = relevant  # Replace with full filtered set


def _on_session_end(
    *,
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    model: str = "",
    platform: str = "",
    **kw,
):
    """Close the MindGraph session with transcript ingestion.

    In the gateway, on_session_end fires after every run_conversation()
    call (every message), NOT just at session end.  The real close happens
    in on_session_start when a *new* session begins.

    We still close here for the CLI case (single run_conversation per
    session) and for interrupted/non-completed sessions.
    """
    if not _session_started:
        return

    # In normal gateway flow, skip — the next on_session_start will close.
    # Only force-close for interruptions or explicit completion signals
    # where there may not be a subsequent on_session_start.
    if completed and not interrupted:
        # Normal completion — let the next on_session_start handle it.
        # This keeps the MindGraph session open across multiple gateway turns.
        # For CLI (single-turn), on_session_start won't fire again, so we
        # register an atexit handler to catch that case (once only).
        global _atexit_registered
        if not _atexit_registered:
            import atexit

            def _atexit_close():
                if _session_started:
                    _close_mindgraph_session(_current_session_id or session_id)

            atexit.register(_atexit_close)
            _atexit_registered = True
        return

    # Interrupted or abnormal end — close immediately
    _close_mindgraph_session(session_id)


# ---------------------------------------------------------------------------
# Plugin entry point — called by Hermes plugin discovery
# ---------------------------------------------------------------------------

def register(ctx):
    """Register MindGraph memory tools and lifecycle hooks with the Hermes plugin system."""

    # --- Register all 11 MindGraph tools ---
    try:
        from hermes_mindgraph_plugin.tools import TOOLS

        for tool in TOOLS:
            ctx.register_tool(
                name=tool["name"],
                toolset=tool["toolset"],
                schema=tool["schema"],
                handler=tool["handler"],
                check_fn=tool.get("check_fn"),
                requires_env=tool.get("requires_env"),
                emoji=tool.get("emoji", ""),
            )
        logger.info("MindGraph plugin: registered %d tools", len(TOOLS))
    except ImportError as exc:
        logger.warning("MindGraph plugin: failed to import tools — %s", exc)
    except Exception as exc:
        logger.warning("MindGraph plugin: tool registration failed — %s", exc)

    # --- Register lifecycle hooks ---
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)

    logger.info("MindGraph memory plugin registered (11 tools + 3 lifecycle hooks)")
