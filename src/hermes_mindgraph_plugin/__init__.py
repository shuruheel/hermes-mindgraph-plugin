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

Hooks registered:
    on_session_start  — Opens a MindGraph session, pre-fetches context.
    pre_llm_call      — Injects session context (goals, decisions, policies)
                        on first turn; score-gated semantic retrieval every turn.
    on_session_end    — Closes the MindGraph session.
"""

__version__ = "0.1.0"

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State — per-process, reset on session_start
# ---------------------------------------------------------------------------

_session_context_cache: Optional[str] = None
_session_started: bool = False


def _is_available() -> bool:
    """Check if MindGraph is configured and ready."""
    try:
        from tools.mindgraph_tool import check_requirements
        return check_requirements()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

def _on_session_start(
    *, session_id: str = "", model: str = "", platform: str = "", **kw
):
    """Open a MindGraph session and pre-fetch session context."""
    global _session_context_cache, _session_started
    _session_context_cache = None
    _session_started = False

    if not _is_available():
        return

    # Open session
    try:
        from tools.mindgraph_tool import auto_open_session

        label = f"hermes-{session_id[:8]}" if session_id else "hermes-session"
        sid = auto_open_session(label=label)
        if sid:
            logger.info("MindGraph session opened: %s", sid)
            _session_started = True
    except Exception as exc:
        logger.debug("MindGraph session open failed (non-fatal): %s", exc)

    # Pre-fetch session context (goals, decisions, policies, weak claims)
    # so it's ready for the first pre_llm_call without blocking.
    try:
        from tools.mindgraph_tool import retrieve_session_context

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
            from tools.mindgraph_tool import proactive_graph_retrieve

            turn_ctx = proactive_graph_retrieve(user_message)
            if turn_ctx:
                parts.append(turn_ctx)
        except Exception as exc:
            logger.debug("MindGraph turn retrieval failed (non-fatal): %s", exc)

    if parts:
        return {"context": "\n\n".join(parts)}
    return None


def _on_session_end(
    *,
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    model: str = "",
    platform: str = "",
    **kw,
):
    """Close the MindGraph session."""
    global _session_context_cache, _session_started

    if not _session_started:
        return

    try:
        from tools.mindgraph_tool import auto_close_session

        auto_close_session(
            summary=(
                f"Session {session_id[:8] if session_id else 'unknown'} "
                f"({'completed' if completed else 'interrupted' if interrupted else 'ended'})"
            ),
        )
        _session_started = False
        _session_context_cache = None
    except Exception as exc:
        logger.debug("MindGraph session close failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Plugin entry point — called by Hermes plugin discovery
# ---------------------------------------------------------------------------

def register(ctx):
    """Register MindGraph memory hooks with the Hermes plugin system."""
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("MindGraph memory plugin registered (3 lifecycle hooks)")
