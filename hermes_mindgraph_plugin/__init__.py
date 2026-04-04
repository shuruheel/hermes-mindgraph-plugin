"""MindGraph semantic graph memory provider for Hermes Agent.

Implements the MemoryProvider interface to bridge MindGraph
(https://mindgraph.cloud) into Hermes Agent, giving any Hermes-powered
agent persistent semantic memory across sessions.

Install:
    pip install hermes-mindgraph-plugin

Configure:
    Set MINDGRAPH_API_KEY in ~/.hermes/.env (get one at https://mindgraph.cloud)

The plugin is auto-discovered via the ``hermes_agent.plugins`` entry point.
It can also be installed manually by copying this package to
``~/.hermes/plugins/memory/mindgraph/`` with an accompanying ``plugin.yaml``.

What you get:
    4 MindGraph tools — remember, retrieve, commit, ingest —
    registered via get_tool_schemas().

    Lifecycle hooks — initialize, system_prompt_block, prefetch, sync_turn,
    on_session_end, on_pre_compress, shutdown — for automatic session
    management, context injection, and transcript ingestion.

    Per-turn hybrid retrieval (FTS + semantic) for natural language queries.
"""

__version__ = "0.6.1"

import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import MemoryProvider ABC — graceful fallback for older Hermes versions
# ---------------------------------------------------------------------------

try:
    from agent.memory_provider import MemoryProvider as _MemoryProviderBase
except ImportError:
    # Hermes version doesn't have MemoryProvider yet — define a stub so the
    # class can still be instantiated and the old register() path still works.
    class _MemoryProviderBase:  # type: ignore[no-redef]
        pass


class MindGraphMemoryProvider(_MemoryProviderBase):
    """MindGraph semantic graph memory provider.

    Implements the Hermes MemoryProvider interface to provide:
    - Session-start context injection (goals, decisions, policies, weak claims)
    - Per-turn hybrid retrieval (FTS + semantic via /retrieve/context)
    - 4 tools: remember, retrieve, commit, ingest
    - Post-session transcript ingestion for cross-session continuity
    """

    def __init__(self):
        self._session_id: Optional[str] = None
        self._session_context_cache: Optional[str] = None
        self._session_started: bool = False
        self._is_cron_session: bool = False
        self._hermes_home: Optional[str] = None
        self._accumulated_messages: list = []
        self._ingested_up_to: int = 0  # High-water mark: messages already pre-compress ingested
        self._sync_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mindgraph"

    def is_available(self) -> bool:
        """Check if MindGraph API key is configured. NO network calls."""
        return bool(os.environ.get("MINDGRAPH_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup.

        Opens a MindGraph session (unless cron mode) and pre-fetches
        session context (goals, decisions, policies, weak claims).

        kwargs includes:
            hermes_home (str): Active HERMES_HOME path
            platform (str): "cli", "gateway", or "cron"
            model (str): Model identifier
        """
        self._hermes_home = kwargs.get("hermes_home", "")
        platform = kwargs.get("platform", "")
        self._is_cron_session = (platform == "cron")

        # Close the *previous* session if one exists with a different ID.
        if (
            not self._is_cron_session
            and self._session_started
            and self._session_id
            and self._session_id != session_id
        ):
            self._close_session()

        self._session_id = session_id
        self._session_started = False
        self._session_context_cache = None
        self._accumulated_messages = []
        self._ingested_up_to = 0

        # Open MindGraph session — skip for cron (no session node pollution)
        if not self._is_cron_session:
            try:
                from hermes_mindgraph_plugin.tools import auto_open_session

                label = f"hermes-{session_id[:8]}" if session_id else "hermes-session"
                sid = auto_open_session(label=label)
                if sid:
                    logger.info("MindGraph session opened: %s", sid)
                    self._session_started = True
            except Exception as exc:
                logger.debug("MindGraph session open failed (non-fatal): %s", exc)
        else:
            logger.info("MindGraph cron mode: read-only (no session open)")

        # Pre-fetch session context for system_prompt_block()
        try:
            from hermes_mindgraph_plugin.tools import retrieve_session_context

            self._session_context_cache = retrieve_session_context()
        except Exception as exc:
            logger.debug("MindGraph session context prefetch failed: %s", exc)

    def get_tool_schemas(self) -> list:
        """Return OpenAI function-calling schemas for all 11 MindGraph tools."""
        try:
            from hermes_mindgraph_plugin.tools import TOOLS
            return [t["schema"] for t in TOOLS]
        except ImportError:
            logger.warning("MindGraph tools module not available")
            return []

    def handle_tool_call(self, name: str, args: dict) -> str:
        """Dispatch a tool call to the appropriate MindGraph tool handler."""
        try:
            from hermes_mindgraph_plugin.tools import TOOLS
        except ImportError:
            return json.dumps({"success": False, "error": "MindGraph tools not available"})

        for t in TOOLS:
            if t["name"] == name:
                try:
                    return t["handler"](args)
                except Exception as exc:
                    return json.dumps({"success": False, "error": str(exc)})

        return json.dumps({"success": False, "error": f"Unknown tool: {name}"})

    def get_config_schema(self) -> list:
        """Declare configuration fields for setup wizard."""
        return [
            {
                "key": "api_key",
                "description": "MindGraph API key (get one at https://mindgraph.cloud)",
                "secret": True,
                "required": True,
                "env_var": "MINDGRAPH_API_KEY",
                "url": "https://mindgraph.cloud",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Save non-secret config. All MindGraph config is via env var (secret)."""
        pass

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Return static provider info + session context for the system prompt.

        Includes the behavioral contract and cached session context
        (goals, decisions, policies, weak claims) fetched during initialize().
        """
        return self._session_context_cache or ""

    def prefetch(self, query: str) -> str:
        """Per-turn proactive semantic retrieval.

        Called before each LLM API call with the user's message.
        Returns topic-relevant knowledge from the semantic graph,
        or empty string if nothing relevant found.

        Disable via MINDGRAPH_PROACTIVE_RETRIEVAL=false environment variable.
        """
        if not self.is_available():
            return ""

        try:
            from hermes_mindgraph_plugin.tools import (
                PROACTIVE_RETRIEVAL_ENABLED,
                proactive_graph_retrieve,
            )

            if not PROACTIVE_RETRIEVAL_ENABLED:
                return ""

            result = proactive_graph_retrieve(query)
            return result or ""
        except Exception as exc:
            logger.debug("MindGraph prefetch failed (non-fatal): %s", exc)
            return ""

    def queue_prefetch(self, query: str) -> None:
        """Pre-warm cache after each turn (no-op for now).

        MindGraph retrieval is fast enough that prefetch() handles
        everything synchronously. This hook is available for future
        optimization if needed.
        """
        pass

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        """Accumulate turn messages for post-session ingestion.

        Non-blocking: only appends to an in-memory list.
        The actual API call happens in on_session_end() or shutdown().
        """
        if self._is_cron_session:
            return

        if user_content:
            self._accumulated_messages.append(
                {"role": "user", "content": user_content}
            )
        if assistant_content:
            self._accumulated_messages.append(
                {"role": "assistant", "content": assistant_content}
            )

    def on_session_end(self, messages: list) -> None:
        """Final extraction/flush on conversation end.

        Closes the MindGraph session and ingests only the portion of
        the transcript that hasn't already been pre-compress ingested.
        """
        if not self._session_started:
            return

        # Use _accumulated_messages sliced from high-water mark to avoid
        # re-ingesting content already sent during pre-compression.
        # If no pre-compress happened (_ingested_up_to == 0), sends everything.
        if self._ingested_up_to > 0:
            transcript = self._accumulated_messages[self._ingested_up_to:]
        else:
            # No pre-compress — use Hermes-provided messages if available,
            # otherwise fall back to full accumulated list
            transcript = messages if messages else self._accumulated_messages
        self._close_session(transcript)

    def on_pre_compress(self, messages: list) -> None:
        """Save insights before context window compression.

        Ingests a snapshot of the conversation being compressed so
        key context is not lost when the window rolls over.

        Sets _ingested_up_to so that on_session_end() only sends
        messages beyond this point, avoiding double ingestion.
        """
        if not self._session_started or not self.is_available():
            return

        try:
            from hermes_mindgraph_plugin.tools import (
                PRE_COMPRESS_LIMIT,
                _filter_transcript_for_ingestion,
                _get_client,
            )

            filtered = _filter_transcript_for_ingestion(messages)
            if filtered and len(filtered) > 50:
                client = _get_client()
                if client:
                    limit = PRE_COMPRESS_LIMIT
                    client.ingest_chunk(
                        content=f"[pre-compression snapshot]\n{filtered[:limit]}"
                    )
                    # Advance high-water mark so session-end doesn't re-ingest
                    self._ingested_up_to = len(self._accumulated_messages)
                    logger.info(
                        "MindGraph pre-compress snapshot ingested (%d chars, limit=%d, hwm=%d)",
                        min(len(filtered), limit), limit, self._ingested_up_to,
                    )
        except Exception as exc:
            logger.debug("MindGraph pre-compress ingest failed (non-fatal): %s", exc)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes into MindGraph.

        Captures Hermes built-in memory writes (e.g., user preferences,
        facts) as journal entries in MindGraph for cross-system continuity.
        """
        if not self.is_available() or self._is_cron_session:
            return

        try:
            from hermes_mindgraph_plugin.tools import mindgraph_journal

            entry_type = "preference" if action == "preference" else "note"
            entry = f"[{action}:{target}] {content}" if target else content
            mindgraph_journal(entry, entry_type=entry_type)
        except Exception as exc:
            logger.debug("MindGraph memory write mirror failed (non-fatal): %s", exc)

    def shutdown(self) -> None:
        """Clean up on process exit.

        Closes any open MindGraph session with only the un-ingested
        portion of the accumulated transcript.
        """
        if self._session_started:
            remaining = self._accumulated_messages[self._ingested_up_to:]
            self._close_session(remaining or None)

        # Wait for any in-flight sync thread
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_session(self, messages: list = None) -> None:
        """Close the MindGraph session with optional transcript ingestion."""
        if not self._session_started:
            return

        try:
            from hermes_mindgraph_plugin.tools import auto_close_session

            summary = (
                f"Session {self._session_id[:8] if self._session_id else 'unknown'} "
                f"(completed)"
            )

            if messages:
                logger.info(
                    "MindGraph session close: passing %d messages for ingestion",
                    len(messages),
                )

            auto_close_session(
                summary=summary,
                transcript_messages=messages,
            )
            logger.info(
                "MindGraph session closed: %s",
                self._session_id[:8] if self._session_id else "unknown",
            )
        except Exception as exc:
            logger.debug("MindGraph session close failed (non-fatal): %s", exc)
        finally:
            self._session_started = False
            self._session_context_cache = None
            self._accumulated_messages = []
            self._ingested_up_to = 0


# ---------------------------------------------------------------------------
# Plugin entry point — called by Hermes plugin discovery
# ---------------------------------------------------------------------------

def register(ctx):
    """Register MindGraph as a memory provider with the Hermes plugin system.

    Called by the memory plugin discovery system. Registers a
    MindGraphMemoryProvider instance via ctx.register_memory_provider().
    """
    provider = MindGraphMemoryProvider()
    ctx.register_memory_provider(provider)
    logger.info("MindGraph memory provider registered")
