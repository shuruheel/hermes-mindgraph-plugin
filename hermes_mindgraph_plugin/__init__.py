"""MindGraph semantic graph memory plugin for Hermes Agent.

Registers 5 tools that give any Hermes-powered agent persistent, structured
memory via the MindGraph knowledge graph (https://mindgraph.cloud):

    mindgraph_remember      store knowledge (entities, claims, notes, preferences)
    mindgraph_retrieve      query the graph (hybrid FTS + semantic, traversal)
    mindgraph_commit        track goals, decisions, plans, tasks, risks, questions
    mindgraph_ingest        bulk long-form content ingestion
    mindgraph_synthesize    project-scoped cross-document synthesis

Also bundles a SKILL.md file that teaches the model when and how to use the
tools — installed to ~/.hermes/skills/mindgraph/ on first load.

Install:

    pip install hermes-mindgraph-plugin

Configure ``MINDGRAPH_API_KEY`` in ``~/.hermes/.env`` (get a key at
https://mindgraph.cloud). The plugin is auto-discovered via the
``hermes_agent.plugins`` entry point.
"""

from __future__ import annotations

__version__ = "0.9.0"

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _install_skill() -> None:
    """Copy ``skill.md`` to ``~/.hermes/skills/mindgraph/SKILL.md`` on first load.

    Hermes discovers skills from that directory at startup, so the agent
    gets our behavioral contract automatically. Existing user edits are
    preserved.
    """
    source = Path(__file__).parent / "skill.md"
    if not source.exists():
        return

    try:
        from hermes_cli.config import get_hermes_home  # type: ignore[import-not-found]

        dest_dir = Path(get_hermes_home()) / "skills" / "mindgraph"
    except Exception:
        dest_dir = Path.home() / ".hermes" / "skills" / "mindgraph"

    dest = dest_dir / "SKILL.md"
    if dest.exists():
        return  # Don't overwrite user edits.

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        logger.debug("Installed MindGraph skill to %s", dest)
    except Exception as exc:
        logger.warning("Could not install MindGraph skill: %s", exc)


def _on_session_start(session_id: str = "", platform: str = "", **kwargs: Any) -> None:
    """Open a MindGraph session when a Hermes conversation begins.

    No-op if MindGraph isn't configured. Safe to re-enter — ``auto_open_session``
    is idempotent across calls within a single process.
    """
    try:
        from hermes_mindgraph_plugin.tools import auto_open_session

        label = f"hermes-{session_id[:8]}" if session_id else "hermes-session"
        auto_open_session(label=label)
    except Exception as exc:
        logger.debug("MindGraph on_session_start failed (non-fatal): %s", exc)


def _on_session_end(session_id: str = "", platform: str = "", **kwargs: Any) -> None:
    """Close the active MindGraph session when a Hermes conversation ends."""
    try:
        from hermes_mindgraph_plugin.tools import auto_close_session

        summary = f"Hermes session {session_id[:8]}" if session_id else ""
        auto_close_session(summary=summary)
    except Exception as exc:
        logger.debug("MindGraph on_session_end failed (non-fatal): %s", exc)


def register(ctx) -> None:
    """Entry point — called by Hermes plugin discovery exactly once at startup.

    Registers all 5 MindGraph tools plus optional session-lifecycle hooks,
    and installs the bundled skill file so the agent sees usage guidance.
    """
    from hermes_mindgraph_plugin.tools import TOOLS

    for tool in TOOLS:
        ctx.register_tool(
            name=tool["name"],
            toolset=tool.get("toolset", "mindgraph"),
            schema=tool["schema"],
            handler=tool["handler"],
            check_fn=tool.get("check_fn"),
            requires_env=tool.get("requires_env"),
            emoji=tool.get("emoji", ""),
        )

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)

    _install_skill()

    logger.info(
        "MindGraph plugin v%s registered (%d tools)", __version__, len(TOOLS)
    )
