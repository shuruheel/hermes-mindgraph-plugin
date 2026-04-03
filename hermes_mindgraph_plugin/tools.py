"""MindGraph semantic graph memory integration.

Provides 11 cognitive tools that map to MindGraph's layered architecture:
  - mindgraph_session: open/close sessions (lifecycle)
  - mindgraph_journal: low-friction capture (observations, notes, preferences)
  - mindgraph_argue: structured epistemic claims with evidence
  - mindgraph_commit: goals, decisions, projects, milestones
  - mindgraph_retrieve: search + structured queries (goals, questions, etc.)
  - mindgraph_capture: entities, observations, concepts (Reality layer)
  - mindgraph_inquire: questions, hypotheses, anomalies (Epistemic layer)
  - mindgraph_action: risks and affordances (Action layer)
  - mindgraph_decide: decisions with options (Intent layer)
  - mindgraph_plan: plans, tasks, execution, governance (Agent layer)
  - mindgraph_ingest: long-form content ingestion

Session-start context (active goals, projects, tasks) is injected into the
system prompt via retrieve_session_context().

Graceful degradation: all tools return error JSON on failure rather than
crashing. If the API is down, tools are still registered but return
friendly errors. Connection errors auto-reset the client for retry.

This module is self-contained — no Hermes internal imports.
Only depends on mindgraph-sdk (the `mindgraph` package) and Python stdlib.
"""

import json
import logging
import os
import re
from difflib import SequenceMatcher

import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client singleton (lazy init)
# ---------------------------------------------------------------------------

_client: Optional[object] = None
_client_error: Optional[str] = None
_active_session_uid: Optional[str] = None  # Track open MindGraph session
_session_lock = threading.Lock()  # Protect _active_session_uid in multi-user gateway

# ---------------------------------------------------------------------------
# Configurable settings (override via environment variables)
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    val = os.getenv(key, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default

def _env_int(key: str, default: int) -> int:
    val = os.getenv(key, "")
    try:
        return int(val) if val else default
    except ValueError:
        return default

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "")
    if not val:
        return default
    return val.lower() in ("true", "1", "yes")

# Proactive retrieval settings
PROACTIVE_RETRIEVAL_ENABLED = _env_bool("MINDGRAPH_PROACTIVE_RETRIEVAL", True)
PROACTIVE_K = _env_int("MINDGRAPH_PROACTIVE_K", 5)

# Dedup settings
DEDUP_FUZZY_THRESHOLD = _env_float("MINDGRAPH_DEDUP_FUZZY_THRESHOLD", 0.85)

# Pre-compression settings
PRE_COMPRESS_LIMIT = _env_int("MINDGRAPH_PRE_COMPRESS_LIMIT", 4000)

# ---------------------------------------------------------------------------
# Proactive injection metrics (lightweight, in-memory)
# ---------------------------------------------------------------------------

class _ProactiveMetrics:
    """Track proactive graph retrieval performance for measurement."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_calls = 0
        self.hits = 0           # Returned context (non-None)
        self.misses = 0         # Returned None
        self.skip_short = 0     # Skipped due to short message
        self.skip_threshold = 0 # Skipped due to low score
        self.errors = 0
        self.total_latency_ms = 0.0
        self.score_buckets = {  # Distribution of top scores
            "0.0-0.3": 0,
            "0.3-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.9": 0,
            "0.9-1.0": 0,
        }

    def record(self, *, hit: bool, latency_ms: float = 0, top_score: float = 0,
               skip_reason: str = "", error: bool = False):
        with self._lock:
            self.total_calls += 1
            if error:
                self.errors += 1
                return
            if skip_reason == "short":
                self.skip_short += 1
                return
            self.total_latency_ms += latency_ms
            if top_score > 0:
                if top_score < 0.3:
                    self.score_buckets["0.0-0.3"] += 1
                elif top_score < 0.5:
                    self.score_buckets["0.3-0.5"] += 1
                elif top_score < 0.7:
                    self.score_buckets["0.5-0.7"] += 1
                elif top_score < 0.9:
                    self.score_buckets["0.7-0.9"] += 1
                else:
                    self.score_buckets["0.9-1.0"] += 1
            if hit:
                self.hits += 1
            else:
                self.misses += 1
                if skip_reason == "threshold":
                    self.skip_threshold += 1

    def snapshot(self) -> dict:
        """Return a copy of current metrics."""
        with self._lock:
            retrieval_calls = self.hits + self.misses
            return {
                "total_calls": self.total_calls,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{self.hits / retrieval_calls:.1%}" if retrieval_calls else "n/a",
                "skip_short": self.skip_short,
                "skip_threshold": self.skip_threshold,
                "errors": self.errors,
                "avg_latency_ms": round(self.total_latency_ms / retrieval_calls, 1) if retrieval_calls else 0,
                "score_distribution": dict(self.score_buckets),
            }


proactive_metrics = _ProactiveMetrics()


def _get_client():
    """Get or create the MindGraph client singleton."""
    global _client, _client_error

    if _client is not None:
        return _client
    if _client_error is not None:
        return None

    api_key = os.getenv("MINDGRAPH_API_KEY", "")
    if not api_key:
        _client_error = "MINDGRAPH_API_KEY not set"
        return None

    try:
        from mindgraph import MindGraph
        _client = MindGraph(
            "https://api.mindgraph.cloud",
            api_key=api_key,
        )
        return _client
    except Exception as e:
        _client_error = str(e)
        logger.warning("MindGraph client init failed: %s", e)
        return None


def _reload_env_key():
    """Re-read MINDGRAPH_API_KEY from ~/.hermes/.env.

    Called on auth errors (401/403) so that key rotations self-heal
    without a process restart.
    """
    try:
        from pathlib import Path
        env_path = Path.home() / ".hermes" / ".env"
        if not env_path.exists():
            return
        # Minimal .env parser — no dotenv dependency in the plugin
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key == "MINDGRAPH_API_KEY":
                value = value.strip().strip("\"'")
                if value:
                    os.environ["MINDGRAPH_API_KEY"] = value
                    logger.info("MindGraph API key reloaded from .env")
                return
    except Exception as e:
        logger.debug("Failed to reload .env: %s", e)


def _reset_client(auth_error: bool = False):
    """Reset client singleton (for reconnection after errors).

    If auth_error=True, also reloads the API key from ~/.hermes/.env
    so that key rotations self-heal without a process restart.
    """
    global _client, _client_error
    if auth_error:
        _reload_env_key()
    _client = None
    _client_error = None


def check_requirements() -> bool:
    """Check if MindGraph API key is available."""
    return bool(os.getenv("MINDGRAPH_API_KEY"))


def _fts_search(client, query, limit=None, node_type=None,
                include_edges=False, include_chunks=False):
    """FTS search via POST /search with optional edges and chunks.

    Tries the public SDK search() first with enrichment params.
    Falls back to client._request() for older SDK versions that don't
    accept include_edges/include_chunks kwargs.

    Response shape:
      - Old server / no flags: flat list of node dicts
      - New server + flags:    {"results": [...], "edges": [...], "chunks": [...]}
    """
    kwargs = {}
    if limit is not None:
        kwargs["limit"] = limit
    if node_type:
        kwargs["node_type"] = node_type
    if include_edges:
        kwargs["include_edges"] = True
    if include_chunks:
        kwargs["include_chunks"] = True

    # Prefer public SDK method — passes enrichment params if SDK supports them
    try:
        return client.search(query, **kwargs)
    except TypeError:
        # SDK's search() doesn't accept include_edges/include_chunks yet —
        # fall back to raw request (will break if SDK removes _request)
        params = {"query": query}
        params.update(kwargs)
        return client._request("POST", "/search", params)


def _normalize_label(text: str) -> str:
    """Normalize a label for fuzzy comparison.

    Lowercases, strips, collapses whitespace, removes punctuation,
    and expands common abbreviations so "Ship v2.0" and "Ship version 2"
    compare as near-identical.
    """
    s = text.strip().lower()
    # Expand common abbreviations before stripping punctuation
    s = re.sub(r"\bv(\d)", r"version \1", s)
    s = re.sub(r"\bver\.?\s*(\d)", r"version \1", s)
    # Remove punctuation (keep alphanumeric and spaces)
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _label_similarity(a: str, b: str) -> float:
    """Compute similarity between two labels after normalization.

    Returns a float 0.0–1.0. Uses SequenceMatcher (stdlib, no dependencies).
    """
    na = _normalize_label(a)
    nb = _normalize_label(b)
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _safe_call(fn, *args, **kwargs):
    """Call a MindGraph SDK method with error handling.

    Returns (result, None) on success or (None, error_string) on failure.
    """
    client = _get_client()
    if client is None:
        return None, _client_error or "MindGraph client not available"

    try:
        result = fn(client, *args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = str(e)
        err_lower = error_msg.lower()
        # If it looks like a connection error, reset client for next attempt
        if any(k in err_lower for k in ("connection", "timeout", "refused", "dns", "401", "403", "unauthorized")):
            is_auth = any(k in err_lower for k in ("401", "403", "unauthorized"))
            _reset_client(auth_error=is_auth)
        logger.warning("MindGraph call failed: %s", e)
        return None, error_msg


def _json_response(success: bool, data=None, error=None) -> str:
    """Format a consistent JSON response."""
    resp = {"success": success}
    if data is not None:
        resp["data"] = data
    if error is not None:
        resp["error"] = error
    return json.dumps(resp, default=str)


# ---------------------------------------------------------------------------
# Session-start context retrieval (called from run_agent.py)
# ---------------------------------------------------------------------------

# Caps for session-start retrieval (prioritized by salience)
_SESSION_CONTEXT_CAPS = {
    "goals": 5,
    "projects": 5,
    "tasks": 10,
    "open_questions": 5,
    "weak_claims": 5,
    "policies": 5,
    "open_decisions": 5,
}


def retrieve_session_context() -> Optional[str]:
    """Retrieve active goals, projects, tasks for system prompt injection.

    Returns a formatted string for the system prompt, or None if unavailable.
    Called once at session start — results are baked into the cached system prompt.
    Does NOT retrieve weak claims (per design decision).

    Uses dedicated API endpoints (get_goals, get_open_questions, get_open_decisions)
    when available, falling back to search.
    """
    if not check_requirements():
        return None

    client = _get_client()
    if client is None:
        return None

    sections = []

    try:
        # ── Governance policies (behavioral constraints) ──
        # These shape HOW the agent acts, not just what it knows.
        # Policies are read via retrieve/search (governance endpoint is write-only).
        try:
            policies = client.search("policy", node_type="Policy", limit=_SESSION_CONTEXT_CAPS["policies"])
            if isinstance(policies, list) and policies:
                lines = []
                for p in policies[:_SESSION_CONTEXT_CAPS["policies"]]:
                    label = p.get("label", "")
                    if not label:
                        continue
                    desc = _get_prop(p, "description", "")
                    line = f"  - {label}"
                    if desc:
                        line += f": {desc[:150]}"
                    lines.append(line)
                if lines:
                    sections.append(
                        "## Active Policies (follow these)\n"
                        "These are governance directives. Apply them to your reasoning and behavior:\n"
                        + "\n".join(lines)
                    )
        except Exception as e:
            logger.debug("MindGraph policies retrieval failed: %s", e)

        # ── Active goals (as directives, not just information) ──
        try:
            goals = client.get_goals()
            if isinstance(goals, list) and goals:
                lines = []
                for g in goals[:_SESSION_CONTEXT_CAPS["goals"]]:
                    label = g.get("label", "")
                    if not label:
                        continue
                    status = _get_prop(g, "status", "")
                    sal = _get_prop(g, "salience", "")
                    line = f"  - {label}"
                    if status:
                        line += f" [{status}]"
                    if sal:
                        line += f" (salience: {sal})"
                    lines.append(line)
                if lines:
                    sections.append(
                        "## Active Goals (advance these when relevant)\n"
                        "When the conversation touches on these goals, actively work toward them — "
                        "don't just acknowledge them:\n"
                        + "\n".join(lines)
                    )
        except Exception as e:
            logger.debug("MindGraph goals retrieval failed: %s", e)

        # ── Projects (structured query by node type) ──
        try:
            results = client.get_nodes(node_type="Project", limit=_SESSION_CONTEXT_CAPS["projects"])
            if isinstance(results, list) and results:
                lines = []
                for p in results[:_SESSION_CONTEXT_CAPS["projects"]]:
                    label = p.get("label", "")
                    if not label:
                        continue
                    status = _get_prop(p, "status", "")
                    line = f"  - {label}"
                    if status:
                        line += f" [{status}]"
                    lines.append(line)
                if lines:
                    sections.append("## Active Projects\n" + "\n".join(lines))
        except Exception as e:
            logger.debug("MindGraph projects retrieval failed: %s", e)

        # ── Tasks (structured query by node type) ──
        try:
            results = client.get_nodes(node_type="Task", limit=_SESSION_CONTEXT_CAPS["tasks"])
            if isinstance(results, list) and results:
                lines = []
                for t in results[:_SESSION_CONTEXT_CAPS["tasks"]]:
                    label = t.get("label", "")
                    if not label:
                        continue
                    status = _get_prop(t, "status", "")
                    line = f"  - {label}"
                    if status:
                        line += f" [{status}]"
                    lines.append(line)
                if lines:
                    sections.append("## Open Tasks\n" + "\n".join(lines))
        except Exception as e:
            logger.debug("MindGraph tasks retrieval failed: %s", e)

        # ── Open decisions (require deliberation) ──
        try:
            decisions = client.get_open_decisions()
            if isinstance(decisions, list) and decisions:
                lines = []
                for d in decisions[:_SESSION_CONTEXT_CAPS["open_decisions"]]:
                    label = d.get("label", "")
                    if not label:
                        continue
                    lines.append(f"  - {label}")
                if lines:
                    sections.append(
                        "## Open Decisions (help resolve when relevant)\n"
                        + "\n".join(lines)
                    )
        except Exception as e:
            logger.debug("MindGraph open decisions retrieval failed: %s", e)

        # ── Open questions (research triggers) ──
        try:
            questions = client.get_open_questions()
            if isinstance(questions, list) and questions:
                lines = []
                for q in questions[:_SESSION_CONTEXT_CAPS["open_questions"]]:
                    label = q.get("label", "")
                    if not label:
                        continue
                    lines.append(f"  - {label}")
                if lines:
                    sections.append(
                        "## Open Questions (investigate when relevant)\n"
                        "If the conversation touches on these, actively pursue answers or new evidence:\n"
                        + "\n".join(lines)
                    )
        except Exception as e:
            logger.debug("MindGraph open questions retrieval failed: %s", e)

        # ── Weak claims (epistemic caution flags) ──
        try:
            weak = client.get_weak_claims()
            if isinstance(weak, list) and weak:
                lines = []
                for c in weak[:_SESSION_CONTEXT_CAPS["weak_claims"]]:
                    label = c.get("label", "")
                    if not label:
                        continue
                    conf = _get_prop(c, "confidence", "")
                    line = f"  - [{conf}] {label}" if conf else f"  - {label}"
                    lines.append(line)
                if lines:
                    sections.append(
                        "## Weak Claims (treat with epistemic caution)\n"
                        "These claims have low confidence. If the conversation touches on them, "
                        "flag the uncertainty rather than asserting them as fact:\n"
                        + "\n".join(lines)
                    )
        except Exception as e:
            logger.debug("MindGraph weak claims retrieval failed: %s", e)

    except Exception as e:
        logger.warning("MindGraph session context retrieval failed: %s", e)
        return None

    header = (
        "# MindGraph — Semantic Graph Memory\n\n"
        "You have a persistent semantic graph that stores structured knowledge across sessions. "
        "MindGraph is not a tool you use when asked — it is how you think. Every conversation "
        "produces knowledge. Your job is to route that knowledge to the right place in the graph "
        "so your future selves can act on it.\n\n"
        #
        # ── Behavioral contract ──
        #
        "## Behavioral Contract (hard rules)\n\n"
        #
        "### 1. Persist Triggers — ALWAYS store these, don't wait to be asked\n"
        "- **Person discussed substantively** → capture(entity_type='person') + at least one observation "
        "(who they are, what they think, how they communicate). A Person node without observations is a "
        "failure state — bare CRM data (name, role, email) is useless for future interactions.\n"
        "- **Organization mentioned with context** → capture(entity_type='organization') + observation "
        "linking it to relevant people or topics.\n"
        "- **User states a preference or corrects you** → journal(entry_type='preference')\n"
        "- **Analytical conclusion reached** → argue() with evidence and calibrated confidence. "
        "If you formed a view, the graph needs it.\n"
        "- **User expresses intent** → commit() as goal/project/milestone. User's own intent only — "
        "never commit content-derived goals.\n"
        "- **Decision point identified** → decide(action='open') + options.\n"
        "- **Factual claim that could be wrong** → argue(), not journal(). Claims should be falsifiable.\n"
        "- **Risk or opportunity surfaced** → action().\n"
        "- **Open question worth tracking** → inquire().\n\n"
        #
        "### 2. Retrieve-Before-Act — ALWAYS retrieve before acting on stored knowledge\n"
        "Retrieval uses full-text search (keyword matching, not semantic). Write queries as "
        "keywords and names, not natural-language questions.\n"
        "- **Before drafting any communication about/to a person** → retrieve(query='Alice Smith'). "
        "Draft from retrieved context, not session memory.\n"
        "- **Before making recommendations involving a person or org** → retrieve() first.\n"
        "- **When user references someone discussed before** → retrieve() before responding.\n"
        "- **When user says 'remember when...' or 'what did we...'** → retrieve() with key terms, "
        "not the full question.\n"
        "- **Broaden on miss**: If retrieve returns nothing, try synonyms, related terms, or "
        "node_type filter. FTS matches exact words — rephrase if needed.\n"
        "- **If retrieve returns nothing for someone you supposedly researched** → something broke. "
        "Flag it, don't proceed on session memory alone.\n\n"
        #
        "### 3. Research Loop (any time you research a person, not just outreach)\n"
        "Research → Persist → Retrieve → Act. Each step gates the next.\n"
        "1. Research: gather substantive sources (blog, talks, papers, interviews)\n"
        "2. Persist: store Person entity + observations (intellectual profile, communication style, "
        "technical positions, hook/relevance rationale). Do NOT proceed until these are in MindGraph.\n"
        "3. Retrieve: query MindGraph for what you just stored + any prior context. "
        "Use the RETRIEVED context to act.\n"
        "4. Act: draft email, make recommendation, update user — grounded in graph, not session memory.\n\n"
        #
        # ── Decision tree ──
        #
        "## Which Tool? (decision tree)\n"
        "Something worth remembering?\n"
        "- Belief, conclusion, or position that could be wrong? → **argue**\n"
        "- Goal, project, or milestone? → **commit**\n"
        "- Choice point with options? → **decide**\n"
        "- Risk or opportunity? → **action**\n"
        "- Plan, task, or policy? → **plan**\n"
        "- Question, hypothesis, or anomaly? → **inquire**\n"
        "- Person, org, nation, place, event, or concept? → **capture**(entity) — each type creates a distinct node\n"
        "- Factual observation about the world? → **capture**(observation)\n"
        "- Long document, article, or transcript? → **ingest**\n"
        "- Preference, note, insight, or reflection? → **journal**\n"
        "- Need to find something? → **retrieve**\n\n"
        #
        # ── Tool patterns ──
        #
        "## Tool Patterns\n\n"
        #
        "**journal** — low-friction capture (default when genuinely unsure)\n"
        "Entry types: observation (factual, default), preference (likes/dislikes), "
        "note (general), reflection (meta/process), insight (connections/patterns).\n"
        "Anti-pattern: Don't journal claims → argue. Don't journal goals → commit. "
        "Don't journal entity facts → capture(observation).\n\n"
        #
        "**argue** — structured epistemic claims\n"
        "Confidence: 0.9+ near-certain; 0.7-0.8 confident (default 0.7); "
        "0.5-0.6 uncertain; 0.3-0.4 speculative; 0.1-0.2 intuition.\n"
        "Anti-pattern: Don't argue bare observations or preferences. Claims should be falsifiable.\n\n"
        #
        "**commit** — goals, projects, milestones (with dedup + update)\n"
        "Status: active → completed/paused/abandoned. Don't auto-complete — let user confirm.\n"
        "Dedup: automatically finds existing commitments with the same label — safe to call repeatedly.\n"
        "Update: pass uid to update an existing commitment's status (e.g., mark a goal completed).\n"
        "Anti-pattern: Commits represent user intent and agency. Never commit content-derived goals.\n\n"
        #
        "**decide** — decisions with deliberation\n"
        "Actions: open → add_option → add_constraint → resolve.\n\n"
        #
        "**capture** — Reality layer entities and facts\n"
        "Entity types (each creates a DISTINCT typed node):\n"
        "  person: occupation, nationality, birth_date, death_date, identifiers, attributes\n"
        "  organization: domain, description\n"
        "  nation, place, event: description\n"
        "  concept: domain, description\n"
        "  work, other: generic fallback\n"
        "Also: capture_type='observation' for factual observations — pass entity_uid to link "
        "the observation to a specific entity (creates HAS_OBSERVATION edge).\n"
        "KEY PRINCIPLE: Entity nodes are for stable identity. Observations are for everything you learn "
        "ABOUT them. Use descriptive keywords in observation labels — FTS retrieves by keyword match, "
        "so clear terms make observations findable. Don't cram findings into entity properties.\n\n"
        #
        "**inquire** — questions, hypotheses, anomalies\n"
        "question: open questions (surface at session start). hypothesis: testable conjectures. "
        "anomaly: things that don't fit.\n\n"
        #
        "**action** — risks and affordances\n"
        "assess_risk: threats, failure modes. add_affordance: capabilities, leverage points, opportunities.\n\n"
        #
        "**plan** — plans, tasks, execution, governance\n"
        "Planning: create_plan, create_task, add_step, update_status, get_plan.\n"
        "Execution: start → complete/fail.\n"
        "Governance: create_policy (rules/constraints the agent should follow).\n\n"
        #
        "**ingest** — long-form content\n"
        "Under 500 chars: sync. Over 500 chars: async.\n"
        "Anti-pattern: Don't ingest trivial content or duplicates.\n\n"
        #
        "**retrieve** — querying the graph (FTS-based)\n"
        "Modes: search (FTS, default — supports node_type filter, returns nodes + edges + chunks), "
        "goals, questions, decisions, neighborhood (from a node UID), weak_claims, contradictions.\n"
        "Topic-relevant context is auto-injected each turn — use retrieve for deeper/specific queries.\n"
        "**Query tips (FTS = keyword matching):**\n"
        "  - Use names and key terms, not questions: 'Alice Smith AI safety' not 'what does Alice think about AI?'\n"
        "  - Use node_type filter to narrow: retrieve(query='Smith', node_type='Person')\n"
        "  - If no results, try fewer/broader keywords or drop qualifiers\n"
        "  - Combine with neighborhood mode: search for a node, then explore its connections\n\n"
        #
        # ── Social graph ──
        #
        "## Building a Social Graph\n\n"
        "Every conversation should contribute to a living social graph. This is not optional "
        "and not limited to specific workflows — it's how the graph becomes genuinely useful.\n\n"
        "Pattern:\n"
        "1. capture(entity_type='person', props={occupation, nationality}) — typed Person node (returns uid)\n"
        "2. capture(entity_type='organization') — their company/institution\n"
        "3. capture(observation, entity_uid=person_uid) per distinct insight — one observation per fact, linked to entity:\n"
        "   - Intellectual profile: what they think about, their core frames\n"
        "   - Communication style: formal/informal, academic/practitioner, contrarian/consensus\n"
        "   - Technical positions: specific claims they've taken\n"
        "   - Career arc: trajectory, pivots (if notable)\n"
        "   - Relationships: 'X co-founded Y', 'X worked at Z before joining W'\n"
        "4. argue() when you form a view — 'X would be a strong collaborator because...' "
        "with evidence and confidence\n\n"
        "The bar: 'would I want to retrieve this in a future conversation?' If yes, create the node.\n\n"
        #
        # ── Compound patterns ──
        #
        "## Compound Patterns\n"
        "- **Learning:** ingest → argue key conclusions → journal insights → commit goals\n"
        "- **Deciding:** retrieve context → argue reasoning → decide → journal what was rejected\n"
        "- **Completing a goal:** retrieve goals → commit(uid=goal_uid, status='completed') → journal lessons\n"
        "- **Researching a person:** capture(person) → capture(org) → multiple observations → "
        "argue(relevance) → plan(follow-up)\n"
        "- **New session:** session-start context is auto-injected. "
        "Retrieve specifics before proceeding.\n\n"
        #
        # ── Anti-patterns ──
        #
        "## Anti-Patterns\n"
        "- **Bare entity nodes**: A Person with name and email but no observations is useless. "
        "Always pair with at least one qualitative observation.\n"
        "- **Session-memory drafting**: Never draft communications from what you researched "
        "'5 minutes ago' — retrieve from graph. Session memory dies; graph persists.\n"
        "- **Over-journaling**: Skip transient, obvious, or easily re-discovered facts.\n"
        "- **Under-arguing**: If you formed a conclusion, argue it. The graph needs claims to reason over.\n"
        "- **Confidence inflation**: 0.7 is a good default. Don't inflate to 0.9 without strong evidence.\n"
        "- **Stale goals**: Update promptly. Stale active goals pollute session-start context.\n"
        "- **Content-derived commits**: Don't create goals from articles or research. Only user intent.\n"
        "- **Cramming**: Entity properties = stable identity. Observations = everything else.\n"
        "- **Performative retrieval**: Don't retrieve to show you can — only when it would change your response.\n\n"
        #
        # ── Judgment calls ──
        #
        "## Judgment Calls\n"
        "- During coding/execution, persistence is less frequent. During planning, research, "
        "or reflective conversations, persist more.\n"
        "- Post-session extraction runs automatically — focus on high-signal items during session.\n"
        "- When in doubt between journal and argue: is this falsifiable? If yes → argue. If no → journal."
    )

    if sections:
        return header + "\n\n## Current Context\n\n" + "\n\n".join(sections)
    else:
        return header


def _get_prop(node: dict, key: str, default=""):
    """Get a property from a node, checking both top-level and nested props."""
    val = node.get(key, "")
    if val:
        return val
    props = node.get("props", {})
    if isinstance(props, dict):
        return props.get(key, default)
    return default


# ---------------------------------------------------------------------------
# Auto session open/close (called from run_agent.py and gateway)
# ---------------------------------------------------------------------------

def auto_open_session(label: str = "hermes-session") -> Optional[str]:
    """Auto-open a MindGraph session at conversation start.
    
    Returns the session UID on success, None on failure.
    Called from run_agent.py at the start of the first conversation turn.
    """
    global _active_session_uid
    with _session_lock:
        if _active_session_uid:
            return _active_session_uid  # Already open

    if not check_requirements():
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        result = client.session(action="open", label=label)
        if isinstance(result, dict) and result.get("uid"):
            with _session_lock:
                _active_session_uid = result["uid"]
            logger.info("MindGraph session opened: %s", result["uid"])
            return result["uid"]
    except Exception as e:
        logger.debug("MindGraph auto session open failed (non-fatal): %s", e)
    return None


def _filter_transcript_for_ingestion(messages: list) -> str:
    """Filter conversation messages to user+assistant text only.
    
    Strips tool calls, tool results, system messages, and internal metadata.
    Returns a formatted transcript string suitable for ingest_session().
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        # Extract text content (handles both string and list-of-blocks formats)
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multimodal messages: extract only text blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        if not content or not content.strip():
            continue

        # Skip tool-result-like messages that snuck into assistant role
        content_stripped = content.strip()
        if content_stripped.startswith("{") and '"tool_call_id"' in content_stripped:
            continue

        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content_stripped}")

    return "\n\n".join(lines)


def auto_close_session(
    summary: str = "",
    transcript_messages: list = None,
    session_title: str = None,
) -> bool:
    """Auto-close the active MindGraph session.
    
    Called from gateway session expiry watcher or explicit reset.
    
    If transcript_messages is provided, runs ingest_session() to extract
    structured knowledge (entities, claims, goals, decisions, etc.) across
    all graph layers. This replaces the old distill-only approach.
    
    Returns True on success, False on failure.
    """
    global _active_session_uid
    with _session_lock:
        if not _active_session_uid:
            return False  # No session to close
        uid = _active_session_uid
        _active_session_uid = None  # Clear immediately to prevent double-close

    if not check_requirements():
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        close_kwargs = {"action": "close", "session_uid": uid}
        if summary:
            close_kwargs["summary"] = summary
        client.session(**close_kwargs)
        logger.info("MindGraph session closed: %s", uid)

        # Ingest full transcript for 5-layer extraction if provided
        if transcript_messages:
            try:
                filtered = _filter_transcript_for_ingestion(transcript_messages)
                if filtered and len(filtered) > 50:  # Skip trivially short sessions
                    title = session_title or summary[:120] or f"Hermes session {uid[:8]}"
                    result = client.ingest_session(
                        content=filtered,
                        title=title,
                        session_uid=uid,
                        agent_id="hermes",
                    )
                    job_id = result.get("job_id", "unknown") if isinstance(result, dict) else "unknown"
                    logger.info(
                        "MindGraph session transcript ingested (job_id=%s, %d chars)",
                        job_id, len(filtered),
                    )
                else:
                    logger.debug("MindGraph session transcript too short to ingest (%d chars)", len(filtered) if filtered else 0)
            except Exception as e:
                logger.debug("MindGraph session ingest failed (non-fatal): %s", e)
        elif summary:
            # Fallback: distill summary only (no transcript available, e.g. CLI mode)
            try:
                client.distill(label=summary[:120], summary=summary)
                logger.info("MindGraph session distilled (summary only)")
            except Exception as e:
                logger.debug("MindGraph distill failed (non-fatal): %s", e)

        return True
    except Exception as e:
        logger.debug("MindGraph auto session close failed (non-fatal): %s", e)
        return False


def get_active_session_uid() -> Optional[str]:
    """Return the active session UID, or None."""
    return _active_session_uid


# ---------------------------------------------------------------------------
# Tool: mindgraph_session
# ---------------------------------------------------------------------------

def mindgraph_session(action: str, summary: str = "", label: str = "") -> str:
    """Open or close a MindGraph session.

    action: "open" or "close"
    label: (open only) session label
    summary: (close only) session summary for distillation
    """
    global _active_session_uid

    if action == "open":
        kwargs = {"action": "open"}
        if label:
            kwargs["label"] = label
        result, err = _safe_call(
            lambda c: c.session(**kwargs),
        )
        if err:
            return _json_response(False, error=f"Failed to open session: {err}")
        # Track the session UID for close
        if isinstance(result, dict) and result.get("uid"):
            with _session_lock:
                _active_session_uid = result["uid"]
        with _session_lock:
            sid = _active_session_uid
        return _json_response(True, data={
            "session_id": sid,
            "session": result,
            "message": "MindGraph session opened",
        })

    elif action == "close":
        with _session_lock:
            if not _active_session_uid:
                return _json_response(False, error="No active session to close")
            closed_uid = _active_session_uid
            _active_session_uid = None  # Clear immediately to prevent double-close

        # Close session (requires session_uid)
        close_kwargs = {"action": "close", "session_uid": closed_uid}
        if summary:
            close_kwargs["summary"] = summary
        result, err = _safe_call(
            lambda c: c.session(**close_kwargs),
        )

        if err:
            return _json_response(False, error=f"Failed to close session: {err}")

        # Attempt distillation if we have a summary
        if summary:
            dist_result, dist_err = _safe_call(
                lambda c: c.distill(label=summary[:120], summary=summary),
            )
            if dist_err:
                logger.debug("MindGraph distill failed: %s", dist_err)

        return _json_response(True, data={
            "session_id": closed_uid,
            "session": result,
            "message": "MindGraph session closed",
        })

    else:
        return _json_response(False, error=f"Unknown action: {action}. Use 'open' or 'close'.")


# ---------------------------------------------------------------------------
# Tool: mindgraph_journal
# ---------------------------------------------------------------------------

def mindgraph_journal(entry: str, entry_type: str = "observation") -> str:
    """Low-friction capture: observations, notes, preferences, reflections.

    entry_type: observation, preference, note, reflection, insight
    """
    if not entry or not entry.strip():
        return _json_response(False, error="Entry text is required")
    # SDK journal() signature: journal(label, props=None, *, summary=None, ...)
    result, err = _safe_call(
        lambda c: c.journal(entry, props={"entry_type": entry_type}),
    )
    if err:
        return _json_response(False, error=f"Journal capture failed: {err}")
    return _json_response(True, data={"node": result, "message": f"Journaled {entry_type}"})


# ---------------------------------------------------------------------------
# Tool: mindgraph_argue
# ---------------------------------------------------------------------------

def mindgraph_argue(
    claim: str,
    evidence: str = "",
    warrant: str = "",
    confidence: float = 0.7,
) -> str:
    """Structured epistemic claim with evidence and warrant.

    Use for beliefs, conclusions, and reasoned positions that need
    to be tracked and potentially challenged later.
    """
    if not claim or not claim.strip():
        return _json_response(False, error="Claim text is required")
    # SDK argue() takes **kwargs -> POST /epistemic/argument
    # claim must be a dict: {"label": str, "confidence": float}
    # evidence must be a list of dicts: [{"label": str}]
    claim_obj = {"label": claim, "confidence": confidence}
    kwargs = {"claim": claim_obj}
    if evidence:
        kwargs["evidence"] = [{"label": evidence}]
    if warrant:
        kwargs["warrant"] = {"label": warrant}

    result, err = _safe_call(
        lambda c: c.argue(**kwargs),
    )
    if err:
        return _json_response(False, error=f"Argue failed: {err}")
    return _json_response(True, data={"result": result, "message": f"Claim recorded (confidence: {confidence})"})


# ---------------------------------------------------------------------------
# Tool: mindgraph_commit
# ---------------------------------------------------------------------------

def mindgraph_commit(
    label: str = "",
    commit_type: str = "goal",
    status: str = "active",
    description: str = "",
    uid: str = "",
) -> str:
    """Record or update goals, projects, and milestones.

    commit_type: goal, project, milestone (NOT decision — use mindgraph_decide)
    status: active, completed, paused, abandoned
    uid: (optional) UID of existing commitment to update

    Deduplication: before creating, searches for an existing node with the same
    label and type. If found, returns the existing node (updating status if changed).
    Pass uid explicitly to update a known commitment.
    """
    if not label and not uid:
        return _json_response(False, error="Label or uid is required")

    # ── Explicit update by UID ──
    if uid:
        update_kwargs = {}
        if status:
            update_kwargs["status"] = status
        if description:
            update_kwargs["description"] = description
        if label:
            update_kwargs["label"] = label
        result, err = _safe_call(
            lambda c: c.update_node(uid, **update_kwargs),
        )
        if err:
            return _json_response(False, error=f"Update failed: {err}")
        return _json_response(True, data={
            "result": result,
            "uid": uid,
            "message": f"Updated {commit_type}: {label or uid}",
        })

    # ── Client-side dedup: search for existing commitment with same label ──
    # Two-pass: exact match first (cheap), then fuzzy match via normalized
    # SequenceMatcher (catches "Ship v2" ≈ "Ship version 2.0").
    # Threshold is configurable: MINDGRAPH_DEDUP_FUZZY_THRESHOLD (default 0.85).
    _node_type_map = {"goal": "Goal", "project": "Project", "milestone": "Milestone"}
    node_type = _node_type_map.get(commit_type)

    if node_type and label:
        try:
            client = _get_client()
            if client:
                raw = _fts_search(client, label, node_type=node_type, limit=5)
                # Normalize — _fts_search may return enriched dict or flat list
                existing = raw.get("results", []) if isinstance(raw, dict) else raw
                if isinstance(existing, list):
                    # Pass 1: exact match (case-insensitive)
                    match = None
                    match_kind = ""
                    label_lower = label.strip().lower()
                    for node in existing:
                        existing_label = (node.get("label") or "").strip().lower()
                        if existing_label == label_lower:
                            match = node
                            match_kind = "exact"
                            break

                    # Pass 2: fuzzy match if no exact hit
                    if match is None and DEDUP_FUZZY_THRESHOLD < 1.0:
                        best_score = 0.0
                        for node in existing:
                            existing_label = node.get("label") or ""
                            if not existing_label:
                                continue
                            score = _label_similarity(label, existing_label)
                            if score >= DEDUP_FUZZY_THRESHOLD and score > best_score:
                                best_score = score
                                match = node
                                match_kind = f"fuzzy ({best_score:.0%})"

                    if match is not None:
                        node_uid = match.get("uid", "")
                        existing_status = _get_prop(match, "status", "")
                        needs_update = (
                            (status and existing_status != status)
                            or description
                        )
                        if needs_update and node_uid:
                            update_kw = {}
                            if status and existing_status != status:
                                update_kw["status"] = status
                            if description:
                                update_kw["description"] = description
                            try:
                                client.update_node(node_uid, **update_kw)
                                match.update(update_kw)
                            except Exception as ue:
                                logger.debug("Dedup update failed (non-fatal): %s", ue)
                        return _json_response(True, data={
                            "result": match,
                            "uid": node_uid,
                            "message": f"Found existing {commit_type}: {match.get('label', label)} ({match_kind} dedup)",
                            "deduplicated": True,
                        })
        except Exception:
            pass  # Fall through to create

    # ── Create new commitment ──
    kwargs = {"action": commit_type, "label": label, "status": status}
    if description:
        kwargs["description"] = description

    result, err = _safe_call(
        lambda c: c.commit(**kwargs),
    )
    if err:
        return _json_response(False, error=f"Commit failed: {err}")
    return _json_response(True, data={"result": result, "message": f"Committed {commit_type}: {label}"})


# ---------------------------------------------------------------------------
# Tool: mindgraph_retrieve
# ---------------------------------------------------------------------------

def mindgraph_retrieve(
    query: str = "",
    mode: str = "search",
    limit: int = 10,
    include_chunks: bool = True,
    include_graph: bool = True,
    node_type: str = "",
) -> str:
    """Search and query the semantic graph.

    All retrieval uses FTS or structured endpoints (no embedding/vector search).

    Modes:
      - search: full-text search (FTS) via POST /search — the default for most queries.
        Returns nodes, edges, and provenance chunks. Supports node_type filter.
      - goals: active goals sorted by salience
      - questions: open questions and hypotheses
      - decisions: open decisions needing resolution
      - neighborhood: get nodes connected to a specific node (query=node_uid)
      - weak_claims: claims with low confidence
      - contradictions: contradictory claims in the graph
    """
    # Validate query for modes that need it
    if mode == "search" and not query:
        return _json_response(False, error="Query required for search mode")
    if mode == "neighborhood" and not query:
        return _json_response(False, error="Node UID required for neighborhood mode")
    if mode not in ("search", "goals", "questions", "decisions", "neighborhood", "weak_claims", "contradictions"):
        return _json_response(False, error=f"Unknown mode: {mode}. Use: search, goals, questions, decisions, neighborhood, weak_claims, contradictions")

    def _do_retrieve(client):
        if mode == "search":
            return _fts_search(
                client, query, limit=limit, node_type=node_type or None,
                include_edges=include_graph, include_chunks=include_chunks,
            )
        elif mode == "goals":
            return client.get_goals()
        elif mode == "questions":
            return client.get_open_questions()
        elif mode == "decisions":
            return client.get_open_decisions()
        elif mode == "neighborhood":
            return client.neighborhood(query, max_depth=1)
        elif mode == "weak_claims":
            return client.get_weak_claims()
        elif mode == "contradictions":
            return client.get_contradictions()

    # Hard caps for structured modes (these endpoints return everything)
    _STRUCTURED_CAPS = {
        "goals": 10,
        "questions": 10,
        "decisions": 10,
        "weak_claims": 10,
        "contradictions": 10,
        "neighborhood": 20,
    }
    effective_limit = min(limit, _STRUCTURED_CAPS.get(mode, limit))

    result, err = _safe_call(_do_retrieve)
    if err:
        return _json_response(False, error=f"Retrieve failed: {err}")

    # Normalize results — handle both enriched dict and flat list responses:
    # - Enriched (search with include_*): {"results": [...], "edges": [...], "chunks": [...]}
    # - Flat list: [{node}, ...] or [{node: {...}, score: N}, ...]
    edges = []
    chunks = []
    if isinstance(result, dict):
        nodes = result.get("results", result.get("nodes", []))
        edges = result.get("edges", [])
        chunks = result.get("chunks", [])
    elif isinstance(result, list):
        nodes = result
    else:
        nodes = [result] if result else []

    # Format nodes — lean output for structured modes, full for search
    formatted = []
    for item in nodes[:effective_limit]:
        if not isinstance(item, dict):
            formatted.append(str(item))
            continue

        # Unwrap {node: {...}, score: N} envelope
        n = item.get("node", item) if "node" in item else item

        if mode in ("goals", "questions", "decisions"):
            # Lean: only uid, label, status
            entry = {
                "uid": n.get("uid", ""),
                "label": n.get("label", ""),
            }
            status = _get_prop(n, "status", "")
            if status:
                entry["status"] = status
            formatted.append(entry)

        elif mode == "weak_claims":
            # Lean: uid, label, confidence
            entry = {
                "uid": n.get("uid", ""),
                "label": n.get("label", ""),
            }
            conf = n.get("confidence", _get_prop(n, "confidence", ""))
            if conf:
                entry["confidence"] = conf
            formatted.append(entry)

        elif mode == "contradictions":
            # Lean: uid, label, confidence
            entry = {
                "uid": n.get("uid", ""),
                "label": n.get("label", ""),
            }
            conf = n.get("confidence", _get_prop(n, "confidence", ""))
            if conf:
                entry["confidence"] = conf
            formatted.append(entry)

        else:
            # Full format for search / neighborhood
            score = item.get("score", "")
            entry = {
                "uid": n.get("uid", ""),
                "label": n.get("label", ""),
                "type": n.get("node_type", n.get("type", "")),
            }
            status = _get_prop(n, "status", "")
            if status:
                entry["status"] = status
            if score:
                entry["score"] = score
            # Include summary only if label is short/generic
            summary = n.get("summary", "")
            if summary and (not entry["label"] or entry["label"].startswith("chunk-")):
                entry["summary"] = summary[:200]
            formatted.append(entry)

    data = {
        "results": formatted,
        "count": len(formatted),
        "mode": mode,
    }
    if query:
        data["query"] = query

    # Include edges and chunks when present (FTS enriched response)
    if edges:
        data["edges"] = [
            {
                "type": e.get("edge_type", e.get("type", "")),
                "source": e.get("source_label", e.get("from_label", e.get("source_uid", ""))),
                "target": e.get("target_label", e.get("to_label", e.get("target_uid", ""))),
            }
            for e in edges[:effective_limit]
        ]
        data["edge_count"] = len(edges)
    if chunks:
        data["chunks"] = [
            {
                "content": c.get("content", c.get("text", ""))[:500],
                "document_title": c.get("document_title", c.get("source", "")),
            }
            for c in chunks[:effective_limit]
        ]
        data["chunk_count"] = len(chunks)

    return _json_response(True, data=data)


# ---------------------------------------------------------------------------
# Tool: mindgraph_capture — Reality layer (entities, observations)
# ---------------------------------------------------------------------------

def mindgraph_capture(
    label: str,
    capture_type: str = "entity",
    entity_type: str = "concept",
    properties: dict = None,
    entity_uid: str = "",
) -> str:
    """Capture facts in the Reality layer — entities, observations, and concepts.

    capture_type:
      - entity: A concrete thing — each entity_type creates a distinct node type with its
        own properties and edge types in the graph. Uses deduplication — safe to call repeatedly.
      - observation: A factual observation about the world — something noticed or reported.
        Pass entity_uid to explicitly link the observation to an entity.
      - concept: An abstract concept in the Epistemic layer (via structure endpoint).

    entity_type (entity only): concept, person, organization, nation, place, event, work, other
    properties: additional properties dict (type-specific props passed to the server)
    entity_uid: (observation only) UID of an entity to link this observation to via HAS_OBSERVATION edge
    """
    if not label or not label.strip():
        return _json_response(False, error="Label is required")

    if capture_type == "entity":
        props = {}
        if properties and isinstance(properties, dict):
            props.update(properties)

        # Route to typed SDK methods for proper node type creation
        _typed_creators = {
            "person": "find_or_create_person",
            "organization": "find_or_create_organization",
            "nation": "find_or_create_nation",
            "event": "find_or_create_event",
            "place": "find_or_create_place",
            "concept": "find_or_create_concept",
        }
        creator_method = _typed_creators.get(entity_type)
        if creator_method:
            result, err = _safe_call(
                lambda c, m=creator_method, p=props: getattr(c, m)(label, props=p if p else None),
            )
        else:
            # "work", "other", or unrecognized — fall back to generic with entity_type in props
            props["entity_type"] = entity_type
            result, err = _safe_call(
                lambda c: c.find_or_create_entity(label, props=props),
            )
        if err:
            return _json_response(False, error=f"Entity creation failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Entity ({entity_type}): {label}"})

    elif capture_type == "observation":
        kwargs = {"action": "observation", "label": label}
        if properties and isinstance(properties, dict):
            kwargs.update(properties)
        result, err = _safe_call(
            lambda c: c.capture(**kwargs),
        )
        if err:
            return _json_response(False, error=f"Observation capture failed: {err}")

        # Link observation to entity if entity_uid provided
        linked = False
        if entity_uid and result and isinstance(result, dict):
            obs_uid = result.get("uid", "")
            if obs_uid:
                _, link_err = _safe_call(
                    lambda c: c.add_edge(
                        source_uid=entity_uid,
                        target_uid=obs_uid,
                        edge_type="HAS_OBSERVATION",
                    ),
                )
                if link_err:
                    logger.debug("Observation-entity link failed (non-fatal): %s", link_err)
                else:
                    linked = True

        msg = f"Observation: {label}"
        if linked:
            msg += f" (linked to {entity_uid[:8]})"
        return _json_response(True, data={"result": result, "message": msg})

    elif capture_type == "concept":
        result, err = _safe_call(
            lambda c: c.structure(action="concept", label=label),
        )
        if err:
            return _json_response(False, error=f"Concept creation failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Concept: {label}"})

    else:
        return _json_response(False, error=f"Unknown capture_type: {capture_type}. Use: entity, observation, concept")


# ---------------------------------------------------------------------------
# Tool: mindgraph_inquire — Epistemic layer (open questions)
# ---------------------------------------------------------------------------

def mindgraph_inquire(label: str, inquiry_type: str = "question") -> str:
    """Create questions, hypotheses, or anomalies in the Epistemic layer.

    inquiry_type:
      - question: An open question worth tracking (shows up as "Open Questions" at session start)
      - hypothesis: A testable hypothesis or conjecture
      - anomaly: Something that doesn't fit existing models — worth investigating
    """
    if not label or not label.strip():
        return _json_response(False, error="Label is required")
    if inquiry_type not in ("question", "hypothesis", "anomaly"):
        return _json_response(False, error=f"Unknown inquiry_type: {inquiry_type}. Use: question, hypothesis, anomaly")
    result, err = _safe_call(
        lambda c: c.inquire(action=inquiry_type, label=label),
    )
    if err:
        return _json_response(False, error=f"Inquire failed: {err}")
    return _json_response(True, data={"result": result, "message": f"{inquiry_type.title()} recorded: {label[:80]}"})


# ---------------------------------------------------------------------------
# Tool: mindgraph_action — Action layer (risk assessments)
# ---------------------------------------------------------------------------

def mindgraph_action(
    action: str = "assess_risk",
    label: str = "",
    description: str = "",
) -> str:
    """Record risks and affordances in the Action layer.

    Actions:
      - assess_risk: Record a risk, threat, or potential failure mode.
      - add_affordance: Record an affordance — an action that is available or enabled
        by the current state of affairs (e.g. diplomatic leverage, technical capability).
    """
    if not label or not label.strip():
        return _json_response(False, error="Label is required")

    if action == "assess_risk":
        kwargs = {"action": "assess", "label": label}
        if description:
            kwargs["description"] = description
        result, err = _safe_call(lambda c: c.risk(**kwargs))
        if err:
            return _json_response(False, error=f"Risk assessment failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Risk assessed: {label[:80]}"})

    elif action == "add_affordance":
        kwargs = {"action": "add_affordance", "label": label}
        if description:
            kwargs["description"] = description
        result, err = _safe_call(lambda c: c.procedure(**kwargs))
        if err:
            return _json_response(False, error=f"Add affordance failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Affordance: {label[:80]}"})

    else:
        return _json_response(False, error=f"Unknown action: {action}. Use: assess_risk, add_affordance")


# ---------------------------------------------------------------------------
# Tool: mindgraph_decide — Intent layer (decisions with options)
# ---------------------------------------------------------------------------

def mindgraph_decide(
    action: str = "open",
    label: str = "",
    decision_uid: str = "",
    option_label: str = "",
    chosen_option_uid: str = "",
    summary: str = "",
) -> str:
    """Manage decisions and constraints in the Intent layer.

    Actions:
      - open: Create a new open decision (requires label)
      - add_option: Add an option to a decision (requires decision_uid + option_label)
      - add_constraint: Add a constraint to a decision (requires decision_uid + label)
      - resolve: Resolve by choosing an option (requires decision_uid + chosen_option_uid)
    """
    if action == "open":
        if not label or not label.strip():
            return _json_response(False, error="Label required to open a decision")
        result, err = _safe_call(
            lambda c: c.open_decision(label, summary=summary or None),
        )
        if err:
            return _json_response(False, error=f"Open decision failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Decision opened: {label[:80]}"})

    elif action == "add_option":
        if not decision_uid or not option_label:
            return _json_response(False, error="decision_uid and option_label required")
        result, err = _safe_call(
            lambda c: c.add_option(decision_uid, option_label, summary=summary or None),
        )
        if err:
            return _json_response(False, error=f"Add option failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Option added: {option_label[:80]}"})

    elif action == "add_constraint":
        if not decision_uid or not label:
            return _json_response(False, error="decision_uid and label required for constraint")
        result, err = _safe_call(
            lambda c: c.deliberate(action="add_constraint", decision_uid=decision_uid, label=label),
        )
        if err:
            return _json_response(False, error=f"Add constraint failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Constraint added: {label[:80]}"})

    elif action == "resolve":
        if not decision_uid or not chosen_option_uid:
            return _json_response(False, error="decision_uid and chosen_option_uid required")
        result, err = _safe_call(
            lambda c: c.resolve_decision(decision_uid, chosen_option_uid, summary=summary or None),
        )
        if err:
            return _json_response(False, error=f"Resolve decision failed: {err}")
        return _json_response(True, data={"result": result, "message": "Decision resolved"})

    else:
        return _json_response(False, error=f"Unknown action: {action}. Use: open, add_option, add_constraint, resolve")


# ---------------------------------------------------------------------------
# Tool: mindgraph_plan — Agent layer (plans, tasks, execution, governance)
# ---------------------------------------------------------------------------

def mindgraph_plan(
    action: str = "create_task",
    label: str = "",
    plan_uid: str = "",
    task_uid: str = "",
    execution_uid: str = "",
    status: str = "",
    description: str = "",
) -> str:
    """Manage plans, tasks, execution, and governance in the Agent layer.

    Plan actions:
      - create_plan: Create a plan (requires label)
      - create_task: Create a task (requires label)
      - add_step: Add a step to a plan (requires plan_uid + label)
      - update_status: Update a task/plan status (requires task_uid + status)
      - get_plan: Retrieve a plan's details (requires plan_uid)

    Execution actions:
      - start: Start executing a task (requires label or task_uid)
      - complete: Mark execution complete (requires execution_uid)
      - fail: Mark execution failed (requires execution_uid)

    Governance actions:
      - create_policy: Create a governance policy (requires label)
    """
    if action in ("create_plan", "create_task") and not label:
        return _json_response(False, error=f"Label required for {action}")

    if action in ("create_plan", "create_task", "add_step", "update_status", "get_plan"):
        kwargs = {"action": action, "label": label}
        if plan_uid:
            kwargs["plan_uid"] = plan_uid
        if task_uid:
            kwargs["task_uid"] = task_uid
        if status:
            kwargs["status"] = status
        if description:
            kwargs["description"] = description
        result, err = _safe_call(lambda c: c.plan(**kwargs))
        if err:
            return _json_response(False, error=f"Plan action failed: {err}")
        return _json_response(True, data={"result": result, "message": f"{action}: {label[:80]}"})

    elif action in ("start", "complete", "fail"):
        kwargs = {"action": action}
        if label:
            kwargs["label"] = label
        if execution_uid:
            kwargs["execution_uid"] = execution_uid
        if task_uid:
            kwargs["task_uid"] = task_uid
        result, err = _safe_call(lambda c: c.execution(**kwargs))
        if err:
            return _json_response(False, error=f"Execution action failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Execution {action}"})

    elif action == "create_policy":
        if not label:
            return _json_response(False, error="Label required for create_policy")
        kwargs = {"action": "create_policy", "label": label}
        if description:
            kwargs["description"] = description
        result, err = _safe_call(lambda c: c.governance(**kwargs))
        if err:
            return _json_response(False, error=f"Create policy failed: {err}")
        return _json_response(True, data={"result": result, "message": f"Policy: {label[:80]}"})

    else:
        return _json_response(False, error=f"Unknown action: {action}. Use: create_plan, create_task, add_step, update_status, get_plan, start, complete, fail, create_policy")


# ---------------------------------------------------------------------------
# Proactive graph retrieval (called from run_agent.py per-turn)
# ---------------------------------------------------------------------------

def proactive_graph_retrieve(user_message: str, k: int = 0) -> Optional[str]:
    """Retrieve relevant graph context for the current user message.

    Called at the start of each turn to give the agent topic-relevant knowledge
    from the semantic graph without requiring an explicit tool call.

    Uses FTS (full-text search via POST /search) with include_edges=true and
    include_chunks=true for low-latency per-turn retrieval that returns graph
    nodes, their connecting edges, and provenance chunks in a single round-trip.

    Response shape (server v0.5+):
        {"results": [nodes...], "edges": [edges...], "chunks": [chunks...]}

    Falls back gracefully to flat node list on older servers.

    Metrics are tracked in `proactive_metrics` (call .snapshot() for a summary).

    Configurable via environment variables:
        MINDGRAPH_PROACTIVE_RETRIEVAL: "true"/"false" to enable/disable (default: true)
        MINDGRAPH_PROACTIVE_K: max results per retrieval (default: 5)
    """
    import time as _time

    if not PROACTIVE_RETRIEVAL_ENABLED:
        return None

    if k <= 0:
        k = PROACTIVE_K

    if not check_requirements():
        return None

    client = _get_client()
    if client is None:
        return None

    # Skip very short messages (greetings, acknowledgments) — not worth a retrieval
    stripped = user_message.strip()
    if len(stripped) < 15:
        proactive_metrics.record(hit=False, skip_reason="short")
        return None

    t0 = _time.monotonic()
    try:
        raw = _fts_search(
            client, stripped[:500], limit=k,
            include_edges=True, include_chunks=True,
        )
    except Exception as e:
        latency_ms = (_time.monotonic() - t0) * 1000
        logger.debug("MindGraph proactive retrieval failed (non-fatal): %s (%.0fms)", e, latency_ms)
        proactive_metrics.record(hit=False, error=True)
        return None

    latency_ms = (_time.monotonic() - t0) * 1000

    # Normalize response — handle both enriched dict and flat list (old server)
    if isinstance(raw, dict):
        nodes = raw.get("results", [])
        edges = raw.get("edges", [])
        chunks = raw.get("chunks", [])
    elif isinstance(raw, list):
        nodes = raw
        edges = []
        chunks = []
    else:
        proactive_metrics.record(hit=False, latency_ms=latency_ms, top_score=0)
        return None

    if not nodes and not chunks:
        proactive_metrics.record(hit=False, latency_ms=latency_ms, top_score=0)
        return None

    sections = []

    # ── Format chunks (provenance source text) ──
    if chunks:
        chunk_lines = []
        for c in chunks[:3]:
            content = c.get("content", c.get("text", ""))[:300]
            title = c.get("document_title", c.get("source", ""))
            if not content:
                continue
            header = f"[{title}]" if title else "[chunk]"
            chunk_lines.append(f"{header}\n{content}")
        if chunk_lines:
            sections.append("Source Knowledge:\n" + "\n---\n".join(chunk_lines))

    # ── Format graph nodes — categorize by behavioral implications ──
    knowledge_lines = []
    question_lines = []
    caution_lines = []
    decision_lines = []

    for n in nodes[:k]:
        label = n.get("label", "")
        ntype = n.get("node_type", n.get("type", ""))
        conf = n.get("confidence", _get_prop(n, "confidence", ""))
        if not label:
            continue

        line = f"  - [{ntype}] {label}"
        if conf:
            line += f" (confidence: {conf})"
            try:
                if float(conf) < 0.5:
                    caution_lines.append(f"  - [{conf}] {label}")
                    continue
            except (ValueError, TypeError):
                pass

        if ntype in ("Question", "OpenQuestion", "Hypothesis", "Anomaly"):
            question_lines.append(f"  - {label}")
        elif ntype in ("Decision", "Option"):
            decision_lines.append(f"  - {label}")
        else:
            knowledge_lines.append(line)

    if knowledge_lines:
        sections.append("Related Graph Knowledge:\n" + "\n".join(knowledge_lines[:8]))
    if question_lines:
        sections.append(
            "Open Questions (investigate if relevant to your response):\n"
            + "\n".join(question_lines[:4])
        )
    if caution_lines:
        sections.append(
            "Weak Claims (flag uncertainty, don't assert as fact):\n"
            + "\n".join(caution_lines[:3])
        )
    if decision_lines:
        sections.append(
            "Open Decisions (help resolve if relevant):\n"
            + "\n".join(decision_lines[:3])
        )

    # ── Format edges (relationships between nodes) ──
    if edges:
        edge_lines = []
        for e in edges[:5]:
            etype = e.get("edge_type", e.get("type", ""))
            src = e.get("source_label", e.get("from_label", ""))
            tgt = e.get("target_label", e.get("to_label", ""))
            if src and tgt and etype:
                edge_lines.append(f"  - {src} —[{etype}]→ {tgt}")
        if edge_lines:
            sections.append("Relationships:\n" + "\n".join(edge_lines))

    if not sections:
        proactive_metrics.record(hit=False, latency_ms=latency_ms, top_score=0)
        return None

    proactive_metrics.record(hit=True, latency_ms=latency_ms, top_score=0)
    logger.info("MindGraph proactive: HIT (%.0fms, %d nodes, %d edges, %d chunks)",
                latency_ms, len(nodes), len(edges), len(chunks))

    return (
        "# MindGraph Context (Topic-Relevant)\n\n"
        "The following was proactively retrieved from semantic graph memory "
        "based on this message's topic. Use it to inform your response.\n\n"
        + "\n\n".join(sections)
    )


# ---------------------------------------------------------------------------
# Tool: mindgraph_ingest
# ---------------------------------------------------------------------------

def mindgraph_ingest(
    content: str,
    source: str = "",
    content_type: str = "text",
) -> str:
    """Ingest long-form content into the graph.

    Handles embedding, entity extraction, and linking automatically.
    For short content (<500 chars), uses sync ingest_chunk.
    For longer content, uses async ingest_document (returns job_id).

    content_type: text, conversation, document, code
    """
    if not content or not content.strip():
        return _json_response(False, error="Content is required")

    if len(content) < 500:
        # Sync ingestion for short content
        result, err = _safe_call(
            lambda c: c.ingest_chunk(content=content),
        )
        if err:
            return _json_response(False, error=f"Ingest failed: {err}")
        return _json_response(True, data={
            "result": result,
            "method": "sync_chunk",
            "message": "Content ingested synchronously",
        })
    else:
        # Async ingestion for longer content
        kwargs = {}
        if source:
            kwargs["source_uri"] = source
        if content_type:
            kwargs["content_type"] = content_type
        result, err = _safe_call(
            lambda c: c.ingest_document(content=content, **kwargs),
        )
        if err:
            return _json_response(False, error=f"Ingest failed: {err}")
        job_id = result.get("job_id", "") if isinstance(result, dict) else ""
        return _json_response(True, data={
            "result": result,
            "job_id": job_id,
            "method": "async_document",
            "message": f"Content submitted for async ingestion{f' (job: {job_id})' if job_id else ''}",
        })


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI format)
# ---------------------------------------------------------------------------

MINDGRAPH_SESSION_SCHEMA = {
    "name": "mindgraph_session",
    "description": (
        "Open or close a MindGraph semantic memory session. Sessions track "
        "what happens during a conversation and enable distillation at the end. "
        "Open at conversation start, close with a summary at the end."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "close"],
                "description": "Whether to open or close the session.",
            },
            "summary": {
                "type": "string",
                "description": "(Close only) Summary of what happened in the session for distillation.",
            },
            "label": {
                "type": "string",
                "description": "(Open only) Session label for identification.",
            },
        },
        "required": ["action"],
    },
}

MINDGRAPH_JOURNAL_SCHEMA = {
    "name": "mindgraph_journal",
    "description": (
        "Capture observations, notes, preferences, and reflections into semantic memory. "
        "Use this for low-friction knowledge capture — things worth remembering but not "
        "formal enough for argue() or commit(). Examples: user preferences, environmental "
        "observations, interesting patterns, session insights."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": "The content to capture.",
            },
            "entry_type": {
                "type": "string",
                "enum": ["observation", "preference", "note", "reflection", "insight"],
                "description": "Type of journal entry. Default: observation.",
            },
        },
        "required": ["entry"],
    },
}

MINDGRAPH_ARGUE_SCHEMA = {
    "name": "mindgraph_argue",
    "description": (
        "Record a structured epistemic claim with optional evidence and warrant. "
        "Use for beliefs, conclusions, and reasoned positions that should be tracked "
        "and potentially challenged later. Claims have confidence scores that can "
        "strengthen or weaken over time as evidence accumulates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "The claim or belief to record.",
            },
            "evidence": {
                "type": "string",
                "description": "Supporting evidence for the claim.",
            },
            "warrant": {
                "type": "string",
                "description": "The reasoning connecting evidence to claim.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level 0.0-1.0. Default: 0.7.",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["claim"],
    },
}

MINDGRAPH_COMMIT_SCHEMA = {
    "name": "mindgraph_commit",
    "description": (
        "Record or update goals, projects, and milestones. Provide label to create "
        "or deduplicate; provide uid to update an existing commitment. At least one "
        "of label or uid is required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Name of the goal, project, or milestone. Required for new commitments.",
            },
            "commit_type": {
                "type": "string",
                "enum": ["goal", "project", "milestone"],
                "description": "Type of commitment. Default: goal.",
            },
            "status": {
                "type": "string",
                "enum": ["active", "completed", "paused", "abandoned"],
                "description": "Current status. Default: active.",
            },
            "description": {
                "type": "string",
                "description": "Detailed description or context.",
            },
            "uid": {
                "type": "string",
                "description": "UID of existing commitment to update. When provided, label is optional. Get UIDs from mindgraph_retrieve(mode='goals').",
            },
        },
        "required": [],
    },
}

MINDGRAPH_RETRIEVE_SCHEMA = {
    "name": "mindgraph_retrieve",
    "description": (
        "Search the semantic graph memory. Uses keyword-based full-text search (FTS), "
        "not semantic search — use names and key terms as queries, not natural-language "
        "questions. Basic context is auto-injected each turn; use this for deeper queries.\n"
        "Modes: search (FTS, default), goals, questions, decisions, neighborhood (by UID), "
        "weak_claims, contradictions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search for (e.g. 'Alice Smith AI'), or node UID for neighborhood mode. Use key terms, not questions.",
            },
            "mode": {
                "type": "string",
                "enum": ["search", "goals", "questions", "decisions", "neighborhood", "weak_claims", "contradictions"],
                "description": "Retrieval mode. Default: search (FTS).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default: 10.",
                "minimum": 1,
                "maximum": 50,
            },
            "node_type": {
                "type": "string",
                "description": "(search mode only) Filter results by node type (e.g. 'Person', 'Goal', 'Project', 'Task', 'Claim').",
            },
            "include_chunks": {
                "type": "boolean",
                "description": "(search mode) Include provenance source text chunks. Default: true.",
            },
            "include_graph": {
                "type": "boolean",
                "description": "(search mode) Include connecting edges between result nodes. Default: true.",
            },
        },
        "required": [],
    },
}

MINDGRAPH_INGEST_SCHEMA = {
    "name": "mindgraph_ingest",
    "description": (
        "Ingest long-form content into semantic memory. Server-side processing extracts "
        "entities, chunks text, and links to the graph. Short content (<500 chars) is "
        "processed synchronously; longer content is queued for async processing. Use for: "
        "articles, documentation, conversation transcripts, code, research papers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to ingest.",
            },
            "source": {
                "type": "string",
                "description": "Source attribution (URL, filename, etc.).",
            },
            "content_type": {
                "type": "string",
                "enum": ["text", "conversation", "document", "code"],
                "description": "Type of content being ingested. Default: text.",
            },
        },
        "required": ["content"],
    },
}

MINDGRAPH_CAPTURE_SCHEMA = {
    "name": "mindgraph_capture",
    "description": (
        "Capture facts into the knowledge graph.\n"
        "- entity: Concrete things (people, orgs, concepts, places, events, works) in the Reality layer. "
        "Uses deduplication — safe to call repeatedly for the same entity.\n"
        "- observation: Factual observations about the world in the Reality layer. "
        "Pass entity_uid to link the observation to an entity (recommended).\n"
        "- concept: Abstract concepts in the Epistemic layer (theories, frameworks, patterns)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Name or description.",
            },
            "capture_type": {
                "type": "string",
                "enum": ["entity", "observation", "concept"],
                "description": "What to capture. Default: entity.",
            },
            "entity_type": {
                "type": "string",
                "enum": ["concept", "person", "organization", "nation", "place", "event", "work", "other"],
                "description": "(entity only) Type of entity. Default: concept.",
            },
            "properties": {
                "type": "object",
                "description": "Additional properties (e.g. domain, author, description).",
            },
            "entity_uid": {
                "type": "string",
                "description": "(observation only) UID of entity to link this observation to. Creates a HAS_OBSERVATION edge. Get entity UIDs from prior capture() or retrieve() calls.",
            },
        },
        "required": ["label"],
    },
}

MINDGRAPH_INQUIRE_SCHEMA = {
    "name": "mindgraph_inquire",
    "description": (
        "Record questions, hypotheses, or anomalies in the Epistemic layer.\n"
        "- question: An open question worth tracking (shows up at session start).\n"
        "- hypothesis: A testable conjecture or proposed explanation.\n"
        "- anomaly: Something that doesn't fit existing models — worth investigating."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "The question, hypothesis, or anomaly to record.",
            },
            "inquiry_type": {
                "type": "string",
                "enum": ["question", "hypothesis", "anomaly"],
                "description": "Type of inquiry. Default: question.",
            },
        },
        "required": ["label"],
    },
}

MINDGRAPH_ACTION_SCHEMA = {
    "name": "mindgraph_action",
    "description": (
        "Record risks and affordances in the Action layer.\n"
        "- assess_risk: A risk, threat, or potential failure mode to track.\n"
        "- add_affordance: An action enabled by current state — a capability, leverage point, "
        "or opportunity that exists and could be exercised."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["assess_risk", "add_affordance"],
                "description": "Action type. Default: assess_risk.",
            },
            "label": {
                "type": "string",
                "description": "Short name for the risk or affordance.",
            },
            "description": {
                "type": "string",
                "description": "Detailed description.",
            },
        },
        "required": ["label"],
    },
}

MINDGRAPH_DECIDE_SCHEMA = {
    "name": "mindgraph_decide",
    "description": (
        "Manage decisions in the Intent layer. Open a decision when a choice point is identified, "
        "add options as they are discussed, and resolve when a decision is made. Decisions track "
        "the deliberation process and show up at session start as 'Open Decisions'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "add_option", "add_constraint", "resolve"],
                "description": "Action: open, add_option, add_constraint, or resolve.",
            },
            "label": {
                "type": "string",
                "description": "(open) Name of the decision.",
            },
            "decision_uid": {
                "type": "string",
                "description": "(add_option, resolve) UID of the decision.",
            },
            "option_label": {
                "type": "string",
                "description": "(add_option) Label for the option.",
            },
            "chosen_option_uid": {
                "type": "string",
                "description": "(resolve) UID of the chosen option.",
            },
            "summary": {
                "type": "string",
                "description": "Optional summary or rationale.",
            },
        },
        "required": ["action"],
    },
}

MINDGRAPH_PLAN_SCHEMA = {
    "name": "mindgraph_plan",
    "description": (
        "Manage plans, tasks, execution tracking, and governance in the Agent layer.\n\n"
        "Planning: create_plan, create_task, add_step, update_status, get_plan.\n"
        "Execution: start (begin work), complete (done), fail (failed).\n"
        "Governance: create_policy (rules/constraints the agent should follow).\n\n"
        "Use this to track your own work — create tasks for multi-step operations, "
        "mark them complete/failed, and maintain provenance of what was done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_plan", "create_task", "add_step", "update_status",
                         "get_plan", "start", "complete", "fail", "create_policy"],
                "description": "Action to perform.",
            },
            "label": {
                "type": "string",
                "description": "Name/description for the plan, task, step, or policy.",
            },
            "plan_uid": {
                "type": "string",
                "description": "(add_step, get_plan) UID of the plan.",
            },
            "task_uid": {
                "type": "string",
                "description": "(update_status, start) UID of the task.",
            },
            "execution_uid": {
                "type": "string",
                "description": "(complete, fail) UID of the execution to finalize.",
            },
            "status": {
                "type": "string",
                "description": "(update_status) New status value.",
            },
            "description": {
                "type": "string",
                "description": "Detailed description or context.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# TOOLS list for plugin registration
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "mindgraph_session",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_SESSION_SCHEMA,
        "handler": lambda args, **kw: mindgraph_session(
            action=args.get("action", "open"),
            summary=args.get("summary", ""),
            label=args.get("label", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "🧠",
    },
    {
        "name": "mindgraph_journal",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_JOURNAL_SCHEMA,
        "handler": lambda args, **kw: mindgraph_journal(
            entry=args.get("entry", ""),
            entry_type=args.get("entry_type", "observation"),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "📓",
    },
    {
        "name": "mindgraph_argue",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_ARGUE_SCHEMA,
        "handler": lambda args, **kw: mindgraph_argue(
            claim=args.get("claim", ""),
            evidence=args.get("evidence", ""),
            warrant=args.get("warrant", ""),
            confidence=args.get("confidence", 0.7),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "⚖️",
    },
    {
        "name": "mindgraph_commit",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_COMMIT_SCHEMA,
        "handler": lambda args, **kw: mindgraph_commit(
            label=args.get("label", ""),
            commit_type=args.get("commit_type", "goal"),
            status=args.get("status", "active"),
            description=args.get("description", ""),
            uid=args.get("uid", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "🎯",
    },
    {
        "name": "mindgraph_retrieve",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_RETRIEVE_SCHEMA,
        "handler": lambda args, **kw: mindgraph_retrieve(
            query=args.get("query", ""),
            mode=args.get("mode", "search"),
            limit=args.get("limit", 10),
            include_chunks=args.get("include_chunks", True),
            include_graph=args.get("include_graph", True),
            node_type=args.get("node_type", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "🔮",
    },
    {
        "name": "mindgraph_ingest",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_INGEST_SCHEMA,
        "handler": lambda args, **kw: mindgraph_ingest(
            content=args.get("content", ""),
            source=args.get("source", ""),
            content_type=args.get("content_type", "text"),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "📥",
    },
    {
        "name": "mindgraph_capture",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_CAPTURE_SCHEMA,
        "handler": lambda args, **kw: mindgraph_capture(
            label=args.get("label", ""),
            capture_type=args.get("capture_type", "entity"),
            entity_type=args.get("entity_type", "concept"),
            properties=args.get("properties"),
            entity_uid=args.get("entity_uid", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "🏷️",
    },
    {
        "name": "mindgraph_inquire",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_INQUIRE_SCHEMA,
        "handler": lambda args, **kw: mindgraph_inquire(
            label=args.get("label", ""),
            inquiry_type=args.get("inquiry_type", "question"),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "❓",
    },
    {
        "name": "mindgraph_action",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_ACTION_SCHEMA,
        "handler": lambda args, **kw: mindgraph_action(
            action=args.get("action", "assess_risk"),
            label=args.get("label", ""),
            description=args.get("description", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "⚠️",
    },
    {
        "name": "mindgraph_decide",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_DECIDE_SCHEMA,
        "handler": lambda args, **kw: mindgraph_decide(
            action=args.get("action", "open"),
            label=args.get("label", ""),
            decision_uid=args.get("decision_uid", ""),
            option_label=args.get("option_label", ""),
            chosen_option_uid=args.get("chosen_option_uid", ""),
            summary=args.get("summary", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "⚖️",
    },
    {
        "name": "mindgraph_plan",
        "toolset": "mindgraph",
        "schema": MINDGRAPH_PLAN_SCHEMA,
        "handler": lambda args, **kw: mindgraph_plan(
            action=args.get("action", "create_task"),
            label=args.get("label", ""),
            plan_uid=args.get("plan_uid", ""),
            task_uid=args.get("task_uid", ""),
            execution_uid=args.get("execution_uid", ""),
            status=args.get("status", ""),
            description=args.get("description", ""),
        ),
        "check_fn": check_requirements,
        "requires_env": ["MINDGRAPH_API_KEY"],
        "emoji": "📋",
    },
]
