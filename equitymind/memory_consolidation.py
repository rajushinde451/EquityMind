"""Asynchronous, background memory consolidation.

After every root-agent turn, ``schedule_consolidation`` (an ``after_agent_callback``)
returns immediately and spawns a background ``asyncio`` task that:

1. **Ingests the session into the ADK MemoryService**
   (``add_session_to_memory``). Locally this is ``InMemoryMemoryService``; in
   production point ``EQUITYMIND_MEMORY_URI`` at Vertex AI Memory Bank
   (``agentengine://<id>``), where ingestion does LLM-based fact extraction,
   which is slow, so it must never block the user's response.
2. **Consolidates the analysis history** into a compact per-ticker digest
   (call count, recommendation trajectory, price drift since first call,
   average score). The CPU work runs in a worker thread
   (``asyncio.to_thread``).

The digest is persisted with a **write-behind** pattern. Background tasks
cannot safely mutate session state (the runner owns event ordering), so
results are staged in ``_pending`` and applied by ``apply_pending_digest``
(a ``before_agent_callback``) at the start of the user's next turn, where
ADK records it as a normal state delta.

The prompt then injects the digest instead of raw history (context compaction),
and ``recall_past_conversations`` lets the agent semantically search older
conversations stored in the MemoryService.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from collections import defaultdict
from typing import Any

from google.adk.tools.tool_context import ToolContext

from .tools.memory import HISTORY_KEY

DIGEST_KEY = "user:memory_digest"
logger = logging.getLogger("equitymind.memory")

_pending: dict[str, dict[str, Any]] = {}
_tasks: set[asyncio.Task] = set()


# --------------------------------------------------------------------------- #
# Pure consolidation logic
# --------------------------------------------------------------------------- #
def build_digest(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a raw analysis history into a per-ticker digest."""
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in history:
        if h.get("ticker"):
            by_ticker[h["ticker"]].append(h)

    tickers: dict[str, Any] = {}
    for symbol, items in by_ticker.items():
        recs = [i.get("recommendation") for i in items if i.get("recommendation")]
        trajectory: list[str] = []
        for r in recs:  # compress repeats: BUY,BUY,HOLD -> BUY→HOLD
            if not trajectory or trajectory[-1] != r:
                trajectory.append(r)
        first_px, last_px = items[0].get("price"), items[-1].get("price")
        drift = (
            round((last_px / first_px - 1) * 100, 2)
            if isinstance(first_px, (int, float)) and isinstance(last_px, (int, float)) and first_px
            else None
        )
        scores = [i["composite_score"] for i in items if isinstance(i.get("composite_score"), (int, float))]
        tickers[symbol] = {
            "analyses": len(items),
            "first_date": items[0].get("date"),
            "last_date": items[-1].get("date"),
            "trajectory": "→".join(trajectory),
            "latest_recommendation": recs[-1] if recs else None,
            "latest_price": last_px,
            "currency": items[-1].get("currency"),
            "price_drift_since_first_pct": drift,
            "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
        }

    counts = defaultdict(int)
    for t in tickers.values():
        if t["latest_recommendation"]:
            counts[t["latest_recommendation"]] += 1
    return {
        "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "total_analyses": len(history),
        "tickers": tickers,
        "stance_counts": dict(counts),
    }


def render_digest(digest: dict[str, Any] | None, limit: int = 8) -> list[str]:
    """Compact prompt lines for the digest (most recently analysed first)."""
    if not digest or not digest.get("tickers"):
        return []
    items = sorted(digest["tickers"].items(), key=lambda kv: kv[1].get("last_date") or "", reverse=True)
    lines = [f"- Consolidated coverage ({digest.get('total_analyses', 0)} analyses):"]
    for symbol, t in items[:limit]:
        drift = t.get("price_drift_since_first_pct")
        lines.append(
            f"  - {symbol}: {t['analyses']}x, calls {t['trajectory']}, last {t.get('latest_price')} "
            f"{t.get('currency') or ''} on {t.get('last_date')}"
            + (f", {drift:+.1f}% since first call" if drift is not None else "")
        )
    return lines


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
async def _consolidate(ctx: Any, user_id: str, history: list[dict[str, Any]]) -> None:
    started = time.perf_counter()
    ingested = False
    try:
        await ctx.add_session_to_memory()
        ingested = True
    except Exception as exc:  # noqa: BLE001 - memory service may be absent
        logger.debug("memory ingestion skipped: %s", exc)
    try:
        digest = await asyncio.to_thread(build_digest, history)
        _pending[user_id] = digest
    except Exception:  # noqa: BLE001
        logger.exception("memory consolidation failed")
        return
    logger.info(
        "memory_consolidated",
        extra={
            "event": "memory_consolidated",
            "phase": "background",
            "user_id": user_id,
            "session_ingested": ingested,
            "tickers": len(digest["tickers"]),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )


def schedule_consolidation(callback_context: Any) -> None:
    """after_agent_callback: fire-and-forget background consolidation."""
    user_id = getattr(callback_context, "user_id", None) or "anonymous"
    history = list(callback_context.state.get(HISTORY_KEY) or [])
    try:
        task = asyncio.get_running_loop().create_task(_consolidate(callback_context, user_id, history))
    except RuntimeError:  # no running loop (sync test context)
        return None
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return None


def apply_pending_digest(callback_context: Any) -> None:
    """before_agent_callback: write-behind — persist the latest digest as a state delta."""
    user_id = getattr(callback_context, "user_id", None) or "anonymous"
    digest = _pending.pop(user_id, None)
    if digest is not None:
        callback_context.state[DIGEST_KEY] = digest
    return None


async def drain(timeout: float = 10.0) -> None:
    """Wait for in-flight consolidation tasks (graceful shutdown / tests)."""
    if _tasks:
        await asyncio.wait(list(_tasks), timeout=timeout)


# --------------------------------------------------------------------------- #
# Tool: semantic recall over past conversations
# --------------------------------------------------------------------------- #
async def recall_past_conversations(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Search the user's past conversations (long-term memory) for relevant context.

    Use when the user refers to something discussed in an earlier session that
    is not in the USER CONTEXT block, e.g. "what was that risk you flagged on
    Tesla last month?".

    Args:
        query: Natural-language search query, e.g. "Tesla regulatory risk".

    Returns:
        dict with status and up to 5 matching memory snippets (author, text).
    """
    try:
        response = await tool_context.search_memory(query)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_message": f"Long-term conversational memory unavailable: {exc}"}

    snippets = []
    for mem in (getattr(response, "memories", None) or [])[:5]:
        parts = getattr(getattr(mem, "content", None), "parts", None) or []
        text = " ".join(p.text for p in parts if getattr(p, "text", None)).strip()
        if text:
            snippets.append({"author": getattr(mem, "author", None), "text": text[:400]})
    return {"status": "success", "count": len(snippets), "memories": snippets}
