"""Long-term, user-scoped memory tools.

ADK state prefixes give us three memory tiers for free:

* ``user:*``  – persists across ALL sessions of the same user (with a
  persistent SessionService such as the SQLite/Postgres DatabaseSessionService
  configured in ``server.py``). Used for risk profile, watchlist and history.
* (no prefix) – scoped to the current conversation/session, e.g. the
  ``quant:<TICKER>:*`` snapshots and ``last_ticker``.
* ``temp:*`` – scoped to a single invocation (used by observability).

The current memory is also injected into the system prompt on every turn (see
``prompt.build_root_instruction``) so the agent can personalise answers
("you asked about NVDA last week, it was a HOLD at $118").
"""

from __future__ import annotations

from typing import Any, Literal

from google.adk.tools.tool_context import ToolContext

RISK_PROFILE_KEY = "user:risk_profile"
WATCHLIST_KEY = "user:watchlist"
HISTORY_KEY = "user:analysis_history"
DEFAULT_RISK_PROFILE = "moderate"
MAX_WATCHLIST = 25


def set_risk_profile(
    risk_profile: Literal["conservative", "moderate", "aggressive"],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Save the user's investment risk profile. It changes how the scorecard weights signals.

    conservative = fundamentals-heavy and a higher bar to BUY; aggressive =
    momentum/technicals-heavy and a lower bar to BUY. Persists across sessions.

    Args:
        risk_profile: One of "conservative", "moderate" or "aggressive".

    Returns:
        dict with status and the saved profile.
    """
    if risk_profile not in ("conservative", "moderate", "aggressive"):
        return {"status": "error", "error_message": "risk_profile must be conservative, moderate or aggressive."}
    tool_context.state[RISK_PROFILE_KEY] = risk_profile
    return {"status": "success", "risk_profile": risk_profile}


def manage_watchlist(
    action: Literal["add", "remove", "list"],
    tool_context: ToolContext,
    ticker: str = "",
) -> dict[str, Any]:
    """Add, remove or list tickers on the user's persistent watchlist.

    Args:
        action: "add", "remove" or "list".
        ticker: Ticker symbol (required for add/remove).

    Returns:
        dict with status and the current watchlist.
    """
    watchlist: list[str] = list(tool_context.state.get(WATCHLIST_KEY, []))
    symbol = (ticker or "").strip().upper()

    if action in ("add", "remove") and not symbol:
        return {"status": "error", "error_message": f"A ticker is required to {action}."}
    if action == "add":
        if symbol not in watchlist:
            if len(watchlist) >= MAX_WATCHLIST:
                return {"status": "error", "error_message": f"Watchlist is full ({MAX_WATCHLIST})."}
            watchlist.append(symbol)
    elif action == "remove":
        if symbol not in watchlist:
            return {"status": "error", "error_message": f"{symbol} is not on the watchlist.", "watchlist": watchlist}
        watchlist.remove(symbol)
    elif action != "list":
        return {"status": "error", "error_message": "action must be add, remove or list."}

    tool_context.state[WATCHLIST_KEY] = watchlist
    return {"status": "success", "watchlist": watchlist}


def get_analysis_history(tool_context: ToolContext, ticker: str = "") -> dict[str, Any]:
    """Recall previous scorecards produced for this user (across sessions).

    Use when the user asks things like "what did you say about TSLA last time?"
    or "how has your view on AAPL changed?".

    Args:
        ticker: Optional ticker to filter by. Empty returns all recent analyses.

    Returns:
        dict with status and a list of past analyses (newest last).
    """
    history: list[dict[str, Any]] = list(tool_context.state.get(HISTORY_KEY, []))
    symbol = (ticker or "").strip().upper()
    if symbol:
        history = [h for h in history if h.get("ticker") == symbol]
    return {"status": "success", "count": len(history), "analyses": history}
