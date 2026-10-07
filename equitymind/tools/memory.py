"""Long-term, user-scoped memory tools (with human-in-the-loop confirmation).

ADK state prefixes give us three memory tiers for free:

* ``user:*``  – persists across ALL sessions of the same user (with a
  persistent SessionService such as the SQLite/Postgres DatabaseSessionService
  configured in ``server.py``). Used for risk profile, watchlist and history.
* (no prefix) – scoped to the current conversation/session, e.g. the
  ``quant:<TICKER>:*`` snapshots and ``last_ticker``.
* ``temp:*`` – scoped to a single invocation.

Human-in-the-loop
-----------------
Operations that change persistent user settings or destroy data pause
execution and request explicit human approval via ADK's tool-confirmation
protocol (``tool_context.request_confirmation``):

* changing an *existing* risk profile (it alters every future recommendation),
* removing a ticker from the watchlist,
* clearing the analysis history (irreversible).

ADK emits an ``adk_request_confirmation`` event; the ADK Web UI renders an
approve/reject prompt, and REST clients answer via ``POST /v1/confirm``.
The tool is then re-invoked with ``tool_context.tool_confirmation`` set.
"""

from __future__ import annotations

from typing import Any, Literal

from google.adk.tools.tool_context import ToolContext

RISK_PROFILE_KEY = "user:risk_profile"
WATCHLIST_KEY = "user:watchlist"
HISTORY_KEY = "user:analysis_history"
DEFAULT_RISK_PROFILE = "moderate"
MAX_WATCHLIST = 25


def _confirmation_gate(tool_context: ToolContext, hint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a pending/cancelled response, or None if approved and execution may proceed."""
    confirmation = getattr(tool_context, "tool_confirmation", None)
    if confirmation is None:
        tool_context.request_confirmation(hint=hint, payload=payload)
        return {
            "status": "pending_confirmation",
            "message": f"Awaiting user approval: {hint}",
        }
    if not confirmation.confirmed:
        return {"status": "cancelled", "message": "The user declined this action. Nothing was changed."}
    return None


def set_risk_profile(
    risk_profile: Literal["conservative", "moderate", "aggressive"],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Save the user's investment risk profile. It changes how the scorecard weights signals.

    conservative = fundamentals-heavy and a higher bar to BUY; aggressive =
    momentum/technicals-heavy and a lower bar to BUY. Persists across sessions.
    Changing an already-saved profile requires the user's explicit confirmation.

    Args:
        risk_profile: One of "conservative", "moderate" or "aggressive".

    Returns:
        dict with status ("success", "pending_confirmation", "cancelled" or "error").
    """
    if risk_profile not in ("conservative", "moderate", "aggressive"):
        return {"status": "error", "error_message": "risk_profile must be conservative, moderate or aggressive."}

    current = tool_context.state.get(RISK_PROFILE_KEY)
    if current and current != risk_profile:
        gate = _confirmation_gate(
            tool_context,
            hint=f"Change your saved risk profile from '{current}' to '{risk_profile}'? "
            "This changes how all future recommendations are scored.",
            payload={"from": current, "to": risk_profile},
        )
        if gate:
            return gate

    tool_context.state[RISK_PROFILE_KEY] = risk_profile
    return {"status": "success", "risk_profile": risk_profile, "previous": current}


def manage_watchlist(
    action: Literal["add", "remove", "list"],
    tool_context: ToolContext,
    ticker: str = "",
) -> dict[str, Any]:
    """Add, remove or list tickers on the user's persistent watchlist.

    Removing a ticker requires the user's explicit confirmation.

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
        gate = _confirmation_gate(
            tool_context, hint=f"Remove {symbol} from your watchlist?", payload={"ticker": symbol}
        )
        if gate:
            return {**gate, "watchlist": watchlist}
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


def clear_analysis_history(tool_context: ToolContext, ticker: str = "") -> dict[str, Any]:
    """Permanently delete the user's stored analysis history (all, or one ticker).

    This is irreversible and ALWAYS requires the user's explicit confirmation.

    Args:
        ticker: Optional ticker; if given only that ticker's analyses are deleted.

    Returns:
        dict with status ("success", "pending_confirmation", "cancelled") and how many entries were removed.
    """
    history: list[dict[str, Any]] = list(tool_context.state.get(HISTORY_KEY, []))
    symbol = (ticker or "").strip().upper()
    to_remove = [h for h in history if not symbol or h.get("ticker") == symbol]
    if not to_remove:
        return {"status": "success", "removed": 0, "message": "Nothing to delete."}

    scope = f"all {len(to_remove)} stored analyses for {symbol}" if symbol else f"ALL {len(to_remove)} stored analyses"
    gate = _confirmation_gate(
        tool_context,
        hint=f"Permanently delete {scope}? This cannot be undone.",
        payload={"ticker": symbol or None, "count": len(to_remove)},
    )
    if gate:
        return gate

    tool_context.state[HISTORY_KEY] = [h for h in history if h not in to_remove]
    return {"status": "success", "removed": len(to_remove)}
