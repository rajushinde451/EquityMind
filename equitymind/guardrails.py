"""Policy guardrails implemented as ADK callbacks (deterministic, not prompt-only).

* ``validate_ticker_args`` – before any tool runs, normalises the ``ticker``
  argument and rejects malformed symbols without hitting the network.
* ``enforce_disclaimer`` – after the root model produces its final text,
  guarantees the mandatory disclaimer footer is present.
"""

from __future__ import annotations

import re
from typing import Any

from google.genai import types

from .config import DISCLAIMER

# Letters/digits plus the separators used by Yahoo symbols: BRK-B, RELIANCE.NS, ^GSPC, EURUSD=X
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")
_DISCLAIMER_MARKER = "not a certified financial advisor"


def validate_ticker_args(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """Normalise/validate `ticker` in-place; short-circuit the tool on bad input."""
    if "ticker" not in args:
        return None
    raw = args.get("ticker")
    if not isinstance(raw, str):
        return {"status": "error", "error_message": "ticker must be a string."}
    symbol = raw.strip().upper().lstrip("$")
    if not symbol and tool.name == "manage_watchlist":
        return None  # 'list' action needs no ticker
    if not TICKER_RE.match(symbol):
        return {
            "status": "error",
            "error_message": (
                f"'{raw}' is not a valid ticker format. Ask the user to confirm the symbol "
                "or resolve the company name with market_sentiment_researcher."
            ),
        }
    args["ticker"] = symbol
    return None


def enforce_disclaimer(callback_context: Any, llm_response: Any) -> Any | None:
    """Append the disclaimer to any final (non-tool-call) text response missing it."""
    if getattr(llm_response, "partial", False):
        return None
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    if not parts or any(getattr(p, "function_call", None) for p in parts):
        return None
    text = "".join(p.text or "" for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    if not text.strip() or _DISCLAIMER_MARKER in text.lower():
        return None
    parts.append(types.Part(text=f"\n\n{DISCLAIMER}"))
    llm_response.content = types.Content(role=content.role or "model", parts=parts)
    return llm_response
