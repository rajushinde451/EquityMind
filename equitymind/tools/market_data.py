"""Quantitative market-data tools for EquityMind (backed by yfinance).

Design principles
-----------------
* Every number the agent reports must come from these tools. Missing fields are
  returned as the literal string ``"Data unavailable"`` so the model never has
  to guess.
* Tools return a uniform envelope: ``{"status": "success" | "error", ...}``.
* Results are cached in-process (TTL, see ``EQUITYMIND_CACHE_TTL_S``) and also
  snapshotted into session state so downstream tools (``score_recommendation``)
  can consume them deterministically without re-fetching.
* Each provider call is wrapped in an OpenTelemetry span.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import pandas as pd
import yfinance as yf
from google.adk.tools.tool_context import ToolContext
from opentelemetry import trace

from ..config import settings

UNAVAILABLE = "Data unavailable"
DATA_SOURCE = "Yahoo Finance via yfinance"

logger = logging.getLogger("equitymind.tools.market_data")
tracer = trace.get_tracer("equitymind.tools")


# --------------------------------------------------------------------------- #
# State keys (shared contract with scoring.py)
# --------------------------------------------------------------------------- #
def fundamentals_key(symbol: str) -> str:
    return f"quant:{symbol}:fundamentals"


def technicals_key(symbol: str) -> str:
    return f"quant:{symbol}:technicals"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _TTLCache:
    """Tiny thread-safe TTL cache so repeated questions don't hammer the API."""

    def __init__(self, ttl_s: int) -> None:
        self.ttl_s = ttl_s
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            hit = self._data.get(key)
            if hit and time.monotonic() - hit[0] < self.ttl_s:
                return hit[1]
            self._data.pop(key, None)
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache = _TTLCache(settings.market_data_cache_ttl_s)


def _clean(value: Any, digits: int = 2) -> Any:
    """Round numbers, map None/NaN/inf to UNAVAILABLE."""
    if value is None:
        return UNAVAILABLE
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f) or math.isinf(f):
        return UNAVAILABLE
    return round(f, digits)


def _pct(info: dict[str, Any], key: str) -> Any:
    v = info.get(key)
    return _clean(v * 100) if isinstance(v, (int, float)) else UNAVAILABLE


def _normalize(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _error(symbol: str, message: str) -> dict[str, Any]:
    return {"status": "error", "ticker": symbol, "error_message": message}


def _not_recognized(symbol: str) -> dict[str, Any]:
    return _error(
        symbol,
        f"Ticker '{symbol}' was not recognized by the market-data provider. "
        "Verify the symbol (non-US listings may need an exchange suffix, "
        "e.g. '.NS', '.L').",
    )


def _fetch(symbol: str, kind: str, loader) -> tuple[Any, bool]:
    """Load data through the cache inside an OTel span. Returns (data, cache_hit)."""
    key = f"{kind}:{symbol}"
    with tracer.start_as_current_span(f"market_data.{kind}") as span:
        span.set_attribute("equitymind.ticker", symbol)
        cached = _cache.get(key)
        span.set_attribute("equitymind.cache_hit", cached is not None)
        if cached is not None:
            return cached, True
        started = time.perf_counter()
        data = loader()
        span.set_attribute("equitymind.provider_latency_ms", (time.perf_counter() - started) * 1000)
        _cache.set(key, data)
        return data, False


def _eps_trend(ticker: yf.Ticker) -> list[dict[str, Any]] | str:
    """Return up to the last 4 quarters of diluted (or basic) EPS, oldest first."""
    try:
        stmt = ticker.quarterly_income_stmt
        if stmt is None or stmt.empty:
            return UNAVAILABLE
        for row in ("Diluted EPS", "Basic EPS"):
            if row in stmt.index:
                series = stmt.loc[row].dropna().sort_index().tail(4)
                if series.empty:
                    continue
                return [{"quarter_end": str(idx.date()), "eps": _clean(val)} for idx, val in series.items()]
    except Exception:  # noqa: BLE001 - yfinance raises many error types
        logger.debug("EPS trend unavailable", exc_info=True)
    return UNAVAILABLE


def _snapshot(tool_context: ToolContext | None, key: str, payload: dict[str, Any]) -> None:
    if tool_context is not None and payload.get("status") == "success":
        tool_context.state[key] = payload
        tool_context.state["last_ticker"] = payload["ticker"]


# --------------------------------------------------------------------------- #
# Tool 1: fundamentals
# --------------------------------------------------------------------------- #
def get_stock_fundamentals(ticker: str, tool_context: ToolContext = None) -> dict[str, Any]:
    """Fetch fundamental metrics for a stock ticker (QUANTITATIVE LOOKUP, part 1).

    Returns price, trailing/forward P/E, dividend yield, debt-to-equity, EPS
    (trailing, forward and the last four quarters), revenue/earnings growth,
    profit margin and Wall Street consensus.

    Args:
        ticker: The stock ticker symbol, e.g. "AAPL" or "RELIANCE.NS".

    Returns:
        dict with "status" ("success" or "error"). On success it contains the
        metrics; any metric the provider did not return is the string
        "Data unavailable". On error it contains "error_message" explaining that
        the ticker could not be verified — do NOT produce a scorecard then.
    """
    symbol = _normalize(ticker)
    if not symbol:
        return _error(symbol, "No ticker symbol provided.")

    def load() -> dict[str, Any]:
        t = yf.Ticker(symbol)
        info = t.info or {}
        hist = t.history(period="5d", interval="1d", auto_adjust=False)
        return {"info": info, "eps_trend": _eps_trend(t), "last_close": _last_close(hist)}

    try:
        raw, cache_hit = _fetch(symbol, "fundamentals", load)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fundamentals fetch failed", extra={"ticker": symbol, "error": str(exc)})
        return _error(symbol, f"Failed to retrieve data for '{symbol}': {exc}")

    info: dict[str, Any] = raw["info"]
    price = info.get("currentPrice") or info.get("regularMarketPrice") or raw["last_close"]
    if price is None and not info.get("longName") and not info.get("shortName"):
        return _not_recognized(symbol)

    # Dividend yield from annual rate / price, because yfinance's own
    # `dividendYield` field has changed units across versions.
    dividend_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    if dividend_rate and price:
        dividend_yield_pct: Any = _clean(float(dividend_rate) / float(price) * 100)
    elif price:
        dividend_yield_pct = 0.0  # No dividend reported -> company does not pay one.
    else:
        dividend_yield_pct = UNAVAILABLE

    # yfinance reports debt-to-equity as a percentage (150.0 == 1.5x).
    de_raw = info.get("debtToEquity")
    de_ratio = _clean(float(de_raw) / 100) if isinstance(de_raw, (int, float)) else UNAVAILABLE

    result = {
        "status": "success",
        "ticker": symbol,
        "company_name": info.get("longName") or info.get("shortName") or UNAVAILABLE,
        "currency": info.get("currency") or UNAVAILABLE,
        "sector": info.get("sector") or UNAVAILABLE,
        "industry": info.get("industry") or UNAVAILABLE,
        "current_price": _clean(price),
        "market_cap": _clean(info.get("marketCap"), 0),
        "trailing_pe": _clean(info.get("trailingPE")),
        "forward_pe": _clean(info.get("forwardPE")),
        "dividend_yield_percent": dividend_yield_pct,
        "debt_to_equity_ratio": de_ratio,
        "trailing_eps": _clean(info.get("trailingEps")),
        "forward_eps": _clean(info.get("forwardEps")),
        "quarterly_eps_trend": raw["eps_trend"],
        "revenue_growth_yoy_percent": _pct(info, "revenueGrowth"),
        "earnings_growth_yoy_percent": _pct(info, "earningsGrowth"),
        "profit_margin_percent": _pct(info, "profitMargins"),
        "analyst_consensus": info.get("recommendationKey") or UNAVAILABLE,
        "analyst_target_mean_price": _clean(info.get("targetMeanPrice")),
        "data_source": DATA_SOURCE,
        "cache_hit": cache_hit,
    }
    _snapshot(tool_context, fundamentals_key(symbol), result)
    return result


# --------------------------------------------------------------------------- #
# Tool 2: technicals
# --------------------------------------------------------------------------- #
def _last_close(hist: pd.DataFrame | None) -> float | None:
    if hist is None or hist.empty or "Close" not in hist:
        return None
    closes = hist["Close"].dropna()
    return float(closes.iloc[-1]) if not closes.empty else None


def _rsi(closes: pd.Series, period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = float(gain.iloc[-1]), float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    return 100 - 100 / (1 + last_gain / last_loss)


def _return_pct(closes: pd.Series, days: int) -> Any:
    if len(closes) <= days:
        return UNAVAILABLE
    return _clean((closes.iloc[-1] / closes.iloc[-1 - days] - 1) * 100)


def compute_technicals(symbol: str, closes: pd.Series) -> dict[str, Any]:
    """Pure function: derive indicators from a series of daily closes."""
    closes = closes.dropna().astype(float)
    price = float(closes.iloc[-1])
    sma_50 = _clean(closes.tail(50).mean()) if len(closes) >= 50 else UNAVAILABLE
    sma_200 = _clean(closes.tail(200).mean()) if len(closes) >= 200 else UNAVAILABLE

    def vs(ma: Any) -> str:
        return UNAVAILABLE if not isinstance(ma, float) else ("above" if price > ma else "below")

    macd_hist: Any = UNAVAILABLE
    if len(closes) >= 35:
        macd = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
        macd_hist = _clean((macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1], 3)

    cross: Any = UNAVAILABLE
    if isinstance(sma_50, float) and isinstance(sma_200, float):
        cross = "golden_cross (50d > 200d)" if sma_50 > sma_200 else "death_cross (50d < 200d)"

    daily = closes.pct_change().dropna()
    vol = _clean(daily.tail(252).std() * math.sqrt(252) * 100) if len(daily) > 20 else UNAVAILABLE
    window = closes.tail(252)

    return {
        "status": "success",
        "ticker": symbol,
        "last_close": _clean(price),
        "sma_50_day": sma_50,
        "sma_200_day": sma_200,
        "price_vs_sma_50": vs(sma_50),
        "price_vs_sma_200": vs(sma_200),
        "ma_crossover": cross,
        "rsi_14": _clean(_rsi(closes)),
        "macd_histogram": macd_hist,
        "return_1m_percent": _return_pct(closes, 21),
        "return_3m_percent": _return_pct(closes, 63),
        "return_6m_percent": _return_pct(closes, 126),
        "annualized_volatility_percent": vol,
        "fifty_two_week_high": _clean(window.max()),
        "fifty_two_week_low": _clean(window.min()),
        "observations": len(closes),
        "data_source": DATA_SOURCE,
    }


def get_technical_indicators(ticker: str, tool_context: ToolContext = None) -> dict[str, Any]:
    """Compute technical indicators from one year of daily prices (QUANTITATIVE LOOKUP, part 2).

    Returns 50/200-day simple moving averages and price position relative to
    them, golden/death cross, RSI(14), MACD histogram, 1/3/6-month returns,
    annualized volatility and the 52-week range.

    Args:
        ticker: The stock ticker symbol, e.g. "AAPL".

    Returns:
        dict with "status" ("success" or "error") and the indicators. Indicators
        that cannot be computed (insufficient history) are "Data unavailable".
    """
    symbol = _normalize(ticker)
    if not symbol:
        return _error(symbol, "No ticker symbol provided.")

    def load() -> pd.Series:
        hist = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            return pd.Series(dtype=float)
        return hist["Close"]

    try:
        closes, cache_hit = _fetch(symbol, "technicals", load)
    except Exception as exc:  # noqa: BLE001
        logger.warning("technicals fetch failed", extra={"ticker": symbol, "error": str(exc)})
        return _error(symbol, f"Failed to retrieve price history for '{symbol}': {exc}")

    if closes is None or closes.dropna().empty:
        return _not_recognized(symbol)

    result = compute_technicals(symbol, closes)
    result["cache_hit"] = cache_hit
    _snapshot(tool_context, technicals_key(symbol), result)
    return result
