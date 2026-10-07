"""Deterministic decision engine (DECISION SYNTHESIS).

The LLM is good at reading news; it is bad at consistent arithmetic. So the
final BUY / SELL / HOLD call is produced by a transparent, rule-based scorer
that consumes:

* the fundamentals + technicals snapshots written to session state by the
  market-data tools (so the numbers cannot be hallucinated), and
* a sentiment label supplied by the agent after it has consulted the
  ``market_sentiment_researcher``.

The tool *refuses* to score until both quantitative lookups have run for the
ticker, which hard-enforces the 3-step reasoning loop.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from google.adk.tools.tool_context import ToolContext

from ..config import settings
from .market_data import UNAVAILABLE, fundamentals_key, technicals_key
from .memory import DEFAULT_RISK_PROFILE, HISTORY_KEY, RISK_PROFILE_KEY

Sentiment = Literal["positive", "neutral", "negative"]

# Category weights by user risk profile (must sum to 1.0).
WEIGHTS: dict[str, dict[str, float]] = {
    "conservative": {"fundamentals": 0.5, "sentiment": 0.2, "technicals": 0.3},
    "moderate": {"fundamentals": 0.4, "sentiment": 0.3, "technicals": 0.3},
    "aggressive": {"fundamentals": 0.3, "sentiment": 0.3, "technicals": 0.4},
}
# Composite score thresholds (score range is [-1, 1]).
THRESHOLDS: dict[str, tuple[float, float]] = {
    "conservative": (0.35, -0.20),
    "moderate": (0.25, -0.25),
    "aggressive": (0.20, -0.30),
}
SENTIMENT_SCORES = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _num(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _verdict(score: float) -> str:
    if score >= 0.2:
        return "🟢 Positive"
    if score <= -0.2:
        return "🔴 Negative"
    return "🟡 Neutral"


def score_fundamentals(f: dict[str, Any]) -> tuple[float, list[str]]:
    """Average of individual signals in [-1, 1]; returns (score, reasons)."""
    signals: list[tuple[float, str]] = []

    pe, fpe = _num(f, "trailing_pe"), _num(f, "forward_pe")
    eps = _num(f, "trailing_eps")
    if eps is not None and eps <= 0:
        signals.append((-1.0, "negative trailing EPS"))
    elif pe is not None:
        if pe < 15:
            signals.append((1.0, f"low P/E {pe}"))
        elif pe > 40:
            signals.append((-1.0, f"rich P/E {pe}"))
        elif pe > 28:
            signals.append((-0.5, f"elevated P/E {pe}"))
        else:
            signals.append((0.0, f"fair P/E {pe}"))
    if pe is not None and fpe is not None and fpe > 0:
        signals.append((0.5, "forward P/E below trailing") if fpe < pe else (-0.5, "forward P/E above trailing"))

    de = _num(f, "debt_to_equity_ratio")
    if de is not None:
        signals.append(
            (1.0, f"low leverage D/E {de}")
            if de < 1
            else (-1.0, f"high leverage D/E {de}")
            if de > 2
            else (0.0, f"moderate leverage D/E {de}")
        )

    trend = f.get("quarterly_eps_trend")
    if isinstance(trend, list) and len(trend) >= 2:
        first, last = _num(trend[0], "eps"), _num(trend[-1], "eps")
        if first is not None and last is not None:
            signals.append(
                (1.0, "quarterly EPS rising")
                if last > first
                else (-1.0, "quarterly EPS falling")
                if last < first
                else (0.0, "quarterly EPS flat")
            )

    rg = _num(f, "revenue_growth_yoy_percent")
    if rg is not None:
        signals.append(
            (1.0, f"revenue growth {rg}%")
            if rg > 10
            else (-1.0, f"revenue decline {rg}%")
            if rg < 0
            else (0.25, f"modest revenue growth {rg}%")
        )

    pm = _num(f, "profit_margin_percent")
    if pm is not None:
        signals.append(
            (0.5, f"strong margin {pm}%")
            if pm > 15
            else (-1.0, f"unprofitable, margin {pm}%")
            if pm < 0
            else (0.0, f"thin margin {pm}%")
        )

    if not signals:
        return 0.0, ["fundamental data unavailable"]
    return sum(s for s, _ in signals) / len(signals), [r for _, r in signals]


def score_technicals(t: dict[str, Any]) -> tuple[float, list[str]]:
    signals: list[tuple[float, str]] = []
    for key, label in (("price_vs_sma_50", "50-day MA"), ("price_vs_sma_200", "200-day MA")):
        pos = t.get(key)
        if pos in ("above", "below"):
            signals.append((1.0 if pos == "above" else -1.0, f"price {pos} {label}"))
    cross = t.get("ma_crossover")
    if isinstance(cross, str) and cross != UNAVAILABLE:
        signals.append((0.5, "golden cross") if cross.startswith("golden") else (-0.5, "death cross"))
    rsi = _num(t, "rsi_14")
    if rsi is not None:
        signals.append(
            (-0.5, f"overbought RSI {rsi}")
            if rsi > 70
            else (0.5, f"oversold RSI {rsi}")
            if rsi < 30
            else (0.0, f"neutral RSI {rsi}")
        )
    macd = _num(t, "macd_histogram")
    if macd is not None:
        signals.append((0.5, "MACD momentum positive") if macd > 0 else (-0.5, "MACD momentum negative"))
    r3 = _num(t, "return_3m_percent")
    if r3 is not None:
        signals.append(
            (0.5, f"3m return {r3}%")
            if r3 > 5
            else (-0.5, f"3m return {r3}%")
            if r3 < -5
            else (0.0, f"3m return {r3}%")
        )
    if not signals:
        return 0.0, ["technical data unavailable"]
    return sum(s for s, _ in signals) / len(signals), [r for _, r in signals]


def compute_recommendation(
    fundamentals: dict[str, Any],
    technicals: dict[str, Any],
    sentiment: str,
    risk_profile: str = DEFAULT_RISK_PROFILE,
) -> dict[str, Any]:
    """Pure scoring function (unit-testable, no I/O)."""
    profile = risk_profile if risk_profile in WEIGHTS else DEFAULT_RISK_PROFILE
    w = WEIGHTS[profile]
    f_score, f_reasons = score_fundamentals(fundamentals)
    t_score, t_reasons = score_technicals(technicals)
    s_score = SENTIMENT_SCORES.get(sentiment.lower().strip(), 0.0)

    composite = w["fundamentals"] * f_score + w["technicals"] * t_score + w["sentiment"] * s_score
    buy_at, sell_at = THRESHOLDS[profile]
    action = "BUY" if composite >= buy_at else "SELL" if composite <= sell_at else "HOLD"

    # Confidence: how decisive the score is, discounted when data is missing.
    coverage = sum(1 for r in (f_reasons, t_reasons) if "unavailable" not in r[0]) / 2
    decisiveness = min(1.0, abs(composite) / max(buy_at, abs(sell_at)))
    confidence = round(0.5 * coverage + 0.5 * decisiveness, 2)

    return {
        "recommendation": action,
        "composite_score": round(composite, 3),
        "confidence": confidence,
        "risk_profile": profile,
        "weights": w,
        "thresholds": {"buy_at_or_above": buy_at, "sell_at_or_below": sell_at},
        "categories": {
            "fundamentals": {"score": round(f_score, 3), "verdict": _verdict(f_score), "drivers": f_reasons},
            "market_sentiment": {"score": s_score, "verdict": _verdict(s_score), "label": sentiment},
            "technical_indicators": {"score": round(t_score, 3), "verdict": _verdict(t_score), "drivers": t_reasons},
        },
    }


def score_recommendation(
    ticker: str,
    sentiment: Sentiment,
    sentiment_summary: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Produce the final BUY/SELL/HOLD decision for a ticker (DECISION SYNTHESIS).

    Call this ONLY after `get_stock_fundamentals`, `get_technical_indicators`
    and `market_sentiment_researcher` have been called for the same ticker.
    It combines the stored quantitative data with your sentiment label using
    a weighted, risk-profile-aware model and records the result in the user's
    analysis history.

    Args:
        ticker: The stock ticker symbol that was analysed.
        sentiment: Overall market sentiment from the researcher: "positive", "neutral" or "negative".
        sentiment_summary: One-sentence summary of the key headlines driving sentiment.

    Returns:
        dict with the recommendation, composite score, confidence and a
        per-category verdict (use these verdicts in the scorecard table).
    """
    symbol = (ticker or "").strip().upper()
    state = tool_context.state
    fundamentals = state.get(fundamentals_key(symbol))
    technicals = state.get(technicals_key(symbol))

    missing = [
        name
        for name, data in (("get_stock_fundamentals", fundamentals), ("get_technical_indicators", technicals))
        if not data
    ]
    if missing:
        return {
            "status": "error",
            "error_message": (
                f"Cannot score {symbol}: quantitative data missing. Call {', '.join(missing)} "
                f"for {symbol} first, then the market_sentiment_researcher, then retry."
            ),
        }
    if sentiment not in SENTIMENT_SCORES:
        return {"status": "error", "error_message": "sentiment must be 'positive', 'neutral' or 'negative'."}

    risk_profile = state.get(RISK_PROFILE_KEY, DEFAULT_RISK_PROFILE)
    result = compute_recommendation(fundamentals, technicals, sentiment, risk_profile)

    # Long-term memory: append to the user-scoped analysis history.
    entry = {
        "ticker": symbol,
        "company_name": fundamentals.get("company_name"),
        "date": dt.date.today().isoformat(),
        "price": fundamentals.get("current_price"),
        "currency": fundamentals.get("currency"),
        "recommendation": result["recommendation"],
        "composite_score": result["composite_score"],
        "sentiment": sentiment,
        "sentiment_summary": sentiment_summary[:300],
    }
    history = list(state.get(HISTORY_KEY, []))
    history.append(entry)
    state[HISTORY_KEY] = history[-settings.history_limit :]

    return {"status": "success", "ticker": symbol, **result}
