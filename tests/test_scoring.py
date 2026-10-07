from unittest.mock import patch

from conftest import STRONG_INFO, WEAK_INFO, FakeTicker

from equitymind.tools import market_data as md
from equitymind.tools.memory import HISTORY_KEY, RISK_PROFILE_KEY
from equitymind.tools.scoring import WEIGHTS, compute_recommendation, score_recommendation


def _load(ctx, info, closes, symbol="ACME", eps=None):
    with patch.object(md.yf, "Ticker", return_value=FakeTicker(info, closes, eps=eps)):
        md.get_stock_fundamentals(symbol, ctx)
        md.get_technical_indicators(symbol, ctx)


def test_refuses_without_quant_data(ctx):
    out = score_recommendation("ACME", "positive", "great news", ctx)
    assert out["status"] == "error"
    assert "get_stock_fundamentals" in out["error_message"]
    assert "get_technical_indicators" in out["error_message"]


def test_refuses_with_partial_data(ctx, uptrend):
    with patch.object(md.yf, "Ticker", return_value=FakeTicker(STRONG_INFO, uptrend)):
        md.get_stock_fundamentals("ACME", ctx)
    out = score_recommendation("ACME", "positive", "x", ctx)
    assert out["status"] == "error"
    assert "get_technical_indicators" in out["error_message"]
    assert "get_stock_fundamentals" not in out["error_message"]


def test_strong_stock_is_buy_and_recorded(ctx, uptrend):
    _load(ctx, STRONG_INFO, uptrend, eps=[1.0, 1.2, 1.4])
    out = score_recommendation("acme", "positive", "Beat earnings", ctx)
    assert out["status"] == "success"
    assert out["recommendation"] == "BUY"
    assert out["categories"]["fundamentals"]["verdict"] == "🟢 Positive"
    assert out["categories"]["technical_indicators"]["verdict"] == "🟢 Positive"
    assert 0 < out["confidence"] <= 1
    assert ctx.state[HISTORY_KEY][-1]["ticker"] == "ACME"
    assert ctx.state[HISTORY_KEY][-1]["recommendation"] == "BUY"


def test_weak_stock_is_sell(ctx, downtrend):
    _load(ctx, WEAK_INFO, downtrend, symbol="BUST", eps=[0.5, 0.3, 0.1])
    out = score_recommendation("BUST", "negative", "Guidance cut", ctx)
    assert out["recommendation"] == "SELL"
    assert out["categories"]["fundamentals"]["verdict"] == "🔴 Negative"


def test_invalid_sentiment_rejected(ctx, uptrend):
    _load(ctx, STRONG_INFO, uptrend)
    assert score_recommendation("ACME", "bullish", "x", ctx)["status"] == "error"


def test_risk_profile_changes_weights(ctx, uptrend):
    _load(ctx, STRONG_INFO, uptrend)
    ctx.state[RISK_PROFILE_KEY] = "aggressive"
    out = score_recommendation("ACME", "neutral", "quiet", ctx)
    assert out["risk_profile"] == "aggressive"
    assert out["weights"] == WEIGHTS["aggressive"]


def test_weights_sum_to_one():
    for w in WEIGHTS.values():
        assert abs(sum(w.values()) - 1.0) < 1e-9


def test_mixed_signals_hold():
    # Good fundamentals but bearish tape and negative news -> HOLD.
    f = {"trailing_pe": 20, "trailing_eps": 5, "debt_to_equity_ratio": 0.5, "revenue_growth_yoy_percent": 12}
    t = {"price_vs_sma_50": "below", "price_vs_sma_200": "below", "rsi_14": 45}
    assert compute_recommendation(f, t, "negative")["recommendation"] == "HOLD"


def test_missing_data_lowers_confidence():
    full = compute_recommendation({"trailing_pe": 10, "trailing_eps": 1}, {"price_vs_sma_50": "above"}, "positive")
    empty = compute_recommendation({}, {}, "positive")
    assert empty["confidence"] < full["confidence"]
    assert "unavailable" in empty["categories"]["fundamentals"]["drivers"][0]


def test_history_is_capped(ctx, uptrend):
    _load(ctx, STRONG_INFO, uptrend)
    for _ in range(30):
        score_recommendation("ACME", "neutral", "x", ctx)
    assert len(ctx.state[HISTORY_KEY]) == 20
