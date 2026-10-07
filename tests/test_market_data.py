from unittest.mock import patch

from conftest import STRONG_INFO, FakeTicker

from equitymind.tools import market_data as md


def test_fundamentals_success_and_state_snapshot(ctx, uptrend):
    fake = FakeTicker(STRONG_INFO, uptrend, eps=[1.0, 1.1, 1.2])
    with patch.object(md.yf, "Ticker", return_value=fake):
        out = md.get_stock_fundamentals("acme", ctx)

    assert out["status"] == "success"
    assert out["ticker"] == "ACME"
    assert out["trailing_pe"] == 12.0
    assert out["dividend_yield_percent"] == 1.0
    assert out["debt_to_equity_ratio"] == 0.5
    assert out["revenue_growth_yoy_percent"] == 20.0
    assert [q["eps"] for q in out["quarterly_eps_trend"]] == [1.0, 1.1, 1.2]
    assert out["analyst_target_mean_price"] == md.UNAVAILABLE
    assert ctx.state[md.fundamentals_key("ACME")] == out
    assert ctx.state["last_ticker"] == "ACME"


def test_fundamentals_cache_hit(ctx, uptrend):
    fake = FakeTicker(STRONG_INFO, uptrend)
    with patch.object(md.yf, "Ticker", return_value=fake) as mocked:
        first = md.get_stock_fundamentals("ACME", ctx)
        second = md.get_stock_fundamentals("ACME", ctx)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert mocked.call_count == 1


def test_non_dividend_payer_reports_zero(ctx, uptrend):
    info = {k: v for k, v in STRONG_INFO.items() if k != "dividendRate"}
    with patch.object(md.yf, "Ticker", return_value=FakeTicker(info, uptrend)):
        assert md.get_stock_fundamentals("ACME", ctx)["dividend_yield_percent"] == 0.0


def test_unknown_ticker_returns_error_and_no_snapshot(ctx):
    with patch.object(md.yf, "Ticker", return_value=FakeTicker({}, [])):
        out = md.get_stock_fundamentals("ZZZZQ", ctx)
    assert out["status"] == "error"
    assert "not recognized" in out["error_message"]
    assert ctx.state == {}


def test_provider_exception_is_wrapped(ctx):
    with patch.object(md.yf, "Ticker", side_effect=RuntimeError("boom")):
        out = md.get_stock_fundamentals("AAPL", ctx)
    assert out["status"] == "error" and "boom" in out["error_message"]


def test_empty_ticker():
    assert md.get_stock_fundamentals("  ")["status"] == "error"
    assert md.get_technical_indicators("")["status"] == "error"


def test_technicals_uptrend(ctx, uptrend):
    with patch.object(md.yf, "Ticker", return_value=FakeTicker({}, uptrend)):
        out = md.get_technical_indicators("ACME", ctx)
    assert out["status"] == "success"
    assert out["price_vs_sma_50"] == "above"
    assert out["price_vs_sma_200"] == "above"
    assert out["ma_crossover"].startswith("golden")
    assert out["rsi_14"] > 50
    assert out["return_3m_percent"] > 0
    assert ctx.state[md.technicals_key("ACME")] == out


def test_technicals_downtrend(downtrend):
    out = md.compute_technicals("X", downtrend)
    assert out["price_vs_sma_200"] == "below"
    assert out["ma_crossover"].startswith("death")
    assert out["macd_histogram"] != md.UNAVAILABLE


def test_technicals_short_history_marks_unavailable():
    out = md.compute_technicals("NEW", __import__("pandas").Series([10.0, 11.0, 12.0]))
    assert out["sma_50_day"] == md.UNAVAILABLE
    assert out["sma_200_day"] == md.UNAVAILABLE
    assert out["rsi_14"] == md.UNAVAILABLE


def test_technicals_unknown_ticker(ctx):
    with patch.object(md.yf, "Ticker", return_value=FakeTicker({}, [])):
        assert md.get_technical_indicators("ZZZZQ", ctx)["status"] == "error"
