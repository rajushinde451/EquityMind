from equitymind.prompt import build_root_instruction, build_sentiment_instruction, render_user_context
from equitymind.tools.memory import (
    HISTORY_KEY,
    RISK_PROFILE_KEY,
    WATCHLIST_KEY,
    get_analysis_history,
    manage_watchlist,
    set_risk_profile,
)


def test_user_scoped_keys():
    # `user:` prefix is what makes ADK persist these across sessions.
    assert all(k.startswith("user:") for k in (RISK_PROFILE_KEY, WATCHLIST_KEY, HISTORY_KEY))


def test_risk_profile(ctx):
    assert set_risk_profile("conservative", ctx)["status"] == "success"
    assert ctx.state[RISK_PROFILE_KEY] == "conservative"
    assert set_risk_profile("yolo", ctx)["status"] == "error"


def test_watchlist_lifecycle(ctx):
    assert manage_watchlist("add", ctx, "aapl")["watchlist"] == ["AAPL"]
    assert manage_watchlist("add", ctx, "AAPL")["watchlist"] == ["AAPL"]  # idempotent
    manage_watchlist("add", ctx, "msft")
    assert manage_watchlist("list", ctx)["watchlist"] == ["AAPL", "MSFT"]
    assert manage_watchlist("remove", ctx, "AAPL")["watchlist"] == ["MSFT"]
    assert manage_watchlist("remove", ctx, "TSLA")["status"] == "error"
    assert manage_watchlist("add", ctx)["status"] == "error"


def test_history_filter(ctx):
    ctx.state[HISTORY_KEY] = [{"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "AAPL"}]
    assert get_analysis_history(ctx, "aapl")["count"] == 2
    assert get_analysis_history(ctx)["count"] == 3


def test_context_injection_defaults():
    text = render_user_context({})
    assert "Risk profile: moderate (default)" in text
    assert "Watchlist: empty" in text
    assert "Recent analyses: none" in text


def test_context_injection_with_memory():
    state = {
        RISK_PROFILE_KEY: "aggressive",
        WATCHLIST_KEY: ["NVDA", "AMD"],
        HISTORY_KEY: [
            {
                "date": "2026-10-01",
                "ticker": "NVDA",
                "recommendation": "HOLD",
                "price": 118.2,
                "currency": "USD",
                "composite_score": 0.1,
            }
        ],
        "last_ticker": "NVDA",
    }
    text = build_root_instruction(state)
    assert "Risk profile: aggressive" in text and "(default)" not in text
    assert "NVDA, AMD" in text
    assert "NVDA -> HOLD at 118.2 USD" in text
    assert "Ticker in focus this session: NVDA" in text
    assert "__TODAY__" not in text


def test_sentiment_instruction_has_date_and_label_contract():
    text = build_sentiment_instruction()
    assert "__TODAY__" not in text
    assert "SENTIMENT_LABEL" in text
