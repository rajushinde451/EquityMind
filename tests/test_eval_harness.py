"""Unit tests for the eval harness scoring logic (offline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from run_eval import check_format, check_trajectory  # noqa: E402

GROUPS = [
    ["get_stock_fundamentals", "get_technical_indicators", "market_sentiment_researcher"],
    ["score_recommendation"],
]


def test_trajectory_ok():
    called = [
        "get_technical_indicators",
        "get_stock_fundamentals",
        "market_sentiment_researcher",
        "score_recommendation",
    ]
    assert check_trajectory(called, GROUPS, []) == []


def test_trajectory_order_violation():
    called = [
        "score_recommendation",
        "get_stock_fundamentals",
        "get_technical_indicators",
        "market_sentiment_researcher",
        "score_recommendation",
    ]
    # first score happened too early -> index of first score < last lookup
    assert any("order violation" in p for p in check_trajectory(called, GROUPS, []))


def test_trajectory_missing_and_forbidden():
    problems = check_trajectory(["score_recommendation"], GROUPS, ["score_recommendation"])
    assert "forbidden tool called: score_recommendation" in problems
    assert "missing tool: get_stock_fundamentals" in problems


def test_format_checks():
    assert check_format("### 📌 Recommendation: BUY", ["Recommendation: (BUY|SELL|HOLD)"], []) == []
    assert check_format("hello", ["Recommendation"], ["hello"]) == [
        "missing pattern: Recommendation",
        "forbidden pattern present: hello",
    ]
