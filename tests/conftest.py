"""Shared offline test fixtures (no network, no API key)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("EQUITYMIND_LOG_JSON", "true")


@dataclass
class FakeToolContext:
    """Duck-typed stand-in for ADK ToolContext / CallbackContext."""

    state: dict[str, Any] = field(default_factory=dict)
    function_call_id: str = "call-1"
    invocation_id: str = "inv-1"
    agent_name: str = "equitymind"
    user_id: str = "test-user"
    session: Any = None


class FakeTicker:
    def __init__(self, info: dict, closes: list[float] | pd.Series, eps: list[float] | None = None):
        self.info = info
        self._closes = pd.Series(closes, dtype=float)
        idx = pd.to_datetime(["2025-12-31", "2026-03-31", "2026-06-30"][: len(eps or [])])
        self.quarterly_income_stmt = pd.DataFrame([eps], index=["Diluted EPS"], columns=idx) if eps else pd.DataFrame()

    def history(self, **_):
        return pd.DataFrame({"Close": self._closes})


@pytest.fixture
def ctx() -> FakeToolContext:
    return FakeToolContext()


@pytest.fixture(autouse=True)
def _clear_cache():
    from equitymind.tools import market_data

    market_data._cache.clear()
    yield
    market_data._cache.clear()


@pytest.fixture
def uptrend() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(np.linspace(100, 200, 252) + rng.normal(0, 0.5, 252))


@pytest.fixture
def downtrend() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(np.linspace(200, 100, 252) + rng.normal(0, 0.5, 252))


STRONG_INFO = {
    "longName": "Acme Corp",
    "currency": "USD",
    "currentPrice": 200.0,
    "trailingPE": 12.0,
    "forwardPE": 10.0,
    "trailingEps": 16.0,
    "dividendRate": 2.0,
    "debtToEquity": 50.0,
    "revenueGrowth": 0.20,
    "profitMargins": 0.25,
}

WEAK_INFO = {
    "longName": "Bustco",
    "currency": "USD",
    "currentPrice": 10.0,
    "trailingPE": 80.0,
    "forwardPE": 95.0,
    "trailingEps": 0.12,
    "debtToEquity": 350.0,
    "revenueGrowth": -0.15,
    "profitMargins": -0.05,
}
