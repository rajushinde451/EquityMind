"""Persistence test: `user:` memory must survive across sessions in a real DB."""

import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService


@pytest.mark.asyncio
async def test_user_state_persists_across_sessions(tmp_path):
    svc = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    s1 = await svc.create_session(app_name="equitymind", user_id="raj")
    await svc.append_event(
        s1,
        Event(
            author="equitymind",
            invocation_id="inv-1",
            actions=EventActions(state_delta={"user:watchlist": ["NVDA"], "last_ticker": "NVDA"}),
        ),
    )

    s2 = await svc.create_session(app_name="equitymind", user_id="raj")
    assert s2.state.get("user:watchlist") == ["NVDA"]  # user-scoped: shared
    assert s2.state.get("last_ticker") is None  # session-scoped: isolated

    other = await svc.create_session(app_name="equitymind", user_id="someone-else")
    assert other.state.get("user:watchlist") is None
