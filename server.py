"""Production HTTP entrypoint for EquityMind (Cloud Run / Docker).

Exposes:
* The full ADK API (``/run``, ``/run_sse``, ``/apps/...`` sessions) and,
  optionally, the ADK dev Web UI — via ``get_fast_api_app``.
* ``GET  /healthz``            – liveness/readiness probe.
* ``POST /v1/analyze``         – simple typed endpoint: ticker in, scorecard out.
* ``GET  /v1/users/{id}/memory`` – inspect a user's long-term memory.

Persistence: set ``EQUITYMIND_SESSION_URI`` (e.g. ``sqlite+aiosqlite:///./data/sessions.db``
or ``postgresql+asyncpg://...``) so ``user:`` scoped memory survives restarts.
Tracing: ``EQUITYMIND_TRACE_TO_CLOUD=true`` exports OTel spans to Cloud Trace.
"""

from __future__ import annotations

import os
import time
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import HTTPException
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(AGENTS_DIR, "equitymind", ".env"))

from equitymind import __version__  # noqa: E402  (env must be loaded first)
from equitymind.agent import root_agent  # noqa: E402
from equitymind.config import settings  # noqa: E402
from equitymind.guardrails import TICKER_RE  # noqa: E402
from equitymind.observability import logger  # noqa: E402
from equitymind.tools.memory import HISTORY_KEY, RISK_PROFILE_KEY, WATCHLIST_KEY  # noqa: E402

APP_NAME = "equitymind"

app = get_fast_api_app(
    agents_dir=AGENTS_DIR,
    session_service_uri=settings.session_service_uri,
    allow_origins=list(settings.allowed_origins),
    web=settings.serve_web_ui,
    trace_to_cloud=settings.trace_to_cloud,
)

# Dedicated runner for the typed REST endpoints (shares the same session DB).
_session_service = (
    DatabaseSessionService(db_url=settings.session_service_uri)
    if settings.session_service_uri
    else InMemorySessionService()
)
_runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=_session_service)


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], description="Ticker symbol to analyse.")
    user_id: str = Field("anonymous", description="Stable user id; scopes long-term memory.")
    session_id: str | None = Field(None, description="Reuse a session for follow-ups.")
    question: str | None = Field(None, description="Optional custom question; defaults to a scorecard request.")


class AnalyzeResponse(BaseModel):
    ticker: str
    user_id: str
    session_id: str
    scorecard_markdown: str
    tools_called: list[str]
    latency_ms: float


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "model": settings.model}


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["equitymind"])
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    symbol = req.ticker.strip().upper()
    if not TICKER_RE.match(symbol):
        raise HTTPException(status_code=422, detail=f"Invalid ticker format: {req.ticker!r}")

    session = None
    if req.session_id:
        session = await _session_service.get_session(app_name=APP_NAME, user_id=req.user_id, session_id=req.session_id)
    if session is None:
        session = await _session_service.create_session(
            app_name=APP_NAME, user_id=req.user_id, session_id=req.session_id or str(uuid.uuid4())
        )

    prompt = req.question or f"Should I buy {symbol}? Produce the full scorecard."
    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    started = time.perf_counter()
    final_text, tools_called = "", []
    async for event in _runner.run_async(user_id=req.user_id, session_id=session.id, new_message=message):
        for call in event.get_function_calls() or []:
            tools_called.append(call.name)
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False))
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    logger.info(
        "analyze_request",
        extra={
            "event": "analyze_request",
            "ticker": symbol,
            "user_id": req.user_id,
            "session_id": session.id,
            "tools_called": tools_called,
            "latency_ms": latency_ms,
        },
    )
    if not final_text:
        raise HTTPException(status_code=502, detail="Agent returned no response.")
    return AnalyzeResponse(
        ticker=symbol,
        user_id=req.user_id,
        session_id=session.id,
        scorecard_markdown=final_text,
        tools_called=tools_called,
        latency_ms=latency_ms,
    )


@app.get("/v1/users/{user_id}/memory", tags=["equitymind"])
async def user_memory(user_id: str) -> dict:
    """Return the user-scoped long-term memory (risk profile, watchlist, history)."""
    listing = await _session_service.list_sessions(app_name=APP_NAME, user_id=user_id)
    sessions = getattr(listing, "sessions", None) or []
    if not sessions:
        return {"user_id": user_id, "risk_profile": None, "watchlist": [], "analysis_history": []}
    latest = await _session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=sessions[-1].id)
    state = latest.state if latest else {}
    return {
        "user_id": user_id,
        "risk_profile": state.get(RISK_PROFILE_KEY),
        "watchlist": state.get(WATCHLIST_KEY, []),
        "analysis_history": state.get(HISTORY_KEY, []),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
