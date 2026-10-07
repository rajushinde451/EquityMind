"""Centralised runtime configuration (12-factor: everything from env vars)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("EQUITYMIND_MODEL", "gemini-2.5-flash")
    sentiment_model: str = os.getenv("EQUITYMIND_SENTIMENT_MODEL", os.getenv("EQUITYMIND_MODEL", "gemini-2.5-flash"))
    market_data_cache_ttl_s: int = int(os.getenv("EQUITYMIND_CACHE_TTL_S", "300"))
    history_limit: int = int(os.getenv("EQUITYMIND_HISTORY_LIMIT", "20"))
    log_level: str = os.getenv("EQUITYMIND_LOG_LEVEL", "INFO")
    log_json: bool = _bool("EQUITYMIND_LOG_JSON", True)
    # Server / infra
    session_service_uri: str | None = os.getenv("EQUITYMIND_SESSION_URI") or None
    # e.g. "agentengine://<agent_engine_id>" for Vertex AI Memory Bank; None = in-memory.
    memory_service_uri: str | None = os.getenv("EQUITYMIND_MEMORY_URI") or None
    redact_model_input: bool = _bool("EQUITYMIND_REDACT_MODEL_INPUT", True)
    trace_to_cloud: bool = _bool("EQUITYMIND_TRACE_TO_CLOUD", False)
    serve_web_ui: bool = _bool("EQUITYMIND_SERVE_WEB_UI", True)
    allowed_origins: tuple[str, ...] = tuple(o for o in os.getenv("EQUITYMIND_ALLOWED_ORIGINS", "*").split(",") if o)


settings = Settings()

DISCLAIMER = (
    '*"Disclaimer: I am an AI, not a certified financial advisor. This analysis is '
    'for informational purposes only. Invest at your own risk."*'
)
