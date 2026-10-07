"""Observability: structured logging, metrics-in-state and OpenTelemetry enrichment.

Three layers:

1. **Distributed tracing** – ADK natively emits OpenTelemetry spans for
   ``invoke_agent``, ``call_llm`` and ``execute_tool``. Our tools add child spans
   (``market_data.*``) and these callbacks attach business attributes
   (ticker, tokens, latency, cache hits) to the active span. Export to Google
   Cloud Trace with ``EQUITYMIND_TRACE_TO_CLOUD=true`` (see ``server.py``) or
   ``adk web --trace_to_cloud``.
2. **Structured logs** – one JSON line per model/tool call, correlated by
   ``invocation_id`` / ``session_id``. Cloud Logging parses these natively.
3. **Per-session metrics** – running counters in session state
   (``obs:*``) visible in the ADK Web UI "State" tab.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from opentelemetry import trace

from .config import settings

logger = logging.getLogger("equitymind")

_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON (Cloud Logging compatible)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            payload["trace_id"] = format(span_ctx.trace_id, "032x")
            payload["span_id"] = format(span_ctx.span_id, "016x")
        payload.update({k: v for k, v in vars(record).items() if k not in _STD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Idempotently configure the ``equitymind`` logger."""
    if getattr(configure_logging, "_done", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if settings.log_json else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    configure_logging._done = True  # type: ignore[attr-defined]


# In-flight timers keyed by invocation/agent or function-call id.
_timers: dict[str, float] = {}


def _ctx_fields(ctx: Any) -> dict[str, Any]:
    session = getattr(ctx, "session", None)
    return {
        "invocation_id": getattr(ctx, "invocation_id", None),
        "session_id": getattr(session, "id", None),
        "user_id": getattr(ctx, "user_id", None),
        "agent": getattr(ctx, "agent_name", None),
    }


def _bump(ctx: Any, key: str, amount: float = 1) -> None:
    try:
        ctx.state[key] = (ctx.state.get(key) or 0) + amount
    except Exception:  # noqa: BLE001 - never let telemetry break the agent
        pass


# ---------------------------- model callbacks ------------------------------ #
def before_model_observe(callback_context: Any, llm_request: Any) -> None:
    _timers[f"llm:{callback_context.invocation_id}:{callback_context.agent_name}"] = time.perf_counter()
    return None


def after_model_observe(callback_context: Any, llm_response: Any) -> None:
    if getattr(llm_response, "partial", False):
        return None
    started = _timers.pop(f"llm:{callback_context.invocation_id}:{callback_context.agent_name}", None)
    latency_ms = round((time.perf_counter() - started) * 1000, 1) if started else None

    usage = getattr(llm_response, "usage_metadata", None)
    prompt_toks = getattr(usage, "prompt_token_count", None) or 0
    output_toks = getattr(usage, "candidates_token_count", None) or 0
    total_toks = getattr(usage, "total_token_count", None) or (prompt_toks + output_toks)

    parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
    tool_calls = [p.function_call.name for p in parts if getattr(p, "function_call", None)]

    span = trace.get_current_span()
    span.set_attribute("equitymind.llm.latency_ms", latency_ms or 0)
    span.set_attribute("equitymind.llm.total_tokens", total_toks)

    _bump(callback_context, "obs:llm_calls")
    _bump(callback_context, "obs:total_tokens", total_toks)

    logger.info(
        "llm_call",
        extra={
            **_ctx_fields(callback_context),
            "event": "llm_call",
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_toks,
            "output_tokens": output_toks,
            "total_tokens": total_toks,
            "requested_tools": tool_calls,
            "error_code": getattr(llm_response, "error_code", None),
        },
    )
    return None


# ----------------------------- tool callbacks ------------------------------ #
def before_tool_observe(tool: Any, args: dict[str, Any], tool_context: Any) -> None:
    _timers[f"tool:{tool_context.function_call_id}"] = time.perf_counter()
    return None


def after_tool_observe(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any) -> None:
    started = _timers.pop(f"tool:{tool_context.function_call_id}", None)
    latency_ms = round((time.perf_counter() - started) * 1000, 1) if started else None
    status = tool_response.get("status", "success") if isinstance(tool_response, dict) else "success"

    span = trace.get_current_span()
    span.set_attribute("equitymind.tool.status", status)
    if isinstance(args.get("ticker"), str):
        span.set_attribute("equitymind.ticker", args["ticker"])

    _bump(tool_context, "obs:tool_calls")
    if status == "error":
        _bump(tool_context, "obs:tool_errors")

    logger.info(
        "tool_call",
        extra={
            **_ctx_fields(tool_context),
            "event": "tool_call",
            "tool": tool.name,
            "tool_args": args,
            "status": status,
            "latency_ms": latency_ms,
            "cache_hit": tool_response.get("cache_hit") if isinstance(tool_response, dict) else None,
            "recommendation": tool_response.get("recommendation") if isinstance(tool_response, dict) else None,
        },
    )
    return None
