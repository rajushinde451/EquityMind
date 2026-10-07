"""Observability: intent/outcome logging, PII-safe structured logs, OTel enrichment.

Layers
------
1. **Distributed tracing** – ADK natively emits OpenTelemetry spans for
   ``invoke_agent``, ``call_llm`` and ``execute_tool``. Our tools add child spans
   (``market_data.*``) and these callbacks attach business attributes
   (ticker, tokens, latency, cache hits, PII counts) to the active span. Export
   to Google Cloud Trace with ``EQUITYMIND_TRACE_TO_CLOUD=true``.
2. **Intent → outcome logging** – every unit of work is logged *before* it
   executes (what the agent intends to do and why) and *after* (what happened):

   ==========================  ==========================
   before (intent)             after (outcome)
   ==========================  ==========================
   ``agent_turn_start``        ``agent_turn_end``
   ``llm_request``             ``llm_response``
   ``tool_intent``             ``tool_result``
   ==========================  ==========================

   All share ``invocation_id`` / ``session_id`` / ``trace_id`` for correlation.
3. **PII safety** – ``JsonFormatter`` recursively redacts every message and
   structured field (``pii.redact_obj``) before emission.
4. **Per-session metrics** – running counters in session state (``obs:*``),
   visible in the ADK Web UI "State" tab.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from opentelemetry import trace

from .config import settings
from .pii import redact_obj, redact_text

logger = logging.getLogger("equitymind")

_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_MAX_FIELD_CHARS = 500


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + "…"
    return value


class JsonFormatter(logging.Formatter):
    """Render records as single-line, PII-redacted JSON (Cloud Logging compatible)."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {k: _truncate(v) for k, v in vars(record).items() if k not in _STD_ATTRS}
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            **redact_obj(extras),
        }
        # Correlation ids are added after redaction so they are never altered.
        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            payload["trace_id"] = format(span_ctx.trace_id, "032x")
            payload["span_id"] = format(span_ctx.span_id, "016x")
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class RedactingTextFormatter(logging.Formatter):
    """Plain-text formatter that still redacts PII (used when LOG_JSON=false)."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def configure_logging() -> None:
    """Idempotently configure the ``equitymind`` logger."""
    if getattr(configure_logging, "_done", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if settings.log_json
        else RedactingTextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
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


def _user_text(ctx: Any) -> str:
    content = getattr(ctx, "user_content", None)
    parts = getattr(content, "parts", None) or []
    return " ".join(p.text for p in parts if getattr(p, "text", None))


# ---------------------------- agent callbacks ------------------------------ #
def before_agent_observe(callback_context: Any) -> None:
    """INTENT: a new agent turn is starting for this user request."""
    _timers[f"agent:{callback_context.invocation_id}:{callback_context.agent_name}"] = time.perf_counter()
    logger.info(
        "agent_turn_start",
        extra={
            **_ctx_fields(callback_context),
            "event": "agent_turn_start",
            "phase": "intent",
            "user_request": _user_text(callback_context),  # redacted by formatter
        },
    )
    return None


def after_agent_observe(callback_context: Any) -> None:
    """OUTCOME: agent turn finished."""
    started = _timers.pop(f"agent:{callback_context.invocation_id}:{callback_context.agent_name}", None)
    state = callback_context.state
    logger.info(
        "agent_turn_end",
        extra={
            **_ctx_fields(callback_context),
            "event": "agent_turn_end",
            "phase": "outcome",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1) if started else None,
            "session_llm_calls": state.get("obs:llm_calls"),
            "session_tool_calls": state.get("obs:tool_calls"),
            "session_total_tokens": state.get("obs:total_tokens"),
        },
    )
    return None


# ---------------------------- model callbacks ------------------------------ #
def before_model_observe(callback_context: Any, llm_request: Any) -> None:
    """INTENT: about to call the LLM (what context and tools it will see)."""
    _timers[f"llm:{callback_context.invocation_id}:{callback_context.agent_name}"] = time.perf_counter()
    tools = list((getattr(llm_request, "tools_dict", None) or {}).keys())
    logger.info(
        "llm_request",
        extra={
            **_ctx_fields(callback_context),
            "event": "llm_request",
            "phase": "intent",
            "model": getattr(llm_request, "model", None),
            "context_messages": len(getattr(llm_request, "contents", None) or []),
            "available_tools": tools,
        },
    )
    return None


def after_model_observe(callback_context: Any, llm_response: Any) -> None:
    """OUTCOME: LLM responded (latency, tokens, which tools it decided to call)."""
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
        "llm_response",
        extra={
            **_ctx_fields(callback_context),
            "event": "llm_response",
            "phase": "outcome",
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_toks,
            "output_tokens": output_toks,
            "total_tokens": total_toks,
            "decided_tool_calls": tool_calls,
            "is_final_answer": not tool_calls,
            "error_code": getattr(llm_response, "error_code", None),
        },
    )
    return None


# ----------------------------- tool callbacks ------------------------------ #
def before_tool_observe(tool: Any, args: dict[str, Any], tool_context: Any) -> None:
    """INTENT: about to execute a tool, with its (redacted) arguments."""
    _timers[f"tool:{tool_context.function_call_id}"] = time.perf_counter()
    span = trace.get_current_span()
    span.set_attribute("equitymind.tool.intent", tool.name)
    logger.info(
        "tool_intent",
        extra={
            **_ctx_fields(tool_context),
            "event": "tool_intent",
            "phase": "intent",
            "tool": tool.name,
            "tool_args": args,
            "function_call_id": tool_context.function_call_id,
        },
    )
    return None


def after_tool_observe(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any) -> None:
    """OUTCOME: tool finished (status, latency, cache hit, decision)."""
    started = _timers.pop(f"tool:{tool_context.function_call_id}", None)
    latency_ms = round((time.perf_counter() - started) * 1000, 1) if started else None
    resp = tool_response if isinstance(tool_response, dict) else {}
    status = resp.get("status", "success")

    span = trace.get_current_span()
    span.set_attribute("equitymind.tool.status", status)
    if isinstance(args.get("ticker"), str):
        span.set_attribute("equitymind.ticker", args["ticker"])

    _bump(tool_context, "obs:tool_calls")
    if status == "error":
        _bump(tool_context, "obs:tool_errors")

    logger.info(
        "tool_result",
        extra={
            **_ctx_fields(tool_context),
            "event": "tool_result",
            "phase": "outcome",
            "tool": tool.name,
            "tool_args": args,
            "function_call_id": tool_context.function_call_id,
            "status": status,
            "latency_ms": latency_ms,
            "cache_hit": resp.get("cache_hit"),
            "recommendation": resp.get("recommendation"),
            "error_message": resp.get("error_message"),
        },
    )
    return None
