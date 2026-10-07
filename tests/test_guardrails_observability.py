import json
import logging
from types import SimpleNamespace

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from equitymind import observability as obs
from equitymind.agent import root_agent, sentiment_agent
from equitymind.guardrails import enforce_disclaimer, validate_ticker_args


def _tool(name="get_stock_fundamentals"):
    return SimpleNamespace(name=name)


# ------------------------------- guardrails -------------------------------- #
def test_ticker_normalised_in_place(ctx):
    args = {"ticker": "  $brk-b "}
    assert validate_ticker_args(_tool(), args, ctx) is None
    assert args["ticker"] == "BRK-B"


def test_suffix_and_index_tickers_allowed(ctx):
    for sym in ("RELIANCE.NS", "^GSPC", "EURUSD=X", "7203.T"):
        assert validate_ticker_args(_tool(), {"ticker": sym}, ctx) is None


def test_bad_ticker_short_circuits(ctx):
    out = validate_ticker_args(_tool(), {"ticker": "DROP TABLE;"}, ctx)
    assert out["status"] == "error"


def test_watchlist_list_without_ticker_allowed(ctx):
    assert validate_ticker_args(_tool("manage_watchlist"), {"action": "list", "ticker": ""}, ctx) is None


def test_tools_without_ticker_untouched(ctx):
    assert validate_ticker_args(_tool("set_risk_profile"), {"risk_profile": "moderate"}, ctx) is None


def _resp(*parts):
    return LlmResponse(content=types.Content(role="model", parts=list(parts)))


def test_disclaimer_appended(ctx):
    out = enforce_disclaimer(ctx, _resp(types.Part(text="### 📌 Recommendation: BUY")))
    assert "not a certified financial advisor" in "".join(p.text for p in out.content.parts)


def test_disclaimer_not_duplicated(ctx):
    text = "BUY. Disclaimer: I am an AI, not a certified financial advisor."
    assert enforce_disclaimer(ctx, _resp(types.Part(text=text))) is None


def test_disclaimer_skipped_for_tool_calls(ctx):
    call = types.Part(function_call=types.FunctionCall(name="get_stock_fundamentals", args={"ticker": "AAPL"}))
    assert enforce_disclaimer(ctx, _resp(call)) is None


def test_disclaimer_skipped_for_partial(ctx):
    r = _resp(types.Part(text="partial"))
    r.partial = True
    assert enforce_disclaimer(ctx, r) is None


# ------------------------------ observability ------------------------------ #
def test_model_callbacks_record_metrics(ctx):
    obs.before_model_observe(ctx, None)
    resp = _resp(types.Part(text="hi"))
    resp.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100, candidates_token_count=20, total_token_count=120
    )
    obs.after_model_observe(ctx, resp)
    assert ctx.state["obs:llm_calls"] == 1
    assert ctx.state["obs:total_tokens"] == 120


def test_tool_callbacks_count_errors(ctx):
    obs.before_tool_observe(_tool(), {"ticker": "X"}, ctx)
    obs.after_tool_observe(_tool(), {"ticker": "X"}, ctx, {"status": "error"})
    assert ctx.state["obs:tool_calls"] == 1
    assert ctx.state["obs:tool_errors"] == 1


def test_json_formatter_includes_extras():
    record = logging.LogRecord("equitymind", logging.INFO, __file__, 1, "tool_call", None, None)
    record.tool = "get_stock_fundamentals"
    record.latency_ms = 12.5
    payload = json.loads(obs.JsonFormatter().format(record))
    assert payload["severity"] == "INFO"
    assert payload["message"] == "tool_call"
    assert payload["tool"] == "get_stock_fundamentals"
    assert payload["latency_ms"] == 12.5


# ------------------------------ agent wiring ------------------------------- #
def test_agent_wiring():
    names = {getattr(t, "name", getattr(t, "__name__", None)) for t in root_agent.tools}
    assert {
        "get_stock_fundamentals",
        "get_technical_indicators",
        "market_sentiment_researcher",
        "score_recommendation",
        "set_risk_profile",
        "manage_watchlist",
        "get_analysis_history",
    } <= names
    # google_search must be isolated in the sub-agent.
    assert [getattr(t, "name", None) for t in sentiment_agent.tools] == ["google_search"]
    assert enforce_disclaimer in root_agent.after_model_callback
    assert root_agent.after_model_callback[0] is obs.after_model_observe
    assert validate_ticker_args in root_agent.before_tool_callback
