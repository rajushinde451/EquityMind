"""EquityMind: a Buy/Sell/Hold stock scorecard agent built on Google ADK.

Architecture
------------
::

    equitymind (root LlmAgent, orchestrator)
    ├── get_stock_fundamentals        FunctionTool  ─┐ step 1: quantitative
    ├── get_technical_indicators      FunctionTool  ─┘ (parallel-callable)
    ├── market_sentiment_researcher   AgentTool       step 2: sentiment
    │     └── google_search  (built-in Gemini grounding, isolated context)
    ├── score_recommendation          FunctionTool    step 3: deterministic decision
    └── set_risk_profile / manage_watchlist / get_analysis_history   (memory)

* ``google_search`` lives in its own sub-agent because Gemini's built-in search
  cannot be combined with function tools on one agent. Wrapping it as an
  AgentTool also keeps raw search results out of the root agent's context;
  only the distilled report comes back.
* Guardrails and observability are attached as ADK callbacks.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from .config import settings
from .guardrails import enforce_disclaimer, validate_ticker_args
from .memory_consolidation import apply_pending_digest, recall_past_conversations, schedule_consolidation
from .observability import (
    after_agent_observe,
    after_model_observe,
    after_tool_observe,
    before_agent_observe,
    before_model_observe,
    before_tool_observe,
    configure_logging,
)
from .pii import redact_pii_in_request
from .prompt import build_root_instruction, build_sentiment_instruction
from .tools import (
    clear_analysis_history,
    get_analysis_history,
    get_stock_fundamentals,
    get_technical_indicators,
    manage_watchlist,
    score_recommendation,
    set_risk_profile,
)

configure_logging()

# Model-input PII scrubbing runs first (mutates the request, returns None).
_BEFORE_MODEL = ([redact_pii_in_request] if settings.redact_model_input else []) + [before_model_observe]


def _root_instruction(ctx) -> str:
    return build_root_instruction(ctx.state)


def _sentiment_instruction(ctx) -> str:
    return build_sentiment_instruction(ctx.state)


sentiment_agent = Agent(
    name="market_sentiment_researcher",
    model=settings.sentiment_model,
    description=(
        "Searches the web for the latest 48-hour news headlines and analyst "
        "reports on a stock and returns a dated, sourced summary ending with a "
        "SENTIMENT_LABEL (positive/neutral/negative). Input: ticker and company "
        "name. Also useful for verifying unfamiliar tickers or company names."
    ),
    instruction=_sentiment_instruction,
    tools=[google_search],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
    before_model_callback=_BEFORE_MODEL,
    after_model_callback=after_model_observe,
)

root_agent = Agent(
    name="equitymind",
    model=settings.model,
    description="Financial analyst agent producing Buy/Sell/Hold stock scorecards.",
    instruction=_root_instruction,
    tools=[
        # Analysis loop
        get_stock_fundamentals,
        get_technical_indicators,
        AgentTool(agent=sentiment_agent),
        score_recommendation,
        # Memory (HITL-gated where state changes are consequential)
        set_risk_profile,
        manage_watchlist,
        get_analysis_history,
        clear_analysis_history,
        recall_past_conversations,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    # Turn lifecycle: apply last turn's background digest, log intent ...
    before_agent_callback=[apply_pending_digest, before_agent_observe],
    # ... log outcome, then kick off async consolidation (non-blocking).
    after_agent_callback=[after_agent_observe, schedule_consolidation],
    before_model_callback=_BEFORE_MODEL,
    # ADK stops at the first callback returning non-None, so the observer
    # (always returns None) runs first, then the disclaimer guardrail.
    after_model_callback=[after_model_observe, enforce_disclaimer],
    # Intent is logged for every attempted call, including ones the validator rejects.
    before_tool_callback=[before_tool_observe, validate_ticker_args],
    after_tool_callback=after_tool_observe,
)
