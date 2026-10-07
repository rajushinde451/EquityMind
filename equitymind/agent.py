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
from .observability import (
    after_model_observe,
    after_tool_observe,
    before_model_observe,
    before_tool_observe,
    configure_logging,
)
from .prompt import build_root_instruction, build_sentiment_instruction
from .tools import (
    get_analysis_history,
    get_stock_fundamentals,
    get_technical_indicators,
    manage_watchlist,
    score_recommendation,
    set_risk_profile,
)

configure_logging()


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
    before_model_callback=before_model_observe,
    after_model_callback=after_model_observe,
)

root_agent = Agent(
    name="equitymind",
    model=settings.model,
    description="Financial analyst agent producing Buy/Sell/Hold stock scorecards.",
    instruction=_root_instruction,
    tools=[
        get_stock_fundamentals,
        get_technical_indicators,
        AgentTool(agent=sentiment_agent),
        score_recommendation,
        set_risk_profile,
        manage_watchlist,
        get_analysis_history,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    before_model_callback=before_model_observe,
    # ADK stops at the first callback returning non-None, so the observer
    # (always returns None) runs first, then the disclaimer guardrail.
    after_model_callback=[after_model_observe, enforce_disclaimer],
    before_tool_callback=[validate_ticker_args, before_tool_observe],
    after_tool_callback=after_tool_observe,
)
