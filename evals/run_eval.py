"""Live agent evaluation: tool-trajectory + response-format checks.

Runs each case in ``eval_cases.json`` against the real agent (Gemini + yfinance
+ Google Search) and scores:

* **trajectory** – every tool group in ``expected_tools_in_order`` was called,
  and each group's tools all occurred before any tool of the next group
  (e.g. all lookups before ``score_recommendation``). ``forbidden_tools`` must
  not be called.
* **format** – all ``required_patterns`` match and no ``forbidden_patterns`` do.

Usage::

    uv run python evals/run_eval.py               # all cases
    uv run python evals/run_eval.py --case unknown_ticker_no_scorecard
    uv run python evals/run_eval.py --min-pass-rate 0.8   # CI gate

Writes a JSON report to ``evals/results/``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import re
import sys
import time
import uuid

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "equitymind" / ".env")

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from equitymind.agent import root_agent  # noqa: E402

CASES = pathlib.Path(__file__).with_name("eval_cases.json")


def check_trajectory(called: list[str], groups: list[list[str]], forbidden: list[str]) -> list[str]:
    problems = []
    for bad in forbidden:
        if bad in called:
            problems.append(f"forbidden tool called: {bad}")
    last_index_prev = -1
    for group in groups:
        idxs = []
        for tool in group:
            if tool not in called:
                problems.append(f"missing tool: {tool}")
                continue
            idxs.append(called.index(tool))
        if idxs and min(idxs) < last_index_prev:
            problems.append(f"order violation: {group} started before previous group finished")
        if idxs:
            last_index_prev = max(max(i for i, t in enumerate(called) if t == tool) for tool in group if tool in called)
    return problems


def check_format(text: str, required: list[str], forbidden: list[str]) -> list[str]:
    problems = [f"missing pattern: {p}" for p in required if not re.search(p, text)]
    problems += [f"forbidden pattern present: {p}" for p in forbidden if re.search(p, text)]
    return problems


async def run_case(case: dict) -> dict:
    sessions = InMemorySessionService()
    runner = Runner(app_name="equitymind-eval", agent=root_agent, session_service=sessions)
    session = await sessions.create_session(app_name="equitymind-eval", user_id=f"eval-{uuid.uuid4().hex[:8]}")
    msg = types.Content(role="user", parts=[types.Part(text=case["prompt"])])

    called: list[str] = []
    final = ""
    started = time.perf_counter()
    async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=msg):
        called += [c.name for c in event.get_function_calls() or []]
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False))
    latency = round(time.perf_counter() - started, 2)

    problems = check_trajectory(called, case.get("expected_tools_in_order", []), case.get("forbidden_tools", []))
    problems += check_format(final, case.get("required_patterns", []), case.get("forbidden_patterns", []))
    return {
        "id": case["id"],
        "passed": not problems,
        "problems": problems,
        "tools_called": called,
        "latency_s": latency,
        "response": final,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Run a single case id")
    ap.add_argument("--min-pass-rate", type=float, default=0.0)
    args = ap.parse_args()

    cases = json.loads(CASES.read_text())
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    results = []
    for case in cases:
        try:
            res = await run_case(case)
        except Exception as exc:  # noqa: BLE001
            res = {"id": case["id"], "passed": False, "problems": [f"exception: {exc}"], "tools_called": []}
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        print(f"[{mark}] {res['id']}  tools={res['tools_called']}")
        for p in res["problems"]:
            print(f"       - {p}")

    pass_rate = sum(r["passed"] for r in results) / max(len(results), 1)
    out_dir = pathlib.Path(__file__).with_name("results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"pass_rate": pass_rate, "results": results}, indent=2, ensure_ascii=False))
    print(f"\nPass rate: {pass_rate:.0%}  ->  {out.relative_to(ROOT)}")
    return 0 if pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
