"""PII detection & redaction.

Applied in two places:

1. **Logs** – ``observability.JsonFormatter`` scrubs every log message and
   every (nested) structured field before it is written, so PII never reaches
   Cloud Logging even if a developer logs raw user input.
2. **Model input** – ``redact_pii_in_request`` (a ``before_model_callback``)
   scrubs user-authored text before it is sent to Gemini. A stock analyst never
   needs a user's card number or national ID to evaluate a ticker.

Patterns are deliberately conservative to avoid mangling financial data
(tickers, prices, ratios, market caps).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums))
    return total % 10 == 0


# Order matters: more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API_KEY", re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{35}|sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{36})\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("IN_AADHAAR", re.compile(r"(?<!\d)(?<!\d[ \-])[2-9]\d{3}[ \-]\d{4}[ \-]\d{4}(?![ \-]?\d)")),
    ("IN_PAN", re.compile(r"\b[A-Z]{3}[PCHABGJLFT][A-Z]\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?<![\w.])(?<!\d[ \-])\+?\(?\d[\d \-()]{8,16}\d(?![\w.])(?![ \-]\d)")),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]


def redact(text: str) -> tuple[str, Counter]:
    """Return (redacted_text, Counter of PII types found)."""
    counts: Counter = Counter()
    if not text:
        return text, counts

    for label, pattern in _PATTERNS:

        def _sub(m: re.Match[str], label: str = label) -> str:
            raw = m.group(0)
            if label == "CREDIT_CARD":
                digits = re.sub(r"\D", "", raw)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return raw
            if label == "PHONE" and not (10 <= len(re.sub(r"\D", "", raw)) <= 13):
                return raw
            counts[label] += 1
            return f"[REDACTED_{label}]"

        text = pattern.sub(_sub, text)
    return text, counts


def redact_text(text: str) -> str:
    return redact(text)[0]


def redact_obj(obj: Any, _depth: int = 0) -> Any:
    """Recursively redact strings inside dicts/lists/tuples (for structured logs)."""
    if _depth > 6:
        return obj
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v, _depth + 1) for v in obj)
    return obj


def redact_pii_in_request(callback_context: Any, llm_request: Any) -> None:
    """before_model_callback: scrub PII from user-authored text parts sent to the LLM."""
    total: Counter = Counter()
    for content in getattr(llm_request, "contents", None) or []:
        if getattr(content, "role", None) != "user":
            continue
        for part in content.parts or []:
            if getattr(part, "text", None):
                part.text, found = redact(part.text)
                total.update(found)
    if total:
        try:
            callback_context.state["obs:pii_redactions"] = (
                callback_context.state.get("obs:pii_redactions") or 0
            ) + sum(total.values())
        except Exception:  # noqa: BLE001
            pass
        from .observability import logger  # local import avoids a cycle

        logger.warning(
            "pii_redacted_from_model_input",
            extra={
                "event": "pii_redacted",
                "pii_types": dict(total),
                "agent": getattr(callback_context, "agent_name", None),
            },
        )
    return None
