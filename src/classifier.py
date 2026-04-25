"""
Query complexity classifier using Groq Llama 3.2 3B.

All Groq calls go through groq_client.groq_chat — never the raw SDK.

Simple  = single-step factual lookup (password reset, how to connect to VPN).
Complex = multi-condition, involves timeline, account state change, or
          follow-up troubleshooting likely.
"""

import json
import logging
from typing import Dict, Any

from groq_client import groq_chat

log = logging.getLogger(__name__)

_MODEL = "llama-3.1-8b-instant"

# Static system prompt — byte-identical every call for prefix cache efficiency.
# Do NOT modify this string at runtime.
_SYSTEM_PROMPT = (
    "You are a query complexity classifier for a UW-Madison IT support system. "
    "Classify the user's IT support query as either 'simple' or 'complex'.\n\n"
    "simple: single-step factual lookup answerable from one KB article. "
    "Examples: password reset steps, how to install VPN, eduroam setup.\n\n"
    "complex: requires multiple KB articles, involves account state changes, "
    "affiliation or eligibility timelines, or likely needs follow-up troubleshooting. "
    "Examples: O365 access lost after role change, Duo locked out with no backup device, "
    "forwarding not working after account deactivation.\n\n"
    "Respond with ONLY valid JSON in this exact format, no other text:\n"
    '{"complexity": "simple" | "complex", "confidence": <float 0.0-1.0>, '
    '"reasoning": "<one sentence>"}'
)


def classify_query(query: str) -> Dict[str, Any]:
    """
    Classify a user query as simple or complex.

    Returns:
        {"complexity": "simple"|"complex", "confidence": float, "reasoning": str}

    Raises:
        EnvironmentError  – GROQ_API_KEY not set
        ValueError        – model returned unparseable JSON
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    result = groq_chat(
        model=_MODEL,
        messages=messages,
        max_tokens=100,
        temperature=0.0,
    )

    log.info(
        "classifier | latency_ms=%.0f input_tokens=%d output_tokens=%d",
        result["latency_ms"], result["input_tokens"], result["output_tokens"],
    )

    raw = result["content"].strip()

    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Classifier returned non-JSON response: {raw!r}") from exc

    complexity = parsed.get("complexity", "").lower()
    if complexity not in ("simple", "complex"):
        raise ValueError(f"Unexpected complexity value: {complexity!r}")

    return {
        "complexity": complexity,
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": str(parsed.get("reasoning", "")),
        "latency_ms": result["latency_ms"],
    }
