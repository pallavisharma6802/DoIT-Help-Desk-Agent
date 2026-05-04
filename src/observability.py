"""
Langfuse observability wrapper.

Logs every session to Langfuse (free tier). Fails silently if
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set — the rest
of the system keeps working.

Per-session fields logged:
    session_id, user_type, query
    turns_taken, resolved, escalated
    total_input_tokens, total_output_tokens, estimated_cost_usd
    ttft_ms (list, one per turn), kb_ids_retrieved (list)

Usage:
    from observability import log_session
    log_session(session_id, user_type, query, result, ttft_ms_per_turn)
"""

import logging
import os
import uuid
from typing import List, Optional

log = logging.getLogger(__name__)

# Groq free tier approximate pricing (as of 2025)
# llama-3.1-8b-instant:  $0.05 / 1M input,  $0.08 / 1M output
# llama-3.3-70b-versatile: $0.59 / 1M input, $0.79 / 1M output
_COST_PER_INPUT_TOKEN  = 0.59 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 0.79 / 1_000_000


def _get_client():
    public_key  = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key  = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host        = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        return None

    try:
        from langfuse import Langfuse
        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as e:
        log.warning("Langfuse init failed: %s", e)
        return None


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens  * _COST_PER_INPUT_TOKEN +
        output_tokens * _COST_PER_OUTPUT_TOKEN,
        6,
    )


def log_session(
    session_id: str,
    user_type: str,                  # "student" | "agent"
    query: str,
    result: dict,                    # output of agent.run()
    ttft_ms_per_turn: List[float],   # one entry per generate call
) -> None:
    """
    Log a completed session to Langfuse.
    Silently no-ops if Langfuse credentials are missing.
    """
    client = _get_client()

    session  = result.get("session")
    in_tok   = session.total_input_tokens  if session else 0
    out_tok  = session.total_output_tokens if session else 0
    cost     = estimate_cost(in_tok, out_tok)
    kb_ids   = session.seen_kb_ids if session else []

    payload = {
        "session_id":           session_id,
        "user_type":            user_type,
        "query":                query,
        "turns_taken":          result.get("turn", 0),
        "resolved":             result.get("resolved", False),
        "escalated":            result.get("escalated", False),
        "total_input_tokens":   in_tok,
        "total_output_tokens":  out_tok,
        "estimated_cost_usd":   cost,
        "ttft_ms":              ttft_ms_per_turn,
        "kb_ids_retrieved":     kb_ids,
    }

    log.info("session_log %s", payload)

    if client is None:
        return

    try:
        trace = client.trace(
            id=session_id,
            name="doit-kb-session",
            user_id=user_type,
            input={"query": query},
            output={"answer": result.get("answer", "")},
            metadata={
                "turns_taken":         payload["turns_taken"],
                "resolved":            payload["resolved"],
                "escalated":           payload["escalated"],
                "total_input_tokens":  in_tok,
                "total_output_tokens": out_tok,
                "estimated_cost_usd":  cost,
                "ttft_ms":             ttft_ms_per_turn,
                "kb_ids_retrieved":    kb_ids,
            },
        )
        client.flush()
    except Exception as e:
        log.warning("Langfuse log failed (non-fatal): %s", e)


def new_session_id() -> str:
    return str(uuid.uuid4())
