import base64
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

import requests

log = logging.getLogger(__name__)

_COST_PER_INPUT_TOKEN  = 0.59 / 1_000_000   # llama-3.3-70b-versatile
_COST_PER_OUTPUT_TOKEN = 0.79 / 1_000_000


def _get_credentials():
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host       = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if not public_key or not secret_key:
        return None
    return public_key, secret_key, host.rstrip("/")


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens  * _COST_PER_INPUT_TOKEN +
        output_tokens * _COST_PER_OUTPUT_TOKEN,
        6,
    )


def log_full_session(
    session_id: str,
    user_type: str,
    metrics: dict,
    session_state,
) -> None:
    creds = _get_credentials()

    turns_data  = metrics.get("turns_data", [])
    history     = session_state.history if session_state else []
    last_answer = turns_data[-1]["answer"] if turns_data else ""

    log.info(
        "Logging full session to Langfuse: session=%s turns=%d resolved=%s",
        session_id[:8], len(turns_data), metrics.get("resolved"),
    )

    if creds is None:
        log.warning("Langfuse credentials not set — skipping cloud log")
        return

    public_key, secret_key, host = creds
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "x-langfuse-ingestion-version": "4",
    }

    trace_body = {
        "id":        session_id,
        "name":      session_id,
        "userId":    user_type,
        "sessionId": session_id,
        "input":     {"conversation": history},
        "output":    {"answer": last_answer},
        "metadata": {
            # number of user messages in the session, not the loop count of the last message
            "user_messages":       len(turns_data),
            "resolved":            metrics.get("resolved", False),
            "escalated":           metrics.get("escalated", False),
            "total_input_tokens":  metrics.get("total_input_tokens", 0),
            "total_output_tokens": metrics.get("total_output_tokens", 0),
            "estimated_cost_usd":  metrics.get("estimated_cost_usd", 0),
            # all KB articles retrieved across the whole session
            "kb_ids_retrieved":    metrics.get("kb_ids_retrieved", []),
        },
        "tags": [
            user_type,
            "resolved" if metrics.get("resolved") else "escalated",
        ],
    }

    # -----------------------------------------------------------------------
    # Reconstruct a plausible timeline working backwards from now.
    # Each user message lasted ttft_ms (wall-clock time in the FastAPI handler).
    # We assume ~10s gap between consecutive user messages.
    # -----------------------------------------------------------------------
    _MSG_GAP_MS = 10_000  # assumed think-time between messages
    now_dt = datetime.now(timezone.utc)

    # Compute start/end time for each message
    msg_windows: list[tuple[datetime, datetime]] = []
    cursor_end = now_dt
    for td in reversed(turns_data):
        msg_end_t   = cursor_end
        msg_start_t = msg_end_t - timedelta(milliseconds=td["ttft_ms"])
        msg_windows.insert(0, (msg_start_t, msg_end_t))
        cursor_end  = msg_start_t - timedelta(milliseconds=_MSG_GAP_MS)

    batch = []

    for msg_idx, (td, (msg_start_t, msg_end_t)) in enumerate(
        zip(turns_data, msg_windows), start=1
    ):
        # ------------------------------------------------------------------
        # Parent span — one per user message, covers the full response time.
        # All classify/iteration generations are nested under this span.
        # The "Latency" shown in Langfuse on this span = ttft_ms = total
        # round-trip time for that user message (equivalent to TTFT in
        # non-streaming mode).
        # ------------------------------------------------------------------
        parent_span_id = str(uuid.uuid4())
        batch.append({
            "id":        str(uuid.uuid4()),
            "type":      "span-create",
            "timestamp": msg_start_t.isoformat(),
            "body": {
                "id":        parent_span_id,
                "traceId":   session_id,
                "name":      f"msg-{msg_idx}",
                "startTime": msg_start_t.isoformat(),
                "endTime":   msg_end_t.isoformat(),
                "input":     {"query": td["query"]},
                "output":    {"answer": td["answer"]},
                "metadata": {
                    "msg_index":       msg_idx,
                    "complexity":      td.get("complexity", ""),
                    "resolved":        td["resolved"],
                    "escalated":       td["escalated"],
                    "graph_loops":     td.get("turn", 1),
                    # ttft_ms = total response latency (classify + retrieve + generate)
                    "ttft_ms":         td["ttft_ms"],
                    "kb_citations":    td.get("kb_citations", []),
                    "clf_confidence":  td.get("clf_confidence", 0.0),
                    "clf_reasoning":   td.get("clf_reasoning", ""),
                },
            },
        })

        # ------------------------------------------------------------------
        # Classifier generation — child of the parent span above.
        # Latency = time for llama-3.1-8b to return complexity label.
        # ------------------------------------------------------------------
        clf_latency = td.get("clf_latency_ms", 300)
        clf_start   = msg_start_t
        clf_end     = clf_start + timedelta(milliseconds=clf_latency)
        batch.append({
            "id":        str(uuid.uuid4()),
            "type":      "generation-create",
            "timestamp": clf_start.isoformat(),
            "body": {
                "id":                   str(uuid.uuid4()),
                "traceId":              session_id,
                "parentObservationId":  parent_span_id,
                "name":                 f"msg-{msg_idx}-classify",
                "model":                "llama-3.1-8b-instant",
                "startTime":            clf_start.isoformat(),
                "endTime":              clf_end.isoformat(),
                "input":                {"query": td["query"]},
                "output": {
                    "complexity":  td.get("complexity", ""),
                    "confidence":  td.get("clf_confidence", 0.0),
                    "reasoning":   td.get("clf_reasoning", ""),
                },
                "usage": {
                    "input":     td.get("clf_input_tokens", 0),
                    "output":    td.get("clf_output_tokens", 0),
                    "totalCost": 0.0,
                },
                "metadata": {
                    "node":      "classify",
                    "msg_index": msg_idx,
                },
            },
        })

        # ------------------------------------------------------------------
        # One generation per internal retrieve→generate iteration.
        # Each spans from right after the previous step ends.
        # Latency = time for llama-3.3-70b to return that iteration's answer.
        # ------------------------------------------------------------------
        cursor = clf_end
        for it in td.get("graph_trace", []):
            it_latency = it.get("latency_ms", 0)
            it_start   = cursor
            it_end     = it_start + timedelta(milliseconds=it_latency)
            cursor     = it_end
            iteration_num = it.get("iteration", 1)
            batch.append({
                "id":        str(uuid.uuid4()),
                "type":      "generation-create",
                "timestamp": it_start.isoformat(),
                "body": {
                    "id":                   str(uuid.uuid4()),
                    "traceId":              session_id,
                    "parentObservationId":  parent_span_id,
                    "name":                 f"msg-{msg_idx}-iter-{iteration_num}",
                    "model":                "llama-3.3-70b-versatile",
                    "startTime":            it_start.isoformat(),
                    "endTime":              it_end.isoformat(),
                    "input":                {"query": td["query"]},
                    "output":               {"answer_snippet": it.get("answer_snippet", "")},
                    "usage": {
                        "input":     it.get("in_tok", 0),
                        "output":    it.get("out_tok", 0),
                        "totalCost": estimate_cost(it.get("in_tok", 0), it.get("out_tok", 0)),
                    },
                    "metadata": {
                        "node":            "generate",
                        "msg_index":       msg_idx,
                        "iteration":       iteration_num,
                        "kb_ids_fetched":  it.get("kb_ids_fetched", []),
                        "kb_citations":    it.get("kb_citations", []),
                        "router_decision": it.get("router_decision", ""),
                        "latency_ms":      round(it_latency),
                    },
                },
            })

    try:
        r1 = requests.post(
            f"{host}/api/public/traces",
            headers=headers,
            data=json.dumps(trace_body),
            timeout=10,
        )
        r1.raise_for_status()

        if batch:
            r2 = requests.post(
                f"{host}/api/public/ingestion",
                headers=headers,
                data=json.dumps({"batch": batch}),
                timeout=10,
            )
            r2.raise_for_status()

        log.info(
            "Langfuse: session=%s logged (%d messages, $%.6f)",
            session_id[:8], len(turns_data), metrics.get("estimated_cost_usd", 0),
        )
    except Exception as e:
        log.warning("Langfuse log failed (non-fatal): %s", e)


def new_session_id() -> str:
    return str(uuid.uuid4())
