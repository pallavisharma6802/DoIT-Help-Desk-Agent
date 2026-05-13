"""
FastAPI backend for DoIT KB Agentic Assistant.

Endpoints:
    GET  /health         — liveness check
    POST /chat           — full agent pipeline (student mode)
    POST /agent-assist   — streaming SSE response for low TTFT (agent mode)
    GET  /metrics        — per-session token counts and cost estimates
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load .env so the server works when started with plain `uvicorn api.main:app`
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent as _agent
from groq_client import groq_stream
from classifier import classify_query
from retriever import retrieve as kb_retrieve
from context_manager import new_session, build_turn_payload
from observability import log_session, new_session_id, estimate_cost

log = logging.getLogger(__name__)

app = FastAPI(title="DoIT KB Agentic Assistant")

# In-memory stores
_session_metrics: dict = {}   # session_id -> metrics dict
_session_state: dict = {}     # session_id -> SessionState (persists across messages)

_REASONING_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_type: str = "student"   # "student" | "agent"


class Citation(BaseModel):
    id: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    kb_citations: List[Citation]
    turn: int
    resolved: bool
    escalated: bool
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session_id = body.session_id or new_session_id()
    t0 = time.monotonic()

    # Reuse existing session so history carries across messages
    existing_session = _session_state.get(session_id)

    try:
        result = _agent.run(body.query, session=existing_session)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("agent.run failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Persist session for next message
    _session_state[session_id] = result["session"]

    ttft_ms = (time.monotonic() - t0) * 1000

    session = result["session"]
    cost = estimate_cost(session.total_input_tokens, session.total_output_tokens)

    _session_metrics[session_id] = {
        "session_id":          session_id,
        "user_type":           body.user_type,
        "query":               body.query,
        "turns_taken":         result["turn"],
        "resolved":            result["resolved"],
        "escalated":           result["escalated"],
        "total_input_tokens":  session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "estimated_cost_usd":  cost,
        "ttft_ms":             [round(ttft_ms, 1)],
        "kb_ids_retrieved":    session.seen_kb_ids,
    }

    log_session(session_id, body.user_type, body.query, result, [ttft_ms])

    return ChatResponse(
        answer=result["answer"],
        kb_citations=[Citation(**c) for c in result["kb_citations"]],
        turn=result["turn"],
        resolved=result["resolved"],
        escalated=result["escalated"],
        session_id=session_id,
    )


@app.post("/agent-assist")
def agent_assist(body: ChatRequest):
    """
    Streaming SSE endpoint for agent-assist mode.
    Classifies the query, retrieves delta KB nodes, then streams
    the 70B response token by token for low TTFT.

    Event format: data: <token>\n\n
    Final event:  data: [DONE]\n\n
    """
    session_id = body.session_id or new_session_id()

    def generate():
        t0 = time.monotonic()
        first_token_sent = False

        try:
            clf = classify_query(body.query)
            nodes = kb_retrieve(body.query, top_k=3)
            session = new_session()
            messages = build_turn_payload(session, body.query, nodes)

            for chunk in groq_stream(_REASONING_MODEL, messages, max_tokens=512):
                if not first_token_sent:
                    ttft_ms = (time.monotonic() - t0) * 1000
                    log.info("agent-assist TTFT: %.0fms session=%s", ttft_ms, session_id)
                    first_token_sent = True
                yield f"data: {chunk}\n\n"

        except Exception as e:
            log.error("agent-assist stream error: %s", e)
            yield f"data: [ERROR] {e}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    return {
        "sessions": list(_session_metrics.values()),
        "total_sessions": len(_session_metrics),
        "total_cost_usd": round(
            sum(s["estimated_cost_usd"] for s in _session_metrics.values()), 6
        ),
    }
