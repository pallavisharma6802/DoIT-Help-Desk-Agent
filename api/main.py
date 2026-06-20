import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

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
from observability import log_full_session, new_session_id, estimate_cost

log = logging.getLogger(__name__)

app = FastAPI(title="DoIT KB Agentic Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session_metrics: dict = {}   # session_id -> metrics dict
_session_state: dict = {}     # session_id -> SessionState



class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_type: str = "student"


class EndSessionRequest(BaseModel):
    session_id: str


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
    complexity: str = ""
    graph_trace: List[dict] = []



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph")
def graph_image():
    png = _agent._graph.get_graph().draw_mermaid_png()
    return Response(content=png, media_type="image/png")


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session_id = body.session_id or new_session_id()
    t0 = time.monotonic()

    existing_session = _session_state.get(session_id)

    try:
        result = _agent.run(body.query, session=existing_session)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("agent.run failed")
        raise HTTPException(status_code=500, detail=str(e))

    _session_state[session_id] = result["session"]

    ttft_ms = (time.monotonic() - t0) * 1000
    session = result["session"]
    cost = estimate_cost(session.total_input_tokens, session.total_output_tokens)

    # Compute per-turn token delta
    prev = _session_metrics.get(session_id, {})
    turn_in_tok  = session.total_input_tokens  - prev.get("total_input_tokens", 0)
    turn_out_tok = session.total_output_tokens - prev.get("total_output_tokens", 0)

    # Accumulate per-turn data for end-session logging
    turns_data = prev.get("turns_data", []) + [{
        "turn":              result["turn"],
        "query":             body.query,
        "answer":            result["answer"],
        "ttft_ms":           round(ttft_ms, 1),
        "in_tok":            turn_in_tok,
        "out_tok":           turn_out_tok,
        "resolved":          result["resolved"],
        "escalated":         result["escalated"],
        "complexity":        result.get("complexity", ""),
        "clf_confidence":    result.get("clf_confidence", 0.0),
        "clf_reasoning":     result.get("clf_reasoning", ""),
        "clf_input_tokens":  result.get("clf_input_tokens", 0),
        "clf_output_tokens": result.get("clf_output_tokens", 0),
        "clf_latency_ms":    result.get("clf_latency_ms", 0.0),
        "kb_citations":      result.get("kb_citations", []),
        "graph_trace":       result.get("graph_trace", []),
    }]

    # Accumulate all KB IDs seen across every turn (seen_kb_ids resets per message)
    all_kb_ids = list({
        kb_id
        for td in turns_data
        for it in td.get("graph_trace", [])
        for kb_id in it.get("kb_ids_fetched", [])
    })

    _session_metrics[session_id] = {
        "session_id":          session_id,
        "user_type":           body.user_type,
        "turns_taken":         len(turns_data),   # number of user messages, not loop count
        "resolved":            result["resolved"],
        "escalated":           result["escalated"],
        "complexity":          result.get("complexity", ""),
        "total_input_tokens":  session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "estimated_cost_usd":  cost,
        "kb_ids_retrieved":    all_kb_ids,
        "turns_data":          turns_data,
    }

    return ChatResponse(
        answer=result["answer"],
        kb_citations=[Citation(**c) for c in result["kb_citations"]],
        turn=result["turn"],
        resolved=result["resolved"],
        escalated=result["escalated"],
        session_id=session_id,
        complexity=result.get("complexity", ""),
        graph_trace=result.get("graph_trace", []),
    )


@app.post("/end-session")
def end_session(body: EndSessionRequest):
    """Log the completed session to Langfuse and clean up local state."""
    session_id = body.session_id
    metrics = _session_metrics.get(session_id)
    state   = _session_state.get(session_id)

    if not metrics:
        raise HTTPException(status_code=404, detail="Session not found")

    log_full_session(session_id, metrics["user_type"], metrics, state)

    # Clean up so the session doesn't linger in memory
    _session_metrics.pop(session_id, None)
    _session_state.pop(session_id, None)

    return {"status": "logged", "session_id": session_id}


_REASONING_MODEL = "llama-3.3-70b-versatile"


@app.post("/agent-assist")
def agent_assist(body: ChatRequest):
    session_id = body.session_id or new_session_id()

    def generate():
        t0 = time.monotonic()
        first_token_sent = False

        try:
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
        "sessions": [
            {k: v for k, v in s.items() if k != "turns_data"}
            for s in _session_metrics.values()
        ],
        "total_sessions": len(_session_metrics),
        "total_cost_usd": round(
            sum(s["estimated_cost_usd"] for s in _session_metrics.values()), 6
        ),
    }
