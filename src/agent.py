import re
import logging
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from groq_client import groq_chat
from classifier import classify_query
import retriever as _retriever
from context_manager import (
    new_session, build_turn_payload,
    SessionState, SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

_REASONING_MODEL = "llama-3.3-70b-versatile"
_MAX_TURNS = 4
_TOP_K = 3


class AgentState(TypedDict):
    query: str
    complexity: str
    clf_confidence: float
    clf_reasoning: str
    clf_input_tokens: int
    clf_output_tokens: int
    clf_latency_ms: float
    # Session fields (flat)
    seen_kb_ids: List[str]
    turn_count: int
    resolved: bool
    escalated: bool
    total_input_tokens: int
    total_output_tokens: int
    history: List[dict]
    # Per-turn
    kb_nodes: List[dict]
    answer: str
    kb_citations: List[dict]
    answer_cites_kb: bool  # True if model mentioned KB-XXXXX ids in answer text
    # Graph trace: one entry per internal retrieve→generate iteration
    graph_trace: List[dict]



def _classify(state: AgentState) -> dict:
    result = classify_query(state["query"])
    log.info(
        "classifier | complexity=%s confidence=%.2f reasoning=%s",
        result["complexity"], result["confidence"], result["reasoning"],
    )
    return {
        "complexity":        result["complexity"],
        "clf_confidence":    result["confidence"],
        "clf_reasoning":     result["reasoning"],
        "clf_input_tokens":  result.get("input_tokens", 0),
        "clf_output_tokens": result.get("output_tokens", 0),
        "clf_latency_ms":    result.get("latency_ms", 0.0),
    }


def _retrieve(state: AgentState) -> dict:
    nodes = _retriever.retrieve(
        state["query"],
        seen_kb_ids=state["seen_kb_ids"],
        top_k=_TOP_K,
    )
    return {"kb_nodes": nodes}


def _generate(state: AgentState) -> dict:
    # Reconstruct session from flat state so context_manager can update it
    session = SessionState(
        seen_kb_ids=list(state["seen_kb_ids"]),
        turn_count=state["turn_count"],
        resolved=state["resolved"],
        escalated=state["escalated"],
        total_input_tokens=state["total_input_tokens"],
        total_output_tokens=state["total_output_tokens"],
        history=list(state["history"]),
    )

    messages = build_turn_payload(session, state["query"], state["kb_nodes"])

    result = groq_chat(
        model=_REASONING_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )

    answer = result["content"]
    session.total_input_tokens += result["input_tokens"]
    session.total_output_tokens += result["output_tokens"]

    # cited_ids: KB IDs the model actually mentioned in its answer text
    cited_ids = set(re.findall(r"KB-(\d+)", answer))
    kb_citations = [
        {"id": n["id"], "url": n["url"]}
        for n in state["kb_nodes"]
        if n["id"] in cited_ids
    ]
    if not kb_citations and state["kb_nodes"]:
        kb_citations = [{"id": n["id"], "url": n["url"]} for n in state["kb_nodes"]]

    # True only when citations are present AND the model isn't saying it couldn't answer.
    # The model often mentions KB-XXXXX even in negative answers like
    # "KB-12345 does not contain enough information" — those should keep looping.
    _NEGATIVE = (
        "do not contain", "does not contain", "not enough information",
        "does not address", "not directly address", "no direct information",
        "does not mention", "does not provide", "cannot answer",
        "unable to answer", "no information", "cannot find",
    )
    answer_lower = answer.lower()
    answer_cites_kb = bool(cited_ids) and not any(p in answer_lower for p in _NEGATIVE)

    session.history.append({"role": "user", "content": state["query"]})
    session.history.append({"role": "assistant", "content": answer})

    iteration_record = {
        "iteration":      session.turn_count,
        "kb_ids_fetched": [n["id"] for n in state["kb_nodes"]],
        "kb_citations":   kb_citations,
        "in_tok":         result["input_tokens"],
        "out_tok":        result["output_tokens"],
        "latency_ms":     result["latency_ms"],
        "answer_snippet": answer[:120],
    }

    return {
        "answer":          answer,
        "kb_citations":    kb_citations,
        "answer_cites_kb": answer_cites_kb,
        "seen_kb_ids":     session.seen_kb_ids,
        "turn_count":      session.turn_count,
        "total_input_tokens":  session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "history":    session.history,
        "graph_trace": list(state.get("graph_trace") or []) + [iteration_record],
    }



def _route(state: AgentState) -> str:
    if state.get("answer_cites_kb"):
        # Answer is grounded — model cited real KB content and answered positively
        log.info("router | substantive KB answer at turn %d → done", state["turn_count"])
        decision = "done"
    elif state["complexity"] == "simple":
        # Simple query but answer was negative (KB didn't have it) → escalate after 1 try
        log.info("router | simple query, no substantive answer → escalate")
        decision = "escalate"
    elif state["turn_count"] >= _MAX_TURNS:
        log.info("router | turn_count=%d reached max → escalate", state["turn_count"])
        decision = "escalate"
    elif not state["kb_nodes"]:
        log.info("router | no new KB nodes at turn %d → escalate", state["turn_count"])
        decision = "escalate"
    else:
        log.info("router | complex, no substantive answer yet, turn %d → retrieve again", state["turn_count"])
        decision = "retrieve"

    # Stamp router decision on the last graph_trace entry
    trace = list(state.get("graph_trace") or [])
    if trace:
        trace[-1]["router_decision"] = decision

    return decision


def _mark_resolved(state: AgentState) -> dict:
    return {"resolved": True}


_ESCALATION_SUFFIX = (
    "\n\n---\n**Need more help?** A DoIT support agent can look up your account directly:\n"
    "- **Phone:** 608-264-4357 (608-264-HELP)\n"
    "- **Chat / Email:** https://it.wisc.edu/help\n"
    "- **Walk-in:** 1210 W Dayton St, Madison"
)

_ESCALATION_FALLBACK = (
    "I couldn't find a specific resolution in the Knowledge Base for this issue.\n\n"
    "Please contact the DoIT Help Desk directly:\n"
    "- **Phone:** 608-264-4357 (608-264-HELP)\n"
    "- **Chat / Email:** https://it.wisc.edu/help\n"
    "- **Walk-in:** 1210 W Dayton St, Madison"
)


def _mark_escalated(state: AgentState) -> dict:
    last_answer = state.get("answer", "").strip()
    if last_answer:
        # Keep the best answer found and append the help desk note
        answer = last_answer + _ESCALATION_SUFFIX
        kb_citations = state.get("kb_citations", [])
    else:
        answer = _ESCALATION_FALLBACK
        kb_citations = []
    return {"escalated": True, "answer": answer, "kb_citations": kb_citations}



def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify", _classify)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("resolve", _mark_resolved)
    g.add_node("escalate", _mark_escalated)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges(
        "generate",
        _route,
        {"retrieve": "retrieve", "done": "resolve", "escalate": "escalate"},
    )
    g.add_edge("resolve", END)
    g.add_edge("escalate", END)

    return g.compile()


_graph = _build_graph()



def run(query: str, session: SessionState = None) -> dict:
    if session is None:
        session = new_session()

    initial: AgentState = {
        "query": query,
        "complexity": "",
        "clf_confidence":  0.0,
        "clf_reasoning":   "",
        "clf_input_tokens":  0,
        "clf_output_tokens": 0,
        "clf_latency_ms":    0.0,
        # Reset per-query state so each user message gets a fresh loop budget
        # and can access all KB articles again
        "seen_kb_ids": [],
        "turn_count": 0,
        "resolved": False,
        "escalated": False,
        # Carry over cross-query state: conversation history and running token totals
        "total_input_tokens": session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "history": list(session.history),
        "kb_nodes": [],
        "answer": "",
        "kb_citations": [],
        "answer_cites_kb": False,
        "graph_trace": [],
    }

    final = _graph.invoke(initial)

    out_session = SessionState(
        seen_kb_ids=final["seen_kb_ids"],
        turn_count=final["turn_count"],
        resolved=final["resolved"],
        escalated=final["escalated"],
        total_input_tokens=final["total_input_tokens"],
        total_output_tokens=final["total_output_tokens"],
        history=final["history"],
    )

    return {
        "answer":           final["answer"],
        "kb_citations":     final["kb_citations"],
        "turn":             final["turn_count"],
        "resolved":         final["resolved"],
        "escalated":        final["escalated"],
        "complexity":       final["complexity"],
        "clf_confidence":    final["clf_confidence"],
        "clf_reasoning":     final["clf_reasoning"],
        "clf_input_tokens":  final["clf_input_tokens"],
        "clf_output_tokens": final["clf_output_tokens"],
        "clf_latency_ms":    final["clf_latency_ms"],
        "graph_trace":      final["graph_trace"],
        "session":          out_session,
    }
