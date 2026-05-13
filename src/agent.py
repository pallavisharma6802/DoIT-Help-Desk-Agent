"""
LangGraph agentic pipeline for DoIT KB assistant.

Flow:
    query -> classify -> retrieve -> generate -> [route]
                                          |-> retrieve (if unresolved, turn < 4)
                                          |-> END      (if resolved or escalated)

Public API:
    run(query, session=None) -> dict
        {answer, kb_citations, turn, resolved, escalated, session}
"""

import re
import logging
from dataclasses import asdict
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from groq_client import groq_chat
from classifier import classify_query
import retriever as _retriever
from context_manager import (
    new_session, build_turn_payload, mark_resolved, mark_escalated,
    SessionState, SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

_REASONING_MODEL = "llama-3.3-70b-versatile"
_MAX_TURNS = 4
_TOP_K = 3


# ---------------------------------------------------------------------------
# Graph state — flat TypedDict so LangGraph handles merges cleanly
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    complexity: str
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _classify(state: AgentState) -> dict:
    result = classify_query(state["query"])
    return {"complexity": result["complexity"]}


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

    # Citations: any KB-ID pattern in answer, fallback to all retrieved nodes
    cited_ids = set(re.findall(r"KB-(\d+)", answer))
    kb_citations = [
        {"id": n["id"], "url": n["url"]}
        for n in state["kb_nodes"]
        if n["id"] in cited_ids
    ]
    if not kb_citations and state["kb_nodes"]:
        kb_citations = [{"id": n["id"], "url": n["url"]} for n in state["kb_nodes"]]

    # Update conversation history for next turn
    session.history.append({"role": "user", "content": state["query"]})
    session.history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "kb_citations": kb_citations,
        "seen_kb_ids": session.seen_kb_ids,
        "turn_count": session.turn_count,
        "total_input_tokens": session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "history": session.history,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    # Simple queries always done in 1 turn
    if state["complexity"] == "simple":
        return "done"

    # Hit max turns
    if state["turn_count"] >= _MAX_TURNS:
        return "escalate"

    # Retriever exhausted (no more delta nodes)
    if not state["kb_nodes"]:
        return "escalate"

    return "retrieve"


def _mark_resolved(state: AgentState) -> dict:
    return {"resolved": True}


_ESCALATION_ANSWER = (
    "I wasn't able to fully resolve this through the Knowledge Base. "
    "Please contact the DoIT Help Desk directly:\n\n"
    "- **Phone:** 608-264-4357 (608-264-HELP)\n"
    "- **Chat / Email:** https://it.wisc.edu/help\n"
    "- **Walk-in:** 1210 W Dayton St, Madison\n\n"
    "A support agent can look up your account directly and resolve this for you."
)


def _mark_escalated(state: AgentState) -> dict:
    return {"escalated": True, "answer": _ESCALATION_ANSWER, "kb_citations": []}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(query: str, session: SessionState = None) -> dict:
    """
    Run the agentic pipeline for a query.

    Returns:
        {answer, kb_citations, turn, resolved, escalated, session}
    """
    if session is None:
        session = new_session()

    initial: AgentState = {
        "query": query,
        "complexity": "",
        "seen_kb_ids": list(session.seen_kb_ids),
        "turn_count": session.turn_count,
        "resolved": session.resolved,
        "escalated": session.escalated,
        "total_input_tokens": session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "history": list(session.history),
        "kb_nodes": [],
        "answer": "",
        "kb_citations": [],
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
        "answer": final["answer"],
        "kb_citations": final["kb_citations"],
        "turn": final["turn_count"],
        "resolved": final["resolved"],
        "escalated": final["escalated"],
        "session": out_session,
    }
