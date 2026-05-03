"""
Delta context injection for multi-turn sessions.

Each turn injects only KB nodes not seen in prior turns (delta), keeping
the per-turn payload small and the system prompt byte-identical for prefix caching.

Public API:
    new_session() -> SessionState
    build_turn_payload(session, query, kb_nodes) -> list[dict]   # messages list
    mark_resolved(session) -> None
    count_tokens(messages) -> int
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

# ---------------------------------------------------------------------------
# Static system prompt — must be byte-identical across all sessions.
# Do NOT modify at runtime. Structured for prefix cache efficiency.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a UW-Madison DoIT IT support assistant. "
    "Answer the user's question using ONLY the KB articles provided below. "
    "Every response MUST cite at least one KB article by including its ID and URL. "
    "If the provided articles do not contain enough information to answer fully, "
    "say so explicitly — do not guess or hallucinate. "
    "Be concise. Use plain language suitable for a student or staff member."
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    seen_kb_ids: List[str] = field(default_factory=list)
    turn_count: int = 0
    resolved: bool = False
    escalated: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def new_session() -> SessionState:
    return SessionState()


# ---------------------------------------------------------------------------
# Token counting — 4 chars ≈ 1 token (consistent with groq_client estimate)
# ---------------------------------------------------------------------------

def count_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_turn_payload(
    session: SessionState,
    query: str,
    kb_nodes: List[dict],
) -> List[Dict[str, str]]:
    """
    Build the messages list for one turn.

    - System prompt is always identical (prefix cache hit).
    - Only KB nodes not in session.seen_kb_ids are injected (delta).
    - Updates session.seen_kb_ids and session.turn_count in place.

    Returns a messages list ready to pass to groq_client.groq_chat().
    """
    seen_set = set(session.seen_kb_ids)
    delta_nodes = [n for n in kb_nodes if n["id"] not in seen_set]

    # Format delta KB nodes as a compact block
    if delta_nodes:
        kb_block = "\n\n".join(
            f"[KB-{n['id']}] {n['title']}\nURL: {n['url']}\n{n['body'][:600]}"
            for n in delta_nodes
        )
        user_content = f"KB Articles:\n{kb_block}\n\nQuestion: {query}"
    else:
        user_content = f"Question: {query}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # Update session state
    for n in delta_nodes:
        if n["id"] not in seen_set:
            session.seen_kb_ids.append(n["id"])
            seen_set.add(n["id"])
    session.turn_count += 1

    return messages


def mark_resolved(session: SessionState) -> None:
    session.resolved = True


def mark_escalated(session: SessionState) -> None:
    session.escalated = True
