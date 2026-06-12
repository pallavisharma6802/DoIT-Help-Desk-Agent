from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a UW-Madison DoIT IT support assistant. "
    "Answer the user's question using ONLY the KB articles provided below. "
    "Every response MUST cite at least one KB article by including its ID and URL. "
    "If the provided articles do not contain enough information to answer fully, "
    "say so explicitly — do not guess or hallucinate. "
    "Be concise. Use plain language suitable for a student or staff member."
)



@dataclass
class SessionState:
    seen_kb_ids: List[str] = field(default_factory=list)
    turn_count: int = 0
    resolved: bool = False
    escalated: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]


def new_session() -> SessionState:
    return SessionState()



def count_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


def build_turn_payload(
    session: SessionState,
    query: str,
    kb_nodes: List[dict],
) -> List[Dict[str, str]]:
    seen_set = set(session.seen_kb_ids)
    delta_nodes = [n for n in kb_nodes if n["id"] not in seen_set]

    if delta_nodes:
        kb_block = "\n\n".join(
            f"[KB-{n['id']}] {n['title']}\nURL: {n['url']}\n{n['body'][:600]}"
            for n in delta_nodes
        )
        user_content = f"KB Articles:\n{kb_block}\n\nQuestion: {query}"
    else:
        user_content = f"Question: {query}"

    # Build messages: system prompt + conversation history (last 4 turns) + current question
    history_window = session.history[-4:] if session.history else []
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history_window
        + [{"role": "user", "content": user_content}]
    )

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
