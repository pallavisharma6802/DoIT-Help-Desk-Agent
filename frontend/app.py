"""
Streamlit dual-mode frontend.

Student mode: ?mode=student  (default)
Agent mode:   ?mode=agent

Talks to the FastAPI backend at API_BASE_URL (default: http://localhost:8000).
"""

import os
import sys
import time
import uuid

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# URL param → mode
params = st.query_params
mode = params.get("mode", "student")
is_agent = mode == "agent"

# Page config
st.set_page_config(
    page_title="DoIT KB Assistant" if not is_agent else "DoIT Agent Assist",
    page_icon="🎓" if not is_agent else "🛠️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Styling
ACCENT = "#c5050c"   # UW-Madison red

st.markdown(f"""
<style>
  .header {{
      background: {ACCENT};
      color: white;
      padding: 14px 20px;
      border-radius: 8px;
      margin-bottom: 20px;
  }}
  .header h2 {{ margin: 0; font-size: 1.3rem; }}
  .header p  {{ margin: 0; font-size: 0.85rem; opacity: 0.85; }}
  .citation {{
      background: #f5f5f5;
      border-left: 3px solid {ACCENT};
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 0.82rem;
      margin-top: 6px;
  }}
  .badge-resolved  {{ background:#2ecc71; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; }}
  .badge-escalated {{ background:#e74c3c; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; }}
  .badge-agent     {{ background:#f39c12; color:white; padding:2px 8px; border-radius:10px; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)

# Header
if is_agent:
    st.markdown("""
    <div class="header">
      <h2>🛠️ DoIT Agent Assist</h2>
      <p>Real-time KB lookup during live calls &nbsp;·&nbsp;
         <span class="badge-agent">AGENT MODE</span></p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header">
      <h2>🎓 DoIT Help Desk Assistant</h2>
      <p>Ask any UW-Madison IT question — powered by the Knowledge Base</p>
    </div>""", unsafe_allow_html=True)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            for c in msg["citations"]:
                st.markdown(
                    f'<div class="citation">📄 <b>KB-{c["id"]}</b> &nbsp;'
                    f'<a href="{c["url"]}" target="_blank">{c["url"]}</a></div>',
                    unsafe_allow_html=True,
                )
        if msg.get("meta"):
            m = msg["meta"]
            badge = ""
            if m.get("resolved"):
                badge = '<span class="badge-resolved">✓ Resolved</span>'
            elif m.get("escalated"):
                badge = '<span class="badge-escalated">⚠ Escalated</span>'
            if badge:
                st.markdown(badge, unsafe_allow_html=True)
            complexity_label = f"· {m.get('complexity','')}" if m.get("complexity") else ""
            st.caption(f"Turn {m.get('turn','-')} {complexity_label} · session `{st.session_state.session_id[:8]}`")

# Input
placeholder = (
    "Describe the issue you're troubleshooting..." if is_agent
    else "Ask a UW-Madison IT question..."
)

if query := st.chat_input(placeholder):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        if is_agent:
            # Streaming via /agent-assist
            placeholder_box = st.empty()
            answer_chunks = []
            t0 = time.monotonic()
            first_token_ms = None

            try:
                with requests.post(
                    f"{API_BASE}/agent-assist",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id,
                        "user_type": "agent",
                    },
                    stream=True,
                    timeout=30,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = line.decode().removeprefix("data: ")
                        if chunk == "[DONE]":
                            break
                        if chunk.startswith("[ERROR]"):
                            st.error(chunk)
                            break
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - t0) * 1000
                        answer_chunks.append(chunk)
                        placeholder_box.markdown("".join(answer_chunks) + "▌")

                answer = "".join(answer_chunks)
                placeholder_box.markdown(answer)

                ttft_label = f"TTFT: {first_token_ms:.0f}ms" if first_token_ms else ""
                st.caption(f"Streaming · {ttft_label} · session `{st.session_state.session_id[:8]}`")

            except Exception as e:
                answer = f"API error: {e}"
                st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        else:
            # Full response via /chat
            with st.spinner("Looking up KB articles..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/chat",
                        json={
                            "query": query,
                            "session_id": st.session_state.session_id,
                            "user_type": "student",
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            answer = data["answer"]
            citations = data.get("kb_citations", [])
            graph_trace = data.get("graph_trace", [])
            meta = {
                "turn":        data.get("turn"),
                "resolved":    data.get("resolved"),
                "escalated":   data.get("escalated"),
                "complexity":  data.get("complexity", ""),
                "graph_trace": graph_trace,
            }

            st.markdown(answer)
            for c in citations:
                st.markdown(
                    f'<div class="citation">📄 <b>KB-{c["id"]}</b> &nbsp;'
                    f'<a href="{c["url"]}" target="_blank">{c["url"]}</a></div>',
                    unsafe_allow_html=True,
                )

            badge = ""
            if meta["resolved"]:
                badge = '<span class="badge-resolved">✓ Resolved</span>'
            elif meta["escalated"]:
                badge = '<span class="badge-escalated">⚠ Escalated — human agent recommended</span>'
            if badge:
                st.markdown(badge, unsafe_allow_html=True)

            complexity_label = f"· classified as **{meta['complexity']}**" if meta["complexity"] else ""
            st.caption(f"Turn {meta['turn']} {complexity_label} · session `{st.session_state.session_id[:8]}`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "meta": meta,
            })

# Sidebar
with st.sidebar:
    st.markdown("### Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎓 Student", use_container_width=True,
                     type="primary" if not is_agent else "secondary"):
            st.query_params["mode"] = "student"
            st.rerun()
    with col2:
        if st.button("🛠️ Agent", use_container_width=True,
                     type="primary" if is_agent else "secondary"):
            st.query_params["mode"] = "agent"
            st.rerun()

    st.divider()
    st.markdown("### Session")
    st.code(st.session_state.session_id[:8], language=None)

    if st.button("End & Log Session", use_container_width=True, type="primary"):
        if st.session_state.messages:
            try:
                resp = requests.post(
                    f"{API_BASE}/end-session",
                    json={"session_id": st.session_state.session_id},
                    timeout=10,
                )
                resp.raise_for_status()
                st.success("Session logged to Langfuse.")
            except Exception as e:
                st.error(f"Logging failed: {e}")
        else:
            st.warning("No messages to log.")
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("New session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Categories covered")
    # Dynamically read from scraped KB articles
    try:
        import json
        from pathlib import Path
        from collections import Counter
        kbs_dir = Path(__file__).parent.parent / "data" / "kbs"
        cats = Counter(
            json.loads(f.read_text())["category"]
            for f in kbs_dir.glob("*.json")
        )
        for cat, count in sorted(cats.items()):
            st.markdown(f"- {cat} &nbsp;<span style='color:#888;font-size:11px'>({count})</span>",
                        unsafe_allow_html=True)
        st.caption(f"{sum(cats.values())} articles total")
    except Exception:
        st.markdown("30 topic areas covered")
