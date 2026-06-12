# DoIT KB Agentic Assistant

An agentic RAG system that answers UW-Madison IT support questions using the [DoIT Knowledge Base](https://kb.wisc.edu). Built with a LangGraph pipeline, ChromaDB vector search, and a Streamlit chat UI.

---

## How it works

Queries go through a 4-node LangGraph pipeline:

1. **Classify** — Llama 3.1 8B labels the query as `simple` or `complex`
2. **Retrieve** — ChromaDB vector search returns the top-3 KB articles not seen yet
3. **Generate** — Llama 3.3 70B answers using only the retrieved articles, citing KB IDs
4. **Route** — simple queries resolve in one turn; complex queries loop up to 4 turns, then escalate to a human agent

Delta context injection keeps token usage low across turns: already-seen KB articles are never re-injected into the prompt.

---

## Results

- **392 KB articles** scraped and indexed across 30 topic categories (NetID, Duo MFA, O365, VPN, Canvas, Zoom, etc.)
- Simple queries resolve in **1 turn** with at least 1 KB citation
- Complex queries (affiliation changes, multi-step troubleshooting) use up to 4 turns before escalating
- Multi-turn token cost is measurably lower than naive per-turn RAG (verified in test T15)
- Classifier latency under 500ms on Groq free tier
- Streaming SSE agent-assist mode with TTFT under 2s

---

## Tech stack

| Layer | Tools |
|---|---|
| LLM inference | Groq API — Llama 3.1 8B (classifier), Llama 3.3 70B (generation) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Knowledge graph | NetworkX — semantic similarity edges (threshold ≥ 0.60) |
| Orchestration | LangGraph |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit (student mode + agent-assist mode) |
| Observability | Langfuse |
| Visualization | pyvis — interactive HTML graph |

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=...
HF_TOKEN=...
LANGFUSE_PUBLIC_KEY=...   # optional
LANGFUSE_SECRET_KEY=...   # optional
```

**Scrape and index KB articles:**
```bash
python src/ingest.py          # scrape + index
python src/ingest.py scrape   # scrape only
python src/ingest.py index    # index only (requires HF_TOKEN)
```

**Run the app:**
```bash
uvicorn api.main:app --reload          # backend on :8000
streamlit run frontend/app.py          # frontend on :8501
```

Student mode: `http://localhost:8501`  
Agent-assist mode (streaming): `http://localhost:8501?mode=agent`

**Visualize the knowledge graph:**
```bash
python visualize_graph.py   # opens kb_graph.html
```

---

## Tests

```bash
pytest tests/test_suite.py -v
```

20 tests covering the classifier, retriever, context manager, agent pipeline, API endpoints, and a 10-thread load test.
