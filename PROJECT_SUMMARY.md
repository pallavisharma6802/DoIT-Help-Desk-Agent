# DoIT KB Agentic Assistant - Project Summary

## What is Built

A multi-turn conversational IT support assistant for UW-Madison students and staff. It answers questions by searching a Knowledge Base of 392 scraped KB articles, routing queries through a LangGraph agent, and logging everything to Langfuse. The system distinguishes simple factual questions from complex multi-step issues and handles them differently.

---

## Architecture Overview

```
User (Streamlit)
     │
     ▼
FastAPI  /chat
     │
     ▼
LangGraph Agent
  classify → retrieve → generate → route
                          ↑____________|  (loop for complex, unresolved)
     │
     ▼
  resolve or escalate
     │
     ▼
Langfuse  (logged at /end-session)
```

### Stack

| Layer           | Choice                         | Why                                                    |
| --------------- | ------------------------------ | ------------------------------------------------------ |
| LLM             | Groq (llama-3.3-70b-versatile) | Fast inference, free tier                              |
| Classifier LLM  | Groq (llama-3.1-8b-instant)    | Lightweight, cheap for binary classification           |
| Agent framework | LangGraph                      | Built-in state graph, conditional edges, loop support  |
| Vector store    | ChromaDB (persistent)          | Local, no infra needed                                 |
| Embeddings      | HuggingFace (via API)          | Free, no local GPU needed                              |
| Knowledge graph | NetworkX                       | Semantic graph over KB articles for neighbor retrieval |
| Observability   | Langfuse                       | Trace + generation spans per session                   |
| API             | FastAPI                        | Async, clean REST, Pydantic models                     |
| Frontend        | Streamlit                      | Rapid UI, session state, dual student/agent modes      |

---

## Data Pipeline

- **392 KB articles** scraped from kb.wisc.edu across 29 categories (VPN, Duo MFA, O365, Canvas, NetID, Zoom, etc.)
- Each article stored as JSON: `{id, title, body, category, url}`
- Indexed into **ChromaDB** using HuggingFace embeddings
- **NetworkX graph** built on top: nodes = articles, edges = pairs with cosine similarity ≥ 0.60 (top-10 neighbors per article). Isolated nodes connected to their single nearest neighbor as fallback.

---

## LangGraph Agent (`src/agent.py`)

### Graph nodes

| Node       | What it does                                                                                             |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| `classify` | Calls llama-3.1-8b to label query as `simple` or `complex` with confidence + reasoning                   |
| `retrieve` | Queries ChromaDB for top-3 KB articles not yet seen in this loop                                         |
| `generate` | Calls llama-3.3-70b with system prompt + KB articles + conversation history. Produces answer + citations |
| `resolve`  | Marks session as resolved, exits                                                                         |
| `escalate` | Keeps best generated answer, appends DoIT help desk contact info, exits                                  |

### Routing logic (`_route`)

```
if answer cited real KB articles AND answer isn't negative  → done (resolve)
elif complexity == simple                                   → escalate (1 try only)
elif turn_count >= 4                                       → escalate
elif no new KB articles available                          → escalate
else                                                       → retrieve again (loop)
```

**Key design decision:** The router checks `answer_cites_kb` - True only when the model mentioned specific KB-XXXXX IDs in its answer AND the answer doesn't contain negative phrases like "do not contain", "not enough information", "does not address". This prevents prematurely resolving when the model says "KB-12345 does not contain what you need."

### Session state (what resets vs. persists per user message)

| Field                                        | Resets each message | Persists across messages |
| -------------------------------------------- | ------------------- | ------------------------ |
| `turn_count`                                 | ✓                   |                          |
| `seen_kb_ids`                                | ✓                   |                          |
| `resolved` / `escalated`                     | ✓                   |                          |
| `history`                                    |                     | ✓                        |
| `total_input_tokens` / `total_output_tokens` |                     | ✓                        |

This is the fix for multi-turn conversations: without resetting, a complex query that ran 4 loops would leave `turn_count=4` and the next user message would immediately escalate.

### Escalation behavior

When escalation is triggered, the agent **keeps the last generated answer** and appends help desk contact info at the bottom. Previously it threw away the answer entirely and replaced with boilerplate - this was wrong. Now users always see the best KB-based answer found, with human support as a supplement.

---

## Classifier (`src/classifier.py`)

- Model: `llama-3.1-8b-instant`
- Output: `{complexity, confidence, reasoning, latency_ms, input_tokens, output_tokens}`
- **simple**: single-step factual lookup answerable from one KB article (e.g., password reset, eduroam setup)
- **complex**: involves account state changes, affiliation timelines, or likely needs follow-up (e.g., O365 lost after role change, Duo locked out)

---

## Retriever (`src/retriever.py`)

- Embeds query using HuggingFace API
- Queries ChromaDB for `top_k=3` nearest articles, **excluding already-seen IDs**
- On each loop iteration, fetches 3 _new_ articles not seen before in this query
- Also exposes `graph_neighbors()` - BFS up to 2 hops in the NetworkX graph - available for future use

---

## Context Manager (`src/context_manager.py`)

- `build_turn_payload()` constructs the messages array for each LLM call
- Includes system prompt + last 4 turns of conversation history + new KB articles
- Only sends **delta nodes** (new articles not seen before) to avoid token bloat
- Increments `turn_count` and appends to `seen_kb_ids` in place

---

## Token Rate Limiting (`src/groq_client.py`)

- Self-imposed `_TokenGuard`: 14,000 tokens per 60s window (conservative ceiling before Groq's real limits)
- Tracks **actual** tokens (post-call) not estimates (pre-call) - earlier bug was recording estimated tokens before the call, causing the window to inflate
- Retry logic: 3 retries with exponential backoff for TPM (per-minute) errors
- **Daily limit (TPD) errors**: detected by checking for "tokens per day" in the error string - raises `EnvironmentError` immediately instead of retrying uselessly
- FastAPI catches `EnvironmentError` as 503 with a human-readable message

---

## FastAPI Backend (`api/main.py`)

### Endpoints

| Endpoint             | Purpose                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `GET /health`        | Liveness check                                                           |
| `POST /chat`         | Main endpoint - runs the agent, returns answer + citations + graph_trace |
| `POST /end-session`  | Logs full session to Langfuse, clears local state                        |
| `POST /agent-assist` | Streaming endpoint for agent mode (SSE)                                  |
| `GET /graph`         | Returns LangGraph structure as PNG (Mermaid rendered via mermaid.ink)    |
| `GET /metrics`       | Returns all active session metrics + running cost                        |

### Session management

Sessions keyed by `session_id` (UUID). Two in-memory dicts on the server:

- `_session_state`: maps session_id → `SessionState` (history, tokens)
- `_session_metrics`: maps session_id → accumulated turns data for Langfuse

---

## Langfuse Observability (`src/observability.py`)

Logging happens at **end-session** (not per message) - batched and sent to Langfuse Cloud when user clicks "End & Log Session" in the UI.

### What gets logged

For a session with N user messages, Langfuse receives:

```
Trace (session level)
│  metadata: turns_taken, resolved, escalated, complexity,
│            total tokens, cost, kb_ids_retrieved
│
├── msg-1-classify        llama-3.1-8b call, complexity/confidence/reasoning, tokens
├── msg-1-iter-1          llama-3.3-70b, kb_ids_fetched, kb_citations, router_decision, latency_ms
├── msg-1-iter-2          (only if complex and looped)
├── msg-1-answer          full answer, ttft_ms, resolved/escalated, complexity
│
├── msg-2-classify
├── msg-2-iter-1
└── msg-2-answer
```

**Naming fix made:** Originally used `turn-{td['turn']}-*` which meant "turn 1" and "turn 4" referred to the internal loop count, not the user message number. Changed to `msg-{index}-*` so Langfuse shows msg-1, msg-2 etc. in sequence.

**TTFT fix:** Per-iteration latency (`latency_ms` from `groq_chat`) is now stored in `graph_trace` and used as `endTime - startTime` for each iteration span. Previously all spans had the same timestamp.

---

## Streamlit Frontend (`frontend/app.py`)

### Two modes

- **Student mode** (default): full async chat via `/chat`, shows citations + resolved/escalated badge
- **Agent mode**: streaming via `/agent-assist` SSE, shows TTFT, designed for live helpdesk calls

### Sidebar

- Mode switcher
- Session ID display
- "End & Log Session" button → POSTs to `/end-session`, triggers Langfuse logging
- "New session" button → resets session ID and message history
- Categories covered (dynamically read from `data/kbs/*.json`)

### Per-message display

- Answer text
- KB citation cards (article ID + URL)
- Resolved ✓ or Escalated ⚠ badge
- Caption: `Turn N · classified as [simple/complex] · session [id]`

---

## Key Bugs Fixed

| Bug                                                      | Root cause                                                                                                                         | Fix                                                                                           |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Complex queries always escalated                         | `_route` had no "done" path for complex queries - always looped to max turns                                                       | Added `answer_cites_kb` check: if model cited real KB articles positively, stop and resolve   |
| "Resolved" with negative answers                         | `answer_cites_kb` was True whenever KB IDs appeared in text, even in "KB-X does not contain..."                                    | Added `_NEGATIVE` phrase check - answer must cite KB AND not contain negative phrases         |
| Multi-turn: second message immediately escalates         | `turn_count=4` and `escalated=True` from previous query bled into next message                                                     | Reset `turn_count`, `seen_kb_ids`, `resolved`, `escalated` at the start of each `agent.run()` |
| Escalation threw away the answer                         | `_mark_escalated` replaced `state["answer"]` with boilerplate                                                                      | Now keeps last generated answer and appends help desk note as suffix                          |
| Token guard too conservative                             | `_TOKEN_WINDOW_LIMIT = 5000` (below Groq's actual 6k TPM limit), tracked estimates not actuals                                     | Raised to 14,000; split into `check()` before call and `record()` after with actual tokens    |
| Daily rate limit caused 500 crash                        | `groq.RateLimitError` caught as generic exception, server crashed                                                                  | Detect TPD errors by string match, raise `EnvironmentError`, FastAPI returns 503 with message |
| Langfuse spans all named "turn-4" / "turn-1"             | Used `td['turn']` (internal loop count) for naming                                                                                 | Changed to `enumerate(turns_data)` → `msg-1`, `msg-2`, etc.                                   |
| No per-iteration latency in Langfuse                     | `latency_ms` from `groq_chat` was not stored                                                                                       | Added to `iteration_record` in `graph_trace`, used for span timestamps                        |
| Streamlit sidebar not visible                            | `st.set_page_config` missing `initial_sidebar_state` - defaulted to `"auto"` which collapses on narrow viewports                   | Added `initial_sidebar_state="expanded"` to `set_page_config`                                 |
| `classify_query` called but discarded in `/agent-assist` | `clf = classify_query(body.query)` result never used - burned tokens on every streaming request                                    | Removed the dead call entirely                                                                |
| `groq_stream` bypassed rate-limit guard                  | `groq_stream` never called `_token_guard.check()` or `_token_guard.record()` - streaming requests were invisible to the TPM window | Added pre-call `check()` and post-stream `record()` to `groq_stream`                          |
| `_REASONING_MODEL` defined after use                     | Constant defined at line 202, used at line 186 inside `agent_assist` - works at runtime but fragile                                | Moved definition above `agent_assist`                                                         |

---

## Retrieval Strategy: Semantic Search vs. Keyword Search

### What the system currently does

Retrieval is **semantic search only** - query is embedded via HuggingFace (`all-MiniLM-L6-v2`) and matched against ChromaDB vectors using cosine similarity. The NetworkX knowledge graph is built but `graph_neighbors()` is **never called in the agent loop** - it exists as dead code for now.

### Why semantic search fits this project

| Reason                           | Example                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| Users don't use technical jargon | "my wifi won't work" → correctly retrieves eduroam articles                          |
| Synonyms handled naturally       | "forgot password" = "password recovery" = "reset credentials"                        |
| Terse, fragmented queries work   | Test queries like "duo setup new phone" have no grammar - embeddings handle it       |
| Cross-concept matching           | "can't log in after leaving UW" can pull O365 deactivation AND NetID expiry together |

### Cons - where pure semantic search hurts

- **Exact-match failures**: product names like `GlobalProtect` or `KB-148522` get approximated through embedding space when a direct match would be more reliable.
- **General-purpose embedding model**: `all-MiniLM-L6-v2` is not fine-tuned on IT support. Terms like `eduroam`, `WiscVPN`, `Workspace ONE` may have weak or miscalibrated embeddings.
- **False positives from topical proximity**: "can't log in" is semantically close to every auth-related article - top-3 results may not be the right 3.
- **No hard filtering**: keyword search allows constraints like _must contain "GlobalProtect" AND "macOS"_; semantic search only scores.
- **Embedding latency**: every query requires a HuggingFace API round-trip before ChromaDB runs. BM25 keyword search would run locally in milliseconds.

### The right long-term approach: hybrid retrieval

The industry standard for RAG systems is **BM25 + semantic → Reciprocal Rank Fusion (RRF)**:

```
BM25 keyword score  +  semantic score  →  RRF  →  top-k
```

Keyword search catches exact product names and article IDs; semantic search handles paraphrases and vague descriptions. Combined recall is almost always higher than either alone.

### Graph traversal on complex query re-retrieval

The NetworkX graph (edges = cosine similarity ≥ 0.60) is now wired into the agent loop for complex queries. The retrieval strategy is:

```
iteration 1 (any complexity)   → semantic search (ChromaDB)
iteration 2+ AND complex       → graph traversal from seen articles (max_hops=1)
                                  fallback to semantic search if graph yields nothing
iteration 2+ AND simple        → semantic search (simple queries escalate after 1 try anyway)
```

On re-retrieval, `graph_neighbors()` does BFS from every previously seen article, collecting direct neighbors not yet seen. This finds articles that are _semantically adjacent_ to what the model already tried - rather than re-running the same vector query and risking the same results. `retrieve_by_ids()` (added to `retriever.py`) loads those articles from disk by ID.

---

## Current Limits / Known Constraints

- **Groq free tier**: 100k tokens/day for llama-3.3-70b-versatile. Heavy multi-turn testing exhausts this quickly.
- **`_MAX_TURNS = 4`**, **`_TOP_K = 3`**: agent searches at most 12 unique KB articles per user message before escalating.
- **Langfuse logging is end-of-session only**: no real-time streaming of spans. This is by design to avoid per-message API overhead.
- **Session state is in-memory** on the FastAPI server: restarting the API clears all active sessions.
- **No authentication**: the app has no login - anyone who can reach the Streamlit URL can use it.

---

## Running the App

```bash
# Terminal 1 - Backend
cd "DoIT KB Agentic Assistant"
PYTHONPATH=src uvicorn api.main:app --port 8000 --log-level info

# Terminal 2 - Frontend
streamlit run frontend/app.py --server.port 8501


# Get the architecture diagram
Run  FastAPI server and hit the /graph endpoint
curl http://localhost:8000/graph --output architecture.png

```

Open `http://localhost:8501` for the UI. API docs at `http://localhost:8000/docs`.
