"""
DoIT KB Agentic Assistant — test suite.
Never delete or skip a test. Add new tests only.

Run: pytest tests/test_suite.py -v
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Make src/ and api/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

# Environment: load .env if python-dotenv is available, otherwise rely on
# the shell environment having the keys set.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # Keys must be set in the environment directly


# ===========================================================================
# T01 – T03: Classifier tests
# ===========================================================================

@pytest.fixture(scope="module")
def classifier():
    from classifier import classify_query
    return classify_query


class TestClassifier:

    def test_T01_simple_query(self, classifier):
        """T01: Simple query → complexity == 'simple'."""
        result = classifier("how do I reset my NetID password")
        assert result["complexity"] == "simple", (
            f"Expected 'simple', got {result['complexity']!r}. "
            f"Reasoning: {result['reasoning']}"
        )

    def test_T02_complex_query(self, classifier):
        """T02: Complex multi-condition query → complexity == 'complex'."""
        result = classifier(
            "my O365 stopped working after my affiliation changed "
            "and forwarding isn't set up"
        )
        assert result["complexity"] == "complex", (
            f"Expected 'complex', got {result['complexity']!r}. "
            f"Reasoning: {result['reasoning']}"
        )

    def test_T03_latency_under_500ms(self, classifier):
        """T03: Classifier (including retry overhead) returns in under 500ms
        on a non-rate-limited call measured wall-clock from the caller side.
        The groq_client 2-second gap enforcer applies between calls, so this
        test is run in isolation — the 2-second gap is between Groq requests,
        not between test runs. We measure only the classify_query() call itself
        which on a warm, non-rate-limited path should be well under 500ms for
        a 3B model on Groq's infrastructure.
        """
        # Brief sleep to ensure the 2-second inter-request gap in groq_client
        # is satisfied from the previous test so this call isn't queued.
        time.sleep(2.1)
        t0 = time.monotonic()
        result = classifier("how do I connect to eduroam WiFi")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500, (
            f"Classifier took {elapsed_ms:.0f}ms — exceeds 500ms budget. "
            "Check Groq free-tier latency or retry logic."
        )
        assert result["complexity"] in ("simple", "complex")


# ===========================================================================
# T04 – T07: Retriever tests
# ===========================================================================

_DATA_DIR = Path(__file__).parent.parent / "data" / "kbs"


@pytest.fixture(scope="module")
def test_chroma_collection(tmp_path_factory):
    """Build a test ChromaDB from local KB JSON files using ChromaDB's built-in EF."""
    import chromadb

    tmp = tmp_path_factory.mktemp("chroma_test")
    client = chromadb.PersistentClient(path=str(tmp))
    # No embedding_function specified → chromadb uses DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX)
    col = client.create_collection("kb_articles", metadata={"hnsw:space": "cosine"})

    ids, documents, metadatas = [], [], []
    for f in _DATA_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        ids.append(d["id"])
        documents.append(f"{d['title']}\n\n{d['body']}")
        metadatas.append({
            "title": d["title"],
            "category": d["category"],
            "url": d["url"],
        })
    col.add(ids=ids, documents=documents, metadatas=metadatas)
    return col


class TestRetriever:

    def test_T04_o365_query_returns_o365_article(self, test_chroma_collection):
        """T04: O365 deactivation query retrieves at least 1 O365 category article."""
        import retriever
        results = retriever.retrieve(
            "O365 deactivation Microsoft email access",
            _collection=test_chroma_collection,
            top_k=3,
        )
        assert results, "retrieve() returned no results"
        categories = [r["category"] for r in results]
        assert "O365" in categories, (
            f"No O365 article in top-3 results. Got categories: {categories}"
        )

    def test_T05_retrieved_nodes_have_body_and_url(self, test_chroma_collection):
        """T05: All retrieved nodes have non-empty body and valid UW KB URL."""
        import retriever
        results = retriever.retrieve(
            "NetID password reset",
            _collection=test_chroma_collection,
            top_k=3,
        )
        assert results, "retrieve() returned no results"
        for node in results:
            assert node["body"], f"Empty body for node {node['id']}"
            assert node["url"].startswith("https://kb.wisc.edu"), (
                f"Invalid URL for node {node['id']}: {node['url']!r}"
            )

    def test_T06_graph_reaches_affiliation_node_in_2_hops(self):
        """T06: O365 deactivation article reaches an affiliation-change article in ≤2 hops."""
        import retriever
        # 79454 = "Leaving the University - Deactivation Notifications for Microsoft 365" (O365)
        neighbors = retriever.graph_neighbors("79454", max_hops=2)
        assert neighbors, "graph_neighbors returned nothing for article 79454"
        G = retriever.build_graph()
        affiliation_neighbors = [
            nid for nid in neighbors
            if "affiliation" in G.nodes[nid].get("title", "").lower()
        ]
        assert affiliation_neighbors, (
            f"No affiliation-related article reachable from 79454 within 2 hops. "
            f"Sample neighbors: {neighbors[:10]}"
        )

    def test_T07_delta_retriever_excludes_seen_ids(self, test_chroma_collection):
        """T07: retrieve() never returns a KB ID already in seen_kb_ids."""
        import retriever
        first = retriever.retrieve(
            "O365 Outlook email",
            _collection=test_chroma_collection,
            top_k=5,
        )
        seen = [r["id"] for r in first]
        assert seen, "First retrieve pass returned nothing — can't test delta"
        second = retriever.retrieve(
            "O365 Outlook email",
            seen_kb_ids=seen,
            _collection=test_chroma_collection,
            top_k=5,
        )
        overlap = {r["id"] for r in second} & set(seen)
        assert not overlap, f"Delta retriever returned already-seen IDs: {overlap}"

# ===========================================================================
# T08 – T12: Agent pipeline tests
# ===========================================================================

# Dummy KB nodes used in mocked tests
_DUMMY_NODES = [
    {"id": str(100 + i), "title": f"KB Article {i}", "body": f"content about netid password reset step {i} " * 15,
     "category": "NetID", "url": f"https://kb.wisc.edu/{100 + i}"}
    for i in range(20)
]

_GROQ_NOOP = {
    "content": "I was unable to fully resolve this issue based on the provided articles.",
    "input_tokens": 100, "output_tokens": 30,
    "model": "llama-3.3-70b-versatile", "latency_ms": 200.0,
}

_GROQ_ANSWER = {
    "content": "To reset your NetID password, visit [KB-1140] https://kb.wisc.edu/1140 and follow the steps.",
    "input_tokens": 120, "output_tokens": 40,
    "model": "llama-3.3-70b-versatile", "latency_ms": 210.0,
}


@pytest.fixture(scope="module")
def simple_agent_result():
    """Run the agent once with a real simple query. Shared by T08, T11, T12."""
    import agent
    return agent.run("how do I reset my NetID password")


class TestAgentPipeline:

    def test_T08_simple_query_resolves_in_one_turn(self, simple_agent_result):
        """T08: Simple query resolves in exactly 1 turn."""
        result = simple_agent_result
        assert result["turn"] == 1, (
            f"Expected 1 turn for simple query, got {result['turn']}."
        )
        assert result["resolved"] is True
        assert result["escalated"] is False

    def test_T09_multi_turn_no_duplicate_kb_nodes(self, monkeypatch):
        """T09: Multi-turn session never injects same KB node twice."""
        import agent

        def mock_groq_chat(model, messages, **kwargs):
            return _GROQ_NOOP

        def mock_classify(query):
            return {"complexity": "complex", "confidence": 0.9, "reasoning": "mock", "latency_ms": 10}

        def mock_retrieve(query, seen_kb_ids=None, top_k=3, _collection=None):
            seen = set(seen_kb_ids or [])
            return [n for n in _DUMMY_NODES if n["id"] not in seen][:top_k]

        monkeypatch.setattr(agent, "groq_chat", mock_groq_chat)
        monkeypatch.setattr(agent, "classify_query", mock_classify)
        monkeypatch.setattr(agent._retriever, "retrieve", mock_retrieve)

        result = agent.run("my O365 stopped working after affiliation change")
        seen = result["session"].seen_kb_ids
        assert len(seen) == len(set(seen)), (
            f"Duplicate KB IDs found in seen_kb_ids: {seen}"
        )

    def test_T10_escalates_after_4_unresolved_turns(self, monkeypatch):
        """T10: Agent sets escalated=True after 4 unresolved turns."""
        import agent

        def mock_groq_chat(model, messages, **kwargs):
            return _GROQ_NOOP

        def mock_classify(query):
            return {"complexity": "complex", "confidence": 0.9, "reasoning": "mock", "latency_ms": 10}

        def mock_retrieve(query, seen_kb_ids=None, top_k=3, _collection=None):
            seen = set(seen_kb_ids or [])
            return [n for n in _DUMMY_NODES if n["id"] not in seen][:top_k]

        monkeypatch.setattr(agent, "groq_chat", mock_groq_chat)
        monkeypatch.setattr(agent, "classify_query", mock_classify)
        monkeypatch.setattr(agent._retriever, "retrieve", mock_retrieve)

        result = agent.run("complex unresolvable issue")
        assert result["escalated"] is True, "Agent should have escalated after 4 turns."
        assert result["turn"] == 4, f"Expected 4 turns, got {result['turn']}."

    def test_T11_every_response_has_kb_citation(self, simple_agent_result):
        """T11: Agent response contains at least 1 KB citation with id and url."""
        result = simple_agent_result
        citations = result["kb_citations"]
        assert len(citations) >= 1, "Response must include at least 1 KB citation."
        for c in citations:
            assert "id" in c and c["id"], f"Citation missing id: {c}"
            assert "url" in c and c["url"].startswith("https://kb.wisc.edu"), (
                f"Citation has invalid url: {c}"
            )

    def test_T12_response_grounded_in_kb_nodes(self, simple_agent_result):
        """T12: Response keywords overlap with retrieved KB content (hallucination guard)."""
        import re
        result = simple_agent_result
        answer = result["answer"].lower()

        # Load bodies of cited KB articles
        kb_text = ""
        for c in result["kb_citations"]:
            kb_path = _DATA_DIR / f"{c['id']}.json"
            if kb_path.exists():
                kb_text += json.loads(kb_path.read_text()).get("body", "").lower()

        if not kb_text:
            pytest.skip("No KB article bodies available to check overlap.")

        stopwords = {
            "this", "that", "with", "from", "your", "have", "will", "been",
            "they", "their", "would", "could", "about", "which", "when",
            "then", "than", "also", "just", "more", "some", "what",
        }
        answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer)) - stopwords
        if not answer_words:
            pytest.skip("Answer too short to check overlap.")

        overlap = sum(1 for w in answer_words if w in kb_text)
        ratio = overlap / len(answer_words)
        assert ratio >= 0.3, (
            f"Low KB overlap ratio ({ratio:.2f}) — response may not be grounded. "
            f"Answer words: {answer_words}"
        )

# ===========================================================================
# T13 – T15: Context manager tests
# ===========================================================================

class TestContextManager:

    def test_T13_turn2_tokens_less_than_turn1(self):
        """T13: Turn 2 input tokens < Turn 1 when KB nodes are deduplicated."""
        from context_manager import new_session, build_turn_payload, count_tokens

        kb_nodes = [
            {"id": "1001", "title": "NetID Password Reset", "url": "https://kb.wisc.edu/1001",
             "body": "To reset your NetID password go to netid.wisc.edu and follow the prompts."},
            {"id": "1002", "title": "NetID Account Utilities", "url": "https://kb.wisc.edu/1002",
             "body": "Use the account utilities page to manage your NetID settings and recovery options."},
            {"id": "1003", "title": "NetID Login Problems", "url": "https://kb.wisc.edu/1003",
             "body": "Common login problems include expired passwords and locked accounts."},
        ]

        session = new_session()

        # Turn 1: all 3 nodes are new — full payload
        msgs1 = build_turn_payload(session, "how do I reset my password", kb_nodes)
        tokens1 = count_tokens(msgs1)

        # Turn 2: same nodes already seen — no KB body re-injected
        msgs2 = build_turn_payload(session, "what if I forgot my NetID", kb_nodes)
        tokens2 = count_tokens(msgs2)

        assert tokens2 < tokens1, (
            f"Turn 2 tokens ({tokens2}) should be less than Turn 1 ({tokens1}) "
            "after KB nodes are deduplicated."
        )

    def test_T14_system_prompt_byte_identical(self):
        """T14: System prompt is byte-identical across 10 different sessions."""
        from context_manager import new_session, build_turn_payload

        kb_node = {"id": "999", "title": "Test", "url": "https://kb.wisc.edu/999", "body": "body"}
        prompts = set()
        for _ in range(10):
            session = new_session()
            msgs = build_turn_payload(session, "test query", [kb_node])
            system_content = next(m["content"] for m in msgs if m["role"] == "system")
            prompts.add(system_content)

        assert len(prompts) == 1, (
            f"System prompt varied across sessions — {len(prompts)} distinct versions found. "
            "Must be byte-identical for prefix cache efficiency."
        )

    def test_T15_session_tokens_less_than_naive_rag(self):
        """T15: 3-turn session total tokens < naive RAG baseline (3x single-turn tokens)."""
        from context_manager import new_session, build_turn_payload, count_tokens

        kb_nodes = [
            {"id": str(i), "title": f"Article {i}", "url": f"https://kb.wisc.edu/{i}",
             "body": f"This is the body of article {i}. " * 20}
            for i in range(1, 7)  # 6 articles, 2 new per turn
        ]

        # Naive RAG baseline: every turn gets all 6 nodes fresh
        session_naive = new_session()
        naive_tokens = sum(
            count_tokens(build_turn_payload(session_naive, f"query {t}", kb_nodes))
            for t in range(3)
        )
        # Reset so naive session doesn't interfere
        # (naive_tokens is already 3x single-turn since we use a fresh session each time)
        single_turn_tokens = count_tokens(
            build_turn_payload(new_session(), "query 0", kb_nodes)
        )
        naive_baseline = single_turn_tokens * 3

        # Delta session: nodes accumulate in seen_kb_ids across turns
        session_delta = new_session()
        # Turn 1: nodes 0-1, turn 2: nodes 2-3, turn 3: nodes 4-5
        delta_tokens = sum(
            count_tokens(build_turn_payload(session_delta, f"query {t}", kb_nodes[t*2:(t*2)+2]))
            for t in range(3)
        )

        assert delta_tokens < naive_baseline, (
            f"Delta session tokens ({delta_tokens}) should be less than "
            f"naive RAG baseline ({naive_baseline})."
        )


# ===========================================================================
# T16 – T19: API tests
# ===========================================================================

@pytest.fixture(scope="module")
def api_client():
    """FastAPI test client with mocked agent and streaming."""
    from fastapi.testclient import TestClient
    import main as api_main
    import agent as _agent_mod

    _MOCK_RESULT = {
        "answer": "To reset your NetID password visit [KB-1140] https://kb.wisc.edu/1140.",
        "kb_citations": [{"id": "1140", "url": "https://kb.wisc.edu/1140"}],
        "turn": 1,
        "resolved": True,
        "escalated": False,
        "session": __import__("context_manager").new_session(),
    }

    def mock_agent_run(query, session=None):
        return _MOCK_RESULT

    def mock_groq_stream(model, messages, **kwargs):
        yield "To reset your NetID "
        yield "password visit [KB-1140]."

    mp = pytest.MonkeyPatch()
    mp.setattr(_agent_mod, "run", mock_agent_run)
    mp.setattr(api_main, "groq_stream", mock_groq_stream)
    mp.setattr(api_main, "classify_query",
               lambda q: {"complexity": "simple", "confidence": 0.9, "reasoning": "", "latency_ms": 10})
    mp.setattr(api_main, "kb_retrieve", lambda q, top_k=3: [])

    yield TestClient(api_main.app)
    mp.undo()


class TestAPI:

    def test_T16_health_returns_200(self, api_client):
        """T16: /health returns 200."""
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_T17_chat_returns_required_fields(self, api_client):
        """T17: /chat returns answer, kb_citations[], turn, resolved, escalated."""
        r = api_client.post("/chat", json={"query": "how do I reset my NetID password"})
        assert r.status_code == 200
        body = r.json()
        for field in ("answer", "kb_citations", "turn", "resolved", "escalated"):
            assert field in body, f"Missing field: {field}"
        assert isinstance(body["kb_citations"], list)
        assert isinstance(body["turn"], int)
        assert isinstance(body["resolved"], bool)
        assert isinstance(body["escalated"], bool)

    def test_T18_agent_assist_ttft_under_2000ms(self, api_client):
        """T18: /agent-assist TTFT under 2000ms (request to first token)."""
        t0 = time.monotonic()
        with api_client.stream("POST", "/agent-assist",
                               json={"query": "how do I reset my NetID password"}) as r:
            assert r.status_code == 200
            for chunk in r.iter_bytes():
                if chunk:
                    ttft_ms = (time.monotonic() - t0) * 1000
                    break
        assert ttft_ms < 2000, f"TTFT was {ttft_ms:.0f}ms — exceeds 2000ms budget."

    def test_T19_metrics_returns_token_counts_and_cost(self, api_client):
        """T19: /metrics returns per-session token counts and cost estimate."""
        # Ensure at least one session exists (T17 creates one)
        r = api_client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert "sessions" in body
        assert "total_cost_usd" in body
        assert isinstance(body["sessions"], list)
        if body["sessions"]:
            s = body["sessions"][0]
            for field in ("session_id", "total_input_tokens",
                          "total_output_tokens", "estimated_cost_usd"):
                assert field in s, f"Missing metrics field: {field}"


# ===========================================================================
# T20: Load test
# ===========================================================================

class TestLoad:

    def test_T20_10_concurrent_chat_requests(self):
        """T20: 10 concurrent /chat requests complete without error,
        median TTFT under 3000ms."""
        import threading
        import statistics
        from fastapi.testclient import TestClient
        import main as api_main
        import agent as _agent_mod

        _MOCK_RESULT = {
            "answer": "Reset your NetID at [KB-1140] https://kb.wisc.edu/1140.",
            "kb_citations": [{"id": "1140", "url": "https://kb.wisc.edu/1140"}],
            "turn": 1,
            "resolved": True,
            "escalated": False,
            "session": __import__("context_manager").new_session(),
        }

        mp = pytest.MonkeyPatch()
        mp.setattr(_agent_mod, "run", lambda q, session=None: _MOCK_RESULT)
        client = TestClient(api_main.app)

        results = []
        errors  = []

        def call():
            t0 = time.monotonic()
            try:
                r = client.post("/chat", json={"query": "how do I reset my NetID password"})
                elapsed = (time.monotonic() - t0) * 1000
                if r.status_code == 200:
                    results.append(elapsed)
                else:
                    errors.append(f"HTTP {r.status_code}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mp.undo()

        assert not errors, f"Errors in concurrent requests: {errors}"
        assert len(results) == 10, f"Only {len(results)}/10 requests succeeded."
        median_ms = statistics.median(results)
        assert median_ms < 3000, f"Median TTFT {median_ms:.0f}ms exceeds 3000ms budget."
