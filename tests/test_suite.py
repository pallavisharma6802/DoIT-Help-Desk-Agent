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

# Make src/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Environment: load .env if python-dotenv is available, otherwise rely on
# the shell environment having the keys set.
# ---------------------------------------------------------------------------

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
