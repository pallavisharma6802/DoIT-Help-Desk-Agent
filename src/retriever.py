import json
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
import networkx as nx

from ingest import embed_query

DATA_DIR = Path(__file__).parent.parent / "data" / "kbs"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

log = logging.getLogger(__name__)

_graph: Optional[nx.Graph] = None


def _load_collection() -> Optional[chromadb.Collection]:
    if not CHROMA_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection("kb_articles")
    except Exception:
        return None


def build_graph(similarity_threshold: float = 0.60, top_k_neighbors: int = 10) -> nx.Graph:
    """Build (and cache) a semantic knowledge graph from ChromaDB embeddings.

    Nodes: all KB articles.
    Edges: pairs where cosine similarity >= similarity_threshold, limited to
           top_k_neighbors per article to keep the graph sparse and meaningful.
    Isolated nodes (no edges after threshold pass) are connected to their
    single most-similar neighbor so every article appears in the graph.

    Falls back to category-clique graph if ChromaDB is not indexed yet.
    """
    global _graph
    if _graph is not None:
        return _graph

    articles = {}
    for f in DATA_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            articles[d["id"]] = d
        except Exception:
            continue

    G = nx.Graph()
    for aid, article in articles.items():
        G.add_node(aid, title=article["title"], category=article["category"], url=article["url"])

    collection = _load_collection()
    if collection is not None and collection.count() > 0:
        # Pull all stored embeddings from ChromaDB
        all_data = collection.get(include=["embeddings", "metadatas"])
        ids = all_data["ids"]
        embeddings = all_data["embeddings"]

        # Store best neighbor for each node (fallback for isolated nodes)
        best_neighbor: dict = {}   # aid -> (nid, similarity)

        # For each article query ChromaDB for its top_k_neighbors+1 nearest (includes self)
        for i, (aid, vec) in enumerate(zip(ids, embeddings)):
            if aid not in G:
                continue
            results = collection.query(
                query_embeddings=[vec],
                n_results=min(top_k_neighbors + 1, len(ids)),
                include=["metadatas", "distances"],
            )
            neighbor_ids = results["ids"][0]
            distances = results["distances"][0]
            for nid, dist in zip(neighbor_ids, distances):
                if nid == aid:
                    continue
                # ChromaDB cosine distance: 0 = identical, 1 = orthogonal
                similarity = 1.0 - dist
                # Track best neighbor regardless of threshold
                if aid not in best_neighbor or similarity > best_neighbor[aid][1]:
                    best_neighbor[aid] = (nid, similarity)
                if similarity >= similarity_threshold and not G.has_edge(aid, nid):
                    G.add_edge(aid, nid, weight=round(similarity, 4), reason="semantic")

        # Connect any still-isolated nodes to their best neighbor
        isolated = list(nx.isolates(G))
        for aid in isolated:
            if aid in best_neighbor:
                nid, sim = best_neighbor[aid]
                if nid in G and not G.has_edge(aid, nid):
                    G.add_edge(aid, nid, weight=round(sim, 4), reason="fallback")

        log.info(
            "Graph built from embeddings: %d nodes, %d edges (threshold=%.2f, isolated_fixed=%d)",
            G.number_of_nodes(), G.number_of_edges(), similarity_threshold, len(isolated),
        )
    else:
        # Fallback: category cliques (used when ChromaDB not yet indexed)
        log.warning("ChromaDB not indexed — building fallback category-clique graph.")
        by_category: dict = {}
        for aid, article in articles.items():
            by_category.setdefault(article.get("category", ""), []).append(aid)
        for cat_ids in by_category.values():
            for i, a in enumerate(cat_ids):
                for b in cat_ids[i + 1:]:
                    G.add_edge(a, b, weight=0.5, reason="same_category")
        log.info("Fallback graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    _graph = G
    return G


def retrieve(
    query: str,
    seen_kb_ids: List[str] = None,
    top_k: int = 3,
    _collection=None,
) -> List[dict]:
    """
    Return top_k KB articles matching query, excluding seen_kb_ids.

    Each returned dict: {id, title, body, category, url}

    Pass _collection to override the default persisted ChromaDB (useful for tests).
    If _collection has an embedding_function, query_texts is used; otherwise
    embed_query() is called to produce the query vector.
    """
    if seen_kb_ids is None:
        seen_kb_ids = []
    seen_set = set(seen_kb_ids)

    collection = _collection if _collection is not None else _load_collection()
    if collection is None:
        log.warning("ChromaDB not indexed. Run: python src/ingest.py index")
        return []

    n_total = collection.count()
    if n_total == 0:
        return []

    fetch_n = min(top_k + len(seen_set) + 5, n_total)

    try:
        has_ef = getattr(collection, "_embedding_function", None) is not None
        if has_ef:
            results = collection.query(
                query_texts=[query],
                n_results=fetch_n,
                include=["documents", "metadatas"],
            )
        else:
            query_vec = embed_query(query)
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=fetch_n,
                include=["documents", "metadatas"],
            )
    except Exception as e:
        log.error("ChromaDB query failed: %s", e)
        return []

    nodes = []
    for rid, meta, doc in zip(
        results["ids"][0], results["metadatas"][0], results["documents"][0]
    ):
        if rid in seen_set:
            continue
        article_path = DATA_DIR / f"{rid}.json"
        if article_path.exists():
            body = json.loads(article_path.read_text()).get("body", "")
        else:
            parts = doc.split("\n\n", 1)
            body = parts[1] if len(parts) > 1 else doc

        nodes.append({
            "id": rid,
            "title": meta.get("title", ""),
            "body": body,
            "category": meta.get("category", ""),
            "url": meta.get("url", ""),
        })
        if len(nodes) >= top_k:
            break

    return nodes


def graph_neighbors(
    start_id: str,
    seen_kb_ids: List[str] = None,
    max_hops: int = 2,
) -> List[str]:
    """
    BFS from start_id up to max_hops, returning reachable node IDs.
    Excludes start_id and seen_kb_ids from the result.
    """
    if seen_kb_ids is None:
        seen_kb_ids = []
    visited = set(seen_kb_ids) | {start_id}

    G = build_graph()
    if start_id not in G:
        return []

    reachable = []
    frontier = {start_id}
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    reachable.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return reachable
