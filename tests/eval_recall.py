"""
Single-pass Recall@3 and Recall@5 comparison:
  Semantic-only  vs  Semantic + Graph BFS

Semantic-only  (baseline from test_retrieval.py):
    retrieve(query, top_k=k)  →  89.9% @3, 100% @5

Semantic + Graph BFS:
    1. Semantic: retrieve top_k seeds via ChromaDB vector search
    2. BFS: collect 1-hop graph neighbors of those seeds
    3. Combined pool: seeds ∪ BFS neighbors
    4. Re-rank pool: score = cosine_sim(query, article)
                            + alpha * graph_affinity(article, seeds)
       where graph_affinity rewards articles connected to high-ranked seeds
    5. Return top_k from re-ranked pool

Run:
    python tests/eval_recall.py               # semantic vs BFS, top_k=3 and 5
    python tests/eval_recall.py --top-k 3     # only @3
    python tests/eval_recall.py --alpha 0.05  # override graph weight
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import retrieve, retrieve_by_ids, graph_neighbors, build_graph, _load_collection
from test_retrieval import TEST_CASES

DATA_DIR = Path(__file__).parent.parent / "data" / "kbs"

# ChromaDB's built-in local embedding (all-MiniLM-L6-v2 via ONNX) — no API token needed
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as _ChromaEF
_local_ef = _ChromaEF()


# ---------------------------------------------------------------------------
# BFS retrieval (single pass)
# ---------------------------------------------------------------------------

def _cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def retrieve_semantic_bfs(query: str, top_k: int = 3, alpha: float = 0.05) -> list[dict]:
    """
    Single-pass: semantic seeds → 1-hop BFS expansion → graph-boosted re-rank → top_k.

    graph_affinity(article) = Σ  edge_weight(seed→article) * seed_sem_score / (1 + seed_rank)
                               for each seed that is a graph-neighbor of this article

    final_score = cosine_sim(query, article) + alpha * graph_affinity
    """
    collection = _load_collection()
    if collection is None:
        return []

    query_vec = _local_ef([query])[0]

    # Step 1: semantic seeds
    seed_results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )
    seed_ids   = seed_results["ids"][0]
    seed_dists = seed_results["distances"][0]   # cosine distance: 0 = identical
    seed_sem   = {sid: 1.0 - d for sid, d in zip(seed_ids, seed_dists)}
    seed_ranks = {sid: rank for rank, sid in enumerate(seed_ids)}

    seed_meta_data = collection.get(ids=seed_ids, include=["metadatas"])
    seed_metas = {pid: m for pid, m in zip(seed_meta_data["ids"], seed_meta_data["metadatas"])}

    # Step 2: 1-hop BFS neighbors (IDs not already in seeds)
    G = build_graph()
    neighbor_ids = []
    seen = set(seed_ids)
    for sid in seed_ids:
        if sid in G:
            for nid in G.neighbors(sid):
                if nid not in seen:
                    neighbor_ids.append(nid)
                    seen.add(nid)

    # Step 3: semantic scores for BFS candidates
    new_sem: dict[str, float] = {}
    new_metas: dict[str, dict] = {}
    if neighbor_ids:
        new_data = collection.get(ids=neighbor_ids, include=["embeddings", "metadatas"])
        for pid, emb, meta in zip(new_data["ids"], new_data["embeddings"], new_data["metadatas"]):
            new_sem[pid]   = _cosine(query_vec, emb)
            new_metas[pid] = meta

    all_sem   = {**seed_sem,   **new_sem}
    all_metas = {**seed_metas, **new_metas}

    # Step 4: graph-boosted scoring
    pool = list(seed_ids) + neighbor_ids
    scored = []
    for pid in pool:
        sem = all_sem.get(pid, 0.0)
        graph_aff = 0.0
        if pid in G:
            for nbr in G.neighbors(pid):
                if nbr in seed_sem:
                    ew      = G[pid][nbr].get("weight", 0.5)
                    rank_w  = 1.0 / (1.0 + seed_ranks[nbr])
                    graph_aff += ew * seed_sem[nbr] * rank_w
        scored.append((pid, sem + alpha * graph_aff, all_metas.get(pid, {})))

    scored.sort(key=lambda t: t[1], reverse=True)

    # Step 5: build result nodes
    nodes = []
    for pid, _score, meta in scored[:top_k]:
        article_path = DATA_DIR / f"{pid}.json"
        body = json.loads(article_path.read_text()).get("body", "") if article_path.exists() else ""
        nodes.append({
            "id":       pid,
            "title":    meta.get("title", ""),
            "body":     body,
            "category": meta.get("category", ""),
            "url":      meta.get("url", ""),
        })
    return nodes


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(retrieval_fn, top_k: int, label: str) -> dict:
    results = []
    cat_stats: dict = defaultdict(lambda: {"pass": 0, "fail": 0})

    for tc in TEST_CASES:
        try:
            hits = retrieval_fn(tc["query"], top_k)
            returned_ids = {h["id"] for h in hits}
        except Exception as e:
            returned_ids = set()
            print(f"  ⚠  [{tc['id']}] error: {e}")

        expected = set(tc["expected_ids"])
        passed   = bool(expected & returned_ids)
        results.append({**tc, "returned_ids": sorted(returned_ids), "passed": passed})
        cat_stats[tc["category"]]["pass" if passed else "fail"] += 1

    n_total  = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    return {
        "label":      label,
        "top_k":      top_k,
        "results":    results,
        "cat_stats":  dict(cat_stats),
        "n_total":    n_total,
        "n_passed":   n_passed,
        "recall_pct": n_passed / n_total * 100,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _category_table(sem3, sem5, bfs3, bfs5):
    all_cats = sorted({c for ev in (sem3, sem5, bfs3, bfs5) for c in ev["cat_stats"]})
    HDR = f"  {'Category':<22}  {'Sem@3':>6}  {'Sem@5':>6}  {'BFS@3':>6}  {'BFS@5':>6}"
    SEP = "─" * 68
    print(f"\n{SEP}\n{HDR}\n{SEP}")
    for cat in all_cats:
        def rate(ev):
            s = ev["cat_stats"].get(cat, {"pass": 0, "fail": 0})
            t = s["pass"] + s["fail"]
            return f"{s['pass']/t*100:>5.0f}%" if t else "   n/a"
        print(f"  {cat:<22}  {rate(sem3)}  {rate(sem5)}  {rate(bfs3)}  {rate(bfs5)}")
    print(SEP)


def _failure_diff(sem_ev, bfs_ev, k):
    sem_fails = {r["id"] for r in sem_ev["results"] if not r["passed"]}
    bfs_fails = {r["id"] for r in bfs_ev["results"] if not r["passed"]}
    fixed     = sem_fails - bfs_fails
    broken    = bfs_fails - sem_fails
    both_fail = sem_fails & bfs_fails

    tc_by_id = {tc["id"]: tc for tc in TEST_CASES}

    if fixed:
        print(f"\n  ✓ BFS fixed ({len(fixed)} — semantic missed, BFS hit):")
        for tid in sorted(fixed):
            tc = tc_by_id[tid]
            print(f"    [{tid}] {tc['category']} — {tc['description']}")
            print(f"           Q: {tc['query']}")
    if broken:
        print(f"\n  ✗ Regressions ({len(broken)} — semantic hit, BFS missed):")
        for tid in sorted(broken):
            tc = tc_by_id[tid]
            print(f"    [{tid}] {tc['category']} — {tc['description']}")
            print(f"           Q: {tc['query']}")
    if both_fail:
        print(f"\n  △ Still failing in both ({len(both_fail)}):")
        for tid in sorted(both_fail):
            tc = tc_by_id[tid]
            print(f"    [{tid}] {tc['category']} — {tc['description']}")
            print(f"           Q: {tc['query']}")


def print_report(sem3, sem5, bfs3, bfs5, alpha):
    W = 72
    print(f"\n{'═'*W}")
    print(f"  DoIT KB — Single-pass Recall  (Semantic vs Semantic+Graph BFS)")
    print(f"  109 queries   |   alpha={alpha}")
    print(f"{'═'*W}")

    print(f"\n  {'Metric':<34}  {'Recall':>8}  {'Pass/Total':>10}")
    print(f"  {'─'*34}  {'─'*8}  {'─'*10}")
    for ev in (sem3, sem5, bfs3, bfs5):
        k    = ev["top_k"]
        kind = "Semantic-only  " if "Semantic" in ev["label"] else "Semantic+BFS   "
        print(f"  Recall@{k} — {kind}  {ev['recall_pct']:>7.1f}%  {ev['n_passed']:>4}/{ev['n_total']}")

    print(f"\n  Δ Recall@3  (BFS − semantic):  "
          f"{bfs3['recall_pct'] - sem3['recall_pct']:+.1f} pp")
    print(f"  Δ Recall@5  (BFS − semantic):  "
          f"{bfs5['recall_pct'] - sem5['recall_pct']:+.1f} pp")

    _category_table(sem3, sem5, bfs3, bfs5)

    print(f"\n{'─'*W}")
    print(f"  RECALL@3 — failure analysis")
    _failure_diff(sem3, bfs3, 3)

    print(f"\n{'─'*W}")
    print(f"  RECALL@5 — failure analysis")
    _failure_diff(sem5, bfs5, 5)

    print(f"\n{'═'*W}\n")


# ---------------------------------------------------------------------------
# Agentic evaluation  (2-turn loop: semantic → graph BFS)
# ---------------------------------------------------------------------------

def evaluate_agentic(top_k: int = 3) -> dict:
    """
    Simulate the agent's 2-turn retrieval loop for every TEST_CASE query.

    Turn 1 — semantic:
        retrieve(query, seen_kb_ids=[], top_k=top_k)

    Turn 2 — graph BFS (only reached if turn 1 missed):
        graph_neighbors(id, seen_kb_ids=turn1_ids, max_hops=1)  for each turn-1 id
        retrieve_by_ids(neighbor_ids, seen_kb_ids=turn1_ids, top_k=top_k)

    A query passes if the expected article appears in turn-1 OR turn-2 results.
    """
    results = []
    cat_stats: dict = defaultdict(lambda: {"pass": 0, "fail": 0})

    for tc in TEST_CASES:
        expected = set(tc["expected_ids"])

        # ── Turn 1: semantic ────────────────────────────────────────────────
        try:
            t1_nodes = retrieve(tc["query"], seen_kb_ids=[], top_k=top_k)
        except Exception as e:
            print(f"  ⚠  [{tc['id']}] turn-1 error: {e}")
            t1_nodes = []

        t1_ids  = [n["id"] for n in t1_nodes]
        t1_hit  = bool(expected & set(t1_ids))

        t2_ids  = []
        t2_hit  = False

        if not t1_hit:
            # ── Turn 2: graph BFS from turn-1 results ───────────────────────
            try:
                seen_set  = set(t1_ids)
                neighbor_ids: list[str] = []
                for start_id in t1_ids:
                    for nid in graph_neighbors(start_id, seen_kb_ids=t1_ids, max_hops=1):
                        if nid not in seen_set:
                            neighbor_ids.append(nid)
                            seen_set.add(nid)

                t2_nodes = retrieve_by_ids(neighbor_ids, seen_kb_ids=t1_ids, top_k=top_k)
            except Exception as e:
                print(f"  ⚠  [{tc['id']}] turn-2 error: {e}")
                t2_nodes = []

            t2_ids = [n["id"] for n in t2_nodes]
            t2_hit = bool(expected & set(t2_ids))

        passed     = t1_hit or t2_hit
        turn_found = (1 if t1_hit else (2 if t2_hit else None))

        results.append({
            **tc,
            "t1_ids":     t1_ids,
            "t2_ids":     t2_ids,
            "t1_hit":     t1_hit,
            "t2_hit":     t2_hit,
            "passed":     passed,
            "turn_found": turn_found,
        })
        cat_stats[tc["category"]]["pass" if passed else "fail"] += 1

    n_total  = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    n_t1     = sum(1 for r in results if r["t1_hit"])
    n_t2     = sum(1 for r in results if r["t2_hit"])

    return {
        "label":      f"Agentic@{top_k}",
        "top_k":      top_k,
        "results":    results,
        "cat_stats":  dict(cat_stats),
        "n_total":    n_total,
        "n_passed":   n_passed,
        "n_turn1":    n_t1,
        "n_turn2":    n_t2,
        "recall_pct": n_passed / n_total * 100,
    }


def print_agentic_report(sem_ev: dict, ag_ev: dict) -> None:
    """Print the agentic vs semantic-only comparison report."""
    top_k  = ag_ev["top_k"]
    W      = 72
    SEP    = "─" * W

    print(f"\n{'═'*W}")
    print(f"  DoIT KB — Agentic Retrieval Evaluation  (top_k={top_k} per turn)")
    print(f"  Turn 1: semantic only  |  Turn 2: graph BFS from turn-1 results")
    print(f"  109 queries")
    print(f"{'═'*W}")

    # Summary table
    print(f"\n  {'Metric':<40}  {'Recall':>7}  {'Count':>9}")
    print(f"  {'─'*40}  {'─'*7}  {'─'*9}")
    print(f"  {'Recall@'+str(top_k)+' — Semantic-only (turn 1)':<40}  "
          f"{sem_ev['recall_pct']:>6.1f}%  "
          f"{sem_ev['n_passed']:>4}/{sem_ev['n_total']}")
    print(f"  {'Recall@'+str(top_k)+' — Agentic (turn 1 + turn 2 BFS)':<40}  "
          f"{ag_ev['recall_pct']:>6.1f}%  "
          f"{ag_ev['n_passed']:>4}/{ag_ev['n_total']}")
    delta = ag_ev["recall_pct"] - sem_ev["recall_pct"]
    print(f"\n  Δ (agentic − semantic):  {delta:+.1f} pp")

    # Turn breakdown
    print(f"\n  Turn breakdown (agentic):")
    print(f"    Turn 1 (semantic)    : {ag_ev['n_turn1']} queries resolved")
    print(f"    Turn 2 (graph BFS)   : {ag_ev['n_turn2']} additional queries resolved")
    print(f"    Still failing        : {ag_ev['n_total'] - ag_ev['n_passed']} queries")

    # Queries fixed by graph BFS on turn 2
    tc_by_id  = {tc["id"]: tc for tc in TEST_CASES}
    sem_fails = {r["id"] for r in sem_ev["results"]  if not r["passed"]}
    ag_fails  = {r["id"] for r in ag_ev["results"]   if not r["passed"]}

    fixed     = sem_fails - ag_fails          # graph BFS rescued these
    still_bad = sem_fails & ag_fails          # neither strategy found them

    print(f"\n{SEP}")
    if fixed:
        print(f"  ✓ Fixed by graph BFS on turn 2  ({len(fixed)} queries):")
        for r in ag_ev["results"]:
            if r["id"] not in fixed:
                continue
            tc = tc_by_id[r["id"]]
            print(f"\n    [{r['id']}] {tc['category']} — {tc['description']}")
            print(f"    Query      : {tc['query']}")
            print(f"    Expected   : {tc['expected_ids']}")
            print(f"    Turn-1 got : {r['t1_ids']}")
            print(f"    Turn-2 got : {r['t2_ids']}")
    else:
        print(f"  (no queries rescued by graph BFS)")

    print(f"\n{SEP}")
    if still_bad:
        print(f"  △ Still failing after both turns  ({len(still_bad)} queries):")
        for r in ag_ev["results"]:
            if r["id"] not in still_bad:
                continue
            tc = tc_by_id[r["id"]]
            print(f"\n    [{r['id']}] {tc['category']} — {tc['description']}")
            print(f"    Query      : {tc['query']}")
            print(f"    Expected   : {tc['expected_ids']}")
            print(f"    Turn-1 got : {r['t1_ids']}")
            print(f"    Turn-2 got : {r['t2_ids']}")
    else:
        print(f"  ✓ All queries resolved within 2 turns!")

    print(f"\n{'═'*W}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recall evaluation: semantic-only vs graph BFS"
    )
    parser.add_argument("--mode", choices=["single", "agentic"], default="agentic",
                        help="'single' = single-pass comparison; "
                             "'agentic' = 2-turn agent loop (default)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Articles retrieved per turn (default: 3)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="[single mode] Graph affinity weight (default: 0.05)")
    args = parser.parse_args()

    print("Building knowledge graph (cached after first call)…")
    build_graph()

    if args.mode == "agentic":
        k = args.top_k
        print(f"\nRunning Semantic-only@{k}  (turn 1 baseline) …")
        sem_ev = evaluate(
            lambda q, top_k: retrieve(q, top_k=top_k),
            top_k=k, label=f"Semantic@{k}",
        )
        print(f"Running Agentic@{k}  (turn 1 semantic + turn 2 graph BFS) …")
        ag_ev = evaluate_agentic(top_k=k)
        print_agentic_report(sem_ev, ag_ev)

    else:  # single
        top_ks = [3, 5] if args.top_k == 3 else [args.top_k]
        evs: dict = {}
        for k in top_ks:
            print(f"\nRunning Semantic-only@{k}  (baseline) …")
            evs[f"sem{k}"] = evaluate(
                lambda q, top_k: retrieve(q, top_k=top_k),
                top_k=k, label=f"Semantic@{k}",
            )
            print(f"Running Semantic+BFS@{k}  (alpha={args.alpha}) …")
            evs[f"bfs{k}"] = evaluate(
                lambda q, top_k, _a=args.alpha: retrieve_semantic_bfs(q, top_k=top_k, alpha=_a),
                top_k=k, label=f"BFS@{k}",
            )
        if set(top_ks) == {3, 5}:
            print_report(evs["sem3"], evs["sem5"], evs["bfs3"], evs["bfs5"], args.alpha)
        else:
            k = top_ks[0]
            sem, bfs = evs[f"sem{k}"], evs[f"bfs{k}"]
            print(f"\n  Recall@{k} — Semantic-only : {sem['recall_pct']:.1f}%  ({sem['n_passed']}/{sem['n_total']})")
            print(f"  Recall@{k} — Semantic+BFS  : {bfs['recall_pct']:.1f}%  ({bfs['n_passed']}/{bfs['n_total']})")
            print(f"  Δ : {bfs['recall_pct'] - sem['recall_pct']:+.1f} pp\n")
