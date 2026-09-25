"""Graph cleanup of transitive false positives (PDF: GraLMatch / TransClean).

Analysis for THIS task (Source-1-centric):
  * We never have to cluster the whole dataset: the output is, for each
    Source-1 entity, its set of Source-2/3 ids. Full transitive closure is
    neither required nor desirable.
  * Source 1 is de-duplicated, so two different Source-1 nodes are, by
    definition, different businesses. In the bipartite graph
    (S1 nodes <-> vendor nodes, edge weight = match probability) the only
    transitive error that can hurt the output is a vendor record linked to
    two S1 entities: transitivity would claim S1_a == S1_b, contradicting
    de-duplication, so at least one of those edges is a false merge.
  * Therefore a connected component may contain at most ONE Source-1 node.

Modes (chosen on the calib split by Macro F0.5; "off" wins unless cleanup
actually helps):
  off          no cleanup
  exclusive    each vendor record keeps only its highest-probability S1 edge
               (the minimal fix of the above contradiction)
  betweenness  PDF procedure on the thresholded graph: while a component has
               more than one S1 node (or exceeds `max_component_size`), first
               cut the weakest *separating bridge* ("minimum edge cut
               analysis": one edge whose removal splits S1 nodes apart), else
               cut the edge with the highest edge-betweenness centrality
               weighted by (1 - p) (weak bottleneck edges); repeat.

The ground-truth statistic `vendor_records_matched_to_multiple_s1` (labels
report) tells whether the at-most-one-S1-per-vendor assumption holds.
"""
from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
import pandas as pd


def exclusive_mask(s1_idx: np.ndarray, v_idx: np.ndarray, p: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"v": v_idx, "p": p})
    best = df.groupby("v")["p"].transform("max").to_numpy()
    return p >= best


def _separating_bridges(g: nx.Graph):
    """Bridges whose removal leaves S1 nodes on both sides."""
    out = []
    for u, w in list(nx.bridges(g)):
        g.remove_edge(u, w)
        side = nx.node_connected_component(g, u)
        g.add_edge(u, w, **g.graph["_attrs"][(min(u, w), max(u, w))])
        s1_side = sum(1 for n in side if n[0] == "s")
        s1_total = g.graph["_n_s1"]
        if 0 < s1_side < s1_total:
            out.append((u, w))
    return out


def betweenness_mask(s1_idx: np.ndarray, v_idx: np.ndarray, p: np.ndarray, min_edge_prob: float = 0.2,
                     max_component_size: int = 50) -> tuple[np.ndarray, dict]:
    keep = np.ones(len(p), dtype=bool)
    in_graph = p >= min_edge_prob
    edge_id = {}
    G = nx.Graph()
    for i in np.where(in_graph)[0]:
        a, b = ("s", int(s1_idx[i])), ("v", int(v_idx[i]))
        G.add_edge(a, b, p=float(p[i]), dist=1.0 - float(p[i]) + 1e-6)
        edge_id[(a, b)] = i
    stats = {"edges": G.number_of_edges(), "components": 0, "conflict_components": 0,
             "oversized_components": 0, "edges_removed": 0, "bridges_removed": 0}
    queue = deque()
    for comp in nx.connected_components(G):
        stats["components"] += 1
        n_s1 = sum(1 for n in comp if n[0] == "s")
        if n_s1 > 1 or len(comp) > max_component_size:
            stats["conflict_components"] += int(n_s1 > 1)
            stats["oversized_components"] += int(len(comp) > max_component_size)
            queue.append(comp)
    guard = 0
    while queue and guard < 100_000:
        guard += 1
        comp = queue.popleft()
        sub = G.subgraph(comp).copy()
        n_s1 = sum(1 for n in sub if n[0] == "s")
        if n_s1 <= 1 and sub.number_of_nodes() <= max_component_size:
            continue
        if sub.number_of_edges() == 0:
            continue
        sub.graph["_attrs"] = {(min(u, w), max(u, w)): d for u, w, d in sub.edges(data=True)}
        sub.graph["_n_s1"] = n_s1
        cand = _separating_bridges(sub) if n_s1 > 1 else []
        if cand:
            u, w = min(cand, key=lambda e: sub.edges[e]["p"])
            stats["bridges_removed"] += 1
        else:
            eb = nx.edge_betweenness_centrality(sub, weight="dist")
            u, w = max(eb, key=lambda e: eb[e] * (1.0 - sub.edges[e]["p"]) + 1e-9 * eb[e])
        G.remove_edge(u, w)
        stats["edges_removed"] += 1
        a, b = (u, w) if u[0] == "s" else (w, u)
        keep[edge_id[(a, b)]] = False
        for new_comp in nx.connected_components(G.subgraph(comp)):
            queue.append(set(new_comp))
    return keep, stats


def normalize_mode(mode) -> str:
    """YAML turns a bare `off` into False - accept that spelling too."""
    if mode is False or mode is None:
        return "off"
    return str(mode).lower()


def graph_keep_mask(pairs: pd.DataFrame, prob_col: str, mode: str, min_edge_prob: float = 0.2,
                    max_component_size: int = 50) -> tuple[np.ndarray, dict]:
    mode = normalize_mode(mode)
    s1 = pairs["s1_idx"].to_numpy()
    v = pairs["v_idx"].to_numpy()
    p = pairs[prob_col].to_numpy(dtype=float)
    if mode == "off":
        return np.ones(len(p), dtype=bool), {"mode": "off"}
    if mode == "exclusive":
        keep = exclusive_mask(s1, v, p)
        return keep, {"mode": "exclusive", "edges_removed": int((~keep).sum())}
    if mode == "betweenness":
        keep, stats = betweenness_mask(s1, v, p, min_edge_prob, max_component_size)
        stats["mode"] = "betweenness"
        return keep, stats
    raise ValueError(f"unknown graph mode {mode}")
