"""Decision layer: turn calibrated pair probabilities into one id SET per
Source-1 entity, tuned for Macro F0.5 on the calib split.

Rules (all deterministic, applied per Source-1 entity):
  1. graph cleanup mask (off / exclusive / betweenness)
  2. keep pair if p >= threshold of its source (Source 2 and 3 may differ -
     the vendors have different noise levels)
  3. keep pair only if p >= (best p of the entity) - margin
  4. singleton gate: the entity outputs anything only if its best p >= gate
  5. optional entity-level singleton model: output empty if
     P(entity has a true match among candidates) < matchable_threshold
  6. optional cap on matches per source

The search maximises the EXACT entity-level Macro F0.5 (every entity of the
split counts, including entities with no candidates and true matches lost in
blocking), NOT pairwise F1. No threshold value is assumed; everything is
swept on validation data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .metrics import entity_fbeta
from .utils import get_logger


@dataclass
class DecisionParams:
    threshold_s2: float = 0.5
    threshold_s3: float = 0.5
    margin: float = 1.0
    gate: float = 0.0
    max_per_source: int | None = None
    matchable_threshold: float = 0.0
    graph_mode: str = "off"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionParams":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        if "graph_mode" in known and (known["graph_mode"] is False or known["graph_mode"] is None):
            known["graph_mode"] = "off"   # YAML parses bare `off` as False
        return cls(**known)


@dataclass
class DecisionTable:
    """Pre-computed arrays for fast repeated evaluation on one entity set."""
    ent: np.ndarray                 # entity position of every pair
    src: np.ndarray
    p: np.ndarray
    n_entities: int
    is_true: np.ndarray | None = None
    n_true: np.ndarray | None = None
    graph_masks: dict = field(default_factory=dict)
    matchable: np.ndarray | None = None
    beta: float = 0.5

    def __post_init__(self):
        self.order = np.lexsort((-self.p, self.src, self.ent))
        o_ent, o_src = self.ent[self.order], self.src[self.order]
        new_grp = np.r_[True, (o_ent[1:] != o_ent[:-1]) | (o_src[1:] != o_src[:-1])]
        self.grp_start_pos = np.maximum.accumulate(np.where(new_grp, np.arange(len(self.order)), 0))
        self._cache: dict = {}

    @classmethod
    def build(cls, pairs: pd.DataFrame, prob_col: str, entity_idx: np.ndarray, n_true: np.ndarray | None = None,
              graph_masks: dict | None = None, matchable: np.ndarray | None = None, beta: float = 0.5):
        pos = pd.Series(np.arange(len(entity_idx)), index=entity_idx)
        ent = pos.reindex(pairs["s1_idx"].to_numpy()).to_numpy()
        if np.isnan(ent.astype(float)).any():
            raise ValueError("pairs contain entities outside entity_idx")
        return cls(ent=ent.astype(np.int64), src=pairs["source"].to_numpy(np.int64),
                   p=pairs[prob_col].to_numpy(np.float64), n_entities=len(entity_idx),
                   is_true=(pairs["is_true"].to_numpy(bool) if "is_true" in pairs
                            else (pairs["label"].to_numpy() != 0) if "label" in pairs else None),
                   n_true=n_true, graph_masks=graph_masks or {}, matchable=matchable, beta=beta)

    def _peff_top(self, mode: str):
        if mode not in self._cache:
            keep = self.graph_masks.get(mode)
            p = self.p if keep is None else np.where(keep, self.p, -1.0)
            top = np.full(self.n_entities, -1.0)
            np.maximum.at(top, self.ent, p)
            self._cache[mode] = (p, top)
        return self._cache[mode]

    def select(self, prm: DecisionParams) -> np.ndarray:
        p, top = self._peff_top(prm.graph_mode)
        thr = np.where(self.src == 2, prm.threshold_s2, prm.threshold_s3)
        top_e = top[self.ent]
        ok = (p >= thr) & (p >= top_e - prm.margin) & (top_e >= prm.gate)
        if self.matchable is not None and prm.matchable_threshold > 0:
            ok &= self.matchable[self.ent] >= prm.matchable_threshold
        if prm.max_per_source:
            okc = ok[self.order].astype(np.int64)
            cs = np.cumsum(okc)
            before = cs[self.grp_start_pos] - okc[self.grp_start_pos]
            rank = cs - before
            capped = np.zeros(len(ok), dtype=bool)
            capped[self.order] = (okc == 1) & (rank <= prm.max_per_source)
            ok = capped
        return ok

    def score(self, prm: DecisionParams) -> float:
        sel = self.select(prm)
        tp = np.bincount(self.ent[sel & self.is_true], minlength=self.n_entities)
        npred = np.bincount(self.ent[sel], minlength=self.n_entities)
        return float(entity_fbeta(tp, npred, self.n_true, self.beta).mean())


def plateau_center(values: list, scores: list, tol: float = 1e-12):
    """Among grid values reaching the best score, return the middle of the
    longest contiguous run. Calibrated scores are often nearly binary, so
    many thresholds tie; the first one (argmax) sits on the edge of the
    plateau and generalises worst."""
    scores = np.asarray(scores, dtype=float)
    best = scores.max()
    good = scores >= best - tol
    runs, start = [], None
    for i, g in enumerate(good):
        if g and start is None:
            start = i
        if not g and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(good) - 1))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return values[(a + b) // 2]


def effective_min_improvement(dcfg: dict, n_entities: int) -> float:
    """Gain an extra rule/blend must show on validation before it is adopted:
    max(min_improvement, min_entities_gain / n_entities). One entity is worth
    1/n of Macro F0.5, so on small validation sets a gain of a single entity
    is indistinguishable from noise."""
    return max(float(dcfg.get("min_improvement", 0.0005)),
               float(dcfg.get("min_entities_gain", 2)) / max(1, n_entities))


def search_decision(table: DecisionTable, dcfg: dict, graph_modes: list[str] | None = None,
                    quick: bool = False) -> tuple[DecisionParams, float, list]:
    """Coordinate-ascent search of the decision parameters (Macro F0.5)."""
    g = dcfg.get("threshold_grid", {"start": 0.05, "stop": 0.99, "step": 0.01})
    T = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4)
    min_imp = effective_min_improvement(dcfg, table.n_entities)
    trace: list = []
    best = DecisionParams(graph_mode="off")
    # 1) global threshold sweep (shared by both sources)
    scores = []
    for t in T:
        prm = DecisionParams(threshold_s2=float(t), threshold_s3=float(t))
        scores.append(table.score(prm))
        trace.append(("init", "threshold", float(t), scores[-1]))
    t_best = plateau_center([float(t) for t in T], scores)
    best.threshold_s2 = best.threshold_s3 = t_best
    best_score = float(max(scores))
    if quick:
        return best, best_score, trace

    def sweep(name, values, simplest, needs_gain=True, plateau=False):
        """Evaluate `values` for one parameter (or a tuple of parameters).
        A non-simplest value is adopted only if it beats the current score by
        the required gain; thresholds use the centre of the best plateau."""
        nonlocal best, best_score
        names = name if isinstance(name, tuple) else (name,)
        results = []
        for val in values:
            vals = val if isinstance(name, tuple) else (val,)
            prm = DecisionParams(**{**best.to_dict(), **dict(zip(names, vals))})
            s = table.score(prm)
            results.append((s, val))
            trace.append((r, str(name), val, s))
        s_best = max(sc for sc, _ in results)
        if plateau:
            v_best = plateau_center([val for _, val in results], [sc for sc, _ in results])
        else:
            v_best = max(results, key=lambda x: x[0])[1]
        current = tuple(getattr(best, n) for n in names)
        current = current if isinstance(name, tuple) else current[0]
        gain_needed = min_imp if (needs_gain and v_best != simplest) else 0.0
        if v_best == current:
            return False
        if s_best > best_score + gain_needed or (not needs_gain and s_best >= best_score):
            vals = v_best if isinstance(name, tuple) else (v_best,)
            best = DecisionParams(**{**best.to_dict(), **dict(zip(names, vals))})
            best_score = s_best
            return True
        return False

    for r in range(int(dcfg.get("rounds", 3))):
        changed = False
        # shared threshold: free to move (centre of the best plateau)
        shared = [(float(t), float(t)) for t in T]
        if best.threshold_s2 == best.threshold_s3:
            changed |= sweep(("threshold_s2", "threshold_s3"), shared, None, needs_gain=False, plateau=True)
        if dcfg.get("per_source_thresholds", True):
            # a separate Source-2 / Source-3 threshold is extra complexity:
            # it must earn the required gain over the shared one
            changed |= sweep("threshold_s2", [float(t) for t in T], None, needs_gain=True, plateau=True)
            changed |= sweep("threshold_s3", [float(t) for t in T], None, needs_gain=True, plateau=True)
        changed |= sweep("margin", [float(m) for m in dcfg.get("margins", [1.0])], 1.0)
        tmax = max(best.threshold_s2, best.threshold_s3)
        gates = [0.0] + sorted({round(min(0.999, tmax + float(o)), 4) for o in dcfg.get("gate_offsets", [0.0])})
        changed |= sweep("gate", gates, 0.0)
        changed |= sweep("max_per_source", list(dcfg.get("max_per_source", [None])), None)
        if table.matchable is not None:
            changed |= sweep("matchable_threshold", [float(x) for x in dcfg.get("matchable_thresholds", [0.0])], 0.0)
        if graph_modes:
            changed |= sweep("graph_mode", list(graph_modes), "off")
        if not changed:
            break
    get_logger().info("Decision search: best Macro F%.1f = %.5f with %s", table.beta, best_score, best.to_dict())
    return best, best_score, trace


# ------------------------------------------------------ singleton model
# pair-level feature columns that entity_features() reads (besides the scores)
ENTITY_INPUT_COLUMNS = ["name_jw", "addr_char_cos", "contradictions", "ctx_mutual_best"]
ENTITY_FEATURES = ["top1", "top2", "margin12", "n_ge_03", "n_ge_05", "n_ge_07", "n_ge_09", "n_cands",
                   "top1_s2", "top1_s3", "top1_lgb", "top1_ce", "top1_model_gap", "top1_name_jw",
                   "top1_addr_cos", "top1_contradictions", "top1_mutual_best", "mean_p", "has_ce"]


def entity_features(scored: pd.DataFrame, entity_idx: np.ndarray, prob_col: str = "p_final") -> pd.DataFrame:
    """Score-distribution features per Source-1 entity (label-free)."""
    df = scored[["s1_idx", "source", prob_col]].copy()
    df["p"] = df[prob_col]
    for c, src in (("p_lgb", "p_lgb"), ("p_ce", "p_ce"), ("name_jw", "name_jw"),
                   ("addr_char_cos", "addr_char_cos"), ("contradictions", "contradictions"),
                   ("ctx_mutual_best", "ctx_mutual_best")):
        df[c] = scored[src].to_numpy() if src in scored else np.nan
    df = df.sort_values(["s1_idx", "p"], ascending=[True, False])
    g = df.groupby("s1_idx", sort=False)
    first = g.head(1).set_index("s1_idx")
    second = g.nth(1).set_index("s1_idx")["p"] if len(df) else pd.Series(dtype=float)
    out = pd.DataFrame(index=pd.Index(entity_idx, name="s1_idx"))
    out["top1"] = first["p"]
    out["top2"] = second
    out["margin12"] = out["top1"] - out["top2"].fillna(0)
    for thr, name in ((0.3, "n_ge_03"), (0.5, "n_ge_05"), (0.7, "n_ge_07"), (0.9, "n_ge_09")):
        out[name] = df[df["p"] >= thr].groupby("s1_idx").size()
    out["n_cands"] = g.size()
    out["top1_s2"] = df[df["source"] == 2].groupby("s1_idx")["p"].max()
    out["top1_s3"] = df[df["source"] == 3].groupby("s1_idx")["p"].max()
    out["top1_lgb"] = first["p_lgb"]
    out["top1_ce"] = first["p_ce"]
    out["top1_model_gap"] = (first["p_lgb"] - first["p_ce"]).abs()
    out["top1_name_jw"] = first["name_jw"]
    out["top1_addr_cos"] = first["addr_char_cos"]
    out["top1_contradictions"] = first["contradictions"]
    out["top1_mutual_best"] = first["ctx_mutual_best"]
    out["mean_p"] = g["p"].mean()
    out["has_ce"] = first["p_ce"].notna().astype(float)
    for c in ("n_ge_03", "n_ge_05", "n_ge_07", "n_ge_09", "n_cands"):
        out[c] = out[c].fillna(0)
    return out[ENTITY_FEATURES].astype(float)


_SINGLETON_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10,
                     "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0,
                     "verbose": -1,
                     # tiny data: a single thread avoids OpenMP spin-wait slowdowns
                     # when other processes share the CPU (seen: 289 s vs 0.2 s)
                     "num_threads": 1}


def train_singleton_model(ent_feats: pd.DataFrame, y: np.ndarray, folds: int = 5, seed: int = 42,
                          rounds: int = 200):
    """Out-of-fold P(matchable) on calib entities + final model on all of them."""
    y = np.asarray(y, dtype=int)
    params = {**_SINGLETON_PARAMS, "seed": seed}
    oof = np.full(len(y), np.nan)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < folds:
        return None, None, {"skipped": "not enough entities of both classes"}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(ent_feats, y):
        b = lgb.train(params, lgb.Dataset(ent_feats.iloc[tr], label=y[tr]), num_boost_round=rounds)
        oof[te] = b.predict(ent_feats.iloc[te])
    final = lgb.train(params, lgb.Dataset(ent_feats, label=y), num_boost_round=rounds)
    from sklearn.metrics import roc_auc_score
    report = {"entities": int(len(y)), "matchable_rate": float(y.mean()), "oof_auc": float(roc_auc_score(y, oof))}
    return final, oof, report
