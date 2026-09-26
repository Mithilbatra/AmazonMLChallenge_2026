"""Pairwise features for the structural matcher (PDF: "Structural Matching via
Gradient Boosted Trees").

* String similarities optimised for short administrative text: Jaro-Winkler
  (prefix scale p = 0.1, prefix capped at 4 - as in the PDF), Levenshtein,
  token-sort/set ratios, char n-gram TF-IDF cosine.
* Soft-TFIDF (Cohen et al.): tokens align when their secondary similarity
  (Jaro-Winkler by default, Levenshtein optional) >= 0.90, weighted by IDF so
  frequent administrative words count little and rare tokens a lot.
* Exact-match boolean flags on parsed address components (house number,
  road, city, state, postcode, unit) + conflict and missing indicators.
* Legal-suffix agreement / conflict (the sequestered suffix column).
* Combined and contradiction features.
* Label-free context features computed from the candidate set only
  (rank of the pair among the entity's candidates, margin to the best
  alternative, reciprocal-best flag). They use no labels and no model output,
  so they are identical at training and inference time.

Missing information is encoded as NaN (LightGBM routes NaN natively).
"""
from __future__ import annotations

import gc
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel, JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .blocking import METHODS, TextRepresentations, rowwise_cosine
from .utils import ensure_dir, get_logger, load_json, mem_str, resolve_n_jobs, save_json

# Blocking flags usable as features. The semantic flag is excluded: the
# bi-encoder is trained on the fit split, so on fit pairs its retrieval is
# in-sample and the flag would leak label information into LightGBM.
BLOCKING_FEATURE_METHODS = [m for m in METHODS if m != "semantic"]


class FeatureContext:
    """Corpus statistics of ONE dataset (train or test): unpruned TF-IDF
    matrices and token IDF tables. Unsupervised - no labels involved."""

    def __init__(self, s1: pd.DataFrame, v: pd.DataFrame, cfg: dict):
        self.reps = TextRepresentations(s1, v, ngram_range=(2, 4), char_max_df=1.0, word_max_df=1.0)
        self.name_idf = self._idf(pd.concat([s1["name_tokens"], v["name_tokens"]]))
        self.addr_idf = self._idf(pd.concat([s1["addr_clean"], v["addr_clean"]]))
        fcfg = cfg.get("features", {})
        self._cols: dict = {}
        self.soft_thr = float(fcfg.get("soft_tfidf_threshold", 0.9))
        self.soft_secondary = fcfg.get("soft_tfidf_secondary", "jaro_winkler")
        self.jw_prefix = float(fcfg.get("jw_prefix_weight", 0.1))

    def column(self, df: pd.DataFrame, side: str, col: str) -> np.ndarray:
        """Object array of a record column, converted once and reused by every chunk."""
        key = (side, col)
        if key not in self._cols:
            self._cols[key] = df[col].to_numpy(object)
        return self._cols[key]

    @staticmethod
    def _idf(texts: pd.Series) -> dict:
        df = Counter()
        n = 0
        for t in texts:
            n += 1
            df.update(set(t.split()))
        return {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}


# ----------------------------------------------------------------- helpers
def _sim(a, b, scorer, n_jobs, **kw) -> np.ndarray:
    out = cpdist(list(a), list(b), scorer=scorer, workers=n_jobs, **kw).astype(np.float32)
    return out


def _nan_if_missing(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x[(a == "") | (b == "")] = np.nan
    return x


def _eq(a: np.ndarray, b: np.ndarray, nan_missing: bool = True) -> np.ndarray:
    out = (a == b).astype(np.float32)
    if nan_missing:
        out[(a == "") | (b == "")] = np.nan
    return out


# Globals for worker processes (set by the pool initializer).
_G: dict = {}


def _init_worker(name_idf, addr_idf, soft_thr, secondary, jw_prefix):
    _G.update(name_idf=name_idf, addr_idf=addr_idf, soft_thr=soft_thr,
              secondary=secondary, jw_prefix=jw_prefix,
              name_default=max(name_idf.values()) if name_idf else 1.0,
              addr_default=max(addr_idf.values()) if addr_idf else 1.0)


def soft_tfidf(ta: list[str], tb: list[str], idf: dict, thr: float = 0.9, secondary: str = "jaro_winkler",
               jw_prefix: float = 0.1, default: float | None = None) -> float:
    """Soft-TFIDF(S, T) = sum_{w in CLOSE(thr,S,T)} V(w,S) * V(w*,T) * sim(w, w*).

    `default` is the IDF of unseen tokens (pass it pre-computed in hot loops)."""
    if not ta or not tb:
        return float("nan")
    if default is None:
        default = max(idf.values()) if idf else 1.0

    def weights(tokens):
        c = Counter(tokens)
        w = {t: n * idf.get(t, default) for t, n in c.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {t: x / norm for t, x in w.items()}

    wa, wb = weights(ta), weights(tb)
    total = 0.0
    for t, w in wa.items():
        if t in wb:
            total += w * wb[t]
            continue
        best, best_u = 0.0, None
        for u in wb:
            if secondary == "levenshtein":
                s = Levenshtein.normalized_similarity(t, u)
            else:
                s = JaroWinkler.normalized_similarity(t, u, prefix_weight=jw_prefix)
            if s > best:
                best, best_u = s, u
        if best_u is not None and best >= thr:
            total += w * wb[best_u] * best
    return min(total, 1.0)


PY_FEATURES = ["name_soft_tfidf", "name_tok_jaccard", "name_tok_overlap_min", "name_idf_overlap",
               "name_max_shared_idf", "name_unshared_idf", "name_acronym", "addr_soft_tfidf",
               "addr_tok_jaccard", "num_jaccard", "num_conflict", "unit_eq", "unit_conflict",
               "suburb_overlap"]


def _python_block(args) -> np.ndarray:
    (n1, n2, a1, a2, i1, i2, num1, num2, u1, u2, sub1, sub2) = args
    name_idf, addr_idf = _G["name_idf"], _G["addr_idf"]
    thr, sec, jwp = _G["soft_thr"], _G["secondary"], _G["jw_prefix"]
    out = np.full((len(n1), len(PY_FEATURES)), np.nan, dtype=np.float32)
    for r in range(len(n1)):
        ta, tb = n1[r].split(), n2[r].split()
        sa, sb = set(ta), set(tb)
        out[r, 0] = soft_tfidf(ta, tb, name_idf, thr, sec, jwp, _G["name_default"])
        if sa and sb:
            inter, union = sa & sb, sa | sb
            out[r, 1] = len(inter) / len(union)
            out[r, 2] = len(inter) / min(len(sa), len(sb))
            wi = sum(name_idf.get(t, 1.0) for t in inter)
            wu = sum(name_idf.get(t, 1.0) for t in union)
            out[r, 3] = wi / wu if wu else 0.0
            out[r, 4] = max((name_idf.get(t, 1.0) for t in inter), default=0.0)
            out[r, 5] = sum(name_idf.get(t, 1.0) for t in (union - inter))
            ini_a, ini_b = i1[r], i2[r]
            nos_a, nos_b = "".join(ta), "".join(tb)
            out[r, 6] = float((len(ini_a) >= 2 and ini_a == nos_b) or (len(ini_b) >= 2 and ini_b == nos_a))
        xa, xb = a1[r].split(), a2[r].split()
        out[r, 7] = soft_tfidf(xa, xb, addr_idf, thr, sec, jwp, _G["addr_default"])
        if xa and xb:
            out[r, 8] = len(set(xa) & set(xb)) / len(set(xa) | set(xb))
        na, nb = set(num1[r].split()), set(num2[r].split())
        if na and nb:
            out[r, 9] = len(na & nb) / len(na | nb)
            out[r, 10] = float(not (na & nb))
        ua, ub = set(u1[r].split()), set(u2[r].split())
        if ua and ub:
            out[r, 11] = float(bool(ua & ub))
            out[r, 12] = float(not (ua & ub))
        sa2, sb2 = set(sub1[r].split()), set(sub2[r].split())
        if sa2 and sb2:
            out[r, 13] = len(sa2 & sb2) / min(len(sa2), len(sb2))
    return out


def _init_args(ctx: FeatureContext) -> tuple:
    return (ctx.name_idf, ctx.addr_idf, ctx.soft_thr, ctx.soft_secondary, ctx.jw_prefix)


@contextmanager
def feature_workers(ctx: FeatureContext, n_jobs: int):
    """One process pool for a whole feature run (created once, reused by every
    chunk). Yields None when running single-process."""
    if n_jobs <= 1:
        _init_worker(*_init_args(ctx))
        yield None
        return
    # Freeze the parent's objects before forking: otherwise the garbage
    # collector in every worker touches all inherited objects, triggering
    # copy-on-write and silently duplicating the parent's heap per worker.
    gc.collect()
    gc.freeze()
    try:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=_init_args(ctx)) as ex:
            yield ex
    finally:
        gc.unfreeze()


def _python_features(cols: dict, ctx: FeatureContext, executor=None, chunk: int = 25_000) -> np.ndarray:
    n = len(cols["n1"])
    keys = ["n1", "n2", "a1", "a2", "i1", "i2", "num1", "num2", "u1", "u2", "sub1", "sub2"]
    blocks = [tuple(list(cols[k][s:s + chunk]) for k in keys) for s in range(0, n, chunk)]
    if executor is None:
        if not _G or _G.get("name_idf") is not ctx.name_idf:
            _init_worker(*_init_args(ctx))
        results = [_python_block(b) for b in blocks]
    else:
        results = list(executor.map(_python_block, blocks))
    return np.vstack(results) if results else np.zeros((0, len(PY_FEATURES)), np.float32)


# -------------------------------------------------------------- main entry
def _base_features(pairs: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, ctx: FeatureContext,
                   n_jobs: int, executor=None) -> pd.DataFrame:
    """All non-context features for ONE chunk of pairs (float32 columns)."""
    ia = pairs["s1_idx"].to_numpy(np.int64)
    ib = pairs["v_idx"].to_numpy(np.int64)
    g1 = lambda c: ctx.column(s1, "s1", c)[ia]
    g2 = lambda c: ctx.column(v, "v", c)[ib]
    F: dict[str, np.ndarray] = {}
    jw = lambda a, b: _sim(a, b, JaroWinkler.normalized_similarity, n_jobs, scorer_kwargs={"prefix_weight": ctx.jw_prefix})

    # ---------------- name
    raw1 = np.array([x.lower().strip() for x in g1("name")], dtype=object)
    raw2 = np.array([x.lower().strip() for x in g2("name")], dtype=object)
    core1, core2 = g1("name_core"), g2("name_core")
    tok1, tok2 = g1("name_tokens"), g2("name_tokens")
    nos1, nos2 = g1("name_nospace"), g2("name_nospace")
    F["name_exact_raw"] = _eq(raw1, raw2)
    F["name_exact_clean"] = _eq(g1("name_clean"), g2("name_clean"))
    F["name_exact_core"] = _eq(core1, core2)
    F["name_exact_nospace"] = _eq(nos1, nos2)
    F["name_jw"] = _nan_if_missing(jw(core1, core2), core1, core2)
    F["name_jw_nospace"] = _nan_if_missing(jw(nos1, nos2), nos1, nos2)
    F["name_lev"] = _nan_if_missing(_sim(core1, core2, Levenshtein.normalized_similarity, n_jobs), core1, core2)
    F["name_indel"] = _nan_if_missing(_sim(core1, core2, Indel.normalized_similarity, n_jobs), core1, core2)
    F["name_token_sort"] = _nan_if_missing(_sim(tok1, tok2, fuzz.token_sort_ratio, n_jobs) / 100, tok1, tok2)
    F["name_token_set"] = _nan_if_missing(_sim(tok1, tok2, fuzz.token_set_ratio, n_jobs) / 100, tok1, tok2)
    F["name_partial"] = _nan_if_missing(_sim(nos1, nos2, fuzz.partial_ratio, n_jobs) / 100, nos1, nos2)
    F["name_char_cos"] = rowwise_cosine(ctx.reps.s1("name_char"), ctx.reps.v("name_char"), ia, ib)
    F["name_word_cos"] = rowwise_cosine(ctx.reps.s1("name_word"), ctx.reps.v("name_word"), ia, ib)
    F["name_phonetic_eq"] = _eq(g1("name_phonetic"), g2("name_phonetic"))
    F["name_first_phonetic_eq"] = _eq(g1("name_first_phonetic"), g2("name_first_phonetic"))
    p1 = np.array([x[:4] for x in nos1], dtype=object)
    p2 = np.array([x[:4] for x in nos2], dtype=object)
    F["name_prefix4_eq"] = _eq(p1, p2)
    F["name_common_prefix"] = np.array([len(_common_prefix(a, b)) for a, b in zip(nos1, nos2)], np.float32)
    ft1 = np.array([x.split()[0] if x else "" for x in tok1], dtype=object)
    ft2 = np.array([x.split()[0] if x else "" for x in tok2], dtype=object)
    F["name_first_token_eq"] = _eq(ft1, ft2)
    l1 = np.array([len(x) for x in nos1], np.float32)
    l2 = np.array([len(x) for x in nos2], np.float32)
    F["name_len_1"], F["name_len_2"] = l1, l2
    F["name_len_ratio"] = np.where(np.maximum(l1, l2) > 0, np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1), np.nan).astype(np.float32)
    F["name_ntok_1"] = np.array([len(x.split()) for x in tok1], np.float32)
    F["name_ntok_2"] = np.array([len(x.split()) for x in tok2], np.float32)
    F["name_contains"] = np.array([float(bool(a) and bool(b) and (a in b or b in a)) for a, b in zip(nos1, nos2)], np.float32)
    d1, d2 = g1("name_digits"), g2("name_digits")
    F["name_digits_eq"] = _eq(d1, d2)
    # legal suffix (sequestered column) agreement
    sf1, sf2 = g1("name_suffix"), g2("name_suffix")
    fam1, fam2 = g1("name_suffix_family"), g2("name_suffix_family")
    F["suffix_present_1"] = (sf1 != "").astype(np.float32)
    F["suffix_present_2"] = (sf2 != "").astype(np.float32)
    F["suffix_eq"] = _eq(sf1, sf2)
    F["suffix_family_eq"] = _eq(fam1, fam2)
    F["suffix_conflict"] = ((sf1 != "") & (sf2 != "") & (fam1 != fam2)).astype(np.float32)

    # ---------------- address
    ad1, ad2 = g1("addr_clean"), g2("addr_clean")
    F["addr_missing_1"] = (ad1 == "").astype(np.float32)
    F["addr_missing_2"] = (ad2 == "").astype(np.float32)
    F["addr_exact"] = _eq(ad1, ad2)
    F["addr_jw"] = _nan_if_missing(jw(ad1, ad2), ad1, ad2)
    F["addr_token_set"] = _nan_if_missing(_sim(ad1, ad2, fuzz.token_set_ratio, n_jobs) / 100, ad1, ad2)
    F["addr_char_cos"] = rowwise_cosine(ctx.reps.s1("addr_char"), ctx.reps.v("addr_char"), ia, ib)
    F["full_char_cos"] = rowwise_cosine(ctx.reps.s1("full_char"), ctx.reps.v("full_char"), ia, ib)
    comps = {}
    for comp in ("house_number", "road", "city", "state", "postcode", "unit"):
        a, b = g1(f"addr_{comp}"), g2(f"addr_{comp}")
        comps[comp] = (a, b)
        both = (a != "") & (b != "")
        F[f"{comp}_both"] = both.astype(np.float32)
    hn1, hn2 = comps["house_number"]
    F["hn_eq"] = _eq(hn1, hn2)
    F["hn_conflict"] = ((hn1 != "") & (hn2 != "") & (hn1 != hn2)).astype(np.float32)
    r1, r2 = comps["road"]
    F["road_jw"] = _nan_if_missing(jw(r1, r2), r1, r2)
    F["road_token_set"] = _nan_if_missing(_sim(r1, r2, fuzz.token_set_ratio, n_jobs) / 100, r1, r2)
    c1, c2 = comps["city"]
    F["city_eq"] = _eq(c1, c2)
    F["city_jw"] = _nan_if_missing(jw(c1, c2), c1, c2)
    F["city_conflict"] = ((c1 != "") & (c2 != "") & (np.nan_to_num(F["city_jw"]) < 0.85)).astype(np.float32)
    s_1, s_2 = comps["state"]
    F["state_eq"] = _eq(s_1, s_2)
    F["state_conflict"] = ((s_1 != "") & (s_2 != "") & (s_1 != s_2)).astype(np.float32)
    pc1, pc2 = comps["postcode"]
    F["pc_eq"] = _eq(pc1, pc2)
    F["pc_prefix3_eq"] = _eq(np.array([x[:3] for x in pc1], dtype=object), np.array([x[:3] for x in pc2], dtype=object))
    F["pc_conflict"] = ((pc1 != "") & (pc2 != "") & (pc1 != pc2)).astype(np.float32)
    F["landmark_1"] = (g1("addr_landmark") != "").astype(np.float32)
    F["landmark_2"] = (g2("addr_landmark") != "").astype(np.float32)

    # ---------------- python-loop features (multiprocess)
    py = _python_features({
        "n1": tok1, "n2": tok2, "a1": ad1, "a2": ad2, "i1": g1("name_initials"), "i2": g2("name_initials"),
        "num1": g1("addr_numbers"), "num2": g2("addr_numbers"), "u1": g1("addr_unit"), "u2": g2("addr_unit"),
        "sub1": g1("addr_suburb"), "sub2": g2("addr_suburb")}, ctx, executor)
    for j, name in enumerate(PY_FEATURES):
        F[name] = py[:, j]

    # ---------------- cross-field and combined
    F["name_in_other_addr"] = np.maximum(
        _nan_if_missing(_sim(tok1, ad2, fuzz.token_set_ratio, n_jobs) / 100, tok1, ad2),
        _nan_if_missing(_sim(tok2, ad1, fuzz.token_set_ratio, n_jobs) / 100, tok2, ad1))
    name_strength = np.nan_to_num(F["name_soft_tfidf"])
    addr_strength = np.nan_to_num(F["addr_char_cos"])
    F["name_x_addr"] = (name_strength * addr_strength).astype(np.float32)
    F["strong_name_weak_addr"] = ((np.nan_to_num(F["name_jw"]) >= 0.92) & (addr_strength < 0.3)).astype(np.float32)
    F["weak_name_strong_addr"] = ((np.nan_to_num(F["name_jw"]) < 0.7) & (addr_strength >= 0.8)).astype(np.float32)
    agree = (np.nan_to_num(F["hn_eq"]) + (np.nan_to_num(F["road_jw"]) >= 0.9) + np.nan_to_num(F["city_eq"])
             + np.nan_to_num(F["state_eq"]) + np.nan_to_num(F["pc_eq"]) + np.nan_to_num(F["unit_eq"]))
    conflict = (F["hn_conflict"] + F["city_conflict"] + F["state_conflict"] + F["pc_conflict"]
                + np.nan_to_num(F["unit_conflict"]))
    F["addr_components_agree"] = agree.astype(np.float32)
    F["addr_components_conflict"] = conflict.astype(np.float32)
    F["addr_components_both"] = sum(F[f"{c}_both"] for c in ("house_number", "road", "city", "state", "postcode", "unit")).astype(np.float32)
    F["contradictions"] = (conflict + F["suffix_conflict"] + np.nan_to_num(F["num_conflict"])).astype(np.float32)
    F["src_is_3"] = (pairs["source"].to_numpy() == 3).astype(np.float32)

    # ---------------- blocking provenance (label-free)
    for m in BLOCKING_FEATURE_METHODS:
        col = f"blk_{m}"
        F[col] = pairs[col].to_numpy(np.float32) if col in pairs else np.zeros(len(pairs), np.float32)
    F["blk_n_methods_syntactic"] = sum(F[f"blk_{m}"] for m in BLOCKING_FEATURE_METHODS).astype(np.float32)

    return pd.DataFrame({k: np.asarray(val, dtype=np.float32) for k, val in F.items()}, index=pairs.index)


def _chunks(n: int, size: int):
    for a in range(0, n, max(1, size)):
        yield a, min(n, a + max(1, size))


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, ctx: FeatureContext,
                     cfg: dict) -> pd.DataFrame:
    """In-memory feature frame (small inputs, tests). Large runs should use
    `build_feature_matrix`, which streams chunks to a disk-backed matrix."""
    log = get_logger()
    n_jobs = resolve_n_jobs(cfg.get("n_jobs", 1))
    size = int(cfg.get("features", {}).get("chunk_pairs", 250_000))
    with feature_workers(ctx, n_jobs if len(pairs) > 25_000 else 1) as ex:
        parts = [_base_features(pairs.iloc[a:b], s1, v, ctx, n_jobs, ex) for a, b in _chunks(len(pairs), size)]
    out = pd.concat(parts) if parts else _base_features(pairs, s1, v, ctx, n_jobs)
    if cfg.get("features", {}).get("context_features", True):
        out = add_context_features(pairs, out)
    log.info("Computed %d features for %d pairs", out.shape[1], len(out))
    return out


class FeatureMatrix:
    """Disk-backed float32 feature matrix (rows = candidate pairs).

    Stored as one or more row-major .npy parts (base features + context
    features) opened read-only with mmap: only the rows / columns a step needs
    are read into RAM, so the full matrix never has to fit in memory. (Pages
    read through the map are clean page cache that the OS can drop at will.)
    """

    def __init__(self, path: str, columns: list[str] | None = None):
        self.path = path
        meta = load_json(path + ".columns.json")
        if isinstance(meta, list):            # single-part layout
            meta = {"parts": [[os.path.basename(path), meta]]}
        folder = os.path.dirname(path)
        self.parts = [(np.load(os.path.join(folder, f), mmap_mode="r"), list(cols)) for f, cols in meta["parts"]]
        self.columns = [c for _, cols in self.parts for c in cols]
        if columns is not None and list(columns) != self.columns:
            raise ValueError("feature columns on disk differ from the expected columns")
        self.col_idx = {c: (pi, j) for pi, (_, cols) in enumerate(self.parts) for j, c in enumerate(cols)}

    def __len__(self) -> int:
        return int(self.parts[0][0].shape[0])

    @staticmethod
    def _rows(rows):
        if rows is None:
            return None
        rows = np.asarray(rows)
        return np.flatnonzero(rows) if rows.dtype == bool else rows.astype(np.int64)

    def _gather(self, r, cols: list[str], a: int | None = None, b: int | None = None,
                out: np.ndarray | None = None) -> np.ndarray:
        n = (b - a) if r is None else len(r)
        if out is None:
            out = np.empty((n, len(cols)), dtype=np.float32)
        for pi, (arr, _) in enumerate(self.parts):
            pos = [k for k, c in enumerate(cols) if self.col_idx[c][0] == pi]
            if not pos:
                continue
            lci = np.array([self.col_idx[cols[k]][1] for k in pos], dtype=np.int64)
            if r is None:
                out[:, pos] = arr[a:b][:, lci]
            else:
                out[:, pos] = arr[np.ix_(r, lci)]
        return out

    def frame(self, rows=None, cols: list[str] | None = None, index=None) -> pd.DataFrame:
        cols = list(cols) if cols is not None else self.columns
        r = self._rows(rows)
        data = self._gather(r, cols, 0, len(self)) if r is None else self._gather(r, cols)
        return pd.DataFrame(data, columns=cols, index=index)

    def fill(self, rows, cols: list[str], out: np.ndarray) -> np.ndarray:
        """Gather rows/cols straight into a pre-allocated float32 array."""
        return self._gather(self._rows(rows), list(cols), out=out)

    def blocks(self, cols: list[str], size: int = 500_000):
        for a, b in _chunks(len(self), size):
            yield a, b, self._gather(None, cols, a, b)


def build_feature_matrix(pairs: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, ctx: FeatureContext,
                         cfg: dict, path: str) -> FeatureMatrix:
    """Compute features chunk by chunk and append them to `path` (.npy) with
    plain sequential writes (no writable memory map, so the process never
    accumulates dirty file pages). The three columns the context features need
    are kept in RAM; context features go to a small second file."""
    log = get_logger()
    fcfg = cfg.get("features", {})
    n_jobs = resolve_n_jobs(cfg.get("n_jobs", 1))
    size = int(fcfg.get("chunk_pairs", 250_000))
    n = len(pairs)
    use_ctx = bool(fcfg.get("context_features", True))
    ensure_dir(os.path.dirname(path))
    base_cols, fh, offset = None, None, 0
    keep = {c: np.empty(n, dtype=np.float32) for c in ("full_char_cos", "name_soft_tfidf", "name_char_cos")}
    try:
        with feature_workers(ctx, n_jobs) as ex:
            for i, (a, b) in enumerate(_chunks(n, size)):
                part = _base_features(pairs.iloc[a:b], s1, v, ctx, n_jobs, ex)
                if fh is None:
                    base_cols = list(part.columns)
                    # header + correctly sized (sparse) file, then plain appends
                    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, len(base_cols)))
                    offset = mm.offset
                    del mm
                    fh = open(path, "r+b")
                block = np.ascontiguousarray(part[base_cols].to_numpy(np.float32))
                fh.seek(offset + a * len(base_cols) * 4)
                fh.write(block.tobytes())
                for c in keep:
                    keep[c][a:b] = part[c].to_numpy(np.float32)
                del part, block
                if i % 10 == 0 or b == n:
                    log.info("  features %d / %d pairs  [%s]", b, n, mem_str())
    finally:
        if fh is not None:
            fh.close()
    if base_cols is None:  # no pairs at all
        base_cols = list(_base_features(pairs, s1, v, ctx, 1).columns)
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(0, len(base_cols)))
    parts = [[os.path.basename(path), base_cols]]
    if use_ctx:
        ctx_path = path[:-4] + "_context.npy" if path.endswith(".npy") else path + "_context.npy"
        arrs = context_arrays(pairs, keep["full_char_cos"], keep["name_soft_tfidf"], keep["name_char_cos"])
        np.save(ctx_path, np.column_stack([arrs[c] for c in CONTEXT_COLUMNS]).astype(np.float32)
                if n else np.zeros((0, len(CONTEXT_COLUMNS)), np.float32))
        del arrs
        parts.append([os.path.basename(ctx_path), CONTEXT_COLUMNS])
    del keep
    save_json({"parts": parts}, path + ".columns.json")
    fm = FeatureMatrix(path)
    log.info("Computed %d features for %d pairs -> %s  [%s]", len(fm.columns), n, path, mem_str())
    return fm


def remove_feature_matrix(path: str) -> None:
    """Delete a feature matrix and all of its parts (features.keep_matrix: false)."""
    meta_path = path + ".columns.json"
    files = [path]
    if os.path.exists(meta_path):
        meta = load_json(meta_path)
        if isinstance(meta, dict):
            files += [os.path.join(os.path.dirname(path), f) for f, _ in meta.get("parts", [])]
        files.append(meta_path)
    for f in dict.fromkeys(files):
        if os.path.exists(f):
            os.remove(f)


def predict_in_blocks(booster, feats, cols: list[str], size: int = 500_000) -> np.ndarray:
    """LightGBM prediction without materialising the whole feature matrix."""
    n = len(feats)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    if isinstance(feats, FeatureMatrix):
        for a, b, X in feats.blocks(cols, size):
            out[a:b] = booster.predict(X)
    else:
        for a, b in _chunks(n, size):
            out[a:b] = booster.predict(feats.iloc[a:b][cols].to_numpy(np.float32))
    return out


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _rank_and_margin(codes: np.ndarray, values: np.ndarray):
    """Per group (integer `codes`): rank (1 = best), gap to the best, margin
    over the best OTHER member, group size. Fully vectorised."""
    v = np.nan_to_num(values.astype(np.float64), nan=-1.0)
    codes = np.asarray(codes, dtype=np.int64)
    n = len(v)
    if n == 0:
        z = np.zeros(0, np.float32)
        return z, z, z, z
    order = np.lexsort((-v, codes))
    sc, sv = codes[order], v[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sc)) + 1]
    sizes = np.diff(np.r_[starts, n])
    grp = np.repeat(np.arange(len(starts)), sizes)
    top = sv[starts]
    second = np.where(sizes > 1, sv[np.minimum(starts + 1, n - 1)], -1.0)
    # rank with ties -> 'min' method: position of first element with same value
    new_val = np.r_[True, (sv[1:] != sv[:-1]) | (sc[1:] != sc[:-1])]
    first_pos = np.maximum.accumulate(np.where(new_val, np.arange(n), 0))
    rank_sorted = (first_pos - starts[grp] + 1).astype(np.float32)
    top_s, second_s = top[grp], second[grp]
    is_top = sv >= top_s
    margin_s = sv - np.where(is_top, second_s, top_s)
    gap_s = top_s - sv
    size_s = sizes[grp].astype(np.float32)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return (rank_sorted[inv], gap_s[inv].astype(np.float32), margin_s[inv].astype(np.float32),
            size_s[inv])


CONTEXT_COLUMNS = ["ctx_rank", "ctx_gap_to_best", "ctx_margin", "ctx_n_cands", "ctx_name_rank",
                   "ctx_name_gap", "ctx_name_margin", "ctx_v_rank", "ctx_v_margin", "ctx_v_n_s1",
                   "ctx_mutual_best"]


def context_arrays(pairs: pd.DataFrame, full_char_cos: np.ndarray, name_soft_tfidf: np.ndarray,
                   name_char_cos: np.ndarray) -> dict[str, np.ndarray]:
    """Label-free features describing a pair relative to its competitors."""
    s1i = pairs["s1_idx"].to_numpy()
    src = pairs["source"].to_numpy()
    vi = pairs["v_idx"].to_numpy()
    base = np.nan_to_num(full_char_cos) * 0.5 + np.nan_to_num(name_soft_tfidf) * 0.5
    ent_src = s1i.astype(np.int64) * 4 + src.astype(np.int64)
    out = {}
    rank, gap, margin, size = _rank_and_margin(ent_src, base)
    out["ctx_rank"], out["ctx_gap_to_best"], out["ctx_margin"], out["ctx_n_cands"] = rank, gap, margin, size
    nrank, ngap, nmargin, _ = _rank_and_margin(ent_src, name_char_cos)
    out["ctx_name_rank"], out["ctx_name_gap"], out["ctx_name_margin"] = nrank, ngap, nmargin
    vrank, _, vmargin, vsize = _rank_and_margin(vi, base)
    out["ctx_v_rank"], out["ctx_v_margin"] = vrank, vmargin
    out["ctx_v_n_s1"] = np.minimum(vsize, 20).astype(np.float32)
    out["ctx_mutual_best"] = ((rank == 1) & (vrank == 1)).astype(np.float32)
    return {c: out[c].astype(np.float32) for c in CONTEXT_COLUMNS}


def add_context_features(pairs: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    for c, arr in context_arrays(pairs, feats["full_char_cos"].to_numpy(), feats["name_soft_tfidf"].to_numpy(),
                                 feats["name_char_cos"].to_numpy()).items():
        feats[c] = arr
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("s1_idx", "v_idx", "source", "label")]
