"""Hybrid candidate generation (blocking).

PDF "Hybrid Candidate Generation": a syntactic path (exact keys, phonetic
hashing, LSH/MinHash, TF-IDF) and a semantic path (SC-Block bi-encoder +
FAISS, see train_biencoder.py). The UNION of all paths is the candidate set
written to candidate_pairs.tsv. Transcript: "records can group either through
a similar name or through a shared address" - so both name-driven and
address-driven blocks are used.

Every method retrieves Source-2 and Source-3 candidates SEPARATELY for each
Source-1 entity (the task is Source-1-centric). Methods:

  name_prefix        first `prefix_length` chars of the normalised core name
  postcode           exact normalised postcode
  postcode_phonetic  postcode + phonetic code of the first name token (PDF)
  city_state         parsed city + state
  phonetic_name      metaphone code of the whole core name
  address_key        house number + first significant road token ("shared address")
  tfidf_name_char    char n-gram TF-IDF cosine top-k on the name
  tfidf_name_word    word TF-IDF cosine top-k on the name
  tfidf_full_char    char n-gram TF-IDF top-k on name + address
  tfidf_address      char n-gram TF-IDF top-k on the address only
  minhash_lsh        MinHash signatures of name shingles, banded LSH buckets
  semantic           bi-encoder embeddings + FAISS top-k (full mode)

Key blocks larger than `top_k` are ranked inside the block by TF-IDF cosine
(so a big postcode block keeps the k most name-similar records); blocks larger
than `max_block_size` are purged (standard block purging). Finally every
(entity, source) keeps at most `max_candidates_per_source` pairs ranked by a
cheap similarity. TF-IDF statistics are fitted on the unlabelled text of the
dataset being blocked; no label is ever used for blocking.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import get, semantic_enabled
from .metrics import entity_fbeta, reduction_ratio
from .utils import describe_counts, get_logger

METHODS = ["name_prefix", "postcode", "postcode_phonetic", "city_state", "phonetic_name",
           "address_key", "tfidf_name_char", "tfidf_name_word", "tfidf_full_char",
           "tfidf_address", "minhash_lsh", "semantic"]
METHOD_BIT = {m: 1 << i for i, m in enumerate(METHODS)}


# ---------------------------------------------------------------- TF-IDF
def _fit_tfidf(texts_a: pd.Series, texts_b: pd.Series, **kwargs):
    corpus = pd.concat([texts_a, texts_b], ignore_index=True).fillna("")
    vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, lowercase=False, **kwargs)
    try:
        vec.fit(corpus)
    except ValueError:  # vocabulary pruned away (tiny corpus) -> no max_df
        kwargs = {k: val for k, val in kwargs.items() if k != "max_df"}
        vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, lowercase=False, **kwargs)
        try:
            vec.fit(corpus)
        except ValueError:
            return None, None, None
    return vec, vec.transform(texts_a.fillna("")).tocsr(), vec.transform(texts_b.fillna("")).tocsr()


class TextRepresentations:
    """Sparse, L2-normalised TF-IDF matrices for S1 and vendor records."""

    def __init__(self, s1: pd.DataFrame, v: pd.DataFrame, ngram_range=(2, 4), char_max_df=1.0,
                 word_max_df=1.0):
        ngram_range = tuple(ngram_range)
        self.mats: dict[str, tuple] = {}
        specs = {
            "name_char": ("name_tokens", dict(analyzer="char_wb", ngram_range=ngram_range, max_df=char_max_df)),
            "name_word": ("name_tokens", dict(analyzer="word", token_pattern=r"\S+", max_df=word_max_df)),
            "full_char": ("full_text", dict(analyzer="char_wb", ngram_range=ngram_range, max_df=char_max_df)),
            "addr_char": ("addr_clean", dict(analyzer="char_wb", ngram_range=ngram_range, max_df=char_max_df)),
        }
        for name, (col, kw) in specs.items():
            _, a, b = _fit_tfidf(s1[col], v[col], **kw)
            if a is None:
                a = sparse.csr_matrix((len(s1), 1), dtype=np.float32)
                b = sparse.csr_matrix((len(v), 1), dtype=np.float32)
            self.mats[name] = (a, b)

    def s1(self, name):
        return self.mats[name][0]

    def v(self, name):
        return self.mats[name][1]


def rowwise_cosine(A: sparse.csr_matrix, B: sparse.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                   chunk: int = 200_000) -> np.ndarray:
    out = np.zeros(len(ia), dtype=np.float32)
    for start in range(0, len(ia), chunk):
        sl = slice(start, start + chunk)
        prod = A[ia[sl]].multiply(B[ib[sl]])
        out[sl] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def sparse_topk(A: sparse.csr_matrix, B: sparse.csr_matrix, k: int, chunk: int = 2000,
                min_sim: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k rows of B for every row of A by cosine (both L2-normalised)."""
    rows, cols, sims = [], [], []
    BT = B.T.tocsr()
    for start in range(0, A.shape[0], chunk):
        S = (A[start:start + chunk] @ BT).tocsr()
        indptr, indices, data = S.indptr, S.indices, S.data
        for r in range(S.shape[0]):
            lo, hi = indptr[r], indptr[r + 1]
            if hi == lo:
                continue
            d = data[lo:hi]
            if hi - lo > k:
                sel = np.argpartition(-d, k - 1)[:k]
            else:
                sel = np.arange(hi - lo)
            sel = sel[d[sel] >= min_sim]
            rows.append(np.full(len(sel), start + r, dtype=np.int64))
            cols.append(indices[lo:hi][sel].astype(np.int64))
            sims.append(d[sel])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)


# ------------------------------------------------------------- key blocks
def key_block(s1_keys: np.ndarray, v_keys: np.ndarray, v_global: np.ndarray,
              rank_s1: sparse.csr_matrix, rank_v: sparse.csr_matrix, top_k: int,
              max_block: int) -> tuple[pd.DataFrame, dict]:
    """Candidates sharing an exact key; oversized blocks ranked / purged.

    `v_keys`, `rank_v` and `v_global` are aligned (vendor rows of ONE source).
    """
    s1df = pd.DataFrame({"s1_idx": np.arange(len(s1_keys)), "key": s1_keys})
    s1df = s1df[s1df["key"] != ""]
    vdf = pd.DataFrame({"vpos": np.arange(len(v_keys)), "key": v_keys})
    vdf = vdf[vdf["key"] != ""]
    counts = vdf["key"].value_counts()
    counts = counts[counts.index.isin(set(s1df["key"]))]
    small = set(counts.index[counts <= top_k])
    mid = set(counts.index[(counts > top_k) & (counts <= max_block)])
    purged = counts[counts > max_block]
    parts = []
    if small:
        m = s1df[s1df["key"].isin(small)].merge(vdf[vdf["key"].isin(small)], on="key")
        parts.append(m[["s1_idx", "vpos"]])
    if mid:
        s1_groups = s1df[s1df["key"].isin(mid)].groupby("key")["s1_idx"].apply(np.asarray)
        v_groups = vdf[vdf["key"].isin(mid)].groupby("key")["vpos"].apply(np.asarray)
        for key, s1_rows in s1_groups.items():
            v_rows = v_groups[key]
            S = (rank_s1[s1_rows] @ rank_v[v_rows].T).toarray()
            kk = min(top_k, S.shape[1])
            top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            parts.append(pd.DataFrame({"s1_idx": np.repeat(s1_rows, kk), "vpos": v_rows[top].ravel()}))
    stats = {"blocks": int(len(counts)), "blocks_ranked": len(mid), "blocks_purged": int(len(purged)),
             "vendor_rows_in_purged_blocks": int(purged.sum())}
    if not parts:
        return pd.DataFrame({"s1_idx": [], "v_idx": []}, dtype=np.int64), stats
    out = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"s1_idx": out["s1_idx"].to_numpy(np.int64),
                         "v_idx": v_global[out["vpos"].to_numpy(np.int64)]}), stats


# ---------------------------------------------------------------- MinHash
_MERSENNE = (1 << 31) - 1


def _shingles(texts, k: int) -> tuple[np.ndarray, np.ndarray]:
    flat, lens = [], []
    for t in texts:
        if not t:
            lens.append(0)
            continue
        sh = {t[i:i + k] for i in range(max(1, len(t) - k + 1))}
        flat.extend(zlib.crc32(x.encode("utf-8")) for x in sh)
        lens.append(len(sh))
    return np.asarray(flat, dtype=np.int64), np.asarray(lens, dtype=np.int64)


def minhash_signatures(texts, num_perm: int = 64, k: int = 3, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised MinHash (universal hashing a*x+b mod p). Returns (sig, valid)."""
    flat, lens = _shingles(texts, k)
    n = len(lens)
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, num_perm, dtype=np.int64)
    b = rng.integers(0, _MERSENNE, num_perm, dtype=np.int64)
    sig = np.full((n, num_perm), _MERSENNE, dtype=np.int64)
    valid = lens > 0
    if flat.size == 0:
        return sig, valid
    x = flat % _MERSENNE
    offsets = np.concatenate([[0], np.cumsum(lens)])[:-1][valid]
    step = max(1, int(2e7 // max(1, x.size)))
    for s in range(0, num_perm, step):
        cols = slice(s, min(num_perm, s + step))
        h = (x[:, None] * a[None, cols] + b[None, cols]) % _MERSENNE
        sig[valid, cols] = np.minimum.reduceat(h, offsets, axis=0)
    return sig, valid


def _band_keys(sig: np.ndarray, bands: int) -> np.ndarray:
    rows = sig.shape[1] // bands
    rng = np.random.default_rng(12345)
    mult = rng.integers(1, 2**62, rows, dtype=np.int64).astype(np.uint64) | np.uint64(1)
    keys = np.empty((sig.shape[0], bands), dtype=np.uint64)
    u = sig.astype(np.uint64)
    with np.errstate(over="ignore"):
        for bi in range(bands):
            keys[:, bi] = (u[:, bi * rows:(bi + 1) * rows] * mult).sum(axis=1)
    return keys


def minhash_block(s1_texts, v_texts, v_global: np.ndarray, top_k: int, num_perm: int, bands: int,
                  k: int, max_bucket: int) -> tuple[pd.DataFrame, dict]:
    sig1, ok1 = minhash_signatures(list(s1_texts), num_perm, k)
    sig2, ok2 = minhash_signatures(list(v_texts), num_perm, k)
    keys1, keys2 = _band_keys(sig1, bands), _band_keys(sig2, bands)
    parts, purged = [], 0
    idx1, idx2 = np.where(ok1)[0], np.where(ok2)[0]
    for bi in range(bands):
        d1 = pd.DataFrame({"s1_idx": idx1, "key": keys1[idx1, bi]})
        d2 = pd.DataFrame({"vpos": idx2, "key": keys2[idx2, bi]})
        cnt = d2["key"].value_counts()
        big = cnt.index[cnt > max_bucket]
        purged += len(big)
        d2 = d2[~d2["key"].isin(big)]
        parts.append(d1.merge(d2, on="key")[["s1_idx", "vpos"]])
    if not parts:
        return pd.DataFrame({"s1_idx": [], "v_idx": []}, dtype=np.int64), {}
    pairs = pd.concat(parts, ignore_index=True).drop_duplicates()
    if pairs.empty:
        return pd.DataFrame({"s1_idx": [], "v_idx": []}, dtype=np.int64), {"buckets_purged": purged}
    a = pairs["s1_idx"].to_numpy()
    bpos = pairs["vpos"].to_numpy()
    est = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), 500_000):
        est[s:s + 500_000] = (sig1[a[s:s + 500_000]] == sig2[bpos[s:s + 500_000]]).mean(axis=1)
    pairs = pairs.assign(sim=est).sort_values(["s1_idx", "sim"], ascending=[True, False])
    pairs = pairs.groupby("s1_idx", sort=False).head(top_k)
    return (pd.DataFrame({"s1_idx": pairs["s1_idx"].to_numpy(np.int64),
                          "v_idx": v_global[pairs["vpos"].to_numpy(np.int64)]}),
            {"buckets_purged": int(purged)})


# ---------------------------------------------------------------- blocker
class Blocker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bcfg = get(cfg, "blocking", {})
        self.methods_cfg = self.bcfg.get("methods", {})

    def _enabled(self, method: str) -> bool:
        if method == "semantic":
            return semantic_enabled(self.cfg)
        return bool(self.methods_cfg.get(method, {}).get("enabled", False))

    def _k(self, method: str, default: int = 20) -> int:
        return int(self.methods_cfg.get(method, {}).get("top_k", default))

    def generate(self, s1: pd.DataFrame, v: pd.DataFrame, semantic_retriever=None):
        """Return (candidates, precap_pairs, stats).

        candidates: [s1_idx, v_idx, source, blk_<method>..., n_methods, blk_score]
        """
        log = get_logger()
        b = self.bcfg
        reps = TextRepresentations(s1, v, b.get("char_ngram_range", [2, 4]),
                                   b.get("char_max_df", 0.05), b.get("word_max_df", 0.10))
        prefix_len = int(b.get("prefix_length", 4))
        max_block = int(b.get("max_block_size", 2000))
        chunk = int(b.get("chunk_size", 2000))
        stats: dict = {"methods": {}}
        frames = []

        s1_keys = {
            "name_prefix": np.where(s1["name_nospace"].str.len() >= 2,
                                    s1["name_nospace"].str[:prefix_len], "").astype(object),
            "postcode": s1["addr_postcode"].to_numpy(object),
            "postcode_phonetic": np.where((s1["addr_postcode"] != "") & (s1["name_first_phonetic"] != ""),
                                          s1["addr_postcode"] + "|" + s1["name_first_phonetic"], "").astype(object),
            "city_state": s1["city_state_key"].to_numpy(object),
            "phonetic_name": s1["name_phonetic"].str[:10].to_numpy(object),
            "address_key": s1["addr_street_key"].to_numpy(object),
        }
        v_keys_all = {
            "name_prefix": np.where(v["name_nospace"].str.len() >= 2,
                                    v["name_nospace"].str[:prefix_len], "").astype(object),
            "postcode": v["addr_postcode"].to_numpy(object),
            "postcode_phonetic": np.where((v["addr_postcode"] != "") & (v["name_first_phonetic"] != ""),
                                          v["addr_postcode"] + "|" + v["name_first_phonetic"], "").astype(object),
            "city_state": v["city_state_key"].to_numpy(object),
            "phonetic_name": v["name_phonetic"].str[:10].to_numpy(object),
            "address_key": v["addr_street_key"].to_numpy(object),
        }
        # ranking representation inside oversized key blocks
        rank_rep = {"name_prefix": "full_char", "postcode": "name_char", "postcode_phonetic": "name_char",
                    "city_state": "name_char", "phonetic_name": "full_char", "address_key": "addr_char"}
        tfidf_rep = {"tfidf_name_char": "name_char", "tfidf_name_word": "name_word",
                     "tfidf_full_char": "full_char", "tfidf_address": "addr_char"}

        for source in (2, 3):
            vmask = (v["source"] == source).to_numpy()
            v_global = np.where(vmask)[0]
            if len(v_global) == 0:
                continue
            for method in METHODS:
                if not self._enabled(method):
                    continue
                k = self._k(method)
                mstats = {}
                if method in s1_keys:
                    rep = rank_rep[method]
                    df, mstats = key_block(s1_keys[method], v_keys_all[method][vmask], v_global,
                                           reps.s1(rep), reps.v(rep)[v_global], k, max_block)
                elif method in tfidf_rep:
                    rep = tfidf_rep[method]
                    r, c, _ = sparse_topk(reps.s1(rep), reps.v(rep)[v_global], k, chunk)
                    df = pd.DataFrame({"s1_idx": r, "v_idx": v_global[c]})
                elif method == "minhash_lsh":
                    mc = self.methods_cfg.get("minhash_lsh", {})
                    df, mstats = minhash_block(s1["name_nospace"], v.loc[vmask, "name_nospace"], v_global, k,
                                               int(mc.get("num_perm", 64)), int(mc.get("bands", 16)),
                                               int(mc.get("shingle_size", 3)),
                                               int(mc.get("max_bucket_size", 300)))
                elif method == "semantic":
                    if semantic_retriever is None:
                        log.warning("semantic blocking enabled but no retriever supplied - skipped")
                        continue
                    df = semantic_retriever.search(s1, v, source, k)[["s1_idx", "v_idx"]]
                else:  # pragma: no cover
                    continue
                df = df.assign(bit=METHOD_BIT[method])
                frames.append(df)
                stats["methods"][f"{method}_s{source}"] = {"pairs": int(len(df)), **mstats}
                log.info("  blocking %-18s source %d -> %9d pairs", method, source, len(df))

        if not frames:
            raise RuntimeError("No blocking method produced candidates - check blocking.methods")
        allp = pd.concat(frames, ignore_index=True)
        allp["s1_idx"] = allp["s1_idx"].astype(np.int64)
        allp["v_idx"] = allp["v_idx"].astype(np.int64)
        allp = allp.drop_duplicates(["s1_idx", "v_idx", "bit"])
        cand = allp.groupby(["s1_idx", "v_idx"], sort=False)["bit"].sum().reset_index()
        cand.rename(columns={"bit": "blk_flags"}, inplace=True)
        cand["source"] = v["source"].to_numpy()[cand["v_idx"].to_numpy()]
        ia, ib = cand["s1_idx"].to_numpy(), cand["v_idx"].to_numpy()
        # Cap ranking: joint name+address similarity plus a small bonus per
        # independent method that retrieved the pair. (Ranking by name
        # similarity alone lets many same-name branches of a common business
        # push the true, address-matching record out of the cap.)
        flags = cand["blk_flags"].to_numpy()
        n_methods = sum(((flags & bit) > 0).astype(np.float32) for bit in METHOD_BIT.values())
        bonus = float(b.get("cap_method_bonus", 0.02))
        cand["blk_score"] = rowwise_cosine(reps.s1("full_char"), reps.v("full_char"), ia, ib) + bonus * n_methods
        precap = cand[["s1_idx", "v_idx", "source", "blk_flags"]].copy()
        cap = int(b.get("max_candidates_per_source", 60))
        cand = cand.sort_values(["s1_idx", "source", "blk_score"], ascending=[True, True, False])
        cand = cand.groupby(["s1_idx", "source"], sort=False).head(cap).reset_index(drop=True)
        for m in METHODS:
            cand[f"blk_{m}"] = ((cand["blk_flags"].to_numpy() & METHOD_BIT[m]) > 0).astype(np.int8)
        cand["n_methods"] = cand[[f"blk_{m}" for m in METHODS]].sum(axis=1).astype(np.int16)
        cand = cand.sort_values(["s1_idx", "source", "v_idx"]).reset_index(drop=True)
        stats["pairs_before_cap"] = int(len(precap))
        stats["pairs_after_cap"] = int(len(cand))
        log.info("Blocking union: %d pairs before cap, %d after cap (%d per source)", len(precap), len(cand), cap)
        return cand, precap, stats


# ------------------------------------------------------------ evaluation
def blocking_report(cand: pd.DataFrame, precap: pd.DataFrame | None, s1: pd.DataFrame, v: pd.DataFrame,
                    li=None, split: np.ndarray | None = None, beta: float = 0.5) -> dict:
    """Pair completeness, reduction ratio, candidate distribution, lost matches,
    per-method contribution and the Macro-F0.5 ceiling imposed by blocking."""
    n1 = len(s1)
    n2 = int((v["source"] == 2).sum())
    n3 = int((v["source"] == 3).sum())
    possible = n1 * (n2 + n3)
    rep: dict = {
        "source1_records": n1, "source2_records": n2, "source3_records": n3,
        "possible_pairs": possible, "possible_pairs_s2": n1 * n2, "possible_pairs_s3": n1 * n3,
        "candidate_pairs": int(len(cand)),
        "candidate_pairs_s2": int((cand["source"] == 2).sum()),
        "candidate_pairs_s3": int((cand["source"] == 3).sum()),
    }
    rep["reduction_ratio"] = reduction_ratio(len(cand), possible)
    rep["reduction_ratio_s2"] = reduction_ratio(rep["candidate_pairs_s2"], n1 * n2)
    rep["reduction_ratio_s3"] = reduction_ratio(rep["candidate_pairs_s3"], n1 * n3)
    per_ent = cand.groupby("s1_idx").size().reindex(range(n1), fill_value=0)
    rep["candidates_per_entity"] = describe_counts(per_ent.to_numpy())
    rep["entities_without_candidates"] = int((per_ent == 0).sum())
    for s in (2, 3):
        pe = cand[cand["source"] == s].groupby("s1_idx").size().reindex(range(n1), fill_value=0)
        rep[f"candidates_per_entity_s{s}"] = describe_counts(pe.to_numpy())
    rep["candidate_count_histogram"] = {
        str(k): int(val) for k, val in pd.cut(per_ent, [-1, 0, 5, 10, 20, 50, 100, 200, 10**9]).value_counts(sort=False).items()}
    if li is None:
        return rep

    tp = li.true_pairs.merge(v[["v_idx", "source"]], on="v_idx")
    key = lambda df: set(zip(df["s1_idx"].to_numpy(), df["v_idx"].to_numpy()))
    cand_set = key(cand)
    found = np.array([(a, b) in cand_set for a, b in zip(tp["s1_idx"], tp["v_idx"])], dtype=bool)
    tp = tp.assign(found=found)
    rep["true_pairs_resolved"] = int(len(tp))
    rep["pair_completeness"] = float(found.mean()) if len(tp) else float("nan")
    rep["true_pairs_lost"] = int((~found).sum())
    for s in (2, 3):
        m = tp["source"] == s
        rep[f"pair_completeness_s{s}"] = float(tp.loc[m, "found"].mean()) if m.any() else float("nan")
    if precap is not None:
        pre_set = key(precap)
        pre_found = np.array([(a, b) in pre_set for a, b in zip(tp["s1_idx"], tp["v_idx"])], dtype=bool)
        rep["pair_completeness_before_cap"] = float(pre_found.mean()) if len(tp) else float("nan")
        rep["pairs_before_cap"] = int(len(precap))
        flags = precap.set_index(["s1_idx", "v_idx"])["blk_flags"]
        tp_flags = flags.reindex(list(zip(tp["s1_idx"], tp["v_idx"]))).fillna(0).astype(np.int64).to_numpy()
        pc_flags = precap["blk_flags"].to_numpy()
        method_stats = {}
        for m in METHODS:
            bit = METHOD_BIT[m]
            has = (tp_flags & bit) > 0
            only = has & (tp_flags == bit)
            method_stats[m] = {
                "candidate_pairs": int(((pc_flags & bit) > 0).sum()),
                "recall": float(has.mean()) if len(tp) else float("nan"),
                "unique_true_pairs": int(only.sum()),
            }
        rep["per_method"] = method_stats

    # entity-level view and the ceiling blocking imposes on Macro F0.5
    n_true = li.n_true
    found_per_ent = np.bincount(tp.loc[tp["found"], "s1_idx"], minlength=n1)
    oracle = entity_fbeta(found_per_ent, found_per_ent, n_true, beta)
    labeled = li.labeled
    nonsingle = (n_true > 0) & labeled
    rep["nonsingleton_entities"] = int(nonsingle.sum())
    rep["nonsingleton_entities_all_matches_lost"] = int((nonsingle & (found_per_ent == 0)).sum())
    rep["oracle_macro_f0.5_given_blocking"] = float(oracle[labeled].mean()) if labeled.any() else float("nan")
    if split is not None:
        per_split = {}
        ent_split = split[tp["s1_idx"].to_numpy()] if len(tp) else np.array([])
        for name in pd.unique(split):
            ms = ent_split == name
            em = (split == name) & labeled
            per_split[str(name)] = {
                "pair_completeness": float(tp.loc[ms, "found"].mean()) if ms.any() else float("nan"),
                "oracle_macro_f0.5": float(oracle[em].mean()) if em.any() else float("nan"),
                "entities": int(em.sum()),
            }
        rep["per_split"] = per_split
    return rep
