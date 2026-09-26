import numpy as np
import pandas as pd

from src.blocking import Blocker, blocking_report, minhash_signatures, sparse_topk
from src.features import FeatureContext, add_context_features, compute_features, soft_tfidf
from src.labels import LabelIndex

CFG = {"n_jobs": 1, "blocking": {"prefix_length": 4, "max_block_size": 100, "max_candidates_per_source": 10,
                                 "char_max_df": 1.0, "word_max_df": 1.0,
                                 "methods": {m: {"enabled": True, "top_k": 3} for m in [
                                     "name_prefix", "postcode", "postcode_phonetic", "city_state",
                                     "phonetic_name", "address_key", "tfidf_name_char", "tfidf_name_word",
                                     "tfidf_full_char", "tfidf_address", "minhash_lsh"]}},
       "features": {"soft_tfidf_threshold": 0.9}}


def test_soft_tfidf_tolerates_typos():
    idf = {"acme": 3.0, "robotics": 2.0, "robotiks": 3.0, "inc": 0.5}
    assert soft_tfidf(["acme", "robotics"], ["acme", "robotics"], idf) > 0.99
    assert soft_tfidf(["acme", "robotics"], ["acme", "robotiks"], idf) > 0.8
    assert soft_tfidf(["acme", "robotics"], ["zeta", "bakery"], idf) == 0.0
    assert np.isnan(soft_tfidf([], ["a"], idf))


def test_sparse_topk_matches_bruteforce():
    rng = np.random.default_rng(0)
    from scipy import sparse
    A = sparse.random(20, 30, density=0.3, random_state=1, format="csr")
    B = sparse.random(25, 30, density=0.3, random_state=2, format="csr")
    r, c, s = sparse_topk(A, B, 3, chunk=7)
    dense = (A @ B.T).toarray()
    for i in range(20):
        got = sorted(c[r == i].tolist())
        nz = np.where(dense[i] > 0)[0]
        exp = sorted(nz[np.argsort(-dense[i][nz])][:3].tolist())
        assert set(got) == set(exp) or np.allclose(sorted(dense[i][got]), sorted(dense[i][exp]))


def test_minhash_similar_strings_collide():
    sig, ok = minhash_signatures(["acmerobotics", "acmerobotic", "zetabakery", ""], num_perm=64)
    assert ok.tolist() == [True, True, True, False]
    assert (sig[0] == sig[1]).mean() > (sig[0] == sig[2]).mean()


def test_blocking_retrieves_true_pairs_and_reports(tiny_tables):
    s1, v = tiny_tables
    cand, precap, stats = Blocker(CFG).generate(s1, v)
    pairs = set(zip(cand["s1_idx"], cand["v_idx"]))
    for a, b in [(0, 0), (0, 3), (1, 2), (2, 4)]:   # true matches
        assert (a, b) in pairs
    assert (cand.groupby(["s1_idx", "source"]).size() <= 10).all()
    li = LabelIndex(key_mode="id", true_keys=[{"B1", "C1"}, {"B3"}, {"C2"}],
                    labeled=np.ones(3, bool),
                    true_pairs=pd.DataFrame({"s1_idx": [0, 0, 1, 2], "v_idx": [0, 3, 2, 4]}))
    rep = blocking_report(cand, precap, s1, v, li)
    assert rep["possible_pairs"] == 3 * 6
    assert rep["pair_completeness"] == 1.0
    assert 0 <= rep["reduction_ratio"] < 1
    assert rep["true_pairs_lost"] == 0
    assert "per_method" in rep


def test_features_sensible(tiny_tables):
    s1, v = tiny_tables
    pairs = pd.DataFrame({"s1_idx": [0, 0, 0], "v_idx": [0, 1, 5], "source": [2, 2, 3]})
    ctx = FeatureContext(s1, v, CFG)
    f = compute_features(pairs, s1, v, ctx, CFG)
    # same business (suffix variant) vs same name at other address vs unrelated
    assert f.loc[0, "name_exact_core"] == 1 and f.loc[0, "pc_eq"] == 1 and f.loc[0, "hn_eq"] == 1
    assert f.loc[1, "name_exact_core"] == 1 and f.loc[1, "hn_conflict"] == 1 and f.loc[1, "state_conflict"] == 1
    assert f.loc[1, "suffix_conflict"] == 1          # inc vs llc
    assert f.loc[0, "suffix_conflict"] == 0          # inc vs inc.
    assert f.loc[2, "name_jw"] < 0.7
    assert f.loc[0, "addr_char_cos"] > f.loc[1, "addr_char_cos"]
    assert f.loc[0, "contradictions"] == 0 and f.loc[1, "contradictions"] >= 2
    assert not f.isin([np.inf, -np.inf]).any().any()


def test_missing_fields_are_nan(tiny_tables):
    s1, v = tiny_tables
    s1 = s1.copy()
    s1.loc[0, "addr_postcode"] = ""
    pairs = pd.DataFrame({"s1_idx": [0], "v_idx": [0], "source": [2]})
    f = compute_features(pairs, s1, v, FeatureContext(s1, v, CFG), CFG)
    assert np.isnan(f.loc[0, "pc_eq"]) and f.loc[0, "postcode_both"] == 0


def test_context_features_rank():
    pairs = pd.DataFrame({"s1_idx": [0, 0, 1], "v_idx": [5, 6, 5], "source": [2, 2, 2]})
    feats = pd.DataFrame({"full_char_cos": [0.9, 0.2, 0.4], "name_soft_tfidf": [0.9, 0.1, 0.5],
                          "name_char_cos": [0.9, 0.1, 0.5]})
    out = add_context_features(pairs, feats)
    assert out["ctx_rank"].tolist() == [1, 2, 1]
    assert out["ctx_v_rank"].tolist() == [1, 1, 2]   # vendor 6 has one S1 candidate
    assert out["ctx_mutual_best"].tolist() == [1, 0, 0]


def test_feature_matrix_matches_in_memory_features(tiny_tables, tmp_path):
    from src.features import FeatureMatrix, build_feature_matrix, predict_in_blocks
    s1, v = tiny_tables
    pairs = pd.DataFrame({"s1_idx": [0, 0, 0, 1, 2, 2], "v_idx": [0, 1, 3, 2, 4, 5],
                          "source": [2, 2, 3, 2, 3, 3]})
    cfg = {**CFG, "features": {**CFG["features"], "chunk_pairs": 2}}   # force several chunks
    ctx = FeatureContext(s1, v, cfg)
    mem = compute_features(pairs, s1, v, ctx, cfg)
    fm = build_feature_matrix(pairs, s1, v, ctx, cfg, str(tmp_path / "f.npy"))
    assert list(fm.columns) == list(mem.columns)
    disk = fm.frame()
    np.testing.assert_allclose(disk.to_numpy(), mem.to_numpy(np.float32), equal_nan=True)
    again = FeatureMatrix(str(tmp_path / "f.npy"))            # columns reloaded from json
    sub = again.frame([1, 3], ["name_jw", "pc_eq"])
    np.testing.assert_allclose(sub.to_numpy(), mem.iloc[[1, 3]][["name_jw", "pc_eq"]].to_numpy(), equal_nan=True)

    class Mean:   # stand-in booster
        def predict(self, X):
            return np.nanmean(np.asarray(X, dtype=float), axis=1)
    cols = ["name_jw", "addr_char_cos"]
    np.testing.assert_allclose(predict_in_blocks(Mean(), fm, cols, size=4), predict_in_blocks(Mean(), mem, cols, size=4))
