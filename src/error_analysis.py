"""Error analysis on labelled splits (calib / holdout).

Writes, for every erroneous pair, the Source-1 and candidate raw and
normalised name/address, the key similarity features, the LightGBM,
cross-encoder and final scores:

  false_positives.tsv            selected but not a true match (false merges)
  false_negatives.tsv            true matches not selected (incl. lost in blocking)
  singleton_errors.tsv           singletons that received >=1 match
  nonsingletons_predicted_empty.tsv
  top_confident_errors.tsv       highest-probability false positives
  low_confidence_true_matches.tsv  true matches with the lowest scores
  difficult_cases.tsv            pairs near the threshold or where models disagree
  error_summary.json
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data_io import write_tsv
from .utils import ensure_dir, save_json

KEY_FEATURES = ["name_jw", "name_soft_tfidf", "name_token_set", "name_char_cos", "addr_char_cos",
                "addr_soft_tfidf", "hn_eq", "road_jw", "city_eq", "state_eq", "pc_eq", "unit_eq",
                "suffix_eq", "suffix_conflict", "contradictions", "ctx_rank", "ctx_mutual_best", "n_methods"]
SCORE_COLS = ["p_lgb_raw", "p_lgb", "sent_to_ce", "p_ce_raw", "p_ce", "p_blend", "p_final", "selected"]


def _describe_pairs(df: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    a = s1.set_index("s1_idx").loc[df["s1_idx"].to_numpy()]
    b = v.set_index("v_idx").loc[df["v_idx"].to_numpy()]
    out = pd.DataFrame({
        "s1_id": a["id"].to_numpy(), "s1_name": a["name"].to_numpy(), "s1_address": a["address"].to_numpy(),
        "s1_name_core": a["name_core"].to_numpy(), "s1_suffix": a["name_suffix"].to_numpy(),
        "s1_addr_clean": a["addr_clean"].to_numpy(),
        "cand_id": b["id"].to_numpy(), "cand_source": b["source"].to_numpy(), "cand_name": b["name"].to_numpy(),
        "cand_address": b["address"].to_numpy(), "cand_name_core": b["name_core"].to_numpy(),
        "cand_suffix": b["name_suffix"].to_numpy(), "cand_addr_clean": b["addr_clean"].to_numpy(),
    })
    for c in [*SCORE_COLS, *KEY_FEATURES, "is_true", "in_candidates"]:
        if c in df:
            out[c] = df[c].to_numpy()
    return out


def run_error_analysis(scored: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, true_pairs: pd.DataFrame,
                       n_true: np.ndarray, entity_idx: np.ndarray, threshold: float, out_dir: str,
                       top_n: int = 200) -> dict:
    ensure_dir(out_dir)
    ents = set(int(e) for e in entity_idx)
    df = scored[scored["s1_idx"].isin(ents)].copy()
    df["in_candidates"] = True
    sel = df["selected"].to_numpy(bool)
    tru = df["is_true"].to_numpy(bool)
    fp = df[sel & ~tru].sort_values("p_final", ascending=False)
    fn_cand = df[~sel & tru]
    tp_all = true_pairs[true_pairs["s1_idx"].isin(ents)]
    lost = tp_all.merge(df[["s1_idx", "v_idx"]], on=["s1_idx", "v_idx"], how="left", indicator=True)
    lost = lost.loc[lost["_merge"] == "left_only", ["s1_idx", "v_idx"]]
    lost = lost.assign(in_candidates=False, is_true=True, selected=False)
    fn = pd.concat([fn_cand, lost], ignore_index=True)
    fn["error_type"] = np.where(fn["in_candidates"].astype(bool), "missed_by_matcher", "lost_in_blocking")

    n_pred = df[sel].groupby("s1_idx").size()
    ent = pd.DataFrame({"s1_idx": list(ents)})
    ent["n_true"] = n_true[ent["s1_idx"].to_numpy()]
    ent["n_pred"] = ent["s1_idx"].map(n_pred).fillna(0).astype(int)
    singleton_err = ent[(ent["n_true"] == 0) & (ent["n_pred"] > 0)]
    nonsingle_empty = ent[(ent["n_true"] > 0) & (ent["n_pred"] == 0)]

    fp_desc = _describe_pairs(fp, s1, v) if len(fp) else pd.DataFrame()
    fn_desc = _describe_pairs(fn, s1, v) if len(fn) else pd.DataFrame()
    if len(fn_desc):
        fn_desc["error_type"] = fn["error_type"].to_numpy()
    write_tsv(fp_desc, os.path.join(out_dir, "false_positives.tsv"))
    write_tsv(fn_desc, os.path.join(out_dir, "false_negatives.tsv"))
    se = fp[fp["s1_idx"].isin(set(singleton_err["s1_idx"]))]
    write_tsv(_describe_pairs(se, s1, v) if len(se) else pd.DataFrame(), os.path.join(out_dir, "singleton_errors.tsv"))
    ne = df[df["s1_idx"].isin(set(nonsingle_empty["s1_idx"])) & tru]
    write_tsv(_describe_pairs(ne, s1, v) if len(ne) else pd.DataFrame(),
              os.path.join(out_dir, "nonsingletons_predicted_empty.tsv"))
    write_tsv(fp_desc.head(top_n), os.path.join(out_dir, "top_confident_errors.tsv"))
    low_true = df[tru].sort_values("p_final").head(top_n)
    write_tsv(_describe_pairs(low_true, s1, v) if len(low_true) else pd.DataFrame(),
              os.path.join(out_dir, "low_confidence_true_matches.tsv"))
    near = (df["p_final"] - threshold).abs() < 0.1
    disagree = (df["p_lgb"] - df["p_ce"]).abs() > 0.5 if "p_ce" in df else pd.Series(False, index=df.index)
    hard = df[near | disagree.fillna(False)].sort_values("p_final", ascending=False).head(top_n * 2)
    write_tsv(_describe_pairs(hard, s1, v) if len(hard) else pd.DataFrame(), os.path.join(out_dir, "difficult_cases.tsv"))
    summary = {
        "entities": int(len(ent)),
        "false_positive_pairs": int(len(fp)),
        "false_negative_pairs": int(len(fn)),
        "false_negatives_missed_by_matcher": int((fn["error_type"] == "missed_by_matcher").sum()),
        "false_negatives_lost_in_blocking": int((fn["error_type"] == "lost_in_blocking").sum()),
        "singletons_with_false_merges": int(len(singleton_err)),
        "nonsingletons_predicted_empty": int(len(nonsingle_empty)),
        "false_positives_by_source": fp["source"].value_counts().to_dict() if len(fp) else {},
        "difficult_cases": int(len(hard)),
    }
    save_json(summary, os.path.join(out_dir, "error_summary.json"))
    return summary
