"""Test-time pipeline as RESUMABLE STAGES (python predict.py --config config.yaml).

  prepare   load test TSVs -> normalise names / parse addresses (offline)
  blocking  [full] encode + FAISS index -> hybrid blocking
            -> WRITE candidate_pairs.tsv (before any model scoring)
  features  pairwise features streamed to a disk-backed matrix
  score     LightGBM -> calibration -> cascade -> cross-encoder -> blend
            -> final calibration -> graph cleanup -> decision rules
            -> WRITE matching_results.tsv (+ scored_candidates.tsv) -> validate

Checkpoints live in outputs/<run>/<split>/checkpoints/, so a notebook can
restart its kernel between stages. No labels exist for the test set and none
are used; every threshold comes from models/<run_name>/pipeline.json, fitted
on the training calib split.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .blocking import METHOD_BIT, METHODS, Blocker, blocking_report
from .config import get, model_dir, output_dir
from .data_io import detect_list_format, load_sources, read_sample_submission_header, read_tsv, write_tsv
from .decision import ENTITY_INPUT_COLUMNS
from .features import FeatureContext, FeatureMatrix, build_feature_matrix, remove_feature_matrix
from .inference import TrainedPipeline, decide, score_candidates
from .schema import label_schema, submission_schema
from .submission import build_matching_results, validate_submission, write_candidate_pairs
from .utils import (ensure_dir, load_json, load_pickle, mem_str, release_memory, save_json, save_pickle, set_seed,
                    setup_logging, timer)

PREDICT_STAGES = ["prepare", "blocking", "features", "score"]


def _list_format_from_labels(cfg: dict) -> str:
    path = get(cfg, "data.train.labels")
    if path and os.path.exists(path):
        df = read_tsv(path, cfg)
        cols = [c for c in label_schema(cfg).match_columns if c in df]
        if cols:
            return detect_list_format(df[cols[0]])
    return "plain"


def _odir(cfg: dict, split: str) -> str:
    return os.path.join(output_dir(cfg), split)


def _ck(cfg: dict, split: str, name: str) -> str:
    return os.path.join(_odir(cfg, split), "checkpoints", name)


def _start(cfg: dict, split: str, stage: str):
    ensure_dir(os.path.join(_odir(cfg, split), "checkpoints"))
    log = setup_logging(os.path.join(_odir(cfg, split), "predict.log"))
    set_seed(int(cfg.get("seed", 42)))
    log.info("=== PREDICT STAGE %-9s split=%s run=%s  [%s]", stage, split, cfg.get("run_name"), mem_str())
    return log


def _schema(cfg: dict):
    return submission_schema(cfg, label_schema(cfg), _list_format_from_labels(cfg), read_sample_submission_header(cfg))


def _timing(cfg: dict, split: str, stage: str, seconds: float):
    path = _ck(cfg, split, "timings.json")
    t = load_json(path) if os.path.exists(path) else {}
    t[stage] = round(seconds, 1)
    save_json(t, path)


def predict_prepare(cfg: dict, split: str = "test") -> dict:
    _start(cfg, split, "prepare")
    t0 = time.time()
    with timer("load data"):
        sources, load_stats = load_sources(cfg, split)
    from .preprocess import preprocess_dataset
    with timer("normalise + parse addresses"):
        s1, v, prep_stats = preprocess_dataset(sources, cfg)
    del sources
    save_pickle({"s1": s1, "v": v, "load": load_stats, "preprocess": prep_stats}, _ck(cfg, split, "prepare.pkl"))
    _timing(cfg, split, "prepare", time.time() - t0)
    return {"source1": len(s1), "vendors": len(v)}


def predict_blocking(cfg: dict, split: str = "test") -> dict:
    log = _start(cfg, split, "blocking")
    t0 = time.time()
    prep = load_pickle(_ck(cfg, split, "prepare.pkl"))
    s1, v = prep["s1"], prep["v"]
    pipe_meta = load_json(os.path.join(model_dir(cfg), "pipeline.json"))
    retriever = None
    if pipe_meta.get("has_semantic_blocking"):
        from .train_biencoder import SemanticRetriever
        retriever = SemanticRetriever(os.path.join(model_dir(cfg), "biencoder"), cfg)
    with timer("blocking"):
        cand, precap, bstats = Blocker(cfg).generate(s1, v, retriever)
    if retriever is not None:
        retriever.save_indices(os.path.join(_odir(cfg, split), "semantic_index"))
        del retriever
    cand_path = os.path.join(_odir(cfg, split), get(cfg, "submission.candidate_pairs", "candidate_pairs.tsv"))
    write_candidate_pairs(cand, s1, v, cand_path, _schema(cfg))   # BEFORE the matcher runs
    brep = blocking_report(cand, precap, s1, v)
    del precap
    brep["generation"] = bstats
    save_json(brep, os.path.join(_odir(cfg, split), "blocking_report.json"))
    save_pickle(cand, _ck(cfg, split, "candidates.pkl"))
    log.info("Wrote %s (%d candidate pairs, reduction ratio %.6f)", cand_path, len(cand), brep["reduction_ratio"])
    _timing(cfg, split, "blocking", time.time() - t0)
    return {"candidate_pairs": len(cand), "reduction_ratio": brep["reduction_ratio"], "file": cand_path}


def predict_features(cfg: dict, split: str = "test") -> dict:
    _start(cfg, split, "features")
    t0 = time.time()
    prep = load_pickle(_ck(cfg, split, "prepare.pkl"))
    cand = load_pickle(_ck(cfg, split, "candidates.pkl"))
    with timer("pairwise features (streamed to disk)"):
        ctx = FeatureContext(prep["s1"], prep["v"], cfg)
        fm = build_feature_matrix(cand, prep["s1"], prep["v"], ctx, cfg, _ck(cfg, split, "features.npy"))
    _timing(cfg, split, "features", time.time() - t0)
    return {"pairs": len(fm), "features": len(fm.columns)}


def _write_scored(scored: pd.DataFrame, cand: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, info: dict,
                  entity_idx: np.ndarray, path: str, block: int = 1_000_000):
    """Debug file written in blocks (no full-size copy of the scored frame)."""
    names = [m for m in METHODS if f"blk_{m}" in cand]
    flags = cand["blk_flags"].to_numpy() if "blk_flags" in cand else None
    matchable = None
    if info.get("matchable") is not None:
        pos = pd.Series(np.arange(len(entity_idx)), index=entity_idx)
        matchable = info["matchable"][pos.reindex(scored["s1_idx"]).to_numpy()]
    s1_ids, v_ids = s1["id"].to_numpy(), v["id"].to_numpy()
    ensure_dir(os.path.dirname(path))
    for i, a in enumerate(range(0, max(len(scored), 1), block)):
        b = min(len(scored), a + block)
        part = scored.iloc[a:b].copy()
        part.insert(0, "source1_id", s1_ids[part["s1_idx"].to_numpy()])
        part.insert(1, "candidate_id", v_ids[part["v_idx"].to_numpy()])
        if flags is not None:
            f = flags[a:b]
            part["blocking_methods"] = [",".join(m for m in names if x & METHOD_BIT[m]) for x in f]
        part["graph_keep"] = info["graph_keep"][a:b]
        if matchable is not None:
            part["entity_p_matchable"] = matchable[a:b]
        part.to_csv(path, sep="\t", index=False, mode="w" if i == 0 else "a", header=i == 0, lineterminator="\n")


def predict_score(cfg: dict, split: str = "test") -> dict:
    log = _start(cfg, split, "score")
    t0 = time.time()
    odir = _odir(cfg, split)
    pipe = TrainedPipeline(model_dir(cfg), cfg)
    log.info("decision parameters: %s", pipe.decision.to_dict())
    prep = load_pickle(_ck(cfg, split, "prepare.pkl"))
    s1, v = prep["s1"], prep["v"]
    cand = load_pickle(_ck(cfg, split, "candidates.pkl"))
    fm = FeatureMatrix(_ck(cfg, split, "features.npy"))
    with timer("scoring (LightGBM / cascade / cross-encoder / blend)"):
        scored = score_candidates(cand, fm, s1, v, pipe)
    entity_idx = s1["s1_idx"].to_numpy()
    with timer("graph cleanup + decision"):
        extra = fm.frame(None, [c for c in ENTITY_INPUT_COLUMNS if c in fm.col_idx], index=scored.index)
        selected, info = decide(scored, entity_idx, pipe, extra_cols=extra)
        del extra
    scored["selected"] = selected
    sch = _schema(cfg)
    results = build_matching_results(s1, v, scored, selected, sch)
    res_path = os.path.join(odir, get(cfg, "submission.matching_results", "matching_results.tsv"))
    write_tsv(results, res_path)
    if get(cfg, "submission.write_scored_candidates", True):
        with timer("write scored_candidates.tsv"):
            _write_scored(scored, cand, s1, v, info, entity_idx,
                          os.path.join(odir, get(cfg, "submission.scored_candidates", "scored_candidates.tsv")))
    val = validate_submission(res_path, s1["id"].tolist(), set(v.loc[v["source"] == 2, "id"]),
                              set(v.loc[v["source"] == 3, "id"]), sch)
    n_match = int(selected.sum())
    per_ent = pd.Series(selected).groupby(scored["s1_idx"].to_numpy()).sum().reindex(entity_idx, fill_value=0)
    brep = load_json(os.path.join(odir, "blocking_report.json"))
    cand_path = os.path.join(odir, get(cfg, "submission.candidate_pairs", "candidate_pairs.tsv"))
    _timing(cfg, split, "score", time.time() - t0)
    report = {
        "split": split, "records": {"source1": len(s1), "source2": int((v["source"] == 2).sum()),
                                    "source3": int((v["source"] == 3).sum())},
        "load": prep["load"], "preprocess": prep["preprocess"],
        "blocking": {k: brep[k] for k in ("candidate_pairs", "reduction_ratio", "candidates_per_entity",
                                          "entities_without_candidates") if k in brep},
        "cascade_pairs_sent_to_ce": int(scored["sent_to_ce"].sum()),
        "graph": info["graph"], "decision": pipe.decision.to_dict(),
        "predicted_matches": n_match,
        "predicted_matches_s2": int((selected & (scored["source"] == 2).to_numpy()).sum()),
        "predicted_matches_s3": int((selected & (scored["source"] == 3).to_numpy()).sum()),
        "predicted_singletons": int((per_ent == 0).sum()),
        "predicted_singleton_rate": float((per_ent == 0).mean()) if len(per_ent) else 0.0,
        "validation": val, "outputs": {"matching_results": res_path, "candidate_pairs": cand_path},
        "timings_sec": load_json(_ck(cfg, split, "timings.json")),
    }
    save_json(report, os.path.join(odir, "prediction_report.json"))
    log.info("Predicted %d matches; %d of %d Source-1 entities predicted as singletons (%.1f%%)", n_match,
             report["predicted_singletons"], len(entity_idx), 100 * report["predicted_singleton_rate"])
    log.info("Validation: %s", "OK" if val["ok"] else val["errors"])
    log.info("Wrote %s  [%s]", res_path, mem_str())
    return report


PREDICT_FUNCS = {"prepare": predict_prepare, "blocking": predict_blocking, "features": predict_features,
                 "score": predict_score}


def run_prediction(cfg: dict, split: str = "test", stages: list[str] | None = None) -> dict:
    out = {}
    for stage in stages or PREDICT_STAGES:
        if stage not in PREDICT_FUNCS:
            raise ValueError(f"unknown prediction stage {stage!r}; stages are {PREDICT_STAGES}")
        out = PREDICT_FUNCS[stage](cfg, split)
        release_memory()
    if get(cfg, "features.keep_matrix", True) is False:
        remove_feature_matrix(_ck(cfg, split, "features.npy"))
    return out
