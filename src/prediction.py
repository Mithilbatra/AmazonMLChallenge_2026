"""Test-time pipeline (python predict.py --config config.yaml).

  load test TSVs -> normalise/parse -> [full] encode + FAISS index
  -> blocking -> WRITE candidate_pairs.tsv (before any model scoring)
  -> features -> LightGBM -> calibration -> cascade -> cross-encoder
  -> blend -> final calibration -> graph cleanup -> decision rules
  -> WRITE matching_results.tsv (+ scored_candidates.tsv for debugging)
  -> format validation

No labels exist for the test set and none are used; every threshold comes
from models/<run_name>/pipeline.json, fitted on the training calib split.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .blocking import METHODS, Blocker, blocking_report
from .config import get, model_dir, output_dir
from .data_io import read_sample_submission_header, read_tsv, detect_list_format, load_sources, write_tsv
from .features import FeatureContext, compute_features
from .inference import TrainedPipeline, decide, score_candidates
from .schema import label_schema, submission_schema
from .submission import build_matching_results, validate_submission, write_candidate_pairs
from .utils import ensure_dir, save_json, set_seed, setup_logging, timer


def _list_format_from_labels(cfg: dict) -> str:
    path = get(cfg, "data.train.labels")
    if path and os.path.exists(path):
        df = read_tsv(path, cfg)
        cols = [c for c in label_schema(cfg).match_columns if c in df]
        if cols:
            return detect_list_format(df[cols[0]])
    return "plain"


def run_prediction(cfg: dict, split: str = "test") -> dict:
    t0 = time.time()
    mdir = model_dir(cfg)
    odir = os.path.join(output_dir(cfg), split)
    ensure_dir(odir)
    log = setup_logging(os.path.join(odir, "predict.log"))
    set_seed(int(cfg.get("seed", 42)))
    timings: dict = {}
    pipe = TrainedPipeline(mdir, cfg)
    log.info("=== PREDICTION  split=%s  mode=%s  decision=%s ===", split, pipe.meta["mode"], pipe.decision.to_dict())

    with timer("load data", timings):
        sources, load_stats = load_sources(cfg, split)
    from .preprocess import preprocess_dataset
    with timer("normalise + parse addresses", timings):
        s1, v, prep_stats = preprocess_dataset(sources, cfg)

    retriever = None
    if pipe.meta.get("has_semantic_blocking"):
        from .train_biencoder import SemanticRetriever
        retriever = SemanticRetriever(os.path.join(mdir, "biencoder"), cfg)
    with timer("blocking", timings):
        cand, precap, bstats = Blocker(cfg).generate(s1, v, retriever)
    if retriever is not None:
        retriever.save_indices(os.path.join(odir, "semantic_index"))

    sch = submission_schema(cfg, label_schema(cfg), _list_format_from_labels(cfg), read_sample_submission_header(cfg))
    cand_path = os.path.join(odir, get(cfg, "submission.candidate_pairs", "candidate_pairs.tsv"))
    write_candidate_pairs(cand, s1, v, cand_path, sch)   # BEFORE the matcher runs
    brep = blocking_report(cand, precap, s1, v)
    brep["generation"] = bstats
    save_json(brep, os.path.join(odir, "blocking_report.json"))
    log.info("Wrote %s (%d candidate pairs, reduction ratio %.6f)", cand_path, len(cand), brep["reduction_ratio"])

    with timer("pairwise features", timings):
        ctx = FeatureContext(s1, v, cfg)
        feats = compute_features(cand, s1, v, ctx, cfg)
    with timer("scoring (LightGBM / cascade / cross-encoder / blend)", timings):
        scored = score_candidates(cand, feats, s1, v, pipe)
    entity_idx = s1["s1_idx"].to_numpy()
    with timer("graph cleanup + decision", timings):
        selected, info = decide(scored, entity_idx, pipe, extra_cols=feats)
    scored["selected"] = selected

    results = build_matching_results(s1, v, scored, selected, sch)
    res_path = os.path.join(odir, get(cfg, "submission.matching_results", "matching_results.tsv"))
    write_tsv(results, res_path)
    dbg = scored.copy()
    dbg.insert(0, "source1_id", s1["id"].to_numpy()[dbg["s1_idx"].to_numpy()])
    dbg.insert(1, "candidate_id", v["id"].to_numpy()[dbg["v_idx"].to_numpy()])
    flags = [f"blk_{m}" for m in METHODS if f"blk_{m}" in cand]
    dbg["blocking_methods"] = [",".join(c[4:] for c, f in zip(flags, row) if f) for row in cand[flags].to_numpy()]
    dbg["graph_keep"] = info["graph_keep"]
    if info["matchable"] is not None:
        dbg["entity_p_matchable"] = info["matchable"][pd.Series(np.arange(len(entity_idx)), index=entity_idx)
                                                      .reindex(dbg["s1_idx"]).to_numpy()]
    write_tsv(dbg, os.path.join(odir, get(cfg, "submission.scored_candidates", "scored_candidates.tsv")))

    val = validate_submission(res_path, s1["id"].tolist(), set(v.loc[v["source"] == 2, "id"]),
                              set(v.loc[v["source"] == 3, "id"]), sch)
    n_match = int(selected.sum())
    per_ent = pd.Series(selected).groupby(scored["s1_idx"].to_numpy()).sum().reindex(entity_idx, fill_value=0)
    report = {
        "split": split, "records": {"source1": len(s1), "source2": int((v["source"] == 2).sum()),
                                    "source3": int((v["source"] == 3).sum())},
        "load": load_stats, "preprocess": prep_stats,
        "blocking": {k: brep[k] for k in ("candidate_pairs", "reduction_ratio", "candidates_per_entity",
                                          "entities_without_candidates")},
        "cascade_pairs_sent_to_ce": int(scored["sent_to_ce"].sum()),
        "graph": info["graph"], "decision": pipe.decision.to_dict(),
        "predicted_matches": n_match,
        "predicted_matches_s2": int((selected & (scored["source"] == 2).to_numpy()).sum()),
        "predicted_matches_s3": int((selected & (scored["source"] == 3).to_numpy()).sum()),
        "predicted_singletons": int((per_ent == 0).sum()),
        "predicted_singleton_rate": float((per_ent == 0).mean()),
        "validation": val, "outputs": {"matching_results": res_path, "candidate_pairs": cand_path},
    }
    timings["total"] = round(time.time() - t0, 1)
    report["timings_sec"] = timings
    save_json(report, os.path.join(odir, "prediction_report.json"))
    log.info("Predicted %d matches; %d of %d Source-1 entities predicted as singletons (%.1f%%)", n_match,
             report["predicted_singletons"], len(entity_idx), 100 * report["predicted_singleton_rate"])
    log.info("Validation: %s", "OK" if val["ok"] else val["errors"])
    log.info("Wrote %s  (total %.1fs)", res_path, timings["total"])
    return report
