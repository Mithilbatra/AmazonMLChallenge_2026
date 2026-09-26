"""Training pipeline as RESUMABLE STAGES (python train.py --config config.yaml).

  prepare       ingest TSVs, validate schema, parse labels, normalise names,
                parse addresses (offline), resolve labels, entity-level split
  biencoder     [full] SupCon bi-encoder on fit entities (early stop on fit_val)
  blocking      hybrid blocking over ALL training Source-1 entities + report
  features      pairwise features streamed to a disk-backed matrix, labels,
                hard-negative tags, features of true pairs lost in blocking
  lightgbm      LightGBM on fit pairs (early stop on fit_val) + isotonic
                calibration on calib
  crossencoder  [full] Ditto cross-encoder on fit pairs, cascade scoring of
                calib / holdout, isotonic calibration on calib
  decide        blend selection, final calibration, singleton model, graph
                modes and Macro-F0.5 decision search on calib; unbiased holdout
                evaluation + error analysis; pipeline.json

Every stage reads its inputs from outputs/<run>/train/checkpoints/ and writes
its outputs there, then returns - so its memory is released. `run_training`
simply runs the stages in order; `--resume` skips completed stages, and a
notebook can restart its kernel between any two stages.
"""
from __future__ import annotations

import gc
import os
import platform
import time

import numpy as np
import pandas as pd
import yaml

from .blocking import Blocker, blocking_report
from .calibration import Calibrator, fit_calibrator
from .config import get, is_full, model_dir, output_dir, semantic_enabled
from .data_io import load_labels, load_sources
from .decision import (ENTITY_INPUT_COLUMNS, DecisionParams, DecisionTable, effective_min_improvement,
                       entity_features, search_decision, train_singleton_model)
from .error_analysis import KEY_FEATURES, run_error_analysis
from .features import (FeatureContext, FeatureMatrix, build_feature_matrix, compute_features, predict_in_blocks,
                       remove_feature_matrix)
from .graph_cleanup import graph_keep_mask, normalize_mode
from .inference import TrainedPipeline, decide
from .labels import attach_labels, build_label_index, missed_positive_pairs, tag_negative_types
from .metrics import pairwise_prf, summarize
from .preprocess import preprocess_dataset
from .scoring import Blender, cascade_mask
from .splits import make_splits, split_summary
from .train_lightgbm import train_lightgbm
from .utils import (ensure_dir, get_logger, load_json, load_pickle, mem_str, release_memory, save_json, save_pickle,
                    set_seed, setup_logging, timer)

FIT_SPLITS = ("fit", "fit_val")
STAGES = ["prepare", "biencoder", "blocking", "features", "lightgbm", "crossencoder", "decide"]
NEG_TYPE_COLUMNS = ["addr_char_cos", "hn_eq", "pc_eq", "name_jw", "name_char_cos"]


# ------------------------------------------------------------ bookkeeping
def train_dir(cfg: dict) -> str:
    return os.path.join(output_dir(cfg), "train")


def ckpt(cfg: dict, name: str) -> str:
    return os.path.join(train_dir(cfg), "checkpoints", name)


def _start(cfg: dict, stage: str):
    ensure_dir(model_dir(cfg))
    ensure_dir(os.path.join(train_dir(cfg), "checkpoints"))
    log = setup_logging(os.path.join(train_dir(cfg), "train.log"))
    set_seed(int(cfg.get("seed", 42)))
    log.info("=== TRAIN STAGE %-12s mode=%s run=%s  [%s]", stage, cfg.get("mode"), cfg.get("run_name"), mem_str())
    return log


def _report_path(cfg: dict) -> str:
    return os.path.join(train_dir(cfg), "training_report.json")


def load_report(cfg: dict) -> dict:
    path = _report_path(cfg)
    return load_json(path) if os.path.exists(path) else {}


def _update_report(cfg: dict, **sections) -> dict:
    rep = load_report(cfg)
    rep.update(sections)
    save_json(rep, _report_path(cfg))
    return rep


def _record_time(cfg: dict, stage: str, seconds: float, extra: dict | None = None):
    rep = load_report(cfg)
    t = rep.get("timings_sec", {})
    t[stage] = round(seconds, 1)
    t.update(extra or {})
    t["total"] = round(sum(val for k, val in t.items() if k in STAGES), 1)
    rep["timings_sec"] = t
    save_json(rep, _report_path(cfg))


def stage_done(cfg: dict, stage: str) -> bool:
    return os.path.exists(ckpt(cfg, f"{stage}.done"))


def _mark_done(cfg: dict, stage: str):
    with open(ckpt(cfg, f"{stage}.done"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S"))


def _invalidate_from(cfg: dict, stage: str):
    """Re-running a stage makes every later stage stale."""
    for st in STAGES[STAGES.index(stage):]:
        path = ckpt(cfg, f"{st}.done")
        if os.path.exists(path):
            os.remove(path)


def _require(cfg: dict, *stages: str):
    for st in stages:
        if not stage_done(cfg, st):
            raise RuntimeError(f"stage '{st}' has not been run for run_name={cfg.get('run_name')!r} "
                               f"(expected {ckpt(cfg, st + '.done')}). Run it first.")


def _load_prepare(cfg: dict) -> dict:
    return load_pickle(ckpt(cfg, "prepare.pkl"))


def _entity_metrics(scored: pd.DataFrame, selected: np.ndarray, entity_idx: np.ndarray, n_true_all: np.ndarray,
                    beta: float) -> dict:
    ents = pd.Index(entity_idx)
    pos = pd.Series(np.arange(len(ents)), index=ents)
    e = pos.reindex(scored["s1_idx"].to_numpy()).to_numpy()
    tp = np.bincount(e[selected & scored["is_true"].to_numpy(bool)], minlength=len(ents))
    npred = np.bincount(e[selected], minlength=len(ents))
    out = summarize(tp, npred, n_true_all[entity_idx], beta)
    out["pairwise_on_candidates"] = pairwise_prf(selected, scored["is_true"].to_numpy(bool), beta)
    return out


# ------------------------------------------------------------------ stages
def stage_prepare(cfg: dict) -> dict:
    log = _start(cfg, "prepare")
    t0 = time.time()
    save_json({"mode": cfg.get("mode"), "run_name": cfg.get("run_name"), "config": cfg.get("_config_path")},
              _report_path(cfg))
    with open(os.path.join(model_dir(cfg), "config_snapshot.yaml"), "w") as fh:
        yaml.safe_dump({k: val for k, val in cfg.items() if not k.startswith("_")}, fh, sort_keys=False)
    with timer("load training data"):
        sources, load_stats = load_sources(cfg, "train")
        labels_df, label_file_stats = load_labels(get(cfg, "data.train.labels"), cfg)
    with timer("normalise + parse addresses"):
        s1, v, prep_stats = preprocess_dataset(sources, cfg)
    del sources
    li = build_label_index(labels_df, s1, v, cfg)
    split = make_splits(li, v["source"], get(cfg, "splits.fractions", {"fit": .7, "calib": .15, "holdout": .15}),
                        float(get(cfg, "splits.inner_val_fraction", 0.15)), int(cfg.get("seed", 42)))
    save_pickle({"s1": s1, "v": v, "li": li, "split": split}, ckpt(cfg, "prepare.pkl"))
    splits = split_summary(split, li)
    log.info("Splits: %s", splits)
    _update_report(cfg, data={"load": load_stats, "label_file": label_file_stats, "preprocess": prep_stats},
                   labels=li.stats, splits=splits)
    _record_time(cfg, "prepare", time.time() - t0)
    return {"source1": len(s1), "vendors": len(v), "splits": splits}


def stage_biencoder(cfg: dict) -> dict:
    log = _start(cfg, "biencoder")
    if not semantic_enabled(cfg):
        log.info("semantic blocking disabled (mode=%s) - bi-encoder stage skipped", cfg.get("mode"))
        return {"skipped": True}
    _require(cfg, "prepare")
    from .train_biencoder import train_biencoder
    t0 = time.time()
    prep = _load_prepare(cfg)
    li, split = prep["li"], prep["split"]
    fit_mask = np.isin(split[li.true_pairs["s1_idx"].to_numpy()], FIT_SPLITS)
    with timer("train SupCon bi-encoder"):
        rep = train_biencoder(prep["s1"], prep["v"], li.true_pairs[fit_mask], split, cfg,
                              os.path.join(model_dir(cfg), "biencoder"), int(cfg.get("seed", 42)))
    _update_report(cfg, biencoder=rep)
    _record_time(cfg, "biencoder", time.time() - t0)
    return rep


def stage_blocking(cfg: dict) -> dict:
    log = _start(cfg, "blocking")
    _require(cfg, "prepare")
    t0 = time.time()
    prep = _load_prepare(cfg)
    s1, v, li, split = prep["s1"], prep["v"], prep["li"], prep["split"]
    retriever = None
    if semantic_enabled(cfg):
        from .train_biencoder import SemanticRetriever
        _require(cfg, "biencoder")
        retriever = SemanticRetriever(os.path.join(model_dir(cfg), "biencoder"), cfg)
    with timer("blocking"):
        cand, precap, bstats = Blocker(cfg).generate(s1, v, retriever)
    if retriever is not None:
        retriever.save_indices(os.path.join(model_dir(cfg), "biencoder_index_train"))
        del retriever
    brep = blocking_report(cand, precap, s1, v, li, split, float(get(cfg, "decision.beta", 0.5)))
    del precap
    brep["generation"] = bstats
    save_json(brep, os.path.join(train_dir(cfg), "blocking_report.json"))
    save_pickle(cand, ckpt(cfg, "candidates.pkl"))
    summary = {k: brep[k] for k in ("candidate_pairs", "reduction_ratio", "pair_completeness",
                                    "pair_completeness_s2", "pair_completeness_s3", "true_pairs_lost",
                                    "oracle_macro_f0.5_given_blocking", "pair_completeness_before_cap") if k in brep}
    log.info("Blocking: %d candidates, reduction ratio %.6f, pair completeness %.4f (before cap %.4f), "
             "%d true pairs lost, oracle Macro F0.5 ceiling %.4f", brep["candidate_pairs"], brep["reduction_ratio"],
             brep.get("pair_completeness", float("nan")), brep.get("pair_completeness_before_cap", float("nan")),
             brep.get("true_pairs_lost", 0), brep.get("oracle_macro_f0.5_given_blocking", float("nan")))
    _update_report(cfg, blocking=summary)
    _record_time(cfg, "blocking", time.time() - t0)
    return summary


def stage_features(cfg: dict) -> dict:
    log = _start(cfg, "features")
    _require(cfg, "prepare", "blocking")
    t0 = time.time()
    prep = _load_prepare(cfg)
    s1, v, li, split = prep["s1"], prep["v"], prep["li"], prep["split"]
    cand = load_pickle(ckpt(cfg, "candidates.pkl"))
    with timer("pairwise features (streamed to disk)"):
        ctx = FeatureContext(s1, v, cfg)
        if is_full(cfg):
            save_pickle({**ctx.addr_idf, **ctx.name_idf}, os.path.join(model_dir(cfg), "serializer_idf.pkl"))
        fm = build_feature_matrix(cand, s1, v, ctx, cfg, ckpt(cfg, "features.npy"))
    feat_cols = list(fm.columns)
    cand["label"] = attach_labels(cand, li, v).to_numpy()
    cand["is_true"] = cand["label"] != 0          # id is in the entity's label list
    cand["split"] = pd.Categorical(split[cand["s1_idx"].to_numpy()])
    neg = (cand["label"] == 0).to_numpy()
    # report statistic only -> a sample, so the whole matrix is not read back
    neg_rows = np.flatnonzero(neg)
    if len(neg_rows) > 500_000:
        neg_rows = np.sort(np.random.default_rng(0).choice(neg_rows, 500_000, replace=False))
    neg_types = tag_negative_types(fm.frame(neg_rows, NEG_TYPE_COLUMNS))
    summary = {
        "candidate_pairs": int(len(cand)), "positives": int((cand["label"] == 1).sum()),
        "negatives": int(neg.sum()), "uncertain_excluded": int((cand["label"] == -1).sum()),
        "positive_rate": float((cand["label"] == 1).mean()) if len(cand) else 0.0,
        "hard_negative_types(sample)": neg_types.value_counts().to_dict(),
    }
    log.info("Pairs: %s", summary)
    extra = None
    if get(cfg, "training.add_missed_positives", True):
        mp = missed_positive_pairs(cand, li)
        mp = mp[np.isin(split[mp["s1_idx"].to_numpy()], FIT_SPLITS)].copy()
        if len(mp):
            mp["source"] = v["source"].to_numpy()[mp["v_idx"].to_numpy()]
            mp_feats = compute_features(mp, s1, v, ctx, {**cfg, "features": {**cfg.get("features", {}),
                                                                          "context_features": False}})
            for c in feat_cols:
                if c not in mp_feats:
                    mp_feats[c] = np.float32(np.nan)   # context of an unretrieved pair is undefined
            extra = {"X": mp_feats[feat_cols].astype(np.float32), "split": split[mp["s1_idx"].to_numpy()]}
            log.info("Added %d true pairs missed by blocking as extra fit positives", len(mp))
    del ctx, fm
    save_pickle(cand, ckpt(cfg, "pairs.pkl"))
    save_pickle({"feature_columns": feat_cols, "missed_positives": extra}, ckpt(cfg, "features_meta.pkl"))
    _update_report(cfg, training_pairs=summary)
    _record_time(cfg, "features", time.time() - t0)
    return summary


def stage_lightgbm(cfg: dict) -> dict:
    log = _start(cfg, "lightgbm")
    _require(cfg, "features")
    t0 = time.time()
    seed = int(cfg.get("seed", 42))
    mdir = model_dir(cfg)
    pairs = load_pickle(ckpt(cfg, "pairs.pkl"))
    meta = load_pickle(ckpt(cfg, "features_meta.pkl"))
    feat_cols = meta["feature_columns"]
    fm = FeatureMatrix(ckpt(cfg, "features.npy"), feat_cols)
    label = pairs["label"].to_numpy()
    sp = pairs["split"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    ds = float(get(cfg, "training.neg_downsample", 1.0))
    usable = label >= 0
    if ds < 1.0:
        usable &= (label == 1) | (rng.random(len(pairs)) < ds)
    tr = np.flatnonzero(usable & (sp == "fit"))
    va = np.flatnonzero(usable & (sp == "fit_val"))
    extra = meta.get("missed_positives")
    m_fit = (extra["split"] == "fit") if extra is not None else np.zeros(0, dtype=bool)
    n_extra = int(m_fit.sum())
    # one pre-allocated training matrix (no concat copy of the large block)
    X = np.empty((len(tr) + n_extra, len(feat_cols)), dtype=np.float32)
    fm.fill(tr, feat_cols, X[:len(tr)])
    if n_extra:
        X[len(tr):] = extra["X"][m_fit][feat_cols].to_numpy(np.float32)
    X_tr = pd.DataFrame(X, columns=feat_cols, copy=False)
    y_tr = np.r_[label[tr].astype(int), np.ones(n_extra, dtype=int)]
    X_va, y_va = fm.frame(va, feat_cols), label[va].astype(int)
    log.info("LightGBM training matrix %s (neg_downsample=%.2f)  [%s]", X_tr.shape, ds, mem_str())
    with timer("train LightGBM"):
        booster, lgb_report = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg, mdir, seed)
    del X_tr, X_va, X
    release_memory()
    with timer("LightGBM scoring of all candidate pairs (blocks)"):
        p_raw = predict_in_blocks(booster, fm, feat_cols, int(get(cfg, "features.predict_block", 500_000)))
    calib_m = (sp == "calib") & (label >= 0)
    cal_lgb, cal_rep = fit_calibrator(p_raw[calib_m], label[calib_m], get(cfg, "calibration.method", "isotonic"), seed)
    save_pickle(cal_lgb, os.path.join(mdir, "calibrator_lgb.pkl"))
    scores = pd.DataFrame({"p_lgb_raw": p_raw, "p_lgb": cal_lgb.transform(p_raw)}, index=pairs.index)
    save_pickle(scores, ckpt(cfg, "scores_lgb.pkl"))
    _update_report(cfg, lightgbm=lgb_report, calibration_lgb=cal_rep)
    _record_time(cfg, "lightgbm", time.time() - t0)
    return {"best_iteration": lgb_report.get("best_iteration"), "val": lgb_report.get("val")}


def stage_crossencoder(cfg: dict) -> dict:
    log = _start(cfg, "crossencoder")
    if not is_full(cfg):
        log.info("baseline mode - cross-encoder stage skipped")
        return {"skipped": True}
    _require(cfg, "prepare", "features", "lightgbm")
    from .train_crossencoder import CrossEncoderPredictor, DittoSerializer, train_crossencoder
    t0 = time.time()
    seed = int(cfg.get("seed", 42))
    mdir = model_dir(cfg)
    prep = _load_prepare(cfg)
    s1, v, li, split = prep["s1"], prep["v"], prep["li"], prep["split"]
    pairs = load_pickle(ckpt(cfg, "pairs.pkl"))
    cand = pd.concat([pairs[["s1_idx", "v_idx", "source", "label", "split"]],
                      load_pickle(ckpt(cfg, "scores_lgb.pkl"))], axis=1)
    del pairs
    cand["split"] = cand["split"].astype(str)
    ser = DittoSerializer(cfg, load_pickle(os.path.join(mdir, "serializer_idf.pkl")))
    t1, tv = ser.serialize_frame(s1), ser.serialize_frame(v)
    ccfg = cfg.get("crossencoder", {})
    tr_df = _crossencoder_training_pairs(cand, ccfg, seed,
                                         li if get(cfg, "training.add_missed_positives", True) else None, split)
    mask_all = cascade_mask(cand, cand["p_lgb"].to_numpy(), cfg.get("cascade", {}))
    va_df = cand[mask_all & (cand["split"] == "fit_val").to_numpy() & (cand["label"].to_numpy() >= 0)]
    with timer("train cross-encoder"):
        ce_rep = train_crossencoder(
            [t1[i] for i in tr_df["s1_idx"]], [tv[j] for j in tr_df["v_idx"]], tr_df["label"].to_numpy(),
            [t1[i] for i in va_df["s1_idx"]], [tv[j] for j in va_df["v_idx"]], va_df["label"].to_numpy(),
            cfg, os.path.join(mdir, "crossencoder"), seed)
    predictor = CrossEncoderPredictor(os.path.join(mdir, "crossencoder"), cfg)
    eval_m = mask_all & cand["split"].isin(["calib", "holdout"]).to_numpy()
    out = pd.DataFrame({"sent_to_ce": eval_m, "p_ce_raw": np.nan, "p_ce": np.nan}, index=cand.index)
    with timer("cross-encoder scoring of calib/holdout cascade pairs"):
        p = predictor.predict([t1[i] for i in cand.loc[eval_m, "s1_idx"]], [tv[j] for j in cand.loc[eval_m, "v_idx"]])
    out.loc[eval_m, "p_ce_raw"] = p
    calib_m = (cand["split"] == "calib").to_numpy() & (cand["label"].to_numpy() >= 0)
    cm = eval_m & calib_m
    cal_ce, cal_ce_rep = fit_calibrator(out.loc[cm, "p_ce_raw"], cand.loc[cm, "label"],
                                        get(cfg, "calibration.method", "isotonic"), seed)
    out.loc[eval_m, "p_ce"] = cal_ce.transform(out.loc[eval_m, "p_ce_raw"].to_numpy())
    save_pickle(cal_ce, os.path.join(mdir, "calibrator_ce.pkl"))
    save_pickle(out, ckpt(cfg, "scores_ce.pkl"))
    n_eval = max(1, int(cand["split"].isin(["calib", "holdout"]).sum()))
    cascade = {"pairs_sent_calib_holdout": int(eval_m.sum()), "share_of_calib_holdout_pairs": float(eval_m.sum() / n_eval)}
    _update_report(cfg, crossencoder=ce_rep, calibration_ce=cal_ce_rep, cascade=cascade)
    _record_time(cfg, "crossencoder", time.time() - t0)
    return cascade


def stage_decide(cfg: dict) -> dict:
    log = _start(cfg, "decide")
    _require(cfg, "prepare", "features", "lightgbm")
    t0 = time.time()
    seed = int(cfg.get("seed", 42))
    beta = float(get(cfg, "decision.beta", 0.5))
    mdir, odir = model_dir(cfg), train_dir(cfg)
    prep = _load_prepare(cfg)
    s1, v, li, split = prep["s1"], prep["v"], prep["li"], prep["split"]
    pairs = load_pickle(ckpt(cfg, "pairs.pkl"))
    feat_cols = load_pickle(ckpt(cfg, "features_meta.pkl"))["feature_columns"]
    fm = FeatureMatrix(ckpt(cfg, "features.npy"), feat_cols)
    has_ce = is_full(cfg)
    if has_ce:
        _require(cfg, "crossencoder")
        ce = load_pickle(ckpt(cfg, "scores_ce.pkl"))
    else:
        ce = pd.DataFrame({"sent_to_ce": False, "p_ce_raw": np.nan, "p_ce": np.nan}, index=pairs.index)
    cand = pd.concat([pairs[["s1_idx", "v_idx", "source", "label", "is_true", "split", "n_methods"]],
                      load_pickle(ckpt(cfg, "scores_lgb.pkl")), ce], axis=1)
    del pairs, ce
    cand["split"] = cand["split"].astype(str)
    n_true = li.n_true
    dcfg = cfg.get("decision", {})
    eval_cols = [c for c in dict.fromkeys([*KEY_FEATURES, *ENTITY_INPUT_COLUMNS]) if c in fm.col_idx]

    # ------------------------------------------- blend + decision on calib
    calib_ents = np.where((split == "calib") & li.labeled)[0]
    calib_rows = np.flatnonzero((cand["split"] == "calib").to_numpy())
    cdf = cand.iloc[calib_rows].copy()
    cfeats = fm.frame(calib_rows, [c for c in ENTITY_INPUT_COLUMNS if c in fm.col_idx], index=cdf.index)
    blender, cal_final, blend_trace = _select_blend(cdf, calib_ents, n_true, cfg, has_ce, seed, beta)
    cdf["p_blend"] = blender.transform(cdf["p_lgb"], cdf["p_ce"], cdf["p_ce"].notna())
    cdf["p_final"] = cal_final.transform(cdf["p_blend"].to_numpy())

    gcfg = cfg.get("graph", {})
    graph_modes = [normalize_mode(m) for m in gcfg.get("modes_to_try", ["off"])]
    masks, gstats = {}, {}
    for mode in graph_modes:
        masks[mode], gstats[mode] = graph_keep_mask(cdf, "p_final", mode, float(gcfg.get("min_edge_prob", 0.2)),
                                                    int(gcfg.get("max_component_size", 50)))

    matchable, singleton_model, sm_rep = None, None, None
    if get(cfg, "singleton_model.enabled", True):
        ef = entity_features(pd.concat([cdf, cfeats], axis=1), calib_ents)
        y_match = np.zeros(len(calib_ents), dtype=int)
        y_match[np.isin(calib_ents, cdf.loc[cdf["is_true"], "s1_idx"].unique())] = 1
        singleton_model, oof, sm_rep = train_singleton_model(ef, y_match, int(get(cfg, "singleton_model.folds", 5)), seed)
        if singleton_model is not None:
            matchable = oof
            singleton_model.save_model(os.path.join(mdir, "singleton_model.txt"))

    table = DecisionTable.build(cdf, "p_final", calib_ents, n_true[calib_ents], masks, matchable, beta)
    if dcfg.get("search", True):
        with timer("Macro-F0.5 decision search (calib)"):
            params, calib_score, trace = search_decision(table, dcfg, graph_modes)
        pd.DataFrame(trace, columns=["round", "param", "value", "macro_f0.5"]).to_csv(
            os.path.join(odir, "decision_search_trace.tsv"), sep="\t", index=False)
    else:
        params = DecisionParams.from_dict(dcfg.get("default", {}))
        calib_score = table.score(params)
    del table, cdf, cfeats

    # save pipeline so that holdout is scored through the exact inference path
    save_pickle(blender, os.path.join(mdir, "blender.pkl"))
    save_pickle(cal_final, os.path.join(mdir, "calibrator_final.pkl"))
    cal_lgb = load_pickle(os.path.join(mdir, "calibrator_lgb.pkl"))
    cal_ce = load_pickle(os.path.join(mdir, "calibrator_ce.pkl")) if has_ce else None
    meta = {
        "mode": cfg.get("mode"), "feature_columns": feat_cols, "decision": params.to_dict(),
        "has_ce": has_ce, "has_singleton_model": singleton_model is not None,
        "has_semantic_blocking": semantic_enabled(cfg), "graph": gcfg, "cascade": cfg.get("cascade", {}),
        "blend": blender.describe(), "calibration": {"lgb": cal_lgb.method, "ce": cal_ce.method if cal_ce else None,
                                                     "final": cal_final.method},
        "label_key_mode": li.key_mode, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
    }
    save_json(meta, os.path.join(mdir, "pipeline.json"))
    pipe = TrainedPipeline(mdir, cfg)

    # ------------------------------------------------ holdout evaluation
    evaluation, scored_all = {}, []
    for name in ("calib", "holdout"):
        ents = np.where((split == name) & li.labeled)[0]
        rows = np.flatnonzero((cand["split"] == name).to_numpy())
        if len(ents) == 0:
            continue
        sc = cand.iloc[rows][["s1_idx", "v_idx", "source", "label", "is_true", "p_lgb_raw", "p_lgb", "sent_to_ce",
                              "p_ce_raw", "p_ce"]].copy()
        sc["p_blend"] = pipe.blender.transform(sc["p_lgb"], sc["p_ce"], sc["p_ce"].notna())
        sc["p_final"] = pipe.cal_final.transform(sc["p_blend"].to_numpy())
        extra = fm.frame(rows, eval_cols, index=sc.index)
        sel, info = decide(sc, ents, pipe, extra_cols=extra)
        sc["selected"] = sel
        m = _entity_metrics(sc, sel, ents, n_true, beta)
        m["graph"] = info["graph"]
        evaluation[name] = m
        sc = pd.concat([sc, extra, cand.iloc[rows][["n_methods", "split"]]], axis=1)
        scored_all.append(sc)
        if name == "holdout":
            with timer("error analysis (holdout)"):
                evaluation["holdout_errors"] = run_error_analysis(
                    sc, s1, v, li.true_pairs, n_true, ents, min(params.threshold_s2, params.threshold_s3),
                    os.path.join(odir, "error_analysis_holdout"), int(get(cfg, "evaluation.error_analysis_top_n", 200)))
    evaluation["ablation"] = _ablation(cand, split, li, n_true, dcfg, beta, has_ce)
    if "holdout" in evaluation:
        h = evaluation["holdout"]
        log.info("HOLDOUT  Macro F0.5 = %.5f | pair P = %.4f R = %.4f | singletons: %d true, %d false merges | "
                 "predicted singletons %d", h["macro_f0.5"], h["pair_precision"], h["pair_recall"],
                 h["true_singletons"], h["singleton_false_merges"], h["predicted_singletons"])
    save_pickle({"scored": pd.concat(scored_all) if scored_all else pd.DataFrame(), "split": split,
                 "n_true": n_true, "true_pairs": li.true_pairs, "true_keys": li.true_keys,
                 "labeled": li.labeled, "key_mode": li.key_mode, "s1": s1, "v": v},
                os.path.join(odir, "eval_state.pkl"))
    _update_report(cfg, blend={"chosen": blender.describe(), "calibration": cal_final.method, "trace": blend_trace},
                   graph_calib=gstats, singleton_model=sm_rep,
                   decision={"params": params.to_dict(), "calib_macro_f0.5(optimistic, tuned here)": calib_score},
                   evaluation=evaluation)
    _record_time(cfg, "decide", time.time() - t0)
    log.info("Artefacts: %s | reports: %s  [%s]", mdir, odir, mem_str())
    return evaluation.get("holdout", {})


STAGE_FUNCS = {"prepare": stage_prepare, "biencoder": stage_biencoder, "blocking": stage_blocking,
               "features": stage_features, "lightgbm": stage_lightgbm, "crossencoder": stage_crossencoder,
               "decide": stage_decide}


def run_stage(cfg: dict, stage: str) -> dict:
    """Run ONE stage (what each notebook cell calls)."""
    if stage not in STAGE_FUNCS:
        raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
    noop = (stage == "biencoder" and not semantic_enabled(cfg)) or (stage == "crossencoder" and not is_full(cfg))
    if not noop:   # a stage that does nothing in this mode must not invalidate later ones
        _invalidate_from(cfg, stage)
    out = STAGE_FUNCS[stage](cfg)
    _mark_done(cfg, stage)
    release_memory()
    get_logger().info("stage %s finished  [%s]", stage, mem_str())
    return out


def run_training(cfg: dict, stages: list[str] | None = None, resume: bool = False) -> dict:
    """Run the stages in order. resume=True skips stages already completed."""
    ensure_dir(os.path.join(train_dir(cfg), "checkpoints"))
    setup_logging(os.path.join(train_dir(cfg), "train.log"))
    for stage in stages or STAGES:
        if resume and stage_done(cfg, stage):
            get_logger().info("stage %s already done - skipped (resume)", stage)
            continue
        run_stage(cfg, stage)
    if get(cfg, "features.keep_matrix", True) is False and stage_done(cfg, "decide"):
        remove_feature_matrix(ckpt(cfg, "features.npy"))
    return load_report(cfg)


def _crossencoder_training_pairs(cand: pd.DataFrame, ccfg: dict, seed: int, li, split) -> pd.DataFrame:
    """Positives + LightGBM-hard negatives + random negatives of fit entities."""
    fit = cand[(cand["split"] == "fit") & (cand["label"] >= 0)]
    pos = fit[fit["label"] == 1][["s1_idx", "v_idx", "label"]]
    neg = fit[fit["label"] == 0].sort_values(["s1_idx", "p_lgb_raw"], ascending=[True, False])
    hard = neg.groupby("s1_idx").head(int(ccfg.get("hard_negatives_per_entity", 8)))
    rest = neg.drop(hard.index)
    rnd = rest.sample(frac=1.0, random_state=seed).groupby("s1_idx").head(int(ccfg.get("random_negatives_per_entity", 2)))
    parts = [pos, hard[["s1_idx", "v_idx", "label"]], rnd[["s1_idx", "v_idx", "label"]]]
    if li is not None:
        mp = missed_positive_pairs(cand, li)
        mp = mp[split[mp["s1_idx"].to_numpy()] == "fit"]
        parts.append(mp.assign(label=1))
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_idx", "v_idx"])
    cap = int(ccfg.get("max_train_pairs", 400000))
    if len(out) > cap:
        pos_part = out[out["label"] == 1]
        neg_part = out[out["label"] == 0].sample(n=max(0, cap - len(pos_part)), random_state=seed)
        out = pd.concat([pos_part, neg_part])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _select_blend(cdf, calib_ents, n_true, cfg, has_ce, seed, beta):
    """Choose blend method/weight + final calibration by calib Macro F0.5."""
    log = get_logger()
    bcfg = cfg.get("blend", {})
    cal_method = get(cfg, "calibration.method", "isotonic")
    dcfg = cfg.get("decision", {})
    y = cdf["label"].clip(lower=0).to_numpy()
    has = cdf["p_ce"].notna().to_numpy()
    if not has_ce or not has.any():
        return Blender("weighted", 1.0), Calibrator("none"), []
    # A blend fitted on a handful of cascaded pairs is noise-fitting: require
    # enough cross-encoder-scored calib pairs of BOTH classes, else keep LightGBM.
    n_pos, n_neg = int((y[has] == 1).sum()), int((y[has] == 0).sum())
    min_pairs, min_cls = int(bcfg.get("min_ce_pairs", 200)), int(bcfg.get("min_ce_per_class", 20))
    if has.sum() < min_pairs or min(n_pos, n_neg) < min_cls:
        log.warning("Only %d cascaded calib pairs (%d pos / %d neg) < blend.min_ce_pairs=%d or "
                    "min_ce_per_class=%d -> keeping LightGBM alone", int(has.sum()), n_pos, n_neg, min_pairs, min_cls)
        return Blender("weighted", 1.0), Calibrator("none"), [{"skipped": "too few cross-encoder calib pairs",
                                                               "pairs": int(has.sum()), "pos": n_pos, "neg": n_neg}]
    methods = ["weighted", "rank", "stack"] if bcfg.get("method", "auto") == "auto" else [bcfg["method"]]
    min_imp = effective_min_improvement(dcfg, len(calib_ents))
    trace, best = [], None
    # LightGBM alone is the reference; a blend must beat it by min_improvement
    candidates = [("weighted", 1.0)] + [(m, float(w)) for m in methods
                                        for w in (bcfg.get("weight_grid", [0.5]) if m != "stack" else [1.0])
                                        if not (m == "weighted" and float(w) == 1.0)]
    for method, w in candidates:
        bl = Blender(method, w).fit(cdf["p_lgb"], cdf["p_ce"], has, y)
        pb = bl.transform(cdf["p_lgb"], cdf["p_ce"], has)
        lgb_only = method == "weighted" and w == 1.0     # p_lgb is already calibrated
        if get(cfg, "calibration.calibrate_blend", True) and not lgb_only:
            cal = Calibrator(cal_method if cal_method != "auto" else "isotonic").fit(pb, y)
        else:
            cal = Calibrator("none")
        tmp = cdf.assign(p_final=cal.transform(pb))
        table = DecisionTable.build(tmp, "p_final", calib_ents, n_true[calib_ents], beta=beta)
        _, score, _ = search_decision(table, dcfg, quick=True)
        trace.append({"method": method, "weight_lgb": w, "calib_macro_f0.5_quick": score})
        if best is None or score > best[0] + min_imp:
            best = (score, bl, cal)
    log.info("Blend selection: %s", best[1].describe())
    return best[1], best[2], trace


def _ablation(cand, split, li, n_true, dcfg, beta, has_ce) -> dict:
    """Holdout Macro F0.5 of simpler scorers, each thresholded on calib."""
    out = {}
    cols = ["p_lgb"] + (["p_ce_or_lgb"] if has_ce else [])
    ce = np.where((split == "calib") & li.labeled)[0]
    he = np.where((split == "holdout") & li.labeled)[0]
    if len(ce) == 0 or len(he) == 0:
        return out
    keep = ["s1_idx", "v_idx", "source", "label", "is_true", "p_lgb", "p_ce"]
    c = cand.loc[(cand["split"] == "calib").to_numpy(), keep].copy()
    h = cand.loc[(cand["split"] == "holdout").to_numpy(), keep].copy()
    for df in (c, h):
        df["p_ce_or_lgb"] = df["p_ce"].fillna(df["p_lgb"])
    for col in cols:
        tc = DecisionTable.build(c.assign(p=c[col]), "p", ce, n_true[ce], beta=beta)
        prm, s_c, _ = search_decision(tc, dcfg, quick=True)
        th = DecisionTable.build(h.assign(p=h[col]), "p", he, n_true[he], beta=beta)
        out[col] = {"threshold": prm.threshold_s2, "calib_macro_f0.5": s_c, "holdout_macro_f0.5": th.score(prm)}
    return out
