"""Training pipeline (python train.py --config config.yaml).

  1  ingest TSVs, validate schema, parse labels
  2  normalise names / parse addresses (offline)
  3  resolve labels, entity-level fit / fit_val / calib / holdout split
  4  [full] train SupCon bi-encoder on fit entities (early stop on fit_val)
  5  blocking over ALL training Source-1 entities + blocking report
  6  pairwise features, labels, hard-negative tags
  7  LightGBM on fit pairs (early stop on fit_val pairs)
  8  isotonic calibration of LightGBM on calib
  9  [full] cross-encoder on fit pairs (hard negatives, focal loss, MixDA),
     cascade scoring of calib / holdout, isotonic calibration on calib
 10  blend selection + final calibration + singleton model + graph modes
     + Macro-F0.5 decision search, all on calib
 11  unbiased evaluation on holdout + error analysis
 12  save artefacts (models/<run_name>/) and reports (outputs/<run_name>/train/)
"""
from __future__ import annotations

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
from .decision import (DecisionParams, DecisionTable, effective_min_improvement, entity_features,
                       search_decision, train_singleton_model)
from .error_analysis import run_error_analysis
from .features import FeatureContext, compute_features, feature_columns
from .graph_cleanup import graph_keep_mask, normalize_mode
from .inference import TrainedPipeline, decide, score_candidates
from .labels import attach_labels, build_label_index, missed_positive_pairs, tag_negative_types
from .metrics import pairwise_prf, summarize
from .preprocess import preprocess_dataset
from .scoring import Blender, cascade_mask
from .splits import make_splits, split_summary
from .train_lightgbm import predict_lightgbm, train_lightgbm
from .utils import (ensure_dir, get_logger, save_json, save_pickle, set_seed, setup_logging, timer)

FIT_SPLITS = ("fit", "fit_val")


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


def run_training(cfg: dict) -> dict:
    t_start = time.time()
    mdir, odir = model_dir(cfg), os.path.join(output_dir(cfg), "train")
    ensure_dir(mdir)
    ensure_dir(odir)
    log = setup_logging(os.path.join(odir, "train.log"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    beta = float(get(cfg, "decision.beta", 0.5))
    timings: dict = {}
    report: dict = {"mode": cfg.get("mode"), "run_name": cfg.get("run_name"), "config": cfg.get("_config_path")}
    log.info("=== TRAINING  mode=%s  run=%s ===", cfg.get("mode"), cfg.get("run_name"))
    with open(os.path.join(mdir, "config_snapshot.yaml"), "w") as fh:
        yaml.safe_dump({k: val for k, val in cfg.items() if not k.startswith("_")}, fh, sort_keys=False)

    # 1-2 ------------------------------------------------------------ data
    with timer("load training data", timings):
        sources, load_stats = load_sources(cfg, "train")
        labels_df, label_file_stats = load_labels(get(cfg, "data.train.labels"), cfg)
    with timer("normalise + parse addresses", timings):
        s1, v, prep_stats = preprocess_dataset(sources, cfg)
    report["data"] = {"load": load_stats, "label_file": label_file_stats, "preprocess": prep_stats}

    # 3 --------------------------------------------------------- labels/split
    li = build_label_index(labels_df, s1, v, cfg)
    split = make_splits(li, v["source"], get(cfg, "splits.fractions", {"fit": .7, "calib": .15, "holdout": .15}),
                        float(get(cfg, "splits.inner_val_fraction", 0.15)), seed)
    report["labels"] = li.stats
    report["splits"] = split_summary(split, li)
    log.info("Splits: %s", report["splits"])

    # 4 ------------------------------------------------------ bi-encoder
    retriever = None
    if semantic_enabled(cfg):
        from .train_biencoder import SemanticRetriever, train_biencoder
        with timer("train SupCon bi-encoder", timings):
            fit_mask = np.isin(split[li.true_pairs["s1_idx"].to_numpy()], FIT_SPLITS)
            report["biencoder"] = train_biencoder(s1, v, li.true_pairs[fit_mask], split, cfg,
                                                  os.path.join(mdir, "biencoder"), seed)
        retriever = SemanticRetriever(os.path.join(mdir, "biencoder"), cfg)

    # 5 -------------------------------------------------------- blocking
    with timer("blocking", timings):
        cand, precap, bstats = Blocker(cfg).generate(s1, v, retriever)
    if retriever is not None:
        retriever.save_indices(os.path.join(mdir, "biencoder_index_train"))
    brep = blocking_report(cand, precap, s1, v, li, split, beta)
    brep["generation"] = bstats
    save_json(brep, os.path.join(odir, "blocking_report.json"))
    report["blocking"] = {k: brep[k] for k in ("candidate_pairs", "reduction_ratio", "pair_completeness",
                                                "pair_completeness_s2", "pair_completeness_s3",
                                                "true_pairs_lost", "oracle_macro_f0.5_given_blocking",
                                                "pair_completeness_before_cap") if k in brep}
    log.info("Blocking: %d candidates, reduction ratio %.6f, pair completeness %.4f (before cap %.4f), "
             "%d true pairs lost, oracle Macro F0.5 ceiling %.4f", brep["candidate_pairs"], brep["reduction_ratio"],
             brep.get("pair_completeness", float("nan")), brep.get("pair_completeness_before_cap", float("nan")),
             brep.get("true_pairs_lost", 0), brep.get("oracle_macro_f0.5_given_blocking", float("nan")))

    # 6 -------------------------------------------------------- features
    with timer("pairwise features", timings):
        ctx = FeatureContext(s1, v, cfg)
        feats = compute_features(cand, s1, v, ctx, cfg)
    cand["label"] = attach_labels(cand, li, v).to_numpy()
    cand["is_true"] = cand["label"] != 0          # id is in the entity's label list
    cand["split"] = split[cand["s1_idx"].to_numpy()]
    feat_cols = feature_columns(feats)
    neg = cand["label"] == 0
    neg_types = tag_negative_types(feats[neg])
    report["training_pairs"] = {
        "candidate_pairs": int(len(cand)), "positives": int((cand["label"] == 1).sum()),
        "negatives": int(neg.sum()), "uncertain_excluded": int((cand["label"] == -1).sum()),
        "positive_rate": float((cand["label"] == 1).mean()),
        "hard_negative_types": neg_types.value_counts().to_dict(),
    }
    log.info("Pairs: %s", report["training_pairs"])

    X_extra = None
    if get(cfg, "training.add_missed_positives", True):
        mp = missed_positive_pairs(cand, li)
        mp = mp[np.isin(split[mp["s1_idx"].to_numpy()], FIT_SPLITS)].copy()
        if len(mp):
            mp["source"] = v["source"].to_numpy()[mp["v_idx"].to_numpy()]
            mp_feats = compute_features(mp, s1, v, ctx, {**cfg, "features": {**cfg.get("features", {}),
                                                                          "context_features": False}})
            for c in feat_cols:
                if c not in mp_feats:
                    mp_feats[c] = np.nan      # context of an unretrieved pair is undefined
            X_extra = (mp_feats[feat_cols], split[mp["s1_idx"].to_numpy()])
            log.info("Added %d true pairs missed by blocking as extra fit positives", len(mp))

    # 7 -------------------------------------------------------- LightGBM
    rng = np.random.default_rng(seed)
    ds = float(get(cfg, "training.neg_downsample", 1.0))
    usable = cand["label"].to_numpy() >= 0
    if ds < 1.0:
        usable &= (cand["label"].to_numpy() == 1) | (rng.random(len(cand)) < ds)
    tr = usable & (cand["split"].to_numpy() == "fit")
    va = usable & (cand["split"].to_numpy() == "fit_val")
    X_tr, y_tr = feats.loc[tr, feat_cols], cand.loc[tr, "label"].to_numpy()
    X_va, y_va = feats.loc[va, feat_cols], cand.loc[va, "label"].to_numpy()
    if X_extra is not None:
        m_fit = X_extra[1] == "fit"
        X_tr = pd.concat([X_tr, X_extra[0][m_fit]], ignore_index=True)
        y_tr = np.r_[y_tr, np.ones(int(m_fit.sum()), dtype=int)]
    with timer("train LightGBM", timings):
        booster, lgb_report = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg, mdir, seed)
    report["lightgbm"] = lgb_report
    cand["p_lgb_raw"] = predict_lightgbm(booster, feats[feat_cols])

    # 8 ------------------------------------------- calibration of LightGBM
    calib_m = (cand["split"] == "calib").to_numpy() & (cand["label"].to_numpy() >= 0)
    cal_method = get(cfg, "calibration.method", "isotonic")
    cal_lgb, cal_lgb_rep = fit_calibrator(cand.loc[calib_m, "p_lgb_raw"], cand.loc[calib_m, "label"], cal_method, seed)
    cand["p_lgb"] = cal_lgb.transform(cand["p_lgb_raw"].to_numpy())
    report["calibration_lgb"] = cal_lgb_rep
    save_pickle(cal_lgb, os.path.join(mdir, "calibrator_lgb.pkl"))

    # 9 --------------------------------------------------- cross-encoder
    has_ce = is_full(cfg)
    cand["sent_to_ce"] = False
    cand["p_ce_raw"] = np.nan
    cand["p_ce"] = np.nan
    cal_ce = None
    if has_ce:
        from .train_crossencoder import CrossEncoderPredictor, DittoSerializer, train_crossencoder
        idf = {**ctx.addr_idf, **ctx.name_idf}
        save_pickle(idf, os.path.join(mdir, "serializer_idf.pkl"))
        ser = DittoSerializer(cfg, idf)
        t1, tv = ser.serialize_frame(s1), ser.serialize_frame(v)
        ccfg = cfg.get("crossencoder", {})
        cascade_cfg = cfg.get("cascade", {})
        tr_df = _crossencoder_training_pairs(cand, ccfg, seed, li if get(cfg, "training.add_missed_positives", True) else None, split)
        mask_all = cascade_mask(cand, cand["p_lgb"].to_numpy(), cascade_cfg)
        va_df = cand[mask_all & (cand["split"] == "fit_val").to_numpy() & (cand["label"].to_numpy() >= 0)]
        with timer("train cross-encoder", timings):
            report["crossencoder"] = train_crossencoder(
                [t1[i] for i in tr_df["s1_idx"]], [tv[j] for j in tr_df["v_idx"]], tr_df["label"].to_numpy(),
                [t1[i] for i in va_df["s1_idx"]], [tv[j] for j in va_df["v_idx"]], va_df["label"].to_numpy(),
                cfg, os.path.join(mdir, "crossencoder"), seed)
        predictor = CrossEncoderPredictor(os.path.join(mdir, "crossencoder"), cfg)
        eval_m = mask_all & cand["split"].isin(["calib", "holdout"]).to_numpy()
        with timer("cross-encoder scoring of calib/holdout cascade pairs", timings):
            p = predictor.predict([t1[i] for i in cand.loc[eval_m, "s1_idx"]], [tv[j] for j in cand.loc[eval_m, "v_idx"]])
        cand.loc[eval_m, "p_ce_raw"] = p
        cand.loc[eval_m, "sent_to_ce"] = True
        cm = eval_m & calib_m
        cal_ce, cal_ce_rep = fit_calibrator(cand.loc[cm, "p_ce_raw"], cand.loc[cm, "label"], cal_method, seed)
        cand.loc[eval_m, "p_ce"] = cal_ce.transform(cand.loc[eval_m, "p_ce_raw"].to_numpy())
        report["calibration_ce"] = cal_ce_rep
        report["cascade"] = {"pairs_sent_calib_holdout": int(eval_m.sum()),
                             "share_of_calib_holdout_pairs": float(eval_m.sum() / max(1, cand["split"].isin(["calib", "holdout"]).sum()))}
        save_pickle(cal_ce, os.path.join(mdir, "calibrator_ce.pkl"))

    # 10 ------------------------------------- blend + decision on calib
    calib_ents = np.where((split == "calib") & li.labeled)[0]
    cdf = cand[cand["split"] == "calib"].copy()
    cfeats = feats.loc[cdf.index]
    n_true = li.n_true
    dcfg = cfg.get("decision", {})
    blender, cal_final, blend_trace = _select_blend(cdf, calib_ents, n_true, cfg, has_ce, seed, beta)
    report["blend"] = {"chosen": blender.describe(), "calibration": cal_final.method, "trace": blend_trace}
    cdf["p_blend"] = blender.transform(cdf["p_lgb"], cdf["p_ce"], cdf["p_ce"].notna())
    cdf["p_final"] = cal_final.transform(cdf["p_blend"].to_numpy())

    gcfg = cfg.get("graph", {})
    graph_modes = [normalize_mode(m) for m in gcfg.get("modes_to_try", ["off"])]
    masks, gstats = {}, {}
    for mode in graph_modes:
        masks[mode], gstats[mode] = graph_keep_mask(cdf, "p_final", mode, float(gcfg.get("min_edge_prob", 0.2)),
                                                    int(gcfg.get("max_component_size", 50)))
    report["graph_calib"] = gstats

    matchable, singleton_model = None, None
    if get(cfg, "singleton_model.enabled", True):
        ef = entity_features(pd.concat([cdf, cfeats], axis=1), calib_ents)
        y_match = np.zeros(len(calib_ents), dtype=int)
        has_true = cdf.loc[cdf["is_true"], "s1_idx"].unique()
        y_match[np.isin(calib_ents, has_true)] = 1
        singleton_model, oof, sm_rep = train_singleton_model(ef, y_match, int(get(cfg, "singleton_model.folds", 5)), seed)
        report["singleton_model"] = sm_rep
        if singleton_model is not None:
            matchable = oof
            singleton_model.save_model(os.path.join(mdir, "singleton_model.txt"))

    table = DecisionTable.build(cdf, "p_final", calib_ents, n_true[calib_ents], masks, matchable, beta)
    if dcfg.get("search", True):
        with timer("Macro-F0.5 decision search (calib)", timings):
            params, calib_score, trace = search_decision(table, dcfg, graph_modes)
        pd.DataFrame(trace, columns=["round", "param", "value", "macro_f0.5"]).to_csv(
            os.path.join(odir, "decision_search_trace.tsv"), sep="\t", index=False)
    else:
        params = DecisionParams.from_dict(dcfg.get("default", {}))
        calib_score = table.score(params)
    report["decision"] = {"params": params.to_dict(), "calib_macro_f0.5(optimistic, tuned here)": calib_score}

    # save pipeline so that holdout is scored through the exact inference path
    save_pickle(blender, os.path.join(mdir, "blender.pkl"))
    save_pickle(cal_final, os.path.join(mdir, "calibrator_final.pkl"))
    meta = {
        "mode": cfg.get("mode"), "feature_columns": feat_cols, "decision": params.to_dict(),
        "has_ce": has_ce, "has_singleton_model": singleton_model is not None,
        "has_semantic_blocking": retriever is not None, "graph": gcfg, "cascade": cfg.get("cascade", {}),
        "blend": blender.describe(), "calibration": {"lgb": cal_lgb.method, "ce": cal_ce.method if cal_ce else None,
                                                     "final": cal_final.method},
        "label_key_mode": li.key_mode, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
    }
    save_json(meta, os.path.join(mdir, "pipeline.json"))
    pipe = TrainedPipeline(mdir, cfg)

    # 11 ----------------------------------------------- holdout evaluation
    evaluation = {}
    scored_all = []
    for name in ("calib", "holdout"):
        ents = np.where((split == name) & li.labeled)[0]
        sub = cand[cand["split"] == name]
        if len(ents) == 0:
            continue
        sc = sub[["s1_idx", "v_idx", "source", "label", "is_true", "p_lgb_raw", "p_lgb", "sent_to_ce",
                  "p_ce_raw", "p_ce"]].copy()
        sc["p_blend"] = pipe.blender.transform(sc["p_lgb"], sc["p_ce"], sc["p_ce"].notna())
        sc["p_final"] = pipe.cal_final.transform(sc["p_blend"].to_numpy())
        sel, info = decide(sc, ents, pipe, extra_cols=feats.loc[sc.index])
        sc["selected"] = sel
        m = _entity_metrics(sc, sel, ents, n_true, beta)
        m["graph"] = info["graph"]
        evaluation[name] = m
        sc = pd.concat([sc, feats.loc[sc.index], cand.loc[sc.index, ["n_methods", "split"]]], axis=1)
        scored_all.append(sc)
        if name == "holdout":
            with timer("error analysis (holdout)", timings):
                evaluation["holdout_errors"] = run_error_analysis(
                    sc, s1, v, li.true_pairs, n_true, ents, min(params.threshold_s2, params.threshold_s3),
                    os.path.join(odir, "error_analysis_holdout"), int(get(cfg, "evaluation.error_analysis_top_n", 200)))
    # simple ablation on holdout: LightGBM alone with its own calib-tuned threshold
    evaluation["ablation"] = _ablation(cand, split, li, n_true, dcfg, beta, has_ce, pipe)
    report["evaluation"] = evaluation
    if "holdout" in evaluation:
        h = evaluation["holdout"]
        log.info("HOLDOUT  Macro F0.5 = %.5f | pair P = %.4f R = %.4f | singletons: %d true, %d false merges | "
                 "predicted singletons %d", h["macro_f0.5"], h["pair_precision"], h["pair_recall"],
                 h["true_singletons"], h["singleton_false_merges"], h["predicted_singletons"])

    # 12 --------------------------------------------------------- save
    save_pickle({"scored": pd.concat(scored_all) if scored_all else pd.DataFrame(), "split": split,
                 "n_true": n_true, "true_pairs": li.true_pairs, "true_keys": li.true_keys,
                 "labeled": li.labeled, "key_mode": li.key_mode,
                 "s1": s1, "v": v}, os.path.join(odir, "eval_state.pkl"))
    timings["total"] = round(time.time() - t_start, 1)
    report["timings_sec"] = timings
    save_json(report, os.path.join(odir, "training_report.json"))
    log.info("Artefacts: %s | reports: %s | total %.1fs", mdir, odir, timings["total"])
    return report


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


def _ablation(cand, split, li, n_true, dcfg, beta, has_ce, pipe) -> dict:
    """Holdout Macro F0.5 of simpler scorers, each thresholded on calib."""
    out = {}
    cols = ["p_lgb"] + (["p_ce_or_lgb"] if has_ce else [])
    df = cand.copy()
    df["p_ce_or_lgb"] = df["p_ce"].fillna(df["p_lgb"])
    for col in cols:
        ce = np.where((split == "calib") & li.labeled)[0]
        he = np.where((split == "holdout") & li.labeled)[0]
        c = df[df["split"] == "calib"]
        h = df[df["split"] == "holdout"]
        if len(ce) == 0 or len(he) == 0:
            continue
        tc = DecisionTable.build(c.assign(p=c[col]), "p", ce, n_true[ce], beta=beta)
        prm, s_c, _ = search_decision(tc, dcfg, quick=True)
        th = DecisionTable.build(h.assign(p=h[col]), "p", he, n_true[he], beta=beta)
        out[col] = {"threshold": prm.threshold_s2, "calib_macro_f0.5": s_c, "holdout_macro_f0.5": th.score(prm)}
    return out
