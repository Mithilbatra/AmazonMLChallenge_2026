"""Structural matcher: LightGBM binary classifier on handcrafted features.

PDF: "LightGBM provides an exceptionally fast and highly accurate baseline
when supplied with rich, handcrafted features" and feature importance by GAIN
lets us audit which name/address components drive decisions - the gain table
is written to feature_importance.tsv.
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .utils import get_logger, save_json


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "positives": int(y.sum())}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        out["logloss"] = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
    return out


def train_lightgbm(X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                   cfg: dict, out_dir: str, seed: int = 42) -> tuple[lgb.Booster, dict]:
    log = get_logger()
    lcfg = cfg.get("lightgbm", {})
    params = dict(lcfg.get("params", {}))
    params.setdefault("objective", "binary")
    params.setdefault("verbose", -1)
    params["seed"] = seed
    params.setdefault("metric", ["binary_logloss", "average_precision"])
    params.setdefault("num_threads", max(1, int(cfg.get("n_jobs", 1) or 1)))
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    valid_sets, valid_names, callbacks = [dtrain], ["train"], [lgb.log_evaluation(100)]
    has_val = X_val is not None and len(X_val) > 0 and len(np.unique(y_val)) == 2
    if has_val:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(int(lcfg.get("early_stopping_rounds", 100)),
                                            first_metric_only=True, verbose=False))
    num_rounds = int(lcfg.get("num_boost_round", 3000)) if has_val else int(lcfg.get("num_boost_round_no_val", 300))
    log.info("LightGBM: %d train rows (%d pos), %d val rows (%d pos), %d features",
             len(X_tr), int(y_tr.sum()), 0 if X_val is None else len(X_val),
             0 if y_val is None else int(y_val.sum()), X_tr.shape[1])
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=valid_sets,
                        valid_names=valid_names, callbacks=callbacks)
    best_iter = booster.best_iteration or booster.current_iteration()
    report = {"best_iteration": int(best_iter), "params": params,
              "train": _safe_metrics(y_tr, booster.predict(X_tr, num_iteration=best_iter))}
    if has_val:
        report["val"] = _safe_metrics(y_val, booster.predict(X_val, num_iteration=best_iter))
    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(os.path.join(out_dir, "lightgbm.txt"), num_iteration=best_iter)
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain", iteration=best_iter),
        "split": booster.feature_importance("split", iteration=best_iter),
    }).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / max(imp["gain"].sum(), 1e-12)
    imp.to_csv(os.path.join(out_dir, "feature_importance.tsv"), sep="\t", index=False)
    report["top_features_by_gain"] = imp.head(25)[["feature", "gain_share"]].to_dict("records")
    save_json(report, os.path.join(out_dir, "lightgbm_report.json"))
    log.info("LightGBM best_iteration=%d  val=%s", best_iter, report.get("val"))
    return booster, report


def load_lightgbm(model_dir: str) -> lgb.Booster:
    return lgb.Booster(model_file=os.path.join(model_dir, "lightgbm.txt"))


def predict_lightgbm(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    if len(X) == 0:
        return np.zeros(0)
    return booster.predict(X[booster.feature_name()])
