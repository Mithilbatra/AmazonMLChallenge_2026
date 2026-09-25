"""Probability calibration (PDF: "Isotonic Regression and Decision Threshold
Calibration").

Calibrators are ALWAYS fitted on the `calib` split (entities never seen by
the matchers) and never on test data. Isotonic regression (the PDF's choice)
and Platt/logistic scaling are both available; `auto` picks the one with the
lower cross-validated Brier score and the comparison is reported either way.

Isotonic regression is a step function, so many pairs share one calibrated
value; a 1e-4 * raw-score tie-breaker keeps the order of the raw scores
(monotone, so it cannot change which pairs pass a threshold by more than
that epsilon) which the per-entity ranking rules rely on.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

_EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt", "none"):
            raise ValueError(f"unknown calibration method {method}")
        self.method = method
        self.model = None

    def fit(self, scores, y) -> "Calibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=int)
        if self.method == "none" or len(np.unique(y)) < 2:
            self.method = "none" if len(np.unique(y)) < 2 else self.method
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)
        else:
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y)
        return self

    def transform(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "none" or self.model is None:
            return scores
        if self.method == "isotonic":
            cal = self.model.predict(scores)
        else:
            cal = self.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return np.clip(cal * (1 - _EPS) + _EPS * np.clip(scores, 0, 1), 0.0, 1.0)


def compare_methods(scores, y, folds: int = 3, seed: int = 0) -> dict:
    """Cross-validated Brier score / log loss of raw, isotonic and Platt."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    out = {"raw": {"brier": float(brier_score_loss(y, np.clip(scores, 0, 1)))}}
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < folds:
        return out
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for method in ("isotonic", "platt"):
        oof = np.zeros(len(y))
        for tr, te in skf.split(scores, y):
            oof[te] = Calibrator(method).fit(scores[tr], y[tr]).transform(scores[te])
        out[method] = {"brier": float(brier_score_loss(y, oof)),
                       "logloss": float(log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7))),
                       "ece": expected_calibration_error(oof, y)}
    return out


def fit_calibrator(scores, y, method: str = "isotonic", seed: int = 0) -> tuple[Calibrator, dict]:
    report = compare_methods(scores, y, seed=seed)
    chosen = method
    if method == "auto":
        cands = [m for m in ("isotonic", "platt") if m in report]
        chosen = min(cands, key=lambda m: report[m]["brier"]) if cands else "none"
    cal = Calibrator(chosen).fit(scores, y)
    report["chosen"] = cal.method
    if len(scores):
        report["reliability"] = reliability_table(cal.transform(scores), y)
    return cal, report


def expected_calibration_error(p, y, bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(p, y, bins: int = 10) -> list[dict]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else None,
                     "frac_pos": float(y[m].mean()) if m.any() else None})
    return rows
