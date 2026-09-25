"""Evaluation metrics.

Challenge metric (transcript): Macro F0.5 over Source-1 entities. For each
Source-1 entity the predicted SET of Source-2/3 ids is compared with the true
set; F-beta (beta = 0.5, precision weighted twice as heavily as recall) is
computed per entity and averaged with equal weight.

    F_beta = (1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP)

Special case (transcript): a singleton (true set empty) scores 1.0 when the
prediction is empty and 0.0 otherwise. A non-singleton predicted empty scores 0.
The formula above already gives these values except the empty/empty case,
which is set to 1 explicitly.
"""
from __future__ import annotations

import numpy as np


def entity_fbeta(tp, n_pred, n_true, beta: float = 0.5) -> np.ndarray:
    tp = np.asarray(tp, dtype=float)
    n_pred = np.asarray(n_pred, dtype=float)
    n_true = np.asarray(n_true, dtype=float)
    fp = n_pred - tp
    fn = n_true - tp
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + b2) * tp / np.where(denom > 0, denom, 1), 1.0)
    return f


def macro_fbeta_sets(pred: dict, true: dict, entities, beta: float = 0.5) -> dict:
    """Reference implementation on Python sets (used by the scorer and tests)."""
    tp, npred, ntrue = [], [], []
    for e in entities:
        p = set(pred.get(e, ()))
        t = set(true.get(e, ()))
        tp.append(len(p & t))
        npred.append(len(p))
        ntrue.append(len(t))
    return summarize(np.array(tp), np.array(npred), np.array(ntrue), beta)


def summarize(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray, beta: float = 0.5) -> dict:
    tp = np.asarray(tp, dtype=float)
    n_pred = np.asarray(n_pred, dtype=float)
    n_true = np.asarray(n_true, dtype=float)
    f = entity_fbeta(tp, n_pred, n_true, beta)
    n = len(f)
    single = n_true == 0
    pred_empty = n_pred == 0
    micro_p = tp.sum() / n_pred.sum() if n_pred.sum() else 1.0
    micro_r = tp.sum() / n_true.sum() if n_true.sum() else 1.0
    b2 = beta * beta
    micro_f = ((1 + b2) * micro_p * micro_r / (b2 * micro_p + micro_r)) if (micro_p + micro_r) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ent_p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), np.nan)
        ent_r = np.where(n_true > 0, tp / np.maximum(n_true, 1), np.nan)
    return {
        "macro_f0.5" if beta == 0.5 else f"macro_f{beta}": float(f.mean()) if n else float("nan"),
        "entities": int(n),
        "entity_precision_mean(pred_nonempty)": float(np.nanmean(ent_p)) if np.any(n_pred > 0) else float("nan"),
        "entity_recall_mean(true_nonempty)": float(np.nanmean(ent_r)) if np.any(n_true > 0) else float("nan"),
        "pair_precision": float(micro_p),
        "pair_recall": float(micro_r),
        "pair_f0.5": float(micro_f),
        "true_singletons": int(single.sum()),
        "predicted_singletons": int(pred_empty.sum()),
        "singleton_correct(empty_pred)": int((single & pred_empty).sum()),
        "singleton_false_merges": int((single & ~pred_empty).sum()),
        "singleton_accuracy": float((single & pred_empty).sum() / single.sum()) if single.any() else float("nan"),
        "nonsingleton_predicted_empty": int((~single & pred_empty).sum()),
        "nonsingleton_exact_set": int((~single & (tp == n_true) & (n_pred == n_true)).sum()),
        "macro_f_singletons": float(f[single].mean()) if single.any() else float("nan"),
        "macro_f_nonsingletons": float(f[~single].mean()) if (~single).any() else float("nan"),
        "predicted_matches": int(n_pred.sum()),
        "true_matches": int(n_true.sum()),
    }


def macro_from_arrays(ent: np.ndarray, selected: np.ndarray, is_true: np.ndarray,
                      n_true: np.ndarray, beta: float = 0.5) -> float:
    """Fast Macro F-beta. `ent` = dense entity codes 0..E-1 for each pair,
    `n_true` = true-set sizes for ALL E entities (including entities that have
    no candidate at all and true matches lost in blocking)."""
    E = len(n_true)
    sel = selected.astype(bool)
    tp = np.bincount(ent[sel & is_true], minlength=E)
    npred = np.bincount(ent[sel], minlength=E)
    return float(entity_fbeta(tp, npred, n_true, beta).mean())


def pairwise_prf(selected: np.ndarray, y: np.ndarray, beta: float = 0.5) -> dict:
    sel = selected.astype(bool)
    y = y.astype(bool)
    tp = int((sel & y).sum())
    fp = int((sel & ~y).sum())
    fn = int((~sel & y).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, f"f{beta}": f}


def reduction_ratio(n_candidates: int, n_possible: int) -> float:
    return 1.0 - n_candidates / n_possible if n_possible else float("nan")
