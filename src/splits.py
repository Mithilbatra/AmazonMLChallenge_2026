"""Leakage-safe entity-level splits.

Unit of splitting: ONE Source-1 entity together with every labelled
Source-2/3 record of that entity. Because Source 1 is de-duplicated, a
business appears in exactly one split, and all of its positive pairs,
candidate pairs and features go with it.

  fit      -> trains LightGBM / bi-encoder / cross-encoder
              (fit_val = subset of fit used only for early stopping)
  calib    -> isotonic calibration, blend weights, decision search,
              singleton model
  holdout  -> never used for any fitting; unbiased final estimate

Stratified on (#S2 matches, #S3 matches) capped at 2, so each split has the
same singleton / multi-match mix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import LabelIndex


def make_splits(li: LabelIndex, v_sources: pd.Series, fractions: dict, inner_val_fraction: float,
                seed: int) -> np.ndarray:
    n = len(li.true_keys)
    tp = li.true_pairs.copy()
    tp["source"] = v_sources.to_numpy()[tp["v_idx"].to_numpy()] if len(tp) else []
    n2 = np.bincount(tp.loc[tp["source"] == 2, "s1_idx"], minlength=n) if len(tp) else np.zeros(n, int)
    n3 = np.bincount(tp.loc[tp["source"] == 3, "s1_idx"], minlength=n) if len(tp) else np.zeros(n, int)
    unresolved = li.n_true - (n2 + n3)
    strata = np.minimum(n2, 2) * 3 + np.minimum(n3, 2) + 9 * (unresolved > 0)
    names = list(fractions.keys())
    probs = np.array([fractions[k] for k in names], dtype=float)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    split = np.array(["unlabeled"] * n, dtype=object)
    for s in np.unique(strata):
        idx = np.where((strata == s) & li.labeled)[0]
        rng.shuffle(idx)
        cuts = np.round(np.cumsum(probs) * len(idx)).astype(int)
        start = 0
        for name, end in zip(names, cuts):
            split[idx[start:end]] = name
            start = end
    fit_idx = np.where(split == "fit")[0]
    rng.shuffle(fit_idx)
    n_val = int(round(inner_val_fraction * len(fit_idx)))
    split[fit_idx[:n_val]] = "fit_val"
    return split


def split_summary(split: np.ndarray, li: LabelIndex) -> dict:
    out = {}
    single = li.is_singleton()
    for name in pd.unique(split):
        m = split == name
        out[str(name)] = {"entities": int(m.sum()), "singletons": int((single & m).sum()),
                          "true_matches": int(li.n_true[m].sum())}
    return out
