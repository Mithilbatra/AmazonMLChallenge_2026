"""Cascade and score blending.

Cascade (why): blocking yields tens of candidates per entity; a DeBERTa-v3
cross-encoder costs ~1000x a LightGBM prediction. The cheap structural model
therefore scores EVERY candidate, and only candidates that are plausible but
not already certain are re-scored by the cross-encoder:

    lgb_low <= p_lgb (calibrated)  and  rank within (entity, source) <= max_per_entity_source
    and (p_lgb <= lgb_high  or  rescore_confident)

Pairs outside the cascade keep their (calibrated) LightGBM probability.

Blending (tuned on the calib split by Macro F0.5, never on test):
  weighted  p = w * p_lgb + (1 - w) * p_ce
  rank      same on empirical-CDF ranks (robust to scale differences)
  stack     logistic regression on [logit p_lgb, logit p_ce, their product]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def cascade_mask(pairs: pd.DataFrame, p_lgb: np.ndarray, ccfg: dict) -> np.ndarray:
    low = float(ccfg.get("lgb_low", 0.02))
    high = float(ccfg.get("lgb_high", 0.995))
    k = int(ccfg.get("max_per_entity_source", 10))
    df = pd.DataFrame({"s1": pairs["s1_idx"].to_numpy(), "src": pairs["source"].to_numpy(), "p": p_lgb})
    rank = df.groupby(["s1", "src"], sort=False)["p"].rank(ascending=False, method="first").to_numpy()
    mask = (p_lgb >= low) & (rank <= k)
    if not ccfg.get("rescore_confident", False):
        mask &= p_lgb <= high
    return mask


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Blender:
    def __init__(self, method: str = "weighted", weight: float = 1.0):
        self.method = method
        self.weight = float(weight)
        self.ref_lgb = None
        self.ref_ce = None
        self.stacker = None

    def fit(self, p_lgb, p_ce, has_ce, y=None) -> "Blender":
        has_ce = np.asarray(has_ce, dtype=bool)
        if self.method == "rank":
            self.ref_lgb = np.sort(np.asarray(p_lgb)[has_ce])
            self.ref_ce = np.sort(np.asarray(p_ce)[has_ce])
        elif self.method == "stack":
            X = self._stack_x(np.asarray(p_lgb)[has_ce], np.asarray(p_ce)[has_ce])
            yy = np.asarray(y)[has_ce]
            if len(np.unique(yy)) == 2:
                self.stacker = LogisticRegression(C=1.0, max_iter=1000).fit(X, yy)
        return self

    @staticmethod
    def _stack_x(a, b):
        la, lb = _logit(a), _logit(b)
        return np.column_stack([la, lb, la * lb])

    @staticmethod
    def _cdf(ref, x):
        if ref is None or len(ref) == 0:
            return np.asarray(x, dtype=float)
        return np.searchsorted(ref, x, side="right") / len(ref)

    def transform(self, p_lgb, p_ce, has_ce) -> np.ndarray:
        p_lgb = np.asarray(p_lgb, dtype=float)
        p_ce = np.nan_to_num(np.asarray(p_ce, dtype=float), nan=0.0)
        has_ce = np.asarray(has_ce, dtype=bool)
        out = p_lgb.copy()
        if not has_ce.any():
            return out
        a, b = p_lgb[has_ce], p_ce[has_ce]
        if self.method == "weighted":
            out[has_ce] = self.weight * a + (1 - self.weight) * b
        elif self.method == "rank":
            ra, rb = self._cdf(self.ref_lgb, a), self._cdf(self.ref_ce, b)
            blended_rank = self.weight * ra + (1 - self.weight) * rb
            # map the blended rank back onto the LightGBM probability scale so
            # that cascaded and non-cascaded pairs stay comparable
            ref = self.ref_lgb if self.ref_lgb is not None and len(self.ref_lgb) else np.sort(a)
            out[has_ce] = np.quantile(ref, np.clip(blended_rank, 0, 1))
        elif self.method == "stack":
            if self.stacker is not None:
                out[has_ce] = self.stacker.predict_proba(self._stack_x(a, b))[:, 1]
        else:
            raise ValueError(f"unknown blend method {self.method}")
        return out

    def describe(self) -> dict:
        d = {"method": self.method, "weight_lgb": self.weight}
        if self.stacker is not None:
            d["stack_coef"] = self.stacker.coef_.ravel().tolist()
            d["stack_intercept"] = float(self.stacker.intercept_[0])
        return d
