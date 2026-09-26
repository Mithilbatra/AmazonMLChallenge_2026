"""Shared scoring + decision code. The SAME functions score the holdout split
during training and the test set during prediction, so the reported holdout
numbers describe exactly the procedure used for the submission.

    candidates -> features -> LightGBM -> isotonic -> cascade -> cross-encoder
    -> isotonic -> blend -> final calibration -> graph cleanup -> decision rules
    (+ singleton gate) -> one id set per Source-1 entity
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decision import DecisionParams, DecisionTable, entity_features
from .graph_cleanup import graph_keep_mask
from .scoring import cascade_mask
from .features import predict_in_blocks
from .train_lightgbm import load_lightgbm
from .utils import get_logger, load_json, load_pickle


class TrainedPipeline:
    """Every fitted artefact needed to score a new dataset."""

    def __init__(self, model_dir: str, cfg: dict):
        path = os.path.join(model_dir, "pipeline.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python train.py --config ...` first")
        self.dir = model_dir
        self.cfg = cfg
        self.meta = load_json(path)
        self.feature_columns: list[str] = self.meta["feature_columns"]
        self.booster = load_lightgbm(model_dir)
        self.cal_lgb = load_pickle(os.path.join(model_dir, "calibrator_lgb.pkl"))
        self.cal_ce = load_pickle(os.path.join(model_dir, "calibrator_ce.pkl")) if self.meta.get("has_ce") else None
        self.blender = load_pickle(os.path.join(model_dir, "blender.pkl"))
        self.cal_final = load_pickle(os.path.join(model_dir, "calibrator_final.pkl"))
        self.decision = DecisionParams.from_dict(self.meta["decision"])
        sm = os.path.join(model_dir, "singleton_model.txt")
        self.singleton_model = lgb.Booster(model_file=sm) if self.meta.get("has_singleton_model") and os.path.exists(sm) else None
        self.graph_cfg = self.meta.get("graph", {})
        self.cascade_cfg = self.meta.get("cascade", {})
        self._ce = None
        self._serializer_idf = None

    @property
    def has_ce(self) -> bool:
        return bool(self.meta.get("has_ce"))

    def ce_predictor(self):
        if self._ce is None:
            from .train_crossencoder import CrossEncoderPredictor
            self._ce = CrossEncoderPredictor(os.path.join(self.dir, "crossencoder"), self.cfg)
        return self._ce

    def serializer(self):
        from .train_crossencoder import DittoSerializer
        if self._serializer_idf is None:
            self._serializer_idf = load_pickle(os.path.join(self.dir, "serializer_idf.pkl"))
        return DittoSerializer(self.cfg, self._serializer_idf)


def score_candidates(pairs: pd.DataFrame, feats, s1: pd.DataFrame, v: pd.DataFrame,
                     pipe: TrainedPipeline, ce_rows: np.ndarray | None = None) -> pd.DataFrame:
    """Return pairs + p_lgb_raw, p_lgb, sent_to_ce, p_ce_raw, p_ce, p_blend, p_final.

    `feats` is a DataFrame or a disk-backed FeatureMatrix (scored in blocks)."""
    log = get_logger()
    out = pairs[["s1_idx", "v_idx", "source"]].copy()
    raw = predict_in_blocks(pipe.booster, feats, pipe.feature_columns)
    out["p_lgb_raw"] = raw
    out["p_lgb"] = pipe.cal_lgb.transform(raw)
    out["sent_to_ce"] = False
    out["p_ce_raw"] = np.nan
    out["p_ce"] = np.nan
    if pipe.has_ce:
        mask = cascade_mask(out, out["p_lgb"].to_numpy(), pipe.cascade_cfg)
        if ce_rows is not None:
            mask &= ce_rows
        out["sent_to_ce"] = mask
        if mask.any():
            ser = pipe.serializer()
            t1, tv = ser.serialize_frame(s1), ser.serialize_frame(v)
            ia = out.loc[mask, "s1_idx"].to_numpy()
            ib = out.loc[mask, "v_idx"].to_numpy()
            log.info("Cascade: %d of %d pairs (%.1f%%) sent to the cross-encoder", int(mask.sum()), len(out),
                     100 * mask.mean())
            p = pipe.ce_predictor().predict([t1[i] for i in ia], [tv[j] for j in ib])
            out.loc[mask, "p_ce_raw"] = p
            out.loc[mask, "p_ce"] = pipe.cal_ce.transform(p)
    has_ce = out["p_ce"].notna().to_numpy()
    out["p_blend"] = pipe.blender.transform(out["p_lgb"].to_numpy(), out["p_ce"].to_numpy(), has_ce)
    out["p_final"] = pipe.cal_final.transform(out["p_blend"].to_numpy())
    return out


def decide(scored: pd.DataFrame, entity_idx: np.ndarray, pipe: TrainedPipeline,
           params: DecisionParams | None = None, extra_cols: pd.DataFrame | None = None):
    """Apply graph cleanup + decision rules; returns (selected, info)."""
    params = params or pipe.decision
    gcfg = pipe.graph_cfg
    keep, gstats = graph_keep_mask(scored, "p_final", params.graph_mode, float(gcfg.get("min_edge_prob", 0.2)),
                                   int(gcfg.get("max_component_size", 50)))
    matchable = None
    if pipe.singleton_model is not None and params.matchable_threshold > 0:
        src = scored if extra_cols is None else pd.concat([scored, extra_cols], axis=1)
        ef = entity_features(src, entity_idx)
        matchable = pipe.singleton_model.predict(ef)
    table = DecisionTable.build(scored, "p_final", entity_idx, graph_masks={params.graph_mode: keep},
                                matchable=matchable)
    selected = table.select(params)
    info = {"graph": gstats, "matchable": matchable, "graph_keep": keep}
    return selected, info
