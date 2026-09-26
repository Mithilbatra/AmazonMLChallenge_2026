"""Ground-truth handling: resolve label ids to records, build positive pairs,
attach labels to candidate pairs, tag hard negatives.

Label format (transcript): one row per Source-1 entity; its id maps to a
comma-separated list of ALL its matching Source-2/3 ids; empty list = no match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import SchemaError, label_schema
from .utils import get_logger


@dataclass
class LabelIndex:
    key_mode: str                         # "id" (combined column) or "source_id" (per-source columns)
    true_keys: list[set]                  # per s1_idx: set of metric keys (strings)
    labeled: np.ndarray                   # per s1_idx: bool, entity usable for training/eval
    true_pairs: pd.DataFrame              # [s1_idx, v_idx] resolved positives
    stats: dict = field(default_factory=dict)

    @property
    def n_true(self) -> np.ndarray:
        return np.array([len(s) for s in self.true_keys], dtype=np.int64)

    def is_singleton(self) -> np.ndarray:
        return self.n_true == 0


def metric_key_series(v: pd.DataFrame, key_mode: str) -> pd.Series:
    if key_mode == "id":
        return v["id"]
    return v["source"].astype(str) + ":" + v["id"]


def build_label_index(labels_df: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, cfg: dict) -> LabelIndex:
    log = get_logger()
    schema = label_schema(cfg)
    key_mode = "source_id" if schema.column_sources else "id"
    id2s1 = dict(zip(s1["id"], s1["s1_idx"]))
    ids_by_source = {s: dict(zip(v.loc[v["source"] == s, "id"], v.loc[v["source"] == s, "v_idx"]))
                     for s in (2, 3)}
    true_keys: list[set] = [set() for _ in range(len(s1))]
    labeled = np.zeros(len(s1), dtype=bool)
    pos_rows = []
    unknown_s1, unknown_ids, ambiguous_ids = [], [], []
    for _, row in labels_df.iterrows():
        s1_idx = id2s1.get(row["s1_id"])
        if s1_idx is None:
            unknown_s1.append(row["s1_id"])
            continue
        labeled[s1_idx] = True
        for col in schema.match_columns:
            for mid in row[col]:
                if schema.column_sources:
                    src = schema.column_sources.get(col)
                    if src not in (2, 3):
                        raise SchemaError(f"column_sources must map {col} to 2 or 3")
                    true_keys[s1_idx].add(f"{src}:{mid}")
                    v_idx = ids_by_source[src].get(mid)
                    if v_idx is None:
                        unknown_ids.append(mid)
                    else:
                        pos_rows.append((s1_idx, v_idx))
                    continue
                true_keys[s1_idx].add(mid)
                in2, in3 = mid in ids_by_source[2], mid in ids_by_source[3]
                if in2 and in3:
                    ambiguous_ids.append(mid)
                    if schema.ambiguous_id_policy == "error":
                        raise SchemaError(f"Label id {mid!r} exists in both Source 2 and Source 3")
                    if schema.ambiguous_id_policy == "both":
                        pos_rows.append((s1_idx, ids_by_source[2][mid]))
                        pos_rows.append((s1_idx, ids_by_source[3][mid]))
                elif in2:
                    pos_rows.append((s1_idx, ids_by_source[2][mid]))
                elif in3:
                    pos_rows.append((s1_idx, ids_by_source[3][mid]))
                else:
                    unknown_ids.append(mid)
    if unknown_ids and schema.unknown_id_policy == "error":
        raise SchemaError(f"{len(unknown_ids)} label ids not found in Source 2/3, e.g. {unknown_ids[:5]}")
    missing = int((~labeled).sum())
    if missing:
        policy = schema.missing_source1_policy
        msg = f"{missing} Source-1 entities have no row in the label file"
        if policy == "error":
            raise SchemaError(msg)
        if policy == "singleton":
            log.warning("%s -> treated as singletons (empty list)", msg)
            labeled[:] = True
        else:
            log.warning("%s -> excluded from training/evaluation", msg)
    if unknown_s1:
        log.warning("%d label rows reference unknown Source-1 ids (e.g. %s)", len(unknown_s1), unknown_s1[:3])
    if unknown_ids:
        log.warning("%d label ids not found in Source 2/3 (e.g. %s); they still count in recall",
                    len(unknown_ids), unknown_ids[:3])
    if ambiguous_ids:
        log.warning("%d label ids exist in both Source 2 and 3 (policy=%s)", len(ambiguous_ids),
                    schema.ambiguous_id_policy)
    true_pairs = pd.DataFrame(pos_rows, columns=["s1_idx", "v_idx"]).drop_duplicates()
    true_pairs = true_pairs.merge(v[["v_idx", "source"]], on="v_idx", how="left")

    n_true = np.array([len(s) for s in true_keys])
    vendor_multi = true_pairs.groupby("v_idx")["s1_idx"].nunique()
    stats = {
        "key_mode": key_mode,
        "labeled_entities": int(labeled.sum()),
        "singletons": int(((n_true == 0) & labeled).sum()),
        "singleton_rate": float(((n_true == 0) & labeled).sum() / max(1, labeled.sum())),
        "positive_pairs_resolved": int(len(true_pairs)),
        "positive_pairs_source2": int((true_pairs["source"] == 2).sum()),
        "positive_pairs_source3": int((true_pairs["source"] == 3).sum()),
        "label_ids_total": int(n_true.sum()),
        "unknown_label_ids": len(unknown_ids),
        "ambiguous_label_ids": len(ambiguous_ids),
        "unknown_source1_rows": len(unknown_s1),
        # S1 is de-duplicated, so a vendor record should belong to at most one
        # S1 entity. This counts violations in the ground truth itself.
        "vendor_records_matched_to_multiple_s1": int((vendor_multi > 1).sum()),
        "matches_per_entity": {int(k): int(val) for k, val in
                               pd.Series(n_true[labeled]).value_counts().sort_index().items()},
    }
    log.info("Resolved %d positive pairs (%d S2 / %d S3); %d vendor records labelled to >1 S1 entity",
             stats["positive_pairs_resolved"], stats["positive_pairs_source2"],
             stats["positive_pairs_source3"], stats["vendor_records_matched_to_multiple_s1"])
    return LabelIndex(key_mode=key_mode, true_keys=true_keys, labeled=labeled,
                      true_pairs=true_pairs[["s1_idx", "v_idx"]], stats=stats)


def attach_labels(pairs: pd.DataFrame, li: LabelIndex, v: pd.DataFrame) -> pd.Series:
    """1 = true match, 0 = non-match, -1 = uncertain (excluded from training).

    A candidate whose id appears in the entity's label list but could not be
    resolved unambiguously is marked -1, so a true match can NEVER be used as
    a negative. Fully vectorised (no Python loop over the candidate pairs).
    """
    nv = max(1, len(v))
    s1_idx = pairs["s1_idx"].to_numpy(np.int64)
    v_idx = pairs["v_idx"].to_numpy(np.int64)
    pid = s1_idx * nv + v_idx
    tp_pid = np.sort(li.true_pairs["s1_idx"].to_numpy(np.int64) * nv + li.true_pairs["v_idx"].to_numpy(np.int64))
    out = np.zeros(len(pairs), dtype=np.int8)
    if len(tp_pid):
        pos = np.minimum(np.searchsorted(tp_pid, pid), len(tp_pid) - 1)
        out[tp_pid[pos] == pid] = 1
    # id listed in the entity's truth but not resolved to this record -> uncertain
    truth = pd.DataFrame([(e, k) for e, keys in enumerate(li.true_keys) for k in keys],
                         columns=["s1_idx", "mkey"])
    if len(truth):
        # only non-positive candidates of entities that HAVE true ids can be uncertain
        ent_has = np.zeros(len(li.true_keys), dtype=bool)
        ent_has[truth["s1_idx"].to_numpy()] = True
        rows = np.flatnonzero((out == 0) & ent_has[s1_idx])
        keys = metric_key_series(v, li.key_mode).to_numpy()
        cand = pd.DataFrame({"s1_idx": s1_idx[rows], "mkey": keys[v_idx[rows]], "row": rows})
        hit = cand.merge(truth, on=["s1_idx", "mkey"])["row"].to_numpy()
        listed = np.zeros(len(pairs), dtype=bool)
        listed[hit] = True
        out[listed & (out == 0)] = -1
    return pd.Series(out, index=pairs.index, name="label")


def missed_positive_pairs(pairs: pd.DataFrame, li: LabelIndex) -> pd.DataFrame:
    """True pairs absent from the candidate set (lost in blocking)."""
    merged = li.true_pairs.merge(pairs[["s1_idx", "v_idx"]], on=["s1_idx", "v_idx"],
                                 how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", ["s1_idx", "v_idx"]].reset_index(drop=True)


def tag_negative_types(feats: pd.DataFrame) -> pd.Series:
    """Categorise non-matches (for reporting and cross-encoder sampling)."""
    same_addr = (feats.get("addr_char_cos", 0) >= 0.8) | (
        (feats.get("hn_eq", 0) == 1) & (feats.get("pc_eq", 0) == 1))
    sim_name = (feats.get("name_jw", 0) >= 0.9) | (feats.get("name_char_cos", 0) >= 0.7)
    out = np.where(same_addr & sim_name, "similar_name_same_address",
                   np.where(same_addr, "shared_address",
                            np.where(sim_name, "similar_name", "other")))
    return pd.Series(out, index=feats.index, name="neg_type")
