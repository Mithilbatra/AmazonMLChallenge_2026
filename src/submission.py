"""Output generation, format validation and packaging.

matching_results.tsv  one row per Source-1 entity (every entity, in the
                      original Source-1 order); its predicted Source-2/3 ids as
                      a comma-separated list, EMPTY for predicted singletons.
                      Columns mirror the label file (transcript: the label
                      format "mirrors exactly what you'll submit") unless a
                      sample submission is configured.
candidate_pairs.tsv   the blocking output BEFORE any model scoring
                      ("candidate/unscored pairs").
"""
from __future__ import annotations

import fnmatch
import os
import zipfile

import numpy as np
import pandas as pd

from .data_io import SEP, parse_id_list, read_tsv, write_tsv
from .schema import SubmissionSchema
from .utils import get_logger


def _fmt(ids: list[str], sch: SubmissionSchema) -> str:
    if not ids:
        return "[]" if sch.list_format == "bracketed" else ""
    body = sch.list_separator.join(ids) if sch.list_format != "bracketed" else ", ".join(ids)
    return f"[{body}]" if sch.list_format == "bracketed" else body


def build_matching_results(s1: pd.DataFrame, v: pd.DataFrame, scored: pd.DataFrame, selected: np.ndarray,
                           sch: SubmissionSchema) -> pd.DataFrame:
    sel = scored.loc[selected, ["s1_idx", "v_idx", "p_final"]].sort_values(["s1_idx", "p_final"],
                                                                         ascending=[True, False])
    vid = v["id"].to_numpy()
    vsrc = v["source"].to_numpy()
    by_ent: dict[int, list[int]] = {}
    for a, b in zip(sel["s1_idx"].to_numpy(), sel["v_idx"].to_numpy()):
        by_ent.setdefault(int(a), []).append(int(b))
    order = s1.sort_values("row_order")
    rows = []
    for s1_idx, s1_id in zip(order["s1_idx"].to_numpy(), order["id"].to_numpy()):
        matched = by_ent.get(int(s1_idx), [])
        row = {sch.id_column: s1_id}
        if len(sch.match_columns) == 1:
            row[sch.match_columns[0]] = _fmt(list(dict.fromkeys(vid[j] for j in matched)), sch)
        else:
            for col in sch.match_columns:
                src = sch.column_sources[col]
                row[col] = _fmt(list(dict.fromkeys(vid[j] for j in matched if vsrc[j] == src)), sch)
        rows.append(row)
    return pd.DataFrame(rows, columns=[sch.id_column, *sch.match_columns])


def write_candidate_pairs(cand: pd.DataFrame, s1: pd.DataFrame, v: pd.DataFrame, path: str,
                          sch: SubmissionSchema) -> None:
    cols = sch.candidate_columns
    out = pd.DataFrame({
        cols.get("source1_id", "source1_id"): s1["id"].to_numpy()[cand["s1_idx"].to_numpy()],
        cols.get("candidate_id", "candidate_id"): v["id"].to_numpy()[cand["v_idx"].to_numpy()],
        cols.get("candidate_source", "candidate_source"): v["source"].to_numpy()[cand["v_idx"].to_numpy()],
    })
    write_tsv(out, path)


def validate_submission(path: str, s1_ids: list[str], s2_ids: set, s3_ids: set, sch: SubmissionSchema) -> dict:
    """Format checks mirroring what an organiser validator would reject.

    (The official validation script mentioned in the transcript should still
    be run before uploading.)"""
    errors, warnings = [], []
    if not os.path.exists(path):
        return {"ok": False, "errors": [f"{path} does not exist"], "warnings": []}
    with open(path, "rb") as fh:
        head = fh.readline()
    if b"\t" not in head:
        errors.append("header has no TAB character - file is not tab-separated")
    df = read_tsv(path)
    expected = [sch.id_column, *sch.match_columns]
    if list(df.columns) != expected:
        errors.append(f"columns {list(df.columns)} != expected {expected}")
        return {"ok": False, "errors": errors, "warnings": warnings}
    ids = df[sch.id_column].astype(str).str.strip()
    dup = ids[ids.duplicated()]
    if len(dup):
        errors.append(f"{len(dup)} duplicated Source-1 ids, e.g. {dup.head(3).tolist()}")
    missing = set(s1_ids) - set(ids)
    extra = set(ids) - set(s1_ids)
    if missing:
        errors.append(f"{len(missing)} Source-1 entities missing (every entity needs a row), e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} ids not in Source 1, e.g. {sorted(extra)[:3]}")
    all_vendor = s2_ids | s3_ids
    n_pred, unknown, empty_rows = 0, [], 0
    for col in sch.match_columns:
        allowed = all_vendor
        if sch.column_sources:
            allowed = s2_ids if sch.column_sources[col] == 2 else s3_ids
        for val in df[col]:
            lst = parse_id_list(val, sch.list_separator)
            n_pred += len(lst)
            unknown.extend(x for x in lst if x not in allowed)
    for _, row in df.iterrows():
        if all(not parse_id_list(row[c], sch.list_separator) for c in sch.match_columns):
            empty_rows += 1
    if unknown:
        errors.append(f"{len(unknown)} predicted ids not found in Source 2/3, e.g. {unknown[:3]}")
    if n_pred == 0:
        warnings.append("no matches predicted at all")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "rows": int(len(df)),
            "predicted_ids": int(n_pred), "empty_rows(predicted_singletons)": int(empty_rows)}


DEFAULT_CODE_PATTERNS = ["README.md", "METHODOLOGY.md", "requirements*.txt", "config.yaml", "configs/*.yaml", "pytest.ini",
                         "*.py", "src/*.py", "scripts/*.py", "tests/*.py"]


def package_submission(root: str, files: dict[str, str], archive: str, include_models_dir: str | None = None,
                       patterns: list[str] | None = None) -> dict:
    """Create the single archive: final matches + candidate pairs + complete
    reproducible pipeline + methodology document (transcript deliverables)."""
    log = get_logger()
    patterns = patterns or DEFAULT_CODE_PATTERNS
    added = []
    os.makedirs(os.path.dirname(os.path.abspath(archive)), exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"required submission file missing: {path}")
            zf.write(path, arcname)
            added.append(arcname)
        for dirpath, _, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            if rel_dir.startswith((".git", "data", "outputs", "models", "__pycache__", ".pytest_cache")) \
                    or "/__pycache__" in rel_dir:
                continue
            for fn in filenames:
                rel = os.path.normpath(os.path.join(rel_dir, fn))
                if any(fnmatch.fnmatch(rel, p) for p in patterns):
                    zf.write(os.path.join(dirpath, fn), os.path.join("pipeline", rel))
                    added.append(os.path.join("pipeline", rel))
        if include_models_dir and os.path.isdir(include_models_dir):
            for dirpath, _, filenames in os.walk(include_models_dir):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    arc = os.path.join("pipeline", os.path.relpath(full, root))
                    zf.write(full, arc)
                    added.append(arc)
    log.info("Wrote %s with %d files", archive, len(added))
    return {"archive": archive, "files": added}
