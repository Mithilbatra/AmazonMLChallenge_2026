"""TSV ingestion: loading, schema validation, null handling, duplicate
detection, Source 1/2/3 separation and label parsing.

Named ``data_io`` (not ``io``) so it can never shadow Python's stdlib ``io``.
Every read uses an explicit TAB separator, as the transcript requires.
"""
from __future__ import annotations

import csv
import os
import re

import pandas as pd

from .config import get
from .schema import (LabelSchema, SchemaError, SourceSchema, label_schema, source_schema,
                     validate_columns)
from .utils import get_logger

SEP = "\t"
_EMPTY_TOKENS = {"", "[]", "nan", "none", "null", "na", "n/a", "{}", "()"}


def read_tsv(path: str, cfg: dict | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TSV file not found: {path}")
    cfg = cfg or {}
    encoding = get(cfg, "io.encoding", "utf-8-sig")
    quoting = csv.QUOTE_NONE if get(cfg, "io.quoting", "minimal") == "none" else csv.QUOTE_MINIMAL
    df = pd.read_csv(
        path,
        sep=SEP,
        dtype=str,
        keep_default_na=False,   # never turn "NA"/"null" business names into NaN
        na_values=[],
        quoting=quoting,
        encoding=encoding,
        engine="c",
    )
    df.columns = [str(c).strip() for c in df.columns]
    # Sanity check: parsed rows vs physical lines (detects quoting problems).
    with open(path, "rb") as fh:
        n_lines = sum(1 for line in fh if line.strip())
    if n_lines - 1 != len(df):
        get_logger().warning(
            "%s: %d data lines on disk but %d parsed rows. If names contain quote characters, "
            "try io.quoting: none (or minimal).", path, n_lines - 1, len(df))
    return df


def _clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def load_source(path: str, source: int, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Load one source file into columns [id, name, address] (+ original row order)."""
    log = get_logger()
    schema: SourceSchema = source_schema(cfg, source)
    raw = read_tsv(path, cfg)
    validate_columns(raw, schema.required, f"Source {source} ({path})")
    df = pd.DataFrame({
        "id": _clean_str(raw[schema.id_col]),
        "name": _clean_str(raw[schema.name_col]),
    })
    parts = [_clean_str(raw[c]) for c in schema.address_cols]
    if parts:
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.str.cat(p, sep=schema.address_join)
        # remove empty components produced by missing columns: "a, , b" -> "a, b"
        sep_re = re.escape(schema.address_join.strip() or " ")
        joined = (joined.str.replace(rf"(\s*{sep_re}\s*)+", schema.address_join, regex=True)
                  .str.strip().str.strip(schema.address_join.strip() or " ").str.strip())
        df["address"] = joined
    else:
        df["address"] = ""
    df["row_order"] = range(len(df))

    stats = {"file": path, "rows": int(len(df))}
    empty_id = (df["id"] == "").sum()
    if empty_id:
        log.warning("Source %d: %d rows with empty id are dropped", source, empty_id)
        df = df[df["id"] != ""]
    stats["empty_id_dropped"] = int(empty_id)
    stats["empty_name"] = int((df["name"] == "").sum())
    stats["empty_address"] = int((df["address"] == "").sum())
    stats["empty_name_and_address"] = int(((df["name"] == "") & (df["address"] == "")).sum())

    dup_mask = df["id"].duplicated(keep="first")
    stats["duplicate_ids"] = int(dup_mask.sum())
    if dup_mask.any():
        policy = get(cfg, "schema.duplicate_ids", "warn_keep_first")
        msg = f"Source {source}: {int(dup_mask.sum())} duplicated ids (e.g. {df.loc[dup_mask, 'id'].head(3).tolist()})"
        if policy == "error":
            raise SchemaError(msg)
        log.warning("%s -> keeping first occurrence", msg)
        df = df[~dup_mask]
    key = df["name"].str.lower() + "\x1f" + df["address"].str.lower()
    stats["duplicate_name_address_rows"] = int(key.duplicated().sum())
    df = df.reset_index(drop=True)
    df["source"] = source
    log.info("Source %d: %d records (%s)", source, len(df), path)
    return df, stats


def load_sources(cfg: dict, split: str) -> tuple[dict[int, pd.DataFrame], dict]:
    sources, stats = {}, {}
    for s in (1, 2, 3):
        path = get(cfg, f"data.{split}.source{s}")
        if not path:
            raise SchemaError(f"data.{split}.source{s} is not configured")
        sources[s], stats[f"source{s}"] = load_source(path, s, cfg)
    overlap = set(sources[2]["id"]) & set(sources[3]["id"])
    stats["ids_shared_by_source2_and_source3"] = len(overlap)
    if overlap:
        get_logger().warning(
            "%d ids occur in BOTH Source 2 and Source 3 (e.g. %s). Label ids are then ambiguous "
            "unless the label file has one column per source.", len(overlap), sorted(overlap)[:3])
    return sources, stats


def parse_id_list(value, sep: str = ",") -> list[str]:
    """Parse "a,b", "a, b", "[a, b]", "['a','b']", "" or "[]" into a list."""
    if value is None:
        return []
    text = str(value).strip()
    if text.lower() in _EMPTY_TOKENS:
        return []
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")) \
            or (text.startswith("{") and text.endswith("}")):
        text = text[1:-1]
    out, seen = [], set()
    for tok in text.split(sep):
        tok = tok.strip().strip("'\"").strip()
        if tok and tok.lower() not in _EMPTY_TOKENS and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def detect_list_format(values: pd.Series) -> str:
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return "plain"
    bracketed = sum(v.startswith("[") and v.endswith("]") for v in vals)
    return "bracketed" if bracketed > len(vals) / 2 else "plain"


def load_labels(path: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return a frame with columns [s1_id, <match col>..., all_ids(list)]."""
    log = get_logger()
    schema: LabelSchema = label_schema(cfg)
    raw = read_tsv(path, cfg)
    validate_columns(raw, schema.required, f"Labels ({path})")
    out = pd.DataFrame({"s1_id": _clean_str(raw[schema.source1_id])})
    fmt_votes = []
    for col in schema.match_columns:
        out[col] = [parse_id_list(v, schema.list_separator) for v in raw[col]]
        fmt_votes.append(detect_list_format(raw[col]))
    out["all_ids"] = [
        list(dict.fromkeys(i for col in schema.match_columns for i in row[col]))
        for _, row in out.iterrows()
    ]
    stats = {"file": path, "rows": int(len(out)), "list_format": max(set(fmt_votes), key=fmt_votes.count)}
    dups = out["s1_id"].duplicated(keep=False)
    if dups.any():
        log.warning("Labels: %d rows share a Source-1 id -> their lists are unioned", int(dups.sum()))
        agg = {col: (lambda s: list(dict.fromkeys(x for lst in s for x in lst))) for col in
               [*schema.match_columns, "all_ids"]}
        out = out.groupby("s1_id", as_index=False, sort=False).agg(agg)
    stats["duplicate_source1_rows"] = int(dups.sum())
    sizes = out["all_ids"].map(len)
    stats["entities"] = int(len(out))
    stats["singletons"] = int((sizes == 0).sum())
    stats["singleton_rate"] = float((sizes == 0).mean()) if len(out) else 0.0
    stats["match_list_size_distribution"] = {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
    stats["total_label_ids"] = int(sizes.sum())
    log.info("Labels: %d Source-1 entities, %d singletons (%.1f%%), %d matching ids",
             len(out), stats["singletons"], 100 * stats["singleton_rate"], stats["total_label_ids"])
    return out, stats


def read_sample_submission_header(cfg: dict) -> list[str] | None:
    path = get(cfg, "data.sample_submission")
    if not path:
        return None
    if not os.path.exists(path):
        get_logger().warning("sample_submission %s not found - mirroring the label schema", path)
        return None
    return list(read_tsv(path, cfg).columns)


def write_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep=SEP, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
