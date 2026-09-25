"""Column schema definitions and validation for the challenge TSV files.

The official transcript does not name files or columns, so every column name
is taken from the `schema:` section of the config. Validation errors list the
columns that ARE present so the config can be fixed quickly
(`python inspect_data.py` prints them as well).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import get


class SchemaError(ValueError):
    pass


@dataclass
class SourceSchema:
    source: int
    id_col: str
    name_col: str
    address_cols: list[str]
    address_join: str = ", "

    @property
    def required(self) -> list[str]:
        return [self.id_col, self.name_col, *self.address_cols]


@dataclass
class LabelSchema:
    source1_id: str
    match_columns: list[str]
    column_sources: dict[str, int] | None = None
    list_separator: str = ","
    missing_source1_policy: str = "singleton"
    ambiguous_id_policy: str = "skip"
    unknown_id_policy: str = "warn"

    @property
    def required(self) -> list[str]:
        return [self.source1_id, *self.match_columns]


@dataclass
class SubmissionSchema:
    id_column: str
    match_columns: list[str]
    column_sources: dict[str, int] | None
    list_separator: str = ","
    list_format: str = "plain"
    candidate_columns: dict = field(default_factory=dict)


def source_schema(cfg: dict, source: int) -> SourceSchema:
    node = get(cfg, f"schema.source{source}")
    if node is None:
        raise SchemaError(f"schema.source{source} missing from config")
    address = node.get("address", [])
    if isinstance(address, str):
        address = [address]
    return SourceSchema(
        source=source,
        id_col=node["id"],
        name_col=node["name"],
        address_cols=list(address),
        address_join=get(cfg, "schema.address_join", ", "),
    )


def label_schema(cfg: dict) -> LabelSchema:
    node = get(cfg, "schema.labels", {})
    cols = node.get("match_columns", ["matches"])
    if isinstance(cols, str):
        cols = [cols]
    col_sources = node.get("column_sources")
    if col_sources:
        col_sources = {str(k): int(v) for k, v in col_sources.items()}
    return LabelSchema(
        source1_id=node.get("source1_id", "source1_id"),
        match_columns=list(cols),
        column_sources=col_sources,
        list_separator=node.get("list_separator", ","),
        missing_source1_policy=node.get("missing_source1_policy", "singleton"),
        ambiguous_id_policy=node.get("ambiguous_id_policy", "skip"),
        unknown_id_policy=node.get("unknown_id_policy", "warn"),
    )


def submission_schema(cfg: dict, labels: LabelSchema, detected_list_format: str = "plain",
                      sample_header: list[str] | None = None) -> SubmissionSchema:
    """Output schema. Defaults mirror the label file ("This mirrors exactly
    what you'll submit" - transcript). A sample submission header wins."""
    node = get(cfg, "submission", {})
    id_col = node.get("id_column") or labels.source1_id
    match_cols = node.get("match_columns") or list(labels.match_columns)
    col_sources = labels.column_sources
    if sample_header:
        id_col = sample_header[0]
        new_cols = sample_header[1:]
        if len(new_cols) == len(match_cols) and col_sources:
            col_sources = {new: col_sources.get(old) for new, old in zip(new_cols, match_cols)}
        elif len(new_cols) != len(match_cols):
            col_sources = None
        match_cols = new_cols
    if isinstance(match_cols, str):
        match_cols = [match_cols]
    if len(match_cols) > 1 and not col_sources:
        # Heuristic only used when the per-source mapping is not configured.
        guess = {}
        for c in match_cols:
            if "2" in c and "3" not in c:
                guess[c] = 2
            elif "3" in c and "2" not in c:
                guess[c] = 3
        if len(guess) == len(match_cols) and sorted(guess.values()) == [2, 3]:
            col_sources = guess
        else:
            raise SchemaError(
                f"Several output match columns {match_cols} but no source mapping; "
                "set schema.labels.column_sources, e.g. {col_a: 2, col_b: 3}."
            )
    fmt = node.get("list_format", "auto")
    if fmt == "auto":
        fmt = detected_list_format
    return SubmissionSchema(
        id_column=id_col,
        match_columns=list(match_cols),
        column_sources=col_sources,
        list_separator=node.get("list_separator", labels.list_separator),
        list_format=fmt,
        candidate_columns=node.get("candidate_columns", {
            "source1_id": "source1_id", "candidate_id": "candidate_id",
            "candidate_source": "candidate_source"}),
    )


def validate_columns(df: pd.DataFrame, required: list[str], what: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{what}: missing column(s) {missing}. Columns present: {list(df.columns)}. "
            "Fix the `schema:` section of the config (see `python inspect_data.py`)."
        )
