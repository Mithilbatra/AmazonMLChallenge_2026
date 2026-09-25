import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.data_io import read_tsv
from src.schema import LabelSchema, submission_schema
from src.submission import build_matching_results, validate_submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_matching_results_one_row_per_entity(tiny_tables, tmp_path):
    s1, v = tiny_tables
    scored = pd.DataFrame({"s1_idx": [0, 0, 1, 2], "v_idx": [0, 3, 2, 5], "p_final": [0.9, 0.95, 0.8, 0.1]})
    selected = np.array([True, True, True, False])
    sch = submission_schema({}, LabelSchema("source1_id", ["matches"]))
    out = build_matching_results(s1, v, scored, selected, sch)
    assert list(out.columns) == ["source1_id", "matches"]
    assert out["source1_id"].tolist() == ["A1", "A2", "A3"]            # every entity, original order
    assert out["matches"].tolist() == ["C1,B1", "B3", ""]               # sorted by score; singleton empty
    path = tmp_path / "matching_results.tsv"
    out.to_csv(path, sep="\t", index=False)
    back = read_tsv(str(path))
    assert back.loc[2, "matches"] == ""
    res = validate_submission(str(path), ["A1", "A2", "A3"], {"B1", "B2", "B3"}, {"C1", "C2", "C3"}, sch)
    assert res["ok"], res


def test_per_source_columns(tiny_tables):
    s1, v = tiny_tables
    scored = pd.DataFrame({"s1_idx": [0, 0], "v_idx": [0, 3], "p_final": [0.9, 0.95]})
    sch = submission_schema({}, LabelSchema("sid", ["s2", "s3"], {"s2": 2, "s3": 3}))
    out = build_matching_results(s1, v, scored, np.array([True, True]), sch)
    assert out.loc[0, "s2"] == "B1" and out.loc[0, "s3"] == "C1" and out.loc[1, "s2"] == ""


def test_validator_catches_errors(tmp_path):
    sch = submission_schema({}, LabelSchema("source1_id", ["matches"]))
    bad = pd.DataFrame({"source1_id": ["A1", "A1"], "matches": ["B1,ZZZ", ""]})
    p = tmp_path / "bad.tsv"
    bad.to_csv(p, sep="\t", index=False)
    res = validate_submission(str(p), ["A1", "A2"], {"B1"}, set(), sch)
    assert not res["ok"]
    joined = " ".join(res["errors"])
    assert "duplicated" in joined and "missing" in joined and "not found" in joined


@pytest.mark.slow
def test_end_to_end_baseline(tmp_path):
    """Synthetic data -> train -> predict -> validate (baseline mode)."""
    data = tmp_path / "data"
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True, capture_output=True, text=True)
    run("scripts/make_synthetic_data.py", "--out", str(data), "--n-train", "300", "--n-test", "120")
    overrides = []
    for split in ("train", "test"):
        for s in (1, 2, 3):
            overrides += ["--set", f"data.{split}.source{s}={data}/{split}/source{s}.tsv"]
    overrides += ["--set", f"data.train.labels={data}/train/labels.tsv",
                  "--set", f"paths.models_dir={tmp_path}/models", "--set", f"paths.outputs_dir={tmp_path}/outputs",
                  "--set", "run_name=e2e", "--set", "n_jobs=1"]
    run("train.py", "--config", "configs/synthetic_baseline.yaml", *overrides)
    run("predict.py", "--config", "configs/synthetic_baseline.yaml", *overrides)
    res = read_tsv(str(tmp_path / "outputs/e2e/test/matching_results.tsv"))
    s1 = read_tsv(str(data / "test/source1.tsv"))
    assert len(res) == len(s1) and set(res["source1_id"]) == set(s1["record_id"])
    assert (tmp_path / "outputs/e2e/test/candidate_pairs.tsv").exists()
    out = run("evaluate.py", "--config", "configs/synthetic_baseline.yaml", *overrides,
              "--predictions", str(tmp_path / "outputs/e2e/test/matching_results.tsv"),
              "--labels", str(data / "test_labels_HIDDEN.tsv"))
    assert '"macro_f0.5"' in out.stdout
