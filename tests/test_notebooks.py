"""The generated notebooks must match src/ and must run end to end."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_notebooks_in_sync_with_src():
    pytest.importorskip("nbformat")
    res = subprocess.run([sys.executable, "scripts/build_notebooks.py", "--check"], cwd=ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.slow
def test_notebooks_execute_on_synthetic_demo(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    monkeypatch.setenv("ER_WORK_DIR", str(tmp_path / "work"))
    for name in ("01_train.ipynb", "02_predict_and_submit.ipynb"):
        with open(os.path.join(ROOT, "notebooks", name), encoding="utf-8") as fh:
            nb = nbformat.read(fh, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code" and "USE_SYNTHETIC_DEMO = False" in cell.source:
                cell.source = cell.source.replace("USE_SYNTHETIC_DEMO = False", "USE_SYNTHETIC_DEMO = True")
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path)}}).execute()
        with open(tmp_path / name, "w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
    out = tmp_path / "work" / "outputs" / "baseline_v1"
    assert (out / "test" / "matching_results.tsv").exists()
    assert (out / "test" / "candidate_pairs.tsv").exists()
    assert (out / "submission.zip").exists()
