#!/usr/bin/env python
"""Generate the Jupyter notebooks in notebooks/ from the Python sources.

    python scripts/build_notebooks.py           # (re)write the notebooks
    python scripts/build_notebooks.py --check   # exit 1 if they are out of date

The notebooks are SELF-CONTAINED: every module of src/ (plus the config files
and helper scripts) is embedded as a `%%writefile` cell, so a notebook runs
on Colab / Kaggle / Jupyter without cloning the repository. Because they are
generated from src/, the notebooks and the command-line pipeline can never
drift apart - edit src/, then re-run this script.

  notebooks/01_train.ipynb             setup -> code -> config -> training stages
                                       -> holdout evaluation + error analysis
  notebooks/02_predict_and_submit.ipynb  setup -> code -> config -> prediction
                                       stages -> validation -> submission.zip
"""
import argparse
import json
import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_ORDER = ["__init__", "config", "utils", "schema", "data_io", "normalization", "address_parser",
                "preprocess", "labels", "splits", "metrics", "blocking", "features", "train_lightgbm",
                "calibration", "scoring", "decision", "graph_cleanup", "train_biencoder", "train_crossencoder",
                "inference", "submission", "error_analysis", "training", "prediction"]


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def writefile(rel: str) -> nbformat.NotebookNode:
    return new_code_cell(f"%%writefile {rel}\n" + _read(rel).rstrip("\n") + "\n")


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text.strip("\n"))


def module_cells() -> list:
    present = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "src")) if f.endswith(".py"))
    missing = [m for m in present if m not in MODULE_ORDER]
    if missing:
        raise SystemExit(f"add {missing} to MODULE_ORDER in scripts/build_notebooks.py")
    return [writefile(f"src/{m}.py") for m in MODULE_ORDER]


# ----------------------------------------------------------------- shared
SETUP_WORKDIR = '''
import os, sys

# Everything (code, checkpoints, models, outputs) lives in WORK_DIR.
# On Colab, put it on Google Drive so a crashed / restarted runtime can resume:
#   from google.colab import drive; drive.mount("/content/drive")
#   WORK_DIR = "/content/drive/MyDrive/amazon_er"
if os.path.isdir("/kaggle/working"):
    WORK_DIR = "/kaggle/working/amazon_er"
elif os.path.isdir("/content"):
    WORK_DIR = "/content/amazon_er"
else:
    WORK_DIR = os.path.abspath("amazon_er_work")
WORK_DIR = os.environ.get("ER_WORK_DIR", WORK_DIR)   # (override used by the automated notebook test)

os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
for d in ("src", "configs", "scripts"):
    os.makedirs(d, exist_ok=True)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)
print("working directory:", os.getcwd())
'''

SETUP_DEPS = '''
import importlib.util, subprocess, sys

INSTALL_FULL_MODE_DEPS = False   # True for mode="full" (torch, transformers, faiss; GPU recommended)

core = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
        "lightgbm": "lightgbm", "rapidfuzz": "rapidfuzz>=3.6", "jellyfish": "jellyfish",
        "networkx": "networkx", "yaml": "pyyaml", "unidecode": "unidecode"}
full = {"torch": "torch", "transformers": "transformers", "tokenizers": "tokenizers",
        "sentencepiece": "sentencepiece", "google.protobuf": "protobuf", "faiss": "faiss-cpu"}
wanted = {**core, **(full if INSTALL_FULL_MODE_DEPS else {})}
missing = [pkg for mod, pkg in wanted.items() if importlib.util.find_spec(mod.split(".")[0]) is None]
if missing:
    print("installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("all required packages are installed")
'''

IMPORT_CHECK = '''
import importlib
importlib.invalidate_caches()
if "src" in sys.modules:
    print("NOTE: src was imported before - restart the kernel if you edited a module cell")
import src, src.training, src.prediction
from src.utils import mem_str
print("pipeline", src.__version__, "|", mem_str())
try:
    import psutil
    print(f"available RAM: {psutil.virtual_memory().available / 2**30:.1f} GB")
except ImportError:
    pass
'''

CONFIG_CELL = '''
import subprocess, yaml
from src.config import deep_merge, load_config

RUN_NAME = "baseline_v1"      # artefacts -> models/<RUN_NAME>/, outputs -> outputs/<RUN_NAME>/
MODE = "baseline"             # "full" adds the SupCon bi-encoder + DeBERTa-v3 cross-encoder (GPU, local weights)
LOW_MEMORY = True             # use configs/low_memory.yaml (recommended for ~12-16 GB RAM)
USE_SYNTHETIC_DEMO = False    # True -> generate a small fake dataset and run the whole notebook on it

# ---- the real challenge files (tab-separated) -----------------------------
DATA_DIR = "/kaggle/input/amazon-ml-challenge-2026"     # <- change to where your TSVs are
SETTINGS = {
    "run_name": RUN_NAME,
    "mode": MODE,
    "data": {
        "train": {"source1": f"{DATA_DIR}/train/source1.tsv", "source2": f"{DATA_DIR}/train/source2.tsv",
                  "source3": f"{DATA_DIR}/train/source3.tsv", "labels": f"{DATA_DIR}/train/labels.tsv"},
        "test": {"source1": f"{DATA_DIR}/test/source1.tsv", "source2": f"{DATA_DIR}/test/source2.tsv",
                 "source3": f"{DATA_DIR}/test/source3.tsv"},
    },
    # column names of the real files - check them with the "Inspect the data" cell
    "schema": {
        "source1": {"id": "id", "name": "name", "address": ["address"]},
        "source2": {"id": "id", "name": "name", "address": ["address"]},
        "source3": {"id": "id", "name": "name", "address": ["address"]},
        "labels": {"source1_id": "source1_id", "match_columns": ["matches"]},
    },
    # full mode only: local folders holding the pre-trained weights
    "biencoder": {"pretrained": "models/pretrained/all-MiniLM-L6-v2"},
    "crossencoder": {"pretrained": "models/pretrained/deberta-v3-base"},
}

if USE_SYNTHETIC_DEMO:
    if not os.path.exists("data/synthetic/train/labels.tsv"):
        subprocess.check_call([sys.executable, "scripts/make_synthetic_data.py", "--out", "data/synthetic",
                               "--n-train", "800", "--n-test", "400"])
    d = "data/synthetic"
    SETTINGS["data"] = {
        "train": {**{f"source{i}": f"{d}/train/source{i}.tsv" for i in (1, 2, 3)}, "labels": f"{d}/train/labels.tsv"},
        "test": {f"source{i}": f"{d}/test/source{i}.tsv" for i in (1, 2, 3)},
    }
    SETTINGS["schema"] = {**{f"source{i}": {"id": "record_id", "name": "business_name", "address": ["full_address"]}
                             for i in (1, 2, 3)},
                          "labels": {"source1_id": "source1_id", "match_columns": ["matched_ids"]}}

base_config = "configs/low_memory.yaml" if LOW_MEMORY else "config.yaml"
cfg = deep_merge(load_config(base_config), SETTINGS)
cfg["_config_path"] = os.path.abspath(base_config)
'''

SAVE_CONFIG = '''
# save the exact configuration: the prediction notebook (and the CLI) reuse it
with open(f"configs/{RUN_NAME}.yaml", "w") as fh:
    yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, fh, sort_keys=False)
print(f"run={cfg['run_name']}  mode={cfg['mode']}  base={base_config}  saved configs/{RUN_NAME}.yaml")
print("candidate cap per source:", cfg["blocking"]["max_candidates_per_source"],
      "| feature chunk:", cfg["features"]["chunk_pairs"], "| workers:", cfg["n_jobs"])
'''

INSPECT_CELL = '''
# Look at the real files before training: row counts, columns, examples.
from src.data_io import read_tsv
for split in ("train", "test"):
    for key, path in cfg["data"][split].items():
        if not os.path.exists(path):
            print(f"{split}.{key}: NOT FOUND -> {path}")
            continue
        df = read_tsv(path, cfg)
        print(f"{split}.{key}: {len(df):,} rows | columns {list(df.columns)}")
        display(df.head(3))
'''


def header(title: str, body: str) -> nbformat.NotebookNode:
    return md(f"# {title}\n\n{body}")


RUN_NOTES = '''
**How to run:** run the cells top to bottom. Section 1 writes the pipeline code into `WORK_DIR/src/`, one module
per cell. Section 2 is the only place you need to edit (data paths, column names, run name, mode).

**If the kernel dies or runs out of memory:** every stage saves its results to
`outputs/<RUN_NAME>/.../checkpoints/`. Restart the kernel, re-run the Setup and Config cells, and continue from
the first stage that did not finish. Nothing before it has to be recomputed. Each stage also prints the current
and peak memory (`RSS ... (peak ...)`), so you can see which one is tight.

**Memory:** `LOW_MEMORY = True` uses `configs/low_memory.yaml`, which sets a smaller candidate cap, smaller chunks,
fewer worker processes and negative down-sampling for LightGBM. The pipeline streams the pairwise features to a
disk-backed matrix, so they never need to fit in RAM all at once.
'''


def setup_cells() -> list:
    return [
        md("## 0. Setup"),
        code(SETUP_WORKDIR),
        code(SETUP_DEPS),
        md("## 1. Pipeline code\n\nEach cell below writes one module of the pipeline to `src/` "
           "(identical to the repository's `src/`). Run them all. If you edit one, restart the kernel before "
           "re-running the later cells."),
        *module_cells(),
        md("### Configuration files and helper scripts"),
        writefile("config.yaml"),
        writefile("configs/low_memory.yaml"),
        writefile("scripts/make_synthetic_data.py"),
        code(IMPORT_CHECK),
    ]


# ------------------------------------------------------------- notebook 1
def train_notebook() -> nbformat.NotebookNode:
    stage_md = {
        "prepare": "Load the TSVs (explicit tab separator), validate columns, normalise names (legal suffixes kept as "
                   "a feature), parse addresses offline, resolve labels, split Source-1 entities into fit / calib / "
                   "holdout.",
        "biencoder": "**Full mode only.** Train the supervised-contrastive bi-encoder used for semantic blocking. "
                     "In baseline mode this cell does nothing.",
        "blocking": "Hybrid blocking. Source 2 and Source 3 are processed one after the other and each is "
                    "finalised (union + cap) before the next starts. The report below shows pair completeness "
                    "(recall ceiling), reduction ratio and what each method contributes.",
        "features": "Pairwise features, computed in chunks and streamed to `checkpoints/features.npy` "
                    "(a disk-backed matrix), so the full feature table never has to fit in RAM.",
        "lightgbm": "LightGBM on the fit split (early stopping on fit_val), block-wise scoring of every candidate, "
                    "isotonic calibration on the calib split.",
        "crossencoder": "**Full mode only.** Ditto-style DeBERTa cross-encoder with focal loss and MixDA, applied to "
                        "the uncertain pairs (cascade). In baseline mode this cell does nothing.",
        "decide": "Blend selection, singleton model, graph cleanup and the Macro-F0.5 decision search on calib; then "
                  "an unbiased evaluation on the untouched holdout split plus error analysis.",
    }
    cells = [
        header("Amazon ML Challenge 2026 — Entity Resolution: TRAINING",
               "Trains the pipeline on the labelled training set and reports the holdout Macro F0.5.\n\n" + RUN_NOTES
               + "\nNext: `02_predict_and_submit.ipynb` (same `WORK_DIR` and `RUN_NAME`)."),
        *setup_cells(),
        writefile("scripts/download_pretrained.py"),
        md("## 2. Configuration (edit this cell)"),
        code(CONFIG_CELL + SAVE_CONFIG),
        md("### Inspect the data (optional)"),
        code(INSPECT_CELL),
        md("## 3. Training stages\n\nOne cell per stage. After a restart: re-run sections 0–2, then continue with "
           "the first unfinished stage, or run the *resume* cell at the end of this section."),
        code("from src.training import STAGES, load_report, run_stage, run_training, stage_done\n"
             "import json, pandas as pd\n"
             "print({s: stage_done(cfg, s) for s in STAGES})"),
    ]
    for st in ["prepare", "biencoder", "blocking", "features", "lightgbm", "crossencoder", "decide"]:
        cells.append(md(f"### Stage: `{st}`\n\n{stage_md[st]}"))
        cells.append(code(f'run_stage(cfg, "{st}")'))
        if st == "blocking":
            cells.append(code(
                'brep = json.load(open(f"outputs/{RUN_NAME}/train/blocking_report.json"))\n'
                'keys = ["candidate_pairs", "possible_pairs", "reduction_ratio", "pair_completeness_before_cap",\n'
                '        "pair_completeness", "pair_completeness_s2", "pair_completeness_s3", "true_pairs_lost",\n'
                '        "oracle_macro_f0.5_given_blocking"]\n'
                'display(pd.Series({k: brep.get(k) for k in keys}, name="blocking"))\n'
                'display(pd.DataFrame(brep.get("per_method", {})).T)'))
    cells += [
        md("### Resume (alternative to the stage cells)\n\nRuns every stage that has not finished yet."),
        code("# run_training(cfg, resume=True)"),
        md("## 4. Results"),
        code('rep = load_report(cfg)\n'
             'ev = rep["evaluation"]\n'
             'display(pd.DataFrame({k: ev[k] for k in ("calib", "holdout") if k in ev})\n'
             '        .drop(index=["graph", "pairwise_on_candidates"], errors="ignore"))\n'
             'print("decision parameters:", rep["decision"]["params"])\n'
             'print("ablation (LightGBM alone):", ev.get("ablation"))\n'
             'print("timings (s):", rep.get("timings_sec"))'),
        code('display(pd.read_csv(f"models/{RUN_NAME}/feature_importance.tsv", sep="\\t").head(20))'),
        md("### Error analysis (holdout)\n\nThe files in `outputs/<RUN_NAME>/train/error_analysis_holdout/` hold "
           "every false positive and false negative, with raw and normalised names/addresses, key features and "
           "scores."),
        code('ea = f"outputs/{RUN_NAME}/train/error_analysis_holdout"\n'
             'print(json.load(open(f"{ea}/error_summary.json")))\n'
             'for name in ("top_confident_errors", "false_negatives"):\n'
             '    path = f"{ea}/{name}.tsv"\n'
             '    if os.path.getsize(path) > 1:\n'
             '        print(name); display(pd.read_csv(path, sep="\\t").head(10))'),
    ]
    return _nb(cells)


# ------------------------------------------------------------- notebook 2
def predict_notebook() -> nbformat.NotebookNode:
    load_cfg = CONFIG_CELL + '''
# Use the EXACT configuration the training notebook saved for this RUN_NAME
# (the settings above are only a fallback when it does not exist).
if os.path.exists(f"configs/{RUN_NAME}.yaml"):
    cfg = load_config(f"configs/{RUN_NAME}.yaml")
    print(f"loaded configs/{RUN_NAME}.yaml (saved by 01_train.ipynb)")
else:
    print(f"WARNING: configs/{RUN_NAME}.yaml not found - using the settings above; they must match training")
assert os.path.exists(f"models/{RUN_NAME}/pipeline.json"), "train first (01_train.ipynb) with the same RUN_NAME"
print(f"run={cfg['run_name']}  mode={cfg['mode']}  test source1={cfg['data']['test']['source1']}")
'''
    stage_md = {
        "prepare": "Load and normalise the test Source 1 / 2 / 3 files (no labels exist or are used).",
        "blocking": "Hybrid blocking; writes `candidate_pairs.tsv` **before** any model scores a pair.",
        "features": "Pairwise features streamed to the disk-backed matrix.",
        "score": "LightGBM (block-wise) → calibration → [cascade → cross-encoder] → blend → graph cleanup → "
                 "decision rules tuned on calib → `matching_results.tsv` (one row per Source-1 entity, empty for "
                 "predicted singletons) → format validation.",
    }
    cells = [
        header("Amazon ML Challenge 2026 — Entity Resolution: PREDICT & SUBMIT",
               "Uses the models trained by `01_train.ipynb` (same `WORK_DIR` and `RUN_NAME`) to produce "
               "`candidate_pairs.tsv` and `matching_results.tsv`, validates them, and builds the submission "
               "archive.\n\n" + RUN_NOTES),
        *setup_cells(),
        writefile("METHODOLOGY.md"),
        md("## 2. Configuration (same settings as the training notebook)"),
        code(load_cfg),
        md("## 3. Prediction stages\n\nAfter a restart: re-run sections 0–2, then continue with the first unfinished "
           "stage."),
        code("from src.prediction import PREDICT_STAGES, PREDICT_FUNCS\n"
             "from src.utils import release_memory\n"
             "import json, pandas as pd\n"
             "def run_predict_stage(name):\n"
             "    out = PREDICT_FUNCS[name](cfg, 'test')\n"
             "    release_memory()\n"
             "    return out"),
    ]
    for st in ["prepare", "blocking", "features", "score"]:
        cells.append(md(f"### Stage: `{st}`\n\n{stage_md[st]}"))
        cells.append(code(f'run_predict_stage("{st}")' if st != "score" else 'report = run_predict_stage("score")\n'
                          'print("validation:", report["validation"])'))
    cells += [
        md("## 4. Outputs"),
        code('out_dir = f"outputs/{RUN_NAME}/test"\n'
             'res = pd.read_csv(f"{out_dir}/{cfg[\'submission\'][\'matching_results\']}", sep="\\t", dtype=str,\n'
             '                  keep_default_na=False)\n'
             'print(len(res), "Source-1 rows;", (res.iloc[:, 1:] == "").all(axis=1).sum(), "predicted singletons")\n'
             'display(res.head(10))\n'
             'display(pd.read_csv(f"{out_dir}/{cfg[\'submission\'][\'candidate_pairs\']}", sep="\\t", nrows=5))'),
        md("## 5. Submission archive\n\nContains `matching_results.tsv`, `candidate_pairs.tsv`, the methodology "
           "document, the reports and the pipeline code (`src/`, configs, scripts). Also run the organisers' own "
           "validation script before uploading."),
        code('from src.submission import package_submission\n'
             'files = {\n'
             '    cfg["submission"]["matching_results"]: f"{out_dir}/{cfg[\'submission\'][\'matching_results\']}",\n'
             '    cfg["submission"]["candidate_pairs"]: f"{out_dir}/{cfg[\'submission\'][\'candidate_pairs\']}",\n'
             '    "METHODOLOGY.md": "METHODOLOGY.md",\n'
             '}\n'
             'for name, path in [("reports/train_training_report.json", f"outputs/{RUN_NAME}/train/training_report.json"),\n'
             '                   ("reports/train_blocking_report.json", f"outputs/{RUN_NAME}/train/blocking_report.json"),\n'
             '                   ("reports/test_prediction_report.json", f"{out_dir}/prediction_report.json")]:\n'
             '    if os.path.exists(path):\n'
             '        files[name] = path\n'
             'archive = f"outputs/{RUN_NAME}/{cfg[\'submission\'][\'archive_name\']}"\n'
             'res_zip = package_submission(".", files, archive)\n'
             'print(archive, len(res_zip["files"]), "files")\n'
             '# Colab: from google.colab import files as colab_files; colab_files.download(archive)'),
    ]
    return _nb(cells)


def _nb(cells) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def _canonical(nb) -> str:
    return json.dumps(nbformat.from_dict(nb), sort_keys=True, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify the notebooks are up to date")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "notebooks")
    os.makedirs(out_dir, exist_ok=True)
    stale = []
    for name, builder in (("01_train.ipynb", train_notebook), ("02_predict_and_submit.ipynb", predict_notebook)):
        nb = builder()
        path = os.path.join(out_dir, name)
        if args.check:
            if not os.path.exists(path):
                stale.append(name)
                continue
            with open(path, encoding="utf-8") as fh:
                current = nbformat.read(fh, as_version=4)
            strip = lambda n: [(c["cell_type"], c["source"]) for c in n["cells"]]
            if strip(current) != strip(nb):
                stale.append(name)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                nbformat.write(nb, fh)
            print(f"wrote {os.path.relpath(path, ROOT)} ({len(nb.cells)} cells)")
    if args.check:
        if stale:
            print("out of date:", stale, "-> run python scripts/build_notebooks.py")
            sys.exit(1)
        print("notebooks are up to date")


if __name__ == "__main__":
    main()
