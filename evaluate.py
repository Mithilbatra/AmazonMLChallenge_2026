#!/usr/bin/env python
"""Evaluation and error analysis.

1) Re-evaluate the trained pipeline on the labelled calib / holdout splits
   (uses outputs/<run>/train/eval_state.pkl written by train.py):

       python evaluate.py --config config.yaml [--split holdout|calib]

   Optionally try other decision parameters without retraining:

       python evaluate.py --config config.yaml --decision threshold_s2=0.7 --decision threshold_s3=0.75

2) Score ANY matching_results-style file against ANY label file with the
   challenge metric (entity-level Macro F0.5):

       python evaluate.py --config config.yaml --predictions outputs/x/test/matching_results.tsv \
                          --labels path/to/labels.tsv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from src.config import get, load_config, model_dir, output_dir
from src.data_io import load_labels
from src.decision import DecisionParams
from src.error_analysis import run_error_analysis
from src.inference import TrainedPipeline, decide
from src.metrics import macro_fbeta_sets, pairwise_prf, summarize
from src.schema import label_schema
from src.utils import load_pickle, save_json, setup_logging


def score_files(cfg, pred_path, labels_path):
    """Official-style scoring: per Source-1 id, predicted id set vs true id set."""
    lab, _ = load_labels(labels_path, cfg)
    pcfg = dict(cfg)
    pred, _ = load_labels(pred_path, pcfg)
    per_source = bool(label_schema(cfg).column_sources)

    def keys(row, cols_sources):
        if not per_source:
            return set(row["all_ids"])
        return {f"{cols_sources[c]}:{i}" for c in cols_sources for i in row[c]}

    cs = label_schema(cfg).column_sources or {}
    true = {r["s1_id"]: keys(r, cs) for _, r in lab.iterrows()}
    predd = {r["s1_id"]: keys(r, cs) for _, r in pred.iterrows()}
    missing = set(true) - set(predd)
    res = macro_fbeta_sets(predd, true, list(true.keys()), float(get(cfg, "decision.beta", 0.5)))
    res["source1_ids_missing_in_predictions(scored as empty)"] = len(missing)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", default="holdout", choices=["holdout", "calib"])
    ap.add_argument("--decision", action="append", default=[], metavar="PARAM=VALUE",
                    help="override a decision parameter for this evaluation only")
    ap.add_argument("--predictions")
    ap.add_argument("--labels")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    if args.predictions:
        if not args.labels:
            raise SystemExit("--predictions requires --labels")
        res = score_files(cfg, args.predictions, args.labels)
        print(json.dumps(res, indent=2))
        return

    odir = os.path.join(output_dir(cfg), "train")
    setup_logging(os.path.join(odir, "evaluate.log"))
    state = load_pickle(os.path.join(odir, "eval_state.pkl"))
    pipe = TrainedPipeline(model_dir(cfg), cfg)
    params = pipe.decision.to_dict()
    for item in args.decision:
        k, val = item.split("=", 1)
        params[k] = yaml.safe_load(val)
    params = DecisionParams.from_dict(params)
    split, n_true = state["split"], state["n_true"]
    ents = np.where((split == args.split) & state["labeled"])[0]
    sc = state["scored"]
    sc = sc[sc["split"] == args.split].copy()
    sel, info = decide(sc, ents, pipe, params, extra_cols=None)
    sc["selected"] = sel
    pos = pd.Series(np.arange(len(ents)), index=ents)
    e = pos.reindex(sc["s1_idx"].to_numpy()).to_numpy()
    tp = np.bincount(e[sel & sc["is_true"].to_numpy(bool)], minlength=len(ents))
    npred = np.bincount(e[sel], minlength=len(ents))
    metrics = summarize(tp, npred, n_true[ents], float(get(cfg, "decision.beta", 0.5)))
    metrics["pairwise_on_candidates"] = pairwise_prf(sel, sc["is_true"].to_numpy(bool))
    metrics["decision"] = params.to_dict()
    metrics["graph"] = info["graph"]
    err_dir = os.path.join(odir, f"error_analysis_{args.split}_eval")
    metrics["errors"] = run_error_analysis(sc, state["s1"], state["v"], state["true_pairs"], n_true, ents,
                                           min(params.threshold_s2, params.threshold_s3), err_dir,
                                           int(get(cfg, "evaluation.error_analysis_top_n", 200)))
    save_json(metrics, os.path.join(odir, f"evaluation_{args.split}.json"))
    print(json.dumps({k: val for k, val in metrics.items() if k != "graph"}, indent=2, default=str))
    print(f"\nError analysis files: {err_dir}")


if __name__ == "__main__":
    main()
