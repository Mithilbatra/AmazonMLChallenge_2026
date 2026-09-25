#!/usr/bin/env python
"""Create the single end-of-challenge archive (transcript deliverable 2):
final matches + candidate/unscored pairs + complete reproducible pipeline +
methodology document.

    python package_submission.py --config config.yaml [--include-models]
"""
import argparse
import os

from src.config import get, load_config, model_dir, output_dir
from src.submission import package_submission


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--include-models", action="store_true",
                    help="also bundle models/<run_name>/ (can be large for the full mode)")
    ap.add_argument("--out", help="archive path (default outputs/<run>/<archive_name>)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    test_dir = os.path.join(output_dir(cfg), "test")
    files = {
        get(cfg, "submission.matching_results", "matching_results.tsv"):
            os.path.join(test_dir, get(cfg, "submission.matching_results", "matching_results.tsv")),
        get(cfg, "submission.candidate_pairs", "candidate_pairs.tsv"):
            os.path.join(test_dir, get(cfg, "submission.candidate_pairs", "candidate_pairs.tsv")),
        get(cfg, "submission.methodology_doc", "METHODOLOGY.md"): get(cfg, "submission.methodology_doc", "METHODOLOGY.md"),
    }
    for name in ("training_report.json", "blocking_report.json"):
        path = os.path.join(output_dir(cfg), "train", name)
        if os.path.exists(path):
            files[f"reports/train_{name}"] = path
    for name in ("prediction_report.json", "blocking_report.json"):
        path = os.path.join(test_dir, name)
        if os.path.exists(path):
            files[f"reports/test_{name}"] = path
    archive = args.out or os.path.join(output_dir(cfg), get(cfg, "submission.archive_name", "submission.zip"))
    res = package_submission(".", files, archive, model_dir(cfg) if args.include_models else None)
    print(f"Archive: {res['archive']} ({len(res['files'])} files)")
    for f in list(files):
        print("  +", f)


if __name__ == "__main__":
    main()
