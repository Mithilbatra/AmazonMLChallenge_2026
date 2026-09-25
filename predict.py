#!/usr/bin/env python
"""Generate candidate_pairs.tsv and matching_results.tsv for the test set.

    python predict.py --config config.yaml
"""
import argparse

from src.config import load_config
from src.prediction import run_prediction


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", help="data.<split> section to predict (default: test)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    report = run_prediction(cfg, args.split)
    print(f"\nmatching_results: {report['outputs']['matching_results']}")
    print(f"candidate_pairs : {report['outputs']['candidate_pairs']}")
    print(f"predicted matches {report['predicted_matches']}, predicted singletons {report['predicted_singletons']}")
    if not report["validation"]["ok"]:
        raise SystemExit(f"FORMAT ERRORS: {report['validation']['errors']}")


if __name__ == "__main__":
    main()
