#!/usr/bin/env python
"""Generate candidate_pairs.tsv and matching_results.tsv for the test set.

    python predict.py --config config.yaml
"""
import argparse

from src.config import load_config
from src.prediction import PREDICT_STAGES, run_prediction


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", help="data.<split> section to predict (default: test)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--stages", help=f"comma-separated subset of {','.join(PREDICT_STAGES)}")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    stages = [x.strip() for x in args.stages.split(",")] if args.stages else None
    report = run_prediction(cfg, args.split, stages)
    if "outputs" not in report:
        print(report)
        return
    print(f"\nmatching_results: {report['outputs']['matching_results']}")
    print(f"candidate_pairs : {report['outputs']['candidate_pairs']}")
    print(f"predicted matches {report['predicted_matches']}, predicted singletons {report['predicted_singletons']}")
    if not report["validation"]["ok"]:
        raise SystemExit(f"FORMAT ERRORS: {report['validation']['errors']}")


if __name__ == "__main__":
    main()
