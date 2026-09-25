#!/usr/bin/env python
"""Train the entity-resolution pipeline on the labelled training set.

    python train.py --config config.yaml [--set mode=full --set run_name=full_v1]
"""
import argparse

from src.config import load_config
from src.training import run_training


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set mode=full")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    report = run_training(cfg)
    ev = report.get("evaluation", {}).get("holdout")
    if ev:
        print(f"\nHoldout Macro F0.5 = {ev['macro_f0.5']:.5f} "
              f"(pair precision {ev['pair_precision']:.4f}, pair recall {ev['pair_recall']:.4f})")


if __name__ == "__main__":
    main()
