#!/usr/bin/env python
"""Inspect the configured challenge files BEFORE training: existence, row
counts, columns, sample rows, empty-value rates, label list format. Use the
output to fill in the `data:` and `schema:` sections of config.yaml.

    python inspect_data.py --config config.yaml
"""
import argparse
import os

from src.config import get, load_config
from src.data_io import detect_list_format, parse_id_list, read_tsv


def show(path, cfg, label=False):
    print("=" * 100)
    print(path)
    if not path or not os.path.exists(path):
        print("  -> NOT FOUND")
        return
    df = read_tsv(path, cfg)
    print(f"  rows={len(df)}  columns={list(df.columns)}")
    for c in df.columns:
        empty = (df[c].astype(str).str.strip() == "").mean()
        print(f"    {c!r:30} empty={empty:6.1%}  unique={df[c].nunique():>9}  e.g. {df[c].head(3).tolist()}")
    if label:
        for c in df.columns[1:]:
            fmt = detect_list_format(df[c])
            sizes = df[c].map(lambda x: len(parse_id_list(x)))
            print(f"    list column {c!r}: format={fmt}  empty lists={int((sizes == 0).sum())}  "
                  f"mean size={sizes.mean():.2f}  max={sizes.max()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            show(get(cfg, f"data.{split}.source{s}"), cfg)
    show(get(cfg, "data.train.labels"), cfg, label=True)
    if get(cfg, "data.sample_submission"):
        show(get(cfg, "data.sample_submission"), cfg, label=True)
    print("=" * 100)
    print("Configured schema:", cfg.get("schema"))


if __name__ == "__main__":
    main()
