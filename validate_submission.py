#!/usr/bin/env python
"""Local format check of matching_results.tsv against the test Source-1/2/3
files. The organisers' own validation script (mentioned in the transcript)
must still be run before uploading - this is a complementary check.

    python validate_submission.py --config config.yaml [--file path/to/matching_results.tsv]
"""
import argparse
import json
import os

from src.config import get, load_config, output_dir
from src.data_io import load_sources, read_sample_submission_header
from src.prediction import _list_format_from_labels
from src.schema import label_schema, submission_schema
from src.submission import validate_submission


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--file")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    path = args.file or os.path.join(output_dir(cfg), args.split,
                                     get(cfg, "submission.matching_results", "matching_results.tsv"))
    sources, _ = load_sources(cfg, args.split)
    sch = submission_schema(cfg, label_schema(cfg), _list_format_from_labels(cfg), read_sample_submission_header(cfg))
    res = validate_submission(path, sources[1]["id"].tolist(), set(sources[2]["id"]), set(sources[3]["id"]), sch)
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
