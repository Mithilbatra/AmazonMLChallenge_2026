#!/usr/bin/env python
"""ONE-TIME download of pre-trained transformer weights for full mode.

Run this ONCE, on a machine with internet access, BEFORE training. It fetches
generic pre-trained language-model weights only (no business data, no
lookup service). Training and prediction then load the weights from local
disk with `local_files_only=True`; nothing is fetched while matching.

Compliance note: the transcript forbids "external databases, APIs, and
lookups". Generic pre-trained LM weights are neither a database of
businesses nor a lookup service, and the PDF architecture relies on them
(DeBERTa-v3, MiniLM). If the organisers consider pre-trained weights
out of bounds, use `mode: baseline`, which uses only the provided data.

    python scripts/download_pretrained.py --out models/pretrained
"""
import argparse
import os

MODELS = {
    "deberta-v3-base": "microsoft/deberta-v3-base",                    # cross-encoder (PDF default)
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",      # bi-encoder (PDF: distilled MiniLM)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/pretrained")
    ap.add_argument("--only", nargs="*", choices=list(MODELS), help="subset of models")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    for local, repo in MODELS.items():
        if args.only and local not in args.only:
            continue
        dest = os.path.join(args.out, local)
        print(f"downloading {repo} -> {dest}")
        snapshot_download(repo_id=repo, local_dir=dest,
                          allow_patterns=["*.json", "*.model", "*.txt", "*.safetensors", "*.bin", "spm*"],
                          ignore_patterns=["*onnx*", "*openvino*", "*.h5", "*.msgpack", "*.ot"])
    print("done - set biencoder.pretrained / crossencoder.pretrained in config.yaml to these folders")


if __name__ == "__main__":
    main()
