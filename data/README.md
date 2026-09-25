# Data

Place the official challenge files here (tab-separated). Example layout:

```
data/train/source1.tsv   data/train/source2.tsv   data/train/source3.tsv   data/train/labels.tsv
data/test/source1.tsv    data/test/source2.tsv    data/test/source3.tsv
```

The real file and column names are not given in the challenge transcript: set
them in `config.yaml` (`data:` and `schema:`) after running

    python inspect_data.py --config config.yaml

A synthetic demo dataset with the same structure can be generated with

    python scripts/make_synthetic_data.py --out data/synthetic
