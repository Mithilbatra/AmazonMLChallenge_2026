# Amazon ML Challenge 2026: Business Entity Resolution

This repository is a runnable implementation of the architecture in *Entity Resolution System Architecture.pdf*. It is built to the official problem statement (*ML_Challenge_2026_Video_Transcript.docx*).

For every **Source 1** business, the pipeline finds all matching **Source 2** and **Source 3** records using only the business name and address. It writes:

- **`matching_results.tsv`**: one row per Source-1 entity with the matching ids; empty for singletons.
- **`candidate_pairs.tsv`**: the blocking candidates, written before any model scores them.

Everything runs **offline**. No external database, API, geocoder or lookup is used, as the challenge rules require.

> `METHODOLOGY.md` is the methodology document that goes in the submission archive. It covers every stage, the leakage controls, the design rationale, the assumptions, and the compliance audit.

---

## 1. Quick start (synthetic demo, about 1 minute on CPU)

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py --out data/synthetic        # a fake dataset with the challenge's structure
python train.py    --config configs/synthetic_baseline.yaml
python predict.py  --config configs/synthetic_baseline.yaml
python evaluate.py --config configs/synthetic_baseline.yaml        # holdout metrics + error analysis
python evaluate.py --config configs/synthetic_baseline.yaml \
       --predictions outputs/synthetic_baseline/test/matching_results.tsv \
       --labels data/synthetic/test_labels_HIDDEN.tsv                # score test predictions (demo only)
python -m pytest -q                                                 # 50 tests, including an end-to-end run
```

Full mode on the demo data uses tiny, randomly initialised local transformers, so no download is needed:

```bash
pip install -r requirements-full.txt
python scripts/build_local_tiny_transformer.py --config configs/synthetic_full.yaml
python train.py   --config configs/synthetic_full.yaml
python predict.py --config configs/synthetic_full.yaml
```

## 2. Running on the real challenge data (execution order)

```bash
# 0. install
pip install -r requirements.txt                 # baseline
pip install -r requirements-full.txt            # full architecture (GPU recommended)

# 1. put the TSVs under data/ and look at them
python inspect_data.py --config config.yaml
#    -> edit config.yaml: data.* paths, schema.* column names, label list columns
#       (the transcript does not give file or column names)

# 2. baseline: fast, easy to debug
python train.py    --config config.yaml                   # writes models/default/, outputs/default/train/
python evaluate.py --config config.yaml                   # holdout Macro F0.5 + error analysis
python predict.py  --config config.yaml                   # outputs/default/test/{candidate_pairs,matching_results}.tsv
python validate_submission.py --config config.yaml        # local format check (also run the organisers' script)

# 3. full architecture (one-time weight download on a machine with internet, then offline)
python scripts/download_pretrained.py --out models/pretrained
python train.py   --config config.yaml --set mode=full --set run_name=full
python evaluate.py --config config.yaml --set mode=full --set run_name=full
python predict.py --config config.yaml --set mode=full --set run_name=full

# 4. the single end-of-challenge archive: matches + candidate pairs + pipeline + methodology
python package_submission.py --config config.yaml [--set run_name=full] [--include-models]
```

Upload `outputs/<run>/test/matching_results.tsv` to the leaderboard. At the end, submit `outputs/<run>/submission.zip`. Always compare the baseline and full holdout scores in `outputs/<run>/train/training_report.json` before choosing which run to submit.

## 3. Architecture → code

| Architecture component ([A] = PDF) | Implementation |
|---|---|
| TSV ingestion, schema, labels | `src/data_io.py`, `src/schema.py`: explicit `\t`, schema validation, null / duplicate reports, robust list parsing |
| Name normalisation (NFKD, casefold, punctuation) | `src/normalization.py` `clean_text` |
| Legal suffix abstraction (longest-first, kept as a feature) | `NameNormalizer.extract_suffix`, 46 legal forms plus families |
| Offline address parsing (libpostal) | `src/address_parser.py`: libpostal when installed, otherwise a multi-region rule parser (US / IN / UK / DE / CA / AU) |
| Syntactic blocking (postcode, 4-char prefix, phonetic) | `src/blocking.py`: `name_prefix`, `postcode`, `postcode_phonetic`, `city_state`, `phonetic_name`, `address_key` |
| LSH / MinHash / TF-IDF | `minhash_block` (vectorised MinHash + banding), `sparse_topk` (char / word TF-IDF) |
| SC-Block: SupCon bi-encoder + FAISS | `src/train_biencoder.py` (`supcon_loss`, hard batching, `SemanticRetriever`) |
| Pair completeness / reduction ratio | `blocking_report` → `outputs/<run>/train/blocking_report.json` |
| LightGBM structural matcher (JW, Soft-TFIDF, component flags) | `src/features.py` (103 features), `src/train_lightgbm.py` (gain importance) |
| Ditto cross-encoder, DeBERTa-v3, focal loss, MixDA, domain tags, TF-IDF summarisation | `src/train_crossencoder.py` |
| Cascade | `src/scoring.py` `cascade_mask` |
| Ensemble / blending | `src/scoring.py` `Blender` (weighted / rank / stack), chosen on calib in `src/training.py` |
| Isotonic calibration | `src/calibration.py` (isotonic vs Platt, both reported) |
| Macro F0.5 threshold optimisation, singleton handling | `src/decision.py` (`DecisionTable`, `search_decision`, singleton model) |
| Graph cleanup (bridges, edge betweenness) | `src/graph_cleanup.py` (`off` / `exclusive` / `betweenness`, chosen on validation) |
| Source-1-centric output | `src/submission.py` `build_matching_results` |
| Error analysis | `src/error_analysis.py`, `evaluate.py` |

## 4. Project structure

```text
.
├── README.md                     this file
├── METHODOLOGY.md                methodology document (goes into the submission archive)
├── config.yaml                   single documented config (all parameters)
├── configs/
│   ├── synthetic_baseline.yaml   demo config (inherits config.yaml via `base:`)
│   └── synthetic_full.yaml       demo full mode with tiny local transformers
├── requirements.txt              core / baseline dependencies
├── requirements-full.txt         + torch, transformers, faiss
├── train.py  predict.py  evaluate.py  package_submission.py  validate_submission.py  inspect_data.py
├── src/
│   ├── config.py  utils.py  schema.py  data_io.py
│   ├── normalization.py  address_parser.py  preprocess.py
│   ├── labels.py  splits.py  blocking.py  features.py
│   ├── train_lightgbm.py  train_biencoder.py  train_crossencoder.py
│   ├── calibration.py  scoring.py  decision.py  graph_cleanup.py  metrics.py
│   ├── training.py  inference.py  prediction.py  submission.py  error_analysis.py
├── scripts/
│   ├── make_synthetic_data.py            synthetic dataset with the challenge's structure
│   ├── build_local_tiny_transformer.py   offline tiny BERT / DeBERTa-v2 checkpoints (smoke tests)
│   └── download_pretrained.py            one-time download of DeBERTa-v3 / MiniLM weights
├── tests/                        50 pytest tests (unit + end-to-end)
├── data/   models/   outputs/    (not versioned)
```

`io.py` from the requested layout is named `data_io.py` so that it can never shadow Python's standard `io` module.

## 5. Modes

| | baseline | full |
|---|---|---|
| normalisation + offline address parsing | ✔ | ✔ |
| 11 syntactic / LSH / TF-IDF blocking methods | ✔ | ✔ |
| SupCon bi-encoder + FAISS semantic blocking | – | ✔ |
| LightGBM + isotonic calibration | ✔ | ✔ |
| Ditto / DeBERTa-v3 cross-encoder cascade + blending | – | ✔ |
| singleton model, graph cleanup, Macro-F0.5 decision search | ✔ | ✔ |
| needs | CPU, core requirements | GPU recommended, local pre-trained weights |

Switch with `mode:` in the config, or with `--set mode=full`. Any value can be overridden on the command line with `--set key.sub=value`.

## 6. Outputs

`models/<run>/`:
- `lightgbm.txt`, `feature_importance.tsv`, `calibrator_*.pkl`, `blender.pkl`
- `singleton_model.txt`, `pipeline.json` (feature list, decision parameters, blend, cascade), `config_snapshot.yaml`
- full mode only: `biencoder/` (encoder + tokenizer), `biencoder_index_train/` (FAISS), `crossencoder/`

`outputs/<run>/train/`:
- `training_report.json`: data stats, label stats, splits, blocking summary, class balance, LightGBM metrics, calibration comparison, blend trace, singleton model, chosen decision, calib and holdout metrics, ablation, timings
- `blocking_report.json`: pair completeness overall, per source, per split and before/after the cap; reduction ratio; possible vs candidate pairs; candidates per entity; true pairs lost; per-method recall and unique contribution; Macro-F0.5 ceiling given blocking
- `decision_search_trace.tsv`, `train.log`
- `error_analysis_holdout/`: `false_positives`, `false_negatives` (split into matcher misses and blocking losses), `singleton_errors`, `nonsingletons_predicted_empty`, `top_confident_errors`, `low_confidence_true_matches`, `difficult_cases` (TSV) and `error_summary.json`

`outputs/<run>/test/`:
- `candidate_pairs.tsv`, `matching_results.tsv`, `scored_candidates.tsv`
- `prediction_report.json`, `blocking_report.json`, `predict.log`
- full mode only: `semantic_index/` (FAISS)

`outputs/<run>/submission.zip`: the archive.

The file names and output columns are configurable in `submission:`. By default the output mirrors the label file's columns. If `data.sample_submission` is set, its header is used instead.

## 7. What has actually been verified, and what has not

Verified in this repository's environment (CPU only, no internet access to HuggingFace):

* `pytest`: 50 tests pass. They cover normalisation, suffix extraction, address parsing (including the PDF's parsing table), features, blocking, label parsing, Macro F0.5, decision rules, graph cleanup, output writing / validation, and a full train → predict → score run.
* Both modes run end to end on the **synthetic** demo data (800 train / 400 test Source-1 entities). These numbers only show that the code works; they say **nothing** about challenge performance.

  | synthetic demo | holdout Macro F0.5 | test Macro F0.5 (hidden synthetic labels) |
  |---|---|---|
  | baseline | 0.9916 | 0.9918 |
  | full (tiny *untrained* transformers) | 0.9856 | 0.9877 |

  The full-mode row uses randomly initialised 300k-parameter models. The pipeline correctly detects that blending with them does not help and keeps LightGBM alone. The gap to baseline is 1–2 entities on a 160-entity split.
* A 20k-entity synthetic stress test (2.4M candidate pairs) trains in about 3.5 minutes on 4 CPU cores. It was used to fix two real problems: the candidate-cap ranking (pair completeness after the cap rose from 0.957 to 0.996) and a hot loop in Soft-TFIDF (2.8× faster).

**Not verified:** anything on the real challenge data (it was not available), and full mode with real DeBERTa-v3 / MiniLM weights (the HuggingFace hub was unreachable here). No leaderboard score, real dataset size, threshold or runtime is claimed. When you run `train.py` on the real data, the reports listed above give all of these numbers for your data.

## 8. Tests

```bash
python -m pytest -q                 # everything (about 15 s)
python -m pytest -q -m "not slow"   # unit tests only (about 1 s)
```
