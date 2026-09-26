# Methodology — Business Entity Resolution (Amazon ML Challenge 2026)

This document describes the approach implemented in this repository. It is
included in the submission archive, as the challenge requires. It covers the
problem, each pipeline stage, how validation avoids leakage, the design
decisions, the assumptions, and a compliance audit.

References:
* **[T]** `ML_Challenge_2026_Video_Transcript.docx`, the official problem statement.
* **[A]** `Entity Resolution System Architecture.pdf`, the proposed architecture this code implements.

---

## 1. Problem (from [T])

* There are three independent sources. **Source 1** is the clean, de-duplicated reference list. **Sources 2 and 3** are noisy vendor fragments. No identifier is shared between them, so only the **business name and address** are available.
* For **every** Source-1 entity, find **all** matching records in Sources 2 and 3. An entity can have many matches, exactly one, or none.
* Ground truth has one row per Source-1 entity: its id maps to a comma-separated list of all matching ids, and the list is empty when there is no match. [T] says this "mirrors exactly what you'll submit".
* Every file is **tab-separated**.
* The deliverables are `matching_results.tsv` (one row per Source-1 entity, which is scored) and, at the end, one archive. The archive contains the final matches, the **candidate/unscored pairs** from blocking, the complete reproducible pipeline, and a methodology document.
* The metric is **Macro F0.5**, which weights precision twice as heavily as recall. A singleton scores 1 when its prediction is empty and 0 otherwise.
* Blocking sets the recall ceiling. [T] advises paying attention to region-specific patterns.
* External databases, APIs and lookups are **strictly prohibited**.

## 2. Pipeline

```
TSV ingestion ─► name normalisation + offline address parsing
      ─► hybrid blocking (11 syntactic methods [+ SupCon bi-encoder / FAISS]) ─► candidate_pairs.tsv
      ─► 103 pairwise features ─► LightGBM ─► isotonic calibration
      ─► [cascade ─► Ditto/DeBERTa-v3 cross-encoder ─► isotonic] ─► blend ─► final calibration
      ─► graph cleanup (off / exclusive / betweenness, chosen on validation)
      ─► decision rules tuned for Macro F0.5 (+ singleton gate) ─► matching_results.tsv
```

`mode: baseline` runs everything except the neural parts in brackets. `mode: full` runs the complete architecture from [A].

### 2.1 Ingestion (`src/data_io.py`, `src/schema.py`)
Every read uses `sep="\t"`, `dtype=str` and `keep_default_na=False`, so a business called "NA" is not turned into a missing value. Column names come from the config because [T] does not name them. The loader validates the schema and reports the columns it actually found. It counts empty names and addresses, and detects duplicate ids and duplicate name+address rows. Labels are parsed robustly (`a,b`, `[a, b]`, `['a','b']`, empty). Each label id is resolved to Source 2 or 3 by lookup, or by a per-column mapping when the labels use one column per source.

### 2.2 Name normalisation ([A] "Lexical Standardization and Legal Suffix Abstraction"; `src/normalization.py`)
1. NFKD decomposition. Combining marks are removed only after Latin letters, so Devanagari and similar scripts are not broken.
2. `str.casefold()` (for example, "Straße" becomes "strasse").
3. Dotted acronyms are collapsed ("S.A." → "sa"), letter/letter slashes are joined ("M/s" → "ms"), apostrophes are dropped, "&" becomes "and", and all other punctuation becomes a space.
4. **Legal suffixes are extracted by longest-first matching at the end of the name**, repeated so that compounds like "gmbh & co kg" or "co ltd" are removed whole. The table covers more than 45 legal forms (US, UK, India, DE, LatAm, EU, Asia). As [A] requires, the suffix is **kept**: its canonical form and family (corporation / llc / limited / gmbh / sa / …) become features (`suffix_eq`, `suffix_family_eq`, `suffix_conflict`). A suffix is never removed if that would leave the name empty.
5. Conservative canonicalisation of abbreviations ("manufacturing"/"mfg" → `mfg`, "holdings"/"hldgs" → `hldg`, …). Both the long and short forms map to the same token.
6. Derived keys: content tokens, a no-space string, Metaphone codes (whole name and first token), initials, and digits.

### 2.3 Address parsing ([A] "Statistical Address Parsing in an Offline Context"; `src/address_parser.py`)
* **libpostal**, when installed, is used offline through the `postal` bindings. It is a local C library and model, and it makes no network calls.
* An offline **rule-based parser** covering several regions is always available. It extracts the country, postcode (US ZIP/ZIP+4, Indian PIN including "560 038", UK, Canada, and 5-/4-digit codes), state (US, CA and AU codes and names; Indian state names), unit/suite/shop, landmarks ("Nr/Near/Opp …", the PDF's "house" component), house number (including "No 12", "Plot No 45", and European "Hauptstr 5"), road, city and suburb. It reproduces all three examples in the PDF's parsing table (see `tests/test_address_parser.py`).
* Address tokens are canonicalised (street→st, road→rd, suite→ste, near→nr, fifth→5th, "-strasse"→"-str").
* **No geocoder, map API or OSM query is used anywhere.**

### 2.4 Blocking ([A] "Algorithmic Paradigms for Candidate Generation"; `src/blocking.py`)
Candidates are generated separately for Source 2 and Source 3, for each Source-1 entity. The final set is the union of these methods:

| method | key / technique | handles |
|---|---|---|
| name_prefix | first 4 chars of the core name ([A]) | cheap exact-ish names |
| postcode | exact normalised postcode ([A]) | same area |
| postcode_phonetic | postcode + Metaphone of first token ([A]: "postal codes combined with phonetic hashing") | spelling variants locally |
| city_state | parsed city + state | missing postcodes |
| phonetic_name | Metaphone of the whole name | phonetic misspellings |
| address_key | house number + road token | **different names at the same address** ([T]) |
| tfidf_name_char / _word | TF-IDF cosine top-k ([A]) | typos, token reordering |
| tfidf_full_char | name + address char n-grams | joint evidence |
| tfidf_address | address-only char n-grams | shared address |
| minhash_lsh | MinHash (64 perms, 16 bands) over name shingles ([A]) | fault-tolerant Jaccard |
| semantic (full) | SupCon bi-encoder + FAISS top-k ([A] SC-Block) | abbreviations, paraphrases |

A key block larger than its `top_k` is ranked internally by TF-IDF cosine. Blocks larger than `max_block_size` are purged. Each (entity, source) then keeps at most `max_candidates_per_source` pairs. These are ranked by name+address TF-IDF cosine plus a bonus of 0.02 for each method that retrieved the pair.
*Why this ranking:* ranking by name similarity alone let many same-name branches of a common business push the true, address-matching record out of the cap. On a 20k-entity synthetic stress test the old ranking kept 95.7% of true pairs and the new one keeps 99.6%. The union before the cap held 99.7%.

The blocking report (`outputs/<run>/train/blocking_report.json`) contains pair completeness (overall, per source, per split, and before/after the cap), the reduction ratio, total possible and candidate pairs, the distribution of candidates per entity, the number of true pairs lost, per-method recall and unique contribution, and the **Macro-F0.5 ceiling that blocking imposes**. That ceiling is the score an oracle matcher would reach given these candidates. `candidate_pairs.tsv` is written **before** any model scores the pairs.

### 2.5 Labels, hard negatives and leakage control (`src/labels.py`, `src/splits.py`)
* **Positives** are the (S1, S2) and (S1, S3) pairs from the labels. True pairs that blocking missed are added to the LightGBM training rows. Their context features are left as NaN because the pair was never retrieved.
* **Negatives** are all other candidates. A candidate whose id appears in the entity's label list but cannot be resolved without ambiguity gets label −1 and is excluded, so a true match can **never** become a negative.
* **Hard negatives** come naturally from blocking: similar names, shared addresses, and same-brand branches. They are tagged (`similar_name`, `shared_address`, …) and reported. The cross-encoder trains on the highest-scoring LightGBM negatives of each entity plus a few random ones. The bi-encoder places entities with the same name prefix in the same batch.
* **Splitting unit:** one Source-1 entity together with all of its labelled Source-2/3 records. Because Source 1 is de-duplicated, a business belongs to exactly one split. The split is stratified on (#S2 matches, #S3 matches).
  * `fit` trains every model; `fit_val` (a subset of fit) is used only for early stopping.
  * `calib` fits the isotonic calibrators, the blend, the singleton model, the graph mode and the decision parameters.
  * `holdout` is never used for fitting and gives the reported estimate.
* **Guards against leakage:**
  * No test label exists or is read.
  * Thresholds are tuned only on `calib`.
  * The semantic-retrieval flag is **not** given to LightGBM as a feature, because on fit pairs it is in-sample.
  * Graph cleanup during evaluation only sees the pairs of the split being evaluated.
  * TF-IDF/IDF statistics are unsupervised statistics of the text in the dataset being processed. They use no labels, and exactly the same computation runs at test time.

### 2.6 Structural matcher: LightGBM ([A]; `src/features.py`, `src/train_lightgbm.py`)
The model uses 103 features:
* **Name:** exact (raw, clean, core, no-space), Jaro–Winkler (p = 0.1, prefix ≤ 4, as in [A]), Levenshtein, Indel, token sort/set, partial ratio, char/word TF-IDF cosine, **Soft-TFIDF** (secondary similarity Jaro–Winkler ≥ 0.90, per [A]; Levenshtein optional), token Jaccard/overlap, IDF-weighted overlap, highest shared IDF, IDF of unshared tokens, phonetic equality, prefix equality, common-prefix length, lengths, containment, acronym match, digits, and suffix agreement/conflict.
* **Address:** exact, Jaro–Winkler, token set, char TF-IDF, Soft-TFIDF, token Jaccard, and for each component (house number, road, city, state, postcode, unit) both-present / equality / conflict flags. Also postcode prefix, overlap of numeric tokens and a conflict flag, landmarks, suburb overlap, and missing-field indicators.
* **Combined:** name × address, strong-name/weak-address, weak-name/strong-address, number of agreeing and conflicting components, a total contradiction count, source id, and blocking provenance flags.
* **Label-free context:** the pair's rank, its gap to the best candidate and its margin over the best *other* candidate within the (entity, source) group, the same from the vendor's side, and a reciprocal-best flag.

Missing values are NaN, which LightGBM routes natively. Training uses early stopping on `fit_val`. Gain-based importance is saved to `models/<run>/feature_importance.tsv`, as [A] suggests.

### 2.7 Semantic blocking: SupCon bi-encoder ([A] SC-Block; `src/train_biencoder.py`)
Records are encoded by a shared MiniLM encoder (mean pooling, L2-normalised) trained with the batch-wise **Supervised Contrastive loss**. Each S1 entity and its labelled vendor records share one label. Singletons and extra vendor records get unique labels and act only as negatives. Batches are hard: similar keys go in the same batch, plus vendor negatives with the same key. The checkpoint is chosen by recall@k on `fit_val`. Every vendor record is encoded once and indexed per source in FAISS (`IndexFlatIP`, or HNSW for large data). The encoder and indices are saved.
**AdaFlood is not implemented.** [A] names it but gives no formula, and the per-sample flood levels need auxiliary models whose training procedure for SupCon is not specified. An optional *constant* Flooding term (Ishida et al., 2020) exists and is disabled by default. It is not AdaFlood.

### 2.8 Semantic matcher: Ditto cross-encoder ([A]; `src/train_crossencoder.py`)
* **Serialisation:** `COL name VAL … COL address VAL …` for each side. The tokenizer adds `[CLS]`/`[SEP]`, and a linear head on the `[CLS]` state outputs one logit. The default backbone is DeBERTa-v3.
* **Domain-knowledge injection:** postcodes are wrapped as `[PC] … [/PC]` and other digit-bearing tokens as `[NUM] … [/NUM]`. These tags are new special tokens.
* **MixDA:** each batch gets one of the operators del / swap / drop_col / token_del / attr_shuffle / entry_swap. The `[CLS]` encodings of the original and augmented inputs are interpolated with λ ~ Beta(α, α), and λ = max(λ, 1 − λ). This follows Ditto's MixDA.
* **TF-IDF summarisation** keeps only the highest-IDF words of any value longer than `tfidf_summarize_max_words`.
* **Focal loss** uses γ = 2, with optional α.
* **Cascade:** the cross-encoder only scores pairs where the calibrated LightGBM probability p is at least `lgb_low`, the pair is among the top `max_per_entity_source` of its (entity, source), and p is at most `lgb_high`. All other pairs keep their LightGBM probability.

### 2.9 Calibration, blending and decision ([A] "Isotonic Regression and Decision Threshold Calibration"; `src/calibration.py`, `src/scoring.py`, `src/decision.py`)
* Isotonic regression is fitted on `calib`, separately for each model. Platt scaling is also evaluated, and the cross-validated Brier score, log loss and ECE are reported. `method: auto` picks whichever is better.
* **Blending** can be weighted, rank-based or stacking. It is chosen on `calib` by Macro F0.5, and a blend must beat LightGBM alone by `min_improvement`. A final isotonic calibration is then applied.
* **Decision rules** (per entity):
  * a separate threshold for Source 2 and Source 3;
  * a margin to the entity's best score;
  * a singleton **gate** on the best score;
  * an optional **singleton model** that empties an entity when P(entity has a true match among its candidates) is below a threshold. This is an entity-level LightGBM on top-1/top-2 scores, the margin, counts, model agreement and the evidence of the top candidate, trained out-of-fold on `calib`;
  * an optional per-source cap;
  * a graph mode.
* The parameters are chosen by coordinate ascent that maximises the **exact entity-level Macro F0.5**. Every entity counts, including entities that have no candidates and true matches lost in blocking. For thresholds the search picks the **middle of the best plateau**, because calibrated scores are nearly binary and the first threshold of a tie sits on the edge. Any other rule must improve the score by `min_improvement` to be adopted. No threshold value is assumed in advance.

### 2.10 Graph cleanup ([A] GraLMatch/TransClean; `src/graph_cleanup.py`)
Analysis:
* The output is centred on Source 1, so no global clustering is required.
* Because Source 1 is de-duplicated, the only transitive error that can affect the output is a vendor record linked to **two** Source-1 entities. Transitivity would then declare two distinct reference businesses equal.
* The graph used is therefore bipartite, and each component may contain at most one S1 node.

The modes are:
* `exclusive`: each vendor keeps only its best S1 edge.
* `betweenness`: [A]'s procedure. The weakest separating bridge is cut first ("minimum edge cut"); otherwise the edge with the highest edge-betweenness × (1 − p) is cut. This repeats until no component holds two S1 nodes or exceeds the size limit.
* `off`.

The mode is chosen on `calib`, and cleanup is used only if it helps. The labels report counts vendor ids that appear under more than one S1 entity, which tests the at-most-one-S1-per-vendor assumption against the ground truth.

### 2.11 Output (`src/submission.py`)
`matching_results.tsv` has one row per Source-1 entity, in the original order. Predicted ids are comma-separated and sorted by score, and the field is empty for predicted singletons. The column names mirror the label file, or follow a configured sample submission. `candidate_pairs.tsv` has the columns `source1_id`, `candidate_id` and `candidate_source`. `scored_candidates.tsv` holds every score and decision for debugging. A local validator checks for tabs, the header, one row per entity, no duplicates, and that every id exists.

### 2.12 Engineering: bounded memory and resumable stages
The data size of the real challenge is unknown, and the pipeline has to run on ~12–16 GB notebook machines.
* **Blocking:** each vendor source is blocked, deduplicated and capped before the next source starts. The key-block ranking matrices are limited to about 2M cells, and MinHash-LSH processes Source-1 records in chunks with a running top-k.
* **Reports and labels:** the blocking report and label attachment use sorted integer pair ids, not Python sets of tuples.
* **Features:** pairwise features are computed in chunks by one reused worker pool. The parent calls `gc.freeze()` before forking, so the workers do not duplicate its heap. Results are streamed into a disk-backed float32 matrix (`features.npy`, memory-mapped), and LightGBM trains on the needed rows and predicts in blocks.
* **Stages:** training runs as seven stages and prediction as four. Each stage reads its inputs from checkpoints and writes its outputs back, so a crashed or restarted process resumes (`--resume`, or one notebook cell per stage) without recomputing earlier stages.
* **Memory release:** freed memory is handed back to the OS after every stage (`malloc_trim`).

## 3. Design decisions

1. **Normalisation.** Vendors write the same business differently ([T]: "Acme Robotics Incorporated" vs "Inc"). Without normalisation, string similarity and blocking keys fail on formatting alone. Suffixes are removed from the matching key but kept as evidence ([A]).
2. **Blocking.** Comparing every pair is |S1|·(|S2|+|S3|) comparisons, which is impossible at scale ([T], [A]).
3. **Several blocking methods.** Each key is brittle ([A]: "a single typographical error … will permanently separate a true match"). [T] shows that matches group through the name *or* through the address. A union of independent methods raises recall, and the per-method report shows what each one contributes.
4. **Candidate recall.** "You cannot match a record you never consider" ([T]). The report states the resulting Macro-F0.5 ceiling directly.
5. **LightGBM.** It is fast, handles NaN-rich tabular similarity features well, and is interpretable through gain importance ([A]). It scores every candidate, which makes it the base of the cascade.
6. **Semantic model.** Abbreviations, transliterations and paraphrases defeat character similarity ([A]: "Acme Mfg" vs "Acme Manufacturing").
7. **Bi-encoder for retrieval.** Records are encoded independently, so vendor embeddings are computed once and searched sub-linearly with FAISS ([A]). This is the only affordable way to apply a neural model to *all* records.
8. **Cross-encoder for hard cases.** Joint attention over both records catches token-level contradictions (for example, the house number differs) that a bi-encoder averages away. It costs far more per pair, so it only sees the uncertain part of the cascade.
9. **Calibration.** Raw LightGBM or transformer scores are not probabilities ([A]). Calibrated scores make blending meaningful and thresholds interpretable.
10. **Why Macro F0.5 moves the threshold.** A false merge costs roughly twice what a miss costs ([T]). The averaging is per entity, and a singleton goes from 1 to 0 with a single false id. So the best threshold is the empirical maximiser of *entity-level* F0.5 on validation, usually well above 0.5, and not the pairwise-F1 threshold. The value is measured, not assumed. [A]'s "0.85–0.95" is not taken as given.
11. **Singletons.** Each singleton is worth exactly as much as any other entity, and they are a sizable share of the data (see the label report). The gate and the singleton model target the rule "when in doubt, it is safer not to merge" ([T]).
12. **Graph cleanup.** It is not strictly necessary for a Source-1-centric output. The only meaningful case is one vendor record linked to two reference businesses. It is implemented as an option chosen on validation, and applying it blindly could remove correct links.
13. **A Source-1-centric output.** The task and the submission are defined per Source-1 entity ([T]). Source 2 and 3 records are never matched to each other, and the output always has exactly one row per Source-1 entity, even when the list is empty.

## 4. Assumptions and limitations
* **ASSUMPTION – schema.** [T] names no files or columns. All names come from the config, and `inspect_data.py` prints the real ones.
* **ASSUMPTION – label ids.** The ids in the combined list are resolved to a source by lookup. Ids present in both Source 2 and Source 3 are ambiguous: they are skipped for training and still count at string level in evaluation. One column per source is supported through `column_sources`.
* **ASSUMPTION – output format.** It mirrors the label file (per [T]), or a configured sample submission. The organisers' validation script must still be run.
* **ASSUMPTION – metric details.** Per-entity F0.5 on sets of ids; an empty prediction for an empty truth scores 1; an empty prediction for a non-empty truth scores 0; the average is unweighted over every Source-1 entity. This follows [T]. The official scorer's exact implementation is not public.
* **ASSUMPTION – candidate file format.** [T] only says "candidate/unscored pairs .tsv", so the columns used are (`source1_id`, `candidate_id`, `candidate_source`).
* **ASSUMPTION – calibration order.** [A] places isotonic calibration both on the raw model outputs (§ "Isotonic Regression") and after graph cleanup (§ "Synthesis"). The code calibrates each model, blends, calibrates again, and runs graph cleanup on those calibrated edge weights.
* **Not reproducible from [A]: AdaFlood.** See §2.7.
* **Pre-trained weights.** Full mode needs DeBERTa-v3 and MiniLM weights on local disk (`scripts/download_pretrained.py`). The build environment could not reach the HuggingFace hub, so full mode was verified end to end only with *tiny randomly initialised* local models (`scripts/build_local_tiny_transformer.py`). **No result from real pre-trained weights or real challenge data is claimed.**
* **Region coverage.** The rule-based parser has dictionaries for US/CA/AU/IN states and common postcode formats. Other regions work through the generic rules or libpostal, and the dictionaries can be extended in the config.
* **Refitting.** Models are trained on `fit` only (about 70% of the labelled entities). Refitting on all labels would change score distributions, which would invalidate the calibrators and thresholds, so it is not done by default.

## 5. Compliance audit against [T]

| Requirement | Status |
|---|---|
| No external database / API / lookup / geocoding / business lookup during matching | ✔ No network code in `src/`. Address parsing is local (rules or libpostal). Transformer weights are loaded with `local_files_only=True`. |
| Only the provided data is used for matching | ✔ Every feature, IDF statistic, tokenizer and model is built from the challenge TSVs. The one external input in full mode is generic pre-trained LM weights, downloaded once and explicitly (flagged below). |
| Output format | ✔ One row per Source-1 entity, TSV, comma-separated ids, empty for singletons. Mirrors the label file. Checked by a local validator. |
| candidate_pairs generated before matching | ✔ Written by `predict.py` right after blocking, before any feature or model runs. |
| Singletons can produce empty lists | ✔ Default for any entity with no accepted candidate, plus an explicit gate and singleton model. |
| Macro F0.5 is the optimisation target | ✔ The decision search maximises the exact entity-level Macro F0.5 on the calib split. |
| TSV with an explicit tab separator | ✔ `read_tsv` / `write_tsv` always use `\t`. |
| Reproducible pipeline + methodology in the archive | ✔ `package_submission.py` bundles the code, configs, this document, `candidate_pairs.tsv` and `matching_results.tsv`. |

**Flagged components.**
1. *libpostal* is trained on OpenStreetMap data, but it runs entirely offline, and [A] recommends it. It is optional, and the rule-based parser is the fallback.
2. *Pre-trained transformer weights* contain general language knowledge, not business records, and [A] requires them for full mode. If the organisers consider them out of bounds, the compliant alternative is `mode: baseline`, which uses only the challenge data. A second option is the tiny from-scratch models built by `scripts/build_local_tiny_transformer.py`, whose tokenizer is trained only on the training TSVs.
