# Clinical Trial Outcome Prediction — AIPI 520 Project 2

Predict the probability that a clinical trial succeeds, using data from
[ClinicalTrials.gov](https://clinicaltrials.gov).

This repository contains the **shared data pipeline** the whole team builds on
(download → label → features → split) and one folder for each teammate's model.
Everyone trains and evaluates on exactly the same rows, so the models are
directly comparable.

> **Writeup:** the project writeup (approach, decisions, how course material was
> applied) is written by the team and is not part of this file. This README only
> documents how to run and reproduce the data pipeline.

---

## 1. Quick start

```bash
# 1. environment (Python 3.11 - 3.13; tested on 3.13)
conda create -n aipi520-p2 python=3.13 -y
conda activate aipi520-p2
pip install -r requirements.txt

# 2. check the code works (no network needed, takes a few seconds)
pytest -q

# 3. run the whole pipeline
python run_pipeline.py
```

Step 3 downloads about 81,000 trials (roughly 80 requests, several minutes, a
~150–200 MB compressed file) and writes everything under `data/`.

To try it on a small sample first: `python run_pipeline.py --max-pages 2`.

## 2. What the pipeline does

```
ClinicalTrials.gov API v2
        │  src/fetch_trials.py      query + pagination, sorted, hashed
        ▼
data/raw/trials.jsonl.gz  +  data/manifest.json
        │  src/build_dataset.py     flatten JSON → one row per trial
        │  src/labeling.py          label from overall_status + why_stopped
        ▼
data/processed/trials_labeled.csv.gz        (all fetched trials, for auditing)
        │  src/features.py          cohort filter + feature engineering
        ▼
data/processed/trials_features.csv.gz
        │  src/split.py             time-based split on start date
        ▼
data/processed/train.csv.gz   val.csv.gz   test.csv.gz
data/processed/pipeline_report.json         (counts, config, hashes)
```

Every choice below lives in **`src/config.py`**. To change one, edit it there
and rerun `python run_pipeline.py --skip-fetch`.

### Step 1 — Data: which trials are downloaded

| Choice | Value | Why |
|---|---|---|
| Source | ClinicalTrials.gov API v2 (`/api/v2/studies`) | The legacy bulk XML download was retired with the classic site in June 2024. |
| Study type | Interventional | The task is about trials of a treatment. |
| Intervention | Drug or Biological | Matches the pharma R&D framing of the project. |
| Phase | Phase 2 or Phase 3 (incl. combined 1/2 and 2/3) | Phases where efficacy is tested. |
| Status | `COMPLETED`, `TERMINATED`, `WITHDRAWN` | Only trials with a known ending can be labeled. `SUSPENDED` is excluded because the trial may resume. |
| Fields | Protocol modules only | Posted results are never downloaded, so they cannot leak into features. |

### Step 2 — Labeling

ClinicalTrials.gov has no "success" column, so the label is inferred from two
fields: `overall_status` and the free-text `why_stopped`. Each trial is first
assigned a **stop category**, and each category maps to a label.

| Stop category | Label | Rule |
|---|---|---|
| `completed` | **1** | `overall_status == COMPLETED` |
| `positive_early` | **1** | Stopped early for success or actually finished (e.g. "met primary endpoint", "accrual goal met", "approved by FDA") |
| `efficacy_futility` | **0** | Lack of efficacy, futility, did not meet endpoint, failed interim analysis |
| `safety` | **0** | Safety concern, toxicity, adverse events, clinical hold, DSMB stop |
| `enrollment` | **0** | Slow accrual, could not recruit, no eligible patients |
| `unknown` | **0** | Terminated/withdrawn with no reason, or a reason the rules do not recognize |
| `covid` | excluded | Stopped because of the COVID-19 pandemic |
| `business_funding` | excluded | Sponsor/business decision, funding, strategy, portfolio |
| `logistics` | excluded | Drug supply, PI left, regulatory/administrative, replaced by another study |

How the text rules work (`src/labeling.py`):

- Categories are checked in the order above; the **first match wins**. Evidence
  about the drug (success, efficacy, safety) outranks circumstances (COVID,
  enrollment, money, logistics). Example: *"slow recruitment and lower than
  expected efficacy"* → `efficacy_futility`.
- **Denials are removed first**, so *"terminated due to slow enrollment, not
  due to any safety concern"* is `enrollment`, not `safety`.
- A "success" phrase is ignored when the sentence contains a negative cue
  (*"did **not** meet the primary endpoint"* is not a success).
- Excluded categories are dropped from the modeling set because the stop says
  nothing about the trial's merit. To keep them as failures instead, change
  `LABEL_POLICY` in `src/labeling.py`.

**What the label means — and does not mean.** `label = 1` means the trial *ran
to completion*. It is a proxy for success: a completed trial can still have
missed its endpoint, so the positive class contains some noise. This limitation
should be stated wherever results are reported.

**Auditing the labels.** The rules were developed against ~400 real
`why_stopped` texts, and 35 of them are fixed as regression tests in
`tests/test_pipeline.py`. To review the labels on the full data yourself:

```bash
python -m src.audit_labels                       # 15 random examples per category
python -m src.audit_labels -n 40 --category unknown
```

The sample uses a fixed seed, so every teammate sees the same examples.

### Step 3 — Cohort and features

**Cohort filter** (counts for each step are saved in `pipeline_report.json`):

1. Drop trials with no start date.
2. Keep trials that **started between 2000-01-01 and 2019-12-31**.
   Before 2000 the registry is sparse. After 2019 many trials are still running,
   so the ones that already have an ending are disproportionately early
   terminations, which would bias the label.
3. Drop trials in an excluded stop category.

**Features** — all are facts written in the protocol before the trial ran:

| Group | Columns |
|---|---|
| Numeric | `start_year`, `phase_num`, `n_arms`, `n_interventions`, `n_collaborators`, `n_primary_outcomes`, `n_secondary_outcomes`, `min_age_years`, `max_age_years`, `criteria_n_chars`, `n_inclusion_criteria`, `n_exclusion_criteria`, `n_conditions`, `sponsor_n_prior_trials` |
| Binary (0/1) | `is_combined_phase`, `is_randomized`, `is_blinded`, `has_placebo_arm`, `has_active_comparator`, `has_dmc`, `healthy_volunteers`, `has_drug`, `has_biological`, `is_industry_sponsor`, `min_age_missing`, `max_age_missing`, and 15 therapeutic-area flags `ta_*` (from MeSH condition ancestors) |
| Categorical | `phases`, `allocation`, `intervention_model`, `primary_purpose`, `masking`, `sponsor_class`, `sex` |
| Text (optional) | `brief_title`, `conditions`, `intervention_names`, `primary_outcome_measures`, `eligibility_criteria` |

`sponsor_n_prior_trials` counts how many cohort trials the same sponsor had
already *started* before this one. It only looks backwards in time.

**Columns that must not be used as features** (kept in the files for auditing;
listed in `src.load.LEAKY_COLUMNS`):

| Column | Why it leaks |
|---|---|
| `overall_status`, `why_stopped`, `stop_category` | They define the label. |
| `completion_date`, `has_results` | Only known after the trial ends. |
| `enrollment`, `enrollment_type` | The registry replaces planned enrollment with the **actual** count. Withdrawn trials show 0 and terminated trials show small numbers, so this column gives away the answer. |

**Risky columns** (`src.load.RISKY_FEATURES`: `n_locations`, `n_countries`,
`has_us_site`) are not in the default feature set. Sites are listed as they
open, so a trial stopped early shows fewer sites than planned. Use them only as
a deliberate experiment and say so.

No imputation, scaling or encoding happens in this step, so nothing here is
fitted on data. Fitted preprocessing is in `make_preprocessor()` and must be fit
on the training split only.

### Step 4 — Train / validation / test split

The split is **by time**, on the trial start date:

| Split | Start date |
|---|---|
| train | 2000-01-01 to 2013-12-31 |
| val | 2014-01-01 to 2015-12-31 |
| test | 2016-01-01 to 2019-12-31 |

A model that predicts trial success would be used on *future* trials, so it is
evaluated on trials that started after everything it was trained on. A random
split would mix eras and put trials from the same drug programmes on both
sides. Sizes and class balance of each split are printed at the end of the run
and saved in `pipeline_report.json`.

## 3. Reproducibility

**What is guaranteed.** Given the same raw file, the pipeline produces
**byte-identical** output files on any machine: rows are sorted by NCT id, there
is no randomness in the data steps, package versions are pinned, and files are
written with fixed timestamps and line endings. The run prints a short hash of
`train` / `val` / `test`; if two teammates see the same hashes, they have
exactly the same data. (`tests/test_pipeline.py::test_rebuild_is_byte_identical`
checks this.)

**What changes over time.** ClinicalTrials.gov is updated daily, so a download
made on a different day can differ slightly. Each download therefore writes
`data/manifest.json` with the registry's data timestamp, the exact query, the
number of trials, and the SHA-256 of the raw file.

**Team workflow — one snapshot for everyone:**

1. **One person** runs `python run_pipeline.py`, then commits
   `data/manifest.json`.
2. That person shares `data/raw/trials.jsonl.gz` with the team (Duke Box,
   Google Drive, or a GitHub release — it is too large for a normal commit).
3. Everyone else puts the file at `data/raw/trials.jsonl.gz` and runs

   ```bash
   python run_pipeline.py --skip-fetch
   ```

   This checks the file against the manifest hash and stops with an error if it
   is a different snapshot.
4. Compare the three hashes printed at the end.

Snapshot used for the reported results: *fill in from `data/manifest.json`
after the team download (registry data timestamp, number of trials, SHA-256).*

**To reproduce the reported results from scratch without the shared file**, run
`python run_pipeline.py`. You will get a newer snapshot; counts may differ by a
small amount from the ones reported.

## 4. Using the data in a model

```python
from src.load import load_splits, make_preprocessor, FEATURES, TARGET
from src.evaluate import evaluate

train, val, test = load_splits()

pre = make_preprocessor()            # median impute + scale + one-hot
X_train = pre.fit_transform(train[FEATURES])   # fit on train only
X_val = pre.transform(val[FEATURES])

# ... fit your model ...
evaluate(val[TARGET], p_val, name="my model / val")
```

- `make_preprocessor(scale_numeric=False)` skips scaling for tree models.
- `src.load` also exports `NUMERIC_FEATURES`, `BINARY_FEATURES`,
  `CATEGORICAL_FEATURES`, `TEXT_FEATURES`, `RISKY_FEATURES`, `LEAKY_COLUMNS`.
- `src/evaluate.py` reports ROC-AUC, PR-AUC (for both classes), Brier score
  with its base-rate baseline, log loss, and a calibration table, so all three
  models are scored the same way.
- `models/example_baseline.py` is a runnable template:
  `python -m models.example_baseline`.

**Rules that keep the comparison fair:** load data only through
`load_splits()`; fit everything (preprocessing included) on `train`; tune on
`val`; evaluate on `test` once, at the end.

## 5. Repository layout

```
run_pipeline.py          one command for the whole pipeline
requirements.txt         pinned versions
src/
  config.py              every pipeline choice in one place
  fetch_trials.py        step 1: API download + manifest
  labeling.py            step 2: labeling rules
  build_dataset.py       step 2: JSON → table
  features.py            step 3: cohort + features
  split.py               step 4: time-based split
  load.py                interface for the models
  evaluate.py            shared metrics
  audit_labels.py        print examples per label category
models/
  example_baseline.py    template; add one file or folder per teammate
tests/
  test_pipeline.py       45 tests
  fixtures/              91 real trial records for offline tests
data/                    created by the pipeline (not committed, except manifest.json)
```

## 6. Known limitations

- **The label is a proxy.** Completion is not the same as meeting the primary
  endpoint (see Step 2).
- **Labeling rules are keyword-based.** Unusual wording falls into `unknown`,
  which is treated as failure. Check the size of that category in
  `pipeline_report.json` and audit it.
- **Records are the latest version, not the version at trial start.** Sponsors
  can edit a record during and after the trial, so some protocol fields may
  have changed since registration. The clearly affected fields (enrollment,
  sites) are excluded; subtler effects may remain.
- **The registry changes daily**, which is why results are tied to one snapshot.
