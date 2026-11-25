"""Single source of truth for every choice the pipeline makes.

Change a value here, rerun `python run_pipeline.py`, and every downstream file
is rebuilt consistently. Nothing else in the repo hard-codes these choices.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "trials.jsonl.gz"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_PATH = PROCESSED_DIR / "trials_labeled.csv.gz"
FEATURES_PATH = PROCESSED_DIR / "trials_features.csv.gz"
SPLIT_PATHS = {s: PROCESSED_DIR / f"{s}.csv.gz" for s in ("train", "val", "test")}
REPORT_PATH = PROCESSED_DIR / "pipeline_report.json"

SEED = 520

# ---------------------------------------------------------------- 1. fetch
API_URL = "https://clinicaltrials.gov/api/v2/studies"
API_VERSION_URL = "https://clinicaltrials.gov/api/v2/version"
PAGE_SIZE = 1000

# Only trials whose fate is known can be labeled. SUSPENDED is left out because
# a suspended trial can still resume.
STATUSES = ["COMPLETED", "TERMINATED", "WITHDRAWN"]

# Interventional drug / biologic trials in Phase 2 or 3
# (also matches the combined PHASE1|PHASE2 and PHASE2|PHASE3 designations).
ADVANCED_FILTER = (
    "AREA[StudyType]INTERVENTIONAL "
    "AND AREA[InterventionType](DRUG OR BIOLOGICAL) "
    "AND AREA[Phase](PHASE2 OR PHASE3)"
)

# Only the modules the pipeline reads. This skips the (very large) posted
# results, so the download stays small and no outcome data enters the features.
API_FIELDS = [
    "IdentificationModule", "StatusModule", "SponsorCollaboratorsModule",
    "OversightModule", "ConditionsModule", "DesignModule",
    "ArmsInterventionsModule", "OutcomesModule", "EligibilityModule",
    "LocationCountry", "ConditionBrowseModule", "InterventionBrowseModule",
    "HasResults",
]

# ---------------------------------------------------------------- 2. cohort
# Trials are kept only if they STARTED inside this window.
#  - before 2000 the registry is sparse and inconsistently filled in;
#  - after 2019 many trials are still running, so the ones that already have an
#    ending are disproportionately early terminations (right-censoring bias).
COHORT_START_MIN = "2000-01-01"
COHORT_START_MAX = "2019-12-31"

# ---------------------------------------------------------------- 3. split
# Time-based split on start date: train on the past, evaluate on the future.
# A random split would let the model see trials from the same era (same
# sponsors, same drug programmes) as the ones it is tested on.
TRAIN_END = "2013-12-31"   # train: start <= TRAIN_END
VAL_END = "2015-12-31"     # val  : TRAIN_END < start <= VAL_END ; test: after VAL_END

TARGET = "label"
ID_COL = "nct_id"


def write_csv(df, path):
    """Write a gzip CSV that is byte-identical for identical data on any OS:
    fixed gzip timestamp and Unix line endings (so file hashes are comparable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n",
              compression={"method": "gzip", "mtime": 0, "compresslevel": 6})
