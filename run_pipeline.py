"""Run the whole shared data pipeline: fetch -> label -> features -> split.

    python run_pipeline.py                # download a fresh snapshot, then build
    python run_pipeline.py --skip-fetch   # rebuild from the existing raw file
                                          # (verified against data/manifest.json)
    python run_pipeline.py --max-pages 2  # small trial run (2,000 trials)
"""
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn

from src import config
from src.build_dataset import build
from src.features import (BINARY_FEATURES, CATEGORICAL_FEATURES, LEAKY_COLUMNS,
                          NUMERIC_FEATURES, RISKY_FEATURES, TEXT_FEATURES, build_features)
from src.fetch_trials import fetch, sha256_of, verify_raw
from src.labeling import LABEL_POLICY
from src.split import split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-fetch", action="store_true", help="reuse data/raw/trials.jsonl.gz")
    ap.add_argument("--max-pages", type=int, default=None, help="download only N pages (for a quick test)")
    args = ap.parse_args()

    np.random.seed(config.SEED)
    manifest = verify_raw() if args.skip_fetch else fetch(max_pages=args.max_pages)

    labeled = build()
    features, cohort_steps = build_features(labeled)
    split_summary = split(features)

    report = {
        "raw_snapshot": {k: manifest[k] for k in
                         ("registry_data_timestamp", "fetched_at_utc", "n_trials_saved", "raw_sha256", "partial_download")},
        "config": {
            "statuses": config.STATUSES, "advanced_filter": config.ADVANCED_FILTER,
            "cohort_start_min": config.COHORT_START_MIN, "cohort_start_max": config.COHORT_START_MAX,
            "train_end": config.TRAIN_END, "val_end": config.VAL_END, "seed": config.SEED,
        },
        "label_policy": {k: ("excluded" if v is None else v) for k, v in LABEL_POLICY.items()},
        "stop_category_counts_all_fetched": labeled["stop_category"].value_counts().to_dict(),
        "cohort_filter_steps": cohort_steps,
        "stop_category_counts_in_cohort": features["stop_category"].value_counts().to_dict(),
        "splits": split_summary,
        "features": {"numeric": NUMERIC_FEATURES, "binary": BINARY_FEATURES,
                     "categorical": CATEGORICAL_FEATURES, "text": TEXT_FEATURES,
                     "risky_not_in_default_set": RISKY_FEATURES,
                     "leaky_do_not_use": LEAKY_COLUMNS},
        "output_sha256": {name: sha256_of(path) for name, path in config.SPLIT_PATHS.items()},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    config.REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[done] report -> {config.REPORT_PATH}")
    print("[done] output hashes (compare with your teammates'):")
    for name, digest in report["output_sha256"].items():
        print(f"         {name:5s} {digest[:16]}")


if __name__ == "__main__":
    main()
