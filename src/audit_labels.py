"""Print random examples of each stop category so a human can check the rules.

    python -m src.audit_labels            # 15 examples per category
    python -m src.audit_labels -n 40 --category unknown

The sample is drawn with a fixed seed, so everyone reviews the same trials.
"""
import argparse
import sys

import pandas as pd

from . import config
from .labeling import LABEL_POLICY


def main():
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print every character
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=15)
    ap.add_argument("--category", default=None)
    args = ap.parse_args()

    df = pd.read_csv(config.LABELED_PATH, usecols=["nct_id", "overall_status", "why_stopped", "stop_category"])
    cats = [args.category] if args.category else [c for c in LABEL_POLICY if c != "completed"]
    for cat in cats:
        sub = df[df["stop_category"] == cat]
        label = LABEL_POLICY[cat]
        print(f"\n=== {cat}  ->  label {'excluded' if label is None else label}   ({len(sub)} trials)")
        for _, r in sub.sample(min(args.n, len(sub)), random_state=config.SEED).iterrows():
            reason = r["why_stopped"] if isinstance(r["why_stopped"], str) else "(no reason given)"
            print(f"  {r['nct_id']}  [{r['overall_status']}]  {reason[:220]}")


if __name__ == "__main__":
    main()
