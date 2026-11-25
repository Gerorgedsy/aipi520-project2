"""Step 4 - deterministic time-based train / validation / test split."""
import pandas as pd

from . import config


def assign_split(start_date: pd.Series) -> pd.Series:
    d = pd.to_datetime(start_date)
    out = pd.Series("test", index=start_date.index)
    out[d <= config.VAL_END] = "val"
    out[d <= config.TRAIN_END] = "train"
    return out


def split(features: pd.DataFrame) -> dict:
    features = features.copy()
    features["split"] = assign_split(features["start_date"])
    summary = {}
    for name, path in config.SPLIT_PATHS.items():
        part = (features[features["split"] == name]
                .drop(columns="split")
                .sort_values(config.ID_COL))
        config.write_csv(part, path)
        summary[name] = {
            "n": int(len(part)),
            "positive_rate": round(float(part[config.TARGET].mean()), 4) if len(part) else None,
            "start_min": part["start_date"].min() if len(part) else None,
            "start_max": part["start_date"].max() if len(part) else None,
        }
        print(f"[split] {name:5s} n={summary[name]['n']:6d}  "
              f"positive_rate={summary[name]['positive_rate']}  "
              f"({summary[name]['start_min']} .. {summary[name]['start_max']})")
    return summary
