"""The interface between the shared pipeline and each teammate's model.

    from src.load import load_splits, make_preprocessor

    train, val, test = load_splits()
    pre = make_preprocessor()                       # impute + scale + one-hot
    X_train = pre.fit_transform(train[FEATURES])    # fit on TRAIN only
    X_val   = pre.transform(val[FEATURES])
    y_train = train[TARGET]

Everyone loading through here trains and evaluates on exactly the same rows,
so the three models are directly comparable.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import (BINARY_FEATURES, CATEGORICAL_FEATURES, LEAKY_COLUMNS,
                       NUMERIC_FEATURES, RISKY_FEATURES, TEXT_FEATURES)

TARGET = config.TARGET
ID_COL = config.ID_COL
# The default tabular feature set. TEXT_FEATURES are available on top of this.
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

__all__ = ["load_splits", "make_preprocessor", "FEATURES", "NUMERIC_FEATURES",
           "BINARY_FEATURES", "CATEGORICAL_FEATURES", "TEXT_FEATURES",
           "RISKY_FEATURES", "LEAKY_COLUMNS", "TARGET", "ID_COL"]


def load_splits():
    """Return (train, val, test) DataFrames exactly as the pipeline wrote them."""
    missing = [str(p) for p in config.SPLIT_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data not found. Run `python run_pipeline.py` first.\n  " + "\n  ".join(missing))
    return tuple(pd.read_csv(config.SPLIT_PATHS[s]) for s in ("train", "val", "test"))


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Unfitted preprocessing for the default tabular features.

    - numeric: median imputation (+ standardization unless scale_numeric=False;
      tree models do not need it)
    - binary: passed through
    - categorical: missing -> 'MISSING', then one-hot; categories unseen during
      fit are ignored at transform time
    Fit it on the training split only.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer(
        [("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
         ("bin", "passthrough", BINARY_FEATURES),
         ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=True,
    )
