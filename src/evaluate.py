"""Shared evaluation so the three models are scored the same way.

    from src.evaluate import evaluate
    evaluate(y_val, model.predict_proba(X_val)[:, 1], name="logreg / val")

The project asks for a PROBABILITY of success, so the metrics cover both
ranking (ROC-AUC, PR-AUC) and how trustworthy the probabilities are
(Brier score, log loss, calibration table).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)


def evaluate(y_true, p_success, name: str = "model", verbose: bool = True) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_success, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    metrics = {
        "name": name,
        "n": int(len(y)),
        "positive_rate": float(base),
        "roc_auc": float(roc_auc_score(y, p)),
        # Failure is the minority class and the one worth catching, so PR-AUC is
        # reported for the failure class as well.
        "pr_auc_success": float(average_precision_score(y, p)),
        "pr_auc_failure": float(average_precision_score(1 - y, 1 - p)),
        "brier": float(brier_score_loss(y, p)),
        # Brier score of always predicting the base rate; a model must beat this.
        "brier_baseline": float(base * (1 - base)),
        "log_loss": float(log_loss(y, p)),
    }
    if verbose:
        print(f"\n{name}  (n={metrics['n']}, success rate={base:.3f})")
        for k in ("roc_auc", "pr_auc_success", "pr_auc_failure", "brier", "brier_baseline", "log_loss"):
            print(f"  {k:16s} {metrics[k]:.4f}")
    return metrics


def calibration_table(y_true, p_success, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs. observed success rate per probability decile."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(p_success, dtype=float)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
              .reset_index())
