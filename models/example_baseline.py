"""Template showing how a model plugs into the shared pipeline.

Copy this file, rename it, and replace the estimator with your own model.
Run from the repo root:   python -m models.example_baseline
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.evaluate import calibration_table, evaluate
from src.load import FEATURES, TARGET, load_splits, make_preprocessor


def main():
    train, val, test = load_splits()

    model = Pipeline([
        ("pre", make_preprocessor()),       # fitted on the training split only
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=config.SEED)),
    ])
    model.fit(train[FEATURES], train[TARGET])

    # Tune on validation. Touch the test split once, at the very end.
    p_val = model.predict_proba(val[FEATURES])[:, 1]
    evaluate(val[TARGET], p_val, name="logistic regression / validation")
    print(calibration_table(val[TARGET], p_val).to_string(index=False))


if __name__ == "__main__":
    main()
