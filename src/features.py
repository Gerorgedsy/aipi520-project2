"""Step 3 - cohort filter, cleaning and feature engineering shared by all models.

Every feature here is something written in the protocol BEFORE the trial ran.
Anything that is only known afterwards is kept out of the feature lists below
(see LEAKY_COLUMNS) so that nobody trains on the answer by accident.

Nothing in this step is fitted on data (no imputation, scaling or encoding), so
it cannot leak information from the test set. Fitted preprocessing lives in
`src/load.py::make_preprocessor`, which is fit on the training split only.
"""
import re

import numpy as np
import pandas as pd

from . import config

# Broad therapeutic areas, read from the MeSH ancestors of the trial's conditions.
THERAPEUTIC_AREAS = {
    "ta_oncology": "Neoplasms",
    "ta_cardiovascular": "Cardiovascular Diseases",
    "ta_nervous_system": "Nervous System Diseases",
    "ta_mental": "Mental Disorders",
    "ta_infection": "Infections",
    "ta_respiratory": "Respiratory Tract Diseases",
    "ta_digestive": "Digestive System Diseases",
    "ta_metabolic": "Nutritional and Metabolic Diseases",
    "ta_endocrine": "Endocrine System Diseases",
    "ta_immune": "Immune System Diseases",
    "ta_musculoskeletal": "Musculoskeletal Diseases",
    "ta_urogenital": "Urogenital Diseases",
    "ta_skin": "Skin and Connective Tissue Diseases",
    "ta_hematologic": "Hemic and Lymphatic Diseases",
    "ta_eye": "Eye Diseases",
}

NUMERIC_FEATURES = [
    "start_year", "phase_num", "n_arms", "n_interventions", "n_collaborators",
    "n_primary_outcomes", "n_secondary_outcomes",
    "min_age_years", "max_age_years", "criteria_n_chars", "n_inclusion_criteria",
    "n_exclusion_criteria", "n_conditions", "sponsor_n_prior_trials",
]
BINARY_FEATURES = [
    "is_combined_phase", "is_randomized", "is_blinded", "has_placebo_arm",
    "has_active_comparator", "has_dmc", "healthy_volunteers",
    "has_drug", "has_biological", "is_industry_sponsor",
    "min_age_missing", "max_age_missing",
] + list(THERAPEUTIC_AREAS)
CATEGORICAL_FEATURES = [
    "phases", "allocation", "intervention_model", "primary_purpose", "masking",
    "sponsor_class", "sex",
]
# Free text, for anyone who wants TF-IDF or embeddings. Not used by make_preprocessor.
TEXT_FEATURES = [
    "brief_title", "conditions", "intervention_names", "primary_outcome_measures",
    "eligibility_criteria",
]

# Site information is NOT in the default feature set. The registry lists sites as
# they are activated, so a trial stopped early (or withdrawn before opening)
# shows fewer sites than it planned: the count partly reflects the outcome.
# Use these only as a deliberate, documented experiment.
RISKY_FEATURES = ["n_locations", "n_countries", "has_us_site"]

# Present in the files for auditing, but NOT valid model inputs.
LEAKY_COLUMNS = {
    "overall_status": "defines the label",
    "why_stopped": "defines the label",
    "stop_category": "defines the label",
    "completion_date": "only known once the trial has ended",
    "has_results": "results are posted after the trial ends",
    "enrollment": "the registry overwrites planned enrollment with the ACTUAL count; "
                  "withdrawn trials show 0 and terminated trials show small numbers",
    "enrollment_type": "ACTUAL vs ESTIMATED reveals whether the trial has ended",
}

_AGE_UNITS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52.18, "day": 1 / 365.25,
              "hour": 1 / 8766, "minute": 1 / 525960}


def age_to_years(value):
    """'18 Years' -> 18.0, '6 Months' -> 0.5, missing -> NaN."""
    if not isinstance(value, str):
        return np.nan
    m = re.match(r"\s*([\d.]+)\s*([a-zA-Z]+)", value)
    if not m:
        return np.nan
    unit = m.group(2).lower().rstrip("s")
    return float(m.group(1)) * _AGE_UNITS.get(unit, np.nan)


def _count_bullets(text, section):
    """Count bullet lines under 'Inclusion Criteria' / 'Exclusion Criteria'."""
    if not isinstance(text, str):
        return 0
    lower = text.lower()
    start = lower.find(f"{section} criteria")
    if start == -1:
        return 0
    other = "exclusion" if section == "inclusion" else "inclusion"
    end = lower.find(f"{other} criteria", start + 1)
    block = text[start:end if end > start else len(text)]
    return len(re.findall(r"^\s*(?:\*|-|\d+[.)])\s+", block, flags=re.MULTILINE))


def apply_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep labeled trials that started inside the cohort window. Returns the
    filtered frame and a count of how many rows each rule removed."""
    steps = {"fetched": len(df)}
    df = df.copy()
    # Registry dates are 'YYYY-MM' or 'YYYY-MM-DD'; a missing day becomes the 1st.
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", format="mixed")
    df = df[df["start_date"].notna()]
    steps["after_dropping_missing_start_date"] = len(df)
    df = df[(df["start_date"] >= config.COHORT_START_MIN) & (df["start_date"] <= config.COHORT_START_MAX)]
    steps["after_start_date_window"] = len(df)
    df = df[df[config.TARGET].notna()]
    steps["after_dropping_ambiguous_labels"] = len(df)
    return df, steps


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered feature columns. Expects apply_cohort() output."""
    df = df.copy()

    def contains(col, token):
        return df[col].fillna("").str.split("|").apply(lambda xs: token in xs)

    df["start_year"] = df["start_date"].dt.year

    # Phase: PHASE2 -> 2, PHASE3 -> 3, combined phases -> the mean (1.5 or 2.5).
    def phase_num(s):
        nums = [int(n) for n in re.findall(r"PHASE(\d)", s or "")]
        return float(np.mean(nums)) if nums else np.nan
    df["phase_num"] = df["phases"].apply(phase_num)
    df["is_combined_phase"] = df["phases"].fillna("").str.contains(r"\|", regex=True)

    df["is_randomized"] = df["allocation"].eq("RANDOMIZED")
    df["is_blinded"] = df["masking"].isin(["SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"])
    df["has_placebo_arm"] = contains("arm_types", "PLACEBO_COMPARATOR")
    df["has_active_comparator"] = contains("arm_types", "ACTIVE_COMPARATOR")
    df["has_drug"] = contains("intervention_types", "DRUG")
    df["has_biological"] = contains("intervention_types", "BIOLOGICAL")
    df["is_industry_sponsor"] = df["sponsor_class"].eq("INDUSTRY")
    for col in ("has_dmc", "healthy_volunteers", "has_us_site"):
        df[col] = df[col].fillna(False).astype(bool)

    # Ages: a missing bound means "no limit", which is informative, so it gets a flag.
    df["min_age_years"] = df["minimum_age"].apply(age_to_years)
    df["max_age_years"] = df["maximum_age"].apply(age_to_years)
    df["min_age_missing"] = df["min_age_years"].isna()
    df["max_age_missing"] = df["max_age_years"].isna()

    crit = df["eligibility_criteria"]
    df["criteria_n_chars"] = crit.fillna("").str.len()
    df["n_inclusion_criteria"] = crit.apply(lambda t: _count_bullets(t, "inclusion"))
    df["n_exclusion_criteria"] = crit.apply(lambda t: _count_bullets(t, "exclusion"))
    df["n_conditions"] = df["conditions"].fillna("").apply(lambda s: len([c for c in s.split("|") if c]))

    for col, mesh_term in THERAPEUTIC_AREAS.items():
        df[col] = (contains("condition_mesh_ancestors", mesh_term)
                   | contains("condition_mesh", mesh_term))

    # Sponsor experience: how many cohort trials this sponsor had ALREADY started
    # before this one. Uses only earlier start dates, so it never looks ahead.
    df = df.sort_values(["start_date", config.ID_COL]).reset_index(drop=True)
    df["sponsor_n_prior_trials"] = df.groupby("lead_sponsor", dropna=False).cumcount()
    df.loc[df["lead_sponsor"].isna(), "sponsor_n_prior_trials"] = 0

    for col in BINARY_FEATURES + ["has_us_site"]:
        df[col] = df[col].astype(int)
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def build_features(labeled: pd.DataFrame, out_path=None) -> tuple[pd.DataFrame, dict]:
    out_path = out_path or config.FEATURES_PATH
    cohort, steps = apply_cohort(labeled)
    feats = engineer(cohort)
    feats["start_date"] = feats["start_date"].dt.strftime("%Y-%m-%d")
    config.write_csv(feats, out_path)
    print(f"[features] {len(feats)} trials x {feats.shape[1]} columns -> {out_path}")
    return feats, steps
