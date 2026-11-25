"""Step 2 - flatten the raw API JSON to one row per trial and attach the label.

Output: data/processed/trials_labeled.csv.gz  (every fetched trial, including
those later excluded from modeling, so labeling decisions can be audited).
"""
import gzip
import json

import pandas as pd

from . import config
from .labeling import infer_label


def _join(items) -> str:
    return "|".join(str(i) for i in items if i)


def flatten(study: dict) -> dict:
    p = study.get("protocolSection", {})
    d = study.get("derivedSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    spons = p.get("sponsorCollaboratorsModule", {})
    design = p.get("designModule", {})
    info = design.get("designInfo", {})
    arms = p.get("armsInterventionsModule", {})
    elig = p.get("eligibilityModule", {})
    outcomes = p.get("outcomesModule", {})
    locs = p.get("contactsLocationsModule", {}).get("locations", [])
    arm_groups = arms.get("armGroups", [])
    interventions = arms.get("interventions", [])
    enroll = design.get("enrollmentInfo", {})
    countries = sorted({l.get("country") for l in locs if l.get("country")})

    category, label = infer_label(status.get("overallStatus"), status.get("whyStopped"))

    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        # ---- outcome information: defines the label, never a feature
        "overall_status": status.get("overallStatus"),
        "why_stopped": status.get("whyStopped"),
        "stop_category": category,
        "label": label,
        # ---- dates
        "start_date": status.get("startDateStruct", {}).get("date"),
        "first_submit_date": status.get("studyFirstSubmitDate"),
        "completion_date": status.get("completionDateStruct", {}).get("date"),
        # ---- design
        "phases": _join(design.get("phases", [])),
        "allocation": info.get("allocation"),
        "intervention_model": info.get("interventionModel"),
        "primary_purpose": info.get("primaryPurpose"),
        "masking": info.get("maskingInfo", {}).get("masking"),
        "enrollment": enroll.get("count"),
        "enrollment_type": enroll.get("type"),
        "n_arms": len(arm_groups),
        "arm_types": _join(sorted({a.get("type") for a in arm_groups if a.get("type")})),
        # ---- sponsor
        "lead_sponsor": spons.get("leadSponsor", {}).get("name"),
        "sponsor_class": spons.get("leadSponsor", {}).get("class"),
        "n_collaborators": len(spons.get("collaborators", [])),
        "has_dmc": p.get("oversightModule", {}).get("oversightHasDmc"),
        # ---- interventions and conditions
        "intervention_types": _join(sorted({i.get("type") for i in interventions if i.get("type")})),
        "intervention_names": _join(i.get("name") for i in interventions),
        "n_interventions": len(interventions),
        "conditions": _join(p.get("conditionsModule", {}).get("conditions", [])),
        "keywords": _join(p.get("conditionsModule", {}).get("keywords", [])),
        "condition_mesh": _join(m.get("term") for m in d.get("conditionBrowseModule", {}).get("meshes", [])),
        "condition_mesh_ancestors": _join(m.get("term") for m in d.get("conditionBrowseModule", {}).get("ancestors", [])),
        "intervention_mesh": _join(m.get("term") for m in d.get("interventionBrowseModule", {}).get("meshes", [])),
        # ---- eligibility
        "sex": elig.get("sex"),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "eligibility_criteria": elig.get("eligibilityCriteria"),
        # ---- outcomes (as planned in the protocol) and sites
        "n_primary_outcomes": len(outcomes.get("primaryOutcomes", [])),
        "n_secondary_outcomes": len(outcomes.get("secondaryOutcomes", [])),
        "primary_outcome_measures": _join(o.get("measure") for o in outcomes.get("primaryOutcomes", [])),
        "n_locations": len(locs),
        "n_countries": len(countries),
        "has_us_site": "United States" in countries,
        "has_results": study.get("hasResults"),
    }


def build(raw_path=None, out_path=None) -> pd.DataFrame:
    raw_path = raw_path or config.RAW_PATH
    out_path = out_path or config.LABELED_PATH
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        rows = [flatten(json.loads(line)) for line in f if line.strip()]
    df = (pd.DataFrame(rows)
          .drop_duplicates(config.ID_COL)
          .sort_values(config.ID_COL)
          .reset_index(drop=True))
    df["label"] = df["label"].astype("Int64")
    config.write_csv(df, out_path)
    print(f"[label] {len(df)} trials -> {out_path}")
    print("[label] stop_category counts:")
    print(df["stop_category"].value_counts().to_string())
    return df
