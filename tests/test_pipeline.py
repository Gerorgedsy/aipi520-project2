"""Tests for the shared pipeline. Run with:  pytest -q

The end-to-end tests use tests/fixtures/sample_trials.jsonl.gz: 91 real records
downloaded from the ClinicalTrials.gov v2 API, so no network is needed.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.build_dataset import build, flatten
from src.features import (BINARY_FEATURES, CATEGORICAL_FEATURES, LEAKY_COLUMNS,
                          NUMERIC_FEATURES, age_to_years, build_features)
from src.labeling import LABEL_POLICY, infer_label, stop_category
from src.load import FEATURES, make_preprocessor
from src.split import assign_split

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trials.jsonl.gz"

# Real why_stopped texts from the registry with the category a human reader assigns.
LABEL_CASES = [
    ("COMPLETED", None, "completed"),
    ("COMPLETED", "anything", "completed"),
    ("WITHDRAWN", None, "unknown"),
    ("TERMINATED", "See Detailed Description", "unknown"),
    ("TERMINATED", "Lack of efficacy", "efficacy_futility"),
    ("TERMINATED", "Study did not meet its primary endpoint.", "efficacy_futility"),
    ("TERMINATED", "met stopping rules for futility", "efficacy_futility"),
    ("TERMINATED", "MBX-102 did not meet the target efficacy profile of HbA1c change.", "efficacy_futility"),
    ("TERMINATED", "Due to lack of efficacy shown at the time of the interim analysis.", "efficacy_futility"),
    ("TERMINATED", "The Sponsor terminated the study due to futility. The decision to stop the study was not based on safety concerns.", "efficacy_futility"),
    ("TERMINATED", "The radioligands to be evaluated showed acceptable safety but limited benefit in metastatic prostate cancer", "efficacy_futility"),
    ("TERMINATED", "The study was terminated prematurely on 16 May 2014 due to a safety concern.", "safety"),
    ("TERMINATED", "Hepatic safety signal identified.", "safety"),
    ("TERMINATED", "FDA has placed all trials involving Pacritinib on Full Clinical Hold", "safety"),
    ("TERMINATED", "Low recruitment rate", "enrollment"),
    ("TERMINATED", "slow accrual", "enrollment"),
    ("TERMINATED", "The study was terminated by the Sponsor due to insufficient enrollment.", "enrollment"),
    ("WITHDRAWN", "No patients were enrolled, study cancelled before start", "enrollment"),
    # a denied topic must not trigger that topic
    ("TERMINATED", "Due to enrollment challenges. The termination is not a consequence of any safety concern.", "enrollment"),
    ("TERMINATED", "Recruitment was early terminated due to slow recruitment. Not linked to any safety concern.", "enrollment"),
    ("TERMINATED", "Sponsor decision; Not a safety decision", "business_funding"),
    ("TERMINATED", "Business reasons, not related to safety.", "business_funding"),
    ("TERMINATED", "The Sponsor terminated the trial early for strategic business reasons. This isn't related to any safety/efficacy/clinical activity/benefit-risk concern with petosemtamab.", "business_funding"),
    ("TERMINATED", "Lack of funding", "business_funding"),
    ("TERMINATED", "Discontinuation of all bardoxolone chronic kidney disease programs", "business_funding"),
    # 'hepatitis' is a disease name, not a liver-safety signal
    ("TERMINATED", "Study was terminated due to sponsor's decision to not pursue further development of inarigivir soproxil in chronic hepatitis B.", "business_funding"),
    ("TERMINATED", "The study was stopped due to lack of meaningful enrollment due to the COVID-19 pandemic.", "covid"),
    ("TERMINATED", "PI left site", "logistics"),
    ("TERMINATED", "Drug supply issues", "logistics"),
    ("WITHDRAWN", "We never got Health Canada approval", "logistics"),
    ("TERMINATED", "Limited regulatory value to uncontrolled trials. No safety/efficacy concerns.", "logistics"),
    ("TERMINATED", "ITI-007 approved by FDA", "positive_early"),
    ("TERMINATED", "Accrual goal met", "positive_early"),
    ("TERMINATED", "Study completed", "positive_early"),
    ("TERMINATED", "Primary Analysis and survival follow up completed per protocol. Not stopped due to safety concerns", "positive_early"),
]


@pytest.mark.parametrize("status,why,expected", LABEL_CASES)
def test_stop_category(status, why, expected):
    assert stop_category(status, why) == expected


def test_label_policy_covers_every_category():
    cats = {stop_category(s, w) for s, w, _ in LABEL_CASES}
    assert cats <= set(LABEL_POLICY)
    assert infer_label("COMPLETED", None) == ("completed", 1)
    assert infer_label("TERMINATED", "Lack of efficacy") == ("efficacy_futility", 0)
    assert infer_label("TERMINATED", "Lack of funding") == ("business_funding", None)


def test_age_to_years():
    assert age_to_years("18 Years") == 18
    assert age_to_years("6 Months") == pytest.approx(0.5)
    assert np.isnan(age_to_years(None))


def test_assign_split_boundaries():
    dates = pd.Series([config.TRAIN_END, "2014-01-01", config.VAL_END, "2016-01-01"])
    assert assign_split(dates).tolist() == ["train", "val", "val", "test"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    labeled = build(FIXTURE, out / "labeled.csv.gz")
    feats, steps = build_features(labeled, out / "features.csv.gz")
    return labeled, feats, steps, out


def test_flatten_reads_real_records():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        row = flatten(json.loads(f.readline()))
    assert row["nct_id"].startswith("NCT")
    assert row["overall_status"] in config.STATUSES


def test_end_to_end(built):
    labeled, feats, steps, _ = built
    assert labeled["nct_id"].is_unique and len(labeled) == 91
    assert steps["fetched"] >= steps["after_start_date_window"] >= steps["after_dropping_ambiguous_labels"] == len(feats)
    assert set(feats[config.TARGET].unique()) <= {0, 1}
    d = pd.to_datetime(feats["start_date"])
    assert d.min() >= pd.Timestamp(config.COHORT_START_MIN)
    assert d.max() <= pd.Timestamp(config.COHORT_START_MAX)
    for col in FEATURES:
        assert col in feats.columns, col
    for col in BINARY_FEATURES:
        assert set(feats[col].unique()) <= {0, 1}, col


def test_no_leaky_or_risky_column_is_a_default_feature():
    from src.features import RISKY_FEATURES
    assert not set(LEAKY_COLUMNS) & set(FEATURES)
    assert not set(RISKY_FEATURES) & set(FEATURES)


def test_sponsor_history_only_looks_backwards(built):
    _, feats, _, _ = built
    feats = feats.assign(d=pd.to_datetime(feats["start_date"]))
    for _, r in feats.iterrows():
        earlier = feats[(feats["lead_sponsor"] == r["lead_sponsor"]) & (feats["d"] < r["d"])]
        assert r["sponsor_n_prior_trials"] >= len(earlier)
        same_or_earlier = feats[(feats["lead_sponsor"] == r["lead_sponsor"]) & (feats["d"] <= r["d"])]
        assert r["sponsor_n_prior_trials"] <= len(same_or_earlier) - 1


def test_preprocessor_runs_and_has_no_nans(built):
    _, feats, _, _ = built
    split = assign_split(feats["start_date"])
    train, test = feats[split == "train"], feats[split != "train"]
    pre = make_preprocessor()
    Xtr = pre.fit_transform(train[FEATURES])
    Xte = pre.transform(test[FEATURES])
    Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
    Xte = Xte.toarray() if hasattr(Xte, "toarray") else Xte
    assert Xtr.shape[1] == Xte.shape[1]
    assert not np.isnan(Xtr).any() and not np.isnan(Xte).any()


def test_rebuild_is_byte_identical(built, tmp_path):
    """Same raw input -> byte-identical outputs (the reproducibility guarantee)."""
    _, _, _, first = built
    labeled = build(FIXTURE, tmp_path / "labeled.csv.gz")
    build_features(labeled, tmp_path / "features.csv.gz")
    for name in ("labeled.csv.gz", "features.csv.gz"):
        a = hashlib.sha256((first / name).read_bytes()).hexdigest()
        b = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert a == b, name


def test_fetch_paginates_and_manifest_verifies(monkeypatch, tmp_path):
    """fetch() follows page tokens, writes a sorted file, and verify_raw() checks the hash."""
    from src import fetch_trials

    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        studies = [json.loads(line) for line in f][:6]
    pages = {None: (studies[4:6], "t1"), "t1": (studies[0:2], "t2"), "t2": (studies[2:4], None)}

    class FakeResponse:
        status_code = 200
        def __init__(self, body): self._body = body
        def json(self): return self._body

    class FakeSession:
        headers = {}
        def get(self, url, params=None, timeout=None):
            if url == config.API_VERSION_URL:
                return FakeResponse({"apiVersion": "2.0.5", "dataTimestamp": "2026-11-24T10:00:04"})
            batch, token = pages[(params or {}).get("pageToken")]
            body = {"studies": batch, "totalCount": 6}
            if token:
                body["nextPageToken"] = token
            return FakeResponse(body)

    monkeypatch.setattr(fetch_trials.requests, "Session", FakeSession)
    monkeypatch.setattr(fetch_trials.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    raw, manifest_path = tmp_path / "raw" / "trials.jsonl.gz", tmp_path / "manifest.json"

    manifest = fetch_trials.fetch(raw_path=raw, manifest_path=manifest_path)
    assert manifest["n_trials_saved"] == 6
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        ids = [json.loads(line)["protocolSection"]["identificationModule"]["nctId"] for line in f]
    assert ids == sorted(ids) and len(ids) == 6
    fetch_trials.verify_raw(raw, manifest_path)          # passes on the untouched file

    first = manifest["raw_sha256"]
    assert fetch_trials.fetch(raw_path=raw, manifest_path=manifest_path)["raw_sha256"] == first

    raw.write_bytes(raw.read_bytes() + b"x")             # tamper -> must be detected
    with pytest.raises(ValueError):
        fetch_trials.verify_raw(raw, manifest_path)
