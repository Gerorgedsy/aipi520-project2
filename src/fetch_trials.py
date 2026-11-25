"""Step 1 - download the trial cohort from the ClinicalTrials.gov v2 API.

Writes
    data/raw/trials.jsonl.gz   one study JSON per line, sorted by NCT id
    data/manifest.json         what was fetched, when, and the file's SHA-256

The registry is updated daily, so two downloads on different days differ
slightly. The manifest is what makes a run reproducible: commit it, share the
raw file with the team, and `python run_pipeline.py --skip-fetch` verifies the
file against the recorded hash before using it.
"""
import gzip
import hashlib
import json
import time
import zlib
from datetime import datetime, timezone

import requests

from . import config


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(session, url, params=None, retries=6):
    """GET with exponential backoff on network errors and non-200 answers."""
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:  # timeouts, dropped connections
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {retries} tries: {last}")


def fetch(max_pages=None, raw_path=None, manifest_path=None) -> dict:
    raw_path = raw_path or config.RAW_PATH
    manifest_path = manifest_path or config.MANIFEST_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "aipi520-project2-pipeline"
    version = _get(session, config.API_VERSION_URL)

    params = {
        "pageSize": config.PAGE_SIZE,
        "countTotal": "true",
        "filter.overallStatus": ",".join(config.STATUSES),
        "filter.advanced": config.ADVANCED_FILTER,
        "fields": ",".join(config.API_FIELDS),
    }
    lines, page, total = {}, 0, None
    while True:
        body = _get(session, config.API_URL, params)
        if total is None:
            total = body.get("totalCount")
            print(f"[fetch] {total} trials match the query")
        for study in body.get("studies", []):
            nct = study["protocolSection"]["identificationModule"]["nctId"]
            line = json.dumps(study, sort_keys=True, ensure_ascii=False) + "\n"
            lines[nct] = zlib.compress(line.encode("utf-8"))  # keeps memory low (~2 KB/trial)
        page += 1
        print(f"[fetch] page {page}: {len(lines)} trials")
        token = body.get("nextPageToken")
        if not token or (max_pages and page >= max_pages):
            break
        params["pageToken"] = token
        time.sleep(0.3)  # stay well inside the public API's rate limit

    # Sorted by NCT id and written with a fixed gzip mtime, so the same set of
    # records always produces a byte-identical file (and the same hash).
    with open(raw_path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        for nct in sorted(lines):
            gz.write(zlib.decompress(lines[nct]))

    manifest = {
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "api_version": version.get("apiVersion"),
        "registry_data_timestamp": version.get("dataTimestamp"),
        "query": {k: v for k, v in params.items() if k not in ("pageToken", "countTotal")},
        "total_count_reported": total,
        "n_trials_saved": len(lines),
        "partial_download": bool(max_pages),
        "raw_file": str(raw_path.relative_to(config.ROOT)).replace("\\", "/"),
        "raw_sha256": sha256_of(raw_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[fetch] saved {len(lines)} trials -> {raw_path}")
    return manifest


def verify_raw(raw_path=None, manifest_path=None) -> dict:
    """Check the local raw file against the manifest. Raises if they differ."""
    raw_path = raw_path or config.RAW_PATH
    manifest_path = manifest_path or config.MANIFEST_PATH
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python run_pipeline.py` to download it, "
            "or put the team's shared copy there."
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; cannot verify the raw file.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_of(raw_path)
    if actual != manifest["raw_sha256"]:
        raise ValueError(
            "Raw data does not match data/manifest.json.\n"
            f"  expected {manifest['raw_sha256']}\n  actual   {actual}\n"
            "You have a different snapshot than the one the manifest describes."
        )
    print(f"[fetch] raw file verified (snapshot {manifest['registry_data_timestamp']})")
    return manifest
