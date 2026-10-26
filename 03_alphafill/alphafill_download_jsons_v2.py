
"""
AlphaFill JSON downloader (Spyder-friendly)
-------------------------------------------
Reads your UBS CSV (one row per UBS) and downloads the AlphaFill JSON
metadata for each unique UniProt in column 'site_uniprots' to a local directory.

How to use in Spyder:
  1) Open this file in Spyder.
  2) Edit the CONFIG block below (paths).
  3) Press F5 to run. Progress prints in the console.

Notes:
  * JSON endpoint documented here:
    https://alphafill.eu/v1/aff/${entryid}/json
  * If a UniProt is missing in AlphaFill, you will see HTTP 404.

-----------------------------------------------------
Changes vs original:
  * Adds LOGGING_DIR and writes the manifest there.
  * Skips JSONs that already exist (>50 bytes), so they are not re-downloaded.
"""

import os
import time
import json
import pathlib
import pandas as pd
import requests
import random
import sys

def get_cwd_prefix(levels=3):
    from pathlib import Path
    # levels=3 -> root + 2 dirs; levels=2 -> root + 1 dir, etc.
    return str(Path(*Path.cwd().parts[:levels]))
user_dir     = get_cwd_prefix()


# ========= CONFIG =========
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

SOURCE_CSV_PATH = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
OUTPUT_DIR   = str(cfg.ALPHAFILL_JSONS_DIR)
LOGGING_DIR  = str(cfg.ALPHAFILL_DIR)

# Delay between *consecutive entries* (even when previous succeeded).
BASE_DELAY_BETWEEN_REQUESTS_S = 2.0   # e.g. 1 second base delay
JITTER_ADDITIONAL_DELAY_S     = 0.5   # up to +0.5s random jitter; set 0 to disable
# Delay between retries (used inside http_get_with_retries).
TIMEOUT_S    = 10
MAX_RETRIES  = 3
SLEEP_BETWEEN_RETRIES_S = 2

# Optional: Recheck-only mode
RECHECK_ONLY = False  # set True in Spyder to skip downloads and only write missing list
# You can also list additional directories whose contents mean "already have it"
EXTRA_SCAN_DIRS = []  # e.g., [r"D:\alphafill_data\alphafill\jsons_export"]

# NEW - trigger alphafill server for entry generation
AF_JSON_URL_TMPL = "https://alphafill.eu/v1/aff/{entryid}/json"

# Use plain UniProt ID to trigger generation
AF_MODEL_TRIGGER_URL_TMPL = "https://alphafill.eu/model?id={uniprot}"
# New: HTML model URL used to trigger generation
#AF_MODEL_URL_TMPL = "https://alphafill.eu/model?id={entryid}"


# Control generation-trigger behavior
TRIGGER_MODEL_BEFORE_JSON = True          # enable the warm-up logic
MODEL_TRIGGER_TIMEOUT_S = 10              # allow slow generation, original, prev 60
POST_TRIGGER_WAIT_S = 5                   # small pause before re-checking JSON, prev 5, 3

# ==========================
#AF_JSON_URL_TMPL = "https://alphafill.eu/v1/aff/{entryid}/json"



def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def read_uniprots_from_csv_v1(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if 'site_uniprots' not in df.columns:
        raise ValueError("Column 'site_uniprots' not found in CSV")
    uniprots = []
    for v in df['site_uniprots'].astype(str):
        parts = [p for p in v.split('_') if p]
        if len(parts) == 1:
            uniprots.append(parts[0])
        else:
            uniprots.extend(parts)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for u in uniprots:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq

def read_uniprots_from_csv(
    csv_path: str,
    level_filter: str | None = "entry",
    skip_multi_uniprot_entries: bool = False,
) -> list[str]:
    """
    Read unique UniProt accessions from the unified CSV.

    Parameters
    ----------
    csv_path : str
        Path to the unified CSV.
    level_filter : str | None
        If not None, keep only rows where column 'level' matches this value
        (e.g. "entry" or "ubs").
    skip_multi_uniprot_entries : bool
        If True, discard rows whose 'uniprots_str' contains more than one UniProt.
        This is mainly intended for level='entry'.

    Returns
    -------
    list[str]
        Unique UniProt accessions in first-seen order.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    if level_filter is not None:
        if "level" not in df.columns:
            raise ValueError("Column 'level' not found in CSV")
        df = df[df["level"].astype(str).str.lower() == str(level_filter).lower()].copy()

    if "uniprots_str" not in df.columns:
        raise ValueError("Column 'uniprots_str' not found in CSV")

    # Optional: keep only single-UniProt rows
    if skip_multi_uniprot_entries:
        df = df[
            df["uniprots_str"]
            .fillna("")
            .astype(str)
            .apply(lambda s: len(s.split()) == 1)
        ].copy()

    seen = set()
    uniq = []

    for v in df["uniprots_str"].fillna("").astype(str):
        for u in v.split():
            if u and u not in seen:
                uniq.append(u)
                seen.add(u)

    return uniq

def trigger_model_generation(uniprot: str) -> tuple[bool, str]:
    """
    Hit the HTML model endpoint once to trigger AlphaFill to generate the entry
    for this UniProt. We don't care about the body, just that the backend is poked.
    """
    url = AF_MODEL_TRIGGER_URL_TMPL.format(uniprot=uniprot)
    try:
        r = requests.get(url, timeout=MODEL_TRIGGER_TIMEOUT_S)
        if r.status_code == 200:
            return True, f"model_page_ok: {url}"
        # Even non-200 may still kick off generation
        return False, f"model_page_status_{r.status_code}: {url}"
    except Exception as e:
        return False, f"model_page_error_{e}: {url}"

def download_one0(entry_id: str, out_dir: str) -> tuple[bool, str]:
    url = AF_JSON_URL_TMPL.format(entryid=entry_id)
    out_path = os.path.join(out_dir, f"{entry_id}.json")
    # Skip if already present and looks non-empty
    if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
        return True, f"exists: {out_path}"

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT_S)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True, f"ok: {out_path}"
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        if attempt < MAX_RETRIES:
            time.sleep(SLEEP_BETWEEN_RETRIES_S)

    return False, f"failed after {MAX_RETRIES} attempts: {last_err}"

def download_one1(entry_id: str, out_dir: str) -> tuple[bool, str]:
    """
    Robust single-entry downloader:
      - Skips existing valid JSONs
      - Retries transient errors (with backoff)
      - Treats 404 as permanent miss
      - Atomic write via .partial + JSON parse validation
    """
    url = AF_JSON_URL_TMPL.format(entryid=entry_id)
    out_dir_p = pathlib.Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_p / f"{entry_id}.json"
    tmp_path = out_path.with_suffix(".json.partial")

    # Fast-path: already have it and looks valid
    if out_path.is_file() and out_path.stat().st_size > 50:
        try:
            with out_path.open("r", encoding="utf-8") as fh:
                json.load(fh)  # validate once
            return True, f"exists: {out_path}"
        except Exception:
            # existing file is corrupt -> continue to redownload
            pass

    # Fetch with retries
    try:
        content, code = http_get_with_retries(url)
    except Exception as e:
        return False, f"failed after {MAX_RETRIES} attempts: {e}"

    if code == 404:
        return False, "404 not found on server"

    # Write to temp file first (atomic promote later)
    try:
        with tmp_path.open("wb") as fh:
            fh.write(content)
    except Exception as e:
        return False, f"io_write_error: {e}"

    # Validate JSON before promoting
    try:
        # Decode explicitly; AlphaFill returns UTF-8 JSON
        json.loads(content.decode("utf-8"))
    except Exception:
        # leave the partial for inspection
        return False, "invalid_json_from_server"

    # Promote atomically on same filesystem
    try:
        tmp_path.replace(out_path)
    except Exception as e:
        return False, f"atomic_replace_failed: {e}"

    return True, f"ok: {out_path}"

def download_one(uniprot: str, out_dir: str) -> tuple[bool, str]:
    """
    Download AlphaFill JSON for a UniProt ID.

    Strategy:
      - Always save as <uniprot>.json.
      - Try JSON with:
            1) <uniprot>
            2) AF-<uniprot>-F1-model_v6
      - If JSON 404s, trigger generation via model?id=<uniprot> ONCE,
        wait a bit, then retry JSON.
      - Atomic writes + JSON validation.
    """
    uniprot = uniprot.strip()
    out_dir_p = pathlib.Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_p / f"{uniprot}.json"
    tmp_path = out_path.with_suffix(".json.partial")

    # Fast path: existing valid file
    if out_path.is_file() and out_path.stat().st_size > 50:
        try:
            with out_path.open("r", encoding="utf-8") as fh:
                json.load(fh)
            return True, f"exists: {out_path}"
        except Exception:
            # Corrupt -> try re-download
            pass

    candidate_entryids = [
        uniprot,
        f"AF-{uniprot}-F1-model_v6",
    ]

    last_reason = None
    triggered = False  # ensure we only poke /model?id=<uniprot> once

    for entryid in candidate_entryids:
        url = AF_JSON_URL_TMPL.format(entryid=entryid)

        # 1st attempt: normal JSON fetch
        try:
            content, code = http_get_with_retries(url)
        except Exception as e:
            last_reason = f"{entryid}: failed after retries: {e}"
            continue

        # Success
        if code == 200:
            try:
                with tmp_path.open("wb") as fh:
                    fh.write(content)
                json.loads(content.decode("utf-8"))
                tmp_path.replace(out_path)
                return True, f"ok via {entryid}: {out_path}"
            except Exception as e:
                last_reason = f"{entryid}: io_or_json_error: {e}"
                continue

        # 404: maybe not generated yet
        if code == 404 and TRIGGER_MODEL_BEFORE_JSON and not triggered:
            # Trigger generation once with plain UniProt ID
            ok_trig, trig_msg = trigger_model_generation(uniprot)
            # Optional debug:
            # print(f"{uniprot}: trigger result: {trig_msg}")
            triggered = True

            # Wait and re-try JSON for this same entryid
            time.sleep(POST_TRIGGER_WAIT_S)
            try:
                content2, code2 = http_get_with_retries(
                    url,
                    max_tries=1,
                    timeout=TIMEOUT_S,
                )
            except Exception as e2:
                last_reason = f"{entryid}: after trigger failed: {e2}"
                continue

            if code2 == 200:
                try:
                    with tmp_path.open("wb") as fh:
                        fh.write(content2)
                    json.loads(content2.decode("utf-8"))
                    tmp_path.replace(out_path)
                    return True, f"ok via {entryid} after trigger: {out_path}"
                except Exception as e3:
                    last_reason = f"{entryid}: after trigger io_or_json_error: {e3}"
                    continue

            if code2 == 404:
                last_reason = f"{entryid}: still 404 after trigger"
                # fall through to try next candidate_entryid
                continue

        # If 404 without triggering (already triggered earlier) or other codes
        last_reason = f"{entryid}: status_{code}"
        # try next candidate

    if last_reason is None:
        last_reason = "no_candidate_succeeded"
    return False, last_reason



# NEW
def scan_present_uniprots(*dirs) -> set[str]:
    present = set()
    for d in dirs:
        p = pathlib.Path(d)
        if not p.exists():
            continue
        for f in p.glob("*.json"):
            # We saved files as "<entry_id>.json" where entry_id is UniProt
            up = f.stem.strip()
            if up:
                present.add(up)
    return present

# NEW
def seed_from_manifest(manifest_path: str) -> set[str]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        return {row["uniprot"] for row in m.get("log", []) if row.get("ok")}
    except Exception:
        return set()

def http_get_with_retries0(
    url: str,
    max_tries: int = MAX_RETRIES,
    base_sleep: float = SLEEP_BETWEEN_RETRIES_S,
    timeout: int = TIMEOUT_S,
):
    """
    GET with bounded retries and exponential backoff.
    Returns (content_bytes, status_code) on success or 404.
    Raises the last exception after exhausting retries.
    """
    last_err = None
    for i in range(max_tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content, 200
            if r.status_code == 404:
                # Permanent: do not retry further
                return None, 404
            # Other HTTP codes: retry
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        # Backoff before next try
        time.sleep(base_sleep * (2 ** i))
    # Exhausted
    raise last_err

def http_get_with_retries(
    url: str,
    max_tries: int = MAX_RETRIES,
    base_sleep: float = SLEEP_BETWEEN_RETRIES_S,
    timeout: int = TIMEOUT_S,
):
    last_err = None
    last_status = None

    for i in range(max_tries):
        try:
            r = requests.get(url, timeout=timeout)
            last_status = r.status_code

            if r.status_code == 200:
                return r.content, 200

            # Non-200 -> remember and retry
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e

        if i < max_tries - 1:
            time.sleep(base_sleep * (2 ** i))

    if last_status == 404:
        # Let caller decide how to handle lazy generation / missing entries
        return None, 404

    # Other persistent errors
    raise last_err



# ==========================


def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(LOGGING_DIR)
    uniprots = read_uniprots_from_csv(SOURCE_CSV_PATH)
    print(f"Found {len(uniprots)} unique UniProts to fetch.")
    #sys.exit(0)

    # ========= RECHECK =========
    # Seed “present” set from output dir and optional extra dirs + manifest
    present = set()
    present |= scan_present_uniprots(OUTPUT_DIR, *EXTRA_SCAN_DIRS)
    manifest_path = os.path.join(LOGGING_DIR, "_download_manifest.json")
    present |= seed_from_manifest(manifest_path)

    missing = [u for u in uniprots if u not in present]
    ensure_dir(os.path.join(LOGGING_DIR, "missing"))     # CREATE "missing" DIR
    missing_path = os.path.join(LOGGING_DIR, "missing", "_missing_uniprots.txt")
    with open(missing_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(missing))
    print(f"Recheck summary: total={len(uniprots)}, have={len(present)}, missing={len(missing)}")
    print(f"Missing list written to: {missing_path}")

    if RECHECK_ONLY:
        print("RECHECK_ONLY=True -> Skipping downloads.")
        return
    # ===========================

    
    ok, fail, exist = 0, 0, 0
    logs = []
    for i, up in enumerate(missing, 1):     # Download only missing UniProts
    #for i, up in enumerate(uniprots, 1):
        success, msg = download_one(up, OUTPUT_DIR)
        logs.append((up, success, msg))
        if success:
            if msg.startswith("exists"):
                exist += 1
            else:
                ok += 1
        else:
            fail += 1
        if i % 25 == 0 or not success:
            print(f"[{i}/{len(missing)}] {up}: {msg}")
            #print(f"[{i}/{len(uniprots)}] {up}: {msg}")

        # polite delay between requests (independent of retry backoff)
        if i < len(missing):  # no need to sleep after the last one
            delay = BASE_DELAY_BETWEEN_REQUESTS_S
            if JITTER_ADDITIONAL_DELAY_S > 0:
                delay += random.uniform(0, JITTER_ADDITIONAL_DELAY_S)
            time.sleep(delay)
    # Save manifest to LOGGING_DIR
    manifest = {
        "csv_source": os.path.abspath(SOURCE_CSV_PATH),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "logging_dir": os.path.abspath(LOGGING_DIR),
        "n_uniprots": len(uniprots),
        "n_ok_downloaded": ok,
        "n_existing_skipped": exist,
        "n_fail": fail,
        "log": [{"uniprot": u, "ok": s, "msg": m} for (u, s, m) in logs],
    }
    with open(os.path.join(LOGGING_DIR, "_download_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. OK(new)={ok}, EXISTING_SKIPPED={exist}, FAIL={fail}.")
    print(f"Manifest: {os.path.join(LOGGING_DIR, '_download_manifest.json')}")

if __name__ == "__main__":
    main()
