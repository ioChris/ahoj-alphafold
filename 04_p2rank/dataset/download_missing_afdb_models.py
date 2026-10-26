# -*- coding: utf-8 -*-
"""
Download missing AFDB monomer models for the UniProts present in an AFDB dataset.

The script:
  - reads the dataset CSV and extracts unique `site_uniprot` values
  - checks whether `AF-<UniProt>-F1-model_v*.cif.gz` already exists locally
  - downloads missing models from AlphaFold DB
  - stores them locally as `.cif.gz`
  - writes a manifest CSV and JSON summary
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


DEFAULT_INPUT = cfg.AFDB_ST100_DATASET_CSV
DEFAULT_AFDB_DIR = cfg.AFDB_DIR
DEFAULT_OUTPUT_DIR = cfg.AFDB_DATASET_DIR
BASE_URL = "https://alphafold.ebi.ac.uk/files"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download missing AFDB models for dataset UniProts.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--afdb_dir", type=Path, default=DEFAULT_AFDB_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--latest_version", type=int, default=6)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--limit", type=int, default=0, help="Optional UniProt limit for testing.")
    return p.parse_args()


def find_existing_model(uniprot: str, afdb_dir: Path) -> Optional[Path]:
    hits = sorted(afdb_dir.glob(f"AF-{uniprot}-F1-model_v*.cif.gz"))
    if not hits:
        return None

    def version_key(path: Path) -> tuple[int, str]:
        try:
            version = int(path.name.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            version = -1
        return (version, path.name)

    hits.sort(key=version_key, reverse=True)
    return hits[0]


def download_model(uniprot: str, afdb_dir: Path, latest_version: int, timeout: int) -> tuple[str, str]:
    afdb_dir.mkdir(parents=True, exist_ok=True)

    existing = find_existing_model(uniprot, afdb_dir)
    if existing is not None:
        return ("existing", existing.name)

    last_error = ""
    for version in range(latest_version, 0, -1):
        filename = f"AF-{uniprot}-F1-model_v{version}.cif.gz"
        out_path = afdb_dir / filename
        url = f"{BASE_URL}/AF-{uniprot}-F1-model_v{version}.cif"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                content = resp.read()
            with gzip.open(out_path, "wb") as fh:
                fh.write(content)
            return ("downloaded", filename)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 404:
                continue
            break
        except Exception as exc:
            last_error = str(exc)
            break

    return ("missing", last_error)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.afdb_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, usecols=["site_uniprot"], low_memory=False)
    uniprots = sorted({str(x).strip() for x in df["site_uniprot"] if str(x).strip()})
    if args.limit > 0:
        uniprots = uniprots[:args.limit]

    rows = []
    for idx, uniprot in enumerate(uniprots, start=1):
        status, detail = download_model(
            uniprot=uniprot,
            afdb_dir=args.afdb_dir,
            latest_version=args.latest_version,
            timeout=args.timeout,
        )
        rows.append({"site_uniprot": uniprot, "status": status, "detail": detail})
        print(f"[{idx}/{len(uniprots)}] {uniprot}: {status} {detail}")

    manifest_csv = args.output_dir / "afdb_model_download_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["site_uniprot", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_csv": str(args.input_csv),
        "afdb_dir": str(args.afdb_dir),
        "n_uniprots_requested": len(uniprots),
        "n_existing": sum(1 for r in rows if r["status"] == "existing"),
        "n_downloaded": sum(1 for r in rows if r["status"] == "downloaded"),
        "n_missing": sum(1 for r in rows if r["status"] == "missing"),
        "latest_version": args.latest_version,
        "timeout": args.timeout,
    }
    summary_json = args.output_dir / "afdb_model_download_summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[SAVE] {manifest_csv}")
    print(f"[SAVE] {summary_json}")


if __name__ == "__main__":
    main()
