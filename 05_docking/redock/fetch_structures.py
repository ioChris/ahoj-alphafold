# -*- coding: utf-8 -*-
"""
Fetch the inputs the redock reference-pose build needs from RCSB:
  * experimental source structures  -> <cache>/structures/<pdb>.cif.gz   (full mmCIF)
  * CCD chemical components         -> <cache>/ccd/<CCD>.cif             (atom names+graph)

Only jobs with a docked pose (has_pose) are fetched by default (errored docking jobs
have no pose to score). Threaded, resumable (skips existing non-empty files), and
logs obsolete/removed PDBs (HTTP 404) rather than failing.

The structures cache is large (~45 GB for the full G1 set) and MUST be local, not on
OneDrive — see cfg.DOCKING_REDOCK_CACHE_DIR.

Env: `ahoj-docking` (pandas; stdlib urllib).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# --- repo bootstrap ---------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

RUN3_DIR = cfg.DOCKING_RUN3_DIR
PDB_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
CCD_URL = "https://files.rcsb.org/ligands/download/{ccd}.cif"

_UA = {"User-Agent": "ahoj-af-redock/1.0 (academic; structural superposition)"}


def _download(url: str, out: Path, timeout: int = 120) -> str:
    if out.exists() and out.stat().st_size > 0:
        return "cached"
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return "empty"
        tmp.replace(out)
        return "ok"
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        return f"missing_http_{e.code}"
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return f"error_{type(e).__name__}"


def _fetch_all(items: List[Tuple[str, str, Path]], workers: int, sleep: float, label: str) -> Dict[str, str]:
    status: Dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download, url, out): key for key, url, out in items}
        for fut in as_completed(futs):
            key = futs[fut]
            status[key] = fut.result()
            done += 1
            if sleep:
                time.sleep(sleep)
            if done % 500 == 0 or done == len(items):
                nok = sum(1 for v in status.values() if v in ("ok", "cached"))
                print(f"  [{label}] {done}/{len(items)}  ok/cached={nok}")
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch RCSB source structures + CCD components for redock.")
    p.add_argument("--escort_csv", type=Path, default=RUN3_DIR / "reference_poses" / "escort_g1.csv")
    p.add_argument("--cache_dir", type=Path, default=cfg.DOCKING_REDOCK_CACHE_DIR)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--sleep", type=float, default=0.0, help="per-completion delay (politeness)")
    p.add_argument("--all_jobs", action="store_true", help="fetch for all G1 jobs, not just has_pose")
    p.add_argument("--limit", type=int, default=0, help="limit unique PDBs (debug/subset)")
    p.add_argument("--skip_structures", action="store_true")
    p.add_argument("--skip_ccd", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    e = pd.read_csv(args.escort_csv, low_memory=False)
    if not args.all_jobs and "has_pose" in e.columns:
        e = e[e["has_pose"]]
    struct_dir = args.cache_dir / "structures"
    ccd_dir = args.cache_dir / "ccd"
    struct_dir.mkdir(parents=True, exist_ok=True)
    ccd_dir.mkdir(parents=True, exist_ok=True)

    pdbs = sorted({str(p).lower() for p in e["src_pdb"] if isinstance(p, str) and p and p != "nan"})
    ccds = sorted({str(c).upper() for c in e["ligand_id"] if isinstance(c, str) and c})
    if args.limit:
        pdbs = pdbs[:args.limit]
    print(f"[FETCH] {len(pdbs)} unique PDBs, {len(ccds)} unique CCDs -> {args.cache_dir}")

    struct_status: Dict[str, str] = {}
    ccd_status: Dict[str, str] = {}
    if not args.skip_structures:
        items = [(p, PDB_URL.format(pdb=p), struct_dir / f"{p}.cif.gz") for p in pdbs]
        struct_status = _fetch_all(items, args.workers, args.sleep, "pdb")
    if not args.skip_ccd:
        items = [(c, CCD_URL.format(ccd=c), ccd_dir / f"{c}.cif") for c in ccds]
        ccd_status = _fetch_all(items, args.workers, args.sleep, "ccd")

    missing = sorted(k for k, v in struct_status.items() if v.startswith("missing"))
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "cache_dir": str(args.cache_dir), "escort_csv": str(args.escort_csv),
        "n_pdb": len(pdbs), "n_ccd": len(ccds),
        "pdb_ok": sum(1 for v in struct_status.values() if v in ("ok", "cached")),
        "pdb_missing": len(missing), "missing_pdbs": missing[:100],
        "ccd_ok": sum(1 for v in ccd_status.values() if v in ("ok", "cached")),
        "ccd_missing": sum(1 for v in ccd_status.values() if not (v in ("ok", "cached"))),
    }
    log_dir = args.escort_csv.parent
    with open(log_dir / "fetch_structures_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    # per-pdb status table (so compute can skip obsolete)
    if struct_status:
        pd.DataFrame({"src_pdb": list(struct_status), "status": list(struct_status.values())}).to_csv(
            log_dir / "fetch_pdb_status.csv", index=False)
    print(f"[DONE] PDB ok={meta['pdb_ok']}/{meta['n_pdb']} missing={meta['pdb_missing']}  "
          f"CCD ok={meta['ccd_ok']}/{meta['n_ccd']}")
    if missing:
        print(f"       obsolete/removed PDBs: {missing[:10]}{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
