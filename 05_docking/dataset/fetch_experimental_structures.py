# -*- coding: utf-8 -*-
"""
Download the experimental holo structures that "escort" the AFDB models.

Every pocket has >=1 AHoJ-DB query structure (a real PDB holo entry). This fetches
them (gzipped mmCIF from RCSB) and writes a per-job linking table so each docking
job points to its escort structure + the receptor chain + the bound-ligand location.

Escort choice per manifest job:
  - group 1 (query holo): the ligand's OWN structure (parsed from `ligand_ref` job key)
    — this is the exact holo the docked CCD came from (best for redock reference / AF-vs-PDB).
  - group 2/3 (AlphaFill / probe): the pocket's REPRESENTATIVE query structure (rep_job).

Outputs (under --out_dir):
  experimental_holo/<pdb>.cif.gz              # deduped by PDB id
  experimental_structures.csv                 # job_id -> escort pdb/chain/ligand + status

Notes:
  - Uses RCSB `https://files.rcsb.org/download/{pdb}.cif.gz`. Obsolete/removed entries
    return 404 -> logged as status="missing" (the user's "hopefully still in the PDB" caveat).
  - Receptor chain is `prot_chains` (NOT the ligand chain) — from QUERY_SUMMARIES via
    the receptor-attached query table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

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

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # allow sibling imports
from parse_ahoj_job_ligands import parse_job_key

RCSB = "https://files.rcsb.org/download/{pdb}.cif.gz"
DEFAULT_MANIFEST = cfg.DOCKING_PILOT_DIR / "docking_manifest.csv"
DEFAULT_QUERY = cfg.DOCKING_MANIFEST_DIR / "query_ligands_group1.csv"
DEFAULT_OUT = cfg.DOCKING_PILOT_DIR


def build_escort_rows(man: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """One escort record per manifest job."""
    # rep query per site (is_rep True), and prot_chains per job key
    rep = q[q.get("is_rep", False) == True] if "is_rep" in q.columns else q
    rep_by_site: Dict[str, dict] = {}
    for r in rep.itertuples(index=False):
        rep_by_site.setdefault(str(r.site_uid), {
            "pdb": str(getattr(r, "src_pdb", "")), "prot": str(getattr(r, "prot_chains", "")),
            "lig_chain": str(getattr(r, "src_chain", "")), "lig_id": str(getattr(r, "ligand_id", "")),
            "lig_resnum": str(getattr(r, "src_resnum", "")), "ur": str(getattr(r, "uniprot_residues", "")),
        })
    prot_by_ref, ur_by_ref = {}, {}
    if "ligand_ref" in q.columns and "prot_chains" in q.columns:
        prot_by_ref = dict(zip(q["ligand_ref"].astype(str), q["prot_chains"].astype(str)))
    if "ligand_ref" in q.columns and "uniprot_residues" in q.columns:
        ur_by_ref = dict(zip(q["ligand_ref"].astype(str), q["uniprot_residues"].astype(str)))

    rows = []
    for j in man.itertuples(index=False):
        grp = int(j.ligand_group)
        site = str(j.site_uid)
        if grp == 1:
            rec = parse_job_key(str(j.ligand_ref))
            if rec:
                pdb, lig_chain, lig_id, lig_resnum = rec["pdb"], rec["chain"], rec["ccd"], rec["resnum"]
                prot = prot_by_ref.get(str(j.ligand_ref), "")
                ur = ur_by_ref.get(str(j.ligand_ref), "")
            else:
                pdb = prot = lig_chain = lig_id = lig_resnum = ur = ""
        else:
            rp = rep_by_site.get(site, {})
            pdb, prot = rp.get("pdb", ""), rp.get("prot", "")
            lig_chain, lig_id, lig_resnum = rp.get("lig_chain", ""), rp.get("lig_id", ""), rp.get("lig_resnum", "")
            ur = rp.get("ur", "")
        rows.append({
            "job_id": j.job_id, "site_uid": site, "ligand_group": grp, "ligand_id": j.ligand_id,
            "escort_pdb_id": pdb, "escort_prot_chains": prot,
            "escort_lig_chain": lig_chain, "escort_lig_id": lig_id, "escort_lig_resnum": lig_resnum,
            "escort_uniprot_residues": ur,
        })
    return pd.DataFrame(rows)


def fetch_pdb_gz(pdb: str, out_dir: Path, overwrite: bool, timeout: int = 60) -> str:
    out = out_dir / f"{pdb.lower()}.cif.gz"
    if out.exists() and not overwrite and out.stat().st_size > 0:
        return "cached"
    try:
        urllib.request.urlretrieve(RCSB.format(pdb=pdb.lower()), out)
        return "ok"
    except urllib.error.HTTPError as e:
        return f"missing_http_{e.code}"
    except Exception as e:
        return f"error_{type(e).__name__}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download experimental holo structures escorting the AFDB models.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--query_ligands", type=Path, default=DEFAULT_QUERY)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sleep", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    holo_dir = args.out_dir / "experimental_holo"
    holo_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest, low_memory=False)
    q = pd.read_csv(args.query_ligands, low_memory=False)
    escort = build_escort_rows(man, q)

    # download unique PDBs
    pdbs = sorted({p for p in escort["escort_pdb_id"].astype(str) if p and p.lower() != "nan"})
    print(f"[FETCH] {len(pdbs)} unique experimental structures -> {holo_dir}")
    status: Dict[str, str] = {}
    for i, pdb in enumerate(pdbs, 1):
        status[pdb] = fetch_pdb_gz(pdb, holo_dir, args.overwrite)
        if args.sleep:
            time.sleep(args.sleep)
        if i % 100 == 0:
            print(f"  {i}/{len(pdbs)}")

    escort["escort_cif_gz"] = escort["escort_pdb_id"].map(
        lambda p: f"experimental_holo/{str(p).lower()}.cif.gz" if p else "")
    escort["escort_status"] = escort["escort_pdb_id"].map(lambda p: status.get(str(p), "no_pdb"))
    out_csv = args.out_dir / "experimental_structures.csv"
    escort.to_csv(out_csv, index=False)

    ok = sum(1 for v in status.values() if v in ("ok", "cached"))
    missing = sorted(p for p, v in status.items() if v.startswith("missing"))
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "manifest": str(args.manifest), "out_dir": str(args.out_dir),
        "unique_pdbs": len(pdbs), "downloaded_ok": ok, "missing": len(missing),
        "missing_pdbs": missing[:50],
        "jobs_linked": int(len(escort)),
    }
    with open(args.out_dir / "experimental_structures_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVE] {out_csv}")
    print(f"[DONE] structures ok={ok}/{len(pdbs)}  missing/obsolete={len(missing)}  jobs linked={len(escort)}")
    if missing:
        print(f"       {len(missing)} PDBs unavailable (obsolete/removed): {missing[:8]}{' ...' if len(missing)>8 else ''}")


if __name__ == "__main__":
    main()
