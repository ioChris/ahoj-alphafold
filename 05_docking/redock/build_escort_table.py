# -*- coding: utf-8 -*-
"""
Build the G1 redock escort table (run_3) — everything the reference-pose build needs,
resolved locally with NO network access.

For each ligand_group==1 job in the docking manifest:
  * parse `ligand_ref` ({pdb}-{lig_chain}-{CCD}-{resnum})  -> the experimental ligand;
  * join QUERY_SUMMARIES_CLASS (ligand_ref == job_name)    -> receptor `prot_chains`
    (100% G1 coverage; the job-key chain is the LIGAND chain, not the protein one);
  * pocket residue positions from the manifest `pocket_chain_resi` (AF/UniProt numbering);
  * mark `has_pose` from the docking results (only status==ok jobs have a docked pose).

Output: <out_dir>/escort_g1.csv  (+ escort_g1_manifest.json).

Env: `ahoj-docking` (pandas). No gemmi/rdkit needed here.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
from pathlib import Path
from typing import List

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
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "dataset"))
from parse_ahoj_job_ligands import parse_job_key  # noqa: E402

RUN3_DIR = cfg.DOCKING_RUN3_DIR


def _pocket_positions(chain_resi: object) -> List[int]:
    """'A_85 A_90 ...' -> [85, 90, ...] (AF chain-A residue numbers == UniProt positions)."""
    out: List[int] = []
    for tok in str(chain_resi).split():
        if "_" in tok:
            try:
                out.append(int(tok.rsplit("_", 1)[1]))
            except ValueError:
                continue
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the G1 redock escort table (run_3).")
    p.add_argument("--manifest", type=Path, default=RUN3_DIR / "manifest" / "docking_manifest.csv")
    p.add_argument("--results", type=Path, default=RUN3_DIR / "results" / "docking_results_run3.csv")
    p.add_argument("--query_summaries", type=Path, default=cfg.QUERY_SUMMARIES_CLASS_CSV)
    p.add_argument("--out_dir", type=Path, default=RUN3_DIR / "reference_poses")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest, low_memory=False)
    g1 = m[m["ligand_group"] == 1].copy()
    print(f"[MANIFEST] G1 rows: {len(g1)}")

    # parse ligand_ref
    recs = g1["ligand_ref"].map(parse_job_key)
    g1["src_pdb"] = [r["pdb"] if r else "" for r in recs]
    g1["src_lig_chain"] = [r["chain"] if r else "" for r in recs]
    g1["src_ccd"] = [r["ccd"] if r else "" for r in recs]
    g1["src_resnum"] = [int(r["resnum"]) if r else -1 for r in recs]
    bad = int((g1["src_pdb"] == "").sum())
    if bad:
        print(f"[WARN] {bad} G1 rows with unparseable ligand_ref")

    # receptor protein chain from QUERY_SUMMARIES (join ligand_ref == job_name)
    qs = pd.read_csv(args.query_summaries,
                     usecols=lambda c: c in {"job_name", "prot_chains", "prot_chains_num"},
                     low_memory=False).rename(columns={"job_name": "ligand_ref"})
    qs["prot_chain"] = qs["prot_chains"].astype(str).str.split().str[0]
    g1 = g1.merge(qs[["ligand_ref", "prot_chain", "prot_chains_num"]], on="ligand_ref", how="left")
    miss_chain = int(g1["prot_chain"].isna().sum())
    print(f"[QS] receptor chain resolved: {len(g1) - miss_chain}/{len(g1)}  (missing {miss_chain})")

    # pocket positions (AF/UniProt numbering)
    g1["pocket_positions"] = g1["pocket_chain_resi"].map(lambda s: json.dumps(_pocket_positions(s)))

    # has_pose from docking results
    res = pd.read_csv(args.results, usecols=["job_id", "ligand_group", "status"], low_memory=False)
    ok = set(res[(res["ligand_group"] == 1) & (res["status"] == "ok")]["job_id"])
    g1["has_pose"] = g1["job_id"].isin(ok)
    print(f"[RESULTS] G1 with docked pose (ok): {int(g1['has_pose'].sum())}/{len(g1)}")

    cols = ["job_id", "site_uid", "site_uniprot", "afdb_filename", "af_model_len",
            "ligand_id", "ligand_ref", "src_pdb", "src_lig_chain", "src_ccd", "src_resnum",
            "prot_chain", "prot_chains_num", "pocket_positions", "pocket_n_residues",
            "has_pose", "pocket_class", "pocket_p_apo", "pocket_p_holo"]
    cols = [c for c in cols if c in g1.columns]
    out = g1[cols].copy()
    out_csv = args.out_dir / "escort_g1.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "manifest": str(args.manifest), "results": str(args.results),
        "n_g1": int(len(out)), "n_with_pose": int(out["has_pose"].sum()),
        "n_unique_pdb": int(out["src_pdb"].nunique()),
        "n_unique_pdb_with_pose": int(out[out["has_pose"]]["src_pdb"].nunique()),
        "n_unique_ccd": int(out["ligand_id"].nunique()),
        "receptor_chain_missing": miss_chain, "ligand_ref_unparsed": bad,
    }
    with open(args.out_dir / "escort_g1_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[SAVE] {out_csv}")
    print(f"[DONE] {meta['n_with_pose']} with-pose jobs, {meta['n_unique_pdb_with_pose']} unique PDBs, "
          f"{meta['n_unique_ccd']} CCDs")


if __name__ == "__main__":
    main()
