# -*- coding: utf-8 -*-
"""
Compute symmetry-corrected redock RMSD for docked poses vs experimental references.

Joins docked pose SDFs (`poses/<job_id>.sdf`, all Vina modes) to reference pose SDFs
(`ref_poses/<job_id>.sdf`, experimental ligand in the AF frame) and computes, per job:
  redock_rmsd_top  = in-place symmetry RMSD of the TOP mode to the reference,
  redock_rmsd_best = best (min) over all returned modes.
No re-superposition (both poses share the AF frame). Writes `redock_rmsd.csv`.

Usage:
    python compute_redock_rmsd.py --poses <poses_dir> --refs <ref_poses_dir> \
        --out <redock_rmsd.csv> [--workers N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem

from rmsd_utils import symmetry_rmsd_inplace


def _load_multi(sdf: Path):
    """Load an SDF as a single mol carrying all records as conformers (docked modes)."""
    supp = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True)
    mols = [m for m in supp if m is not None]
    if not mols:
        supp = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
        mols = [m for m in supp if m is not None]
    if not mols:
        return None
    base = mols[0]
    for extra in mols[1:]:
        if extra.GetNumAtoms() == base.GetNumAtoms():
            base.AddConformer(extra.GetConformer(), assignId=True)
    return base


def _one(job_id: str, pose_path: str, ref_path: str) -> dict:
    out = {"job_id": job_id, "redock_rmsd_top": None, "redock_rmsd_best": None,
           "n_isomorphisms": 0, "rmsd_status": ""}
    try:
        docked = _load_multi(Path(pose_path))
        ref = _load_multi(Path(ref_path))
        if docked is None or ref is None:
            out["rmsd_status"] = "load_failed"; return out
        top, best, niso, status = symmetry_rmsd_inplace(docked, ref)
        out.update(redock_rmsd_top=None if top is None else round(top, 3),
                   redock_rmsd_best=None if best is None else round(best, 3),
                   n_isomorphisms=niso, rmsd_status=status)
    except Exception as exc:
        out["rmsd_status"] = f"{type(exc).__name__}: {exc}"
    return out


def _worker(args):
    return _one(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    refs = {p.stem: p for p in args.refs.glob("*.sdf")}
    tasks = []
    for jid, ref in refs.items():
        pose = args.poses / f"{jid}.sdf"
        if pose.exists():
            tasks.append((jid, str(pose), str(ref)))
    print(f"[RMSD] {len(tasks)} jobs with both a pose and a reference  workers={args.workers}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 100 == 0 or i == len(tasks):
                nok = sum(r["rmsd_status"] == "ok" for r in rows)
                print(f"  {i}/{len(tasks)}  ok={nok}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if len(df):
        ok = df[df["rmsd_status"] == "ok"]
        succ = (pd.to_numeric(ok["redock_rmsd_top"], errors="coerce") <= 2.0).mean() if len(ok) else float("nan")
        print(f"[DONE] {len(df)} jobs  ok={len(ok)}  median top RMSD="
              f"{pd.to_numeric(ok['redock_rmsd_top'], errors='coerce').median():.2f}  "
              f"top<=2A success={succ:.1%}  -> {args.out}")


if __name__ == "__main__":
    main()
