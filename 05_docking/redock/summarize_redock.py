# -*- coding: utf-8 -*-
"""
QC summary of the run_3 redock RMSD build (Analysis 6, G1).

Reports build status, reference-frame quality, and the redock-RMSD distribution split
by apo/holo class and pose_in_pocket — enough to sanity-check the build and preview the
AF-Holo>AF-Apo pose-recovery signal. The full size-controlled correlations live in
`runner/analyze_docking_pilot.py --redock_csv`.

Spearman is computed manually (rank + numpy) to avoid the scipy/OpenMP native crash
seen on this Windows box.

Env: `ahoj-docking`.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

RUN3_DIR = cfg.DOCKING_RUN3_DIR


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank-based Spearman, explicit formula (no scipy, no np.corrcoef/BLAS — those
    native-crash via the duplicate OpenMP DLL on this Windows box)."""
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy(dtype=float, copy=True)
    rb = pd.Series(b[m]).rank().to_numpy(dtype=float, copy=True)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redock_csv", type=Path, default=RUN3_DIR / "reference_poses" / "redock_rmsd.csv")
    ap.add_argument("--escort_csv", type=Path, default=RUN3_DIR / "reference_poses" / "escort_g1.csv")
    ap.add_argument("--results", type=Path, default=RUN3_DIR / "results" / "docking_results_run3.csv")
    ap.add_argument("--manifest", type=Path, default=RUN3_DIR / "manifest" / "docking_manifest.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.redock_csv, low_memory=False)
    esc = pd.read_csv(args.escort_csv, usecols=["job_id", "pocket_class", "pocket_p_holo"], low_memory=False)
    res = pd.read_csv(args.results, usecols=["job_id", "affinity_top", "pose_in_pocket", "n_pose_atoms_in_pocket"],
                      low_memory=False)
    man = pd.read_csv(args.manifest, usecols=["job_id", "size_x"], low_memory=False)
    d = d.drop(columns=[c for c in ("affinity_top",) if c in d.columns])
    d = d.merge(esc, on="job_id", how="left").merge(res, on="job_id", how="left").merge(man, on="job_id", how="left")
    d["rmsd"] = pd.to_numeric(d["redock_rmsd_top"], errors="coerce")

    lines = []
    def p(s): lines.append(s); print(s)

    p(f"[BUILD] {len(d)} jobs")
    p(f"  status: {d['rmsd_status'].value_counts().to_dict()}")
    ok = d[d["rmsd_status"] == "ok"].copy()
    p(f"  scored ok: {len(ok)} ({len(ok)/len(d)*100:.1f}%)")

    sp = pd.to_numeric(ok["superpose_rmsd"], errors="coerce")
    p(f"\n[FRAME] superpose RMSD median={sp.median():.3f}  p90={sp.quantile(.9):.3f}")
    p(f"  wholechain fallbacks: {(ok['superpose_mode']=='wholechain').sum()} "
      f"({(ok['superpose_mode']=='wholechain').mean()*100:.1f}%)")
    p(f"  bad frame (superpose_rmsd>3A): {(sp>3).sum()} ({(sp>3).mean()*100:.1f}%)")
    rc = pd.to_numeric(ok["ref_completeness"], errors="coerce")
    p(f"  ref_completeness median={rc.median():.2f}  <0.9: {(rc<0.9).mean()*100:.1f}%")

    # strict set: good frame only
    good = ok[(ok["superpose_mode"] == "pocket") & (sp <= 2.0)].copy()
    p(f"\n[RMSD] over ok jobs (n={len(ok)}):")
    for lab, s in [("all ok", ok), ("good-frame", good)]:
        r = pd.to_numeric(s["rmsd"], errors="coerce")
        p(f"  {lab:12s} n={len(s):6d}  median={r.median():.2f}  "
          + "  ".join(f"<={t}A={ (r<=t).mean()*100:4.1f}%" for t in (2, 3, 5)))

    p(f"\n[BY CLASS] good-frame jobs:")
    for cls, g in good.groupby("pocket_class"):
        r = pd.to_numeric(g["rmsd"], errors="coerce")
        p(f"  {str(cls):5s} n={len(g):6d}  median={r.median():.2f}  <=2A={(r<=2).mean()*100:4.1f}%  <=3A={(r<=3).mean()*100:4.1f}%")

    p(f"\n[BY pose_in_pocket] good-frame jobs:")
    for pip, g in good.groupby("pose_in_pocket"):
        r = pd.to_numeric(g["rmsd"], errors="coerce")
        p(f"  in_pocket={pip}: n={len(g):6d}  median={r.median():.2f}  <=2A={(r<=2).mean()*100:4.1f}%")

    p(f"\n[SPEARMAN] good-frame jobs (RMSD vs ...):")
    p(f"  p_holo               : {spearman(good['rmsd'], good['pocket_p_holo']):+.3f}  (neg = holo docks better)")
    p(f"  box size_x           : {spearman(good['rmsd'], good['size_x']):+.3f}")
    p(f"  ligand size (tmpl)   : {spearman(good['rmsd'], pd.to_numeric(good['n_template_atoms'],errors='coerce')):+.3f}")
    p(f"  n_pose_atoms_in_pocket: {spearman(good['rmsd'], good['n_pose_atoms_in_pocket']):+.3f}")
    p(f"  affinity_top         : {spearman(good['rmsd'], good['affinity_top']):+.3f}")

    summary = {
        "n_jobs": int(len(d)), "status_counts": d["rmsd_status"].value_counts().to_dict(),
        "n_ok": int(len(ok)), "n_good_frame": int(len(good)),
        "median_rmsd_ok": float(pd.to_numeric(ok["rmsd"], errors="coerce").median()),
        "success_2A_ok": float((pd.to_numeric(ok["rmsd"], errors="coerce") <= 2).mean()),
        "success_2A_goodframe": float((pd.to_numeric(good["rmsd"], errors="coerce") <= 2).mean()),
        "median_superpose_rmsd": float(sp.median()),
        "by_class_goodframe": {str(c): {"n": int(len(g)),
                               "median_rmsd": float(pd.to_numeric(g["rmsd"], errors="coerce").median()),
                               "success_2A": float((pd.to_numeric(g["rmsd"], errors="coerce") <= 2).mean())}
                               for c, g in good.groupby("pocket_class")},
        "spearman_rmsd_p_holo_goodframe": spearman(good["rmsd"], good["pocket_p_holo"]),
    }
    out = args.redock_csv.parent / "redock_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    p(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
