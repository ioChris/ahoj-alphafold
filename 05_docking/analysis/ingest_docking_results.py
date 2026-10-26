# -*- coding: utf-8 -*-
"""
Ingest returned docking results and correlate with the apo/holo classification.

Run this AFTER the colleague returns `docking_results.csv` (schema:
05_docking/schemas/docking_results.schema.md). It:
  1. joins results to the ST_100 geometry per-pocket table on `site_uid`,
  2. derives docking outcomes (affinity goodness, redock success, dockability),
  3. computes, PER LIGAND GROUP, zero-order and *size-controlled* correlations
     against pocket_class / pocket_p_holo / pocket_separation_pair / pLDDT,
  4. (optional) cross-checks against AlphaFill clash and P2Rank TNS on shared pockets.

This mirrors the P2Rank analysis (`04_p2rank/analysis/`) so the docking panel can
be presented the same way, and it bakes in the size confound control from the
start (DOCKING_DESIGN.md §9). First-pass / extensible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional

import numpy as np
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


DEFAULT_RESULTS = cfg.DOCKING_RESULTS_DIR / "docking_results.csv"
DEFAULT_GEOM = cfg.AFDB_ST100_DATASET_GEOM_CSV
DEFAULT_OUT_DIR = cfg.DOCKING_ANALYSIS_DIR
DEFAULT_RMSD_CUTOFF = 2.0
DEFAULT_TAU = -6.0  # kcal/mol affinity threshold for the binary "dockability" flag

# geometry columns we join for correlation
GEOM_COLS = ["site_uid", "site_uniprot", "pocket_class", "pocket_p_holo", "pocket_p_apo",
             "pocket_separation_pair", "pocket_score", "model_pocket_plddt",
             "union_rg", "union_n_residues_found", "n_unp_pockets"]


def _spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return None
    xr = x[m].rank()
    yr = y[m].rank()
    if xr.std() == 0 or yr.std() == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> Optional[float]:
    """Spearman(x, y | z): correlation of rank-residuals after removing rank(z)."""
    m = x.notna() & y.notna() & z.notna()
    if m.sum() < 8:
        return None
    xs, ys, zs = x[m], y[m], z[m]
    if xs.nunique() < 2 or ys.nunique() < 2:  # constant outcome/predictor -> undefined
        return None
    xr, yr, zr = xs.rank(), ys.rank(), zs.rank()
    if zr.nunique() < 2:  # constant control -> no adjustment, fall back to zero-order
        return _spearman(xs, ys)
    zc = np.c_[np.ones(m.sum()), zr.values]

    def resid(v):
        beta, *_ = np.linalg.lstsq(zc, v.values, rcond=None)
        return v.values - zc @ beta

    rx, ry = resid(xr), resid(yr)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _class_is_holo(v: object) -> Optional[int]:
    s = str(v).lower()
    if "holo" in s:
        return 1
    if "apo" in s:
        return 0
    return None


def derive_outcomes(df: pd.DataFrame, rmsd_cutoff: float, tau: float) -> pd.DataFrame:
    df = df.copy()
    df["vina_affinity_top"] = pd.to_numeric(df.get("vina_affinity_top"), errors="coerce")
    # higher = better, so correlations with p_holo read intuitively
    df["affinity_goodness"] = -df["vina_affinity_top"]
    if "redock_rmsd_top" in df.columns:
        df["redock_rmsd_top"] = pd.to_numeric(df["redock_rmsd_top"], errors="coerce")
        df["redock_success"] = (df["redock_rmsd_top"] <= rmsd_cutoff).astype("float")
        df.loc[df["redock_rmsd_top"].isna(), "redock_success"] = np.nan
    if "pose_in_pocket" in df.columns:
        pip = df["pose_in_pocket"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["dockability"] = (pip & (df["vina_affinity_top"] <= tau)).astype("float")
    df["is_holo"] = df["pocket_class"].map(_class_is_holo)
    return df


def correlate_group(g: pd.DataFrame) -> pd.DataFrame:
    outcomes = [c for c in ["affinity_goodness", "redock_success", "dockability"] if c in g.columns]
    predictors = [c for c in ["is_holo", "pocket_p_holo", "pocket_separation_pair", "model_pocket_plddt"] if c in g.columns]
    size_col = "union_rg" if "union_rg" in g.columns else None
    rows = []
    for outc in outcomes:
        for pred in predictors:
            rho = _spearman(g[outc], g[pred])
            rho_ctrl = _partial_spearman(g[outc], g[pred], g[size_col]) if size_col else None
            rows.append({
                "outcome": outc, "predictor": pred,
                "spearman": None if rho is None else round(rho, 4),
                "partial_spearman_ctrl_size": None if rho_ctrl is None else round(rho_ctrl, 4),
                "n": int((g[outc].notna() & g[pred].notna()).sum()),
            })
    return pd.DataFrame(rows)


def success_by_class(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in [c for c in ["affinity_goodness", "redock_success", "dockability"] if c in g.columns]:
        for cls, sub in g.groupby(g["is_holo"].map({1: "AF-Holo", 0: "AF-Apo"}), dropna=True):
            v = pd.to_numeric(sub[metric], errors="coerce")
            rows.append({"metric": metric, "class": cls, "n": int(v.notna().sum()),
                         "mean": round(float(v.mean()), 4) if v.notna().any() else None})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest docking results and correlate with apo/holo class.")
    p.add_argument("--results_csv", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--geometry_csv", type=Path, default=DEFAULT_GEOM)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--rmsd_cutoff", type=float, default=DEFAULT_RMSD_CUTOFF)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(args.results_csv, low_memory=False)
    geom = pd.read_csv(args.geometry_csv, low_memory=False)
    geom = geom[[c for c in GEOM_COLS if c in geom.columns]].drop_duplicates("site_uid")

    merged = res.merge(geom, on="site_uid", how="left", suffixes=("", "_geom"))
    # prefer classification cols from geometry if results didn't carry them
    for c in ["pocket_class", "pocket_p_holo", "pocket_separation_pair", "model_pocket_plddt", "union_rg"]:
        if c not in merged.columns and f"{c}_geom" in merged.columns:
            merged[c] = merged[f"{c}_geom"]

    merged = derive_outcomes(merged, args.rmsd_cutoff, args.tau)
    merged.to_csv(args.out_dir / "docking_merged_per_pocket.csv", index=False)

    corr_frames, byclass_frames = [], []
    for grp, gdf in merged.groupby("ligand_group"):
        c = correlate_group(gdf); c.insert(0, "ligand_group", grp); corr_frames.append(c)
        b = success_by_class(gdf); b.insert(0, "ligand_group", grp); byclass_frames.append(b)

    corr = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame()
    byclass = pd.concat(byclass_frames, ignore_index=True) if byclass_frames else pd.DataFrame()
    corr.to_csv(args.out_dir / "docking_correlations.csv", index=False)
    byclass.to_csv(args.out_dir / "docking_success_by_class.csv", index=False)

    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "results_csv": str(args.results_csv),
        "geometry_csv": str(args.geometry_csv),
        "rmsd_cutoff": args.rmsd_cutoff, "tau": args.tau,
        "n_jobs": int(len(merged)),
        "n_pockets": int(merged["site_uid"].nunique()),
        "groups": sorted(int(g) for g in merged["ligand_group"].dropna().unique()),
        "note": "Size-controlled column is partial Spearman controlling union_rg. "
                "For the manuscript, add controlled-pairs + CMH by size quantile "
                "(mirror 04_p2rank/analysis/analyze_p2rank_controlled_pairs.py) and "
                "control ligand heavy-atom count for groups 1-2.",
    }
    with open(args.out_dir / "docking_ingest_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVE] {args.out_dir / 'docking_merged_per_pocket.csv'}")
    print(f"[SAVE] {args.out_dir / 'docking_correlations.csv'}")
    print(f"[SAVE] {args.out_dir / 'docking_success_by_class.csv'}")
    print(f"[JOBS] {len(merged)}  [POCKETS] {merged['site_uid'].nunique()}  [GROUPS] {meta['groups']}")


if __name__ == "__main__":
    main()
