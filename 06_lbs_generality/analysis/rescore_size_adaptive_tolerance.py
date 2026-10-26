# -*- coding: utf-8 -*-
"""
Size-adaptive (per-site Rg) DCA tolerance re-evaluation for the LBS-generality panel.
=====================================================================================

RETROACTIVE re-scoring only -- NO tool re-runs. Replaces the fixed 4 A DCA tolerance
with a per-site tolerance = k * union_rg (k in {0.5, 1.0, 2.0}) applied IDENTICALLY to
every predictor, and reports the AF-Holo vs AF-Apo TNS gradient side-by-side against the
fixed-4 A reference. The point is a built-in pocket-SIZE-confound control: if the
holo>apo gradient survives a tolerance that scales with pocket size, the effect is not a
size artifact (complements the size-matched 1,494 pilot).

  * Fixed-4 A column   : read from the EXISTING per-tool merged tables
                         (`.../{tool}/(full|.)/lbs_{tool}_st100_merged_per_pocket.csv`).
  * k*Rg columns       : re-score the EXISTING predictions with
                         score(..., tolerance_col='union_rg', tolerance_scale=k).
  * Native P2Rank      : SAS-eval TNS from the manuscript P2Rank table, as a LABELLED
                         reference (looser ground truth; NOT re-scorable at k*Rg).

Population (apples-to-apples): every cell is restricted to sites with a finite union_rg
(so the fixed and adaptive modes share one denominator), n ~= 86,946 (apo 34,900 /
holo 51,751). The balanced pilot is the 1,494 size-matched pockets in
`pilot/pilot_pockets.csv` (747 apo / 747 holo), all finite-Rg.

Outputs (under LBS_DIR/analysis_v1/tables/):
  lbs_size_adaptive_tolerance_long.csv   -- tidy one-row-per (tool, scoring, population, class)
  lbs_size_adaptive_tolerance_table.csv  -- wide side-by-side (apo/holo/gap per scoring)
  lbs_size_adaptive_tolerance_manifest.json

Base env (pandas + numpy). Run from anywhere:
    python 06_lbs_generality/analysis/rescore_size_adaptive_tolerance.py
"""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# --- repo root on sys.path so `common` + the scorer import ---------------------
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = None
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        _REPO_ROOT = _root
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg
_SCORE_DIR = _REPO_ROOT / "06_lbs_generality" / "score"
if str(_SCORE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCORE_DIR))
import compute_lbs_success as C  # noqa: E402

THRESHOLD = "ST_100"
KS = [0.5, 1.0, 2.0]
TOL_COL = "union_rg"

# tool -> (predictions.csv, existing fixed-4A merged table)
LBS = cfg.LBS_DIR
TOOLS = {
    "ifsitepred": (cfg.lbs_tool_dir("ifsitepred") / "predictions.csv",
                   cfg.lbs_tool_dir("ifsitepred") / "lbs_ifsitepred_st100_merged_per_pocket.csv"),
    "fpocket":    (cfg.lbs_tool_dir("fpocket") / "predictions.csv",
                   cfg.lbs_tool_dir("fpocket") / "lbs_fpocket_st100_merged_per_pocket.csv"),
    "p2rank":     (cfg.lbs_tool_dir("p2rank") / "predictions.csv",
                   cfg.lbs_tool_dir("p2rank") / "lbs_p2rank_st100_merged_per_pocket.csv"),
}


def summarize(df: pd.DataFrame, *, mask: pd.Series, cutoff_col: str | None) -> dict:
    """apo/holo TNS% + N + gap on df[mask], plus median effective cutoff (A)."""
    d = df[mask]
    out = {}
    for cls in ("apo", "holo"):
        v = pd.to_numeric(d.loc[d["pocket_class"] == cls, "success_tns"], errors="coerce")
        out[f"{cls}_n"] = int(v.notna().sum())
        out[f"{cls}_tns"] = None if v.notna().sum() == 0 else round(float(v.mean()) * 100, 1)
    a, h = out["apo_tns"], out["holo_tns"]
    out["gap_holo_minus_apo"] = None if (a is None or h is None) else round(h - a, 1)
    if cutoff_col is not None and cutoff_col in d.columns:
        cu = pd.to_numeric(d[cutoff_col], errors="coerce")
        out["median_cutoff_A"] = None if cu.notna().sum() == 0 else round(float(cu.median()), 2)
    else:
        out["median_cutoff_A"] = 4.0  # fixed-4A reference
    return out


def main() -> None:
    out_dir = cfg.LBS_ANALYSIS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] geometry + ground-truth atoms ...", flush=True)
    geom = pd.read_csv(cfg.AFDB_ST100_DATASET_GEOM_CSV, low_memory=False)
    gt = C.load_ground_truth_npz(cfg.LBS_GROUND_TRUTH_NPZ)
    pilot_uids = set(pd.read_csv(cfg.LBS_PILOT_DIR / "pilot_pockets.csv")["site_uid"].astype(str).str.strip())
    print(f"[INFO] pilot pockets: {len(pilot_uids)}", flush=True)

    rows = []  # tidy long records

    def emit(tool, scoring, k, df, cutoff_col):
        df = df.copy()
        df["site_uid"] = df["site_uid"].astype(str).str.strip()
        df["pocket_class"] = df["pocket_class"].astype(str).str.lower()
        rg = pd.to_numeric(df.get("rg", df.get(TOL_COL)), errors="coerce")
        finite_rg = rg.notna()
        for pop, mask in (("full", finite_rg),
                          ("pilot", finite_rg & df["site_uid"].isin(pilot_uids))):
            s = summarize(df, mask=mask, cutoff_col=cutoff_col)
            rows.append({"tool": tool, "scoring": scoring, "k": k, "population": pop, **s})

    for tool, (pred_csv, fixed_csv) in TOOLS.items():
        print(f"\n[TOOL] {tool}", flush=True)
        # --- fixed 4 A reference (existing merged table) ---
        fx = pd.read_csv(fixed_csv, low_memory=False)
        fx = fx[fx["threshold"] == THRESHOLD] if "threshold" in fx.columns else fx
        emit(tool, "fixed_4A", 4.0, fx[["site_uid", "pocket_class", "success_tns", "rg"]], cutoff_col=None)
        del fx

        # --- k*Rg size-adaptive re-scores ---
        preds = pd.read_csv(pred_csv, low_memory=False)
        for k in KS:
            print(f"   [SCORE] {tool} k={k} * {TOL_COL} ...", flush=True)
            m = C.score(preds, geom, gt, tool=tool, threshold=THRESHOLD,
                        primary_cutoff=4.0, holo_plus_one=1.0,
                        tolerance_col=TOL_COL, tolerance_scale=k)
            emit(tool, f"rg_k{k:g}", k,
                 m[["site_uid", "pocket_class", "success_tns", "rg", "dca_cutoff_used"]],
                 cutoff_col="dca_cutoff_used")
            del m
        del preds

    # --- native P2Rank SAS-eval reference (labelled; fixed only, not re-scorable) ---
    if Path(cfg.P2RANK_MERGED_PER_POCKET_CSV).is_file():
        nat = pd.read_csv(cfg.P2RANK_MERGED_PER_POCKET_CSV,
                          usecols=["threshold", "pocket_class", "success_tns", "rg"], low_memory=False)
        nat = nat[nat["threshold"] == THRESHOLD].copy()
        nat["pocket_class"] = nat["pocket_class"].astype(str).str.lower()
        rgn = pd.to_numeric(nat["rg"], errors="coerce")
        s = summarize(nat, mask=rgn.notna(), cutoff_col=None)
        rows.append({"tool": "p2rank_native_saseval", "scoring": "native_fixed_4A_SASeval",
                     "k": 4.0, "population": "full", **s})

    long_df = pd.DataFrame(rows)
    long_csv = out_dir / "lbs_size_adaptive_tolerance_long.csv"
    long_df.to_csv(long_csv, index=False)

    # --- wide side-by-side: rows = (tool, population), cols = scoring x {apo,holo,gap} ---
    wide = long_df.pivot_table(index=["tool", "population"], columns="scoring",
                               values=["apo_tns", "holo_tns", "gap_holo_minus_apo"], aggfunc="first")
    wide_csv = out_dir / "lbs_size_adaptive_tolerance_table.csv"
    wide.to_csv(wide_csv)

    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "threshold": THRESHOLD, "tolerance_col": TOL_COL, "ks": KS,
        "population": "sites with finite union_rg (full ~86,946; pilot 1,494 balanced)",
        "tools": list(TOOLS.keys()),
        "note": "Per-site DCA cutoff = k * union_rg, applied identically to every tool. "
                "success_tns / dca_rank use the per-site cutoff; absolute DCA-sweep columns unchanged. "
                "Native P2Rank row is SAS-eval (looser) fixed-4A reference, NOT re-scorable at k*Rg.",
    }
    with (out_dir / "lbs_size_adaptive_tolerance_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # --- console report ---
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n================  SIZE-ADAPTIVE DCA TOLERANCE  ================")
    for pop in ("full", "pilot"):
        print(f"\n--- population: {pop} ---")
        sub = long_df[long_df["population"] == pop]
        show = sub[["tool", "scoring", "median_cutoff_A", "apo_tns", "apo_n",
                    "holo_tns", "holo_n", "gap_holo_minus_apo"]]
        print(show.to_string(index=False))
    print(f"\n[SAVE] {long_csv}")
    print(f"[SAVE] {wide_csv}")


if __name__ == "__main__":
    main()
