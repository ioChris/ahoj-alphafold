# -*- coding: utf-8 -*-
"""
Floored size-adaptive DCA tolerance: cutoff = max(floor, k * union_rg).
=======================================================================

Follow-up to `rescore_size_adaptive_tolerance.py`. The pure per-site tolerance k*Rg lets
SMALL pockets collapse to a sub-Angstrom cutoff (median Rg ~7.5 A, so k=0.5 hands ~62% of
sites a <4 A cutoff -> absolute TNS dips BELOW the fixed-4 A baseline). A FLOOR fixes that:

    per-site cutoff = max(floor, k * union_rg)          floor in {3, 4} A, k in {0.5, 1.0}

Small pockets keep at least `floor` A; only genuinely large pockets loosen with size. This is
a cleaner size-normalized criterion (behaves like fixed-`floor` for small/medium sites, size-
adaptive only in the tail) and a stronger size-confound control. NO tool re-runs -- re-scores
the EXISTING predictions only, identically for every tool.

Baseline rows (fixed_4A, rg_k0.5, rg_k1 -- no floor) are read from the sibling run's
`lbs_size_adaptive_tolerance_long.csv` if present (else recomputed). The 4 floored combos are
scored here. Population = finite-Rg sites (full ~86,946; balanced pilot 1,494).

Outputs (LBS_DIR/analysis_v1/tables/):
  lbs_size_adaptive_floor_long.csv   -- tidy (tool, scoring, population, class)
  lbs_size_adaptive_floor_table.csv  -- wide side-by-side (apo/holo/gap per scoring)
  lbs_size_adaptive_floor_manifest.json

Base env (pandas + numpy):
    python 06_lbs_generality/analysis/rescore_size_adaptive_floor.py
"""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

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
_ANALYSIS_DIR = _REPO_ROOT / "06_lbs_generality" / "analysis"
for _d in (_SCORE_DIR, _ANALYSIS_DIR):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
import compute_lbs_success as C  # noqa: E402
from rescore_size_adaptive_tolerance import summarize, TOOLS, THRESHOLD, TOL_COL  # noqa: E402

# (k, floor) combos requested: max(floor, k*Rg)
FLOOR_COMBOS = [(0.5, 3.0), (0.5, 4.0), (1.0, 3.0), (1.0, 4.0)]
BASELINE_SCORINGS = ["fixed_4A", "rg_k0.5", "rg_k1"]  # reused from the sibling run
# display order for the wide table
SCORING_ORDER = ["fixed_4A", "rg_k0.5", "rg_k0.5_floor3", "rg_k0.5_floor4",
                 "rg_k1", "rg_k1_floor3", "rg_k1_floor4"]


def main() -> None:
    out_dir = cfg.LBS_ANALYSIS_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] geometry + ground-truth atoms ...", flush=True)
    geom = pd.read_csv(cfg.AFDB_ST100_DATASET_GEOM_CSV, low_memory=False)
    gt = C.load_ground_truth_npz(cfg.LBS_GROUND_TRUTH_NPZ)
    pilot_uids = set(pd.read_csv(cfg.LBS_PILOT_DIR / "pilot_pockets.csv")["site_uid"].astype(str).str.strip())

    rows = []

    # --- baseline (no floor): reuse the sibling run's long table if available ---
    sib = out_dir / "lbs_size_adaptive_tolerance_long.csv"
    if sib.is_file():
        base = pd.read_csv(sib)
        base = base[base["scoring"].isin(BASELINE_SCORINGS)].copy()
        rows.extend(base.to_dict("records"))
        print(f"[INFO] reused {len(base)} baseline rows from {sib.name}", flush=True)
    else:
        print("[WARN] sibling long table not found; baseline (no-floor) rows will be missing.", flush=True)

    def emit(tool, scoring, k, floor, df):
        df = df.copy()
        df["site_uid"] = df["site_uid"].astype(str).str.strip()
        df["pocket_class"] = df["pocket_class"].astype(str).str.lower()
        finite_rg = pd.to_numeric(df.get("rg", df.get(TOL_COL)), errors="coerce").notna()
        for pop, mask in (("full", finite_rg),
                          ("pilot", finite_rg & df["site_uid"].isin(pilot_uids))):
            s = summarize(df, mask=mask, cutoff_col="dca_cutoff_used")
            rows.append({"tool": tool, "scoring": scoring, "k": k, "floor": floor,
                         "population": pop, **s})

    # --- floored combos: score the existing predictions ---
    for tool, (pred_csv, _fixed_csv) in TOOLS.items():
        print(f"\n[TOOL] {tool}", flush=True)
        preds = pd.read_csv(pred_csv, low_memory=False)
        for k, floor in FLOOR_COMBOS:
            label = f"rg_k{k:g}_floor{floor:g}"
            print(f"   [SCORE] {tool} max({floor:g}, {k:g}*{TOL_COL}) ...", flush=True)
            m = C.score(preds, geom, gt, tool=tool, threshold=THRESHOLD,
                        primary_cutoff=4.0, holo_plus_one=1.0,
                        tolerance_col=TOL_COL, tolerance_scale=k, tolerance_floor=floor)
            emit(tool, label, k, floor,
                 m[["site_uid", "pocket_class", "success_tns", "rg", "dca_cutoff_used"]])
            del m
        del preds

    long_df = pd.DataFrame(rows)
    # keep floor column tidy for baseline rows (no floor)
    if "floor" not in long_df.columns:
        long_df["floor"] = np.nan
    long_csv = out_dir / "lbs_size_adaptive_floor_long.csv"
    long_df.to_csv(long_csv, index=False)

    order = pd.Categorical(long_df["scoring"], categories=SCORING_ORDER, ordered=True)
    long_df = long_df.assign(_o=order).sort_values(["population", "tool", "_o"]).drop(columns="_o")

    wide = long_df.pivot_table(index=["tool", "population"], columns="scoring",
                               values=["apo_tns", "holo_tns", "gap_holo_minus_apo"], aggfunc="first")
    wide = wide.reindex(columns=[c for c in SCORING_ORDER if c in long_df["scoring"].unique()], level=1)
    wide_csv = out_dir / "lbs_size_adaptive_floor_table.csv"
    wide.to_csv(wide_csv)

    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "threshold": THRESHOLD, "tolerance_col": TOL_COL,
        "rule": "per-site DCA cutoff = max(floor, k * union_rg)",
        "floor_combos": [{"k": k, "floor": f} for k, f in FLOOR_COMBOS],
        "population": "sites with finite union_rg (full ~86,946; pilot 1,494 balanced)",
        "tools": list(TOOLS.keys()),
        "note": "Floor lifts small-Rg sites back to >= floor A so they don't collapse; large sites still "
                "scale with k*Rg. Baseline (no-floor) rows reused from lbs_size_adaptive_tolerance_long.csv.",
    }
    with (out_dir / "lbs_size_adaptive_floor_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("\n============  FLOORED SIZE-ADAPTIVE DCA TOLERANCE  max(floor, k*Rg)  ============")
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
