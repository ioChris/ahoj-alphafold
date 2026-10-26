# -*- coding: utf-8 -*-
"""What share of pockets sit at LOW SEPARATION (apo/holo states barely distinguishable)?

separation = 1 - pocket_overlap_pair ; tiers via confidence_labeling.support_level
(<=0.20 very_low, <=0.30 low, <=0.60 moderate, <=0.85 high, else very_high).

Backs the left-hand table of Supplementary Table S3 (share of pockets in the very-low and low separation
tiers at each clustering level) and the main-text "12.0% of ST 100% pockets at very low separation (25.9%
including low)". Reads cfg.ANALYSIS_UNIFIED_LEVELS_CSV; does not depend on the model set.

    python c30_separation_share.py [c30_separation_share.csv]      (written to P.OUT)
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
from pathlib import Path
import pandas as pd, numpy as np

REPO = P.REPO
sys.path.insert(0, str(REPO))
from common import config as cfg
sys.path.insert(0, str(REPO / "04_p2rank" / "analysis"))
from confidence_labeling import support_level

COLS = ["threshold", "level", "pocket_class", "pocket_overlap_pair", "unit_uid"]
df = pd.read_csv(cfg.ANALYSIS_UNIFIED_LEVELS_CSV, usecols=COLS, low_memory=False)
print("rows total:", len(df))
print("levels:", df["level"].value_counts().to_dict())
print("thresholds:", df["threshold"].value_counts().to_dict())

df["separation"] = 1.0 - pd.to_numeric(df["pocket_overlap_pair"], errors="coerce")
df["sep_tier"] = df["separation"].map(lambda v: support_level(v, merge=False))

ORDER = ["very_low", "low", "moderate", "high", "very_high", "unknown"]
out = []
for thr, sub in df.groupby("threshold"):
    n = len(sub)
    vc = sub["sep_tier"].value_counts()
    row = {"threshold": thr, "n_pockets": n}
    for t in ORDER:
        row[t] = int(vc.get(t, 0))
    defined = n - row["unknown"]
    row["n_defined"] = defined
    # share of ALL pockets and of pockets where separation is defined
    row["very_low_pct_all"]      = 100.0 * row["very_low"] / n if n else np.nan
    row["vlow_plus_low_pct_all"] = 100.0 * (row["very_low"] + row["low"]) / n if n else np.nan
    row["very_low_pct_def"]      = 100.0 * row["very_low"] / defined if defined else np.nan
    row["vlow_plus_low_pct_def"] = 100.0 * (row["very_low"] + row["low"]) / defined if defined else np.nan
    out.append(row)

res = pd.DataFrame(out)
order = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]
res["__o"] = res["threshold"].map(lambda t: order.index(t) if t in order else 99)
res = res.sort_values("__o").drop(columns="__o")
pd.set_option("display.width", 200, "display.max_columns", 50)
print("\n=== separation tier distribution per threshold ===")
print(res.to_string(index=False))

o = Path(P.out(sys.argv[1] if len(sys.argv) > 1 else "c30_separation_share.csv"))
res.to_csv(o, index=False)
print("\n[OK] ->", o)
