# -*- coding: utf-8 -*-
"""Density AND separation vs downstream performance, RAW and SIZE-STANDARDIZED.

density  = pocket_density_pair = (pocket_density_combined + pocket_density_strongest)/2
separation = 1 - pocket_overlap_pair
Both tiered with the database-aligned cuts (support_level: <=.20 very_low, <=.30 low, <=.60 moderate,
<=.85 high, else very_high).

Tasks:  P2Rank / fpocket / IF-SitePred / consensus  (harmonized DCA-TNS, n=86,946)
        docking                                     (mean per-pocket dockable fraction, G1 ok, n=60,689)

Size adjustment: DIRECT standardization to the task's own union_rg decile distribution.

Backs the right-hand table of Supplementary Table S3 (size-standardized gap by separation and density tier,
consensus detection and docking), the per-predictor and raw series quoted under it, and the density-tier
shares (very low 4.0%, low 4.8%, high 43.9%). The manuscript quotes the AFDB v4 set:

    AHOJ_AF_MODEL_SET=afdbv4 python density_separation_gaps.py [density_separation_gaps_afdbv4.csv]

The CSV goes to P.OUT (default name density_separation_gaps{_afdbv4}.csv); density_exclusion_gaps.py reads it.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = P.REPO
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "06_lbs_generality" / "analysis"))
sys.path.insert(0, str(REPO / "04_p2rank" / "analysis"))
from confidence_labeling import support_level
import analyze_fig6a_quality_gates as G
from common import config as cfg

LEVELS = ["very_low", "low", "moderate", "high", "very_high"]
K = 10


def tiered_gaps(df, tier_col, value_col, label):
    """df needs: tier_col, 'cls', value_col (0..1), 'union_rg'."""
    d = df[df["cls"].isin(["apo", "holo"])].copy()
    d["_zb"] = pd.qcut(pd.to_numeric(d["union_rg"], errors="coerce"), K, labels=False, duplicates="drop")
    W = d.loc[d["_zb"].notna(), "_zb"].value_counts(normalize=True).sort_index()  # standard population
    rows = []
    for lv in LEVELS:
        s = d[d[tier_col] == lv]
        rec = {"task": label, "metric": tier_col, "tier": lv, "n": len(s),
               "pct_of_set": 100.0 * len(s) / len(d),
               "n_apo": int((s.cls == "apo").sum()), "n_holo": int((s.cls == "holo").sum()),
               "median_rg": float(pd.to_numeric(s["union_rg"], errors="coerce").median()) if len(s) else np.nan}
        raw, std = {}, {}
        for cl in ("apo", "holo"):
            ss = s[s.cls == cl]
            raw[cl] = 100 * ss[value_col].mean() if len(ss) else np.nan
            t = ss[ss["_zb"].notna()]
            per = t.groupby("_zb")[value_col].mean() * 100.0
            w = W.reindex(per.index)
            std[cl] = float((per * w).sum() / w.sum()) if len(w) and w.sum() > 0 else np.nan
        rec.update({"rate_apo": raw["apo"], "rate_holo": raw["holo"],
                    "gap_raw": raw["holo"] - raw["apo"],
                    "rate_apo_std": std["apo"], "rate_holo_std": std["holo"],
                    "gap_std": std["holo"] - std["apo"]})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---- LBS ------------------------------------------------------------------
base = G.load_base()
base["consensus"] = base[["P2Rank", "fpocket", "IF-SitePred"]].mean(axis=1)
out = []
for m in ["P2Rank", "fpocket", "IF-SitePred", "consensus"]:
    for tier_col in ["density_level_db", "separation_level_db"]:
        out.append(tiered_gaps(base, tier_col, m, m))

# ---- docking ---------------------------------------------------------------
# cfg.DOCKING_RUN3_DIR, not DOCKING_DIR/"run3": it follows AHOJ_AF_MODEL_SET, so this script
# reads the v4 re-run with the switch on and July's tree with it unset (same default path).
dk = pd.read_csv(cfg.DOCKING_RUN3_DIR / "analysis" / "docking_run3_merged.csv", low_memory=False)
dk = dk[(dk.ligand_group == 1) & (dk.status.astype(str).str.lower() == "ok")].copy()
dk["aff"] = pd.to_numeric(dk["affinity_top"], errors="coerce")
dk["inpk"] = dk["pose_in_pocket"].astype(str).str.lower().isin(["true", "1", "yes"])
dk["_ok"] = (dk["inpk"] & (dk["aff"] <= -6)).astype(float)
dk["cls"] = dk["pocket_class"].astype(str).str.lower()
dk["density_level_db"] = pd.to_numeric(dk["pocket_density_pair"], errors="coerce").map(lambda v: support_level(v))
dk["separation_level_db"] = (1.0 - pd.to_numeric(dk["pocket_overlap_pair"], errors="coerce")).map(lambda v: support_level(v))
pk = dk.groupby(["site_uid", "cls", "density_level_db", "separation_level_db"], as_index=False).agg(
    dockable=("_ok", "mean"), union_rg=("union_rg", "first"))
print("docking pockets:", len(pk))
for tier_col in ["density_level_db", "separation_level_db"]:
    out.append(tiered_gaps(pk, tier_col, "dockable", "docking"))

res = pd.concat(out, ignore_index=True)
res["metric"] = res["metric"].str.replace("_level_db", "", regex=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
for met in ["density", "separation"]:
    print(f"\n{'='*88}\n=== {met.upper()} : holo-apo gap (pp), raw vs size-standardized ===\n{'='*88}")
    p = res[res.metric == met].pivot_table(index="tier", columns="task",
                                           values=["gap_raw", "gap_std"]).reindex(LEVELS)
    print(p.round(2).to_string())
    cen = res[(res.metric == met) & (res.task == "consensus")][["tier", "n", "pct_of_set", "median_rg",
                                                               "rate_apo", "rate_holo", "gap_raw", "gap_std"]]
    print("\n  consensus detail:"); print(cen.round(2).to_string(index=False))

out_csv = P.out(sys.argv[1] if len(sys.argv) > 1 else f"density_separation_gaps{P.MODEL_SFX}.csv")
res.to_csv(out_csv, index=False); print("\n[OK] ->", out_csv)
