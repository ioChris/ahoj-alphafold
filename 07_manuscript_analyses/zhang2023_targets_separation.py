# -*- coding: utf-8 -*-
"""Apo-holo separation of the ST 100% sites of the four targets Zhang et al. (2023, J Chem Inf Model) found
clearly apo-like in AlphaFold DB models: ESR1, MET, EGFR and BRAF. Prints the numbers quoted in the Discussion
paragraph on the two published results that seem to run against holo-like models (the 1,012 ST 100% sites of the
four targets, median separation 0.36 against 0.44 across the detection set).

    python zhang2023_targets_separation.py [path/to/afdb_st_100_dataset_with_geometry.csv]

Reads <work_dir>/afdb_dataset/st100/afdb_st_100_dataset_with_geometry.csv (the ST 100% site table with the
pair separation, 1 - mean(overall overlap, mode-weighted overlap); 04_p2rank/dataset/build_st100_afdb_dataset.py
in this repository). Tiers are the database's: very low < 0.20, low 0.20-0.30, moderate 0.30-0.60,
high 0.60-0.85, very high > 0.85.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import os
import sys

import pandas as pd

WD = str(P.WD)

TARGETS = {"P03372": "ESR1", "P08581": "MET", "P00533": "EGFR", "P15056": "BRAF"}
CUTS = [-1, 0.20, 0.30, 0.60, 0.85, 2]
TIERS = ["very low", "low", "moderate", "high", "very high"]


def main(path):
    df = pd.read_csv(path, low_memory=False)
    df["tier"] = pd.cut(df["pocket_separation_pair"], CUTS, labels=TIERS)
    print(f"ST 100% sites: {len(df):,}; median separation {df['pocket_separation_pair'].median():.2f}")
    print("tier shares, all sites:", df["tier"].value_counts(normalize=True).round(3).to_dict())
    sub = df[df["site_uniprot"].isin(TARGETS)].copy()
    sub["gene"] = sub["site_uniprot"].map(TARGETS)
    for gene, d in sub.groupby("gene"):
        cls = d["pocket_class"].value_counts().to_dict()
        hi = int(d["tier"].isin(["high", "very high"]).sum())
        print(f"{gene:<5} {d['site_uniprot'].iloc[0]}  sites {len(d):>4}  median separation "
              f"{d['pocket_separation_pair'].median():.2f}  high or very high {hi:>2}  "
              f"AF-Apo {cls.get('apo', 0)} / AF-Holo {cls.get('holo', 0)} / tie {cls.get('tie', 0)}")
    hi_all = int(sub["tier"].isin(["high", "very high"]).sum())
    print(f"four targets together: {len(sub):,} sites, {hi_all} in the high or very high tier "
          f"({100 * hi_all / len(sub):.1f}%), median separation {sub['pocket_separation_pair'].median():.2f}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        WD, "afdb_dataset", "st100", "afdb_st_100_dataset_with_geometry.csv")
    main(src)
