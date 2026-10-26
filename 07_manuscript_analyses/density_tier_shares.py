# -*- coding: utf-8 -*-
"""Share of pockets in the very-low and low density tiers, per clustering level.

Backs the Results sentence of "Can one structure rule them all?":
"low-density pockets, about one in ten, sit outside the experimental distribution".
Tiers as in the Methods and SI defined terms: very low <= 0.20, low <= 0.30 (pocket_density_pair).

Usage:  python density_tier_shares.py > density_tier_shares.out.txt
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import pandas as pd

UNI = str(P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV)


def main():
    d = pd.read_csv(UNI, usecols=["threshold", "pocket_density_pair"], low_memory=False)
    print(f"source: {UNI}")
    print(f"{'level':12} {'pockets':>8} {'very low %':>11} {'very low + low %':>17}")
    for level in ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030", "ST_LJ"]:
        x = d.loc[d.threshold == level, "pocket_density_pair"].dropna()
        print(f"{level:12} {len(x):8,d} {100 * (x <= 0.20).mean():11.1f} {100 * (x <= 0.30).mean():17.1f}")


if __name__ == "__main__":
    main()
