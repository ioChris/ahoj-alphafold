# -*- coding: utf-8 -*-
"""Size-controlled partial correlations of pocket pLDDT quoted in the Supplementary Information (pLDDT
section, Table S6 text): pLDDT against conformational diversity (mean pocket RMSD), CV of the radius of
gyration and |evidence ratio|, each controlling for pocket size.

    python frontier_at_scale.py          # writes frontier_per_pocket.csv to P.OUT (once)
    python plddt_partial_size.py

Size is the mean experimental pocket volume (``volume_mean``). The partial is the first-order partial of
Pearson correlations on ranks (as in plddt_headtohead.py), over the complete cases of all five columns
(236,087 pockets: the pockets with a defined spread of the radius of gyration). This gives the SI's
-0.478 / -0.386 / +0.082.

Figure 5's caption number is a different estimate of the same relation: plotting/figure5_plddt_reliability.py
controls for the AlphaFold model's pocket residue count, over all 237,922 pockets, with a Spearman
correlation of rank residuals (-0.4786). Both round to the main text's -0.48.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ma_paths as P

d = pd.read_csv(P.out(sys.argv[1] if len(sys.argv) > 1 else "frontier_per_pocket.csv"), low_memory=False)
y = "model_pocket_pLDDT_avg"
d["cv_RoG"] = d.RoG_std_dev / d.RoG_mean.replace(0, np.nan)
d["abs_ratio"] = np.log2((d.n_holo + 1) / d.n_apo.where(d.n_apo > 0)).abs()
sub = d[[y, "pocket_rms_mean", "cv_RoG", "abs_ratio", "volume_mean"]].replace([np.inf, -np.inf], np.nan).dropna()
print("complete cases (pLDDT, mean pocket RMSD, CV RoG, |ratio|, volume_mean):", len(sub))
r = sub.rank()


def partial(a, b, c):
    rab = np.corrcoef(r[a], r[b])[0, 1]
    rac = np.corrcoef(r[a], r[c])[0, 1]
    rbc = np.corrcoef(r[b], r[c])[0, 1]
    return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))


for x in ("pocket_rms_mean", "cv_RoG", "abs_ratio"):
    print(f"  {x:16s} zero-order {spearmanr(sub[x], sub[y])[0]:+.6f}   partial | volume_mean {partial(y, x, 'volume_mean'):+.6f}")
