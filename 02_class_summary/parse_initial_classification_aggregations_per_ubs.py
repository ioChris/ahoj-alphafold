# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:48:33 2025

@author: Christos P. Feidakis
"""

"""
Parse the initial classification aggregations (CSV file) and perform a preliminary analysis
"""

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

csv_path = str(cfg.UBS_AGG_CLASSIFICATIONS_CSV)



# SETTINGS
STANDARD_THRESHOLDS = ['ST_030', 'ST_050', 'ST_070', 'ST_100']
#STANDARD_THRESHOLDS = ['UBS_05', 'UBS_06', 'UBS_07', 'UBS_10']


LARGE_JUMP_CATEGORY = "ST_LJ"
#LARGE_JUMP_CATEGORY = "UBS_LJ"


df = pd.read_csv(csv_path)



def classification_counts(df, col_apo, col_holo):
    n_apo = n_holo = n_tie = n_no_data = n_other = 0
    for _, row in df.iterrows():
        apo_prob = row.get(col_apo)
        holo_prob = row.get(col_holo)
        if pd.isna(apo_prob) and pd.isna(holo_prob):
            n_no_data += 1
        elif pd.isna(apo_prob) or (holo_prob > apo_prob):
            n_holo += 1
        elif pd.isna(holo_prob) or (apo_prob > holo_prob):
            n_apo += 1
        elif apo_prob == holo_prob:
            n_tie += 1
        else:
            n_other += 1
    return n_apo, n_holo, n_tie, n_no_data, n_other

def vote_counts(df, vote_col):
    vote_counts = df[vote_col].value_counts(dropna=False)
    n_apo = int(vote_counts.get("apo", 0))
    n_holo = int(vote_counts.get("holo", 0))
    n_tie = int(vote_counts.get("tie", 0))
    n_unknown = int(vote_counts.get(None, 0) + vote_counts.get(float('nan'), 0))
    n_other = int(sum([count for v, count in vote_counts.items() if v not in {"apo", "holo", "tie", None} and not pd.isna(v)]))
    return n_apo, n_holo, n_tie, n_unknown, n_other

def cutoff_classification_counts_sum1(
    df, col_apo, col_holo, cutoff=0.6,
    density_col=None, density_cutoff=None
):
    """
    Counts UBS classified as apo/holo/ambiguous/no data using a cutoff.
    Optionally, if density_col and density_cutoff are set, counts 'low-density' UBS.
    Returns: n_apo, n_holo, n_ambiguous, n_no_data, n_low_density, n_other
    """
    n_apo = n_holo = n_ambiguous = n_no_data = n_low_density = n_other = 0
    for _, row in df.iterrows():
        # Low-density check (if set)
        if density_col and density_cutoff is not None:
            density = row.get(density_col)
            if pd.notna(density) and density < density_cutoff:
                n_low_density += 1
                continue
        apo_prob = row.get(col_apo)
        holo_prob = row.get(col_holo)
        # Main cutoff logic
        if pd.isna(apo_prob) and pd.isna(holo_prob):
            n_no_data += 1
        elif pd.isna(apo_prob):
            if holo_prob > cutoff:
                n_holo += 1
            else:
                n_ambiguous += 1
        elif pd.isna(holo_prob):
            if apo_prob > cutoff:
                n_apo += 1
            else:
                n_ambiguous += 1
        else:
            max_prob = max(apo_prob, holo_prob)
            if max_prob <= cutoff:
                n_ambiguous += 1
            elif apo_prob > holo_prob:
                n_apo += 1
            elif holo_prob > apo_prob:
                n_holo += 1
            else:
                n_ambiguous += 1
    return n_apo, n_holo, n_ambiguous, n_no_data, n_low_density, n_other



print("\n========== Detailed UBS Classification Summary ==========")
for threshold in STANDARD_THRESHOLDS + [LARGE_JUMP_CATEGORY]:
    if threshold == LARGE_JUMP_CATEGORY:
        df_th = df[df['threshold'].str.startswith(LARGE_JUMP_CATEGORY, na=False)]
    else:
        df_th = df[df['threshold'] == threshold]
    n_total = len(df_th)

    # Pocket probability-based
    n_apo_p, n_holo_p, n_tie_p, n_no_data_p, n_other_p = classification_counts(
        df_th, 'avg_P(apo)_mean_pocket', 'avg_P(holo)_mean_pocket')
    n_with_data_p = n_total - n_no_data_p

    # Pocket probability with cutoff
    #n_apo_cut, n_holo_cut, n_ambiguous_cut, n_no_data_cut, n_other_cut = cutoff_classification_counts_sum1(
    #df_th, 'avg_P(apo)_mean_pocket', 'avg_P(holo)_mean_pocket', cutoff=0.6)
    n_apo_cut, n_holo_cut, n_ambiguous_cut, n_no_data_cut, n_low_density, n_other_cut = cutoff_classification_counts_sum1(
    df_th,
    'avg_P(apo)_mean_pocket', 'avg_P(holo)_mean_pocket',
    cutoff=0.6,
    density_col='avg_normalized_density_combined_mean_pocket',
    density_cutoff=0.10)

    n_with_data_cut = n_total - (n_no_data_cut + n_low_density)


    # Chain probability-based
    n_apo_c, n_holo_c, n_tie_c, n_no_data_c, n_other_c = classification_counts(
        df_th, 'avg_P(apo)_mean_chain', 'avg_P(holo)_mean_chain')
    n_with_data_c = n_total - n_no_data_c

    # Pocket vote-based
    n_apo_vp, n_holo_vp, n_tie_vp, n_unknown_vp, n_other_vp = vote_counts(df_th, 'majority_vote_pocket')
    n_with_vote_p = n_total - n_unknown_vp

    # Chain vote-based
    n_apo_vc, n_holo_vc, n_tie_vc, n_unknown_vc, n_other_vc = vote_counts(df_th, 'majority_vote_chain')
    n_with_vote_c = n_total - n_unknown_vc

    pct = lambda x: 100 * x / n_total if n_total else 0
    pct_data_p = lambda x: 100 * x / n_with_data_p if n_with_data_p else 0
    pct_data_c = lambda x: 100 * x / n_with_data_c if n_with_data_c else 0
    pct_vote_p = lambda x: 100 * x / n_with_vote_p if n_with_vote_p else 0
    pct_vote_c = lambda x: 100 * x / n_with_vote_c if n_with_vote_c else 0

    pct_cut = lambda x: 100 * x / n_total if n_total else 0
    pct_data_cut = lambda x: 100 * x / n_with_data_cut if n_with_data_cut else 0
    #pct_data_cut = lambda x: 100 * x / n_with_data_cut if n_with_data_cut else 0

    print(f"\n--- Threshold: {threshold} ---")
    print(f"  Total UBS:                   {n_total}")
    print("  [Pocket PROB classification]")
    print(f"    With data:           {n_with_data_p} ({pct(n_with_data_p):.1f}%)")
    print(f"    Holo classified:     {n_holo_p} ({pct(n_holo_p):.1f}% of total, {pct_data_p(n_holo_p):.1f}% of with-data)")
    print(f"    Apo classified:      {n_apo_p} ({pct(n_apo_p):.1f}% of total, {pct_data_p(n_apo_p):.1f}% of with-data)")
    print(f"    Tie:                 {n_tie_p} ({pct(n_tie_p):.1f}% of total, {pct_data_p(n_tie_p):.1f}% of with-data)")
    print(f"    No data:             {n_no_data_p} ({pct(n_no_data_p):.1f}%)")
    if n_other_p:
        print(f"    Other/Unexpected:    {n_other_p}")

    print("  [Pocket PROB classification with cutoff 0.6]")
    print(f"    With data:           {n_with_data_cut} ({pct_cut(n_with_data_cut):.1f}%)")
    print(f"    Holo classified:     {n_holo_cut} ({pct_cut(n_holo_cut):.1f}% of total, {pct_data_cut(n_holo_cut):.1f}% of with-data)")
    print(f"    Apo classified:      {n_apo_cut} ({pct_cut(n_apo_cut):.1f}% of total, {pct_data_cut(n_apo_cut):.1f}% of with-data)")
    print(f"    Low-density (<0.10): {n_low_density} ({pct_cut(n_low_density):.1f}% of total)")
    print(f"    Ambiguous:           {n_ambiguous_cut} ({pct_cut(n_ambiguous_cut):.1f}% of total, {pct_data_cut(n_ambiguous_cut):.1f}% of with-data)")
    print(f"    No data:             {n_no_data_cut} ({pct_cut(n_no_data_cut):.1f}%)")
    if n_other_cut:
        print(f"    Other/Unexpected:    {n_other_cut}")

    print("  [Chain PROB classification]")
    print(f"    With data:           {n_with_data_c} ({pct(n_with_data_c):.1f}%)")
    print(f"    Holo classified:     {n_holo_c} ({pct(n_holo_c):.1f}% of total, {pct_data_c(n_holo_c):.1f}% of with-data)")
    print(f"    Apo classified:      {n_apo_c} ({pct(n_apo_c):.1f}% of total, {pct_data_c(n_apo_c):.1f}% of with-data)")
    print(f"    Tie:                 {n_tie_c} ({pct(n_tie_c):.1f}% of total, {pct_data_c(n_tie_c):.1f}% of with-data)")
    print(f"    No data:             {n_no_data_c} ({pct(n_no_data_c):.1f}%)")
    if n_other_c:
        print(f"    Other/Unexpected:    {n_other_c}")

    print("  [Pocket MAJORITY VOTE classification]")
    print(f"    With vote:           {n_with_vote_p} ({pct(n_with_vote_p):.1f}%)")
    print(f"    Holo classified:     {n_holo_vp} ({pct(n_holo_vp):.1f}% of total, {pct_vote_p(n_holo_vp):.1f}% of with-vote)")
    print(f"    Apo classified:      {n_apo_vp} ({pct(n_apo_vp):.1f}% of total, {pct_vote_p(n_apo_vp):.1f}% of with-vote)")
    print(f"    Tie:                 {n_tie_vp} ({pct(n_tie_vp):.1f}% of total, {pct_vote_p(n_tie_vp):.1f}% of with-vote)")
    print(f"    No vote:             {n_unknown_vp} ({pct(n_unknown_vp):.1f}%)")
    if n_other_vp:
        print(f"    Other/Unexpected:    {n_other_vp}")

    print("  [Chain MAJORITY VOTE classification]")
    print(f"    With vote:           {n_with_vote_c} ({pct(n_with_vote_c):.1f}%)")
    print(f"    Holo classified:     {n_holo_vc} ({pct(n_holo_vc):.1f}% of total, {pct_vote_c(n_holo_vc):.1f}% of with-vote)")
    print(f"    Apo classified:      {n_apo_vc} ({pct(n_apo_vc):.1f}% of total, {pct_vote_c(n_apo_vc):.1f}% of with-vote)")
    print(f"    Tie:                 {n_tie_vc} ({pct(n_tie_vc):.1f}% of total, {pct_vote_c(n_tie_vc):.1f}% of with-vote)")
    print(f"    No vote:             {n_unknown_vc} ({pct(n_unknown_vc):.1f}%)")
    if n_other_vc:
        print(f"    Other/Unexpected:    {n_other_vc}")

print(f"\nNote: 'No data' (probabilities missing) and 'No vote' (vote missing) are reported separately. All large jump thresholds ({LARGE_JUMP_CATEGORY}*) are combined into a single '{LARGE_JUMP_CATEGORY}' category.")


