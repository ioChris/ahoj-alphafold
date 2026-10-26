# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:26:56 2025

@author: Christos P. Feidakis
"""

import os
import ast
import pandas as pd
from typing import List, Optional




def aggregate_ubs_cluster(row, df_summary, uniprot, threshold):
    """Aggregate summary for a single UBS cluster row."""
    entry_ids = ast.literal_eval(row['entry_ids'])
    cluster_jobs = df_summary[df_summary['job_name'].isin(entry_ids)]
    missing = set(entry_ids) - set(cluster_jobs['job_name'])
    return {
        'uniprot': uniprot,
        'threshold': threshold,
        'cluster_id': row['cluster_id'],
        'num_entries': row['num_entries'],
        'n_found': len(cluster_jobs),
        'sum_n_apo': cluster_jobs['n_apo'].sum(),
        'sum_n_holo': cluster_jobs['n_holo'].sum(),
        'num_shared_residues': row['num_shared_residues'],
        'num_total_residues': row['num_total_residues'],
        'missing_jobs': ";".join(missing),
        'pocket_avg_P(apo)_mean': cluster_jobs['pocket_avg_P(apo)'].mean(),
        'pocket_avg_P(holo)_mean': cluster_jobs['pocket_avg_P(holo)'].mean(),
        'pocket_avg_normalized_density_combined_mean': cluster_jobs['pocket_avg_normalized_density_combined'].mean(),
        'chain_avg_P(apo)_mean': cluster_jobs['chain_avg_P(apo)'].mean(),
        'chain_avg_P(holo)_mean': cluster_jobs['chain_avg_P(holo)'].mean(),
        'chain_avg_normalized_density_combined_mean': cluster_jobs['chain_avg_normalized_density_combined'].mean(),
        'job_names': ";".join(entry_ids)
    }

def aggregate_uniprot_ubs(
    uniprot, UBS_root, df_summary, 
    thresholds: Optional[List[str]] = None
):
    """Aggregate all clusters for a single UniProt and given thresholds."""
    uniprot_path = os.path.join(UBS_root, uniprot)
    results = []
    # Determine which threshold files to use
    if thresholds is None:
        thresholds = [
            f for f in os.listdir(uniprot_path)
            if f.startswith("UBS_") and f.endswith(".tsv")
        ]
    for fname in thresholds:
        threshold = fname.replace('.tsv', '')
        ubs_file = os.path.join(uniprot_path, fname)
        df_ubs = pd.read_csv(ubs_file, sep='\t')
        for _, row in df_ubs.iterrows():
            results.append(aggregate_ubs_cluster(row, df_summary, uniprot, threshold))
    return results

def aggregate_all_uniprots(
    UBS_root, summary_csv_path, 
    uniprot_include: Optional[List[str]] = None, 
    limit_n_uniprots: Optional[int] = None,
    thresholds: Optional[List[str]] = None
):
    """Aggregate for all (or selected/first N) UniProts and thresholds."""
    df_summary = pd.read_csv(summary_csv_path)
    all_uniprots = [
        d for d in os.listdir(UBS_root) 
        if os.path.isdir(os.path.join(UBS_root, d))
    ]
    # Apply user limit/selection
    if uniprot_include is not None:
        uniprots = [u for u in all_uniprots if u in uniprot_include]
    elif limit_n_uniprots is not None:
        uniprots = all_uniprots[:limit_n_uniprots]
    else:
        uniprots = all_uniprots
    
    all_results = []
    for uniprot in uniprots:
        print(f"Processing UniProt: {uniprot}")
        try:
            results = aggregate_uniprot_ubs(uniprot, UBS_root, df_summary, thresholds)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {uniprot}: {e}")
    return pd.DataFrame(all_results)

# --------- USAGE EXAMPLE ---------



# ---- USER INPUTS ----
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# NOTE: legacy demo/usage example — adjust to your current inputs as needed.
summary_csv_path = str(cfg.QUERY_SUMMARIES_CLASS_CSV)
UBS_root = str(cfg.UBS_ROOT_DIR)

# ---- OUTPUT ----
output_dir = str(cfg.UBS_AGG_DIR)

#uniprot_id = "B3EWQ9"                                  # <--- Set for demo/test
#threshold = "UBS_05.tsv"                               # <--- Can iterate later
# ---------------------


# Setup (edit these):
#summary_csv_path = "PATH/TO/query_summaries_class.csv"
#UBS_root = "PATH/TO/UBS_root"

# Example usage (first 5 UniProts only):
df_agg = aggregate_all_uniprots(
    UBS_root, summary_csv_path, 
    limit_n_uniprots=99
)

# ROUNDING
cols_to_round = [
    'pocket_avg_P(apo)_mean', 'pocket_avg_P(holo)_mean', 
    'pocket_avg_normalized_density_combined_mean',
    'chain_avg_P(apo)_mean', 'chain_avg_P(holo)_mean',
    'chain_avg_normalized_density_combined_mean'
]
for col in cols_to_round:
    if col in df_agg.columns:
        df_agg[col] = df_agg[col].round(3)


df_agg.to_csv(os.path.join(output_dir, "UBS_cluster_aggregations_subset.csv"), index=False)

# Example usage (specific list of UniProts):
# df_agg = aggregate_all_uniprots(
#     UBS_root, summary_csv_path,
#     uniprot_include=['B3EWQ9', 'P12345']
# )

