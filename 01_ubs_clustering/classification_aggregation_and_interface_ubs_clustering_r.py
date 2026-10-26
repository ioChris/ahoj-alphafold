# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:02:32 2025

@author: Christos P. Feidakis
"""

"""
Parse one main file with the classification of each job (both pocket and chain)
1. the per-query classification summary (query_summaries_class.csv; configured as
   QUERY_SUMMARIES_CLASS_CSV in common/config.py)
and the file(s) with the UBSs per threshold (1 file per uniprot per Jaccard threshold)
2. 
3. Add site_uid column with a single unique and explicit string for each single or unified cluster [uniprot_id:threshold:ubs_id]

# v1.4 change to new file names (UBS_05 -> ST_050), columns (cluster_id -> ubs_id) and similarity thresholds used (+030, -080)
# v1.5 Added logging, manifest, avg_n_apo, avg_n_holo, ratio_avg_n_holo_to_apo, add multimeric status to final table
# v1.6 added more density information (+pocket_avg_normalized_density_strongest) and the mean of both densities.
# v1.7 add the UBS-level mean pLDDT (per atom avg) column of the entries (if it exists) as the average over merged entries/pockets
       record UBS groups processed per threshold in run.log and run_manifest.json

# v1.8 Clean ST_LJ thresholds:
#      - output `threshold` as "ST_LJ" for all large-jump unified groups
#      - store the original LJ thresholds involved in a group in `threshold_detail`
#      This prevents category explosion in plots and keeps provenance for debugging.

# v1.9 Standardized UniProt delimiter in site_uniprots to be space-separated (matching upstream tables) 
#     and added a filesystem-safe site_uniprots_key (underscore-joined) for stable IDs/filenames.

# v1.10 Added dual-output mode: keep the existing “reference” aggregated CSV (may include empty/filtered-out member groups), 
#      and also write a second “filtered” aggregated CSV that only reflects members present in the provided filtered_survivors summary input (i.e., groups with ≥1 surviving member).

# v1.11 Updated manifest to report per-threshold row counts for both reference and filtered outputs (rows_by_threshold, rows_filtered_by_threshold plus UniProt-group breakdowns). 
#      Removed unique_threshold_detail_n and unique_threshold_detail_sample from the manifest to keep it uncluttered.

"""


import os
import ast
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import time
from tqdm import tqdm

import sys
import json
import socket
import platform
import logging
from datetime import datetime

STANDARD_THRESHOLDS = ['ST_030', 'ST_050', 'ST_070', 'ST_100']

# --------------------------------- FUNCTIONS ---------------------------------

def get_2level_cwd():
    npath = os.path.normpath(os.getcwd())   # Normalize the path string for the OS
    path0 = os.path.join(npath.split(os.sep)[0], os.sep, npath.split(os.sep)[1], npath.split(os.sep)[2])
    if os.path.exists(path0):
        memo = "Root path found >> " + path0
    else:
        memo = 'Error finding root path in working dir:' + npath
    print(memo)
    return path0


def parse_threshold_from_fname(fname):
    return fname.replace('.tsv', '')


def load_ubs_clusters(UBS_root, threshold=None, lj=False, uniprots=None, limit_n_uniprots=None):
    """
    Load UBS clusters for a specific standard threshold, or all LJ thresholds if lj=True.
    Returns a list of clusters, each as a dict with uniprot, ubs_id, threshold, etc.
    """
    clusters = []
    all_uniprots = sorted([
        d for d in os.listdir(UBS_root)
        if os.path.isdir(os.path.join(UBS_root, d))
    ])
    if uniprots is not None:
        all_uniprots = [u for u in all_uniprots if u in uniprots]
    if limit_n_uniprots is not None:
        all_uniprots = all_uniprots[:limit_n_uniprots]

    for uniprot in all_uniprots:
        ubs_path = os.path.join(UBS_root, uniprot)
        for fname in os.listdir(ubs_path):
            if not fname.startswith('ST_') or not fname.endswith('.tsv'):
                continue
            th = parse_threshold_from_fname(fname)
            is_lj = th.startswith('ST_LJ')
            if lj and not is_lj:
                continue
            if not lj and threshold is not None and th != threshold:
                continue

            df_ubs = pd.read_csv(os.path.join(ubs_path, fname), sep='\t', dtype={'multimeric_status_all': 'object'})
            if 'multimeric_status_all' not in df_ubs.columns:
                df_ubs['multimeric_status_all'] = np.nan
            for _, row in df_ubs.iterrows():
                clusters.append({
                    "uniprot": uniprot,
                    "ubs_id": str(row['ubs_id']),
                    "threshold": th,
                    "entry_ids": set(ast.literal_eval(row['entry_ids'])),
                    "shared_residues": set(ast.literal_eval(row['shared_residues'])),
                    "total_residues": set(ast.literal_eval(row['total_residues'])),
                    # NEW: carry multimeric status for this cluster row
                    "multimeric_status_all": row.get('multimeric_status_all', None)
                })
    return clusters


def build_connected_components(clusters, allow_cross_thresholds=False):
    """
    Merge clusters into connected components via entry_id overlap.
    For standard thresholds, no cross-threshold merging.
    For LJ, allow merging across LJ thresholds.
    """
    entry_to_ubs_idxs = defaultdict(set)
    for idx, cluster in enumerate(clusters):
        for eid in cluster["entry_ids"]:
            entry_to_ubs_idxs[eid].add(idx)

    n = len(clusters)
    seen = set()
    components = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        component = set()
        # Only allow merging across thresholds for LJ, not for standard
        allowed_thresholds = {clusters[i]['threshold']} if not allow_cross_thresholds else None
        while stack:
            j = stack.pop()
            if j in component:
                continue
            component.add(j)
            for eid in clusters[j]["entry_ids"]:
                for neighbor in entry_to_ubs_idxs[eid]:
                    if neighbor in component:
                        continue
                    if not allow_cross_thresholds:
                        # Only merge clusters with the same threshold
                        if clusters[neighbor]['threshold'] not in allowed_thresholds:
                            continue
                    stack.append(neighbor)
        seen |= component
        components.append(component)
    return components


def collect_unique_ints(series, delimiter=";"):
    """
    Collect unique integer values from a Series that may contain:
      - ints / floats (1, 1.0)
      - NaN / None
      - stringified ints ("1")
      - delimited strings ("1;2", "1, 2", "1:2", "1 2")
      - stringified containers ("[1, 2]", "{1,2}")
    Returns (sorted_list_of_ints, joined_string) where joined_string uses `delimiter`.
    """
    vals = set()

    for v in series:
        # skip missing
        if v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NA):
            continue

        # container direct
        if isinstance(v, (list, tuple, set, np.ndarray)):
            items = v
        else:
            s = str(v).strip()
            if not s or s.lower() == "nan":
                continue
            # try to parse literal (handles 1, 1.0, [1,2], {1,2})
            parsed = None
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None

            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                items = parsed
            elif parsed is not None and not isinstance(parsed, dict):
                items = [parsed]
            else:
                # fallback: split by common delimiters ; , : and whitespace
                s2 = s.replace(';', ' ').replace(',', ' ').replace(':', ' ')
                items = s2.split()

        for item in items:
            try:
                vals.add(int(item))
            except Exception:
                # tolerate floats that are integer-like (e.g., "1.0")
                try:
                    vals.add(int(float(item)))
                except Exception:
                    pass

    uniq = sorted(vals)
    joined = delimiter.join(str(x) for x in uniq) if uniq else None
    return uniq, joined


def aggregate_ubs_groups(
    group_idxs, clusters, df_summary, allow_cross_thresholds=False
):
    """
    Aggregates unified sites from cluster groups, using the summary DataFrame.
    """
    results = []
    results_filtered = []
    # Index summary df for faster access in-loop
    df_summary_indexed = df_summary.set_index('job_name')
    summary_columns = df_summary.columns

    for group in tqdm(group_idxs, desc="Aggregating UBS groups"):
    #for group in group_idxs:
        group_clusters = [clusters[idx] for idx in group]
        all_entry_ids = set()
        all_uniprots = set()
        cluster_recipes = []
        cluster_uid_parts = []  # NEW
        thresholds_in_group = set()
        all_shared_residues = []
        all_total_residues = []
        for cl in group_clusters:
            all_entry_ids.update(cl["entry_ids"])
            all_uniprots.add(cl["uniprot"])
            cluster_recipes.append(f"{cl['uniprot']}:{cl['ubs_id']}")
            cluster_uid_parts.append(f"{cl['uniprot']}:{cl['threshold']}:{cl['ubs_id']}")  # NEW
            thresholds_in_group.add(cl['threshold'])
            all_shared_residues.append(cl["shared_residues"])
            all_total_residues.append(cl["total_residues"])

        union_residues = set().union(*all_total_residues)
        intersection_residues = set.intersection(*all_shared_residues) if all_shared_residues else set()

        # Get sub df
        existing_entry_ids = [eid for eid in all_entry_ids if eid in df_summary_indexed.index]
        if existing_entry_ids:
            sub_df = df_summary_indexed.loc[existing_entry_ids]
        else:
            sub_df = pd.DataFrame(columns=summary_columns)

        ## --- multimeric status aggregation (unique values across merged rows) ---
        # NEW: collect from the clusters in this group (ST files source of truth)
        status_series = pd.Series(
            [cl.get('multimeric_status_all', np.nan) for cl in group_clusters],
            dtype='object'
        )
        _, multimeric_joined = collect_unique_ints(status_series)

        # Apo and holo sum
        sum_n_apo = int(pd.to_numeric(sub_df['n_apo'], errors='coerce').sum())
        sum_n_holo = int(pd.to_numeric(sub_df['n_holo'], errors='coerce').sum())
        #sum_n_apo = int(sub_df['n_apo'].sum())
        #sum_n_holo = int(sub_df['n_holo'].sum())

        if sum_n_apo == 0 and sum_n_holo == 0:
            ratio_n_holo_to_apo = None
        elif sum_n_apo == 0:
            ratio_n_holo_to_apo = float('inf')
        elif sum_n_holo == 0:
            ratio_n_holo_to_apo = 0.0
        else:
            ratio_n_holo_to_apo = sum_n_holo / sum_n_apo

        # Round the ratio
        if ratio_n_holo_to_apo is None or (isinstance(ratio_n_holo_to_apo, float) and (math.isnan(ratio_n_holo_to_apo) or math.isinf(ratio_n_holo_to_apo))):
            ratio_n_holo_to_apo_rounded = ratio_n_holo_to_apo
        else:
            ratio_n_holo_to_apo_rounded = round(ratio_n_holo_to_apo, 2)
        
        # --- averages over jobs in this UBS (ignore NaNs safely) ---
        apo_series = pd.to_numeric(sub_df['n_apo'], errors='coerce')
        holo_series = pd.to_numeric(sub_df['n_holo'], errors='coerce')
        
        avg_n_apo_raw = apo_series.mean() if not apo_series.dropna().empty else None
        avg_n_holo_raw = holo_series.mean() if not holo_series.dropna().empty else None
        
        avg_n_apo = round(avg_n_apo_raw, 2) if avg_n_apo_raw is not None and not pd.isna(avg_n_apo_raw) else None
        avg_n_holo = round(avg_n_holo_raw, 2) if avg_n_holo_raw is not None and not pd.isna(avg_n_holo_raw) else None
        
        # --- ratio of averages (holo / apo), robust to zeros/NaNs ---
        if (avg_n_apo is None and avg_n_holo is None):
            ratio_avg_n_holo_to_apo = None
        elif avg_n_apo in (None,) or pd.isna(avg_n_apo):
            # apo avg missing
            ratio_avg_n_holo_to_apo = float('inf') if (avg_n_holo is not None and not pd.isna(avg_n_holo) and avg_n_holo > 0) else None
        elif avg_n_holo in (None,) or pd.isna(avg_n_holo):
            # holo avg missing
            ratio_avg_n_holo_to_apo = 0.0
        else:
            if avg_n_apo == 0:
                ratio_avg_n_holo_to_apo = float('inf') if avg_n_holo > 0 else None
            else:
                ratio_avg_n_holo_to_apo = avg_n_holo / avg_n_apo
        
        # Optional rounding of ratio (preserve None/inf/NaN)
        if isinstance(ratio_avg_n_holo_to_apo, float) and not (math.isinf(ratio_avg_n_holo_to_apo) or math.isnan(ratio_avg_n_holo_to_apo)):
            ratio_avg_n_holo_to_apo = round(ratio_avg_n_holo_to_apo, 2)


        # Pocket-level classification info
        votes_n_apo_pocket = (sub_df['pocket_overall_classification'] == "Apo").sum()
        votes_n_holo_pocket = (sub_df['pocket_overall_classification'] == "Holo").sum()
        votes_n_undefined_pocket = (sub_df['pocket_overall_classification'] == "Undefined").sum()

        # Vote ratio (careful about divide-by-zero)
        if votes_n_apo_pocket == 0 and votes_n_holo_pocket == 0:
            vote_ratio_holo_to_apo_pocket = None
            majority_vote_pocket = None # "unknown"
        elif votes_n_apo_pocket == 0:
            vote_ratio_holo_to_apo_pocket = float('inf')
            majority_vote_pocket = "holo"
        elif votes_n_holo_pocket == 0:
            vote_ratio_holo_to_apo_pocket = 0.0
            majority_vote_pocket = "apo"
        else:
            vote_ratio_holo_to_apo_pocket = votes_n_holo_pocket / votes_n_apo_pocket
            majority_vote_pocket = "holo" if votes_n_holo_pocket > votes_n_apo_pocket else ("apo" if votes_n_apo_pocket > votes_n_holo_pocket else "tie")

        # Round the ratio
        if vote_ratio_holo_to_apo_pocket is None or (isinstance(vote_ratio_holo_to_apo_pocket, float) and (math.isnan(vote_ratio_holo_to_apo_pocket) or math.isinf(vote_ratio_holo_to_apo_pocket))):
            vote_ratio_holo_to_apo_pocket_rounded = vote_ratio_holo_to_apo_pocket  # keep as is if None, NaN, or inf
        else:
            vote_ratio_holo_to_apo_pocket_rounded = round(vote_ratio_holo_to_apo_pocket, 2)

        # Probability means
        pocket_mean_p_apo = sub_df['pocket_avg_P(apo)'].mean()
        pocket_mean_p_holo = sub_df['pocket_avg_P(holo)'].mean()

        if pd.isna(pocket_mean_p_apo) and pd.isna(pocket_mean_p_holo):
            pocket_highest_p_mean = None # "unknown"
        elif pd.isna(pocket_mean_p_apo) or (pocket_mean_p_holo > pocket_mean_p_apo):
            pocket_highest_p_mean = "holo"
        elif pd.isna(pocket_mean_p_holo) or (pocket_mean_p_apo > pocket_mean_p_holo):
            pocket_highest_p_mean = "apo"
        else:
            pocket_highest_p_mean = "tie"

        # Chain-level classification info
        votes_n_apo_chain = (sub_df['chain_overall_classification'] == "Apo").sum()
        votes_n_holo_chain = (sub_df['chain_overall_classification'] == "Holo").sum()
        votes_n_undefined_chain = (sub_df['chain_overall_classification'] == "Undefined").sum()

        # Chain vote ratio and majority
        if votes_n_apo_chain == 0 and votes_n_holo_chain == 0:
            vote_ratio_holo_to_apo_chain = None
            majority_vote_chain = None
        elif votes_n_apo_chain == 0:
            vote_ratio_holo_to_apo_chain = float('inf')
            majority_vote_chain = "holo"
        elif votes_n_holo_chain == 0:
            vote_ratio_holo_to_apo_chain = 0.0
            majority_vote_chain = "apo"
        else:
            vote_ratio_holo_to_apo_chain = votes_n_holo_chain / votes_n_apo_chain
            majority_vote_chain = "holo" if votes_n_holo_chain > votes_n_apo_chain else ("apo" if votes_n_apo_chain > votes_n_holo_chain else "tie")

        # Round the chain ratio
        if vote_ratio_holo_to_apo_chain is None or (isinstance(vote_ratio_holo_to_apo_chain, float) and (math.isnan(vote_ratio_holo_to_apo_chain) or math.isinf(vote_ratio_holo_to_apo_chain))):
            vote_ratio_holo_to_apo_chain_rounded = vote_ratio_holo_to_apo_chain
        else:
            vote_ratio_holo_to_apo_chain_rounded = round(vote_ratio_holo_to_apo_chain, 2)

        # Chain-level probabilities
        chain_mean_p_apo = sub_df['chain_avg_P(apo)'].mean()
        chain_mean_p_holo = sub_df['chain_avg_P(holo)'].mean()

        if pd.isna(chain_mean_p_apo) and pd.isna(chain_mean_p_holo):
            chain_highest_p_mean = None
        elif pd.isna(chain_mean_p_apo) or (chain_mean_p_holo > chain_mean_p_apo):
            chain_highest_p_mean = "holo"
        elif pd.isna(chain_mean_p_holo) or (chain_mean_p_apo > chain_mean_p_holo):
            chain_highest_p_mean = "apo"
        else:
            chain_highest_p_mean = "tie"


        # Formulate UBS string
        all_chains = set()
        for ch in sub_df['chains3'].dropna():
            all_chains.update(str(ch).split('-'))
        site_uniprots = " ".join(sorted(all_uniprots))
        #site_uniprots_key = "_".join(sorted(all_uniprots))  # filesystem-safe key
        site_recipe = "_".join(sorted(cluster_recipes))

        def _sort_key(uid):
            # uid format: UNP:THRESHOLD:UBSID
            # (Exact order will be deterministic by UniProt, then threshold, then ubs_id numeric.)
            try:
                unp, thr, cid = uid.split(':', 2)
                # sort ubs_id numerically when possible
                cid_num = int(cid) if cid.isdigit() else cid
                return (unp, thr, cid_num)
            except Exception:
                return (uid, )
        site_uid = " ".join(sorted(cluster_uid_parts, key=_sort_key))

        # Compute max number of chains for any job in the site
        chains_max = sub_df['n_chains3'].max() if not sub_df.empty else None
        max_n_chains3 = int(chains_max) if chains_max is not None and not pd.isna(chains_max) else None
        is_interface = (len(all_uniprots) > 1) or (max_n_chains3 is not None and max_n_chains3 > 1)

        # Get representative job
        representative_job = None
        representative_job_overlap = None
        representative_job_resis3 = None
        max_resis3 = None  # for UBS

        if not sub_df.empty:
            # Step 1: Store max resis3 for all jobs in this UBS
            max_resis3 = sub_df['#resis3'].max()
            # Step 2: Calculate overlaps for all jobs
            overlap_counts = {}
            if 'uniprot_residues_set' in sub_df.columns and intersection_residues:
                for jobname, row in sub_df.iterrows():
                    job_residues = row['uniprot_residues_set'] if isinstance(row['uniprot_residues_set'], set) else set()
                    overlap = len(job_residues & set(intersection_residues))
                    overlap_counts[jobname] = overlap

                if overlap_counts:
                    max_overlap = max(overlap_counts.values())
                    jobs_with_max_overlap = [jn for jn, ov in overlap_counts.items() if ov == max_overlap]
                    # Step 3: Among jobs with max overlap, pick one with highest resis3
                    if jobs_with_max_overlap:
                        max_resis3_in_overlap = -1
                        for jobname in jobs_with_max_overlap:
                            job_resis3 = sub_df.loc[jobname, '#resis3']
                            if job_resis3 > max_resis3_in_overlap:
                                max_resis3_in_overlap = job_resis3
                                representative_job = jobname
                        representative_job_overlap = max_overlap
                        representative_job_resis3 = sub_df.loc[representative_job, '#resis3'] if representative_job is not None else None
            else:
                # Fallback: just pick the job(s) with max resis3
                jobs_with_max_resis3 = sub_df[sub_df['#resis3'] == max_resis3].index.tolist()
                representative_job = jobs_with_max_resis3[0] if jobs_with_max_resis3 else None
                representative_job_overlap = None
                representative_job_resis3 = max_resis3 if representative_job is not None else None
        else:
            max_resis3 = None
            representative_job = None
            representative_job_overlap = None
            representative_job_resis3 = None

        # Computations
        # --- overlap means & paired averages (Pocket + Chain) ---
        overall_overlap_mean_pocket = sub_df['pocket_overall_overlap'].mean()
        mode_overlap_mean_pocket = sub_df['pocket_overall_mode_overlap'].mean()
        pair_overlap_mean_pocket = (round((overall_overlap_mean_pocket + mode_overlap_mean_pocket) / 2, 2)
            if not pd.isna(overall_overlap_mean_pocket) and not pd.isna(mode_overlap_mean_pocket) else None)

        overall_overlap_mean_chain = sub_df['chain_overall_overlap'].mean()
        mode_overlap_mean_chain = sub_df['chain_overall_mode_overlap'].mean()
        pair_overlap_mean_chain = (round((overall_overlap_mean_chain + mode_overlap_mean_chain) / 2, 2)
            if not pd.isna(overall_overlap_mean_chain) and not pd.isna(mode_overlap_mean_chain) else None)

        # --- density means & paired averages (Pocket + Chain) ---
        # Pocket
        pocket_mean_density_combined = sub_df['pocket_avg_normalized_density_combined'].mean()
        pocket_mean_density_strongest = (
            sub_df['pocket_avg_normalized_density_strongest'].mean()
            if 'pocket_avg_normalized_density_strongest' in sub_df.columns else np.nan)
        pair_density_mean_pocket = (
            round((pocket_mean_density_combined + pocket_mean_density_strongest) / 2, 2)
            if not pd.isna(pocket_mean_density_combined) and not pd.isna(pocket_mean_density_strongest) else None)

        # Chain (mirrors Pocket)
        chain_mean_density_combined = sub_df['chain_avg_normalized_density_combined'].mean()
        chain_mean_density_strongest = (
            sub_df['chain_avg_normalized_density_strongest'].mean()
            if 'chain_avg_normalized_density_strongest' in sub_df.columns else np.nan)
        pair_density_mean_chain = (
            round((chain_mean_density_combined + chain_mean_density_strongest) / 2, 2)
            if not pd.isna(chain_mean_density_combined) and not pd.isna(chain_mean_density_strongest) else None)

        # --- model pocket pLDDT avg (mean over entries in this UBS; ignore NaNs) ---
        plddt_col = "model_pocket_pLDDT_avg"
        if plddt_col in sub_df.columns:
            plddt_series = pd.to_numeric(sub_df[plddt_col], errors="coerce")
            plddt_mean_raw = plddt_series.mean() if not plddt_series.dropna().empty else None
            plddt_mean = round(plddt_mean_raw, 2) if plddt_mean_raw is not None and not pd.isna(plddt_mean_raw) else None
        else:
            plddt_mean = None

        # Threshold labeling
        # - Standard thresholds: keep the single threshold (e.g. ST_050)
        # - Large-jump thresholds (ST_LJ*): use a clean "ST_LJ" for grouping/plotting
        #   and store the original LJ thresholds involved in `threshold_detail` for provenance.
        if allow_cross_thresholds:
            threshold_clean = "ST_LJ"
            threshold_detail = ":".join(sorted(thresholds_in_group))
        else:
            threshold_clean = list(thresholds_in_group)[0] if thresholds_in_group else None
            threshold_detail = threshold_clean
        row_dict = {
            'site_uniprots': site_uniprots,
            #'site_uniprots_key': site_uniprots_key,
            'site_uid': site_uid,
            'site_recipe': site_recipe,
            'threshold': threshold_clean,
            'threshold_detail': threshold_detail,
            'is_interface': is_interface,
            'n_uniprots': len(all_uniprots),
            'n_entries': len(all_entry_ids),
            'max_n_chains3': max_n_chains3, #int(max_n_chains3) if not pd.isna(max_n_chains3) else None,
            'max_n_resis3': int(max_resis3) if max_resis3 is not None and not pd.isna(max_resis3) else None,
            'multimeric_status_all_unique': multimeric_joined, #multimeric_joined, multimeric_joined

            'n_union': len(union_residues),
            'n_intersection': len(intersection_residues),
            'union_residues': sorted(union_residues),
            'intersection_residues': sorted(intersection_residues),
            'sum_n_apo': sum_n_apo,
            'sum_n_holo': sum_n_holo,
            'ratio_n_holo_to_apo': ratio_n_holo_to_apo_rounded,

            'avg_n_apo': avg_n_apo,
            'avg_n_holo': avg_n_holo,
            'ratio_avg_n_holo_to_apo': ratio_avg_n_holo_to_apo,

            'model_pocket_pLDDT_avg_mean': plddt_mean,

            ## Classification stuff
            # Pocket
            'avg_P(apo)_mean_pocket': round(pocket_mean_p_apo, 3),
            'avg_P(holo)_mean_pocket': round(pocket_mean_p_holo, 3),
            #'avg_normalized_density_combined_mean_pocket': round(sub_df['pocket_avg_normalized_density_combined'].mean(), 2),

            'avg_normalized_density_combined_mean_pocket': (round(pocket_mean_density_combined, 2) if not pd.isna(pocket_mean_density_combined) else None),
            'avg_normalized_density_strongest_mean_pocket': (round(pocket_mean_density_strongest, 2) if not pd.isna(pocket_mean_density_strongest) else None),
            'pair_density_mean_pocket': pair_density_mean_pocket,
            
            'overall_overlap_mean_pocket': round(overall_overlap_mean_pocket, 2),
            'mode_overlap_mean_pocket': round(mode_overlap_mean_pocket, 2),
            'pair_overlap_mean_pocket': pair_overlap_mean_pocket,

            'votes_n_apo_pocket': votes_n_apo_pocket,
            'votes_n_holo_pocket': votes_n_holo_pocket,
            'votes_n_undefined_pocket': votes_n_undefined_pocket,
            'vote_ratio_holo_to_apo_pocket': vote_ratio_holo_to_apo_pocket_rounded,
            'highest_p_mean_pocket': pocket_highest_p_mean,
            'majority_vote_pocket': majority_vote_pocket,

            # Chain (mirroring Pocket metrics)
            'avg_P(apo)_mean_chain': round(chain_mean_p_apo, 3),
            'avg_P(holo)_mean_chain': round(chain_mean_p_holo, 3),
            #'avg_normalized_density_combined_mean_chain': round(sub_df['chain_avg_normalized_density_combined'].mean(), 2),

            'avg_normalized_density_combined_mean_chain': (round(chain_mean_density_combined, 2) if not pd.isna(chain_mean_density_combined) else None),
            'avg_normalized_density_strongest_mean_chain': (round(chain_mean_density_strongest, 2) if not pd.isna(chain_mean_density_strongest) else None),
            'pair_density_mean_chain': pair_density_mean_chain,

            'overall_overlap_mean_chain': round(overall_overlap_mean_chain, 2),
            'mode_overlap_mean_chain': round(mode_overlap_mean_chain, 2),
            'pair_overlap_mean_chain': pair_overlap_mean_chain, # round(pair_overlap_mean_chain, 3),

            'votes_n_apo_chain': votes_n_apo_chain,
            'votes_n_holo_chain': votes_n_holo_chain,
            'votes_n_undefined_chain': votes_n_undefined_chain,
            'vote_ratio_holo_to_apo_chain': vote_ratio_holo_to_apo_chain_rounded,
            'highest_p_mean_chain': chain_highest_p_mean,
            'majority_vote_chain': majority_vote_chain,

            # Representative job
            'representative_job': representative_job,
            'representative_job_overlap': representative_job_overlap,
            'representative_job_resis3': int(representative_job_resis3) if representative_job_resis3 is not None and not pd.isna(representative_job_resis3) else None,

            # Clustered jobs
            'job_names': ";".join(sorted(all_entry_ids))
            #'max_resis3_job_names': ";".join(max_resis3_jobs),
            #'representative_job': representative_job
        }
        results.append(row_dict)
        # Filtered view: only keep UBS groups that have at least one surviving member in df_summary
        if existing_entry_ids:
            row_dict_f = row_dict.copy()
            # preserve original size (before intersecting with df_summary) for traceability
            row_dict_f['n_entries_total'] = row_dict_f.get('n_entries')
            row_dict_f['n_entries'] = len(existing_entry_ids)
            row_dict_f['job_names'] = ";".join(sorted(existing_entry_ids))
            results_filtered.append(row_dict_f)
    return results, results_filtered


def run_all_unification(summary_csv_path, residues_csv_path, UBS_root, output_csv, output_csv_filtered=None, uniprots=None, limit_n_uniprots=None):
    """
    Runs true UBS unification for all thresholds, respecting cross-threshold rules.
    Saves a single output CSV with all unified sites.
    """
    # Load
    logger.info("Loading dataframes")
    df_summary = pd.read_csv(summary_csv_path)
    df_residues = pd.read_csv(residues_csv_path, low_memory=False)
    logger.info("Done")

    # Prepare
    logger.info("Pre-processing data")
    all_results = []
    filtered_results = []
    if 'n_chains3' not in df_summary.columns:
        df_summary['n_chains3'] = df_summary['chains3'].fillna('').apply(lambda c: len(str(c).split('-')) if c else 0)

    # Optionally, parse residues right away for all jobs
    df_residues = df_residues.set_index('job_name')
    df_residues['uniprot_residues_set'] = df_residues['uniprot_residues'].apply(
        lambda x: set(ast.literal_eval(x)) if pd.notnull(x) and x != '' else set())
    # Merge Into Main Summary DataFrame
    df_summary = df_summary.set_index('job_name')
    df_summary['uniprot_residues_set'] = df_residues['uniprot_residues_set']
    df_summary = df_summary.reset_index()
    logger.info("Done")
    
    ubs_groups_processed = {}


    # --- Standard thresholds ---
    for threshold in STANDARD_THRESHOLDS:
        logger.info(f"Processing standard threshold: {threshold}")
        clusters = load_ubs_clusters(UBS_root, threshold=threshold, lj=False, uniprots=uniprots, limit_n_uniprots=limit_n_uniprots)
        group_idxs = build_connected_components(clusters, allow_cross_thresholds=False)
        n_groups = len(group_idxs)
        ubs_groups_processed[threshold] = n_groups
        logger.info(f"{threshold}: {len(clusters)} clusters -> {n_groups} UBS groups")
        res_all, res_filt = aggregate_ubs_groups(group_idxs, clusters, df_summary, allow_cross_thresholds=False)
        all_results.extend(res_all)
        filtered_results.extend(res_filt)
    # --- Large-jump thresholds (processed together, merging is allowed) ---
    logger.info("Processing all large-jump thresholds (ST_LJ*) together")
    clusters_lj = load_ubs_clusters(UBS_root, lj=True, uniprots=uniprots, limit_n_uniprots=limit_n_uniprots)
    if clusters_lj:
        group_idxs_lj = build_connected_components(clusters_lj, allow_cross_thresholds=True)
        # Keep a clean key for reporting (output `threshold` for large-jump groups is "ST_LJ")
        lj_key = "ST_LJ"
        n_groups_lj = len(group_idxs_lj)
        ubs_groups_processed[lj_key] = n_groups_lj
        logger.info(f"{lj_key}: {len(clusters_lj)} clusters -> {n_groups_lj} UBS groups")
        res_all, res_filt = aggregate_ubs_groups(group_idxs_lj, clusters_lj, df_summary, allow_cross_thresholds=True)
        all_results.extend(res_all)
        filtered_results.extend(res_filt)

    # --- Save outputs ---
    df_ubs_unified = pd.DataFrame(all_results)
    try:
        df_ubs_unified.to_csv(output_csv, index=False)
        logger.info(f"Saved unified UBS aggregation (reference, may include empty groups) to: {output_csv}")
    except Exception as e:
        logger.info(f"Exception while trying to save unified UBS aggregation: {e}")

    df_ubs_unified_filtered = pd.DataFrame(filtered_results)
    if output_csv_filtered is not None:
        try:
            df_ubs_unified_filtered.to_csv(output_csv_filtered, index=False)
            logger.info(f"Saved unified UBS aggregation (filtered to summary members) to: {output_csv_filtered}")
        except Exception as e:
            logger.info(f"Exception while trying to save filtered unified UBS aggregation: {e}")

    return df_ubs_unified, df_ubs_unified_filtered, ubs_groups_processed

# ------------------------------- FUNCTIONS END -------------------------------




# --------------------------------- SETTINGS ---------------------------------

n_uniprots = None   # None or int (e.g., 5) for testing
VERSION    = 1.11


basedir = get_2level_cwd()

# ---- INPUT ----
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Conformational analysis & classification summary file (filtered survivors, +pLDDT)
summary_csv_path = str(cfg.FILTERED_SURVIVORS_CSV)

# AHoJ-DB entries UniProt residues file
residues_csv_path = str(cfg.QUERY_SUMMARIES_CLASS_CSV)

# Unified Binding Sites root (per-UniProt threshold TSVs)
UBS_root = str(cfg.UBS_ROOT_DIR)


# ---- OUTPUT ----
output_base_dir = str(cfg.UBS_AGG_DIR)

# -------------------------------- SETTINGS END -------------------------------



# Create datetime stamp for output dir
timestamp = f"{time.strftime('%d%m%Y_%H%M%S', time.gmtime())}"
dir_suffix = "" # "_filteredInput"
summary_subdir_timestamp = os.path.join(output_base_dir, f"{timestamp}_v{str(VERSION)}_{str(n_uniprots)}unplimit{str(dir_suffix)}")
os.makedirs(summary_subdir_timestamp, exist_ok=True)
output_csv = os.path.join(summary_subdir_timestamp, "ubs_allthresholds_aggregated_classifications.csv")
output_csv_filtered = os.path.join(summary_subdir_timestamp, "ubs_allthresholds_aggregated_classifications_filtered.csv")


# ---------- Logging setup ----------
log_path = os.path.join(summary_subdir_timestamp, "run.log")

# IMPORTANT: force=True ensures reconfiguration even if logging was set elsewhere
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,  # <---- THIS is the key
)
logger = logging.getLogger("UBS-Aggregation")
logger.info("Initialized logging at %s", log_path)


# ---------- Run manifest (arguments + paths) ----------
run_manifest = {
    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "version": VERSION,
    "n_uniprots": n_uniprots,
    "standard_thresholds": STANDARD_THRESHOLDS,
    "inputs": {
        "summary_csv_path": summary_csv_path,
        "residues_csv_path": residues_csv_path,
        "UBS_root": UBS_root,
    },
    "outputs": {
        "output_folder": summary_subdir_timestamp,
        "output_csv": output_csv,
        "output_csv_filtered": output_csv_filtered,
        "log_path": log_path,
    },
    "env": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "script": __file__ if "__file__" in globals() else None,
    },
}
manifest_path = os.path.join(summary_subdir_timestamp, "run_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(run_manifest, f, indent=2)
logger.info("Wrote run manifest to %s", manifest_path)


# ---------- Processing ----------
logger.info("Start processing")
start_time = time.time()

# You can add limit_n_uniprots or uniprots as arguments if you want to test on a subset
df_all, df_all_filtered, ubs_groups_processed = run_all_unification(
    summary_csv_path=summary_csv_path,
    residues_csv_path=residues_csv_path,
    UBS_root=UBS_root,
    output_csv=output_csv,
    output_csv_filtered=output_csv_filtered,
    limit_n_uniprots=n_uniprots
)

# Append manifest with post-run update
try:
    summary_update = {
        "result": {
            "rows": int(len(df_all)) if df_all is not None else 0,
            "rows_filtered": int(len(df_all_filtered)) if 'df_all_filtered' in globals() and df_all_filtered is not None else 0,
            "output_csv_filtered": str(output_csv_filtered) if 'output_csv_filtered' in globals() else None,

            # `threshold` is the clean grouping key (ST_030, ST_050, ST_070, ST_100, ST_LJ)
            "unique_thresholds": sorted(df_all["threshold"].dropna().unique().tolist())
                               if df_all is not None and "threshold" in df_all.columns else [],

            # Per-threshold row counts (reference vs filtered)
            "rows_by_threshold": (
                {str(k): int(v) for k, v in df_all.groupby("threshold").size().to_dict().items()}
                if df_all is not None and "threshold" in df_all.columns else {}
            ),
            "rows_filtered_by_threshold": (
                {str(k): int(v) for k, v in df_all_filtered.groupby("threshold").size().to_dict().items()}
                if df_all_filtered is not None and "threshold" in df_all_filtered.columns else {}
            ),

            # Per-threshold unique UniProt-group counts (site_uniprots)
            "unique_uniprot_groups": int(df_all["site_uniprots"].nunique())
                                    if df_all is not None and "site_uniprots" in df_all.columns else 0,
            "unique_uniprot_groups_by_threshold": (
                {str(k): int(v) for k, v in df_all.groupby("threshold")["site_uniprots"].nunique().to_dict().items()}
                if df_all is not None and "threshold" in df_all.columns and "site_uniprots" in df_all.columns else {}
            ),
            "unique_uniprot_groups_filtered_by_threshold": (
                {str(k): int(v) for k, v in df_all_filtered.groupby("threshold")["site_uniprots"].nunique().to_dict().items()}
                if df_all_filtered is not None and "threshold" in df_all_filtered.columns and "site_uniprots" in df_all_filtered.columns else {}
            ),

            "ubs_groups_processed": ubs_groups_processed,
        }
    }

# Merge & re-write manifest
    run_manifest.update(summary_update)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    logger.info("Appended run summary to manifest")
except Exception as e:
    logging.exception("Failed to update manifest with run summary: %s", e)

logger.info(f"All done in {time.time() - start_time:.1f} seconds")

logging.shutdown()

