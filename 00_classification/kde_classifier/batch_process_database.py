# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:32:36 2025

@author: Christos P. Feidakis
"""

"""
Given a root directory that includes AHoJ queries (e.g. AHoJ-DB), process all of
the entries/queries and optionally, save the results in a different mirrored root
directory.

# Generate plots that illustrate metric distribution, classify AF-DB models

    Processing: 
    - Characterize the conformational spectrum of each pocket by parsing
    selected metrics
    - Assign apo/holo status to the AlphaFold-DB model in the given entry

    Additional considerations
    - Allow multiprocessing of entries
    - Intermediate batch folders between root directory and entry directories

"""

import os
# Set threading environment variables BEFORE importing numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL  
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["BLIS_NUM_THREADS"] = "1"  # BLIS
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate
os.environ["NUMBA_NUM_THREADS"] = "1"  # Numba
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

import argparse
import time
import json
import multiprocessing
import pandas as pd
import numpy as np
import importlib
import zipfile
#import sys

from parse_AHoJ3_DBv2b import get_2level_cwd, get_all_batch_dirs, collect_job_directories, \
        print_aggregate_results, save_arguments, save_job_categories, select_first_n_jobs

from stats_plot import set_plot_font

# Run types (docs/RUN_TYPES.md): holo-2.1 (legacy v2c), holo-2.4, afc-2.4; run bundles (ahoj-db-pipeline)
import run_types as rt
from run_types import RunTypeError

# Force single-threading for matplotlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode



### Define functions

def parse_job_list_file(path):
    """AHOJ-018. Read a --job_list file into a set of job_names for targeted reprocessing.

    Each non-blank line is either "<job_name>" or "<job_name>: <reason>" (a Discarded_jobs.txt
    line). job_names contain no ':' (holo '<pdb>-<chain>-<lig>-<resnum>' or with '_', afc '<UniProt>_ST_100_<n>';
    a site_uid 'P46883:ST_100:14' is NOT a job key, its key form uses '_'), so we take everything
    before the first colon - letting a prior run's discard log be fed back verbatim. RUN_TYPES item 18.
    """
    with open(path, 'r', encoding='utf-8') as fh:
        return {ln.split(':', 1)[0].strip() for ln in fh if ln.strip()}


def select_batches(args, all_batch_dirs=list):
    """
    Select a number of batches from a list
    """
    if args.batch_limit == 0:
        batch_dirs = all_batch_dirs
    else:
        batch_dirs = all_batch_dirs[:args.batch_limit]
    if args.batch_selection:
        batch_dirs = args.batch_selection

    return batch_dirs


def worker(job_name, query_path, args, job_processing_func):
    # query_path: query result - either a directory or compressed tgz file
    # AHOJ-004: guard every job here so a single bad entry is logged as 'Error' instead of
    #   raising out of pool.starmap and aborting the WHOLE batch (catastrophic for a 500k run).
    #   The 5-tuple matches process_jobs()'s Error branch (detail[4] is filtered out by smart_concat).
    try:
        result = job_processing_func(args, job_name, query_path)
    except Exception as e:
        import traceback
        print(f'[ERROR] Unhandled exception in job {job_name} [{query_path}]: {type(e).__name__}: {e}')
        print(traceback.format_exc())
        result = ('Error', (job_name, f'{type(e).__name__}: {e}', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    return (job_name, result)


def process_jobs_serial(args, job_dict, job_processing_func):
    print(f'Processing {len(job_dict)} in 1 process')

    results = []
    for job_name, query_path in job_dict.items():
        result = worker(job_name, query_path, args, job_processing_func)
        results.append(result)
    return results


def _pool_init(font_name, svg_fonttype, random_seed=None, sites_tsv=None, additives_file=None):
    # Called once in each worker (child) process.
    # AHOJ-011: children do NOT inherit the parent's np.random state under 'spawn' (Windows), so
    #   reseed here. The KDE path is deterministic and the BGM pins its own random_state, so this
    #   is insurance for any future stochastic step rather than a current requirement.
    if random_seed is not None:
        np.random.seed(random_seed)
    set_plot_font(font_name=font_name, svg_fonttype=svg_fonttype)
    # RUN_TYPES: the sites table (afc-2.4 ion/additive rule) is loaded once per worker, not pickled per job.
    if sites_tsv:
        rt.set_sites(rt.load_sites_table(sites_tsv, additives=rt.additive_set(additives_file)))


def process_jobs_parallel(args, job_dict, num_processes, job_processing_func):
    print(f'Processing {len(job_dict)} in parallel with {num_processes} processes')
    with multiprocessing.Pool(
        num_processes,
        initializer=_pool_init,
        initargs=(args.plot_font, args.svg_fonttype, getattr(args, 'random_seed', None),
                  getattr(args, 'sites_tsv', None), getattr(args, 'additives_file', None))
    ) as pool:
        results = pool.starmap(
            worker,
            [(job_name, query_path, args, job_processing_func) for job_name, query_path in job_dict.items()]
        )
    return results


def smart_concat(lst):
    """Aggregate a list of DataFrames, dicts, or list-of-dicts into a single DataFrame."""
    filtered = [x for x in lst if x is not None and not (
        (isinstance(x, pd.DataFrame) and x.empty) or (isinstance(x, (dict, list)) and not x))]
    if not filtered:
        return pd.DataFrame()
    first = filtered[0]
    if isinstance(first, pd.DataFrame):
        return pd.concat(filtered, ignore_index=True)
    elif isinstance(first, dict):
        return pd.DataFrame(filtered)
    elif isinstance(first, list):
        # Expecting list of dicts
        all_dicts = []
        for l in filtered:
            if isinstance(l, dict):
                all_dicts.append(l)
            else:
                all_dicts.extend(l)
        return pd.DataFrame(all_dicts)
    else:
        raise ValueError("Unknown summary data type for aggregation!")


def zip_file(file_path, delete_original=False):
    """Zip one saved output file as <file_path>.zip and optionally remove the original."""
    if not os.path.isfile(file_path) or file_path.lower().endswith(".zip"):
        return None

    zip_path = f"{file_path}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=os.path.basename(file_path))

    if delete_original:
        os.remove(file_path)

    return zip_path


def zip_summary_output_files(summary_dir, delete_original=False):
    """Zip each immediate file in a summary run directory separately."""
    zipped_files = []
    for name in os.listdir(summary_dir):
        file_path = os.path.join(summary_dir, name)
        zipped_path = zip_file(file_path, delete_original=delete_original)
        if zipped_path:
            zipped_files.append(zipped_path)
    return zipped_files


def label_totals(query_summaries_df):
    """
    AHOJ-020: run totals of the per-entry label columns: the A- rows pooled as apo
    (n_apo_query_demoted) and the candidate labels that joined no set (unknown_labels, 'X:3;Y:1').
    None when the summary carries neither column.
    """
    qs = query_summaries_df
    if qs is None or qs.empty:
        return None
    out = {}
    if "n_apo_query_demoted" in qs.columns:
        v = pd.to_numeric(qs["n_apo_query_demoted"], errors="coerce").fillna(0)
        out["n_apo_query_demoted"] = int(v.sum())
        out["n_jobs_with_apo_query_demoted"] = int((v > 0).sum())
    if "unknown_labels" in qs.columns:
        tot = {}
        cells = qs["unknown_labels"].dropna().astype(str)
        for cell in cells:
            for part in cell.split(";"):
                lab, _, n = part.rpartition(":")
                tot[lab] = tot.get(lab, 0) + int(n)
        out["unknown_labels"] = dict(sorted(tot.items()))
        out["n_unknown_label"] = int(sum(tot.values()))
        out["n_jobs_with_unknown_label"] = int(len(cells))
    return out or None


def save_run_manifest(directory, args, funcname, result_counts, query_summaries_df,
                      n_batch_dirs, n_jobs, query_summaries_path, start_time,
                      metric_summaries_df=None, metrics_path=None):
    """
    AHOJ-015: write a single self-describing run_manifest.json consolidating run metadata, aggregate
    job counts, and output pointers. Complements parameters.json (full args) and the job-categories
    file; gives one machine-readable record per run (provenance for the full-DB reruns).
    Best-effort: git commit / host lookups are guarded so they never abort a finished run.
    """
    import socket
    import sys
    import platform
    import subprocess

    def _git_commit():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

    counts = {k: v.get("count", 0) for k, v in result_counts.items()}
    has_rows = query_summaries_df is not None and not query_summaries_df.empty
    manifest = {
        "run_id": os.path.basename(directory),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_function": funcname,
        "command": " ".join(sys.argv),
        "config": getattr(args, "config", None),
        # RUN_TYPES: which path was requested, and the bundle's run.yaml when one was given
        "run_type": getattr(args, "run_type", None),
        "bundle": getattr(args, "bundle", None),
        "run_yaml": getattr(args, "run_yaml", None),
        "sites_tsv": getattr(args, "sites_tsv", None),
        "run_types_in_summary": (sorted(query_summaries_df["run_type"].dropna().astype(str).unique().tolist())
                                 if has_rows and "run_type" in query_summaries_df.columns else None),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "num_processes": getattr(args, "num_processes", None),
        "random_seed": getattr(args, "random_seed", None),
        "input": {
            "root_data_dir": getattr(args, "root_data_dir", None),
            "n_batch_dirs": n_batch_dirs,
            "n_jobs_selected": n_jobs,
        },
        "results": {"counts": counts, "total_jobs": sum(counts.values())},
        # AHOJ-020: A- rows pooled as apo, and candidate labels no set took, over the summarised entries
        "labels": label_totals(query_summaries_df),
        "outputs": {
            "summary_dir": directory,
            "query_summaries_csv": query_summaries_path,
            "n_query_summary_rows": int(len(query_summaries_df)) if query_summaries_df is not None else 0,
            "n_query_summary_cols": int(query_summaries_df.shape[1]) if has_rows else 0,
            "query_metrics_csv": metrics_path,
            "n_query_metrics_cols": int(metric_summaries_df.shape[1]) if (metric_summaries_df is not None and not getattr(metric_summaries_df, "empty", True)) else 0,
        },
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    path = os.path.join(directory, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def process_jobs(job_dict, num_processes, args, job_processing_func):

    aggregate_results = {
        'Success': {'count': 0, 'total_rows': 0, 'jobs': []},
        'File Not Found': {'count': 0, 'jobs': []},
        'Error': {'count': 0, 'jobs': []},
        'Discarded': {'count': 0, 'jobs': []}
    }

    # Processing step (job_processing_func)
    apo_summary_list = []
    holo_summary_list = []
    query_info_list = []
    metrics_summary_list = []  # AHOJ-015: detached metric-stats rows (detail[5]), when present

    apo_values_list, holo_values_list, query_info_list = [], [], []

    if num_processes == 1:
        results = process_jobs_serial(args, job_dict, job_processing_func)
    else:
        results = process_jobs_parallel(args, job_dict, num_processes, job_processing_func)
    print('All directories processed, summarizing results')

    for job_name, result in results:
        status, detail = result

        if status == 'Success':
            aggregate_results['Success']['count'] += 1
            aggregate_results['Success']['total_rows'] += detail[1]
            aggregate_results['Success']['jobs'].append(job_name)
            apo_summary_list.append(detail[2])
            holo_summary_list.append(detail[3])
            query_info_list.append(detail[4])
            if isinstance(detail, tuple) and len(detail) > 5:
                metrics_summary_list.append(detail[5])
            #apo_values_list.append(detail[2].iloc[1:2])
            #holo_values_list.append(detail[3].iloc[1:2])
        elif status == 'File Not Found':
            aggregate_results['File Not Found']['count'] += 1
            aggregate_results['File Not Found']['jobs'].append(job_name)
            if isinstance(detail, tuple) and len(detail) > 4:
                query_info_list.append(detail[4])
        elif status == 'Discarded':
            aggregate_results['Discarded']['count'] += 1
            aggregate_results['Discarded']['jobs'].append((job_name, detail[1]))  # Save job name and reason
            if isinstance(detail, tuple) and len(detail) > 4:
                query_info_list.append(detail[4])
        else:
            aggregate_results['Error']['count'] += 1
            aggregate_results['Error']['jobs'].append(job_name)
            if isinstance(detail, tuple) and len(detail) > 4:
                query_info_list.append(detail[4])

    # Smart aggregation
    apo_summaries = smart_concat(apo_summary_list)
    holo_summaries = smart_concat(holo_summary_list)
    query_info_summaries = smart_concat(query_info_list)
    metric_summaries = smart_concat(metrics_summary_list)  # AHOJ-015 (empty unless split is on)
    #apo_summaries = pd.concat(apo_values_list, ignore_index=True)
    #holo_summaries = pd.concat(holo_values_list, ignore_index=True)
    #query_info_summaries = pd.concat(query_info_list, ignore_index=True)

    print('Done collecting summaries')

    return aggregate_results, apo_summaries, holo_summaries, query_info_summaries, metric_summaries



### Arguments

# Per-entry job functions selectable via --processing_function (imported from batch_job_functions.py).
# name -> one-line description. Keys are the valid --processing_function choices.
PROCESSING_FUNCTIONS = {
    'do_kde_plots_and_classification':
        'PRODUCTION classification run: full KDE + DP/z/robust/composite apo-holo classification, '
        'plots (DP & KDE curves, violins, confidence bars) and per-entry statistics + the aggregate query summary.',
    'do_collect_model_pLDDT':
        'Lightweight collector: AlphaFold model pocket-average pLDDT + query resolution/exp_method; no per-entry files (for very large DBs).',
    'do_collect_uniprot_residues':
        'Collector: UniProt residue indices of each query pocket from the DB residue-mapping files.',
    'do_collect_postcomputed_statistics':
        'Collector: flatten previously-saved apo_holo_all_stats_*.json (pocket/chain) into one row per entry; needs --stats_root_dir.',
    'do_collect_apoholo_metric_stats':
        'Lightweight apo-vs-holo metric statistics (no KDE/plots/classification): per-entry apo/holo means + '
        'diff_means (|mean_apo-mean_holo|) for pocket & extra metrics incl. RoG - the non-cancelling delta for '
        'the apo/holo %-difference figure.',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process zipped AHoJ-DB raw data.")

    # Optional JSON parameter config (e.g. a saved parameters.json or configs/run_config_*.json).
    # Values in the file become defaults; any flag passed explicitly on the CLI still overrides them.
    parser.add_argument('--config',                 type=str,   default=None,   help='Path to a JSON parameters config. Its values become defaults; explicit CLI flags override them.')

    # Entry filtering
    parser.add_argument('--resolution_soft',        type=float, default=None,    help='None or float. Soft max-resolution cutoff applied to CANDIDATE rows only (role C); candidates with NaN resolution are kept here (use --drop_no_resolution to remove them). AHOJ-005.')
    parser.add_argument('--drop_no_resolution',      type=int,   default=0,      help='0/1. Drop CANDIDATE rows with no resolution (e.g. SOLUTION_NMR, stored as "-" -> NaN). Recommended 1 for the production run (NMR query pockets can inflate residue counts downstream). AHOJ-005.')
    parser.add_argument('--bndgrsds_prcnt_seqres',  type=int,   default=75,    help='None or int. Threshold for all percent_mapped_sqr (hard filter). Condition for keeping is >= threshold')
    parser.add_argument('--bndgrsds_prcnt_obs',     type=int,   default=75,    help='None or int. Threshold for all percent_mapped_obs (hard filter). Condition for keeping is >= threshold')

    parser.add_argument('--full_pockets_aligned',   type=int,   default=0,      help='Reject candidate pockets with unobserved residues (also rejects results with poorly mapped pockets)')
    parser.add_argument('--pocket_len_resis3',      type=int,   default=0,      help='Reject candidate pockets with mismatch between number of observed and aligned residues, i.e., pocket_len = #resis3 (to avoid metrics from poorly mapped pockets')
    parser.add_argument('--min_apo',                type=int,   default=0,      help='Minimum number of apo rows required in dataframe to process entry')
    parser.add_argument('--min_holo',               type=int,   default=0,      help='Minimum number of holo rows required in dataframe to process entry')
    parser.add_argument('--min_apoholo',            type=int,   default=1,      help='Minimum number of combined apo and holo rows required in dataframe to process entry')
    parser.add_argument('--keep_no_model',           type=int,   default=0,      help='0/1. AHOJ-018. 0 (default): discard entries with no AlphaFold model (role M) row - nothing to classify (AlphaFold-project use case). 1: KEEP and process them - the conformational analysis/plots are model-independent and still useful (webserver use case); the summary row is flagged has_model=0 with an "N/A (no model)" verdict.')
    parser.add_argument('--columns_nozero',         nargs='+',  default=['RoG', 'bfac_t', 'bfac_avg', 'resolution'], help='Columns that should not have zero values, replace with nan (0: np.nan, "": np.nan)')

    # Relabel apo/holo assignments based on ligand exclusion list (NEW)
    parser.add_argument('--relabel_apoholo',        type=int,   default=1,      help='Change the apoholo_assignment in global_results for holo candidates that only bind a single atom ligand (assign A**)')
    parser.add_argument('--add_query_row',          type=int,   default=None,   help='RUN_TYPES: None = the run-type profile decides (holo: 1, the query row joins the holo set; afc-2.4: 0, the query row IS the model). An explicit 1 on an AF-centric archive is an error.')
    #parser.add_argument('--save_relabeled_df',      type=int,   default=0,      help='Save the relabeled dataframe as global_results_relabeled.csv')

    # Run types (docs/RUN_TYPES.md). Every archive is read for its AHoJ version and query mode; the
    # profile (holo-2.1 | holo-2.4 | afc-2.4) then sets the rules below. Each rule can be overridden
    # from the config; None keeps the profile value. Contradictions are loud errors per job.
    parser.add_argument('--run_type',               type=str,   default='auto', choices=['auto'] + list(rt.RUN_TYPES), help='auto: detect per archive (script_parameters.json script_version + global_results.csv query_mode). An explicit value is checked against every archive and against the bundle.')
    parser.add_argument('--bundle',                 type=str,   default=None,   help='A run bundle directory (ahoj-db-pipeline docs/run-bundle.md): run.yaml gives run_type and the AHoJ version, sites.tsv the locked pockets, output/archives the .tgz files. Sets --root_data_dir and --sites_tsv.')
    parser.add_argument('--sites_tsv',              type=str,   default=None,   help='AF-centric sites table (site_uid, uniprot, residues, members[, ligands]). Needed by ion_rule/additive_rule "site" to know which sites are ion- or additive-defined.')
    parser.add_argument('--relabel_mode',           type=str,   default=None,   choices=[None] + list(rt.RELABEL_MODES), help='v2c_ions (holo-2.1), v2c_ions_demote_only (holo-2.4: never undo the gate), entity_gate_site (afc-2.4: ligands.json entities), none.')
    parser.add_argument('--ion_rule',               type=str,   default=None,   choices=[None] + list(rt.ION_RULES), help='query_disables (v2c: ions count when the query ligand is an ion), site (ions count only at ion-defined sites), nonion, all.')
    parser.add_argument('--additive_rule',          type=str,   default=None,   choices=[None] + list(rt.ADDITIVE_RULES), help='v2c (additives count as holo; default everywhere), site (only at additive-defined sites; the gate_site_noadd sensitivity row).')
    parser.add_argument('--gate',                   type=str,   default=None,   choices=[None] + list(rt.GATE_MODES), help='entity relabel only. archive: apply the gate only if the AHoJ run had holo_overlap_mode=gate; apply: always apply the archive thresholds (2.4 defaults 2/20/30); none: the v2c contact rule.')
    parser.add_argument('--volume_column',          type=str,   default=None,   help='Column copied into "volume" before curation (2.4: volume, volume2, volume_seed, ...). Missing column = error.')
    parser.add_argument('--volume_nan_to_zero',     type=int,   default=None,   help='0/1. The v2c rule: a NaN volume with #resis3 > 0 becomes 0.0 (written for PyVOL). Profile default 1 on every path.')
    parser.add_argument('--model_row',              type=str,   default=None,   choices=[None] + list(rt.MODEL_ROW_MODES), help='AF_model: the model row of the query UniProt (default): structure == "AF_model" before AHoJ PR #143, since then the role-M row whose UNPs are all query UNPs (UniWeave adds other role-M rows). first_M: every role-M row (legacy).')
    parser.add_argument('--additives_file',         type=str,   default=None,   help='One CCD code per line; replaces the built-in crystallisation-additive list (run_types.ADDITIVES_DEFAULT).')
    parser.add_argument('--key_format',             type=str,   default=None,   choices=[None] + list(rt.KEY_FORMATS), help='Job-key shape to enforce: holo <pdb>-<chain>-<lig>-<resi> or <pdb>_<chain>_<lig>_<resi>, afc <UniProt>_ST_100_<n>, any. The profile decides by default; "any" is for the pilot, whose holo pass is named by site.')
    parser.add_argument('--allow_undo_gate',        type=int,   default=0,      help='0/1. Permit relabel_mode v2c_ions on a holo-2.4 run made with holo_overlap_mode=gate (the v2c contact rule then undoes the gate). Off by default; on only for a deliberate comparison.')

    # States filtering (apo - holo results within entry)
    #parser.add_argument('--crypt_rms',              type=float, default=2.0,    help='pocket RMSD threshold for considering cryptic or flexible pockets')
    parser.add_argument('--filter_candidates',      type=int,   default=1,      help='Choose whether to filter alternative states of an entry (global_results.csv) or not')
    parser.add_argument('--rog_diff_thrs_query',    type=float, default=0.6,    help='float or None, e.g., 0.3: candidates whose RoG differs more than 0.3*query_RoG (+/-) from the (mean) query RoG are filtered out (lower=stricter)')
    parser.add_argument('--rog_diff_thrs_mean',     type=float, default=None,   help='float or None, e.g., 0.3: candidates whose RoG differs more than 0.3*mean_RoG (+/-) from the average candidate RoG are filtered out (lower=stricter)')
    # AHOJ-003: pocket_dist is measured AFTER whole-chain superposition, so a large value reflects
    #   GLOBAL conformational change (domain rearrangement / hinge motion), NOT a wrong pocket mapping
    #   or a poor local pocket fit. The pocket classification uses only LOCAL metrics
    #   (pocket_rms/sasa/volume/RoG) and never pocket_dist, so filtering candidates on distance only
    #   discards valid local-pocket evidence. Decision (2026-06-30): DROP this top-level filter
    #   (default None = off). pocket_dist is retained as a chain metric, and extreme global
    #   displacements are FLAGGED (see --dist_outlier_thr) so the secondary chain classification can be
    #   marked downstream rather than losing candidates. See KNOWN_ISSUES.md (AHOJ-003).
    parser.add_argument('--pocket_dist',            type=float, default=None,   help='Optional hard filter: drop candidate pockets farther than X (A) from the query pocket after global chain superposition. Default None = OFF (distance is global, not a local-fit signal; see AHOJ-003). Set a value only to re-enable the old behavior.')
    parser.add_argument('--dist_outlier_thr',       type=float, default=20.0,   help='pocket_dist (A) above which a candidate is counted a global-displacement outlier. Not a filter - only populates chain_pocket_dist_max / chain_n_dist_outliers in the summary to flag chain classifications skewed by large global motions.')

    # KDE settings
    parser.add_argument('--kde_extend',             type=bool,  default=True,     help='Choose whether to extend the KDE density at the edges for smoother tails (affects classification, default cut factor=2)')
    parser.add_argument('--margin_cap_scale',       type=float, default=1.0,      help='Scales max tail extension cap; 1.0 preserves previous defaults')
    parser.add_argument('--bw_mode',                type=str,   default="pooled", help='')
    parser.add_argument('--bw_scale',               type=float, default=0.75,     help='')
    parser.add_argument('--prior_mode',             type=str,   default="equal",  help='')

    # Processing options
    parser.add_argument('--batch_limit',            type=int,   default=0,       help='0: no limit, integer: number of batches. Number of first n batch directories to process, out of total detected')
    parser.add_argument('--batch_selection',        type=list,  default=[],     help='Select specific batch directory/ies to process. i.e., default=["103", "104"]')
    parser.add_argument('--job_limit',              type=int,   default=None,    help='Select the first n entries/jobs to process out of all collected. e.g., default=20')
    parser.add_argument('--job_list',               type=str,   default=None,    help='AHOJ-018. Path to a text file with one job_name per line; restricts processing to EXACTLY those entries (intersected with the collected jobs). One line may be "<job_name>" or "<job_name>: <reason>" (a Discarded_jobs.txt line - the reason after ":" is ignored), so a prior run\'s discard log can be fed back directly for targeted reprocessing. Applied before --job_limit.')
    parser.add_argument('--num_processes',          type=int,   default=None,    help='Number of processes to use for multiprocessing (1 = serial processing), i.e., None (=1/2 cores) / 12 / 28.')
    parser.add_argument('--random_seed',            type=int,   default=42,      help='Random seed for reproducible results')
    parser.add_argument('--apo_holo_sums',          type=int,   default=0,       help='Calculate aggregate apo_summaries and holo_summaries CSV rows for jobs that produce them')
    #parser.add_argument('--union',                  type=int,   default=0,       help='0/1/. 1=Get database statistics (apoholo_averages) for pockets from entries with at least one apo and one holo pocket.')

    # Input & Output directories
    parser.add_argument('--root_data_dir',          type=str,   default=None,   help='Input root directory with batch directories and zipped tar files inside (.tgz)')
    parser.add_argument('--root_mirror_dir',        type=str,   default=None,   help='Output root directory to save data in a mirrored database structure')
    parser.add_argument('--summary_dir',            type=str,   default=None,   help='Output directory to save reports, parameters and overall summary files')
    #parser.add_argument('--db_version',             type=str,   default='v2b',  help='AHoJ-DB version being parsed')

    # Conformational analysis options
    parser.add_argument('--pocket_class_metrics',   type=list,  default=["pocket_rms", "sasa", "volume", "RoG"], help='Select which metrics to use for model pocket classification')
    parser.add_argument('--chain_class_metrics',    type=list,  default=["pocket_dist", "tm_score", "rmsd"],     help='Select which metrics to use for model chain classification')
    parser.add_argument('--separation_weights',     type=dict,  default={'w_center': 0.5, 'w_iqr': 0.3, 'w_extremes': 0.2},   help='Custom weights to use for composite point classification')

    # Select what to plot & save
    parser.add_argument('--general_violins',        type=int,   default=0,      help='Violin plots generated from the apo/holo data points right after the data is filtered')
    parser.add_argument('--dp_curves',              type=int,   default=1,      help='Dirichlet process micture model density estimates & used for AlphaFold model classification')
    parser.add_argument('--kde_curves',             type=int,   default=1,      help='KDE density estimates produced used for AlphaFold model classification')
    parser.add_argument('--kde_violins',            type=int,   default=1,      help='Violin plots generated from the pre-computed KDEs used for the classification')
    parser.add_argument('--conf_bars',              type=int,   default=1,      help='Confidence bar plots created from the KDE data')

    # Plot settings
    parser.add_argument('--separate_metrics',       type=int,   default=1,      help='Plot each metric in a separate figure')
    parser.add_argument('--show_plots',             type=int,   default=0,      help='Only for testing, keep it off')
    parser.add_argument('--plot_font',              type=str,   default='Helvetica Neue',    help='Set the default font to be used in plots. E.g., "Helvetica Neue" or "Arial"')
    parser.add_argument('--svg_fonttype',           type=str,   default='none', help='["none", "path"]: Choose how to save svg font - "none" is default, "path" will save glyphs as svg (larger file size)')

    # Save statistics etc.
    parser.add_argument('--save_statistics',        type=int,   default=1,      help='Save statistics and classification results for each entry')
    parser.add_argument('--save_filtered_df',       type=int,   default=1,      help='Save the filtered dataframe that for each entry, which was used for classification and statistics')
    parser.add_argument('--save_per_method_json',   type=int,   default=0,      help='Also write the 8 legacy per-method classification JSONs (z/robust/composite/dirichlet x pocket/chain). Off by default: their verdicts are in classification_summary.json. AHOJ-013.')
    parser.add_argument('--save_relabeled_csv',     type=int,   default=0,      help='Also write global_results_relabeled.csv. Off by default: the relabeled apoholo_assignment already persists in global_results_plots.csv.')

    # Query-summary metric export
    parser.add_argument('--query_summary_metrics',  nargs='+',  default=['pocket_rms', 'volume', 'sasa', 'RoG', 'tm_score', 'rmsd', 'pocket_len', 'bfac_avg'], help='AHOJ-015. Candidate metrics summarized into the aggregate query summary. pocket_len = #site residues (P2Rank site_residues correlate); bfac_avg = candidate B-factor (experimental flexibility). tm_score_i dropped (redundant with tm_score).')
    parser.add_argument('--query_summary_deltas',   type=int,   default=1,      help='AHOJ-015. 0: raw metrics, 1: raw plus query deltas (keeps raw volume/sasa/RoG correlates AND deltas), 2: replace selected pocket-size metrics with query deltas. Default 1: deltas=2 drops the raw size stats the prior run correlated with P2Rank. Note: with 1, local_flexibility averages CV over both raw and delta of sasa/volume/RoG.')
    parser.add_argument('--query_summary_stats',    nargs='+',  default=['min', 'max', 'mean', 'std_dev', 'iqr'], help='AHOJ-015. Which per-metric stats to emit into the query summary (curated to keep the table lean). Full set: min max range mean std_dev q1 q3 iqr confidence_interval_low confidence_interval_high coefficient_of_variation. Dropped ones are recomputable (range=max-min, etc.). local/global flexibility always emitted.')
    parser.add_argument('--separate_metric_summary', type=int,  default=1,      help='AHOJ-015. 1: write the descriptive metric-stats block to its OWN file (query_metrics_<funcname>.csv, joins on job_name) so the classification summary stays lean. 0: keep the block inline in the classification summary (legacy).')

    # Post-computed statistics JSON collection
    parser.add_argument('--stats_root_dir',         type=str,   default=None,   help='Root directory with mirrored post-computed statistics job folders')
    parser.add_argument('--stats_groups',           nargs='+',  default=['All'], help='Statistic groups to flatten from metric-first JSONs, e.g. All Apo Holo')
    parser.add_argument('--stats_scopes',           nargs='+',  default=['pocket', 'chain'], help='Statistics scopes to collect, e.g. pocket chain')
    parser.add_argument('--stats_group_fields',     nargs='+',  default=['mean', 'median', 'min', 'max', 'std', 'iqr', 'q1', 'q3', 'range', 'range_theor'], help='Group statistic fields to export; use "all" to export all scalar group fields')
    parser.add_argument('--stats_metric_fields',    nargs='+',  default=['diff_means', 'cohen_d', 'overlap_coef', 'robust_sep'], help='Metric-level fields to export; use "all" to export all scalar metric fields')
    parser.add_argument('--stats_include_nested',   type=int,   default=0,      help='1: recursively flatten nested group/metric structures selected by the field arguments')

    # Zipping
    parser.add_argument('--zip_plots_dir',          type=int,   default=0,      help='Choose whether or not to zip the entire plots directory')
    parser.add_argument('--zip_stats_dir',          type=int,   default=0,      help='Choose whether or not to zip the entire statistics directory')
    parser.add_argument('--del_dir_after_zip',      type=bool,  default=True,   help='Delete original directory after zipping')
    parser.add_argument('--zip_summary_files',      type=int,   default=0,      help='Zip each saved file in the summary run directory separately')
    parser.add_argument('--del_summary_files_after_zip', type=int, default=0,   help='Delete original summary files after --zip_summary_files 1')


    # Processing Function: the per-entry job function (see PROCESSING_FUNCTIONS above for the
    # name -> description map; choices and help text are generated from it).
    parser.add_argument('--processing_function',    type=str,   default='do_kde_plots_and_classification',
        choices=list(PROCESSING_FUNCTIONS),
        help='Per-entry job function (default: do_kde_plots_and_classification). Options -- '
             + '  ||  '.join(f'{name}: {desc}' for name, desc in PROCESSING_FUNCTIONS.items()))

    # Function arguments
    parser.add_argument('--collect_query_meta',     type=int, default=1,        help='Also collect query metadata (resolution, exp_method) from global_results.csv')
    parser.add_argument('--query_meta_role',        type=str, default='Q',      help='Role to harvest query metadata from (usually Q; set to M if needed)')

    # transparency

    # Apply --config (if given) as defaults so a single JSON file can drive a full run.
    # Precedence: explicit CLI flag > config file value > built-in default.
    pre_args, _ = parser.parse_known_args()
    if getattr(pre_args, 'config', None):
        with open(pre_args.config, 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
        valid_dests = {a.dest for a in parser._actions}
        unknown = sorted(k for k in cfg if k not in valid_dests)
        if unknown:
            print(f'[config] ignoring {len(unknown)} unknown key(s): {unknown}')
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in valid_dests})
        print(f'[config] loaded {pre_args.config}')

    return parser.parse_args()







def main():

    # Parse arguments, get start time, etc
    args = parse_arguments()
    np.random.seed(args.random_seed)
    start_time = time.time()
    two_level_cwd = get_2level_cwd()
    #print(f'\nInput arguments:\n{args}\n')


    ### Input
    # RUN_TYPES item 19: no default input any more (the old default pointed at AHoJ-DB v2c). Either a
    # run bundle (docs/run-bundle.md) or an explicit --root_data_dir.
    args.run_yaml = None
    if args.bundle:
        try:
            run_yaml = rt.read_run_yaml(args.bundle)
        except RunTypeError as e:
            raise SystemExit(f'[bundle] {e}')
        bp = rt.bundle_paths(args.bundle)
        bundle_rt = run_yaml['_parse_ahoj_run_type']
        if args.run_type not in ('auto', bundle_rt):
            raise SystemExit(f'[bundle] run.yaml says {bundle_rt} (run_type {run_yaml["run_type"]}, AHoJ '
                             f'{run_yaml["ahoj"].get("version")}) but --run_type {args.run_type} was given')
        args.run_type = bundle_rt
        args.root_data_dir = args.root_data_dir or bp['archives']
        if bundle_rt.startswith('afc') and not args.sites_tsv and os.path.isfile(bp['sites_tsv']):
            args.sites_tsv = bp['sites_tsv']
        args.run_yaml = {k: v for k, v in run_yaml.items() if not k.startswith('_')}
        print(f"[bundle] {args.bundle}: run_id {run_yaml.get('run_id')}, run type {bundle_rt}, "
              f"AHoJ {run_yaml['ahoj'].get('version')} @ {run_yaml['ahoj'].get('commit')}")
    if not args.root_data_dir:
        raise SystemExit('--root_data_dir (or --bundle) is required; there is no default input directory')
    args.root_data_dir = os.path.join(two_level_cwd, args.root_data_dir)
    if not os.path.isdir(args.root_data_dir):
        raise SystemExit(f'root_data_dir does not exist: {args.root_data_dir}')
    if args.stats_root_dir:
        args.stats_root_dir = os.path.join(two_level_cwd, args.stats_root_dir)
    if args.sites_tsv and not os.path.isfile(args.sites_tsv):
        raise SystemExit(f'sites_tsv does not exist: {args.sites_tsv}')


    ### Output
    # The root directory of the mirrored output of AHoJ-DB
    #args.root_mirror_dir = args.root_mirror_dir or os.path.join(two_level_cwd, r'<AHoJ runs dir>\v2c\meta-output')

    # Dir for reports, summaries and metadata. RUN_TYPES item 19: no cloud-folder fallback; a bundle is
    # read-only for parse-ahoj (run-bundle.md: only ahoj-compute writes into output/), so the summary
    # directory must be named explicitly.
    if not args.summary_dir:
        raise SystemExit('--summary_dir is required (the summaries, parameters.json and run_manifest.json go there)')
    args.summary_dir = os.path.join(two_level_cwd, args.summary_dir)


    ### Set-up processing preferences

    ## Sites table for the afc-2.4 ion/additive rule: loaded here for the serial path and re-loaded in
    ## every pool worker (_pool_init) so the table is never pickled per job.
    if args.sites_tsv:
        try:
            _sites = rt.load_sites_table(args.sites_tsv, additives=rt.additive_set(args.additives_file))
        except RunTypeError as e:
            raise SystemExit(f'[sites_tsv] {e}')
        rt.set_sites(_sites)
        _uniq = {s.site_uid: s for s in _sites.values()}
        print(f'[sites_tsv] {args.sites_tsv}: {len(_uniq)} sites, '
              f'{sum(s.ion_defined for s in _uniq.values())} ion-defined, '
              f'{sum(s.additive_defined for s in _uniq.values())} additive-defined')

    ## Select directories/entries to process
    # Get the list of immediate sub-directories (batches) in root dir ('000', '001', ..). A flat layout
    # (archives directly under root, as the AF-centric pilot wrote them) is one pseudo-batch ''.
    all_batch_dirs = get_all_batch_dirs(args.root_data_dir)
    if any(f.lower().endswith('.tgz') for f in os.listdir(args.root_data_dir)):
        all_batch_dirs = [''] + all_batch_dirs

    # Select certain batches to process (optional according to argument, default = everything)
    batch_dirs = select_batches(args, all_batch_dirs)
    print(f'\nSelected [{len(batch_dirs)}] out of [{len(all_batch_dirs)}] batch dir(s) in root directory for processing: {batch_dirs}')

    # Get list of entries in selected batches
    job_dict = collect_job_directories(args.root_data_dir, batch_dirs) # key: AHoJ entry, value:tar directory

    # AHOJ-018: optional targeted reprocessing. Restrict to exactly the job_names in --job_list
    # (intersected with the collected jobs). Each line is "<job_name>" or "<job_name>: <reason>"
    # (a Discarded_jobs.txt line); job_names contain no ':' so we split on the first colon.
    if args.job_list:
        wanted = parse_job_list_file(args.job_list)
        present = wanted & set(job_dict)
        missing = wanted - present
        job_dict = {k: v for k, v in job_dict.items() if k in present}
        print(f'[job_list] {args.job_list}: restricting to [{len(job_dict)}] of [{len(wanted)}] '
              f'requested job(s); [{len(missing)}] not found in the selected batches.')
        if missing:
            _preview = ', '.join(sorted(missing)[:10])
            print(f'[job_list] first missing: {_preview}{" ..." if len(missing) > 10 else ""}')

    # Select number of jobs to process
    job_dict_selection = select_first_n_jobs(job_dict, n=args.job_limit)
    #print(f'Collected [{total_job_folders}] zipped jobs from [{len(batch_dirs)}] batch directories.')
    print(f'Selected [{len(job_dict_selection)}] out of [{len(job_dict)}] zipped jobs from [{len(batch_dirs)}] batch directories.')
    #for key, value in job_dict.items(): print(key, value)

    ## Select number of cores for multiprocessing
    num_cores_total = multiprocessing.cpu_count()
    num_processes = args.num_processes if args.num_processes else int(num_cores_total / 2)
    print(f'Processing [{len(job_dict_selection)}] job(s) from [{len(batch_dirs)}] batch dir(s) with [{num_processes}] out of [{num_cores_total}] cores.')
    #sys.exit(0)

    ## Set Matplotlib plot font preferences
    set_plot_font(font_name=args.plot_font, svg_fonttype=args.svg_fonttype) # Do this in child process


    ### Processing

    ## Dynamically select and import the function for batch processing
    funcname = args.processing_function  # e.g., "do_process_job_collect_uniprot"
    job_module = importlib.import_module("batch_job_functions")
    job_processing_func = getattr(job_module, funcname)
    print(f'Using {job_processing_func}.\n')

    ## Process in parallel
    result_counts, apo_summaries, holo_summaries, query_info_summaries, metric_summaries = process_jobs(job_dict_selection, num_processes, args, job_processing_func)
    print('\nProcessing complete')


    ## Print report
    print_aggregate_results(result_counts)
    _labels = label_totals(query_info_summaries) or {}
    if _labels.get("n_unknown_label"):
        print(f'[WARN] {_labels["n_unknown_label"]} candidate row(s) in {_labels["n_jobs_with_unknown_label"]} '
              f'summarised job(s) carry an apoholo_assignment parse-ahoj does not know '
              f'({_labels["unknown_labels"]}); they joined neither the apo nor the holo set (AHOJ-020)')

    ## Save summary files (report, parameters, etc.)
    dir_timestamp = f"run_{time.strftime('%d%m%Y_%H%M%S', time.gmtime())}"
    summary_subdir_timestamp = os.path.join(args.summary_dir, f"{dir_timestamp}_{funcname}")
    os.makedirs(summary_subdir_timestamp, exist_ok=True)
    save_job_categories(result_counts, summary_subdir_timestamp)

    # Save script parameters/arguments
    save_arguments(args, summary_subdir_timestamp)
    print(f'Summary files saved in: {summary_subdir_timestamp}')

    # AHOJ-015: write a summary CSV and zip it SEPARATELY (its own .zip), keeping the original. Each
    # output file is compressed independently so the big tables ship as standalone archives.
    def _save_and_zip(df, filename, label):
        path = os.path.join(summary_subdir_timestamp, filename)
        df.to_csv(path, index=False)
        z = zip_file(path, delete_original=False)
        print(f'{label} saved in: {path}' + (f'  (+ {os.path.basename(z)})' if z else ''))
        return path

    # Save query/classification summaries (lean when the metric-stats block is detached, see
    # --separate_metric_summary).
    query_summaries_dir = _save_and_zip(query_info_summaries, f'query_summaries_{funcname}.csv', 'Query summary')

    # AHOJ-015: detached metric-stats summary. Joins to the classification file on job_name/query_path.
    metrics_summaries_dir = None
    if metric_summaries is not None and not metric_summaries.empty:
        metrics_summaries_dir = _save_and_zip(metric_summaries, f'query_metrics_{funcname}.csv', 'Metric-stats summary')

    # Save optional apo/holo summaries when a job function returns them.
    if not apo_summaries.empty:
        _save_and_zip(apo_summaries, 'apo_summaries.csv', 'Apo summary')
    if not holo_summaries.empty:
        _save_and_zip(holo_summaries, 'holo_summaries.csv', 'Holo summary')

    # Consolidated, machine-readable run manifest (provenance for the full-DB reruns).
    manifest_path = save_run_manifest(
        summary_subdir_timestamp, args, funcname, result_counts, query_info_summaries,
        n_batch_dirs=len(batch_dirs), n_jobs=len(job_dict_selection),
        query_summaries_path=query_summaries_dir, start_time=start_time,
        metric_summaries_df=metric_summaries, metrics_path=metrics_summaries_dir)
    print(f'Run manifest saved in: {manifest_path}')

    # Optionally zip every immediate saved file in this summary run directory.
    if int(args.zip_summary_files) == 1:
        zipped_files = zip_summary_output_files(
            summary_subdir_timestamp,
            delete_original=int(args.del_summary_files_after_zip) == 1,
        )
        print(f'Zipped [{len(zipped_files)}] summary file(s) in: {summary_subdir_timestamp}')

    ### Clean up & print time
    #plt.rcdefaults()  # Resets rcParams to their default settings
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print('All Done')



if __name__ == '__main__':
    main()
