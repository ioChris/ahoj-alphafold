# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:52:03 2025

@author: Christos P. Feidakis
"""

""" Data point classification spectrum """


import os
#os.environ["OMP_NUM_THREADS"] = "1"

import time
import pandas as pd
import numpy as np
import argparse
import sys
import zipfile, shutil
from urllib.request import urlopen, Request


from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions

from stats_api import compute_absolute_statistics_v2

from stats_core import (
    classify_point_modified_z, 
    classify_point_robust, classify_point_composite,
    overall_group_separation_robust, overall_group_separation,
    overall_group_overlap_robust, overall_group_overlap,
    calculate_group_separation_and_overlap, find_closest_candidates,
    point_score_percentage, dp_classify_new_point
    )

from stats_plot import (
    set_plot_font,
    visualize_metric_distributions_composite,
    plot_robust_distances, plot_normalized_distances,
    plot_composite_distances, 
    plot_overall_metrics,
    plot_dp_curves
    #plot_all_metrics_with_all_violin_overlap,
    #visualize_metric_distributions,
    )

from kde_api import classify_point_kde_r3
from kde_plot import VIOLIN_COLOR_SCHEMES, plot_kde_curves, plot_kde_violins, confidence_bars_kde


# Old imports
#from overlap_coefficient_classification import dp_classify_new_point, plot_overall_metrics, plot_density_estimation_per_metric
#from dispersion_and_coverage_functions import compute_KS_uniformity, compute_entropy, compute_nnd_stats # compute_CUS, print_uniformity_metrics 
#from kde_bayesian_point_classification import classify_point_kde_r3, plot_kde_curves, confidence_bars_kde, plot_kde_violins


################################### Script settings / arguments ###################################

def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: 1/0, true/false, yes/no, on/off")


def _parse_gridline_intervals(value):
    if value is None:
        return None

    value = str(value).strip()
    if value.lower() in {"", "none", "off", "false", "0"}:
        return None
    if value.lower() == "auto":
        return "auto"

    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use 'auto', 'none', an integer, or comma-separated integers, e.g. 5,10"
        ) from exc


def _normalize_save_formats(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]

    formats = []
    for item in value:
        for part in str(item).replace(",", " ").split():
            fmt = part.strip().lower().lstrip(".")
            if fmt in {"none", "off", "false", "0"}:
                return []
            if fmt:
                formats.append(fmt)
    return formats


def _coalesce_arg(value, fallback):
    return fallback if value is None else value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process zipped AHoJ-DB raw data.")

    # Test entry
    parser.add_argument('--ahoj_query_str',         type=str,   default='2yiw-A-I46-1354_db_poster_pps26', help='AHoJ entry/job directory to read under the local results directory')

    # Entry filtering
    parser.add_argument('--resolution_soft',        type=float, default=None,   help='None or float. Filter by resolution and also discard entries w/o resolution')
    parser.add_argument('--bndgrsds_prcnt_seqres',  type=int,   default=75,     help='None or int. Threshold for all %%mapped_sqr (hard filter). Condition for keeping is >= threshold')
    parser.add_argument('--bndgrsds_prcnt_obs',     type=int,   default=75,     help='None or int. Threshold for all %%mapped_obs (hard filter). Condition for keeping is >= threshold')

    parser.add_argument('--full_pockets_aligned',   type=int,   default=0,      help='Reject candidate pockets with unobserved residues (also rejects results with poorly mapped pockets)')
    parser.add_argument('--pocket_len_resis3',      type=int,   default=0,      help='Reject candidate pockets with mismatch between number of observed and aligned residues, i.e., pocket_len = #resis3 (to avoid metrics from poorly mapped pockets')
    parser.add_argument('--min_apo',                type=int,   default=1,      help='Minimum number of apo rows required in dataframe to process entry')
    parser.add_argument('--min_holo',               type=int,   default=1,      help='Minimum number of holo rows required in dataframe to process entry')
    parser.add_argument('--columns_nozero',         nargs='+',  default=['RoG', 'bfac_t', 'bfac_avg', 'resolution'], help='Columns that should not have zero values, replace with nan (0: np.nan, "": np.nan)')

    parser.add_argument('--relabel_apoholo',        type=int,   default=1,      help='Change the apoholo_assignment in global_results for holo candidates that only bind a single atom ligand (assign A**)')
    parser.add_argument('--add_query_row',          type=int,   default=1,      help='Choose whether to include query as a candidate')

    # States filtering (apo - holo results within entry)
    parser.add_argument('--filter_candidates',      type=int,   default=1,      help='Choose whether to filter alternative states of an entry (global_results.csv) or not')
    parser.add_argument('--rog_diff_thrs_query',    type=float, default=0.6,    help='float or None, e.g., 0.3: candidates whose RoG differs more than 0.3*query_RoG (+/-) from the (mean) query RoG are filtered out (lower=stricter) (default=0.5)')
    parser.add_argument('--rog_diff_thrs_mean',     type=float, default=0.6,    help='float or None, e.g., 0.3: candidates whose RoG differs more than 0.3*mean_RoG (+/-) from the average candidate RoG are filtered out (lower=stricter) (default=0.5)')
    parser.add_argument('--pocket_dist',            type=float, default=10.0,   help='Filter out pockets that are farther than X after global chain(s) alignment (default=5.0)')

    # KDE
    parser.add_argument('--kde_extend',             type=_str_to_bool, default=True, help='Extend KDE grid by tail chasing to avoid boundary clipping')
    parser.add_argument('--margin_cap_scale',       type=float, default=1.0,    help='Scales max tail extension cap; 1.0 preserves previous defaults')

    # Processing options
    parser.add_argument('--batch_limit',            type=int,   default=0,       help='0: no limit, integer: number of batches. Number of batch directories to process')
    parser.add_argument('--batch_selection',        type=list,  default=['103'], help='Select which batch directory/ies to process')
    parser.add_argument('--job_limit',              type=int,   default=8,    help='Select the first n entries/jobs to process out of all collected. e.g., default=20')
    parser.add_argument('--num_processes',          type=int,   default=None,    help='Number of processes to use for multiprocessing, i.e., None (=50%% cores) / 12 / 28.')
    #parser.add_argument('--apo_holo_sums',          type=int,   default=0,       help='Calculate numerical summaries for apo and holo data frames (slow)')
    #parser.add_argument('--union',                 type=int,   default=0,      help='0/1/. 1=Get database statistics (apoholo_averages) for pockets from entries with at least one apo and one holo pocket.')

    # Directory options
    parser.add_argument('--parsing_dir_suffix',     type=str,   default='_v3c5b', help='Suffix for parsing results directory name, that represents script version')
    parser.add_argument('--db_version',             type=str,   default='v2b',    help='AHoJ-DB version being parsed')
    parser.add_argument('--local_base_dir',         type=str,   default=None,     help='Input directory with unzipped tar files')
    parser.add_argument('--save_base_dir',          type=str,   default=None,     help='Base directory to save parsing results')
    parser.add_argument('--save_meta_dir',          type=str,   default=None,     help='Base directory to save entry meta-processing results (multiple sub-directories')

    parser.add_argument('--plot_font',              type=str,   default='Helvetica Neue',    help='Set the font to be used for plot titles/legends')
    parser.add_argument('--svg_fonttype',           type=str,   default='none',     help='["none", "path"]: Choose how to save svg font - none is default, path will save glyphs as svg (larger file size)')
    #parser.add_argument('--font_file',              type=str,   default='HelveticaNeue-Roman.otf',         help='')

    # Visualizations settings
    parser.add_argument('--separate_metrics',       type=int,   default=1,      help='Plot each metric in a separate figure')
    parser.add_argument('--show_plots',             type=int,   default=1,      help='Display KDE violin and confidence bar figures after saving')
    parser.add_argument('--plot_save_formats',      nargs='+',  default=['svg'], help='Plot file formats to save, e.g. svg or svg png')
    parser.add_argument('--plot_transparency', '--transparent', type=int, default=1, help='Fallback transparency for supported plot families; keeps axes white and makes the outer figure transparent')

    # KDE violin plot settings
    parser.add_argument('--kde_violin_transparency', type=int,   default=None,   help='Override transparency for KDE violin plots')
    parser.add_argument('--kde_violin_color_scheme', type=str,   default='seaborn1', choices=sorted(VIOLIN_COLOR_SCHEMES), help='Color scheme for KDE violin plots')
    parser.add_argument('--kde_violin_gridlines',    type=_parse_gridline_intervals, default=None, help="KDE violin gridlines: 'auto', 'none', an integer, or comma-separated integers")
    parser.add_argument('--kde_violin_margin_ratio', type=float, default=0.04,   help='Relative x-axis margin for KDE violin plots')
    parser.add_argument('--kde_violin_width',        type=float, default=0.80,   help='Violin width for KDE violin plots')
    parser.add_argument('--kde_violin_boxplots',     type=int,   default=1,      help='Show embedded KDE violin boxplots')
    parser.add_argument('--kde_violin_show_legend',  type=int,   default=1,      help='Show KDE violin legends')
    parser.add_argument('--kde_violin_set_title',    type=int,   default=0,      help='Show KDE violin per-metric titles')

    # Confidence bar plot settings
    parser.add_argument('--confidence_bars_transparency', type=int, default=None, help='Override transparency for KDE confidence bar plots')

    return parser.parse_args()


# Global variables
# Dictionary of metric labels
metrics_names_and_units_dict = {
    "pocket_rms": "Pocket all-atom RMSD (Å)",
    "volume": "Volume (Å³)",
    "sasa": "Solvent Accessible Surface Area (Å²)",
    "sasa_avg": "Solvent Accessible Surface Area per-residue avg. (Å²)",
    "asa": "Molecular Surface Area (Å²)",
    "asa_avg": "Molecular Surface Area per-residue avg. (Å²)",
    "sasa_asa_fraction": "Ratio of SASA to ASA",
    "RoG": "Radius of Gyration (Å)",
    "pocket_dist": "Pocket Center Distance After Chain Superposition (Å)",
    "bfac_t": "B-factor (Å²) & model pLDDT",
    "bfac_avg": "B-factor per-atom avg. (Å²) & model pLDDT per-atom avg.",
    "tm_score": "Chain TM-score",
    "tm_score_i": "Chain TM-score inverse",
    "rmsd": "Chain Cα RMSD (Å)",
    "#solvent": "Pocket Water Molecules (#)"
}

metrics_names_dict = {
    "pocket_rms": "Pocket all-atom RMSD",
    "volume": "Volume",
    "sasa": "Solvent Accessible Surface Area",
    "sasa_avg": "Solvent Accessible Surface Area per-residue avg.",
    "asa": "Molecular Surface Area",
    "asa_avg": "Molecular Surface Area per-residue avg.",
    "sasa_asa_fraction": "Ratio of SASA to ASA",
    "RoG": "Radius of Gyration",
    "pocket_dist": "Pocket Center Distance After Chain Superposition",
    "bfac_t": "B-factor",
    "bfac_avg": "B-factor per-atom avg.",
    "tm_score": "Chain TM-score",
    "tm_score_i": "Chain TM-score inverse",
    "rmsd": "Chain Cα RMSD",
    "#solvent": "Pocket Water Molecules"
}

metrics_units_dict = {
    "pocket_rms": "Å",
    "volume": "Å³",
    "sasa": "Å²",
    "sasa_avg": "Å²",
    "asa": "Å²",
    "asa_avg": "Å²",
    "sasa_asa_fraction": "",
    "RoG": "Å",
    "pocket_dist": "Å",
    "bfac_t": "Å²",
    "bfac_avg": "Å²",
    "tm_score": "0-1",
    "tm_score_i": "0-1",
    "rmsd": "Å",
    "#solvent": "#"
}


################################### Functions ###################################

def get_2level_cwd():
    npath = os.path.normpath(os.getcwd())   # Normalize the path string for the OS
    #path0 = os.path.join(npath.split(os.sep)[0], '/', npath.split(os.sep)[1], npath.split(os.sep)[2])
    path0 = os.path.join(npath.split(os.sep)[0], os.sep, npath.split(os.sep)[1], npath.split(os.sep)[2])
    if os.path.exists(path0):
        print(f'Root path found >> {path0}')
    else:
        print(f'Error finding root path in working dir: {npath}')
    return path0

def normalize_query_df(query_df, query_metrics_zero, query_metrics_one):
    """
    Parameters
    ----------
    query_df : pandas.DataFrame
        Input data frame to normalize.
    query_metrics_zero : list of str
        Column names of metrics to set to zero.
    query_metrics_one : list of str
        Column names of metrics to set to one.

    Returns
    -------
    df_q_norm : pandas.DataFrame
        Dataframe with selected columns converted to zero and one respectively.

    """
    df_q_norm = query_df.copy()
    # Replace empty strings with NaN for relevant columns
    df_q_norm.loc[:, query_metrics_zero + query_metrics_one] = df_q_norm[query_metrics_zero + query_metrics_one].replace('', np.nan)
    # Convert columns to float before applying fillna
    df_q_norm.loc[:, query_metrics_zero + query_metrics_one] = df_q_norm[query_metrics_zero + query_metrics_one].astype(float)
    # Apply different fillna values
    df_q_norm.loc[:, query_metrics_zero] = df_q_norm[query_metrics_zero].fillna(0.0)
    df_q_norm.loc[:, query_metrics_one] = df_q_norm[query_metrics_one].fillna(1.0)

    return df_q_norm

def _json_safe_scalar(v):
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if v is None:
        return None
    return v

def _jsonify(obj):
    # recursively convert dict/list/np types → built-in Python types
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if obj is None or (isinstance(obj, float) and (obj != obj)):  # NaN
        return None
    return _json_safe_scalar(obj)

def save_dict_to_json(data_dict, directory, filename):
    """
    Save a dictionary as a JSON file.

    Parameters:
    - data_dict: The dictionary to save.
    - directory: The directory path where the file should be saved.
    - filename: The output file name (should include .json extension).

    Returns:
    - filepath: The full path to the saved JSON file.
    """
    import json
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full file path
    filepath = os.path.join(directory, filename)

    # Write the dictionary to the JSON file with indentation for readability
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(data_dict), f, indent=4, ensure_ascii=False)
    #with open(filepath, 'w') as f:
    #    json.dump(data_dict, f, indent=4)

    #print(f"Dictionary saved to {filepath}")
    return filepath

def add_fraction_metric(df, numerator_col='sasa', denominator_col='asa', new_col_name='sasa_asa_fraction', decimals=3):
    """
    Adds a new column to the dataframe representing the fraction of two specified columns.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        numerator_col (str): The column name to be used as numerator.
        denominator_col (str): The column name to be used as denominator.
        new_col_name (str): The name of the new column to add.
        decimals (int): Number of decimal places to round to.

    Returns:
        pd.DataFrame: DataFrame with the new fraction column added.
    """
    df = df.copy()
    df[new_col_name] = (df[numerator_col] / df[denominator_col]).round(decimals)
    return df

def compress_directory(dir_path: str, delete_original: bool = False, level: int = 3) -> str:
    """
    Compresses the *contents* of the given directory into a single
    ``<folder>.tar.zst`` archive placed alongside the original folder
    (contents at the archive root, no intermediate folder nesting).

    Uses the ``zstd`` CLI (the run environment ships it, see RUN_full_batch.md):
    the directory is tar'd in-process and streamed to ``zstd -<level>`` so no
    intermediate raw tar hits disk. On the near-identical per-entry SVG/JSON
    corpus, zstd -3 compresses far better than the previous plain-DEFLATE zip
    while staying fast enough to run inside every worker without stalling the
    plot-bound pipeline.

    Args:
        dir_path: Path to the directory whose contents you want to archive.
        delete_original: If True, delete the original directory tree afterward.
        level: zstd compression level (default 3).

    Returns:
        Path to the created ``.tar.zst`` archive.

    Raises:
        ValueError: If dir_path is not a valid directory.
        RuntimeError: If the zstd CLI exits non-zero.
    """
    import subprocess, tarfile

    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise ValueError(f"Not a directory: {dir_path!r}")

    parent_dir, folder_name = os.path.split(dir_path)
    out_path = os.path.join(parent_dir, f"{folder_name}.tar.zst")

    # Stream: tar (in-process) -> zstd stdin -> <folder>.tar.zst.
    # -q quiet, -f overwrite (idempotent re-runs), single-threaded (workers own their core).
    proc = subprocess.Popen(
        ["zstd", f"-{level}", "-q", "-f", "-o", out_path, "-"],
        stdin=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    # Archive names relative to dir_path (contents at root).
                    arcname = os.path.relpath(abs_path, dir_path)
                    tar.add(abs_path, arcname=arcname)
    finally:
        if proc.stdin:
            proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"zstd exited with code {rc} while writing {out_path!r}")

    if delete_original:
        shutil.rmtree(dir_path)

    return out_path

def compress_file(file_path: str, delete_original: bool = False) -> str:
    """
    Compress a single file into a ZIP archive placed alongside it.

    Args:
        file_path: Path to the file you want to zip.
        delete_original: If True, delete the file after zipping.

    Returns:
        Path to the created .zip file.
    """
    import zipfile, os

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise ValueError(f"Not a file: {file_path!r}")

    parent_dir, filename = os.path.split(file_path)
    zip_filename = f"{filename}.zip"
    zip_path = os.path.join(parent_dir, zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, arcname=filename)

    if delete_original:
        os.remove(file_path)

    return zip_path

def write_script_parameters(args, datetime, script_version, python_version, 
                            path_results, output_filename = 'pocket_analysis_parameters.json'):
    import json

    # Create a dictionary with additional fields
    additional_info = {
        "date": datetime,
        #"date": str(datetime.datetime.now()),
        "script_version": script_version,
        #"pymol_version": pml_version,
        "python_version": python_version
    }

    # Combine argparse arguments and additional fields
    data = {**vars(args), **additional_info}

    # Save the data to a JSON file
    #output_filename = 'pocket_analysis_parameters.json'
    output_file = os.path.join(path_results, output_filename)
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Parameters saved to {output_file}")

def ensure_ahoj_entry(entry_id: str,
                      results_base_dir: str,
                      *,
                      keep_zip: bool = False,
                      overwrite_dir: bool = False,
                      timeout: int = 120) -> str:
    """
    Ensure a given AHoJ entry directory exists locally; if not, download and unzip it.

    Parameters
    ----------
    entry_id : str
        e.g., "3cqv-A-HEM-601".
    results_base_dir : str
        Directory where entry folders live, e.g. os.path.join(workdir, middle_dir).
    keep_zip : bool
        If True, keep the downloaded .zip alongside the extracted folder.
    overwrite_dir : bool
        If True, re-download and re-extract even if the directory already exists.
    timeout : int
        Download timeout (seconds).

    Returns
    -------
    csv_path : str
        Full path to "<results_base_dir>/<entry_id>/global_results.csv".

    Raises
    ------
    RuntimeError on download / extraction / missing CSV issues.
    """
    assert entry_id and isinstance(entry_id, str)
    os.makedirs(results_base_dir, exist_ok=True)

    entry_dir = os.path.join(results_base_dir, entry_id)
    csv_path = os.path.join(entry_dir, "global_results.csv")

    # If already available and not forcing overwrite, return it
    if os.path.isdir(entry_dir) and os.path.isfile(csv_path) and not overwrite_dir:
        return csv_path

    # If overwriting, remove existing folder first
    if overwrite_dir and os.path.isdir(entry_dir):
        shutil.rmtree(entry_dir)

    # Build URL -> https://dev.apoholo.cz/api/db/entry/<ID>/download/<ID>.zip
    zip_url = f"https://apoholo.cz/api/db/entry/{entry_id}/download/{entry_id}.zip"
    #zip_url = f"https://dev.apoholo.cz/api/db/entry/{entry_id}/download/{entry_id}.zip"
    zip_path = os.path.join(results_base_dir, f"{entry_id}.zip")

    # Download (use urllib to avoid extra deps; set a UA to be nice)
    try:
        req = Request(zip_url, headers={"User-Agent": "AHoJ-fetch/1.0"})
        with urlopen(req, timeout=timeout) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as e:
        raise RuntimeError(f"Failed to download {zip_url}: {e}") from e

    # Verify and extract safely
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"Corrupt ZIP member: {bad}")

            # Safe extraction: prevent path traversal
            def _safe_join(base, *paths):
                final_path = os.path.abspath(os.path.join(base, *paths))
                if not final_path.startswith(os.path.abspath(base) + os.sep):
                    raise RuntimeError(f"Unsafe path detected in ZIP: {final_path}")
                return final_path

            os.makedirs(entry_dir, exist_ok=True)
            for member in zf.infolist():
                # Directories
                if member.is_dir():
                    target_path = _safe_join(entry_dir, member.filename)
                    os.makedirs(target_path, exist_ok=True)
                    continue
                # Files
                target_path = _safe_join(entry_dir, member.filename)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception as e:
        # Cleanup partially extracted dir on failure
        if os.path.isdir(entry_dir):
            shutil.rmtree(entry_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract ZIP for {entry_id}: {e}") from e
    finally:
        if not keep_zip and os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

    # Validate CSV presence
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Expected CSV not found after extraction: {csv_path}")

    return csv_path
### Functions end #############################################################



def main():

    args = parse_arguments()

    ### Input

    ################################# AHoJ test query #################################
    ahoj_query_str = args.ahoj_query_str # '1gx8-A-RTL-1163' # '2gke-A-ZDP-700'
    ###################################################################################
    # AHoJ queries:
    # 1gx8-A-RTL-1163, ambiguous classification
    # 1hbo-A-F43-1550, 1a6g-A-HEM-154 (no apo)
    # 3cqv-A-HEM-601, 1bz0-B-HEM-147, 1bab-B-HEM-147 (same with 1bz0), 1il3-A-7DG-301, 1rnc-A-5GP-161, 1x0x-A-SO4-1002
    # 4q0a-C-4OA-502 (error with KDE), 1cza-N-ADP-922, 3f8c-A-HT1-127 (poor volume and rog classification)
    # 8a3h-A-IDC-500 (), 2gke-A-ZDP-700 (low holo sasa)
    # 3o0o-A-B12-1004 (B12)

    # CBS
    # 1b14-A-NAD-255, 3rvr-A-MOO-131 (Ecoli, 27/71, good), 1ecg-A-ONL-505, 1ecg-A-PIN-506 (ratio), 3f3y-A-4OA-326 (good sep, 2/3)
    # 5jiw-B-GLC-2_, 1n1e-A-NDE-400 (1 apo), 1fp6-A-ADP-5291 (interface)

    # Flexible
    # 6uo1-1A-MG-3167, 5tle-A-RD1-401 (good but not human)
    # 1pzo-A-CBT-300 (low sep), 1axb-A-FOS-291 (low sep)

    # Saldano et al
    # 1lah-E-ORN-240 (holo), 1ex7-A-5GP-187 (apo), 1pun-A-8OG-131 (none)
    # 4pj1-A-ADP-601 (clear holo, good sep), 2vcd-A-RAP-138, 6oy9-R-RET-401
    # 1ssp-E-URA-55, 1aw1-B-PGA-290 (pholo:1), 1cbr-B-REA-200 (1 apo), 1e5q-A-NDP-500, 1eio-A-GCH-128 (1 apo)

    # Homo sapiens
    # 2src-A-ANP-1, 5h8t-A-RTL-201 (retinol), XEKND_01_2npq-a-bog-1000 (good example), 2yiw-A-I46-1354 (good example + AHoJ-DB), 1pin-A-PRO-202, 3ikd-A-J9Z-501 (SBDD)
    # 3wh0-A-O4B-202 (crown ether), 3oob-A-KDH-402
    # 1dfp-B-DFP-501, 5tca-A-J55-500_ (interface), 1bm7-A-FLF-501_(interface many), 1dvs-A-STL-125_ (same), 1e4h-A-PBR-998 (same but monomer, low separation)
    # 4rmg-A-3TE-402 (good ambiguous), 5d7p-B-OCZ-403 (same prot, low apo), 3cqp-C-CU-154, 3ipb-A-JZD-128_
    # 6ypn-B-ADP-403, 8fkd-A-Y5O-501, 3v9v-A-21L-501_ (peroxisome)

    # Large queries
    # 5sfg-D-ZN-801 (broken entry, no model data), 5rsl-A-W51-201 (ok query, no model)
    # 6v2k-E-MN-201 (heterodimer MN), 

    # No apo
    # 3sce-B-HEC-1003
    # Multichain
    # 1aei-D-CA-317

    # Errors
    # 1aby-A-HEM-286, 117e-A-PO4-3002, 3k88-B-NAD-501, 4heu-B-15J-809
    # 7jgn-I-ZN-201 (ok), 5jjf-A-RET-301 (fixed), 4csd-A-MFU-301 (fixed), 7jgl-x-ZN-201
    # 3j6q-A-SAH-1101 (no af row), 5rm3-B-ZN-702 ()
    
    # Tests
    # 3bz8-A-BL6-800, 6okc-A-MRJ-501 (interface pocket)
    
    # Undefined classification
    # 3cxc-2-MG-8078
    
    # Density over 1 (pocket_avg_normalized_density_strongest)
    # 3cyi-A-ATP-1, 1d2u-A-HEM-185
    
    #
    # 6k9v-B-D3L-504 (Tubulin indole derivative at colchicine pocket)
    
    # Ions causing holo classification
    # 1evl-A-TSB-2002, 12ca-A-ZN-262 (test)
    # 3bzu-D-NAP-4
    
    # Tail smoothing same between KDE curves and KDE violins
    # 3bz7-A-MG-997

    # Local jobs
    # job_196 (3cqv A hem concave),   job_201 (convex)
    # job_197 (2yiw A I46 1354, p38), job_203 (convex)

    workdir = get_2level_cwd()
    # Folder holding the AHoJ result folders (<dir>/<entry>/global_results.csv); a missing entry is
    # downloaded there from the AHoJ-DB API. An absolute path, so os.path.join(workdir, ...) keeps it.
    middle_dir = os.path.abspath(os.environ.get("AHOJ_RESULTS_DIR", "ahoj_results"))
    results_base_dir = os.path.join(workdir, middle_dir)

    file_path = os.path.join(workdir, middle_dir, ahoj_query_str, 'global_results.csv')

    ### Output
    output_directory_plots = os.path.join(workdir, middle_dir, ahoj_query_str, 'plots_poster')
    output_directory_stats = os.path.join(workdir, middle_dir, ahoj_query_str, 'statistics')
    os.makedirs(output_directory_plots, exist_ok=True)
    os.makedirs(output_directory_stats, exist_ok=True)

    print(f"\nInput file:\n{file_path}")
    print(f"Output directory:\n{output_directory_plots}")
    print(f"AHoJ query: {ahoj_query_str}")


    ### Arguments
    # Specify which metrics to plot
    #pocket_metrics, chain_metrics = [],[]
    #pocket_metrics = ["pocket_rms", "sasa", "sasa_avg", "asa", "asa_avg", "volume", "RoG", "pocket_dist", "bfac_t", "bfac_avg", "#solvent"]
    #chain_metrics = ["tm_score", "tm_score_i", "rmsd"]
    #total_metrics_to_plot = pocket_metrics + chain_metrics

    # Define pocket & chain metrics used for the point classification

    # Pocket
    pocket_classification_metrics = ["pocket_rms", "sasa", "volume", "RoG"]
    #pocket_classification_metrics = ["pocket_rms", "sasa", "sasa_asa_fraction", "volume", "RoG"]
    #pocket_classification_metrics = ["pocket_rms", "sasa", "asa", "volume", "RoG"]
    #pocket_classification_metrics = ["pocket_rms", "sasa", "asa", "sasa_asa_fraction", "volume", "RoG"]
    #pocket_classification_metrics = ["pocket_rms", "sasa", "volume", "RoG", "pocket_dist"]
    #pocket_classification_metrics = ["sasa", "volume", "RoG", "pocket_dist"]#, "bfac_t", "bfac_avg"]
    #pocket_classification_metrics = ["sasa", "sasa_avg", "volume", "RoG", "pocket_dist"]#, "bfac_t", "bfac_avg"]
    #pocket_classification_metrics = ["pocket_rms", "sasa", "sasa_avg", "volume", "RoG", "pocket_dist"]
    #metrics_to_plotA = ["pocket_rms", "sasa", "volume", "RoG"] # , "asa"
    #metrics_to_plotA = ["pocket_rms", "sasa", "asa", "volume", "RoG"] # 

    # Chain
    chain_classification_metrics = ["pocket_dist", "tm_score", "rmsd"] 
    #chain_classification_metrics = ["pocket_dist", "tm_score", "rmsd"] # "bfac_t", "#solvent"
    #metrics_to_plotB = ["pocket_dist", "tm_score", "rmsd"] # "bfac_t", "#solvent"

    # Pocket & chain metrics
    

    # Set plot font & style
    set_plot_font(font_name=args.plot_font, svg_fonttype=args.svg_fonttype)
    plot_save_formats = _normalize_save_formats(args.plot_save_formats)
    kde_violin_transparency = _coalesce_arg(args.kde_violin_transparency, args.plot_transparency)
    confidence_bars_transparency = _coalesce_arg(args.confidence_bars_transparency, args.plot_transparency)
    #sys.exit(1)

    # Set the weights for the center, middle and extremes (mean & median, IQR and min/max)
    w_center, w_iqr, w_extremes = 0.5, 0.3, 0.2 # 0.4, 0.3, 0.3
    #weights_dict = {'center': w_center, 'iqr': w_iqr, 'extremes': w_extremes}
    #w_center, w_iqr, w_extremes = 1.0, 0.0, 0.0 # robust stats default weight



    ################################ Data loading & filtering ################################

    ### Load the global results CSV (AHoJ 2.6.0 names renamed to parse-ahoj's: run_types.load_global_results)
    import run_types as rt
    df = rt.load_global_results(os.path.dirname(file_path)) if os.path.isfile(file_path) else None
    if df is None:
        print(f'\nFile not found: {file_path}')
        print('Attempting to fetch entry package from server...')
        csv_path = ensure_ahoj_entry(ahoj_query_str, results_base_dir)
        df = rt.load_global_results(os.path.dirname(csv_path))
        #print(f'\nException when loading file: {ex}')
        #sys.exit(1)
    ### Curate & prepare data

    # Check if df is empty
    if df.empty or (df.replace('', np.nan).isnull().all(axis=None)):
        print('Data frame is empty, exiting')
        sys.exit(0)
        #return ('Discarded', (job_name, f'Empty results dataframe: {len(df)}' ))


    # Relabel apo/holo classification in global results according to exclusion list (ions)
    if args.relabel_apoholo == 1:
        df, changed = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=os.path.join(workdir, middle_dir, ahoj_query_str)) # 
        #print(f"\nChanged rows:\n{changed}")

    # Set NaN pocket volume to zero (or 1)
    #print('\nConverting zero or nan volume to 0.0 or 1.0')
    #print(df['volume'].to_string(header=False, index=False))
    s = pd.to_numeric(df['#resis3'], errors='coerce')          # helper: NaN if empty/non-numeric
    v = pd.to_numeric(df['volume'],  errors='coerce')          # convert '' -> NaN, keep numbers
    df['volume'] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v)

    #df['volume'] = df['volume'].replace('', np.nan).astype(float).fillna(0.0) # previous standard
    #print(df['volume'].to_string(header=False, index=False))

    # Add new column with composite metric (ratio of SASA to ASA)
    df = add_fraction_metric(df)

    # Split global results data frame
    df_q = df[df['role'] == 'Q']
    df_af = df[df['role'] == 'M']

    # Test print
    #print(df.to_string(index=False))

    # Filter candidates
    if args.filter_candidates == 1:
        df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=args.full_pockets_aligned, bndgrsds_prcnt_seqres=args.bndgrsds_prcnt_seqres, 
                                bndgrsds_prcnt_obs=args.bndgrsds_prcnt_obs, resolution_soft=args.resolution_soft, rog_diff_threshold_deltaQ=args.rog_diff_thrs_query, 
                                rog_diff_threshold_mean=args.rog_diff_thrs_mean, pocket_distance=args.pocket_dist, pocket_len_resis3=args.pocket_len_resis3)
    else:
        df_f = df.copy()

    # Extract candidates
    df_c = df_f[df_f['role'] == 'C']

    #print(df_c.to_string(index=False)) # Test print candidate dataframe

    # [OPTIONAL] Add query row to Candidates (allows plotting of computed query metrics)
    df_c = pd.concat([df_q, df_c], ignore_index=True)

    # Save the dataframe that will be used for plots
    #df_f.to_csv(os.path.join(output_directory_stats, 'global_results_plots_df_f.csv'), index=False)
    #df_c.to_csv(os.path.join(output_directory_stats, 'global_results_plots_df_c.csv'), index=False)

    # Set superposition-related metrics/deltas to 0.0 or 1.0 in Query DF (for plots)
    query_metrics_zero = ["pocket_rms", "pocket_dist", "rmsd"]
    query_metrics_one = ["tm_score", "tm_score_i"]
    df_q_norm = normalize_query_df(df_q, query_metrics_zero, query_metrics_one)

    # Extract apo and holo data
    apo_df = df_c[df_c['apoholo_assignment'].isin(['A', 'A*', 'A**'])]
    holo_df = df_c[df_c['apoholo_assignment'].isin(['H', 'H*'])]
    #print(apo_df.to_string(header=False, index=False))
    #print(holo_df.to_string(header=False, index=False))
    #sys.exit(0)
    
    ### Done curating data



    ############################### Data processing ###############################

    start_time = time.time()


    ### General: Apo & Holo data distributions of selected metrics with basic seaborn violins

    # Pocket violin
    #plot_all_metrics_with_all_violin_overlap(df_c, pocket_classification_metrics, output_directory_plots, 
    #    model_df=df_af, query_df=df_q_norm, save_formats=['svg'], show_plot=1, filename_suffix="_pocket", transparency=0, gridline_intervals=None, metrics_names_and_units_dict=metrics_names_and_units_dict)

    # Chain violin
    #plot_all_metrics_with_all_violin_overlap(df_c, chain_classification_metrics, output_directory_plots, 
    #    model_df=df_af, query_df=df_q_norm, save_formats=['svg'], show_plot=1, filename_suffix="_chain", transparency=0, gridline_intervals=None, metrics_names_and_units_dict=metrics_names_and_units_dict)


    ### Get closest candidate to model
    all_metrics = list(dict.fromkeys(pocket_classification_metrics + chain_classification_metrics))

    # Run separately to scope "overall" to each set:
    closest_pocket = find_closest_candidates(df_c[df_c["role"]=="C"], df_af, pocket_classification_metrics, top_n=5, scale="range", decimals=5)
    closest_chain  = find_closest_candidates(df_c[df_c["role"]=="C"], df_af, chain_classification_metrics,  top_n=5, scale="range", decimals=5)
    closest_all    = find_closest_candidates(df_c[df_c["role"]=="C"], df_af, all_metrics,    top_n=5, scale="range", decimals=5)

    # Save
    closest_pocket_json = save_dict_to_json(closest_pocket, output_directory_stats, 'closest_pocket.json')
    closest_chain_json  = save_dict_to_json(closest_chain,  output_directory_stats, 'closest_chain.json')
    closest_all_json    = save_dict_to_json(closest_all,    output_directory_stats, 'closest_all.json')
    compress_file(closest_pocket_json, delete_original=True)
    compress_file(closest_chain_json, delete_original=True)
    compress_file(closest_all_json, delete_original=True)
    #sys.exit(0)


    ### Compute descriptive statistics for Apo, Holo & All groups for each metric (mean, median, min/max, ..) plus Apo/Holo comparisons
    #for counter in range(1, 100):

    #comparisons_pocket_v1 = compute_absolute_statistics(apo_df, holo_df, pocket_classification_metrics)
    #comparisons_chain_v1 = compute_absolute_statistics(apo_df, holo_df, chain_classification_metrics)

    comparisons_pocket = compute_absolute_statistics_v2(apo_df, holo_df, pocket_classification_metrics)
    comparisons_chain = compute_absolute_statistics_v2(apo_df, holo_df, chain_classification_metrics)

    # Print a report of the computed statistics
    #print("\n=== Statistics v1 for Apo, Holo, All and comparative per metric ===")
    #for metric, comp in comparisons_pocket_v1.items():
    #    print(f"Metric: {metric.upper()}\n{comp}")

    print("\n=== Statistics v2 for Apo, Holo, All and comparative per metric ===")
    for metric, comp in comparisons_pocket.items():
        print(f"Metric: {metric.upper()}\n{comp}")

    #sys.exit(0)
    ### Classify data point based on its distance from Apo and Holo groups (use computed statistics)
    # 5 approaches: Z-scores, Robust statistics, Composite weighted distance, Dirichlet process mixture model, KDE

    ## Z-scores
    classification_z_pocket = classify_point_modified_z(df_af, comparisons_pocket, pocket_classification_metrics)
    classification_z_chain = classify_point_modified_z(df_af, comparisons_chain, chain_classification_metrics)
    print(f"\nClassification result pocket (Z-scores):\n{classification_z_pocket}")
    print(f"\nClassification result chain (Z-scores):\n{classification_z_chain}")

    ## Robust statistics
    classification_robust_pocket = classify_point_robust(df_af, comparisons_pocket, pocket_classification_metrics)
    classification_robust_chain = classify_point_robust(df_af, comparisons_chain, chain_classification_metrics)
    print(f"\nClassification result pocket (Robust statistics):\n{classification_robust_pocket}")
    print(f"\nClassification result chain (Robust statistics):\n{classification_robust_chain}")

    ## Composite distance
    classification_composite_pocket = classify_point_composite(df_af, comparisons_pocket, pocket_classification_metrics, w_center=w_center, w_iqr=w_iqr, w_extremes=w_extremes)
    classification_composite_chain = classify_point_composite(df_af, comparisons_chain, chain_classification_metrics, w_center=w_center, w_iqr=w_iqr, w_extremes=w_extremes)
    print(f"\nClassification result pocket (Composite weighted distance):\n{classification_composite_pocket}")
    print(f"\nClassification result chain (Composite weighted distance):\n{classification_composite_chain}")
    #sys.exit(0)

    ## Dirichlet process mixture model
    classification_dirichlet_pocket = dp_classify_new_point(df_c, pocket_classification_metrics, df_af)
    classification_dirichlet_chain = dp_classify_new_point(df_c, chain_classification_metrics, df_af)
    print(f"\nClassification result pocket (Dirichlet process mixture model):\n{classification_dirichlet_pocket}")
    print(f"\nClassification result chain (Dirichlet process mixture model):\n{classification_dirichlet_chain}")
    #sys.exit(0)


    ## KDE Bayesian classification

    #start_time = time.time()
    #for i in range(1, 1000):

    # Pocket
    kde_results_r3_pocket, kde_data_r3_pocket, apo_dict_pocket, holo_dict_pocket, all_dict_pocket = classify_point_kde_r3(df_c, apo_df, holo_df,
        df_af, pocket_classification_metrics, kde_extend=args.kde_extend, margin_cap_scale=args.margin_cap_scale, bw_mode="pooled", bw_scale=0.75, prior_mode='equal') # <— NEW: "pooled" | "separate"

    # Chain
    kde_results_r3_chain, kde_data_r3_chain, apo_dict_chain, holo_dict_chain, all_dict_chain = classify_point_kde_r3(df_c, apo_df, holo_df,
        df_af, chain_classification_metrics, kde_extend=args.kde_extend, margin_cap_scale=args.margin_cap_scale, bw_mode="pooled", bw_scale=0.75, prior_mode='equal') # <— NEW: "pooled" | "separate"
    #sys.exit(0)

    # Compare classification outputs
    #kde_results, kde_data = classify_point_kde_r2(df_c, apo_df, holo_df, df_af, pocket_classification_metrics)
    #kde_results_chain, kde_data_chain = classify_point_kde_r2(df_c, apo_df, holo_df, df_af, chain_classification_metrics)
    #print('\nComparing classification results')
    #ok1 = compare_classification_dicts(kde_results, kde_results_r3_pocket)
    # Compare density curve outputs
    #ok2 = compare_kde_data_dicts(kde_data, kde_data_r3_pocket)
    #if ok1 and ok2:     print("✅ Outputs are equivalent.")
    #else:   print("❌ Outputs differ.")

    print(f"\nKDE Classification result - Pocket:\n{kde_results_r3_pocket}")
    print(f"\nKDE Classification result - Chain:\n{kde_results_r3_chain}")
    #print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
    #print(f"\nClassification result (KDE Bayesian classification):\n{kde_results}")


    ### Plot computed distances & probabilities for each method and metric (Classification histograms)
    print("\n-Plotting distances, probabilities, densities per metrics and overall")
    #sys.exit(0)


    ## Classification Histograms
    #plot_normalized_distances(classification_z_pocket, pocket_classification_metrics) # Z-scores
    #plot_robust_distances(classification_robust_pocket, pocket_classification_metrics) # Robust statistics
    #plot_composite_distances(classification_composite_pocket, pocket_classification_metrics, save_dir=output_directory_plots, save_formats=[], apo_label="Apo", holo_label="Holo", include_all=False) # Composite weighted
    #plot_overall_metrics(classification_dirichlet_pocket.get('overall')) # Dirichlet overall

    ## Box plots + data points (composite)
    #visualize_metric_distributions_composite(df_candidates=df_c, df_new=df_af, metrics=pocket_classification_metrics,group_col="apoholo_assignment",
    #                                           classification_composite_pocket=classification_composite_pocket,
    #                                           save_dir=output_directory_plots, save_formats=['svg'], apo_label="Apo", holo_label="Holo")


    ## Probability Density Functions

    # DP-GMM curves - Pocket
    plot_dp_curves(
        apo_group=apo_df, holo_group=holo_df,
        classification_metrics=pocket_classification_metrics,
        model_point=df_af,
        supertitle=0, show_plot=args.show_plots,
        save_dir=output_directory_plots, save_formats=['svg'], filename_suffix='_pocket',
        transparency=0,
        separate_metrics=args.separate_metrics,   # 0→one stacked panel, 1→per-metric figs
        color_scheme="blue_orange1",             # any name from KDE’s VIOLIN_COLOR_SCHEMES
        metrics_names_and_units_dict=metrics_names_and_units_dict,
        set_title=1, show_legend=True, show_points=True
    )

    # DP-GMM curves - Chain
    plot_dp_curves(
        apo_group=apo_df, holo_group=holo_df,
        classification_metrics=chain_classification_metrics,
        model_point=df_af,
        supertitle=0, show_plot=args.show_plots,
        save_dir=output_directory_plots, save_formats=['svg'], filename_suffix='_chain',
        transparency=0,
        separate_metrics=args.separate_metrics,   # 0→one stacked panel, 1→per-metric figs
        color_scheme="blue_orange1",             # any name from KDE’s VIOLIN_COLOR_SCHEMES
        metrics_names_and_units_dict=metrics_names_and_units_dict,
        set_title=1, show_legend=True, show_points=True
    )
    #sys.exit(0)

    # KDE curves - Pocket
    plot_kde_curves(kde_data_r3_pocket, apo_df, holo_df, pocket_classification_metrics, df_af, metrics_names_and_units_dict, show_all=1, # , df_c
        supertitle=0, show_plot=args.show_plots, save_dir=output_directory_plots, save_formats=['svg'], filename_suffix='_pocket', transparency=0,
        separate_metrics=args.separate_metrics, color_scheme="curves_legacy", set_title=1)
    #sys.exit(0)

    # KDE curves - Chain
    plot_kde_curves(kde_data_r3_chain, apo_df, holo_df, chain_classification_metrics, df_af, metrics_names_and_units_dict, show_all=1, # , df_c
        supertitle=0, show_plot=0, save_dir=output_directory_plots, save_formats=['svg'], filename_suffix='_chain', transparency=0, 
        separate_metrics=args.separate_metrics, color_scheme="curves_legacy", set_title=1)

    # KDE violins - Pocket
    plot_kde_violins(
        kde_data=kde_data_r3_pocket,
        metrics=pocket_classification_metrics,
        save_dir=output_directory_plots,
        apo_values_dict=apo_dict_pocket,   # {metric: np.array([...])}
        holo_values_dict=holo_dict_pocket,
        all_values_dict=all_dict_pocket,
        model_df=df_af,
        query_df=df_q_norm,
        save_formats=plot_save_formats,
        show_plot=args.show_plots,
        supertitle=None,
        set_title=args.kde_violin_set_title,
        filename_suffix="_pocket",
        transparency=kde_violin_transparency,
        metrics_names_and_units_dict=metrics_names_and_units_dict,
        margin_ratio=args.kde_violin_margin_ratio, # was 0.04, then 0.06
        violin_width=args.kde_violin_width,
        boxplots=args.kde_violin_boxplots,
        color_scheme=args.kde_violin_color_scheme, # sankey_clr2
        separate_metrics=args.separate_metrics,
        gridline_intervals=args.kde_violin_gridlines, # e.g., 'auto', None, [10]
        show_legend=bool(args.kde_violin_show_legend),
    )

    # KDE violins - Chain
    plot_kde_violins(
        kde_data=kde_data_r3_chain,
        metrics=chain_classification_metrics,
        save_dir=output_directory_plots,
        apo_values_dict=apo_dict_chain,   # {metric: np.array([...])}
        holo_values_dict=holo_dict_chain,
        all_values_dict=all_dict_chain,
        model_df=df_af,
        query_df=df_q_norm,
        save_formats=plot_save_formats, # 'png'
        show_plot=args.show_plots,
        supertitle=None,
        set_title=args.kde_violin_set_title,
        filename_suffix="_chain",
        transparency=kde_violin_transparency,
        metrics_names_and_units_dict=metrics_names_and_units_dict,
        margin_ratio=args.kde_violin_margin_ratio, # was 0.4
        violin_width=args.kde_violin_width,
        boxplots=args.kde_violin_boxplots,
        color_scheme=args.kde_violin_color_scheme,  # sankey_clr2
        separate_metrics=args.separate_metrics,
        gridline_intervals=args.kde_violin_gridlines,
        show_legend=bool(args.kde_violin_show_legend),
    )

    #sys.exit(1)



    ### Compute separation and overlap between Apo and Holo groups

    separation_weights = {'center': w_center, 'iqr': w_iqr, 'extremes': w_extremes}
    overlap_weights = separation_weights #{'center': 0.5, 'iqr': 0.3, 'extremes': 0.2}
    #overlap_weights2 = {'center': 0.2, 'iqr': 0.3, 'extremes': 0.5} # Invert weights, extremes = higher

    # Separation (Robust statistics)
    group_separation_robust = overall_group_separation_robust(comparisons_pocket, pocket_classification_metrics, separation_weights)
    print(f"\nApo - Holo separation score (robust):\n{group_separation_robust}")
    # Separation (Composite weighted distance)
    group_separation = overall_group_separation(comparisons_pocket, pocket_classification_metrics, separation_weights)
    print(f"Apo - Holo separation score (composite, weighted):\n{group_separation}")

    # Overlap (Robust)
    group_overlap_robust = overall_group_overlap_robust(comparisons_pocket, pocket_classification_metrics, overlap_weights)
    print(f"\nApo - Holo overall overlap score (robust):\n{group_overlap_robust}")
    # Overlap (extremes)
    group_overlap_minmax = overall_group_overlap(comparisons_pocket, pocket_classification_metrics)
    print(f"Apo - Holo overall overlap score (min/max):\n{group_overlap_minmax}")
    # Overlap (custom weights)
    group_overlap_weighted = overall_group_overlap(comparisons_pocket, pocket_classification_metrics, overlap_weights)
    print(f"Apo - Holo overall overlap score (weighted: {overlap_weights}):\n{group_overlap_weighted}")

    # Robust separation and overlap combined
    group_separation_overlap2 = calculate_group_separation_and_overlap(comparisons_pocket, pocket_classification_metrics, overlap_weights=overlap_weights, w_center=w_center, w_extremes=w_extremes, w_iqr=w_iqr)
    print(f"\nApo - Holo separation & overlap (weighted):\n{group_separation_overlap2}")
    #group_separation_overlap3 = calculate_group_separation_and_overlap(comparisons_pocket, pocket_classification_metrics, overlap_weights=overlap_weights2, w_center=w_center, w_extremes=w_extremes, w_iqr=w_iqr)
    #print(f"\nApo - Holo separation & overlap (weighted invert):\n{group_separation_overlap3}")


    ## Classification confidence plots (KDE): Gradient Bars (probability, separation, point density)
    print('\n-Plotting scores, separation and point density of KDE by metric and overall (Confidence Bars)')
    #for i in range (0, 20):

    # Pocket
    confidence_bars_kde(kde_results_r3_pocket, metrics_names_dict, save_dir=output_directory_plots, 
                        save_formats=plot_save_formats, show_plot=args.show_plots, supertitle=0, filename_suffix='_pocket', transparency=confidence_bars_transparency)
    # Chain
    confidence_bars_kde(kde_results_r3_chain, metrics_names_dict, save_dir=output_directory_plots, 
                        save_formats=plot_save_formats, show_plot=args.show_plots, supertitle=1, filename_suffix='_chain', transparency=confidence_bars_transparency)

    #sys.exit(1)



    ### Print readable report of point classification
    # Compute point classification score as a percentage of the group separation score

    # Robust
    point_score, group_separation_score_avg = classification_robust_pocket.get('score', float('NaN')), group_separation_robust.get('average_separation', float('NaN'))
    percentage = point_score_percentage(point_score, group_separation_score_avg)
    #percentage_overlap = point_score_percentage(point_score, group_overlap_robust.get('average_overlap'))
    print("\nPoint classification (Robust statistics)")
    print(f"Point score distance ({point_score:.3f}) is {percentage:.1f}% of the overall group separation ({group_separation_score_avg:.3f}).") # and {percentage_overlap:.1f}% of the overall overlap ({group_overlap_robust.get('average_overlap')})
    print(f"Apo distribution distance: {classification_robust_pocket.get('avg_distance_apo', float('NaN')):.3f}\nHolo distribution distance: {classification_robust_pocket.get('avg_distance_holo', float('NaN')):.3f}")
    print(f"Classification relative proximity to [{classification_robust_pocket.get('classification')}]: {classification_robust_pocket.get('score', float('NaN')):.3f}")
    print(f"Apo - Holo overlap: {group_overlap_robust.get('average_overlap', float('NaN')):.3f}\nApo - Holo separation: {group_separation_robust.get('average_separation', float('NaN')):.3f}")

    # Composite
    point_score, group_separation_score_avg = classification_composite_pocket.get('score', float('NaN')), group_separation.get('average_separation', float('NaN'))
    percentage = point_score_percentage(point_score, group_separation_score_avg)
    print("\nPoint classification (Composite weighted distance)")
    try:
        print(f"Point score distance ({point_score:.3f}) is {percentage:.1f}% of the overall group separation ({group_separation_score_avg:.3f}).")
        print(f"Apo distibution distance: {classification_composite_pocket.get('avg_distance_apo', float('NaN'))}\nHolo distribution distance: {classification_composite_pocket.get('avg_distance_holo')}")
        print(f"Classification relative proximity to [{classification_composite_pocket.get('classification')}]: {classification_composite_pocket.get('score', float('NaN')):.3f}")
        print(f"Apo - Holo overlap: {group_overlap_weighted.get('average_overlap', float('NaN')):.3f}\nApo - Holo separation: {group_separation.get('average_separation', float('NaN')):.3f}")
    except:
        print('Error reporting metrics')
    #sys.exit(1)

    ## Print verbose Uniformity coverage and dispersion statistics (for each metric)
    #print_uniformity_metrics(comparisons_pocket)


    # Save script parameters
    VERSION = '0.1' # Script version
    datetime_local = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    write_script_parameters(args, datetime_local, VERSION, python_version, output_directory_stats, output_filename = 'script_parameters_pocket_stats.json')

    ### Save
    save_dict_to_json(comparisons_pocket, output_directory_stats, 'apo_holo_all_metric_stats.json') # STATISTICS

    save_dict_to_json(classification_z_pocket, output_directory_stats, 'model_classification_zscore.json')
    save_dict_to_json(classification_robust_pocket, output_directory_stats, 'model_classification_robust.json')
    save_dict_to_json(classification_composite_pocket, output_directory_stats, 'model_classification_composite.json')
    save_dict_to_json(classification_dirichlet_pocket, output_directory_stats, 'model_classification_dirichlet.json')
    #save_dict_to_json(kde_results, output_directory_stats, 'model_classification_kde.json')
    save_dict_to_json(kde_results_r3_pocket, output_directory_stats, 'model_classification_kde_pocket.json')
    save_dict_to_json(kde_results_r3_chain, output_directory_stats, 'model_classification_kde_chain.json')

    save_dict_to_json(group_separation, output_directory_stats, 'group_separation.json')
    save_dict_to_json(group_overlap_minmax, output_directory_stats, 'group_overlap_minmax.json')
    save_dict_to_json(group_overlap_weighted, output_directory_stats, 'group_overlap_weighted.json')

    #save_kde_data_json(kde_data, os.path.join(output_directory_stats, 'kde_data.json')) # KDE DATA
    #save_kde_data_npz(kde_data, os.path.join(output_directory_stats, 'kde_data.npz')) # KDE DATA

    # Zip stats and plots folder (and optionally delete originals)
    #zip_file_plots = compress_directory(output_directory_plots, delete_original=False)
    #print("Created:", zip_file_plots)
    #zip_file_stats = compress_directory(output_directory_stats, delete_original=False)
    #print("Created:", zip_file_stats)


    ### Clean up & print time
    #plt.rcdefaults()  # Resets rcParams to their default settings
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print('All Done')

if __name__ == '__main__':
    main()
