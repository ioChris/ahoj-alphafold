# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:23:42 2025

@author: Christos P. Feidakis
"""

"""
Functions that are pluggable to a main batch-processing script for AHoJ-DB (zipped or unzipped)
"""

##############################  Imports  ######################################

import os
#os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP

import pandas as pd
import numpy as np
import zipfile
import json
import tarfile
import io
import traceback

# AHoJ results dataframe curation
from parse_AHoJ3_DBv2b import replace_zero_with_nan

# Plotting, KDE, Model classification
from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions, compute_metrics, create_summary_df

from stats_api import compute_absolute_statistics_v2
from stats_plot import plot_dp_curves
from stats_core import (
    classify_point_modified_z, 
    classify_point_robust, classify_point_composite,
    overall_group_separation_robust, overall_group_separation,
    overall_group_overlap_robust, overall_group_overlap,
    calculate_group_separation_and_overlap, find_closest_candidates,
    point_score_percentage, dp_classify_new_point
    )

from conformational_spectrum_api import (
    add_fraction_metric, normalize_query_df,
    metrics_names_and_units_dict, metrics_names_dict, compress_directory,
    save_dict_to_json, compress_file
    #plot_all_metrics_with_all_violin_overlap
    )

from kde_api import classify_point_kde_r3
from kde_plot import plot_kde_curves, plot_kde_violins, confidence_bars_kde

# Run types (docs/RUN_TYPES.md): holo-2.1 (legacy v2c), holo-2.4, afc-2.4
import run_types as rt
from run_types import RunTypeError



###############################################################################

def curate_dataframe():
    # Placeholder function to spin-off dataframe curation including filtering/sorting/value replacement etc.
    # returns all what's needed to be further processed
    pass


def indent_multiline(s, num_spaces=4):
    """
    Indents all lines of a multiline string with a specified number of spaces.
    """
    indent = ' ' * num_spaces
    return indent + s.replace('\n', '\n' + indent)


def print_exception(e):
    """
    Prints the exception message and stack trace.
    """
    print(indent_multiline(traceback.format_exc()))


def load_file_from_tar(query_tgz_path, job_name, file_name):
    with tarfile.open(query_tgz_path, 'r:gz') as tar:
        if file_name in tar.getnames():
            file = tar.extractfile(file_name)
            if file_name.endswith('.csv'):
                df = pd.read_csv(io.TextIOWrapper(file, newline=""))
                return ('Success', (job_name, len(df), df))
            elif file_name.endswith('.json'):
                data = json.load(io.TextIOWrapper(file, encoding='utf-8'))
                return ('Success', (job_name, data))
            # Add more handlers as needed
        return ('File Not Found', (job_name, None))


def load_file_from_dir(query_dir_path, job_name, file_name):
    file_path = os.path.join(query_dir_path, file_name)
    if os.path.exists(file_path):
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
            return ('Success', (job_name, len(df), df))
        elif file_name.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ('Success', (job_name, data))
    return ('File Not Found', (job_name, None))


def load_file_from_query(query_path, job_name, file_name):
    # Loading defaults: CSV -> DataFrame, JSON -> Dictionary
    try:
        if os.path.isdir(query_path):
            return load_file_from_dir(query_path, job_name, file_name)
        else:
            return load_file_from_tar(query_path, job_name, file_name)
    except Exception as e:
        print(f'Exception while processing job {job_name}: {e}')
        print_exception(e)
        return ('Error', (job_name, str(e), None))


def extract_file(query_path, inner_path):
    if os.path.isdir(query_path):
        candidate = os.path.join(query_path, inner_path)
        if os.path.exists(candidate):
            return open(candidate, 'rb')
        else:
            raise FileNotFoundError(candidate)
    else:
        with tarfile.open(query_path, 'r:gz') as tar:
            try:
                member = tar.getmember(inner_path)
                return tar.extractfile(member)
            except KeyError:
                raise FileNotFoundError(f"{inner_path} not found in {query_path}")


def shorten_path(path, levels=4):
    """Return the last path components for compact summaries."""
    try:
        parts = os.path.normpath(str(path)).split(os.path.sep)
        return os.path.join(*parts[-levels:]) if len(parts) >= levels else str(path)
    except Exception:
        return str(path)


def strip_archive_suffix(path):
    """Map an archive path to the matching extracted job-directory path."""
    path = str(path)
    lower = path.lower()
    if lower.endswith(".tar.gz"):
        return path[:-7]
    if lower.endswith(".tgz"):
        return path[:-4]
    return path


def resolve_mirrored_job_dir(root_data_dir, mirror_root_dir, query_path):
    """
    Resolve a query path under root_data_dir to the matching job directory
    under mirror_root_dir, stripping .tgz/.tar.gz when present.
    """
    query_path_no_archive = strip_archive_suffix(query_path)
    if mirror_root_dir:
        try:
            relative_path = os.path.relpath(query_path, root_data_dir)
            relative_path = strip_archive_suffix(relative_path)
        except (TypeError, ValueError):
            relative_path = os.path.join(
                os.path.basename(os.path.dirname(query_path_no_archive)),
                os.path.basename(query_path_no_archive),
            )
        return os.path.join(mirror_root_dir, relative_path)
    return query_path_no_archive


def csv_scalar(value):
    """Normalize JSON values for CSV-friendly one-row summaries."""
    if value is None:
        return np.nan
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def selected_json_keys(data, requested):
    """Return requested keys, supporting 'all'/'*' while preserving JSON order."""
    if not isinstance(data, dict):
        return []
    requested = requested or []
    if any(str(key).lower() in {"all", "*"} for key in requested):
        return list(data.keys())
    return [key for key in requested if key in data]


def flatten_json_value(out, prefix, value):
    """Recursively flatten nested JSON dictionaries into stable column names."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            flatten_json_value(out, f"{prefix}_{key}", nested_value)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, sort_keys=True)
    else:
        out[prefix] = csv_scalar(value)


def flatten_statistics_json(stats_data, scope, groups, group_fields, metric_fields, include_nested=False):
    """
    Flatten metric-first apo/holo/all statistics JSONs into a single row.

    Group columns are named like:
      stats_pocket_All_volume_mean

    Metric-level comparison columns are named like:
      stats_pocket_volume_cohen_d
    """
    row = {}
    if not isinstance(stats_data, dict):
        return row

    for metric, metric_data in stats_data.items():
        if not isinstance(metric_data, dict):
            continue

        for group in groups:
            group_data = metric_data.get(group, {})
            if not isinstance(group_data, dict):
                continue

            for field in selected_json_keys(group_data, group_fields):
                value = group_data.get(field)
                prefix = f"stats_{scope}_{group}_{metric}_{field}"
                if isinstance(value, dict):
                    if include_nested:
                        flatten_json_value(row, prefix, value)
                else:
                    row[prefix] = csv_scalar(value)

        for field in selected_json_keys(metric_data, metric_fields):
            value = metric_data.get(field)
            prefix = f"stats_{scope}_{metric}_{field}"
            if isinstance(value, dict):
                if include_nested:
                    flatten_json_value(row, prefix, value)
            else:
                row[prefix] = csv_scalar(value)

    return row



# Deprecated start
def load_dataframe_from_tar(query_tgz_path, job_name, file_name):
    with tarfile.open(query_tgz_path, 'r:gz') as tar:
        if file_name in tar.getnames():
            file = tar.extractfile(file_name)
            df = pd.read_csv(io.TextIOWrapper(file, newline=""))
            return ('Success', (job_name, len(df), df))
        else:
            return ('File Not Found', (job_name, 0, pd.DataFrame()))

def load_dataframe_from_dir(query_dir_path, job_name, file_name):
    file_path = os.path.join(query_dir_path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return ('Success', (job_name, len(df), df))
    else:
        return ('File Not Found', (job_name, 0, pd.DataFrame()))

def load_dataframe_from_query(query_path, job_name, file_name = './global_results.csv'):
    # query_path: query result - either a directory or compressed tgz file
    try:
        if os.path.isdir(query_path):
            return load_dataframe_from_dir(query_path, job_name, file_name)
        else:
            # query_path is a .tgz file
            return load_dataframe_from_tar(query_path, job_name, file_name)
    except Exception as e:
        print(f'Exception while processing job {job_name}: {e}')
        print_exception(e)
        return ('Error', (job_name, str(e), pd.DataFrame()))

def process_and_filter_job(query_path, job_name, args, job_processing_func):
    try:
        job_status, rest = load_file_from_query(query_path, job_name, './global_results.csv')
        job_name, df_length, df = rest[0], rest[1], rest[2]

        if job_status == 'Success':
            #return do_kde_plots_and_classification(args, df, job_name, query_path, args.root_mirror_dir)
            #return do_process_job(args, df, job_name, query_path, args.root_mirror_dir)
            return job_processing_func(args, df, job_name, query_path)
        else:
            return ('File Not Found', (job_name, 0))

    except Exception as e:
        print(f'[ERROR] Exception while processing job {job_name} [{os.path.join(*query_path.split(os.path.sep)[-4:])}]: {e}')
        print_exception(e)
        return ('Error', (job_name, f'{str(e)} {query_path}', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
# Deprecated end



def collect_unique_ligands(df, columns=("query_ligs", "nonquery_ligs")):
    """
    Non-redundant set of ligand codes found in `columns` (each a '-'-joined string such as
    "EAA-GSH"), excluding blanks and 'nan'. Read-only / NaN-safe; mirrors the splitting logic
    of parse_AHoJ3_DBv2b.extract_unique_ligands. Returns (set_of_codes, count).
    Used to record, per query, the ligand(s) that bind the pocket (query vs candidate scope)
    for downstream integration and a possible redock-known-ligands experiment.
    """
    ligs = set()
    if df is None or getattr(df, "empty", True):
        return ligs, 0
    for col in columns:
        if col in df.columns:
            for item in df[col].astype("string"):
                if item is pd.NA:
                    continue
                for tok in str(item).split("-"):
                    tok = tok.strip()
                    if tok and tok.lower() != "nan":
                        ligs.add(tok)
    return ligs, len(ligs)


def extract_model_pLDDT_and_query_meta(df, meta_role="Q", model_row="AF_model"):
    """
    Reusable extraction of (a) the AlphaFold model pocket-average pLDDT (mean of the role 'M'
    'bfac_avg' column) and (b) query structure metadata (resolution, exp_method) from role
    `meta_role` (usually 'Q'). Mirrors the field logic of do_collect_model_pLDDT so the full
    classification run records the same columns the previous run lacked (resolution + model pLDDT),
    avoiding a separate re-parse. Returns scalar fields (NaN/None when unavailable).

    RUN_TYPES item 6: `model_row` selects the model rows ('AF_model' = the query's own model; UniWeave
    adds neighbour-UniProt rows with role M; see rt.select_model_rows).
    """
    out = {
        "query_resolution": np.nan,
        "query_exp_method": None,
        "model_pocket_pLDDT_avg": np.nan,
        "n_model_rows": 0,
        # AlphaFold model pocket geometry (the pocket actually docked into); see below.
        "model_pocket_volume": np.nan,
        "model_pocket_sasa": np.nan,
        "model_pocket_RoG": np.nan,
        "model_pocket_len": np.nan,
        "model_pocket_rms": np.nan,
    }
    if df is None or getattr(df, "empty", True) or "role" not in df.columns:
        return out

    # AlphaFold model pocket pLDDT: stored in 'bfac_avg' for role == 'M' (the query's own model).
    df_m = rt.select_model_rows(df, model_row)
    out["n_model_rows"] = int(len(df_m))
    if len(df_m) and "bfac_avg" in df_m.columns:
        vals = pd.to_numeric(df_m["bfac_avg"], errors="coerce")
        if vals.notna().any():
            out["model_pocket_pLDDT_avg"] = round(float(np.nanmean(vals)), 3)

    # AHOJ-015: AlphaFold model pocket geometry (same role 'M' row(s)): volume / SASA / radius of
    # gyration / residue count. This is the pocket a docking run would target, and the geometry the
    # "is the AF pocket holo-like / druggable?" question is about. Only model pLDDT was captured
    # before, forcing a re-parse for the rest (cf. AHOJ-014 for resolution/pLDDT).
    #
    # AHOJ-019: `pocket_rms` added. It is the FOURTH classifier metric (pocket_class_metrics =
    # pocket_rms, sasa, volume, RoG), so the model row has always carried a value for it -- the
    # classification could not run otherwise -- but it was never aggregated here, which left the
    # summaries able to describe the model on only three of the four axes it is scored on.
    # NOTE ON MEANING: unlike volume/SASA/RoG, which are intrinsic properties of the model pocket,
    # `pocket_rms` is a DISTANCE FROM THE QUERY structure. Every candidate is measured against the
    # same query, so the model's value is directly comparable to theirs; but a statistic built on it
    # describes spread around a designated reference (itself one of the holo structures), not spread
    # in an absolute property. Do not pool it with the other three without saying so.
    if len(df_m):
        for col, key in (("volume", "model_pocket_volume"), ("sasa", "model_pocket_sasa"),
                         ("RoG", "model_pocket_RoG"), ("pocket_len", "model_pocket_len"),
                         ("pocket_rms", "model_pocket_rms")):
            if col in df_m.columns:
                v = pd.to_numeric(df_m[col], errors="coerce")
                if v.notna().any():
                    out[key] = round(float(np.nanmean(v)), 3)

    # Query structure metadata (role meta_role, usually the experimental query 'Q').
    df_qm = df[df["role"] == meta_role]
    if len(df_qm):
        if "resolution" in df_qm.columns:
            res = pd.to_numeric(df_qm["resolution"], errors="coerce").dropna()
            if not res.empty:
                out["query_resolution"] = float(res.iloc[0])
        if "exp_method" in df_qm.columns:
            em = df_qm["exp_method"].astype("string").str.strip()
            em = em[em.notna() & (em != "")]
            if len(em):
                out["query_exp_method"] = str(em.iloc[0])
    return out


def _norm_class_label(lbl):
    """Normalize a method's class label to 'Apo'/'Holo', or None (Ambiguous/Tie/Undefined/etc.)."""
    s = str(lbl).strip().lower()
    if s == "apo":
        return "Apo"
    if s == "holo":
        return "Holo"
    return None  # not counted toward consensus


def build_classification_summary(kde_res, dp_res, z_res, robust_res, comp_res):
    """
    Collapse the five per-scope classifiers into one compact block: each method's verdict +
    its own native primary score/support, plus a consensus across methods.

    Inputs are the dicts returned by classify_point_kde_r3 (full result), dp_classify_new_point,
    and classify_point_modified_z / classify_point_robust / classify_point_composite.
    Replaces the eight per-method JSONs (z/robust/composite/dirichlet x pocket/chain) for the
    default output while keeping every verdict. See KNOWN_ISSUES.md.
    """
    kde_ov = (kde_res or {}).get("overall", {})
    dp_ov = (dp_res or {}).get("overall", {})
    z_res, robust_res, comp_res = (z_res or {}), (robust_res or {}), (comp_res or {})

    kde = {
        "class": kde_ov.get("overall_classification"),
        "p_holo": kde_ov.get("avg_P(holo)"),
        "p_apo": kde_ov.get("avg_P(apo)"),
        "score": kde_ov.get("overall_score"),
        "support": {
            "norm_density_strongest": kde_ov.get("avg_normalized_density_strongest"),
            "overlap": kde_ov.get("overall_overlap"),
            "mode_overlap": kde_ov.get("overall_mode_overlap"),
        },
    }
    dirichlet = {
        "class": dp_ov.get("overall_classification"),
        "score": dp_ov.get("avg_score"),
        "overlap": dp_ov.get("avg_overlap"),
    }
    modz = {
        "class": z_res.get("classification"),
        "score": z_res.get("score"),
        "d_apo": z_res.get("avg_distance_apo"),
        "d_holo": z_res.get("avg_distance_holo"),
    }
    robust = {
        "class": robust_res.get("classification"),
        "score": robust_res.get("score"),
        "d_apo": robust_res.get("avg_distance_apo"),
        "d_holo": robust_res.get("avg_distance_holo"),
    }
    composite = {
        "class": comp_res.get("classification"),
        "score": comp_res.get("score"),
        "d_apo": comp_res.get("avg_distance_apo"),
        "d_holo": comp_res.get("avg_distance_holo"),
    }

    votes = [v for v in (_norm_class_label(b["class"]) for b in (kde, dirichlet, modz, robust, composite)) if v]
    n_meth = len(votes)
    n_holo = sum(v == "Holo" for v in votes)
    n_apo = sum(v == "Apo" for v in votes)
    if n_meth == 0:
        cons_class, cons_frac = "Undefined", None
    elif n_holo > n_apo:
        cons_class, cons_frac = "Holo", round(n_holo / n_meth, 3)
    elif n_apo > n_holo:
        cons_class, cons_frac = "Apo", round(n_apo / n_meth, 3)
    else:
        cons_class, cons_frac = "Tie", round(max(n_holo, n_apo) / n_meth, 3)

    return {
        "kde": kde, "dirichlet": dirichlet, "modz": modz, "robust": robust, "composite": composite,
        "consensus": {"class": cons_class, "frac": cons_frac, "n_methods": n_meth,
                      "n_holo": n_holo, "n_apo": n_apo},
    }


def do_kde_plots_and_classification(args, job_name, query_path):
    """
    Generate plots that illustrate metric distribution, classify AF-DB models

        Processing: 
        - Characterize the conformational spectrum of each pocket by parsing
        selected metrics
        - Assign apo/holo status to the AlphaFold-DB model in the given entry

        Additional considerations
        - Allow multiprocessing of entries
        - Intermediate batch folders between root directory and entry directories

    """

    ### Load df
    try:
        # RUN_TYPES item 1: both member spellings ('./global_results.csv' and 'global_results.csv').
        # AHOJ-017: a .tgz that exists but holds no global_results.csv is 'File Not Found', not an
        #   IndexError relabelled as 'Error'. See KNOWN_ISSUES.md (AHOJ-017).
        df = rt.load_global_results(query_path)
        if df is None:
            return ('File Not Found', (job_name, 0))

    except Exception as e:
        print(f'[ERROR] Exception while processing job {job_name} [{os.path.join(*query_path.split(os.path.sep)[-4:])}]: {e}')
        print_exception(e)
        return ('Error', (job_name, f'{str(e)} {query_path}', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))


    ### Data curation start

    # Drop rows where all values are nan (Empty AF row)
    # df = df.dropna(how="all") # TODO revisit this option (4heu B 15J 809)

    # Discard empty entries
    if df.empty or (df.replace('', np.nan).isnull().all(axis=None)):
        return ('Discarded', (job_name, f'Empty dataframe: {len(df)}'))  # return df.iloc[0:0]

    # RUN TYPE (docs/RUN_TYPES.md). Which AHoJ wrote the archive, and which rules apply to it. Every
    # contradiction between the archive, the config and the job key is a loud 'Error', never a
    # silently different number.
    try:
        script_params = rt.load_script_parameters(query_path)
        profile = rt.resolve_profile(args, script_params, df, job_name)
        job_key = rt.parse_job_key(job_name, profile.key_format)
        site = rt.site_for_job(job_name, profile.key_format)
        if profile.relabel_mode == 'entity_gate_site' and (profile.ion_rule == 'site' or profile.additive_rule == 'site') and site is None:
            raise RunTypeError(f'[{job_name}] site {job_key.site_uid} is not in the sites table')
    except RunTypeError as e:
        print(f'[ERROR] run type: {e}')
        return ('Error', (job_name, f'RunTypeError: {e}', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))


    ## Create mirrored job dir
    last_two_dirs = os.path.join(*query_path.split(os.path.sep)[-2:])   # Extract last two dirs (incl. filename)
    dir_without_ext = os.path.splitext(last_two_dirs)[0]                # Strip file extension (.tgz)

    # MAKE DIRECTORY - Plots
    mirrored_jobdir_plots = os.path.join(args.root_mirror_dir, dir_without_ext, 'plots')
    if any([args.general_violins == 1,args.kde_curves == 1,args.kde_violins == 1,args.conf_bars == 1]):
        os.makedirs(mirrored_jobdir_plots, exist_ok=True)

    # MAKE DIRECTORY - Statistics/text output
    mirrored_jobdir_stats = os.path.join(args.root_mirror_dir, dir_without_ext, 'statistics')
    if any([args.save_filtered_df == 1,args.save_statistics == 1]):
        os.makedirs(mirrored_jobdir_stats, exist_ok=True)


    # Relabel candidate apoholo_assignment (RUN_TYPES items 2-4; the profile picks the rule)
    #   v2c_ions              holo-2.1: rebuild H/H*/A from the ligand columns, ions excluded (v2c)
    #   v2c_ions_demote_only  holo-2.4: keep AHoJ's gate labels; only ion-only holo calls -> A**
    #   entity_gate_site      afc-2.4: from ligands.json entities, gate + site-aware ion rule
    #   none                  keep AHoJ's own labels
    try:
        if profile.relabel_mode in ('v2c_ions', 'v2c_ions_demote_only'):
            # The relabeled apoholo_assignment column survives into global_results_plots.csv, so the
            # separate global_results_relabeled.csv is redundant; only write it when explicitly asked.
            _relabel_save_dir = mirrored_jobdir_stats if int(getattr(args, "save_relabeled_csv", 0)) == 1 else None
            df, changed = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=_relabel_save_dir, verbose=False,
                                                  protect_labels=set(profile.protect_labels),
                                                  demote_only=(profile.relabel_mode == 'v2c_ions_demote_only'))
        elif profile.relabel_mode == 'entity_gate_site':
            _mode, _thr = rt.gate_from_script_params(script_params)
            _gate = None if profile.gate == 'none' or (profile.gate == 'archive' and _mode != 'gate') else _thr
            df, changed = rt.relabel_by_entities(df, rt.load_ligands_json(query_path), _gate, rt.ion_set(),
                                                 rt.additive_set(getattr(args, 'additives_file', None)),
                                                 profile.ion_rule, profile.additive_rule, site,
                                                 protect_labels=profile.protect_labels)
    except RunTypeError as e:
        print(f'[ERROR] run type: {e}')
        return ('Error', (job_name, f'RunTypeError: {e}', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))

    # AHOJ-020: a candidate label outside rt.KNOWN_CANDIDATE_LABELS joins neither set; count and report it
    label_summary = audit_candidate_labels(df, job_name)

    # Replace zero with np.nan in columns that should not have zero (b-factor)
    df = replace_zero_with_nan(df, args.columns_nozero)

    # Volume (RUN_TYPES item 9): the profile names the column that becomes 'volume' (2.4 archives carry
    # volume, volume2, volume_seed, ...; 2.1 has only PyVOL's volume) and whether the v2c NaN->0 rule
    # applies (a PyVOL pocket with residues and no volume was an empty pocket, so 0.0).
    if profile.volume_column != 'volume':
        df['volume_ahoj'] = df['volume']
        df['volume'] = df[profile.volume_column]
    s = pd.to_numeric(df['#resis3'], errors='coerce')          # helper: NaN if empty/non-numeric
    v = pd.to_numeric(df['volume'],  errors='coerce')          # convert '' -> NaN, keep numbers
    df['volume'] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v) if profile.volume_nan_to_zero else v
    #df['volume'] = df['volume'].replace('', np.nan).astype(float).fillna(0.0)  # previous standard

    # Add new column with composite metric to all rows (ratio of SASA to ASA)
    df = add_fraction_metric(df)

    # Split global results data frame
    df_q = df[df['role'] == 'Q']
    df_af = rt.select_model_rows(df, profile.model_row)     # RUN_TYPES item 6

    # RUN_TYPES item 13: the +/-60% RoG window is anchored on the query row (the model in afc-2.4).
    # A NaN anchor would drop every candidate silently, so it is an error here.
    if args.filter_candidates == 1 and args.rog_diff_thrs_query and not df_q.empty:
        _qrog = pd.to_numeric(df_q['RoG'], errors='coerce')
        if _qrog.isna().all():
            return ('Error', (job_name, f'RunTypeError: query-row RoG is NaN ({profile.rog_anchor} anchor); '
                              'the RoG window would drop every candidate', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))

    # AHOJ-004 / AHOJ-018: an entry may lack a model (role 'M') row. The apo/holo classification of
    #   the model point is then impossible, but the CONFORMATIONAL analysis (apo_holo_all_stats,
    #   KDE/DP densities and the distribution plots) is model-INDEPENDENT and still valuable (e.g. for
    #   the webserver, which shows plots for every entry). Behaviour is parameterised by keep_no_model:
    #     0 (default, AlphaFold-project use case): discard - there is nothing to classify.
    #     1 (webserver use case): KEEP and process. Every downstream point-classifier already guards an
    #       empty model (classify_point_* / dp_classify_new_point / prepare_metric_data ->
    #       "Undefined"/NaN/"Insufficient Data") and every plot call is try/except-wrapped, so the
    #       pipeline runs end-to-end. The summary row is flagged has_model=0 with an "N/A (no model)"
    #       overall verdict (see create_query_summary_row). See KNOWN_ISSUES.md (AHOJ-018).
    has_model = not df_af.empty
    if not has_model and int(getattr(args, 'keep_no_model', 0)) != 1:
        return ('Discarded', (job_name, 'No model (role M) row'))

    # Filter candidates
    if args.filter_candidates == 1:
        df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=args.full_pockets_aligned,
                                  bndgrsds_prcnt_seqres=args.bndgrsds_prcnt_seqres,
                                  bndgrsds_prcnt_obs=args.bndgrsds_prcnt_obs, resolution_soft=args.resolution_soft,
                                  rog_diff_threshold_deltaQ=args.rog_diff_thrs_query,
                                  rog_diff_threshold_mean=args.rog_diff_thrs_mean, pocket_distance=args.pocket_dist,
                                  pocket_len_resis3=args.pocket_len_resis3,
                                  drop_no_resolution=getattr(args, 'drop_no_resolution', 0))
    else:
        df_f = df.copy()  # Unfiltered, but copied for consistency

    # Extract candidates
    df_c = df_f[df_f['role'] == 'C']

    # Skip entry if not enough candidates according to arguments (min_apo, min_holo, min_apoholo)
    min_results_true, n_apo, n_holo, n_total = has_minimum_apo_holo(df_c, min_apo=args.min_apo, min_holo=args.min_holo, min_apoholo=args.min_apoholo)
    if not min_results_true:
        return ('Discarded', (job_name, f'Low candidates after filtering: apo={n_apo}, holo={n_holo}, total={n_total}'))
    # AHOJ-020: A- is pooled as apo; its count keeps the demoted query-ligand hits visible per entry
    label_summary['n_apo_query_demoted'] = int((df_c['apoholo_assignment'] == 'A-').sum())

    # Flat candidate metrics for the aggregate query-summary CSV. Keep this before adding the query row.
    # AHOJ-015: emit a SINGLE clean candidate-only block (anchor_query=False = the AHOJ-007 unbiased
    #   stats, no synthetic query point) with BARE stat names (volume_mean, RoG_mean, ... — matching the
    #   historical P2Rank-correlate columns). The anchored/query-injected block was the count-biased
    #   legacy duplicate and is dropped. Only the stats in --query_summary_stats are emitted (curated to
    #   keep the table lean); local/global flexibility are always included. deltas default is 1 (raw +
    #   query deltas) so both the raw size correlates and the delta/range descriptors are kept.
    _deltas = getattr(args, "query_summary_deltas", 1)
    _qcols = list(getattr(args, "query_summary_metrics", []))
    _qstats = getattr(args, "query_summary_stats", None) or None
    query_metric_summary_df = compute_metrics(df_c, df_q, deltas=_deltas, columns=_qcols,
                                              anchor_query=False, keep_stats=_qstats)

    # Add query row to candidates (for use in plots and classification). RUN_TYPES item 5: the profile
    # decides. holo: the query row joins the holo set (pocket RMSD excluded through normalize_query_df).
    # afc-2.4: the query row IS the AlphaFold model and never joins (resolve_profile refuses 1).
    if profile.add_query_row == 1:
        df_c = pd.concat([df_q, df_c], ignore_index=True)

    # Set superposition-related metrics/deltas to 0 and 1 in Query DF (for plotting query reference point)
    query_metrics_zero = ["pocket_rms", "pocket_dist", "rmsd"]
    query_metrics_one = ["tm_score", "tm_score_i"]
    df_q_norm = normalize_query_df(df_q, query_metrics_zero, query_metrics_one)

    # Extract apo and holo data
    # AHOJ-001: 'A**' (holo candidate relabeled to apo because it only bound an excluded
    #   single-atom ligand) must count as apo during classification. See KNOWN_ISSUES.md.
    # AHOJ-020: so must AHoJ's 'A-' (a gate-demoted query-ligand hit). The label sets live in
    #   run_types (APO_LABELS / HOLO_LABELS); the label itself stays in the column.
    apo_df = df_c[df_c['apoholo_assignment'].isin(rt.APO_LABELS)]
    holo_df = df_c[df_c['apoholo_assignment'].isin(rt.HOLO_LABELS)]

    # Optional old-style apo/holo summary rows for aggregate CSV export.
    # Use candidate rows only; df_c may also include the query row for plotting/classification.
    apo_summary, holo_summary = pd.DataFrame(), pd.DataFrame()
    if int(getattr(args, "apo_holo_sums", 0)) == 1:
        candidate_rows = df_c[df_c["role"] == "C"]
        apo_summary_df = candidate_rows[candidate_rows['apoholo_assignment'].isin(rt.APO_LABELS)]  # AHOJ-001, AHOJ-020
        holo_summary_df = candidate_rows[candidate_rows['apoholo_assignment'].isin(rt.HOLO_LABELS)]
        paired_entry = 1 if not apo_summary_df.empty and not holo_summary_df.empty else 0
        if not apo_summary_df.empty:
            apo_summary = create_summary_df(df_q, apo_summary_df, 'A', paired_entry, job_name).iloc[1:2]
        if not holo_summary_df.empty:
            holo_summary = create_summary_df(df_q, holo_summary_df, 'H', paired_entry, job_name).iloc[1:2]

    # Save the dataframe that will be used for plots
    if args.save_filtered_df == 1:
        df_f.to_csv(os.path.join(mirrored_jobdir_stats, 'global_results_plots.csv'), index=False)

    ### Data curation end


    ### Analysis

    ## Get closest candidate(s) to model
    all_metrics = list(dict.fromkeys(args.pocket_class_metrics + args.chain_class_metrics))

    # Run separately to scope "overall" to each set:
    try:
        closest_pocket = find_closest_candidates(df_c[df_c["role"]=="C"], df_af, args.pocket_class_metrics, top_n=5, scale="range", decimals=5)
        closest_chain  = find_closest_candidates(df_c[df_c["role"]=="C"], df_af, args.chain_class_metrics,  top_n=5, scale="range", decimals=5)
        closest_all    = find_closest_candidates(df_c[df_c["role"]=="C"], df_af, all_metrics,    top_n=5, scale="range", decimals=5)
    except Exception as e:
        closest_pocket, closest_chain, closest_all = {},{},{}
        print(f'[{job_name}] Get closest candidates error: {e}')

    # Plot and save general violin plots (pocket & chain)
    #if args.general_violins == 1:
    #    plot_all_metrics_with_all_violin_overlap(df_c, args.pocket_class_metrics, mirrored_jobdir_plots, model_df=df_af, query_df=df_q_norm, 
    #                                             save_formats=['svg'], show_plot=args.show_plots, filename_suffix="_pocket",
    #                                             transparency=0, gridline_intervals=None, metrics_names_and_units_dict=metrics_names_and_units_dict)

    #    plot_all_metrics_with_all_violin_overlap(df_c, args.chain_class_metrics, mirrored_jobdir_plots, model_df=df_af, query_df=df_q_norm,
    #                                             save_formats=['svg'], show_plot=args.show_plots, filename_suffix="_chain",
    #                                             transparency=0, gridline_intervals=None, metrics_names_and_units_dict=metrics_names_and_units_dict)

    ## Compute general statistics (pocket & chain)
    comparisons_pocket = compute_absolute_statistics_v2(apo_df, holo_df, args.pocket_class_metrics)
    comparisons_chain  = compute_absolute_statistics_v2(apo_df, holo_df, args.chain_class_metrics)

    ## Do point classifications (pocket & chain)
    # Z-scores
    classification_z_pocket = classify_point_modified_z(df_af, comparisons_pocket, args.pocket_class_metrics)
    classification_z_chain  = classify_point_modified_z(df_af, comparisons_chain,  args.chain_class_metrics)

    # Robust statistics
    classification_robust_pocket = classify_point_robust(df_af, comparisons_pocket, args.pocket_class_metrics)
    classification_robust_chain  = classify_point_robust(df_af, comparisons_chain,  args.chain_class_metrics)

    # Composite distance
    w_center, w_iqr, w_extremes     = args.separation_weights.get('w_center'), args.separation_weights.get('w_iqr'), args.separation_weights.get('w_extremes')
    classification_composite_pocket = classify_point_composite(df_af, comparisons_pocket, args.pocket_class_metrics, w_center=w_center, w_iqr=w_iqr, w_extremes=w_extremes)
    classification_composite_chain  = classify_point_composite(df_af, comparisons_chain,  args.chain_class_metrics,  w_center=w_center, w_iqr=w_iqr, w_extremes=w_extremes)

    # Dirichlet process mixture model. AHOJ-013: capture the per-metric densities so the DP plot can
    # draw EXACTLY the curve that drove the classification (fidelity) and skip a redundant BGM refit.
    classification_dirichlet_pocket, dp_curves_pocket = dp_classify_new_point(df_c, args.pocket_class_metrics, df_af, return_curves=True)
    classification_dirichlet_chain,  dp_curves_chain  = dp_classify_new_point(df_c, args.chain_class_metrics,  df_af, return_curves=True)

    # KDE
    kde_results_pocket, kde_data_pocket, apo_values_pocket, holo_values_pocket, all_values_pocket = classify_point_kde_r3(df_c, apo_df, holo_df, df_af,
        args.pocket_class_metrics, kde_extend=args.kde_extend, margin_cap_scale=args.margin_cap_scale, bw_mode=args.bw_mode, bw_scale=args.bw_scale, prior_mode=args.prior_mode)
    kde_results_chain, kde_data_chain, apo_values_chain, holo_values_chain, all_values_chain = classify_point_kde_r3(df_c, apo_df, holo_df, df_af,
        args.chain_class_metrics, kde_extend=args.kde_extend, margin_cap_scale=args.margin_cap_scale, bw_mode=args.bw_mode, bw_scale=args.bw_scale, prior_mode=args.prior_mode)

    ## All-methods compact summary + consensus (pocket & chain).
    # One artifact carrying every classifier's verdict; replaces the 8 per-method JSONs by default
    # and feeds the consensus/support columns used for downstream confidence filtering (F5).
    classification_summary = {
        "pocket": build_classification_summary(kde_results_pocket, classification_dirichlet_pocket,
                    classification_z_pocket, classification_robust_pocket, classification_composite_pocket),
        "chain": build_classification_summary(kde_results_chain, classification_dirichlet_chain,
                    classification_z_chain, classification_robust_chain, classification_composite_chain),
    }


    ## Plots

    # DP-GMM curves
    if args.dp_curves == 1:
        try:
            # Pocket
            plot_dp_curves(
                apo_group=apo_df, holo_group=holo_df, classification_metrics=args.pocket_class_metrics, model_point=df_af,
                supertitle=0, show_plot=args.show_plots, save_dir=mirrored_jobdir_plots, save_formats=['svg'], filename_suffix='_pocket',
                transparency=0, separate_metrics=args.separate_metrics, color_scheme="blue_orange1",
                metrics_names_and_units_dict=metrics_names_and_units_dict,
                set_title=1, show_legend=True, show_points=True, precomputed_curves=dp_curves_pocket)

            # Chain
            plot_dp_curves(
                apo_group=apo_df, holo_group=holo_df, classification_metrics=args.chain_class_metrics, model_point=df_af,
                supertitle=0, show_plot=args.show_plots, save_dir=mirrored_jobdir_plots, save_formats=['svg'], filename_suffix='_chain',
                transparency=0, separate_metrics=args.separate_metrics, color_scheme="blue_orange1",
                metrics_names_and_units_dict=metrics_names_and_units_dict,
                set_title=1, show_legend=True, show_points=True, precomputed_curves=dp_curves_chain)

        except Exception as e:
            print(f'[{job_name}] DP-GMM curves error: {e}')

    # KDE curves
    if args.kde_curves == 1:
        try:
            # Pocket
            plot_kde_curves(kde_data_pocket, apo_df, holo_df, args.pocket_class_metrics, df_af, metrics_names_and_units_dict, show_all=1, supertitle=0,
                                             show_plot=args.show_plots, save_dir=mirrored_jobdir_plots, save_formats=['svg'], filename_suffix='_pocket', 
                                             transparency=0, separate_metrics=args.separate_metrics, color_scheme="curves_legacy", set_title=1)
            # Chain
            plot_kde_curves(kde_data_chain, apo_df, holo_df, args.chain_class_metrics, df_af, metrics_names_and_units_dict, show_all=1, supertitle=0,
                                             show_plot=args.show_plots, save_dir=mirrored_jobdir_plots, save_formats=['svg'], filename_suffix='_chain', 
                                             transparency=0, separate_metrics=args.separate_metrics, color_scheme="curves_legacy", set_title=1)

        except Exception as e:
            print(f'[{job_name}] KDE curves error: {e}')


    # KDE violins
    if args.kde_violins == 1:

        try:
            # Pocket
            plot_kde_violins(
                kde_data_pocket,
                args.pocket_class_metrics,
                mirrored_jobdir_plots,
                apo_values_pocket, holo_values_pocket, all_values_pocket,
                model_df=df_af, query_df=df_q_norm,
                save_formats=['svg'],
                show_plot=args.show_plots,
                supertitle=None, set_title=1,
                filename_suffix="_pocket",
                transparency=0,
                metrics_names_and_units_dict=metrics_names_and_units_dict,
                margin_ratio=0.04, # was 0.04, then 0.06
                violin_width=0.80, # was 0.86
                boxplots=1,
                color_scheme='seaborn1',
                separate_metrics=args.separate_metrics,
                gridline_intervals=None)#, # e.g., 'auto', None, [10]

            # Chain
            plot_kde_violins(
                kde_data_chain,
                args.chain_class_metrics,
                mirrored_jobdir_plots,
                apo_values_chain, holo_values_chain, all_values_chain,
                model_df=df_af, query_df=df_q_norm,
                save_formats=['svg'],
                show_plot=args.show_plots,
                supertitle=None, set_title=1,
                filename_suffix="_chain",
                transparency=0,
                metrics_names_and_units_dict=metrics_names_and_units_dict,
                margin_ratio=0.04, # was 0.04, then 0.06
                violin_width=0.80, # was 0.86
                boxplots=1,
                color_scheme='seaborn1',
                separate_metrics=args.separate_metrics,
                gridline_intervals=None)#, # e.g., 'auto', None, [10]

        except Exception as e:
            print(f'[{job_name}] KDE violins error: {e}')

    # Confidence bars
    if args.conf_bars == 1:
        try:
            # Pocket
            confidence_bars_kde(kde_results_pocket, metrics_names_dict, save_dir=mirrored_jobdir_plots,
                                save_formats=['svg'], show_plot=args.show_plots, supertitle=0, filename_suffix='_pocket', transparency=0)
            # Chain
            confidence_bars_kde(kde_results_chain, metrics_names_dict, save_dir=mirrored_jobdir_plots,
                                save_formats=['svg'], show_plot=args.show_plots, supertitle=0, filename_suffix='_chain', transparency=0)
        except Exception as e:
            print(f'[{job_name}] Confidence bars error: {e}')

    # Save text files (statistics, output summary)
    if args.save_statistics == 1:
        # Rich descriptive stats (consumed downstream by do_collect_postcomputed_statistics) - keep.
        save_dict_to_json(comparisons_pocket, mirrored_jobdir_stats, 'apo_holo_all_stats_pocket.json')
        save_dict_to_json(comparisons_chain, mirrored_jobdir_stats, 'apo_holo_all_stats_chain.json')
        # Headline KDE detail (per-metric P/densities; source for the confidence bars) - keep.
        save_dict_to_json(kde_results_pocket, mirrored_jobdir_stats, 'classification_kde_pocket.json')
        save_dict_to_json(kde_results_chain, mirrored_jobdir_stats, 'classification_kde_chain.json')
        # NEW: one compact all-methods summary + consensus. Replaces the 8 per-method JSONs by default.
        save_dict_to_json(classification_summary, mirrored_jobdir_stats, 'classification_summary.json')

        # Optional verbose per-method JSONs (legacy layout). Off by default - their verdicts are in
        # classification_summary.json and the full descriptive stats are in apo_holo_all_stats_*.json.
        if int(getattr(args, "save_per_method_json", 0)) == 1:
            save_dict_to_json(classification_z_pocket, mirrored_jobdir_stats, 'classification_z_pocket.json')
            save_dict_to_json(classification_z_chain, mirrored_jobdir_stats, 'classification_z_chain.json')
            save_dict_to_json(classification_robust_pocket, mirrored_jobdir_stats, 'classification_robust_pocket.json')
            save_dict_to_json(classification_robust_chain, mirrored_jobdir_stats, 'classification_robust_chain.json')
            save_dict_to_json(classification_composite_pocket, mirrored_jobdir_stats, 'classification_composite_pocket.json')
            save_dict_to_json(classification_composite_chain, mirrored_jobdir_stats, 'classification_composite_chain.json')
            save_dict_to_json(classification_dirichlet_pocket, mirrored_jobdir_stats, 'classification_dirichlet_pocket.json')
            save_dict_to_json(classification_dirichlet_chain, mirrored_jobdir_stats, 'classification_dirichlet_chain.json')

        save_dict_to_json(closest_pocket, mirrored_jobdir_stats, 'closest_pocket.json') # closest_pocket_json =
        save_dict_to_json(closest_chain,  mirrored_jobdir_stats, 'closest_chain.json') # closest_chain_json  =
        save_dict_to_json(closest_all,    mirrored_jobdir_stats, 'closest_all.json') # closest_all_json    =
        #compress_file(closest_pocket_json, delete_original=args.del_dir_after_zip)
        #compress_file(closest_chain_json,  delete_original=args.del_dir_after_zip)
        #compress_file(closest_all_json,    delete_original=args.del_dir_after_zip)

    # TODO save i) summary of processing and classification result

    # Save overlaps/separations?

    # Zip 'plots' & 'statistics' folders and delete originals (optional)
    if args.zip_plots_dir == 1:
        compress_directory(mirrored_jobdir_plots, delete_original=args.del_dir_after_zip) # zip_file_plots = 
    if args.zip_stats_dir == 1:
        compress_directory(mirrored_jobdir_stats, delete_original=args.del_dir_after_zip) # zip_file_stats = 


    # Resolution + AlphaFold model pocket pLDDT (the previous full run lacked these and needed a
    # re-parse). df still carries all roles (Q/M present), with bfac_avg/resolution already cleaned.
    model_query_meta = extract_model_pLDDT_and_query_meta(df, meta_role=getattr(args, 'query_meta_role', 'Q'),
                                                          model_row=profile.model_row)
    # RUN_TYPES items 7 and 10: with the model as the anchor its pocket_rms is 0 by construction and the
    # query has no resolution or experimental method ('-'). Blank them rather than report constants.
    if profile.delta_anchor == 'model':
        model_query_meta['model_pocket_rms'] = np.nan
        model_query_meta['query_resolution'] = np.nan
        model_query_meta['query_exp_method'] = None

    # Chain distance-outlier flag. pocket_dist is no longer a top-level filter (AHOJ-003 decision):
    # it is a GLOBAL displacement, not a local-fit signal, and the pocket classification never uses
    # it. We keep it as a chain metric and FLAG entries whose candidates include extreme global
    # motions, so the SECONDARY chain classification can be marked downstream instead of dropping
    # candidates (which would hurt the local pocket classification).
    _cand_dist = (pd.to_numeric(df_c.loc[df_c['role'] == 'C', 'pocket_dist'], errors='coerce').dropna()
                  if 'pocket_dist' in df_c.columns else pd.Series(dtype=float))
    _dthr = float(getattr(args, 'dist_outlier_thr', 20.0))
    dist_flags = {
        'chain_pocket_dist_max': round(float(_cand_dist.max()), 3) if not _cand_dist.empty else np.nan,
        'chain_n_dist_outliers': int((_cand_dist > _dthr).sum()),
        'chain_dist_outlier_thr': _dthr,
    }

    # Ligand codes binding this pocket: the query pocket's ligand(s) and the non-redundant union
    # across candidate pockets (the "known ligands" observed in equivalent pockets across the PDB).
    # The query ligand is also encoded in job_name, but recording both here aids downstream
    # integration and a redock-known-ligands experiment (see KNOWN_ISSUES.md AHOJ-015).
    # RUN_TYPES item 11: for a residue-set query (afc-2.4) the query row's ligand columns list residue
    # names, so query_ligands is blank; and candidate_ligands is taken from HOLO rows only, because
    # gate-rejected entities stay listed in nonquery_ligs.
    _is_set_query = bool(len(df_q)) and 'POI_type' in df_q.columns and str(df_q['POI_type'].iloc[0]) == 'set'
    if _is_set_query:
        _q_ligs, _q_n = set(), 0
        _c_ligs, _c_n = collect_unique_ligands(df_c[(df_c['role'] == 'C') & df_c['apoholo_assignment'].isin(rt.HOLO_LABELS)])
    else:
        _q_ligs, _q_n = collect_unique_ligands(df_q)
        _c_ligs, _c_n = collect_unique_ligands(df_c.loc[df_c['role'] == 'C'])
    ligand_summary = {
        'query_ligands': '-'.join(sorted(_q_ligs)) if _q_ligs else None,
        'n_query_ligands': _q_n,
        'candidate_ligands': '-'.join(sorted(_c_ligs)) if _c_ligs else None,
        'n_candidate_ligands': _c_n,
    }

    # --- create summary row(s) ---
    # AHOJ-015: optionally DETACH the descriptive metric-stats block (compute_metrics: volume_mean,
    # RoG_iqr, deltas, flexibility, ...) into its own summary file so the per-entry CLASSIFICATION
    # summary stays lean. Both files share job_name (+ query_path) as the join key. With
    # separate_metric_summary=0 the block stays inline in the classification file (legacy behaviour).
    _split = int(getattr(args, "separate_metric_summary", 1)) == 1
    query_summary_df = create_query_summary_row(args=args,job_name=job_name,query_path=query_path,
        df_q=df_q,n_apo=n_apo,n_holo=n_holo,n_all=n_total,
        kde_results_pocket=kde_results_pocket,kde_results_chain=kde_results_chain,
        query_metric_summary_df=(None if _split else query_metric_summary_df),
        classification_summary=classification_summary,
        model_query_meta=model_query_meta,
        dist_flags=dist_flags,
        ligand_summary=ligand_summary,
        label_summary=label_summary,
        has_model=has_model,
        provenance=rt.provenance_columns(profile, script_params, df, site, job_key),
        chain_axes_meaningful=profile.chain_axes_meaningful)

    # Detached metric-stats row (empty when not splitting -> the block already lives in query_summary_df).
    metrics_summary_df = pd.DataFrame()
    if _split and query_metric_summary_df is not None and not query_metric_summary_df.empty:
        _meta = {'job_name': job_name,
                 'query_path': os.path.join(*query_path.split(os.path.sep)[-4:])}
        metrics_summary_df = pd.concat(
            [pd.DataFrame([_meta]), query_metric_summary_df.reset_index(drop=True)], axis=1)

    return ('Success', (job_name, len(df), apo_summary, holo_summary, query_summary_df, metrics_summary_df))
    #return ('Success', (job_name, len(df), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    #return ('Success', (job_name, len(df), apo_summary, holo_summary, query_info_df))


def has_minimum_apo_holo(candidate_dataframe, min_apo=1, min_holo=1, min_apoholo=None):
    """
    Returns (result, num_apo, num_holo, total) where result is True if criteria are met,
    num_apo is the number of apo rows, num_holo is number of holo rows, total is sum.
    """
    if candidate_dataframe is None or candidate_dataframe.empty:
        return (False, 0, 0, 0)
    apo_mask = candidate_dataframe['apoholo_assignment'].isin(rt.APO_LABELS)  # AHOJ-001, AHOJ-020
    holo_mask = candidate_dataframe['apoholo_assignment'].isin(rt.HOLO_LABELS)
    num_apo = apo_mask.sum()
    num_holo = holo_mask.sum()
    total = num_apo + num_holo
    # Minimum checks
    apo_ok = (min_apo is None) or (num_apo >= min_apo)
    holo_ok = (min_holo is None) or (num_holo >= min_holo)
    apoholo_ok = (min_apoholo is None) or (total >= min_apoholo)
    result = apo_ok and holo_ok and apoholo_ok
    return (result, num_apo, num_holo, total)


def audit_candidate_labels(df, job_name):
    """
    AHOJ-020: count the candidate rows whose apoholo_assignment parse-ahoj does not know
    (rt.KNOWN_CANDIDATE_LABELS) and warn about them. Those rows join neither the apo nor the holo set;
    before A- was added to APO_LABELS they vanished with no trace. Returns the summary-row columns.
    """
    unknown = rt.unknown_candidate_labels(df)
    n_unknown = int(sum(unknown.values()))
    if n_unknown:
        print(f'[WARN] [{job_name}] {n_unknown} candidate row(s) carry an apoholo_assignment parse-ahoj does not '
              f'know ({rt.format_label_counts(unknown)}); they join neither the apo nor the holo set '
              f'(apo {" ".join(rt.APO_LABELS)}, holo {" ".join(rt.HOLO_LABELS)}, unpooled {" ".join(rt.UNPOOLED_LABELS)})')
    return {'n_unknown_label': n_unknown, 'unknown_labels': rt.format_label_counts(unknown)}


def create_query_summary_row(
    args, 
    job_name, 
    query_path, 
    df_q, 
    n_apo, 
    n_holo, 
    n_all, 
    kde_results_pocket,
    kde_results_chain,
    query_metric_summary_df=None,
    classification_summary=None,
    model_query_meta=None,
    dist_flags=None,
    ligand_summary=None,
    label_summary=None,
    has_model=True,
    provenance=None,
    chain_axes_meaningful=True
):
    # Start with dictionary for the row
    row = {
        "job_name": job_name,
        "query_path": os.path.join(*query_path.split(os.path.sep)[-4:]), #query_path,
        # You can add more args or script params here if needed
        "n_apo": n_apo,
        "n_holo": n_holo,
        "n_all": n_all,
    }

    # RUN_TYPES item 12: which AHoJ version and schema wrote the archive, which rules applied, and the
    # query triple (query_mode / query_anchor / query_model; None on 2.1 archives).
    if provenance:
        row.update(provenance)

    # Add fields from query dataframe (usually 1 row, but guard for empty)
    for col in ['chains3', 'UNPs', '#resis3']:
        row[col] = df_q.iloc[0][col] if (not df_q.empty and col in df_q.columns) else None

    # AHOJ-015: Query pocket geometry (role Q). Anchors the *_delta columns (deltas are abs(candidate-query),
    # uninterpretable without the reference) and lets absolute candidate values be recovered.
    # The query residue count is already captured as '#resis3' above; query 'pocket_len' is NaN
    # (the query is not aligned to itself), so only RoG/volume/sasa are recorded here.
    for col in ['RoG', 'volume', 'sasa']:
        row[f'query_{col}'] = df_q.iloc[0][col] if (not df_q.empty and col in df_q.columns) else None

    # Query resolution / experimental method + AlphaFold model pocket-average pLDDT.
    # Added to the summary so the next full run does not need a separate re-parse for these.
    if model_query_meta:
        row['query_resolution'] = model_query_meta.get('query_resolution')
        row['query_exp_method'] = model_query_meta.get('query_exp_method')
        row['model_pocket_pLDDT_avg'] = model_query_meta.get('model_pocket_pLDDT_avg')
        row['n_model_rows'] = model_query_meta.get('n_model_rows')
        # AlphaFold model pocket geometry (the pocket a docking run would target).
        row['model_pocket_volume'] = model_query_meta.get('model_pocket_volume')
        row['model_pocket_sasa'] = model_query_meta.get('model_pocket_sasa')
        row['model_pocket_RoG'] = model_query_meta.get('model_pocket_RoG')
        row['model_pocket_len'] = model_query_meta.get('model_pocket_len')
        # AHOJ-019: the fourth classifier metric; see extract_model_pLDDT_and_query_meta for the
        # caveat that this one is a distance from the query, not an intrinsic pocket property.
        row['model_pocket_rms'] = model_query_meta.get('model_pocket_rms')

    # AHOJ-018: explicit has-model flag. 1 = a role 'M' (AlphaFold model) row was present and the
    # apo/holo classification below is meaningful; 0 = no model row (kept via keep_no_model=1) so the
    # conformational analysis/plots are populated but the classification verdicts are "N/A (no model)".
    # Downstream loaders should key off this flag rather than parsing the verdict strings.
    row['has_model'] = int(bool(has_model))

    # Chain distance-outlier flag (pocket_dist is no longer filtered; see AHOJ-003).
    if dist_flags:
        row.update(dist_flags)

    # Non-redundant ligand codes binding this pocket (query scope vs candidate scope), for
    # downstream integration and a possible redock-known-ligands experiment.
    if ligand_summary:
        row.update(ligand_summary)

    # AHOJ-020: A- rows among the pooled apo candidates, and the candidate labels that joined no set.
    if label_summary:
        row.update(label_summary)

    # Add flat candidate metric summaries, including delta/range-like columns.
    if query_metric_summary_df is not None and not query_metric_summary_df.empty:
        row.update(query_metric_summary_df.iloc[0].to_dict())

    # Add pocket metrics KDE overall statistics
    pocket_overall = kde_results_pocket.get('overall', {})
    for key in [
        "avg_P(apo)", "avg_P(holo)", "overall_score",
        "avg_normalized_density_strongest", "avg_normalized_density_combined",
        "overall_overlap", "overall_mode_overlap", "overall_classification"
    ]:
        row[f'pocket_{key}'] = pocket_overall.get(key, None)

    # Add chain metrics KDE overall statistics
    chain_overall = kde_results_chain.get('overall', {})
    for key in [
        "avg_P(apo)", "avg_P(holo)", "overall_score",
        "avg_normalized_density_strongest", "avg_normalized_density_combined",
        "overall_overlap", "overall_mode_overlap", "overall_classification"
    ]:
        row[f'chain_{key}'] = chain_overall.get(key, None)

    # All-methods verdicts + consensus (flat). Per scope: each method's class, the KDE/Dirichlet
    # primary scores, and a consensus class/fraction for downstream confidence filtering (F5).
    if classification_summary:
        for scope in ("pocket", "chain"):
            sc = classification_summary.get(scope, {}) or {}
            for meth in ("kde", "dirichlet", "modz", "robust", "composite"):
                row[f'{scope}_{meth}_class'] = (sc.get(meth) or {}).get("class")
            row[f'{scope}_dirichlet_score'] = (sc.get("dirichlet") or {}).get("score")
            cons = sc.get("consensus", {}) or {}
            row[f'{scope}_consensus_class'] = cons.get("class")
            row[f'{scope}_consensus_frac'] = cons.get("frac")
            row[f'{scope}_n_methods'] = cons.get("n_methods")

    # AHOJ-018: for a no-model entry the classifiers degrade to "Undefined"/"Insufficient Data";
    # relabel the two HEADLINE verdicts + the consensus classes to an unambiguous "N/A (no model)" so
    # the label is self-describing even without joining on has_model. Numeric P(apo)/P(holo)/score and
    # the per-method diagnostic columns stay as-is (already NaN / degraded).
    if not has_model:
        for _c in ("pocket_overall_classification", "chain_overall_classification",
                   "pocket_consensus_class", "chain_consensus_class"):
            if _c in row:
                row[_c] = "N/A (no model)"

    # RUN_TYPES item 7: when the model is the anchor its chain metrics (pocket_dist, tm_score, rmsd)
    # are 0/1/0 by construction, so the chain verdict describes nothing. Keep the numbers, mark the
    # headline verdicts.
    row['chain_axes_meaningful'] = int(bool(chain_axes_meaningful))
    if has_model and not chain_axes_meaningful:
        for _c in ("chain_overall_classification", "chain_consensus_class"):
            if _c in row:
                row[_c] = "N/A (model is anchor)"

    # Return as DataFrame (single row)
    return pd.DataFrame([row])


def do_collect_postcomputed_statistics(args, job_name, query_path):
    """
    Collect flattened post-computed statistics JSONs from a mirrored stats root.

    The batch processor still discovers jobs from args.root_data_dir. This function
    maps each discovered DB job to args.stats_root_dir using the same batch/job
    relative path and strips archive suffixes such as .tgz.
    """
    try:
        scopes = [str(scope) for scope in getattr(args, "stats_scopes", ["pocket", "chain"])]
        groups = [str(group) for group in getattr(args, "stats_groups", ["All"])]
        group_fields = [str(field) for field in getattr(args, "stats_group_fields", [])]
        metric_fields = [str(field) for field in getattr(args, "stats_metric_fields", [])]
        include_nested = int(getattr(args, "stats_include_nested", 0)) == 1

        stats_job_dir = resolve_mirrored_job_dir(
            getattr(args, "root_data_dir", None),
            getattr(args, "stats_root_dir", None),
            query_path,
        )

        out = {
            "job_name": job_name,
            "query_path": shorten_path(query_path),
            "stats_job_dir": shorten_path(stats_job_dir),
            "stats_groups": " ".join(groups),
            "stats_scopes": " ".join(scopes),
            "status": "pending",
        }

        files_found = 0
        files_missing = 0

        for scope in scopes:
            stats_filename = f"apo_holo_all_stats_{scope}.json"
            stats_path = os.path.join(stats_job_dir, "statistics", stats_filename)
            status_key = f"stats_{scope}_status"
            metrics_key = f"stats_{scope}_n_metrics"

            if not os.path.exists(stats_path):
                out[status_key] = "missing"
                out[metrics_key] = 0
                files_missing += 1
                continue

            try:
                with open(stats_path, "r", encoding="utf-8") as handle:
                    stats_data = json.load(handle)
                files_found += 1
                out[status_key] = "ok"
                out[metrics_key] = len(stats_data) if isinstance(stats_data, dict) else 0
                out.update(
                    flatten_statistics_json(
                        stats_data,
                        scope,
                        groups,
                        group_fields,
                        metric_fields,
                        include_nested=include_nested,
                    )
                )
            except Exception as e:
                out[status_key] = f"error:{type(e).__name__}"
                out[f"stats_{scope}_error"] = str(e)

        out["stats_files_found"] = files_found
        out["stats_files_missing"] = files_missing
        out["status"] = "ok" if files_found > 0 else "statistics_missing"

        if files_found == 0:
            return ("File Not Found", (job_name, 0, pd.DataFrame(), pd.DataFrame(), out))

        return ("Success", (job_name, files_found, pd.DataFrame(), pd.DataFrame(), out))

    except Exception as e:
        print(f"[ERROR] Exception in do_collect_postcomputed_statistics for {job_name}: {e}")
        print_exception(e)
        out = {
            "job_name": job_name,
            "query_path": shorten_path(query_path),
            "status": f"error:{type(e).__name__}",
            "error": str(e),
        }
        return ("Error", (job_name, str(e), pd.DataFrame(), pd.DataFrame(), out))



######################## UNP residue index extraction #########################

def load_poi_dict_unp_from_tgz(tgz_path, zip_name="residue_mappings.zip", json_name="poi_dict_unp.json"):
    """
    Extracts poi_dict_unp.json from residue_mappings.zip inside a .tgz archive.
    Returns the parsed JSON as a Python dict.
    """
    with tarfile.open(tgz_path, "r:gz") as tar:
        # Remove leading './' if present in tar paths
        members = {name.lstrip('./'): name for name in tar.getnames()}
        if zip_name not in members:
            raise FileNotFoundError(f"{zip_name} not found in archive")
        zip_member = tar.getmember(members[zip_name])
        with tar.extractfile(zip_member) as zip_bytes:
            with zipfile.ZipFile(io.BytesIO(zip_bytes.read())) as zf:
                if json_name not in zf.namelist():
                    raise FileNotFoundError(f"{json_name} not found in {zip_name}")
                with zf.open(json_name) as jf:
                    return json.load(jf)


def summarize_binding_site(poi_dict):
    """Extracts residue-level and summary information for one binding site dict."""
    # Collect all keys and values from the mapping (could be more than one site per JSON, see outer dict)
    all_keys = []
    all_values = []
    for mapping in poi_dict.values():
        all_keys.extend(mapping.keys())
        all_values.extend(mapping.values())

    # Extract unique chain identifiers from keys (e.g., "A_A_ASN_101" → "A" is the second element)
    prot_chains = sorted(set(k.split('_')[1] for k in all_keys))
    prot_chains_num = len(prot_chains)

    # Extract unique UniProt accessions from values (e.g., "P81947_101" → "P81947")
    UNPs = sorted(set(v.split('_')[0] for v in all_values))
    UNPs_num = len(UNPs)

    return prot_chains, prot_chains_num, UNPs, UNPs_num



def do_collect_uniprot_residues(args, job_name, query_path):
    try:
        # Determine if it's a tar archive or directory
        if str(query_path).endswith('.tgz'):
            poi_dict = load_poi_dict_unp_from_tgz(query_path)
        else:
            # Insert the directory-based loader from before (not shown here)
            raise NotImplementedError("Directory case not implemented in this snippet.")

        # Parse attributes from job_name (RUN_TYPES item 16: holo <pdb>-<chain>-<lig>-<resi> (or '_') or afc
        # <UniProt>_ST_100_<n>; anything else raises RunTypeError -> 'Error', not a silent ValueError).
        jk = rt.parse_job_key(job_name)
        pdb, chain, ligand, ligand_index = jk.pdb, jk.chain, jk.lig, jk.resi

        # For each binding site in the JSON (often just one top-level key, but support multiple)
        all_results = []
        for site_id, mapping in poi_dict.items():
            all_uniprot_residues = list(mapping.values()) # Allow redundancy

            # Get prot_chains and UNPs summary
            prot_chains, prot_chains_num, UNPs, UNPs_num = summarize_binding_site({site_id: mapping})

            result_dict = {
                "job_name": job_name,
                #"site_id": site_id,
                "key_kind": jk.kind,
                "site_uid": jk.site_uid,
                "site_uniprot": jk.uniprot,
                "pdb": pdb,
                "lig_chain": chain,
                "lig_id": ligand,
                "lig_index": ligand_index,
                "prot_chains": " ".join(prot_chains),
                "prot_chains_num": prot_chains_num,
                "UNPs": " ".join(UNPs),
                "UNPs_num": UNPs_num,
                "num_binding_residues": len(all_uniprot_residues),
                #"mapping_detail": mapping,  # PDB res idx
                "uniprot_residues": all_uniprot_residues  # UniProt res idx
            }

            all_results.append(result_dict)
            #print(f"\nDictionary of entry {job_name}:\n{result_dict}")

        # You can return all results as a list for aggregation
        return ('Success', (job_name, len(all_results), pd.DataFrame(), pd.DataFrame(), all_results))

    except Exception as e:
        print(f"[ERROR] Exception in {job_name}: {e}")
        import traceback; print(traceback.format_exc())
        return ('Error', (job_name, 0, pd.DataFrame(), pd.DataFrame(), []))


def do_collect_model_pLDDT(args, job_name, query_path):
    """
    Collect pocket-average pLDDT for the model row(s) (role == 'M') from global_results.csv.

    - In your dataset, model pLDDT is stored in column 'bfac_avg' for role == 'M'.
    - No per-entry file writing (important for very large DBs).
    - Returns a compact summary row (dict) to be aggregated into query_summaries_do_collect_model_pLDDT.csv.

    Status behavior:
      - global_results.csv missing -> status='global_results_missing'
      - model row missing          -> status='no_model_row'
      - multiple model rows        -> status='multiple_model_rows' (anomaly)
      - missing bfac_avg column    -> status='missing_bfac_avg_column'
      - empty dataframe            -> status='empty_global_results'
    """
    try:
        # Both tar member conventions; AHoJ 2.6.0 names renamed to parse-ahoj's (rt.load_global_results)
        df = rt.load_global_results(query_path)

        # helper: shorten query_path for logging/triage
        try:
            query_path_short = os.path.join(*str(query_path).split(os.path.sep)[-4:])
        except Exception:
            query_path_short = str(query_path)

        # Base output row (always returned, even on missing file)
        out = {
            "job_name": job_name,
            "query_path": query_path_short,
            "model_pocket_pLDDT_avg": np.nan,
            "n_model_rows": 0,
            "status": None,

            # AHOJ-019: AlphaFold model pocket geometry, harvested from the same role 'M' row(s).
            # Mirrors extract_model_pLDDT_and_query_meta so this lightweight collector can refresh
            # the model columns for a whole database WITHOUT re-running KDE/classification.
            "model_pocket_volume": np.nan,
            "model_pocket_sasa": np.nan,
            "model_pocket_RoG": np.nan,
            "model_pocket_len": np.nan,
            "model_pocket_rms": np.nan,

            # query-structure metadata (usually role == 'Q')
            "query_resolution": np.nan,
            "query_exp_method": None,
            "n_query_rows": 0,
            "query_meta_status": None,
        }

        if df is None:
            out["status"] = "global_results_missing"
            # Return as File Not Found so your existing category logging stays correct
            return ("File Not Found", (job_name, 0, pd.DataFrame(), pd.DataFrame(), out))

        if df.empty or (df.replace("", np.nan).isnull().all(axis=None)):
            out["status"] = "empty_global_results"
            return ("Discarded", (job_name, "Empty global_results.csv dataframe", pd.DataFrame(), pd.DataFrame(), out))

        # Optional: reuse your standard zero->nan cleanup if configured
        if hasattr(args, "columns_nozero") and args.columns_nozero:
            df = replace_zero_with_nan(df, args.columns_nozero)

        if "role" not in df.columns:
            out["status"] = "missing_role_column"
            return ("Success", (job_name, 1, pd.DataFrame(), pd.DataFrame(), out))

        # --- OPTIONAL: collect query metadata (resolution, exp_method) ---
        # Use getattr so you DON'T have to add CLI args immediately.
        # Defaults: enabled + role 'Q' (experimental/query structure).
        if int(getattr(args, "collect_query_meta", 1)) == 1:
            meta_role = getattr(args, "query_meta_role", "Q")

            df_qmeta = df[df["role"] == meta_role].copy()
            out["n_query_rows"] = int(len(df_qmeta))

            if len(df_qmeta) == 0:
                out["query_meta_status"] = f"no_role_{meta_role}_row"
            else:
                out["query_meta_status"] = "ok" if len(df_qmeta) == 1 else "multiple_query_rows"

                # resolution
                if "resolution" in df_qmeta.columns:
                    res = pd.to_numeric(df_qmeta["resolution"], errors="coerce")
                    res = res.dropna()
                    if not res.empty:
                        out["query_resolution"] = float(res.iloc[0])
                else:
                    out["query_meta_status"] += "|missing_resolution"

                # exp_method
                if "exp_method" in df_qmeta.columns:
                    em = df_qmeta["exp_method"].astype("string").str.strip()
                    em = em[em.notna() & (em != "")]
                    if len(em) > 0:
                        out["query_exp_method"] = str(em.iloc[0])
                else:
                    out["query_meta_status"] += "|missing_exp_method"


        # RUN_TYPES item 6: the query's own model row(s) only (rt.select_model_rows, 'AF_model' by default).
        df_m = rt.select_model_rows(df, getattr(args, "model_row", None) or "AF_model").copy()
        out["n_model_rows"] = int(len(df_m))
        out["ahoj_version"] = rt.ahoj_version(rt.load_script_parameters(query_path))
        out["ahoj_schema"] = rt.schema_of(df)
        out.update(rt.query_triple(df))

        if out["n_model_rows"] == 0:
            out["status"] = "no_model_row"
            return ("Success", (job_name, 1, pd.DataFrame(), pd.DataFrame(), out))

        # AHOJ-019: harvest the model pocket geometry BEFORE the bfac_avg guard below, so an entry
        # missing pLDDT still contributes its geometry rather than returning an all-NaN row.
        for _col, _key in (("volume", "model_pocket_volume"), ("sasa", "model_pocket_sasa"),
                           ("RoG", "model_pocket_RoG"), ("pocket_len", "model_pocket_len"),
                           ("pocket_rms", "model_pocket_rms")):
            if _col in df_m.columns:
                _v = pd.to_numeric(df_m[_col], errors="coerce")
                if _v.notna().any():
                    out[_key] = round(float(np.nanmean(_v)), 3)

        if "bfac_avg" not in df_m.columns:
            out["status"] = "missing_bfac_avg_column"
            return ("Success", (job_name, 1, pd.DataFrame(), pd.DataFrame(), out))

        vals = pd.to_numeric(df_m["bfac_avg"], errors="coerce").astype(float)

        if out["n_model_rows"] > 1:
            # anomaly: multiple model rows
            out["status"] = "multiple_model_rows"
            # keep a small breadcrumb for debugging (optional)
            out["model_pocket_pLDDT_values"] = [round(x, 3) if pd.notna(x) else np.nan for x in vals.tolist()]
            out["model_pocket_pLDDT_avg"] = float(np.nanmean(vals)) if vals.notna().any() else np.nan
        else:
            out["status"] = "ok"
            out["model_pocket_pLDDT_avg"] = float(vals.iloc[0]) if pd.notna(vals.iloc[0]) else np.nan

        if pd.notna(out["model_pocket_pLDDT_avg"]):
            out["model_pocket_pLDDT_avg"] = round(out["model_pocket_pLDDT_avg"], 3)

        return ("Success", (job_name, 1, pd.DataFrame(), pd.DataFrame(), out))

    except Exception as e:
        print(f"[ERROR] Exception in do_collect_model_pLDDT for {job_name}: {e}")
        print_exception(e)
        # Still return a row so it can show up in the aggregate CSV if you adopt the process_jobs tweak below
        out = {
            "job_name": job_name,
            "query_path": str(query_path),
            "model_pocket_pLDDT_avg": np.nan,
            "n_model_rows": np.nan,
            "status": f"error:{type(e).__name__}",
        }
        return ("Error", (job_name, str(e), pd.DataFrame(), pd.DataFrame(), out))


# Default metric set for the lightweight apo/holo metric-stats collector. Superset of the
# classifier's pocket metrics (pocket_rms, sasa, volume, RoG) plus the extra descriptors the
# original SI apo/holo %-difference figure used (bfac_avg, pocket_dist, rmsd, tm_score, resolution,
# sasa_avg). Overridable via args.metric_stats_metrics (e.g. from a config JSON).
APOHOLO_METRIC_STATS_DEFAULT = [
    "pocket_rms", "sasa", "sasa_avg", "volume", "RoG",
    "bfac_avg", "pocket_dist", "rmsd", "tm_score", "tm_score_i", "resolution",
]


def do_collect_apoholo_metric_stats(args, job_name, query_path):
    """
    Lightweight apo-vs-holo metric-statistics collector (no KDE / DP / plots / classification).

    Purpose: produce, per entry, the *apo-vs-holo* statistics that an apo/holo percentage-difference
    figure needs - crucially the per-entry diff_means = |mean_apo - mean_holo|, computed WITHIN each
    entry (absolute), so it does NOT cancel when aggregated across the DB. Every metric (RoG included)
    is treated identically, via the same compute_absolute_statistics_v2 used by the production
    classifier (methodological fidelity).

    NOTE: this is NOT the runtime "RoG distance from query" sanity delta (abs(RoG_candidate -
    RoG_query), used to catch exploding pockets). It is the between-state (apo vs holo) mean
    difference, analogous to how volume/sasa are handled.

    Reuses the standard load -> curate -> filter -> apo/holo split framework of
    do_kde_plots_and_classification, then emits ONE flat row per entry:
        job_name, query_path, n_apo, n_holo, n_all, paired, chains3, UNPs, #resis3,
        query_RoG, query_volume, query_sasa, query_resolution,
        and for each metric m present:
            <m>_apo_mean, <m>_holo_mean, <m>_all_mean,
            <m>_diff_means, <m>_cohen_d, <m>_overlap_coef, <m>_robust_sep
    No per-entry files are written (safe for the full DB). Rows aggregate into
    query_summaries_do_collect_apoholo_metric_stats.csv.
    """
    ### Load df
    try:
        df = rt.load_global_results(query_path)     # both member spellings; AHoJ 2.6.0 names renamed
        if df is None:
            return ("File Not Found", (job_name, 0))
    except Exception as e:
        print(f"[ERROR] Exception loading job {job_name} "
              f"[{os.path.join(*str(query_path).split(os.path.sep)[-4:])}]: {e}")
        print_exception(e)
        return ("Error", (job_name, str(e), pd.DataFrame(), pd.DataFrame(), {}))

    ### Curation (mirror do_kde_plots_and_classification, incl. the run-type profile)
    if df is None or df.empty or (df.replace("", np.nan).isnull().all(axis=None)):
        return ("Discarded", (job_name, f"Empty dataframe: {0 if df is None else len(df)}"))

    try:
        script_params = rt.load_script_parameters(query_path)
        profile = rt.resolve_profile(args, script_params, df, job_name)
        site = rt.site_for_job(job_name, profile.key_format)
        if profile.relabel_mode in ("v2c_ions", "v2c_ions_demote_only"):
            df, _changed = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=None, verbose=False,
                                                   protect_labels=set(profile.protect_labels),
                                                   demote_only=(profile.relabel_mode == "v2c_ions_demote_only"))
        elif profile.relabel_mode == "entity_gate_site":
            _mode, _thr = rt.gate_from_script_params(script_params)
            _gate = None if profile.gate == "none" or (profile.gate == "archive" and _mode != "gate") else _thr
            df, _changed = rt.relabel_by_entities(df, rt.load_ligands_json(query_path), _gate, rt.ion_set(),
                                                  rt.additive_set(getattr(args, "additives_file", None)),
                                                  profile.ion_rule, profile.additive_rule, site,
                                                  protect_labels=profile.protect_labels)
    except RunTypeError as e:
        print(f"[ERROR] run type: {e}")
        return ("Error", (job_name, f"RunTypeError: {e}", pd.DataFrame(), pd.DataFrame(), {}))

    label_summary = audit_candidate_labels(df, job_name)     # AHOJ-020

    df = replace_zero_with_nan(df, args.columns_nozero)

    # Volume column and the NaN -> 0 rule, as the profile says (same as the classifier)
    if profile.volume_column != "volume":
        df["volume"] = df[profile.volume_column]
    s = pd.to_numeric(df["#resis3"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v) if profile.volume_nan_to_zero else v

    df_q = df[df["role"] == "Q"]

    ### Candidate filtering (same framework; controlled by --filter_candidates)
    if int(getattr(args, "filter_candidates", 1)) == 1:
        df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=args.full_pockets_aligned,
                                  bndgrsds_prcnt_seqres=args.bndgrsds_prcnt_seqres,
                                  bndgrsds_prcnt_obs=args.bndgrsds_prcnt_obs, resolution_soft=args.resolution_soft,
                                  rog_diff_threshold_deltaQ=args.rog_diff_thrs_query,
                                  rog_diff_threshold_mean=args.rog_diff_thrs_mean, pocket_distance=args.pocket_dist,
                                  pocket_len_resis3=args.pocket_len_resis3,
                                  drop_no_resolution=getattr(args, "drop_no_resolution", 0))
    else:
        df_f = df.copy()

    df_c = df_f[df_f["role"] == "C"]

    # This figure is about PAIRED entries: require at least one apo and one holo candidate.
    min_ok, n_apo, n_holo, n_total = has_minimum_apo_holo(
        df_c,
        min_apo=max(1, int(getattr(args, "min_apo", 1) or 1)),
        min_holo=max(1, int(getattr(args, "min_holo", 1) or 1)),
        min_apoholo=getattr(args, "min_apoholo", None))
    if not min_ok:
        return ("Discarded", (job_name, f"Low candidates after filtering: apo={n_apo}, holo={n_holo}, total={n_total}"))

    apo_df = df_c[df_c["apoholo_assignment"].isin(rt.APO_LABELS)]   # AHOJ-001, AHOJ-020
    holo_df = df_c[df_c["apoholo_assignment"].isin(rt.HOLO_LABELS)]

    ### Metric set: default superset unioned with the classifier metrics; keep only present columns
    metrics = list(getattr(args, "metric_stats_metrics", None) or APOHOLO_METRIC_STATS_DEFAULT)
    for m in list(getattr(args, "pocket_class_metrics", []) or []) + list(getattr(args, "chain_class_metrics", []) or []):
        if m not in metrics:
            metrics.append(m)
    metrics = [m for m in metrics if m in df_c.columns]

    comparisons = compute_absolute_statistics_v2(apo_df, holo_df, metrics)

    ### Flatten to one row
    out = {
        "job_name": job_name,
        "query_path": os.path.join(*str(query_path).split(os.path.sep)[-4:]),
        "n_apo": int(n_apo), "n_holo": int(n_holo), "n_all": int(n_total),
        "paired": 1,
        "run_type": profile.name,
        "ahoj_version": rt.ahoj_version(script_params),
        "ahoj_schema": rt.schema_of(df),
        "rule_relabel_mode": profile.relabel_mode,
        "rule_volume_column": profile.volume_column,
        "n_apo_query_demoted": int((df_c["apoholo_assignment"] == "A-").sum()),   # AHOJ-020
    }
    out.update(label_summary)
    out.update(rt.query_triple(df))
    for col in ["chains3", "UNPs", "#resis3"]:
        out[col] = df_q.iloc[0][col] if (not df_q.empty and col in df_q.columns) else None
    for col in ["RoG", "volume", "sasa", "resolution"]:
        out[f"query_{col}"] = df_q.iloc[0][col] if (not df_q.empty and col in df_q.columns) else None

    for m, cmp in comparisons.items():
        apo_stats = cmp.get("Apo") or {}
        holo_stats = cmp.get("Holo") or {}
        all_stats = cmp.get("All") or {}
        out[f"{m}_apo_mean"] = apo_stats.get("mean")
        out[f"{m}_holo_mean"] = holo_stats.get("mean")
        out[f"{m}_all_mean"] = all_stats.get("mean")
        out[f"{m}_diff_means"] = cmp.get("diff_means")   # |mean_apo - mean_holo| (non-cancelling)
        out[f"{m}_cohen_d"] = cmp.get("cohen_d")
        out[f"{m}_overlap_coef"] = cmp.get("overlap_coef")
        out[f"{m}_robust_sep"] = cmp.get("robust_sep")

    return ("Success", (job_name, len(df), pd.DataFrame(), pd.DataFrame(), out))



