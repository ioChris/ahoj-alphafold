# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:12:01 2025

@author: Christos P. Feidakis
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

import pandas as pd
import numpy as np
import ast
import time
import logging
from logging.handlers import QueueHandler, QueueListener

import argparse
import warnings
import datetime
import multiprocessing

import sys

# Make `ubs_utils` importable when this file is loaded from another directory
# (e.g. by another pipeline via importlib / runpy).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Optional font helper from the classification code (00_classification/kde_classifier/stats_plot.py).
# It only affects figure rendering, so its absence must not stop table-only runs.
set_plot_font = None
for _cand in (os.path.join(os.path.dirname(os.path.dirname(_HERE)), '00_classification', 'kde_classifier'),):
    if os.path.isdir(_cand):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        try:
            from stats_plot import set_plot_font  # noqa: E402
            break
        except Exception:
            set_plot_font = None
if set_plot_font is None:
    def set_plot_font(*args, **kwargs):  # noqa: D103 - fallback no-op
        return None

from ubs_utils import (
    split_interface_entries,
    hierarchical_cluster_jaccard,
    save_pickle,
    save_ubs_table,
    save_annotated_ubs_dendrogram,
    compress_file,
    get_cluster_labels_from_linkage,
    save_uniprot_binding_histogram,
    write_uniprot_manifest,
)



def start_logging_with_queue(log_dir):
    """Start a single listener in the main process and route root logger through it."""
    os.makedirs(log_dir, exist_ok=True)
    datestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"ubs_unification_{datestamp}.log")

    # Create a queue and a listener with both console + file handlers
    log_queue = multiprocessing.Queue(-1)
    console = logging.StreamHandler()
    fileh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s %(message)s")
    console.setFormatter(fmt)
    fileh.setFormatter(fmt)
    listener = QueueListener(log_queue, console, fileh)
    listener.start()

    # Route the main process logger through the queue too
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    root.addHandler(QueueHandler(log_queue))

    logging.getLogger(__name__).info(f"Log file created at: {log_path}")
    return log_queue, listener


def configure_worker_logger(log_queue):
    """Call this at the top of each worker to send its logs to the main listener."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []  # avoid duplicate handlers if a worker is reused
    root.addHandler(QueueHandler(log_queue))


# --- NEW: multimeric status helpers ------------------------------------------------
def multimeric_status_code(prot_chains_num, unps_num):
    """
    Return numeric code:
      1 = monomer (prot=1 and unp=1)
      2 = multi-homomer (prot>=2 and unp==1)
      3 = multi-heteromer (prot>=2 and unp==prot)
      4 = mixed homo/hetero (prot>=2 and unp>=2 and unp!=prot)
    """
    try:
        p = int(prot_chains_num)
        u = int(unps_num)
    except Exception:
        return 0  # unknown
    if p == 1 and u == 1:
        return 1
    if p >= 2 and u == 1:
        return 2
    if p >= 2 and u == p:
        return 3
    if p >= 2 and u >= 2 and u != p:
        return 4
    return 0


def status_code_to_name(code: int) -> str:
    return {1: "monomer", 2: "homomer", 3: "heteromer", 4: "homo/hetero"}.get(code, "unknown")
# -----------------------------------------------------------------------------------


def process_uniprot(df_uniprot, uniprot_id, thresholds, ubs_dir, save_trees,# logger,
                     dendro_show_threshold: str = "none",
                     dendro_show_thresholds = None,
                     compression_method: str = "zip",
                     save_figures: bool = True):

    """
    dendro_show_threshold:
      "none" (default)
      "LJ"   (large-jump)
      "030","050","070","100" (show a specific Similarity Threshold line)

    save_figures:
      False -> figures-off mode: only the ST_XXX.tsv / ST_LJXXX.tsv tables, the
      compressed input_data.csv and the manifest are written (no dendrogram SVG,
      no binding-position histogram, no tree.pkl). Clustering is unchanged.
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Processing UniProt {uniprot_id} ({len(df_uniprot)} entries)")
        residue_sets = df_uniprot['uniprot_residues'].tolist()
        job_names = df_uniprot['job_name'].tolist()


        uniprot_dir = os.path.join(ubs_dir, uniprot_id)
        os.makedirs(uniprot_dir, exist_ok=True)

        # Manifest collectors
        manifest_thresholds = []
        manifest_figs = {}
        manifest_input = None
        n_jobs_uniprot = int(df_uniprot['job_name'].nunique())


        if len(residue_sets) == 0:
            logger.warning(f"No entries for {uniprot_id}. Skipping.")
            return

        # Build per-entry multimeric status mapping (for later aggregation into the TSVs)
        status_map = {}
        for _, r in df_uniprot[['job_name', 'prot_chains_num', 'UNPs_num']].iterrows():
            code = multimeric_status_code(r['prot_chains_num'], r['UNPs_num'])
            status_map[r['job_name']] = code

        # Ensure threshold 1.0 is included and sorted uniquely
        thresholds_to_run = sorted(set(list(thresholds) + [1.0]))
        Z = None
        large_jump_similarity = None

        # Only cluster if >1 entry
        if len(residue_sets) > 1:
            # Compute linkage tree ONCE (doesn't depend on threshold)
            _, Z = hierarchical_cluster_jaccard(residue_sets, threshold=thresholds_to_run[0])
            linkage_distances = Z[:, 2]
            jumps = np.diff(linkage_distances)
            if len(jumps) > 0:
                max_jump_idx = np.argmax(jumps)
                large_jump_distance = linkage_distances[max_jump_idx]
                large_jump_similarity = 1 - large_jump_distance
                # Always include the large-jump threshold *even if* it matches a default one
                thresholds_to_run.append(large_jump_similarity)

        # Save input_data.csv (provenance) once and compress it
        try:
            csv_path = os.path.join(uniprot_dir, 'input_data.csv')
            df_uniprot.to_csv(csv_path, index=False)
            compressed = compress_file(csv_path, method=compression_method, remove_original=True)
            manifest_input = os.path.basename(compressed)
        except Exception as e:
            logger.error(f"Failed saving input_data.csv for {uniprot_id}: {e}")

        for threshold in thresholds_to_run:
            try:
                if len(residue_sets) > 1:
                    distance_threshold = 1 - threshold
                    cluster_labels = get_cluster_labels_from_linkage(Z, distance_threshold)
                else:
                    cluster_labels = [1]  # Single entry → UBS 1

                clusters = {}
                for idx, ubs_id in enumerate(cluster_labels):
                    clusters.setdefault(ubs_id, {'job_names': [], 'residues': []})
                    clusters[ubs_id]['job_names'].append(job_names[idx])
                    clusters[ubs_id]['residues'].append(residue_sets[idx])

                cluster_rows = []
                for ubs_id, info in clusters.items():
                    merged_residues = set()
                    for residues in info['residues']:
                        merged_residues |= set(residues)
                    merged_residues = sorted(merged_residues)

                    if len(info['residues']) > 1:
                        common_residues = set(info['residues'][0]).intersection(*info['residues'][1:])
                    else:
                        common_residues = set(info['residues'][0])

                    # Aggregate multimeric status across entries in this UBS
                    status_list = [status_map.get(j, 0) for j in info['job_names']]
                    unique_statuses = sorted(set(status_list))
                    # mode (break ties by preferring higher code deterministically)
                    #if len(status_list) > 0:
                    #    vals, counts = np.unique(status_list, return_counts=True)
                    #    mode_code = int(vals[np.argmax(counts)])
                    #else:
                    #    mode_code = 0

                    cluster_rows.append({
                        'ubs_id': ubs_id,  # renaming from former "cluster_id"
                        'num_shared_residues': len(common_residues),
                        'num_total_residues': len(merged_residues),
                        'multimeric_status_all': " ".join(map(str, unique_statuses)), # NEW column #3
                        'num_entries': len(info['job_names']),
                        'entry_ids': info['job_names'],
                        'shared_residues': sorted(common_residues),
                        'total_residues': merged_residues
                        # NEW columns (request #3)
                        #'multimeric_status': mode_code,
                        #'multimeric_status_name': status_code_to_name(mode_code),
                    })

                cluster_rows_sorted = sorted(cluster_rows, key=lambda x: x['ubs_id'])
                df_clusters = pd.DataFrame(cluster_rows_sorted)

                # Always create both files if large jump overlaps a regular threshold
                is_large_jump = (large_jump_similarity is not None and np.isclose(threshold, large_jump_similarity, atol=1e-6))
                is_predefined = (any(np.isclose(threshold, t, atol=1e-6) for t in thresholds))

                # Predefined ST file (3-digit)
                if is_predefined:
                    suffix = f"{int(round(threshold*100)):03d}"
                    out_path = os.path.join(uniprot_dir, f"ST_{suffix}.tsv")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    save_ubs_table(df_clusters, out_path)

                    manifest_thresholds.append({
                        "JaccardSimilarityThreshold": float(threshold),
                        "label": f"ST_{suffix}",
                        "type": "fixed",
                        "file": os.path.basename(out_path),
                        "shortDescription": "Fixed similarity threshold (predefined).",
                        "longDescription": "Unified Binding Sites computed at a predefined Jaccard similarity threshold."
                    })

                # Large-jump ST file (always 3-digit)
                if is_large_jump:
                    lj_pct = int(round(large_jump_similarity*100))
                    lj_suffix = f"LJ{lj_pct:03d}"
                    #lj_suffix = f"LJ{int(round(large_jump_similarity*100)):03d}"
                    out_path = os.path.join(uniprot_dir, f"ST_{lj_suffix}.tsv")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    save_ubs_table(df_clusters, out_path)

                    manifest_thresholds.append({
                        "JaccardSimilarityThreshold": float(large_jump_similarity),
                        "label": f"ST_{lj_suffix}",
                        "type": "optimal",
                        "file": os.path.basename(out_path),
                        "shortDescription": "Optimal similarity threshold (large jump).",
                        "longDescription": "Automatically selected threshold based on the largest linkage jump in the dendrogram."
                    })

            except Exception as e:
                logger.error(f"Failed for threshold {threshold} in {uniprot_id}: {e}")

        # Save dendrogram and pickle only if clustering is possible (more than 1 entry)
        if save_figures and save_trees and Z is not None:
            try:
                tree_pkl_path = os.path.join(uniprot_dir, 'tree.pkl')
                dendro_svg_path = os.path.join(uniprot_dir, 'dendrogram.svg')
                save_pickle(Z, tree_pkl_path)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.cluster.hierarchy")
                    # NOTE: no default 0.7 line; axis shows SIMILARITY (%) and we can
                    # optionally show a specific threshold via 'dendro_show_threshold'
                    # Attach the 3-digit ST codes to Z so "auto" can use the exact set used here
                    try:
                        Z._st_codes = [f"{int(round(t*100)):03d}" for t in sorted(set(thresholds))]
                    except Exception:
                        pass

                    save_annotated_ubs_dendrogram(
                        Z,
                        job_names,
                        residue_sets,
                        save_path=dendro_svg_path,
                        show_threshold=dendro_show_threshold,        # legacy (single)
                        thresholds_to_mark=dendro_show_thresholds,   # NEW (multi)
                        large_jump_similarity=large_jump_similarity,
                        orientation='right',
                        legend_loc="upper right",     # <- fixed position inside the axes
                        legend_outside=False,          # <- set True to park it outside on the right
                        transparency=1,   # NEW: 0=opaque outside, 1=fully transparent outside

                    )

                compressed = compress_file(dendro_svg_path, method=compression_method, remove_original=False)
                manifest_figs["dendrogram"] = os.path.basename(compressed)

            except Exception as e:
                logger.error(f"Failed saving dendrogram/pickle for {uniprot_id}: {e}")


        # --- NEW: per-UniProt binding-residue histogram -----------------------
        if save_figures:
            try:
                hist_svg = os.path.join(uniprot_dir, 'binding_positions.svg')
                save_uniprot_binding_histogram(
                    df_uniprot=df_uniprot,
                    uniprot_id=uniprot_id,
                    save_path=hist_svg,
                    as_percent=True,            # % of jobs; set False for raw counts
                    show_interface_split=True,  # stacked monomer vs interface
                    figsize=(10, 3.2),
                    facecolor='white',
                    transparency=1
                )

                compressed = compress_file(hist_svg, method=compression_method, remove_original=False)
                manifest_figs["histogram"] = os.path.basename(compressed)

            except Exception as e:
                logger.error(f"Failed saving binding histogram for {uniprot_id}: {e}")
        # ----------------------------------------------------------------------

        # Write JSON manifest
        try:
            write_uniprot_manifest(
                out_dir=uniprot_dir,
                uniprot_id=uniprot_id,
                thresholds=manifest_thresholds,
                figures=manifest_figs,
                input_data_file=manifest_input,
                archive_format=compression_method,
                extra={
                    "numEntries": n_jobs_uniprot,
                    "defaultThresholds": [float(t) for t in thresholds],
                },
            )
        except Exception as e:
            logger.error(f"Failed writing manifest for {uniprot_id}: {e}")


    except Exception as e:
        logger.error(f"Critical failure for UniProt {uniprot_id}: {e}")

def init_worker(log_queue):
    configure_worker_logger(log_queue)          # route this process’s logs to the queue
    #set_plot_font(font_name='Helvetica Neue')   # optional: set font once per worker


def worker_process(args):
    (df_uniprot, uniprot_id, thresholds, ubs_dir,
     save_trees, dendro_show_threshold, dendro_show_thresholds, compression_method,
     save_figures) = args

    #set_plot_font(font_name='Helvetica Neue')
    process_uniprot(
         df_uniprot, uniprot_id, thresholds, ubs_dir, save_trees,
         dendro_show_threshold=dendro_show_threshold,
         dendro_show_thresholds=dendro_show_thresholds,
         compression_method=compression_method,
         save_figures=save_figures,
     )

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 1.0)
DEFAULT_COMPRESSION = "gz"          # "zip" or "gz"
DEFAULT_WORKERS = 1                 # single worker keeps runs reproducible / easy to debug


def filter_subset(df, subset):
    """
    Keep the rows that involve at least one accession of `subset`.

    `UNPs` may hold several space-separated accessions (interface rows); a row is
    kept when any of them is in `subset`, so the per-UniProt groups formed after
    `split_interface_entries` are exactly those of a full run restricted to
    `subset`.
    """
    wanted = set(subset)
    mask = df['UNPs'].astype(str).apply(lambda s: any(tok in wanted for tok in s.split()))
    return df[mask].copy()


def default_dendro_marks(thresholds):
    """3-digit ST codes for the thresholds below 1.0, plus the large-jump marker."""
    codes = [f"{int(round(t * 100)):03d}" for t in sorted(set(thresholds)) if t < 1.0 - 1e-9]
    return codes + ["LJ"]


def run_pipeline(
    input_file,
    out_dir,
    thresholds=DEFAULT_THRESHOLDS,
    workers=DEFAULT_WORKERS,
    subset=None,
    figures=True,
    compression=DEFAULT_COMPRESSION,
    log_dir=None,
    save_trees=True,
    dendro_show_threshold="none",          # legacy single marker
    dendro_show_thresholds=None,           # None -> derived from `thresholds` (+ "LJ")
):
    """
    Run the UBS unification for every UniProt in `input_file`, writing one
    directory per accession under `out_dir`.

    Parameters
    ----------
    input_file : str
        CSV with `job_name`, `UNPs`, `uniprot_residues`, `prot_chains_num`,
        `UNPs_num`, `pdb` (and optionally `ligands`).
    out_dir : str
        Root output directory (`<out_dir>/<ACC>/ST_XXX.tsv`, ...).
    thresholds : iterable of float
        Fixed Jaccard similarity thresholds, each in (0, 1]. 1.0 is always
        part of the linkage cut set; ST_100.tsv is written only when 1.0 is
        listed here (it is by default). The large-jump cut is always added.
    workers : int
        Number of processes (1 = run in this process).
    subset : iterable of str or None
        Restrict the run to these accessions (off when None).
    figures : bool
        False = figures-off mode (tables, input_data and manifest only).
    compression : {"zip", "gz"}
        Archive format for input_data.csv and figures.
    log_dir : str or None
        Where to write the run log (defaults to `out_dir`).
    """
    thresholds = [float(t) for t in thresholds]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    for t in thresholds:
        if not (0.0 < t <= 1.0):
            raise ValueError(f"thresholds must be in (0, 1]; got {t}")
    if compression not in ("zip", "gz"):
        raise ValueError(f"compression must be 'zip' or 'gz'; got {compression!r}")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if dendro_show_thresholds is None:
        dendro_show_thresholds = default_dendro_marks(thresholds)

    ubs_dir = out_dir
    os.makedirs(ubs_dir, exist_ok=True)
    if log_dir is None:
        log_dir = ubs_dir

    # Logger
    log_queue, log_listener = start_logging_with_queue(log_dir)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    # Set plot font (figures only)
    if figures:
        try:
            set_plot_font(font_name='Helvetica Neue', svg_fonttype='path')
        except Exception as e:
            logger.warning(f"Could not set plot font: {e}")

    try:
        # --- Data Preparation ---
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, low_memory=False)

        # Optional sub-selection (debugging / partial runs)
        if subset:
            subset = list(subset)
            logger.info(f"Restricting to {len(subset)} UniProt accession(s): {' '.join(subset)}")
            df = filter_subset(df, subset)

        # Convert residue list string to actual list
        df['uniprot_residues'] = df['uniprot_residues'].apply(ast.literal_eval)

        # Split interface entries
        logger.info("Splitting interface entries")
        df = split_interface_entries(df)
        if subset:
            # After the split each row carries a single accession; drop the partners
            # of interface rows that are not themselves requested.
            df = df[df['UNPs'].isin(set(subset))].copy()

        # Group and prepare arguments
        args_list = [
          (group_df, uniprot_id, thresholds, ubs_dir, save_trees,
           dendro_show_threshold, dendro_show_thresholds, compression, figures)
          for uniprot_id, group_df in df.groupby('UNPs')]

        mode = "tables only (no figures)" if not figures else "tables + figures"

        # Do processing
        if workers == 1:
            logger.info(f"Processing {len(args_list)} UniProts with {len(df)} entries with single process ({mode}).")
            for args in args_list:
                worker_process(args)
        else:
            logger.info(f"Processing {len(args_list)} UniProts with {len(df)} entries with {workers} cores ({mode}).")
            with multiprocessing.Pool(
                    processes=workers,
                    initializer=init_worker,
                    initargs=(log_queue,)
            ) as pool:
                pool.map(worker_process, args_list)

        logger.info(f"All done in {time.time() - start_time:.1f} seconds")
    finally:
        # Ensure all queued logs flush before exit
        try:
            log_listener.stop()
        except Exception:
            pass
        logging.shutdown()


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_ubs_unification.py",
        description="Cluster ligand-binding residue sets per UniProt into Unified Binding Sites (UBS).",
    )
    p.add_argument("--input", required=True, metavar="CSV",
                   help="input CSV (job_name, UNPs, uniprot_residues, prot_chains_num, UNPs_num, pdb, ligands)")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="output root; one sub-directory per UniProt accession is created")
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS), metavar="T",
                   help="fixed Jaccard similarity thresholds in (0, 1]; the large-jump cut is always added "
                        "(default: 0.3 0.5 0.7 1.0)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
                   help="number of worker processes (default: 1)")
    p.add_argument("--subset", nargs="+", default=None, metavar="ACC",
                   help="only process these UniProt accessions (off by default)")
    p.add_argument("--no-figures", dest="figures", action="store_false",
                   help="skip dendrogram/histogram SVGs and tree.pkl; write only tables, input_data and manifest")
    p.add_argument("--compression", choices=("zip", "gz"), default=DEFAULT_COMPRESSION,
                   help="archive format for input_data.csv and figures (default: gz)")
    p.add_argument("--log-dir", default=None, metavar="DIR",
                   help="directory for the run log (default: the output root)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_pipeline(
        input_file=args.input,
        out_dir=args.out,
        thresholds=args.thresholds,
        workers=args.workers,
        subset=args.subset,
        figures=args.figures,
        compression=args.compression,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
