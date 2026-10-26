# -*- coding: utf-8 -*-
"""
plot_example_entries.py
=======================

Render the per-entry conformational-spectrum plots + apo/holo classification for a
SMALL, hand-picked set of AHoJ-DB entries (e.g. a manuscript figure example), reusing
a chosen **run config** so the candidate filtering and classification are *identical*
to that full-database run (e.g. `configs/run_config_full_run_3.json`) -- WITHOUT
re-processing the whole database.

Why this exists
---------------
Publication example figures (Figure 1) need the classification of a specific entry to
match the deployed run exactly. This script calls the *same* batch code path
(`batch_job_functions.do_kde_plots_and_classification`) that the full run uses, so the
per-entry numbers (P(apo), densities, separations, verbal labels) reproduce the full
run bit-for-bit. It is the clean, config-driven replacement for the ad-hoc
`conformational_spectrum_api.py`.

It writes, per entry, the standard AHoJ-DB per-entry outputs under OUTPUT_ROOT:
    <OUTPUT_ROOT>/<batch>/<entry>/statistics/classification_kde_pocket.json  (+ chain, all_stats, etc.)
    <OUTPUT_ROOT>/<batch>/<entry>/plots/       (kde curves/violins, DP curves, confidence bars if conf_bars=1)
Those JSONs feed the publication figure builder `figure_pocket_example.py`.

Input resolution (per entry), first hit wins:
    1) LOCAL_DB_ROOT/<batch>/<entry>.tgz              (authoritative AHoJ-DB batch layout)
    2) LOCAL_DB_ROOT/<batch>/<entry>/global_results.csv   (extracted batch layout)
    3) LOCAL_FLAT_DIRS/<entry>/global_results.csv     (flat AHoJ result folders)
    4) fetch from FETCH_BASE_URL (AHoJ-DB API) into <OUTPUT_ROOT>/_fetched_input/  (if FETCH_FALLBACK)

Spyder-friendly: edit the SETTINGS block below and press Run (or run from a shell; a
few of the settings are also exposed as optional CLI flags for scripting).
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse
from urllib.request import urlopen, Request

# --------------------------------------------------------------------------------------
# Locate the parse-ahoj repo root (this file lives in it) so the pipeline imports resolve
# regardless of the current working directory / Spyder.
# --------------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Repo-wide path anchors (common/config.py of the ahoj-alphafold repo; set them with environment
# variables or common/paths.ini). The loop finds the repo root however deeply this file is nested.
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

# ======================================================================================
# SETTINGS  (edit these)
# ======================================================================================
# The run whose filtering / classification parameters we mirror. Any run config works.
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "run_config_full_run_3.json")

# One or more AHoJ-DB entries (job_names) to render.
ENTRIES = ["2yiw-A-I46-1354"]

# Where the per-entry plots + statistics are written (mirrored <batch>/<entry> layout).
OUTPUT_ROOT = str(cfg.EXAMPLE_ENTRIES_DIR)        # AHOJ_EXAMPLE_ENTRIES_DIR

# Input source 1/2: authoritative local AHoJ-DB, batch layout <root>/<batch>/<entry>.tgz.
LOCAL_DB_ROOT = str(cfg.AHOJ_OUTPUT_FIX_DIR)      # AHOJ_OUTPUT_FIX_DIR

# Input source 3: flat AHoJ result folders <dir>/<entry>/global_results.csv.
# Optional: AHOJ_RESULTS_DIR (several folders separated by os.pathsep); none by default.
LOCAL_FLAT_DIRS = [d for d in os.environ.get("AHOJ_RESULTS_DIR", "").split(os.pathsep) if d]

# Input source 4: download the entry from the AHoJ-DB API if not found locally.
FETCH_FALLBACK = True
FETCH_BASE_URL = "https://dev.apoholo.cz"   # dev instance; use "https://apoholo.cz" for production

# Force these plotting/output options ON regardless of the run config (the full run turns
# conf_bars OFF and parallelism ON; for an example entry we want everything, serially).
FORCE_ARGS = {
    "conf_bars": 1,          # also emit the DB-style confidence-bar SVGs (reference / QA)
    "kde_violins": 1,
    "kde_curves": 1,
    "dp_curves": 1,
    "separate_metrics": 1,
    "save_statistics": 1,    # writes classification_kde_*.json (the figure builder's input)
    "save_filtered_df": 1,   # writes global_results_plots.csv (Panel A violin source)
    "save_per_method_json": 0,
    "save_relabeled_csv": 0,
    "num_processes": 1,
    "zip_plots_dir": 0,
    "zip_stats_dir": 0,
    "show_plots": 0,
    "svg_fonttype": "none",
}

# ======================================================================================
# Pipeline imports (after sys.path is set)
# ======================================================================================
import matplotlib
matplotlib.use("Agg")

import batch_process_database as bpd
from batch_job_functions import do_kde_plots_and_classification
from stats_plot import set_plot_font


# ======================================================================================
# Helpers
# ======================================================================================
def build_args_from_config(config_path, overrides):
    """Build the batch `args` namespace from a run config, then apply overrides.

    Reuses batch_process_database.parse_arguments so EVERY classification/filtering
    parameter is exactly what the run config specifies (no drift)."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Run config not found: {config_path}")
    saved_argv = sys.argv
    sys.argv = ["plot_example_entries", "--config", config_path]
    try:
        args = bpd.parse_arguments()
    finally:
        sys.argv = saved_argv
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def fetch_ahoj_entry(entry_id, dest_base_dir, base_url, timeout=120):
    """Download <entry>.zip from the AHoJ-DB API and extract it to
    <dest_base_dir>/<entry_id>/. Returns that directory (which holds global_results.csv).

    Self-contained (stdlib only) so this script does not depend on the ad-hoc test
    script. Mirrors the logic of conformational_spectrum_api.ensure_ahoj_entry
    but with a configurable base URL (default: dev instance)."""
    os.makedirs(dest_base_dir, exist_ok=True)
    entry_dir = os.path.join(dest_base_dir, entry_id)
    csv_path = os.path.join(entry_dir, "global_results.csv")
    if os.path.isfile(csv_path):
        return entry_dir

    zip_url = f"{base_url.rstrip('/')}/api/db/entry/{entry_id}/download/{entry_id}.zip"
    zip_path = os.path.join(dest_base_dir, f"{entry_id}.zip")
    print(f"    fetching {zip_url}")
    try:
        req = Request(zip_url, headers={"User-Agent": "AHoJ-fetch/1.0"})
        with urlopen(req, timeout=timeout) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        with zipfile.ZipFile(zip_path, "r") as zf:
            os.makedirs(entry_dir, exist_ok=True)
            zf.extractall(entry_dir)
    finally:
        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
    if not os.path.isfile(csv_path):
        # some archives nest the files one level down; try to locate the csv
        found = glob.glob(os.path.join(entry_dir, "**", "global_results.csv"), recursive=True)
        if found:
            return os.path.dirname(found[0])
        raise RuntimeError(f"global_results.csv not found after fetching {entry_id}")
    return entry_dir


def resolve_entry_input(entry_id):
    """Return a query_path (a .tgz file OR a directory containing global_results.csv)
    for `entry_id`, trying local sources first, then the API."""
    # 1) local DB, batch layout: <root>/<batch>/<entry>.tgz
    hits = glob.glob(os.path.join(LOCAL_DB_ROOT, "*", entry_id + ".tgz"))
    if hits:
        return hits[0]
    # 2) local DB, extracted: <root>/<batch>/<entry>/global_results.csv
    hits = glob.glob(os.path.join(LOCAL_DB_ROOT, "*", entry_id, "global_results.csv"))
    if hits:
        return os.path.dirname(hits[0])
    # 3) flat result dirs
    for d in LOCAL_FLAT_DIRS:
        if os.path.isfile(os.path.join(d, entry_id, "global_results.csv")):
            return os.path.join(d, entry_id)
    # 4) fetch from the API
    if FETCH_FALLBACK:
        return fetch_ahoj_entry(entry_id, os.path.join(OUTPUT_ROOT, "_fetched_input"), FETCH_BASE_URL)
    raise FileNotFoundError(
        f"No local input for {entry_id} under {LOCAL_DB_ROOT} or {LOCAL_FLAT_DIRS} and FETCH_FALLBACK is off."
    )


def mirror_stats_dir(root_mirror_dir, query_path):
    """Replicate do_kde_plots_and_classification's output-path derivation:
    outputs go to <root_mirror>/<last-two-path-components-without-ext>/{statistics,plots}."""
    parts = os.path.normpath(query_path).split(os.sep)
    last_two = os.path.join(*parts[-2:])
    sub = os.path.splitext(last_two)[0]
    return os.path.join(root_mirror_dir, sub, "statistics")


# ======================================================================================
# Main
# ======================================================================================
def parse_cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=CONFIG_PATH, help="Run config to mirror (default: %(default)s)")
    p.add_argument("--entries", nargs="+", default=None, help="Entry job_names (default: SETTINGS.ENTRIES)")
    p.add_argument("--output_root", default=OUTPUT_ROOT, help="Output root (default: %(default)s)")
    p.add_argument("--base_url", default=FETCH_BASE_URL, help="AHoJ-DB API base URL for fetch fallback")
    return p.parse_args()


def run(config_path=CONFIG_PATH, entries=None, output_root=OUTPUT_ROOT, base_url=None):
    global OUTPUT_ROOT, FETCH_BASE_URL
    OUTPUT_ROOT = output_root
    if base_url:
        FETCH_BASE_URL = base_url
    entries = entries or ENTRIES

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    overrides = dict(FORCE_ARGS)
    overrides["root_mirror_dir"] = OUTPUT_ROOT
    overrides["summary_dir"] = os.path.join(OUTPUT_ROOT, "_summary")
    args = build_args_from_config(config_path, overrides)
    set_plot_font(font_name=getattr(args, "plot_font", "Helvetica Neue"),
                  svg_fonttype=getattr(args, "svg_fonttype", "none"))

    print(f"[config] {config_path}")
    print(f"[params] filter_candidates={args.filter_candidates} drop_no_resolution={getattr(args,'drop_no_resolution',0)} "
          f"pocket_dist={args.pocket_dist} relabel_apoholo={args.relabel_apoholo} "
          f"bndgrsds={args.bndgrsds_prcnt_seqres}/{args.bndgrsds_prcnt_obs} rog_diff_thrs_query={args.rog_diff_thrs_query}")
    print(f"[output] {OUTPUT_ROOT}\n")

    results = {}
    for entry in entries:
        print(f"=== {entry} ===")
        query_path = resolve_entry_input(entry)
        print(f"    input: {query_path}")
        status = do_kde_plots_and_classification(args, entry, query_path)
        status_label = status[0] if isinstance(status, tuple) else str(status)
        stats_dir = mirror_stats_dir(OUTPUT_ROOT, query_path)
        ok = os.path.isfile(os.path.join(stats_dir, "classification_kde_pocket.json"))
        print(f"    status: {status_label} | stats: {stats_dir} | kde_pocket.json written: {ok}\n")
        results[entry] = {"status": status_label, "stats_dir": stats_dir, "query_path": query_path}
    return results


def main():
    cli = parse_cli()
    run(config_path=cli.config, entries=cli.entries, output_root=cli.output_root, base_url=cli.base_url)


if __name__ == "__main__":
    main()
