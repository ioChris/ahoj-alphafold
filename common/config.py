"""
Central path configuration / registry for the AHoJ-AlphaFold analysis repo.
=========================================================================

WHY THIS FILE EXISTS
--------------------
Every machine-specific / personal absolute path used anywhere in this repo is
defined HERE (and only here), built on a handful of *root anchors*. The anchors
themselves live in a local, git-ignored ``common/paths.ini`` (template:
``common/paths.example.ini``). This means:

  * Detach / move machine  -> edit ONE file (``common/paths.ini``).
  * Audit / clean up        -> every input & output path is listed below, each
                               tagged ``# CURRENT`` or ``# DEPRECATED``.
  * Public release          -> no personal absolute path is committed anywhere
                               (``paths.ini`` is git-ignored; only the example
                               template ships).

HOW SCRIPTS USE IT
------------------
Put this small bootstrap at the top of a script (works no matter how deeply the
script is nested, and regardless of the current working directory / Spyder):

    import sys as _sys
    from pathlib import Path as _Path
    for _root in _Path(__file__).resolve().parents:
        if (_root / "common" / "config.py").is_file():
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))
            break
    from common import config as cfg

Then refer to ``cfg.WORK_DIR``, ``cfg.ANALYSIS_UNIFIED_LEVELS_CSV``, etc.

ANCHOR RESOLUTION ORDER (per anchor)
------------------------------------
  1. environment variable  (e.g. ``AHOJ_AF_WORK_DIR``)
  2. ``[paths]`` section of ``common/paths.ini``
  3. a clearly-labelled placeholder  ``<UNSET:NAME>``  (so this module always
     imports; a script that actually tries to read/write will fail with an
     obvious path telling you what to configure)

Call ``cfg.require("WORK_DIR", "AHOJ_DB_DIR")`` near the top of a script if you
want an explicit, friendly error *before* any I/O is attempted.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

__all__ = [
    # anchors
    "WORK_DIR", "AHOJ_DB_DIR", "AFDB_DIR", "P2RANK_PRED_DIR", "PRESENTATION_DIR",
    # helpers
    "require", "is_configured",
]

# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_INI_PATH = _HERE / "paths.ini"

_cfg = configparser.ConfigParser()
# ConfigParser lowercases keys by default; keep them case-insensitive (fine here).
if _INI_PATH.is_file():
    _cfg.read(_INI_PATH, encoding="utf-8")

_UNSET_ANCHORS: list[str] = []


def _anchor(key: str, env_var: str) -> Path:
    """Resolve a root anchor: env var -> paths.ini [paths] -> placeholder."""
    val = os.environ.get(env_var)
    if not val and _cfg.has_section("paths"):
        val = _cfg.get("paths", key, fallback=None)
    if not val:
        _UNSET_ANCHORS.append(env_var)
        return Path(f"<UNSET:{env_var}>")
    return Path(os.path.expanduser(val.strip()))


def is_configured() -> bool:
    """True if every required anchor resolved to a real value."""
    return not _UNSET_ANCHORS


def require(*names: str) -> None:
    """Raise a friendly error if any named anchor is unconfigured.

    ``names`` are attribute names in this module, e.g. ``require("WORK_DIR")``.
    """
    missing = []
    for name in names:
        val = globals().get(name)
        if isinstance(val, Path) and str(val).startswith("<UNSET:"):
            missing.append(name)
    if missing:
        raise RuntimeError(
            "AHoJ-AlphaFold paths are not configured: "
            + ", ".join(missing)
            + f".\nSet them as environment variables, or copy "
            f"'{_HERE / 'paths.example.ini'}' to '{_INI_PATH}' and fill in the roots."
        )


# ===========================================================================
# ROOT ANCHORS  (the ONLY machine-specific values — configured in paths.ini)
# ===========================================================================
WORK_DIR        = _anchor("work_dir",        "AHOJ_AF_WORK_DIR")          # CURRENT  project work_dir (almost everything reads/writes under here)
AHOJ_DB_DIR     = _anchor("ahoj_db_dir",     "AHOJ_DB_DIR")               # CURRENT  AHoJ-DB database root (v2c): classification inputs
AFDB_DIR        = _anchor("afdb_dir",        "AFDB_STRUCTURES_DIR")       # CURRENT  AlphaFold-DB *.cif.gz monomer models
P2RANK_PRED_DIR = _anchor("p2rank_pred_dir", "P2RANK_PRED_DIR")           # CURRENT  raw P2Rank prediction outputs (eval dirs / cases)
UBS_ROOT_DIR    = _anchor("ubs_root_dir",    "UBS_ROOT_DIR")              # CURRENT  per-UniProt UBS threshold TSVs (UBS_root_2/<UniProt>/ST_*.tsv)
PRESENTATION_DIR = _anchor("presentation_dir", "AHOJ_AF_PRESENTATION_DIR")  # optional  poster/talk plot exports (a few plotting scripts)
AHOJ_OUTPUT_FIX_DIR = _anchor("ahoj_output_fix_dir", "AHOJ_OUTPUT_FIX_DIR")  # optional  per-job AHoJ run outputs (output_fix/<batch>/<job>.tgz; each .tgz has pocket_residues.csv with PDB-numbered binding residues)
EXAMPLE_ENTRIES_DIR = _anchor("example_entries_dir", "AHOJ_EXAMPLE_ENTRIES_DIR")  # optional  per-entry AHoJ example outputs (parse-ahoj plot_example_entries.py; Figure 1b reads 093/2yiw-A-I46-1354/statistics)
MANUSCRIPT_FIGURES_DIR = _anchor("manuscript_figures_dir", "AHOJ_AF_MANUSCRIPT_FIGURES_DIR")  # optional  composed manuscript figures (plotting/manuscript_v4/; unset -> PLOTS_DIR/manuscript_v4)
ENTRY_EXAMPLE_PSE = _anchor("entry_example_pse", "AHOJ_AF_ENTRY_EXAMPLE_PSE")  # optional  PyMOL session of the 2yiw worked entry (SI Figure S9d; plotting/manuscript_v4/render_si_entry_pymol_v4.py)


# ===========================================================================
# NAMED PATH REGISTRY  (built from anchors; grouped by analysis stage)
# Tag legend:  # CURRENT = used by the current pipeline / manuscript
#              # DEPRECATED = older run/location, kept for provenance only
# ===========================================================================

# --- Analysis 1 — apo/holo classification inputs (produced by the SEPARATE
#     methodology repo; consumed here). Live under the AHoJ-DB root. -----------
# Merged run_3 classification outputs (per-query apo/holo KDE classification summaries).
# DEPRECATED (run_2 baseline = manuscript run4): AHOJ_DB_DIR/Products/KDEs_and_classification/full_run_2/05102025_153331_do_kde_plots_and_classification/add_pLDDT_v1
CLASSIFICATION_MERGED_DIR = AHOJ_DB_DIR / "Products" / "KDEs_and_classification" / "full_run_3" / "plots_summary" / "full_run_3_merged"  # CURRENT (run_3 merged; 61-col query_summaries w/ inline pLDDT+resolution+has_model)
# INPUT to the survivor filter (00_classification/filter_survivors.py): the full per-query
# classification summary BEFORE analysis-set inclusion filtering (464,869 rows x 61 cols).
CLASSIFICATION_SUMMARIES_CSV = CLASSIFICATION_MERGED_DIR / "query_summaries_do_kde_plots_and_classification.csv"  # CURRENT
# OUTPUT of the survivor filter = the analysis set consumed by stages 01+ (237,922 rows / 12,255 UniProts).
FILTERED_SURVIVORS_DIR = CLASSIFICATION_MERGED_DIR / "filtering_20260704_152000"  # CURRENT (run_3 merged; has_model==1 + drop_no_resolution=1; see ahoj-af-project DECISIONS 2026-07-04)
FILTERED_SURVIVORS_ZIP = FILTERED_SURVIVORS_DIR / "filtered_survivors.zip"   # CURRENT
FILTERED_SURVIVORS_CSV = FILTERED_SURVIVORS_DIR / "filtered_survivors.csv"   # CURRENT
QUERY_SUMMARIES_CLASS_CSV = AHOJ_DB_DIR / "Summaries" / "extract_uniprot_residues" / "run_27062025_130510_do_collect_uniprot_residues" / "query_summaries_class.csv"  # CURRENT
# Per-query descriptor summaries used by the P2Rank descriptor correlations:
QUERY_SUMMARIES_ROG_CSV = AHOJ_DB_DIR / "Summaries" / "parsing_v3c5_DBv2b_561batches_511845jobs_80seqres_Noneobs_4.5res_0fullpcktaln_sums0" / "query_summaries_rog_DQ5.csv"  # CURRENT
# Per-entry apo-vs-holo descriptor means (batch_process_database.py --processing_function
# do_collect_apoholo_metric_stats; full DB, full_run_3 filters): SI Figure S1.
APOHOLO_METRIC_STATS_CSV = AHOJ_DB_DIR / "Summaries" / "apoholo_metric_stats" / "full_run_3" / "query_summaries_do_collect_apoholo_metric_stats.csv"  # CURRENT

# --- Analysis 2 — UBS clustering / aggregation + master unified table ---------
UBS_AGG_DIR = WORK_DIR / "UBS_aggregated_classifications"                     # CURRENT  base dir for aggregation outputs
# DEPRECATED (run_2 baseline = manuscript run4): UBS_AGG_DIR/10032026_130217_v1.11_Noneunplimit/..._filtered.zip
UBS_AGG_FILTERED_ZIP = UBS_AGG_DIR / "04072026_150704_v1.11_Noneunplimit" / "ubs_allthresholds_aggregated_classifications_filtered.zip"  # CURRENT (run_3; reference counts identical to run_2, filtered +15,716)
# DEPRECATED (run_2): UBS_AGG_DIR/21012026_151032_v1.7_Noneunplimit/ubs_allthresholds_aggregated_classifications.csv
UBS_AGG_CLASSIFICATIONS_CSV = UBS_AGG_DIR / "04072026_150704_v1.11_Noneunplimit" / "ubs_allthresholds_aggregated_classifications.csv"     # CURRENT (run_3 plain agg used for QC/sankey)

# DEPRECATED (run_2 classification): WORK_DIR/unified_output/run4  (run4 = manuscript run, built on run_2)
UNIFIED_RUN_DIR             = WORK_DIR / "unified_output" / "run5"            # CURRENT  (run5 = run_3 classification reproduction)
ANALYSIS_UNIFIED_LEVELS_CSV = UNIFIED_RUN_DIR / "analysis_unified_levels.csv"  # CURRENT  *** master table consumed by plotting / alphafill / p2rank ***
ANALYSIS_UNIFIED_LEVELS_ZIP = UNIFIED_RUN_DIR / "analysis_unified_levels.zip"  # CURRENT
PLOTS_DIR                   = UNIFIED_RUN_DIR / "plots"                        # CURRENT  main-figure output root
SI_PLOTS_DIR                = PLOTS_DIR / "SI"                                 # CURRENT  supplementary-figure output
# Composed manuscript figures (plotting/manuscript_v4/): the anchor if set, else a folder under PLOTS_DIR.
# Parts go to <dir>/parts, SI figures to <dir>/SI. The model-set switch below appends `_afdbv4`.
MANUSCRIPT_FIG_DIR = (PLOTS_DIR / "manuscript_v4" if str(MANUSCRIPT_FIGURES_DIR).startswith("<UNSET:")
                      else MANUSCRIPT_FIGURES_DIR)                             # CURRENT

# --- Analysis 4 — AlphaFill ---------------------------------------------------
# run_3 reproduction bumps every stage to a NEW version, leaving the run_2 baseline
# (parse v2 / merge v6 / unified run1 / entries run1) intact (decision 1). The shared
# JSON cache (ALPHAFILL_JSONS_DIR) is unversioned; run_3 topped it up (521 from the
# D:\alphafill_data mirror + server download of the remainder).
ALPHAFILL_DIR        = WORK_DIR / "alphafill"                                  # CURRENT
ALPHAFILL_JSONS_DIR  = ALPHAFILL_DIR / "jsons"                                 # CURRENT  downloaded AlphaFill JSONs (shared cache)
# DEPRECATED (run_2 baseline): ALPHAFILL_DIR/alphafill_analysis/v2/alphafill_parsed.zip
ALPHAFILL_PARSED_ZIP = ALPHAFILL_DIR / "alphafill_analysis" / "v3" / "alphafill_parsed.zip"  # CURRENT (run_3)
# DEPRECATED (run_2 baseline): WORK_DIR/merge_ahoj_alphafill/v6
MERGE_AHOJ_ALPHAFILL_DIR          = WORK_DIR / "merge_ahoj_alphafill" / "v7"   # CURRENT (run_3)
MERGED_AHOJ_ALPHAFILL_SUMMARY_CSV = MERGE_AHOJ_ALPHAFILL_DIR / "merged_ahoj_alphafill_summary.csv"  # CURRENT
MERGED_AHOJ_ALPHAFILL_LONG_CSV    = MERGE_AHOJ_ALPHAFILL_DIR / "merged_ahoj_alphafill_long.csv"     # CURRENT
# DEPRECATED (run_2 baseline = manuscript Fig 5): WORK_DIR/alphafill_unified/run1
ALPHAFILL_UNIFIED_DIR          = WORK_DIR / "alphafill_unified" / "run2"       # CURRENT (run_3) UBS-level branch
ALPHAFILL_UNIFIED_ANALYSIS_DIR = ALPHAFILL_UNIFIED_DIR / "analysis_v1"         # CURRENT
# DEPRECATED (run_2 baseline): WORK_DIR/alphafill_unified_entries/run1
ALPHAFILL_ENTRIES_DIR          = WORK_DIR / "alphafill_unified_entries" / "run2"  # CURRENT (run_3) entry-level branch
ALPHAFILL_ENTRIES_ANALYSIS_DIR = ALPHAFILL_ENTRIES_DIR / "analysis_v1"        # CURRENT

# --- Analysis 5 — AFDB dataset build + P2Rank analysis ------------------------
AFDB_DATASET_DIR            = WORK_DIR / "afdb_dataset"                        # CURRENT
# DEPRECATED (run_2 baseline): AFDB_DATASET_DIR/st100  (run_2 ST_100 dataset + geometry, 3/2026)
AFDB_DATASET_ST100_DIR      = AFDB_DATASET_DIR / "st100_run3"                  # CURRENT (run_3; shared ground truth for P2Rank + docking + LBS)
AFDB_ST100_DATASET_CSV      = AFDB_DATASET_ST100_DIR / "afdb_st_100_dataset.csv"               # CURRENT
AFDB_ST100_DATASET_GEOM_CSV = AFDB_DATASET_ST100_DIR / "afdb_st_100_dataset_with_geometry.csv" # CURRENT
# DEPRECATED (run_2 baseline = manuscript P2Rank): WORK_DIR/P2Rank/ubs_run_4
P2RANK_RUN_DIR               = WORK_DIR / "P2Rank" / "ubs_run_5"              # CURRENT (run_3)
P2RANK_ANALYSIS_DIR          = P2RANK_RUN_DIR / "analysis_v1"                  # CURRENT
P2RANK_TABLES_DIR            = P2RANK_ANALYSIS_DIR / "tables"                  # CURRENT
P2RANK_PLOTS_DIR             = P2RANK_ANALYSIS_DIR / "plots"                   # CURRENT
P2RANK_MERGED_PER_POCKET_CSV = P2RANK_TABLES_DIR / "p2rank_st100_merged_per_pocket.csv"  # CURRENT  *** main P2Rank per-pocket table ***

# --- Analysis 6 — Docking (AutoDock Vina / Uni-Dock) on AFDB pockets ----------
# The docking manifest is built on OUR side from the ST_100 geometry dataset
# (+ merged AlphaFill for ligand group 2) and handed over to a colleague who runs
# the batch. Receptors are the same AFDB models used everywhere (cfg.AFDB_DIR).
DOCKING_DIR                   = WORK_DIR / "docking"                            # CURRENT  base dir for docking datasets / handover / results
DOCKING_MANIFEST_DIR          = DOCKING_DIR / "manifest"                        # CURRENT  built per-(pocket x ligand) docking task manifests (handover bundle)
DOCKING_LIGANDS_DIR           = DOCKING_DIR / "ligands"                         # CURRENT  resolved ligand SMILES/SDF (CCD group1/2 + probe library group3)
DOCKING_REFPOSES_DIR          = DOCKING_DIR / "reference_poses"                 # CURRENT  experimental / AlphaFill reference poses (for redock RMSD)
DOCKING_RUN3_DIR              = DOCKING_DIR / "run3"                            # CURRENT  full_run_3 GPU-docking outputs (manifest/out/results/reference_poses)
# Local cache for the run_3 redock reference-pose build: fetched experimental
# source structures (~45 GB mmCIF.gz) + CCD component cifs. MUST be local, NOT in a
# cloud-synced folder (it would sync tens of GB). Defaults next to the AFDB models' root.
DOCKING_REDOCK_CACHE_DIR      = AFDB_DIR.parent.parent / "redock_run3"          # CURRENT  <AFDB_DIR>/../../redock_run3/{structures,ccd}
DOCKING_PILOT_DIR             = DOCKING_DIR / "pilot"                           # CURRENT  stratified pilot subset + its manifest
DOCKING_HANDOVER_DIR          = DOCKING_DIR / "pilot_handover"                  # CURRENT  self-contained, share-ready bundle (label-stripped manifest + receptors + ligands + docs)
DOCKING_RESULTS_DIR           = DOCKING_DIR / "results"                         # CURRENT  colleague's returned docking results (ingested here)
DOCKING_ANALYSIS_DIR          = DOCKING_DIR / "analysis_v1"                     # CURRENT  merged per-pocket docking table + correlations
DOCKING_MERGED_PER_POCKET_CSV = DOCKING_ANALYSIS_DIR / "docking_merged_per_pocket.csv"  # CURRENT  *** main docking per-pocket table ***
# Optional: a curated probe-ligand library (SMILES .csv) for group-3 dockability.
# Defaults to a small built-in set written by 05_docking/dataset/resolve_ligands.py
# under DOCKING_LIGANDS_DIR if this file is absent.
DOCKING_PROBE_LIBRARY_CSV     = DOCKING_LIGANDS_DIR / "probe_library.csv"       # CURRENT

# --- Analysis 7 — LBS generality (2nd/3rd binding-site predictor; T3-2nd) ------
# Run additional pocket / ligand-binding-site predictors (fpocket, IF-SitePred,
# ...) on the SAME AFDB ST_100 pocket set and recompute the SAME DCA4 / TNS
# success as P2Rank, to show the AF-Holo>AF-Apo detection gradient is not
# P2Rank-specific. Receptors are the same AFDB models (cfg.AFDB_DIR); the
# ground-truth site definition is the union residue set (cfg.AFDB_ST100_DATASET_GEOM_CSV).
LBS_DIR                    = WORK_DIR / "lbs_generality"                        # CURRENT  base dir for this stage
LBS_GROUND_TRUTH_NPZ       = LBS_DIR / "ground_truth_atoms.npz"                 # CURRENT  per-pocket union-residue heavy atoms (built once, gemmi)
LBS_RECEPTORS_CSV          = LBS_DIR / "receptors.csv"                          # CURRENT  site_uid -> AFDB model (predictor input list)
LBS_PILOT_DIR             = LBS_DIR / "pilot"                                   # CURRENT  stratified pilot subset (mirrors docking pilot)
# Per-tool predictions + scored success tables live under LBS_DIR/<tool>/:
#   LBS_DIR/<tool>/predictions.csv                       (normalized adapter output)
#   LBS_DIR/<tool>/analysis_v1/tables/lbs_<tool>_st100_merged_per_pocket.csv  (scorer output)
LBS_PROBE_TOOLS            = ("fpocket", "ifsitepred")                          # CURRENT  recommended 2nd/3rd predictors

# --- AFDB v4 re-alignment (2026-09) -------------------------------------------
# The classification (AHoJ-DB v2c) was computed on AlphaFold DB v4 models; the
# July 2026 detection and docking arms above ran on v6 files. These constants
# name the re-run of those arms on the v4 models. They sit NEXT TO the v6 ones:
# nothing above is repointed and no v6 output is overwritten. The model folder is
# passed per script (--afdb_dir) or through AFDB_STRUCTURES_DIR; it is a staged,
# md5-checked copy of the detection accessions' models from the v2c store
# (10,500 v4 files and one v3 file, AF-G3FFN6-F1-model_v3.cif.gz).
AFDBV4_SUFFIX                      = "_afdbv4"                                                  # CURRENT-AFDBV4
AFDBV4_MODEL_DIR                   = AFDB_DIR.parent / "v4_detection_afdbv4"                    # CURRENT-AFDBV4  staged v4 (+1 v3) detection models
AFDBV4_DATASET_ST100_DIR           = AFDB_DATASET_DIR / "st100_run3_afdbv4"                     # CURRENT-AFDBV4
AFDBV4_ST100_DATASET_GEOM_CSV      = AFDBV4_DATASET_ST100_DIR / "afdb_st_100_dataset_with_geometry.csv"  # CURRENT-AFDBV4  (selection CSV stays AFDB_ST100_DATASET_CSV)
AFDBV4_P2RANK_RUN_DIR              = WORK_DIR / "P2Rank" / "ubs_run_5_afdbv4"                   # CURRENT-AFDBV4
AFDBV4_P2RANK_ANALYSIS_DIR         = AFDBV4_P2RANK_RUN_DIR / "analysis_v1"                      # CURRENT-AFDBV4
AFDBV4_P2RANK_MERGED_PER_POCKET_CSV = AFDBV4_P2RANK_ANALYSIS_DIR / "tables" / "p2rank_st100_merged_per_pocket.csv"  # CURRENT-AFDBV4
AFDBV4_LBS_GROUND_TRUTH_NPZ        = LBS_DIR / "ground_truth_atoms_afdbv4.npz"                  # CURRENT-AFDBV4
AFDBV4_LBS_INPUTS_DIR              = LBS_DIR / "inputs_afdbv4"                                  # CURRENT-AFDBV4  receptors.csv, receptor_site_map.csv
#   per-tool outputs: LBS_DIR/<tool>/full_afdbv4/  (LBS_DIR/p2rank_afdbv4/ for the P2Rank re-score)
AFDBV4_DOCKING_RUN3_DIR            = DOCKING_DIR / "run3_afdbv4"                                # CURRENT-AFDBV4

# --- Model-set switch (opt-in; default = the July run on v6 files, nothing above changes) ---------------
# AHOJ_AF_MODEL_SET=afdbv4 makes the CURRENT names of the detection and docking arms resolve to their
# CURRENT-AFDBV4 counterparts, so every downstream script reads and writes the `_afdbv4` tree without
# per-script path arguments. The selection CSV (AFDB_ST100_DATASET_CSV), the classification tables and the
# AlphaFill merge do not depend on the model files and are not switched.
MODEL_SET = os.environ.get("AHOJ_AF_MODEL_SET", "").strip().lower()
if MODEL_SET not in ("", "afdbv4"):
    raise RuntimeError(f"AHOJ_AF_MODEL_SET={MODEL_SET!r}: expected unset or 'afdbv4'")
LBS_FULL_SUBDIR       = "full"                                   # LBS_DIR/<tool>/full  (fpocket, ifsitepred)
LBS_P2RANK_DIR        = LBS_DIR / "p2rank"                       # harmonized re-score of the P2Rank centres
LBS_ANALYSIS_DIR      = LBS_DIR / "analysis_v1"                  # cross-tool tables/ and plots/
LBS_QUALITY_GATED_DIR = LBS_DIR / "quality_gated"
if MODEL_SET == "afdbv4":
    AFDB_DIR                     = AFDBV4_MODEL_DIR
    AFDB_DATASET_ST100_DIR       = AFDBV4_DATASET_ST100_DIR
    AFDB_ST100_DATASET_GEOM_CSV  = AFDBV4_ST100_DATASET_GEOM_CSV
    P2RANK_RUN_DIR               = AFDBV4_P2RANK_RUN_DIR
    P2RANK_ANALYSIS_DIR          = AFDBV4_P2RANK_ANALYSIS_DIR
    P2RANK_TABLES_DIR            = P2RANK_ANALYSIS_DIR / "tables"
    P2RANK_PLOTS_DIR             = P2RANK_ANALYSIS_DIR / "plots"
    P2RANK_MERGED_PER_POCKET_CSV = AFDBV4_P2RANK_MERGED_PER_POCKET_CSV
    DOCKING_RUN3_DIR             = AFDBV4_DOCKING_RUN3_DIR
    LBS_GROUND_TRUTH_NPZ         = AFDBV4_LBS_GROUND_TRUTH_NPZ
    LBS_RECEPTORS_CSV            = AFDBV4_LBS_INPUTS_DIR / "receptors.csv"
    LBS_FULL_SUBDIR              = "full_afdbv4"
    LBS_P2RANK_DIR               = LBS_DIR / "p2rank_afdbv4"
    LBS_ANALYSIS_DIR             = LBS_DIR / "analysis_v1_afdbv4"
    LBS_QUALITY_GATED_DIR        = LBS_DIR / "quality_gated_afdbv4"
    MANUSCRIPT_FIG_DIR           = MANUSCRIPT_FIG_DIR.with_name(MANUSCRIPT_FIG_DIR.name + AFDBV4_SUFFIX)


def lbs_tool_dir(tool: str) -> Path:
    """Folder of a detection tool's normalized predictions and scored per-pocket table, for the active model set."""
    return LBS_P2RANK_DIR if tool == "p2rank" else LBS_DIR / tool / LBS_FULL_SUBDIR

# ---------------------------------------------------------------------------
# DEPRECATED runs (collected from commented-out alternates across the
# old scripts, for provenance + cleanup). Not used by the current pipeline.
# ---------------------------------------------------------------------------
# Older runs / table versions superseded by the CURRENT entries above:
#   - unified_output\\run3        (superseded by run4)
#   - UBS aggregations v1.2 / v1.6 / v1.7   (superseded by v1.11 filtered)
#   - merge_ubs_alphafill (UBS-only)         (superseded by merge_ahoj_alphafill\\v6)
