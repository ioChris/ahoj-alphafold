# 00 — apo/holo classification

This is **Analysis 1** of the project: the probabilistic apo/holo classification of each
AlphaFold-DB model pocket against AHoJ-DB experimental apo/holo reference distributions
(a KDE density-ratio scorer with equal priors, over four pocket metrics: all-atom RMSD,
SASA, concave volume, radius of gyration).

The classification code is in [`kde_classifier/`](kde_classifier/). Its output, the per-query
classification summary, feeds the survivor filter below, and the classified *survivor* table
(`filtered_survivors.csv/.zip`, `FILTERED_SURVIVORS_*` in [`../common/config.py`](../common/config.py))
feeds [`../01_ubs_clustering/`](../01_ubs_clustering/) and every later stage.

---

### Source and version

`kde_classifier/` is a copy of the classification code of the AHoJ-DB parsing repository
(`parse-ahoj`, not public) at commit **`9128ae1`**. The published classification (`full_run_3`) was run
with `parse-ahoj` commits `0623fb7`, `886d35b`, `f25699a` and `42e127b`, which the run manifests in the
Zenodo deposit name. The commits after those add the AHoJ 2.4 run types (`holo-2.4`, `afc-2.4`, see
[`kde_classifier/docs/RUN_TYPES.md`](kde_classifier/docs/RUN_TYPES.md)) and leave the AHoJ-DB v2c profile
(`holo-2.1`) unchanged: re-classifying 100 v2c archives with this copy reproduces the released
`P(holo)`, `n_apo`, `n_holo` and class of all 73 that the release classifies (the `legacy` check below).

What changed in the copy: `@author` tags carry the full name; machine-specific default paths are
replaced by the repository's path anchors or by required arguments (listed below);
`test_conformational_spectrum_api.py` is renamed `conformational_spectrum_api.py` (it is a library, and
pytest would collect it under its old name). Nothing else.

### Module set

The modules import each other by top-level name, so they run as scripts from `kde_classifier/` (or
with that folder on `sys.path`).

| file | role |
|---|---|
| `batch_process_database.py` | **entry point**: walks an AHoJ-DB tree (`<root>/<batch>/<job>.tgz`) and classifies every entry, in parallel; writes per-entry outputs and the run summaries |
| `batch_job_functions.py` | the per-entry job `do_kde_plots_and_classification` (load, filter, relabel, metrics, KDE classification, plots, summary row) |
| `run_types.py` | run-type detection and profiles (`holo-2.1` = AHoJ-DB v2c, `holo-2.4`, `afc-2.4`), archive and bundle readers |
| `ahoj_names.py` | AHoJ 2.6.0 → internal column/key names |
| `parse_AHoJ3_DBv2b.py`, `parsing_functions_DBv2.py` | AHoJ result parsing, candidate filtering and apo/holo relabelling |
| `kde_core.py`, `kde_api.py` | the KDE density-ratio classifier (`classify_point_kde_r3`) |
| `stats_core.py`, `stats_api.py` | descriptive statistics, separation/overlap, the alternative classifiers reported alongside |
| `kde_plot.py`, `stats_plot.py` | KDE curves/violins, confidence bars and their verbal bands, plot font set-up |
| `conformational_spectrum_api.py` | shared helpers (metric names/units, query normalisation, JSON/zip I/O); its `main` is a single-entry driver |
| `plot_example_entries.py` | re-classifies a few chosen entries with a run config, exactly as the full run does |
| `figure_pocket_example.py` | composes the per-entry classification figure from those outputs |
| `run_full_batch_launcher.py`, `launch_full_run_3.sh` | the resumable, one-batch-at-a-time launcher the full run used |
| `configs/` | `run_config_full_run_3.json` (the published run), `run_config_full_run_3_nomodel.json` (its top-up for entries without a model row), and the 2.4 run-type configs |
| `docs/RUN_TYPES.md` | the run profiles and their fidelity checks; `docs/confidence_bar_spec.md`: the confidence-bar bands |
| `tests/`, `analysis/run_types_fidelity/check_run_types.py` | unit/regression tests on small fixture archives; the fidelity checks against the release |

Requirements: the repository's [`../requirements.txt`](../requirements.txt), plus `pyyaml` (read only for
`--bundle` run directories) and `pytest` for the tests.

### Running a classification

```bash
cd 00_classification/kde_classifier
python batch_process_database.py --config configs/run_config_full_run_3.json \
    --root_data_dir /abs/path/to/AHoJ-DB/v2c/data \
    --root_mirror_dir /abs/path/to/out/plots \
    --summary_dir /abs/path/to/out/plots_summary
```

- **Input:** the AHoJ-DB v2c per-entry archives, `<root_data_dir>/<batch>/<job>.tgz`.
- **Config:** values in the JSON become argument defaults; an explicit flag overrides them. The three
  path keys in the shipped configs are the run host's; override them as above or edit a copy. Use
  absolute paths (relative ones are joined to the first two levels of the working directory).
  `--root_data_dir` and `--summary_dir` are required.
- **Output:** per entry, `<root_mirror_dir>/<batch>/<job>/{statistics,plots}/`
  (`statistics/classification_summary.json`, `classification_kde_pocket.json`, ...); per run,
  `<summary_dir>/run_<timestamp>_do_kde_plots_and_classification/` with
  `query_summaries_do_kde_plots_and_classification.csv` (one row per query, the input of the survivor
  filter), `query_metrics_*.csv`, `parameters.json` and `run_manifest.json`.
- **Full database:** `launch_full_run_3.sh <config>` runs `run_full_batch_launcher.py`, which calls
  `batch_process_database.py` once per batch directory (crash-resumable; two machines can split the run
  and merge by copying; `concat` stitches the per-batch summaries into the single run directory above).
- **Tests:** `python -m pytest 00_classification/kde_classifier/tests` (121 pass; 4 skip: one needs an
  AHoJ2 checkout in `AHOJ2_DIR`, three compare against a JavaScript renderer that stays in `parse-ahoj`).
  The fidelity checks (`check_run_types.py legacy|afc|hq1 --out DIR`) need the AHoJ-DB archives and the
  release summary (`AHOJ_OUTPUT_FIX_DIR`, `AHOJ_DB_DIR`) and, for `afc`/`hq1`, the AHoJ 2.4 pilot runs
  (`AHOJ_PILOT_RUNS`, `WORK_DIR/af_centric_rerun/pilot_sites_collected.csv`).

### Worked example (Supplementary Figure S9)

Supplementary Figure S9 shows the classification of one entry, **`2yiw-A-I46-1354`** (UniProt
**Q16539**, MAPK14 / p38α; pocket class **Apo**, average P(holo) 0.201): **a** the KDE violins of the four
pocket metrics, **b** the per-metric confidence bars, **c** the overall pocket confidence bars, all drawn
by this code; **d** is a PyMOL render
([`../plotting/manuscript_v4/render_si_entry_pymol_v4.py`](../plotting/manuscript_v4/)).

```bash
cd 00_classification/kde_classifier
python plot_example_entries.py --entries 2yiw-A-I46-1354   # re-classifies the entry with configs/run_config_full_run_3.json
python figure_pocket_example.py                             # composes a-c (180 x 234 mm SVG/PDF/PNG)
```

- `plot_example_entries.py` reads the entry from `AHOJ_OUTPUT_FIX_DIR/<batch>/<entry>.tgz` (then from
  the folders in `AHOJ_RESULTS_DIR`, then the AHoJ-DB API) and writes it to
  `AHOJ_EXAMPLE_ENTRIES_DIR/<batch>/<entry>/` (`--output_root` overrides). It calls the same
  `do_kde_plots_and_classification` as the full run, so its numbers are those of the release.
- `figure_pocket_example.py` reads `AHOJ_EXAMPLE_ENTRIES_DIR/*/<entry>/statistics` and writes
  `<entry>/figure/`. Change `ENTRIES` / `ENTRY` at the top of the two scripts for another entry. The manuscript panel is
  assembled by [`../plotting/manuscript_v4/make_si_entry_example_v4.py`](../plotting/manuscript_v4/), which
  reuses these functions and only changes the panel letters.
- The path anchors come from [`../common/config.py`](../common/config.py) (environment variables or
  `common/paths.ini`).

**Fonts.** The figures use Helvetica Neue (the full run's plots: `plot_font` in the config). The font is
proprietary and is **not** distributed. To reproduce the exact typography, install it system-wide, or
put the `HelveticaNeue-*.otf`/`.ttf` files in `kde_classifier/fonts/` (git-ignored) or in a folder named
by `AHOJ_FONT_DIR`. Without it, `figure_pocket_example.py` falls back to the first installed of TeX Gyre
Heros, Nimbus Sans, Arial, DejaVu Sans, and the batch plots to matplotlib's default (a warning is
printed); the numbers do not depend on the font.

---

### The survivor filter (analysis-set selection) — lives here

[`filter_survivors.py`](filter_survivors.py) selects the **analysis set** from the classification
output: it reads the full per-query classification summary
(`query_summaries_do_kde_plots_and_classification.csv`, `cfg.CLASSIFICATION_SUMMARIES_CSV`) and keeps
the pockets that enter stages 01+. This is an **analysis decision, not a classification-method one** —
its thresholds are the study's analysis-set inclusion criteria (they set the manuscript's headline N),
so the code lives with the analysis rather than in the methodology repo (where it used to be
`filter_KDE_classifications.py`).

- **Inclusion criteria:** `has_model == 1` (AHOJ-018), usable pocket classification (non-NaN pocket
  probabilities), `n_apo >= 1`, and query resolution `<= 20 Å`.
- **Run:** edit the top-of-file settings if needed, then run `python 00_classification/filter_survivors.py`
  (Spyder-friendly). Input defaults to `cfg.CLASSIFICATION_SUMMARIES_CSV`; output goes to a fresh
  `filtering_<timestamp>/` beside the input. Point `cfg.FILTERED_SURVIVORS_DIR` at the produced directory.
- **Canonical run_3 output:** `filtering_20260704_152000/` → **237,922** survivors / **12,255** UniProts
  (from 464,869 classified queries).

---

### Robustness — apo/holo pockets are equally completely observed

A natural worry for any apo/holo comparison is that **apo** experimental structures might be less completely
*observed* (more missing density) than **holo** ones, so that apparent apo/holo differences would reflect
observation completeness rather than genuine conformational state. This is a documented limitation of
*whole-chain* apo/holo-versus-AlphaFold comparisons (Saldaño et al. 2022, *Bioinformatics* 38:2742,
[10.1093/bioinformatics/btac202](https://doi.org/10.1093/bioinformatics/btac202)), where holo depositions
tend to be more completely observed and align to more of the full-length model.

Because this classification is defined at the **binding-pocket** level, the relevant test is whether the
*pocket* residues are equally observed in the apo and holo experimental structures that feed each call — and
they are. Across a sample of 12,000 queries drawn from the analysis set (10,592 with both apo and holo matches,
paired within query), the median fraction of unobserved pocket residues is **0.00% for both classes** (higher
for apo in only 26% of queries), observed chain coverage is near-identical (99.4% apo vs 99.7% holo), and there
is only a minor holo resolution edge (2.49 vs 2.26 Å); a random draw over *all* AHoJ queries gives the same
result. The pocket-level classification therefore compares the same fully-observed residue set for both classes
and is **not** confounded by differential observation completeness.

- **Check:** [`../05_docking/redock/check_apo_holo_completeness.py`](../05_docking/redock/check_apo_holo_completeness.py)
  (implemented alongside the docking analysis, but classification-wide). It reads the per-query AHoJ outputs
  (`output_fix/<batch>/<job>.tgz` → `apo/holo_filtered_sorted_results.csv`, columns `%unobserved_rsd`,
  `UNPcov_obs`, `resolution`) and compares apo vs holo paired within each query. The numbers above are the
  `--survivors` run (sample restricted to the `filtered_survivors` analysis set, key `job_name`); omit the flag
  for a random draw over all AHoJ queries.
