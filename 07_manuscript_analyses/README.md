# 07 — Supplementary analyses of the manuscript

The scripts behind the numbers of the Supplementary Information and the supplementary numbers quoted in the
main text and Methods (Supplementary Notes 1–5, Tables S3–S13, Supplementary Data 1, and the Methods counts).
The main and SI figures are drawn by `plotting/` and `plotting/manuscript_v4/`; the analyses of stages 01–06 by
their own folders.

**Running them.** Run each script from this folder (`python <script>.py [subcommand]`). Every script starts with
`import ma_paths as P` ([`ma_paths.py`](ma_paths.py)), which resolves the roots from `common/config.py` and puts the
classification code (`00_classification/kde_classifier/`) on the import path.

- Outputs, and the intermediates the scripts read from each other, go to `P.OUT`: `manuscript_analyses_dir` in
  `common/paths.ini` or `AHOJ_AF_MANUSCRIPT_ANALYSES_DIR`; default `<work_dir>/manuscript_analyses`. The published
  output tables are in the Zenodo record (10.5281/zenodo.23243349).
- `AHOJ_AF_MODEL_SET=afdbv4` selects the AlphaFold DB v4 model set the manuscript quotes. The commands below say
  where it is needed.
- Data roots beyond `common/config.py`'s named paths are anchors (`P.anchor(key, env)`): set the key in
  `common/paths.ini` or the environment variable. The tables in each section list them. A script fails only in
  the mode that uses an unset anchor.

**How they were checked.** Each script was copied from the authors' working copy, with only its paths changed.
Each was re-run from here and its output compared with the published one: CSVs cell by cell, logs line by line
(paths and timings masked). The Verification column says what matched. Whole-database scans over the per-job
archives (30 min to 2 h each) were checked on their built-in subsets against the same rows of the published
per-entry tables. Modes that query live web services or rebuild network caches were not re-run; their caches are
listed under B.

**Extra packages:** `pyarrow` (parquet). For a few modes only: `pymol-open-source`, PyVOL 1.7.8 with MSMS,
`foldcomp`, and the `gcloud` CLI (listed per script).

The scripts are grouped below: A, B and C, then two added at integration.

## A. Classifier-based analyses

Every script starts with `import ma_paths as P` (except the helper `ahoj_job.py`, which has no roots) and takes
its roots from it. Outputs and the intermediates these scripts share go to `P.OUT` (`AHOJ_AF_MANUSCRIPT_ANALYSES_DIR`,
default `<work_dir>/manuscript_analyses`), under their original file names. The per-entry intermediates that used
to sit in `work_dir/balanced_classification/` and `work_dir/af_model_occlusion/` are now in `P.OUT` too.
`afdbv4` below means `AHOJ_AF_MODEL_SET=afdbv4`, the AlphaFold DB v4 model set the manuscript quotes.

The `collect`-type subcommands re-read the per-job AHoJ archives (`cfg.AHOJ_OUTPUT_FIX_DIR/<batch>/<job>.tgz`)
and re-apply the classifier's own curation by importing the classification code by name
(`00_classification/kde_classifier/`: `parsing_functions_DBv2`, `parse_AHoJ3_DBv2b`, `kde_core`, `kde_api`).
Over the whole analysis set (237,922 entries) each takes 30 min to 2 h on 12-22 processes; they were verified
on their `--limit` / `--n` / `--sites` subsets (see Verification), and every `summarise` / `sites` step was
re-run in full from the published per-entry tables.

### Anchors

| Key (`common/paths.ini`) | Env var | Points at | Used by |
|---|---|---|---|
| `afdb_store_dir` | `AHOJ_AFDB_STORE_DIR` | AFDB model store with `v4/` and `v3/` subfolders (`AF-<acc>-F1-model_v4.cif.gz`) | fig1b_site_lazou, temporal_split `templates`, af_model_occlusion `collect`/`verify` |
| `sifts_dir` | `AHOJ_SIFTS_DIR` | SIFTS flat files `uniprot_segments_observed.tsv`, `pdb_chain_uniprot.tsv` | af_model_occlusion `sifts` |
| `webserver_results_dir` | `AHOJ_WEBSERVER_RESULTS_DIR` | AHoJ webserver result folders (`<job>/structure_files`, the aligned chains of the 2yiw entry) | fig1b_site_lazou |
| `pdb_resolutions_dir` | `AHOJ_PDB_RESOLUTIONS_DIR` | folder holding the PDB `entries.idx` (deposition dates) | temporal_split `collect` |

Existing anchors these scripts need beyond `work_dir` / `ahoj_db_dir`: `ahoj_output_fix_dir` (every `collect`),
`example_entries_dir` (fig1b_site_lazou), and `afdb_dir` through `cfg.DOCKING_REDOCK_CACHE_DIR`
(fig1b_site_lazou reads the deposited 2YIW from `<redock_run3>/structures`).
One optional env var, not an anchor: `AHOJ_PARSE_AHOJ` (ligand_relabel_criteria `lists`) points the list and
call-site scan at a parse-ahoj checkout; unset, it reads `00_classification/kde_classifier/`.

### Run order and dependencies

- `ensemble_containment.py summarise` before `apo_holo_state_distance.py summarise` (reads
  `ensemble_containment_summary.csv`); `apo_holo_state_distance.py summarise` also reads `frontier_final_table.txt`
  (frontier_final.py, group C). `apo_holo_state_distance.py collect` before `density_percentile.py summarise`
  (position block reads `apo_holo_state_distance_entries.parquet`; skipped when absent).
- `balanced_classification.py collect` before `temporal_split.py summarise` (reads `balanced_entries.csv`), and
  `classifier_size_bias.py downstream` before `balanced_classification.py sites` (draw-noise block reads
  `downstream_members.csv`; skipped when absent).
- `temporal_split.py dates` and `templates` before its `collect`.
- `af_model_occlusion.py sifts` before `collect`.
- Imports from other groups: `balanced_classification.py sites` imports `separation_shape` and `st100_evidence_numbers`;
  `af_model_occlusion.py summarise` imports `separation_shape`.
- Python modules beyond requirements.txt: `pyarrow` (parquet I/O: apo_holo_state_distance, density_percentile,
  temporal_split); PyMOL (`pymol2`, pymol-open-source) for af_model_occlusion `collect`/`verify`, plus PyVOL 1.7.8
  and MSMS for `verify`; `gemmi` for fig1b_site_lazou (already listed). The PyMOL steps and fig1b ran in a Python
  3.11 env (PyMOL 3.1, gemmi 0.7.3); everything else in base Python 3.12 (pandas 2.2.2, numpy 1.26.4, scipy 1.13.1).

### Scripts

| File (lines) | Computes | Backs | Command | Inputs | Outputs | Verification |
|---|---|---|---|---|---|---|
| `ahoj_job.py` (197) | Shared helper: reads a job archive's `global_results.csv` / `ligands.json` under the column names before AHoJ 2.6.0 (`read_global_results`, `legacy_results`, `read_ligands`) | Used by ensemble_containment, balanced_classification, ligand_relabel_criteria, af_model_occlusion, fig1b_site_lazou and group C's afdb_v4_sources | library; `python ahoj_job.py --check <AHoJ2>/output_names.py` | none | none | Copied unchanged (interface kept); exercised by every archive run below |
| `ensemble_containment.py` (353) | Per entry, from the archives with the classifier's curation: apo states inside the holo min–max / mean ± 1 SD and vice versa; interpenetration; within-entry spread (CV, range); holo ligand states, conformers per state | SI Table S7 (containment table, interpenetration 92.1/88.6/88.5/86.7%, all-four 77.7%, spread and ligand-set paragraph) | `python ensemble_containment.py collect --procs 12`; `python ensemble_containment.py summarise` | `cfg.ANALYSIS_UNIFIED_LEVELS_CSV`, `cfg.CLASSIFICATION_SUMMARIES_CSV`, `cfg.AHOJ_DB_DIR/Summaries/apoholo_metric_stats/full_run_3/parameters.json`, `cfg.AHOJ_OUTPUT_FIX_DIR` | `P.OUT/ensemble_containment_entries.csv` (per entry), `P.OUT/ensemble_containment_summary.csv`; stdout | `summarise` on the published entries CSV: summary CSV identical (exact, 16 × 10). `collect --limit 400`: 400 rows × 79 columns identical (exact) to the same rows of the published entries CSV. Full collect not re-run (archive scan) |
| `apo_holo_state_distance.py` (575) | d′ between the apo and holo centres, trimmed central ranges (99/97/95%), sampling depth, separation; the model against member-to-member distances (within, cross-state, pooled), side-taking, position on the apo→holo axis | SI Table S7, "Distance between the states" (d′ medians 0.63/0.71/0.70/0.91, trimming shifts, ρ with separation, 50.6–56.9% / 60.8–68.9%) | `python apo_holo_state_distance.py collect --procs 10`; `python apo_holo_state_distance.py summarise > apo_holo_state_distance.out.txt` | as ensemble_containment (its curation is imported), plus `cfg.AHOJ_DB_DIR/Summaries/model_metrics/full_run_3/run_23082026_180200_do_collect_model_pLDDT/query_summaries_do_collect_model_pLDDT.csv`, `cfg.APOHOLO_METRIC_STATS_CSV`; `P.OUT/ensemble_containment_summary.csv`, `P.OUT/frontier_final_table.txt` | `P.OUT/apo_holo_state_distance_entries.parquet`, `P.OUT/apo_holo_state_distance_summary.csv`; stdout | `summarise` on the published parquet: summary CSV identical (exact, 1,049 × 6); stdout identical to `apo_holo_state_distance.out.txt`, 512/513 lines (the last, the `wrote <path>` line, differs only in the path). `collect --limit 2000`: 2,000 rows × 287 columns identical (exact). Full collect not re-run (archive scan, ~28 min on 12 processes) |
| `density_percentile.py` (411) | Leave-one-out KDE density of every conformer and the model's percentile among its conformers (combined, strongest, published pair); strata by class, separation, sampling, ratio, pLDDT, distance to the nearer state | SI Table S7, "Typicality" (median rank 0.250, 73.6% / 5.6% / 28.4%, 0.259 at pLDDT ≥ 95, 0.286 / 0.224, 0.517–0.615) | `python density_percentile.py collect --procs 22`; `python density_percentile.py summarise > density_percentile.out.txt` | balanced_classification's curation and job list; kde_core (`build_kde`, `classify_metric`); model-metrics harvest (as above); `cfg.CLASSIFICATION_SUMMARIES_CSV`, `cfg.ANALYSIS_UNIFIED_LEVELS_CSV`; `P.OUT/apo_holo_state_distance_entries.parquet` | `P.OUT/density_percentile_entries.parquet` (`…_limit<N>.parquet` with `--limit`), `P.OUT/density_percentile_summary.csv`; stdout | `summarise` on the published parquets: summary CSV identical (exact, 99 × 12); stdout identical to `density_percentile.out.txt`, 96/97 lines (the `wrote` path line differs). `collect --limit 1000`: 1,000 rows × 72 columns identical (exact). Full collect not re-run (42 min on 22 processes) |
| `balanced_classification.py` (1035) | Size-balanced reclassification (k = min(n_apo, n_holo + 1, 20) per class, 10 draws); known-state validation with one PDB entry held out of both classes; query-row check; site-level aggregation and Figure 4's estimator under both labels | SI Table S9 (`summarise`), SI Table S10a–d (`sites`), Supplementary Note 3 (label changes, k, ρ 0.47 → 0.19, query-row check 5.3% / 7.0%, draw noise 6.5% / 5.9%), the size-balanced labels used in Methods and main Results | `python balanced_classification.py collect --procs 22`; `… validate --n 20000`; `… query_row --n 2000`; `… summarise`; `AHOJ_AF_MODEL_SET=afdbv4 python balanced_classification.py sites` | as ensemble_containment; kde_api (`classify_point_kde_r3`); `sites`: separation_shape frames (`load_lbs`, `load_docking`), review_items (`load_st100`, `partial_spearman`), `P.OUT/downstream_members.csv` | `P.OUT/balanced_entries.csv`, `balanced_known_state_entry_out.csv`, `balanced_known_state.csv` (first design, kept for the design check), `balanced_query_row.csv`; summary CSVs `balanced_classification_summary.csv`, `balanced_by_separation_tier.csv`, `balanced_known_state_{summary,by_tier,by_ratio}.csv`, `balanced_prevalence_by_level.csv`, `balanced_evidence_test.csv`, `balanced_downstream_gaps.csv` (`--csv-dir` to redirect); stdout | `summarise` on the published per-entry CSVs: all five summary CSVs identical (exact). `sites` without the switch: the three committed CSVs identical (exact); with `afdbv4`, prevalence and evidence-test CSVs are the same and `balanced_downstream_gaps.csv` reproduces every number of SI Table S10c (87,091 / 60,988 pockets; consensus +16.8, +10.2, +8.4 [+7.1, +9.8], +5.3 [+4.0, +6.6], +3.2, +3.2; docking +19.4 … +7.9) — see Notes. `collect --limit 600` (600 × 15), `validate --n 300` (300 × 15) and `query_row --n 200` (200 × 11): identical (exact) to the same rows of the published files (both samples are prefixes of the published ones, same seeds). Full collect / validate / query_row not re-run (104 / 25 / few min on 22 processes) |
| `classifier_size_bias.py` (304) | Three measurements of the drift: `simulate` (pure null with the classifier's kde_core), `subsample` (real sites, model fixed, only the reference counts subsampled), `downstream` (site-level released vs balanced labels; leaves the independent re-draw) | Supplementary Note 3, "How large it is without AlphaFold" (91.0% / 53.8% / 9.5%; 71.8% / 31.8%) and "on real sites and models" (92.8% / 51.0% / 20.8%, flips 72.0%); the draw-noise re-draw behind Note 3 / Table S10 | `python classifier_size_bias.py simulate`; `… subsample`; `… downstream` | as balanced_classification (imports its curation and `_balanced`); `downstream`: `P.WD/lbs_generality/{p2rank,fpocket/full,ifsitepred/full}/lbs_*_st100_merged_per_pocket.csv`, `P.WD/docking/run3/analysis/docking_run3_merged.csv` | `P.OUT/size_bias_simulation.csv`, `size_bias_subsample.csv`, `size_bias_downstream.csv`, `downstream_members.csv`; stdout | `simulate` (full, 400 replicates): CSV identical (exact, 55 × 8). `downstream` (full, 25,381 member pockets, 16 min on 14 processes): `size_bias_downstream.csv` identical (exact, 24 × 7) and `downstream_members.csv` identical (exact, 25,381 rows) to the published re-draw. `subsample --sites 40`: 40 rows × 11 columns identical (exact) to the same sites of the published CSV (the sample is a prefix of the published 1,500, same seeds). Full subsample not re-run |
| `fig1b_site_lazou.py` (486) | The 2yiw A I46 1354 showcase entry: classification, pocket against the deposited struct_site, ligands across the entry, curated labels (asserts the released n_apo / n_holo), Lazou's Met198 RMSD to the open (2NPQ) and closed (2ZB1) references for the model and all 264 curated chains, occlusion of the I46 position | SI Figure S9 caption (148 apo + 117 holo states; the model's Met198 1.3 Å from 2ZB1 and 7.7 Å from 2NPQ) and main Figure 1b | `python fig1b_site_lazou.py > out.txt` (needs gemmi) | `cfg.AHOJ_OUTPUT_FIX_DIR/093/2yiw-A-I46-1354.tgz`, `cfg.CLASSIFICATION_SUMMARIES_CSV`, `cfg.EXAMPLE_ENTRIES_DIR/093/2yiw-A-I46-1354/statistics/classification_kde_pocket.json`, `webserver_results_dir/<job>/structure_files`, `cfg.DOCKING_REDOCK_CACHE_DIR/structures/2yiw.cif.gz`, `afdb_store_dir/v4/AF-Q16539-F1-model_v4.cif.gz`; ensemble_containment's parameters | `P.OUT/fig1b_site_lazou_met198.csv`; stdout | Full run (seconds): CSV identical to the committed `fig1b_site_lazou_met198.csv` (exact, 266 × 16); stdout prints n_apo 148 / n_holo 116 (+ query = 117), Met198 1.30 Å closed / 7.70 Å open, as quoted. No committed stdout |
| `ligand_relabel_criteria.py` (517) | The two single-atom exclusion lists (TOTAL_LE1 106 codes, HEAVY_LE1 123) read from the classification code, their call sites; per entry the relabel under no list / total_le1 / heavy_le1 with the curation around each; every old → new label transition | Supplementary Data 1 (106 / 123 / 17 codes); Methods "Apo/holo relabelling of candidate structures" (1,224,472 of 43,153,179 candidates, 56,189 entries, 79,779 query-ion entries, 54,639 / 13,177 entries, heavy_le1 +24,220 rows / 4,198 entries) | `python ligand_relabel_criteria.py lists`; `… collect --procs 22`; `… summarise`; `… transitions --procs 22` | `parsing_functions_DBv2.py` of the classification code (or `AHOJ_PARSE_AHOJ`); as ensemble_containment | `P.OUT/ligand_relabel_criteria_entries.csv`, `ligand_relabel_criteria_summary.csv`, `ligand_relabel_criteria_codes.csv`, `ligand_relabel_criteria_transitions.csv`; stdout | `lists`: 106 / 123 / 17 codes and the 17 HEAVY-only codes as in Supplementary Data 1. `collect --limit 3000` + `summarise`: total_le1 reproduces n_apo and n_holo for 100.000% of the sample (heavy_le1 98.2%); shares match the published full-set ones (relabelled 2.75% vs 2.84% of candidates, entries touched 23.5% vs 23.6%, query-ion 33.1% vs 33.5%, apo only via relabel 5.9% vs 5.5%, counts changed 22.9% vs 23.0%, heavy_le1 1.83% vs 1.76%); the summary CSV is flagged `is_sample` and is not comparable cell by cell. `transitions --limit 3000`: same transition types as the published CSV. Full collect / transitions not re-run (~32 / ~30 min on 22 processes): the summary and transitions CSVs were NOT reproduced in full. `lists` run against the moved copy dates the first `total_le1` call site to this repository's own import commit, not to parse-ahoj's history (set `AHOJ_PARSE_AHOJ` to a parse-ahoj checkout for that line) |
| `temporal_split.py` (629) | Release era of every conformer (E1 ≤ 2018-04-30, E2 to 2021-02-15, E3 later); which era's apo/holo balance goes with the class (thinned), which side of the E1/E3 medians the model is on, the model's own templates, sites with no seen structure; coverage of the balanced draws | Supplementary Note 5 (all tables and numbers); Supplementary Note 3, "73.7% … 47.8% of all conformers" | `python temporal_split.py dates`; `… templates`; `… collect --procs 22`; `… summarise > temporal_split.out.txt` | RCSB search API (`dates`); `afdb_store_dir/{v4,v3}` (`templates`); `pdb_resolutions_dir/entries.idx`; balanced_classification's curation and job list; `P.OUT/balanced_entries.csv`; `cfg.CLASSIFICATION_SUMMARIES_CSV`, `cfg.ANALYSIS_UNIFIED_LEVELS_CSV` | `P.OUT/temporal_split_release_eras.csv`, `temporal_split_templates.csv`, `temporal_split_entries.parquet` (`…_limit<N>` with `--limit`), `temporal_split_summary.csv`; stdout | `summarise` on the published inputs: summary CSV identical (exact, 159 × 40); stdout identical to `temporal_split.out.txt`, 164/165 lines (the `wrote` path line differs). `templates` (full, 2 min): CSV identical (exact, 109,655 rows). `collect --limit 1000`: 1,000 rows × 274 columns identical (exact). `dates` (live RCSB query, 2026-10-26): all 260,720 PDB IDs common to both runs have the same release year and era; the live set adds 826 entries released after 2026-10-05 and lacks 4 since obsoleted, so the file is date-dependent by design. Full collect not re-run (29 min on 22 processes) |
| `af_model_occlusion.py` (1060) | Model residues near the pocket that the experimental structures lack (SIFTS); the model's pocket SASA recomputed in PyMOL without them; re-classification with only the model's SASA replaced; site-level shares and Figure 4's estimator with and without the occluded pockets | Supplementary Note 1, "The model's own context" (16.5% of 235,451, 20.3%, 53 Å², 2.7% calls, 61.9 → 62.3%, 60.6 → 61.2%, gaps +8.4 / +8.8 / +8.2, docking +10.2 / +9.0, top 1% → 56%) and the 13.2% / 5.5% incomplete-pocket shares | `python af_model_occlusion.py sifts`; `… collect --procs 12`; `… verify`; `AHOJ_AF_MODEL_SET=afdbv4 python af_model_occlusion.py summarise` | `sifts_dir`; `afdb_store_dir/{v4,v3}`; `cfg.AHOJ_OUTPUT_FIX_DIR` (`global_results.csv`, `pocket_selections.csv`, `matrices/`); as ensemble_containment; separation_shape frames | `P.OUT/sifts_index_2026-09-13.npz`, `occlusion_entries.csv`, `af_model_occlusion_summary{_afdbv4}.csv`; stdout | `summarise` (afdbv4) on the published entries CSV: `af_model_occlusion_summary_afdbv4.csv` identical (exact, 131 × 53). `collect --limit 300` with the published SIFTS index: 300 rows × 92 columns identical (exact) apart from the timing column `t_s`; its fidelity lines read 100%. `verify`: 18 of 18 volumes and RoGs unchanged, SASA rises in all (e.g. 35.69 → 2733.86 Å² at 3mup D SMK 600), as quoted. `sifts`: NOT reproducible on this PC any more, because the SIFTS files in `sifts_dir` were refreshed on 2026-10-04 after the 2026-09-13 index was built (the index file keeps its 2026-09-13 name but the arrays differ); `collect` needs the published 2026-09-13 index. Full collect not re-run (~2 h on 12 processes) |

### Left out

| Script | Reason |
|---|---|
| `af_centric_rerun.py` | Scoping pilot of an AHoJ re-run with the AlphaFold model as the query (100 sites, 154 known-state chains, cost model). Its numbers (κ 0.54, 78.4 vs 70.0% balanced accuracy, 11,611 core-h, ...) appear in neither `main_draft_v4.md` nor `SI_draft_v4.md`; the 488 no-v6-model sites the SI quotes come from the AFDB v4 re-alignment scripts, not from this one |

### Notes

- `balanced_classification.py sites` writes `balanced_downstream_gaps.csv` under the same name with or without
  `AHOJ_AF_MODEL_SET=afdbv4`. The authors' working copy of that CSV was the v6 run (86,604 pockets, consensus +8.8);
  SI Table S10c quotes the afdbv4 run (87,091 pockets, consensus +8.4), which the moved script reproduces exactly.
  In the same afdbv4 run, the draw-noise block prints an independent-minus-collect consensus difference of
  −0.4 pp, while Supplementary Note 3 says "differs by 0.6 point", which is the v6 run's value.
- `balanced_classification.py sites` and `af_model_occlusion.py summarise` (afdbv4) were re-run a second time
  against group C's moved `separation_shape.py` in this folder: outputs identical to the first runs.
- `fig1b_site_lazou.py` still prints "SI Fig. S10d" in two labels of its stdout (the figure is now SI Figure S9);
  only the comments were renumbered, so that the stdout stays the same.

## B. Model provenance, detection and docking

Every script starts with `import ma_paths as P` and takes its roots from it. Everything these scripts write, their
regenerable caches included, goes to `P.OUT` (`AHOJ_AF_MANUSCRIPT_ANALYSES_DIR`, default
`<work_dir>/manuscript_analyses`): the summary tables and caches that sat beside the scripts keep their names in
`P.OUT`, and the caches that sat in `work_dir` keep their folder names under it (`P.OUT/afdb_v4_sources/`,
`P.OUT/afdb_v4_v6_diff/`, `P.OUT/rank_cut_afdbv4/site_ranks.csv`, `P.OUT/afdb_v4_realign/`). `afdbv4` below means
`AHOJ_AF_MODEL_SET=afdbv4`, the AlphaFold DB v4 model set the manuscript quotes.

### Anchors

All are optional in the sense that a script fails only in the mode that uses them (`P.anchor` returns a
`<UNSET:...>` placeholder otherwise).

| Key (paths.ini) | Env var | Points at | Used by |
|---|---|---|---|
| `afdb_store_dir` | `AHOJ_AFDB_STORE_DIR` | the AFDB model store: `v4/` and `v3/` (the 27,615 v2c models copied from Google Cloud, with `afdb_v2c_store_manifest.csv`) beside `v6/` (the v6 files AFDB serves today) | afdb_v4_sources (`STORE`, `AFDB_V6 = <store>/v6`), afdb_v4_v6_diff and afdb_v4_realign through it, alphafill_model_version (`--fetch`) |
| `afdb_v4_sources_dir` | `AHOJ_AFDB_V4_SOURCES_DIR` | extraction cache of afdb_v4_sources: `v4_ftp/` (members pulled from the EBI tars), `v6/`, `structures/` (query crystals), `tar_headers/`, `gcs_fetch/`, `gcs_check/`, `pymol_fetch/` | afdb_v4_sources, afdb_v4_v6_diff |
| `webserver_cache_dir` | `AHOJ_WEBSERVER_CACHE_DIR` | the AHoJ webserver's local caches: `AF/` (14 local v4 model files), `structures/` (query crystals) | afdb_v4_sources (`tars`, `store`, `validate`, `report` live probes) |
| `af_centric_pilot_dir` | `AHOJ_AF_CENTRIC_PILOT_DIR` | the AF-centric re-run pilot: `workdir/structures/` (cached query crystals), `code_5c3a191/alignment_executables/USalign_20240319.exe` | afdb_v4_sources (`validate`), afdb_v4_v6_diff (crystal sizes in `report`) |
| `alphafill_databank_dir` | `AHOJ_ALPHAFILL_DATABANK_DIR` | the local AlphaFill 2.1.1 databank (alphafill.eu bulk tar, extracted; `<xx>/AF-<acc>-F1-filled_v4.cif.gz`) | afdb_v4_sources (`alphafill`, `validate`, `report`) |
| `p2rank_dev_eval_dir` | `AHOJ_P2RANK_DEV_EVAL_DIR` | the P2Rank 2.6-alpha dev build's `test_output/` (the `UBS2_run3_*` eval runs and their `observed_sites.csv`) | afdb_v4_realign (`p2rank`, `report`) |
| `lbs_raw_v6_dir` | `AHOJ_LBS_RAW_V6_DIR` | the July (v6) raw fpocket and IF-SitePred pocket lists as pulled from the cluster (`fpocket_raw_full.csv`, `ifsitepred_raw_full.csv`) | afdb_v4_realign (`fpocket`, `ifsitepred`, `report`) |
| `cryptobench_dir` | `AHOJ_CRYPTOBENCH_DIR` | the CryptoBench dataset (OSF pz4a9, folder `cryptobench-dataset`: `dataset.json`, `auxiliary-data/non-cryptic-pockets/noncryptic-pockets.json`) | cryptobench_external_label |


Plain environment variables, not anchors (defaults need no configuration): `AHOJ_AF_PY311` (the python whose
`bin/` holds msms for PyVOL; default `sys.executable`), `AHOJ_AF_GCLOUD` (default `gcloud` on `PATH`),
`AHOJ_AF_FOLDCOMP_PY` (a python with `foldcomp`; unset skips the Foldcomp fidelity check of `mirrors`).

### Run order and intermediates

- `afdb_v4_sources.py` caches first (`v2c_scan`, `sets`, `uniprot`, `v6hdr`, `alphafill`, `gcs`, `tars`, `store ...`,
  `validate`), then `afdb_v4_v6_diff.py` (`atoms`, `api`, `scan`, `pockets`, `worklist`), whose
  `afdb_v4_v6_accessions.csv` and `afdb_v4_v6_diff/v4_v6_atoms.csv` are read by `afdb_v4_realign.py` and
  `alphafill_model_version.py`.
- `detection_rank_cut.py` (`ranks` or `all`) writes `rank_cut_afdbv4/site_ranks.csv`, read by `ifsitepred_cap.py`
  (and by group C's left-out `fig3a_neff.py`, which imports `cfg, G, S, HERE, RANKS_CSV, load, PRED` from
  `ifsitepred_cap`; those names are kept). `detection_rank_cut.py` imports `tool_gap_reconciliation.py` (`log_or`,
  `rank_share`), `separation_shape.py`, `balanced_classification.py` and `evidence_amount_vs_ratio.py`; group C's
  `density_exclusion_gaps.py` imports `detection_rank_cut.two_label_frame`.
- Intermediates of other groups' scripts that these read from `P.OUT`: `balanced_entries.csv` and
  `balanced_downstream_gaps.csv` (balanced_classification.py), `density_separation_gaps_afdbv4.csv`,
  `evidence_amount_vs_ratio_afdbv4.csv`, `ensemble_containment_entries.csv`; and in `P.OUT/afdb_v4_realign/` the saved
  stdout of `separation_shape.py` under each setting (`separation_shape_afdbv4.txt`, `separation_shape_v6.txt`) and
  `balanced_sites_csv/balanced_downstream_gaps.csv` (`balanced_classification.py sites --csv-dir
  <P.OUT>/afdb_v4_realign/balanced_sites_csv` under afdbv4). On the author's PC these sat in
  `<work_dir>/afdb_v4_realign/`.

### Scripts

| File (lines) | Computes | Backs | Command | Inputs | Outputs | Verification |
|---|---|---|---|---|---|---|
| `ahojdb_release_residual.py` (212) | Processed (run_3 job lists) against released (db_stats.json, release tar listing, webserver entry endpoint) AHoJ-DB v2c sets and their overlap | Methods, Study design: 515,463 / 121,029 / 37,620 / 29,463 released; 516,794 / 121,227 / 37,662 processed; "overlap in 514,529 entries, 99.8% of the release (1,327 … 938 … 934 …)" | `python ahojdb_release_residual.py` (`--api` re-probes the endpoint, a few thousand GETs) | `cfg.AHOJ_DB_DIR`: `Products/.../full_run_3_merged/{Success,Discarded}_jobs.txt`, `Database/ahoj_db_v2c_tar.log`; `P.OUT/ahojdb_v2c_db_stats.json` (fetched live, cache as fallback); `P.OUT/ahojdb_release_residual_api.json` (cache of 3,416 endpoint statuses) | stdout; the two JSON caches | Original (private copy) and moved copy: stdout identical, 34/34 lines; the author's 16 expected lines (`check_verify_marks_numbers.py` patterns) all present. db_stats.json fetched live and equal to the cache |
| `alphafill_model_version.py` (255) | Which AFDB model each AlphaFill JSON was computed on (archive v4 / archive undated / on demand after v6), joined to the v4/v6 identity; Figure 4a medians without the on-demand sites; `--fetch N` compares served files with the stores | Methods, AlphaFill matching: 84% / 16% (2,849 sites on 1,325 accessions), 2,540 coordinate-identical, at most 309 sites (1.8%), no tier median moves by more than 0.014; Data availability; SI Note 4 | `python alphafill_model_version.py` (`--fetch 40` re-samples alphafill.eu) | `cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR/per_ubs_summary_same_uniprot_ge_0p30.csv`, `cfg.ALPHAFILL_JSONS_DIR`, `P.OUT/afdb_v4_v6_diff/v4_v6_atoms.csv`; `afdb_store_dir` (`--fetch`); `P.OUT/alphafill_model_version_fetch.json` | stdout; the fetch cache | Original and moved copy: stdout identical, 42/42 lines; 17/17 expected lines present |
| `predictor_versions.py` (125) | fpocket version and IF-SitePred commit of the cluster images, from conda-forge's upload history and the GitHub commit history against the build date | Methods, Binding-site prediction: fpocket 4.2.3 (conda-forge build), IF-SitePred commit fb9d778 (25 March 2024) | `python predictor_versions.py` (`--refresh` re-fetches; `--local-wsl` reads the WSL rehearsal) | `P.OUT/predictor_versions_cache.json` (the two public histories; fetched and written when absent) | stdout; the cache | Original and moved copy: stdout identical, 14/14 lines; 7/7 expected lines present |
| `docking_implausible_affinities.py` (151) | Share and character of Uni-Dock affinities outside [−15, 0] and [−20, 0); the three docking estimands under "fail" and "drop" rules | Methods, Molecular docking: 1,645 of 69,738 (2.4%), 1,198 positive (median 40 heavy atoms against 24), 447 below −15; the moves (< 1 point, < 0.2 points, δ 0.006, 0.3 points); Cliff's δ −0.27 (−0.269) and 59.2% of 3,422 | `python docking_implausible_affinities.py` (`--run run3` for July's v6 run) | `P.WD/docking/run3_afdbv4/analysis/docking_run3_merged.csv` | stdout | Original and moved copy: stdout identical, 26/26 lines; 14/14 expected lines present |
| `c41_cut_sweep2.py` (57) | Mean per-pocket dockable fraction by class as the affinity cut runs from −4 to −9 kcal/mol; threshold-free Cliff's δ of the pocket median affinity | SI Table S4 (every row); Methods "positive at every cut between −4 and −9" | `AHOJ_AF_MODEL_SET=afdbv4 python c41_cut_sweep2.py` (defaults; or `<docking_run3_merged.csv> <out.csv>`) | `P.WD/docking/run3{_afdbv4}/analysis/docking_run3_merged.csv` | `P.OUT/c41_cut_sweep{_afdbv4}.csv`; stdout | afdbv4: `c41_cut_sweep_afdbv4.csv` identical to the committed one (exact, 8 × 7), stdout identical to the original's; v6: `c41_cut_sweep.csv` identical (exact, 8 × 7) |
| `c41_affinity_cut.py` (64) | Pose correctness (redock RMSD ≤ 2 Å) against the −6 kcal/mol cut on good-frame jobs; affinity percentiles of correct poses | SI Table S4 text: 65,414 good-frame jobs, 72.2% of correct poses (n = 5,489) and 61.3% of the others (n = 59,925) score ≤ −6; Methods "27.8% … score worse than −6 kcal/mol" | `AHOJ_AF_MODEL_SET=afdbv4 python c41_affinity_cut.py` (or a `redock_rmsd.csv` path) | `P.WD/docking/run3{_afdbv4}/reference_poses/redock_rmsd.csv` | stdout | No committed output. Original and moved copy on afdbv4: stdout identical, 33/33 lines; prints 65,414 / 5,489 / 72.2% / 59,925 / 61.3% / 27.8% |
| `e3_manuscript_numbers.py` (219) | The pooled detection and docking numbers for one model set (Figure 3a overall bars, denominators, missed by all three, docking medians); `--both` gates v6 against the pre-re-run values | Main text and Methods denominators: 87,434 sites (35,124 AF-Apo, 52,014 AF-Holo, 296 tied), docking 61,216 G1 pockets (61,012 classified); pooled rates of Figure 3a; SI Table S2 "detected by no predictor" | `AHOJ_AF_MODEL_SET=afdbv4 python e3_manuscript_numbers.py`; `python e3_manuscript_numbers.py --both` | `P.WD/lbs_generality/{p2rank,fpocket/full,ifsitepred/full}{_afdbv4}/…merged_per_pocket.csv`, `P.WD/docking/run3{_afdbv4}/analysis/{pocket_level_table,docking_run3_merged}.csv` | stdout | No committed output. Original and moved copy: afdbv4 32/32 lines and `--both` 105/105 lines identical; the built-in v6 gate prints PASS |
| `afdb_v4_sources.py` (2,304) | Where the v2c AFDB v4 files can still be obtained (Google Cloud bucket, EBI FTP tars by byte range, carried-over v6, AlphaFill, third parties); which v6 models are carried over or re-predicted; the v2c model store copied from Google Cloud, verified, and its manifest; validation against archived v2c jobs. Also the path/helper module of afdb_v4_v6_diff and afdb_v4_realign | Data availability (v4 no longer served per accession; Google Cloud dataset and EBI FTP archives; the deposited model files with their manifest of accession, version and checksum, `afdb_v2c_store_manifest.csv` from `store verify`); Methods and SI Note 4 "version 6 re-predicted … Swiss-Prot and reference-proteome entries and kept … the rest"; G3FFN6 the one v3 model | `python afdb_v4_sources.py report > afdb_v4_sources.out.txt` after the cache modes listed in the docstring (`v2c_scan`, `sets`, `uniprot`, `v6hdr`, `alphafill`, `gcs`, `tars`, `validate`, `store list/fetch/verify`, `validate --arm store --n 200`, `mirrors`) | `cfg.AHOJ_OUTPUT_FIX_DIR` (518,883 v2c archives), `cfg.CLASSIFICATION_SUMMARIES_CSV`, `cfg.ANALYSIS_UNIFIED_LEVELS_CSV`, `P.WD/lbs_generality/p2rank/lbs_p2rank_st100_merged_per_pocket.csv`; anchors `afdb_store_dir`, `afdb_v4_sources_dir`, `webserver_cache_dir`, `af_centric_pilot_dir`, `alphafill_databank_dir`; network (EBI FTP, Google Cloud listing, UniProt REST, AFDB API, third parties); `P.OUT/afdb_v4_tar_sample.csv`, `afdb_v4_validation_jobs.csv`, `afdb_v4_store_validation_jobs.csv` (the fixed samples) | `P.OUT/afdb_v4_sources/*.csv` caches; the three sample CSVs; `<afdb_store_dir>/{v4,v3}/` and `afdb_v2c_store_manifest.csv` (`store fetch/verify`) | `report --offline` on the author's caches (copied with their dates): original and moved copy identical, 121/121 lines. `sets` and `v6hdr` re-run: `accession_sets.csv` (27,615 × 7) and `local_v6_headers.csv` (10,554 rows; compared sorted, the pool returns files in listing order) identical to the work_dir caches (exact). Not re-run: `v2c_scan` (all 518,883 archives), `uniprot`, `gcs`, `tars`, `mirrors` and `report`'s dated live probes (network), `validate` (PyMOL/PyVOL, query-crystal downloads), `store fetch/verify` (`verify` writes the manifest into the store) |
| `afdb_v4_v6_diff.py` (932) | How many v2c models changed between v4 and v6 (exact atom identity of every local pair; API dates for the rest); pocket-level movement; the per-entry and per-site refresh worklists | Methods and SI Note 4 (which models version 6 re-predicted); the per-accession identity behind SI Note 4's strata and the AlphaFill bound (through afdb_v4_realign and alphafill_model_version) | `python afdb_v4_v6_diff.py report > afdb_v4_v6_diff.out.txt` after `atoms`, `api`, `scan`, `pockets`, `worklist` | afdb_v4_sources paths and caches; `P.WD/afdb_dataset/st100_run3/afdb_st_100_dataset.csv`; `P.WD/af_centric_rerun/model_row_refresh_v6.csv` (the modelv6 prototype's rows; read if present) | `P.OUT/afdb_v4_v6_diff/` caches (`v4_v6_atoms.csv`, `afdb_api_nonlocal.csv`, `entry_model_pockets.csv`, `pocket_movement.csv`, `st100_site_pockets.csv`, `entry_worklist.csv`, `st100_site_worklist.csv`); `P.OUT/afdb_v4_v6_{diff_summary,tiers,accessions}.csv` | `report`: original and moved copy identical, 104/104 lines, and identical to the author's saved `report_2026-09-21.txt` but for the `wrote` path line; its three tables identical to the committed ones (exact and byte-identical: 24 × 3, 8 × 15, 10,554 × 14). `worklist` re-run (5.5 min): `entry_worklist.csv` (237,922 × 12) and `st100_site_worklist.csv` (105,713 × 11) identical to the work_dir caches (exact). Not re-run: `atoms`, `pockets` (every model pair), `scan` (237,922 archives), `api` (AFDB API) |
| `afdb_v4_realign.py` (725) | Detection and docking re-run on the v4 models: validation gates on coordinate-identical models, coverage, sequence-changed sites, docking calibration noise, and the v4-against-v6 sensitivity table with paired cluster-bootstrap intervals | SI Note 4: its table (`afdb_v4_realign_sensitivity.csv`), the 2,746 coordinate-identical accessions (exact gate), 488 sites on 82 accessions without a v6 model, 78 sequence-changed accessions with 847 sites, 63.3%, 338 sites, the 4.3% / 5.7% / 5.4% docking noise floor, 99.8% / 99.9% on identical models | `python afdb_v4_realign.py geometry`, `sensitivity`, `seqchanged` (needs gemmi), then `python afdb_v4_realign.py report > afdb_v4_realign.out.txt` | `P.OUT/afdb_v4_v6_accessions.csv`; `P.WD`: `afdb_dataset/st100_run3{,_afdbv4}/…with_geometry.csv`, `P2Rank/ubs_run_5{,_afdbv4}/…`, `lbs_generality/…{,_afdbv4}`, `docking/run3{,_afdbv4}/…`; anchors `p2rank_dev_eval_dir`, `lbs_raw_v6_dir`, `afdb_store_dir` (through afdb_v4_sources); `P.OUT/afdb_v4_realign/` printouts and `P.OUT/balanced_downstream_gaps.csv` (`downstream` section) | `P.OUT/afdb_v4_realign_{residue_coverage,sensitivity,seqchanged_sites}.csv`; stdout | `geometry`, `sensitivity`, `seqchanged` (py311 env for gemmi) re-run: the three CSVs identical to the committed ones (exact: 78 × 7, 54 × 20, 847 × 6; they differ from the working copies only in line endings). `report`: original and moved copy identical, 265/265 lines; the author's checker `check_afdb_v4_realign_numbers.py` against the results write-up: 5 CI triples, 439 decimals, 143 counts matched, 0 unmatched |
| `tool_gap_reconciliation.py` (444) | Level against slope of the class gap for the three predictors on one row set (raw and standardized, both axes, arms, composition) | None of its numbers is quoted (it ran on the July v6 tables). Included because `detection_rank_cut.py` imports its `log_or` and `rank_share` | `python tool_gap_reconciliation.py` | separation_shape frames; `P.WD/lbs_generality/{p2rank,fpocket/full,ifsitepred/full}/…merged_per_pocket.csv` | `P.OUT/tool_gap_reconciliation.csv`; stdout | Identical to the committed CSV (exact, 296 × 8). Run time 1.5 min |
| `detection_rank_cut.py` (1,044) | Detection under criteria that count each site once (top-n_eff, competition-free top-k, any rank) beside DCA4-TNS: per-site ranks, mechanism, Figure 4's estimator per criterion by separation tier, SI Table S11 per criterion | SI Table S12 (upper table and the ranking-given-detection sentence) | `AHOJ_AF_MODEL_SET=afdbv4 python detection_rank_cut.py all > detection_rank_cut_afdbv4.out.txt` (~35 min; `ranks`, `mechanism`, `tiers`, `s11` separately) | `cfg.AFDB_ST100_DATASET_GEOM_CSV`, `cfg.LBS_RECEPTORS_CSV`, `cfg.LBS_GROUND_TRUTH_NPZ`, `cfg.lbs_tool_dir(...)/predictions.csv`, the scored per-pocket tables (through analyze_fig6a_quality_gates); `P.OUT`: `balanced_entries.csv`, `density_separation_gaps_afdbv4.csv`, `evidence_amount_vs_ratio_afdbv4.csv`, `afdb_v4_realign/balanced_sites_csv/balanced_downstream_gaps.csv` (gates) | `P.OUT/rank_cut_afdbv4/site_ranks.csv`; `P.OUT/detection_rank_cut_{mechanism,tiers,s11}_afdbv4.csv`; stdout | `all` re-run on afdbv4 (31 min): `detection_rank_cut_{mechanism,tiers,s11}_afdbv4.csv` identical to the committed ones (exact: 630 × 5, 930 × 11, 903 × 11); the rebuilt `site_ranks.csv` identical to the work_dir cache (exact, 87,434 × 18); stdout identical to the committed `detection_rank_cut_afdbv4.out.txt`, 1,177 lines, but for the `[OK]` line naming the site_ranks location; built-in gates pass; the author's checker `check_detection_rank_cut_numbers.py`: 202 CI triples, 1,001 decimals, 9 counts, 0 unmatched |
| `ifsitepred_cap.py` (394) | What IF-SitePred's three-site output cap does: proteins by distinct regions, rates and raw / standardized gaps on the subsets where the cap can and cannot bind, other tools held to three sites | SI Table S12, the IF-SitePred cap paragraph (1,222 proteins, 34,141 pockets, 39%; +12 / +36 points; +13.6 / +14.1 / +16.2; +2.6, +2.2, +2.4, +1.7; 1.9 [0.4, 3.4]; +0.3) | `AHOJ_AF_MODEL_SET=afdbv4 python ifsitepred_cap.py > <P.OUT>/ifsitepred_cap_afdbv4/out.txt` | `06_lbs_generality/analysis/analyze_fig6a_quality_gates` (detection base, scored tables), `cfg.AFDB_ST100_DATASET_GEOM_CSV`, `cfg.LBS_RECEPTORS_CSV`; `P.OUT/rank_cut_afdbv4/site_ranks.csv`; `P.OUT/density_separation_gaps_afdbv4.csv` | `P.OUT/ifsitepred_cap_afdbv4/{proteins,counts,subset_rates}.csv`; stdout | The three CSVs identical to the committed ones (exact: 10,501 × 23, 67 × 5, 21 × 25); stdout identical to the committed `out.txt`, 128/128 lines (paths and timings masked). Run time 1 min |
| `cryptobench_external_label.py` (615) | CryptoBench's cryptic / non-cryptic pairs mapped to analysis entries; descriptor AUCs, pLDDT beyond diversity, the model's state at cryptic sites, separation, detection and docking gaps at cryptic sites | SI Table S13 (both tables and the text); main text "the AFDB model shows the ligand-bound pocket at 72% of them" | `python cryptobench_external_label.py summarise > cryptobench_external_label.out.txt` (afdbv4 is its default) | `cryptobench_dir`; `cfg.ANALYSIS_UNIFIED_LEVELS_CSV`, `cfg.CLASSIFICATION_SUMMARIES_CSV` (through balanced_classification), `cfg.CLASSIFICATION_MERGED_DIR/query_metrics_do_kde_plots_and_classification.csv`, `cfg.AHOJ_DB_DIR/Summaries/model_metrics/full_run_3/run_23082026_180200_do_collect_model_pLDDT/…`; `P.OUT/balanced_entries.csv`, `P.OUT/ensemble_containment_entries.csv`; separation_shape frames | `P.OUT/cryptobench_external_label_{summary,entries}.csv`; stdout | Both CSVs identical to the committed ones (exact: 429 × 7, 14,827 × 32); stdout identical to the committed `.out.txt`, 213/213 lines but the `wrote` path line. Run time 1 min |

### Left out

None of the group's 14 scripts. Three cluster-planning and review-bookkeeping scripts were excluded
beforehand, and the `check_*_numbers.py` checkers are not copied (they were used read-only in the verification).

### Caches the published numbers depend on

These are inputs, not code; they belong in the data deposit (or are regenerated against live services that may
have changed since):

- `ahojdb_release_residual_api.json` (HTTP status of the 1,327 empty-table jobs and the 2,089 tar-only archives at
  the webserver's entry endpoint): the 934 / 938 / 514,529 of the Methods. `--api` re-probes the live server.
  `ahojdb_v2c_db_stats.json` is the released statistics (fetched live when reachable).
- `predictor_versions_cache.json` (conda-forge upload history of fpocket, GitHub history of binding-sites): the
  Methods' fpocket 4.2.3 and commit fb9d778.
- `alphafill_model_version_fetch.json` (40 alphafill.eu files compared with the stores): prints the sample result;
  the Methods numbers do not depend on it.
- `afdb_v4_sources/` caches (112 MB in the author's work_dir: UniProt facts at one release, the Google Cloud
  listing, the EBI tar membership sample, the store listing and verification) and the three fixed samples
  `afdb_v4_tar_sample.csv`, `afdb_v4_validation_jobs.csv`, `afdb_v4_store_validation_jobs.csv`. The store manifest
  `afdb_v2c_store_manifest.csv` is already in the deposit with the model files.
- `afdb_v4_v6_diff/afdb_api_nonlocal.csv` (AFDB API facts for the accessions without a local v6 file); the other
  `afdb_v4_v6_diff/` caches are regenerable from the store and the v6 files.

### Caveats

- `afdb_v4_sources.py`, `afdb_v4_v6_diff.py` and `tool_gap_reconciliation.py` still name internal write-ups in a few
  `print` strings (e.g. "AF_CENTRIC_RERUN.md model-row refresh"); they were kept so the stdout matches the
  originals line for line. Docstrings and comments were rewritten.
- Backslash-joined relative paths (`r"afdb_dataset\st100_run3\..."`) were split into `os.path.join` components so
  the scripts also run off Windows; nothing else in the code changed beyond the roots.
- `afdb_v4_realign.py` reads the July raw pocket lists through the `lbs_raw_v6_dir` anchor because the folder that
  holds them on the author's PC is named after the cluster.
- `c41_cut_sweep2.py` and `c41_affinity_cut.py` took their input (and output) paths only as arguments; they now
  default to the work_dir tables and `P.OUT`, following `AHOJ_AF_MODEL_SET`. Arguments still override.
- Packages beyond `requirements.txt`, needed only by `afdb_v4_sources.py validate`: `pymol-open-source` and PyVOL
  (with msms); `foldcomp` (separate python) for `mirrors`; the `gcloud` CLI for `store list/fetch` of the v3 bucket.

## C. Separation, density, evidence and pLDDT

Every script starts with `import ma_paths as P` and takes its roots from it. Outputs and the intermediates these
scripts share go to `P.OUT` (`AHOJ_AF_MANUSCRIPT_ANALYSES_DIR`, default `<work_dir>/manuscript_analyses`). A
positional output or intermediate argument that is a bare file name is placed in `P.OUT`; an absolute path is used
as given. `afdbv4` below means `AHOJ_AF_MODEL_SET=afdbv4`, the AlphaFold DB v4 model set the manuscript quotes.
No new anchors: every root is `P.WD`, `P.REPO` or a named path of `common/config.py`.

### Run order

`frontier_at_scale.py` first (it writes `frontier_per_pocket.csv`, read by five others), and
`density_separation_gaps.py` (afdbv4) before `density_exclusion_gaps.py`. `separation_shape.py` is also a library:
`off_frontier.py`, `density_exclusion_gaps.py` and several group A/B scripts import it.

### Scripts

| File (lines) | Computes | Backs | Command | Inputs | Outputs | Verification |
|---|---|---|---|---|---|---|
| `separation_shape.py` (870) | Figure 4's frames (`load_lbs`, `load_docking`) and estimator (`Gap`: size-standardized AF-Holo − AF-Apo gap, cluster bootstrap over UniProt, B = 400); Q1–Q4: shape of low separation, separation × confidence, which arm moves, gap left at very low separation | Figure 4c/d per-predictor series; main text "no single predictor's gap differs from zero" at very low separation, "AF-Apo falls from 56.7% to 48.9%, AF-Holo rises from 59.1% to 67.1%"; SI Table S3 per-predictor series. Library for off_frontier, density_exclusion_gaps and others | `AHOJ_AF_MODEL_SET=afdbv4 python separation_shape.py > separation_shape_afdbv4.out.txt` | `P.WD`: `lbs_generality/{p2rank,fpocket/full,ifsitepred/full}{_afdbv4}/…merged_per_pocket.csv`, `docking/run3{_afdbv4}/analysis/docking_run3_merged.csv`, `P2Rank/ubs_run_4/analysis_v1/tables/p2rank_st100_descriptor_features_per_ubs.csv` | stdout only | No committed output. Original (private copy) and moved copy re-run on afdbv4: stdout identical, 358/358 lines |
| `off_frontier.py` (836) | Distance to the nearer state (`offmode`: model to the nearer apo/holo class mean in within-class SD, mean of four metrics); its downstream penalty by quintile and top decile, against separation, confidence, pLDDT, evidence depth | Supplementary Note 2 and Table S8; main text ρ = −0.74 with density, consensus −9.0 pp [7.1, 11.0] and docking −11.7 [8.6, 14.8] in the top decile, ρ = −0.04 (pLDDT) and +0.21 (winning probability) | `AHOJ_AF_MODEL_SET=afdbv4 python off_frontier.py off_frontier_afdbv4.csv > off_frontier_afdbv4.out.txt` (optional 2nd arg: a parquet cache of the per-job block) | separation_shape frames; `cfg.CLASSIFICATION_MERGED_DIR` (query summaries + metrics), `cfg.APOHOLO_METRIC_STATS_CSV`, `cfg.AHOJ_DB_DIR/Summaries/model_metrics/full_run_3/run_23082026_180200_do_collect_model_pLDDT/query_summaries_do_collect_model_pLDDT.csv`, `P.WD/unified_output/run5/analysis_unified_levels.csv` | `P.OUT/off_frontier{_afdbv4}.csv`; stdout | afdbv4: `off_frontier_afdbv4.csv` identical to the committed one (exact, 35 × 12); stdout identical to `off_frontier_afdbv4.out.txt` on all 495 committed lines, plus the two trailing lines of the CSV-write notice (the committed log was taken from a run without the CSV argument). v6: `off_frontier.csv` identical (exact, 35 × 12). Run time about 6 min each |
| `density_separation_gaps.py` (106) | Raw and size-standardized gap per separation and per density tier, three predictors + consensus + docking | SI Table S3 right-hand table, the per-predictor and raw series and the density-tier shares under it | `AHOJ_AF_MODEL_SET=afdbv4 python density_separation_gaps.py` (also run without the switch for the v6 table) | `06_lbs_generality/analysis/analyze_fig6a_quality_gates.load_base()`, `04_p2rank/analysis/confidence_labeling`, `cfg.DOCKING_RUN3_DIR/analysis/docking_run3_merged.csv` | `P.OUT/density_separation_gaps{_afdbv4}.csv` (read by density_exclusion_gaps) | Both CSVs identical to the committed ones (`assert_frame_equal`, exact): `density_separation_gaps_afdbv4.csv` and `density_separation_gaps.csv`, 50 × 14 each |
| `density_exclusion_gaps.py` (253) | Pooled gap and separation gradient with and without the very-low / low density pockets, released and size-balanced labels, continuous density floor; gated against Table S3's density column | SI Table S3 caption ("consensus +8.4 to +8.1, docking +10.2 to +10.0"); main text "Leaving them out leaves the class gap unchanged" | `AHOJ_AF_MODEL_SET=afdbv4 python density_exclusion_gaps.py > density_exclusion_gaps_afdbv4.out.txt` | separation_shape frames; `detection_rank_cut.two_label_frame` (group A/B); `P.OUT/density_separation_gaps_afdbv4.csv` | `P.OUT/density_exclusion_gaps{_afdbv4}.csv`; stdout | afdbv4: identical to the committed `density_exclusion_gaps_afdbv4.csv` (exact, 325 × 16) and `.out.txt` (238/238 lines, the `[OK]` path line masked); the built-in gate against Table S3 passes. `detection_rank_cut.py` and `tool_gap_reconciliation.py` were not yet in this folder, so this run imported the authors' unmoved copies from a scratch directory on `PYTHONPATH`; `P.OUT` was seeded with `balanced_entries.csv` (from `<work_dir>/balanced_classification/`, read by `balanced_classification.py`). Re-run once the moved copies are in. Run time about 3 min |
| `c30_separation_share.py` (61) | Share of pockets per separation tier at each clustering level | SI Table S3 left-hand table; main text "12.0% … at very low separation (25.9% including low)" | `python c30_separation_share.py` | `cfg.ANALYSIS_UNIFIED_LEVELS_CSV`, `04_p2rank/analysis/confidence_labeling` | `P.OUT/c30_separation_share.csv` | Identical to the committed CSV (exact), 6 × 13 |
| `density_tier_shares.py` (27) | Share of pockets in the very-low and low density tiers per clustering level | Results, "low-density pockets, about one in ten" | `python density_tier_shares.py > density_tier_shares.out.txt` | `cfg.ANALYSIS_UNIFIED_LEVELS_CSV` | stdout | Identical to the committed `density_tier_shares.out.txt`, 8/8 lines (the `source:` path line masked) |
| `frontier_at_scale.py` (121) | Per pocket: model placement, inside [min, max], centrality z and frontier for volume / SASA / RoG; summary table | Builds the per-pocket table read by frontier_final, frontier_rms, density_vs_ssf, plddt_headtohead, plddt_deep; summary behind SI Table S5 | `python frontier_at_scale.py` | `cfg.ANALYSIS_UNIFIED_LEVELS_CSV`, `cfg.CLASSIFICATION_MERGED_DIR` (query summaries + metrics), `cfg.APOHOLO_METRIC_STATS_CSV` | `P.OUT/frontier_per_pocket.csv` (intermediate, 237,922 rows, ~89 MB; name chosen here, the original was not kept), `P.OUT/frontier_summary.csv` | `frontier_summary.csv` identical to the committed one (exact), 3 × 11 |
| `frontier_final.py` (61) | Final per-metric frontier table for all four metrics with one denominator rule; complete-case conjunctions | SI Table S5 and the counts under it; Results 55–66% frontier, 54–67% within ± 1 SD | `python frontier_final.py > frontier_final_table.txt` | `P.OUT/frontier_per_pocket.csv`; model-metrics harvest (as off_frontier); `cfg.APOHOLO_METRIC_STATS_CSV` | stdout | Identical to the committed `frontier_final_table.txt`, 17/17 lines |
| `frontier_rms.py` (79) | Pocket-RMSD frontier and its redundancy with the class label | SI Table S5 note: ρ = +0.06 with AF-Holo, frontier 62.7% AF-Apo vs 68.4% AF-Holo | `python frontier_rms.py` | model-metrics harvest; `P.OUT/frontier_per_pocket.csv`; `cfg.APOHOLO_METRIC_STATS_CSV` | `P.OUT/frontier_rms.csv`; stdout | No committed output. Original and moved copy on the same inputs: stdout identical (21/21 lines, `[OK]` path masked), CSV identical (exact, 237,922 × 5). Prints the quoted +0.061, 62.7%, 68.4% |
| `density_vs_ssf.py` (77) | Density against the frontier statistics (containment, centrality, frontier count), by tier and by conformer count | Results "Density falls as the model moves away from that spread (ρ = −0.68)"; SI Table S5 note ρ = −0.68 | `python density_vs_ssf.py` | `P.OUT/frontier_per_pocket.csv`, `cfg.ANALYSIS_UNIFIED_LEVELS_CSV` | stdout | No committed output. Original and moved copy on the same inputs: identical, 29/29 lines. Prints ρ = −0.679 |
| `plddt_headtohead.py` (72) | Spearman ρ of pocket pLDDT with diversity and evidence variables; partial correlations ratio \| diversity and diversity \| ratio | SI Table S6 first table; +0.007 and −0.490 under it; main text ρ = −0.49, +0.07, +0.01 | `python plddt_headtohead.py` | `P.OUT/frontier_per_pocket.csv` | `P.OUT/plddt_headtohead.csv`; stdout | Identical to the committed CSV (exact), 16 × 6 |
| `plddt_deep.py` (89) | pLDDT against four spread definitions per metric, composite, effect sizes, within-band ρ | Methods "pLDDT and conformational diversity": CV of RoG ρ = −0.37; the size-proxy argument under SI Table S6 | `python plddt_deep.py` | `P.OUT/frontier_per_pocket.csv` | stdout | No committed output. Original and moved copy on the same input: identical, 40/40 lines. Prints CV RoG −0.372 |
| `evidence_amount_vs_ratio.py` (248) | Partial ρ of success with ratio / p_holo / class under no control, size, size + total evidence; the total's own effect at fixed ratio and size; paired cluster bootstrap | SI Table S11; the "total evidence, ratio fixed" column of SI Table S12 comes through `detection_rank_cut.py` (group A/B), which imports this module | `AHOJ_AF_MODEL_SET=afdbv4 python evidence_amount_vs_ratio.py` | `05_docking/runner/analyze_quality_gated_ratio_vs_class.py` and `06_lbs_generality/analysis/analyze_lbs_ratio_vs_class.py` (their loaders, gates and partial-Spearman code) | `P.OUT/evidence_amount_vs_ratio{_afdbv4}.csv` (`--out` to rename); stdout | afdbv4: identical to the committed `evidence_amount_vs_ratio_afdbv4.csv` (exact, 120 × 10); the pooled block reproduces SI Table S11 (0.324 / 0.181 / 0.166, total evidence +0.040). Run time about 11 min |
| `zhang2023_targets_separation.py` (49) | Separation and calls at the ST 100% sites of ESR1, MET, EGFR and BRAF | Discussion: "the 1,012 ST 100% sites … median 0.36, against 0.44 across the detection set" | `python zhang2023_targets_separation.py` | `P.WD/afdb_dataset/st100/afdb_st_100_dataset_with_geometry.csv` (or a path as argument) | stdout | No committed output. Original and moved copy: identical, 7/7 lines. Prints 1,012 sites, 0.36, 0.44 |

### Left out

| Script | Reason |
|---|---|
| `c30_lowsep_behaviour.py` | Census and raw / standardized gap by separation tier on the July (v6) detection table. Superseded: SI Table S3 cites `density_separation_gaps_afdbv4.csv`, which carries the same columns on the manuscript's set; none of its own values is quoted |
| `fig3a_neff.py` | Figure 3a under a distinct-region rank budget. Diagnostic; its output says "not a manuscript figure" and none of its numbers is quoted (SI Table S12 comes from `detection_rank_cut.py`, the three-site cap paragraph from `ifsitepred_cap.py`) |
| `frontier_robust.py` | Stricter containment bands (± IQR/2) and stricter frontier margins (k = 0.75 … 0.25). Not quoted; the bands the SI uses come from `frontier_final.py` |
| `o6_s3_binning.py` | Binning test of the former SI Figure S3; that figure was dropped from the SI and nothing of it is quoted |
| `o8_size_variable.py` | Which pocket-size variable correlates best with success; it fed a wording decision, no number is quoted |
| `plddt_band_headtohead.py` | Bootstrap intervals for the within-band ρ; not quoted. The band table in SI Table S6 and Figure 5c comes from `plotting/figure5_plddt_reliability.py` |
| `separation_vs_probability.py` | Separation gradient within winning-probability tiers (300 resamples); a precursor of `separation_shape.py`, whose estimator replaced it. No number of its own is quoted |
| `success_vs_diversity.py` | Detection success against diversity vs the evidence ratio; a framing check, no number is quoted |

### Caveats

- `off_frontier.py` and `separation_shape.py` still name internal write-ups in a few `print` strings (e.g.
  "FRONTIER_AND_PLDDT.md", "SUCCESS_VS_DIVERSITY"). They were left as they are so the stdout matches the committed
  logs line for line; docstrings and comments were rewritten.
- `zhang2023_targets_separation.py` defaults to the run_2 ST_100 table (`afdb_dataset/st100`, 82,130 sites). On the
  run_3 table (`cfg.AFDB_ST100_DATASET_GEOM_CSV`, 87,434 sites, pass it as the argument) the quoted numbers are
  unchanged (1,012 sites, median 0.36 against 0.44); the per-target calls, BRAF's median (0.45 vs 0.41) and the
  high-tier count (10 vs 6) differ.
- `separation_shape.py` reads the descriptor table of the run_2 P2Rank run (`P2Rank/ubs_run_4/...`) for its Q1 shape
  variables, as the original did.
- The size-controlled partial rho of pLDDT with diversity in the SI (-0.478) is printed by `plddt_partial_size.py`
  (size = mean pocket volume, 236,087 complete cases); `plotting/figure5_plddt_reliability.py` prints -0.479 for
  Figure 5 (size = model pocket residue count, all 237,922 pockets). See that script's docstring.

## D. Added at integration

| File (lines) | Computes | Backs | Command | Inputs | Outputs | Verification |
|---|---|---|---|---|---|---|
| `st100_evidence_numbers.py` (280) | ST_100 numbers: evidence ratio vs counts as what sets the class (item 1), class counts and evidence asymmetry (4), n_apo / n_holo per class (7), the filter chain 105,713 → 86,946 detection pockets (8), separation tiers by class (9); `balanced_classification.py` imports its `load_st100` and `partial_spearman` | Results on the evidence ratio, Figure 2 legend, Methods detection-set counts | `python st100_evidence_numbers.py all` (or `item1`/`item4`/`item7`/`item8`/`item9`) | `analysis_unified_levels.csv`, the ST_100 detection dataset, the three LBS per-pocket tables (work_dir) | stdout | stdout identical to the original (134/134 lines) |
| `plddt_partial_size.py` (43) | Size-controlled partial correlations of pocket pLDDT with mean pocket RMSD, CV RoG and \|evidence ratio\| (size = mean pocket volume; rank-Pearson partial over the 236,087 complete cases) | SI pLDDT section: -0.478, -0.386, +0.082 | `python frontier_at_scale.py` once, then `python plddt_partial_size.py` | `P.OUT/frontier_per_pocket.csv` | stdout | prints -0.478368 / -0.386075 / +0.082161 |

## Not included

The authors' number checkers (`check_*_numbers.py`), which audit internal write-ups rather than the manuscript, and
three bookkeeping scripts (cluster re-run planning, review-item and cross-reference lists) are not part of this
folder, nor are the exploratory scripts listed under "Left out" in each section.
