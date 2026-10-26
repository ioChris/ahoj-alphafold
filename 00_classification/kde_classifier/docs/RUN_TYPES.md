# Run types: holo-2.1, holo-2.4 and afc-2.4

*Design and record, 2026-09-17, from the classification code's development repository (parse-ahoj),
where this code is the parsing stage of the AHoJ-DB build. Some references below point to that
repository and to the authors' private project notes. The profile used in the manuscript is
`holo-2.1` (the v2c run).*

parse-ahoj used to know one kind of archive: an AHoJ 2.1.0 job on a PDB holo query, as AHoJ-DB v2c was
built. It now recognises three, shares one classifier between them, and refuses to guess.

| path | archive | unit of work | query row | what changes for parse-ahoj |
|---|---|---|---|---|
| **holo-2.1** | AHoJ 2.1.x, v2c layout (52 columns, no gate columns, no `query_mode`) | one job per ligand instance | the holo crystal | nothing: the released numbers are reproduced (§7) |
| **holo-2.4** | AHoJ 2.4.x, 2.5.x or 2.6.x, `query_mode` `pdb` | one job per ligand instance | the holo crystal | gate labels stand; the convex volume; new columns; the query triple recorded |
| **afc-2.4** | AHoJ 2.4.x, 2.5.x or 2.6.x, `query_mode` `uniprot` | one job per binding site (a locked UniProt residue set) | **the AFDB model itself** | `add_query_row = 0`; labels from entity evidence with the gate and a site-aware ion rule; the model is the anchor of every distance |

**The author's key point.** For a PDB holo query the query row joins the holo set with pocket RMSD
excluded (`add_query_row = 1`). In AF-centric mode the query row *is* the AlphaFold model. It must never
be added to the apo or holo set (`add_query_row = 0`), or the model is scored against itself. The
`afc-2.4` profile fixes 0 and `resolve_profile` raises on any config that says 1.

---

## 1. How a run type is determined

**With a bundle** (`--bundle <dir>`, `run-bundle.md`): `run.yaml` gives `run_type` (`afc` | `holo`) and
`ahoj.version`; together they name the profile through `run_types.AHOJ_MINORS`, the table
`detect_run_type` reads too (`afc` + 2.4.x → `afc-2.4`; `holo` + `2.5.1+develop.7fd6825` → `holo-2.4`;
§8 item 9). `output/archives/` becomes
`root_data_dir` and `sites.tsv` becomes `sites_tsv`. An explicit `--run_type` that disagrees with
`run.yaml` stops the run. The bundle's `run.yaml` is copied into `run_manifest.json`.

**Without a bundle**, every archive is read for itself (`run_types.detect_run_type`):

| `script_parameters.json` `script_version` | `global_results.csv` | result |
|---|---|---|
| 2.1.x | v2.1 layout | `holo-2.1` |
| 2.4.x, 2.5.x, 2.6.x | `query_mode` == `uniprot` (or a `<UniProt> <i>+<j>+…` query string when the column is absent) | `afc-2.4` |
| 2.4.x, 2.5.x, 2.6.x | anything else | `holo-2.4` |
| any other minor (2.2, 2.3, 2.7 and later), or a 2.1 archive with 2.4 columns | | `RunTypeError` |
| missing | the layout decides when it can (`schema_of`: v2.1 / v2.4); otherwise `RunTypeError` | |

Detection reads the columns after the name step of §5a: an AHoJ 2.6.0 job, whose columns and JSON keys
carry the 2.6.0 names, is renamed to parse-ahoj's names as it is read, before anything above looks at it.

`--run_type auto` (default) detects per archive; an explicit value is checked against every archive and
a mismatch is an `Error` for that job. Every summary row records `run_type`, `ahoj_version`,
`ahoj_schema`, `ahoj_date`, `ahoj_volume_mode`, `ahoj_pocket_alignment_method`, `ahoj_holo_overlap_mode`,
`ahoj_uniweave`, `ahoj_model_version`, and the query triple `query_mode` / `query_anchor` / `query_model`
(None on 2.1 archives). That is the "which AHoJ version and which schema wrote each archive" record.

## 2. The profile: every rule is a field

`run_types.RunProfile` (frozen dataclass). A config overrides single fields (`add_query_row`,
`relabel_mode`, `ion_rule`, `additive_rule`, `gate`, `volume_column`, `volume_nan_to_zero`, `model_row`,
`key_format`; `null` keeps the profile value) and the summary row records what was in force
(`rule_*`, `delta_anchor`, `rog_anchor`, `mapped_denominator`, `chain_axes_meaningful`).

| field | holo-2.1 | holo-2.4 | afc-2.4 | meaning |
|---|---|---|---|---|
| `key_format` | `holo` | `holo` | `afc` | `<pdb>-<chain>-<lig>-<resi>[.<u\|l…>]` or `<pdb>_<chain>_<lig>_<resi>[.<u\|l…>]` vs `<UniProt>_ST_100_<n>` (`any` accepts both; the pilot named its holo pass by site). The optional suffix is arm 1's case disambiguator; the `_` form is kaleidoscope's export (§4 item 16) |
| `add_query_row` | 1 | 1 | **0** | see above; 1 on an afc archive raises |
| `relabel_mode` | `v2c_ions` | `v2c_ions_demote_only` | `entity_gate_site` | §3 |
| `ion_rule` | `query_disables` | `query_disables` | `site` | §3 |
| `additive_rule` | `v2c` | `v2c` | `v2c` | additives count as holo (v2c); `site` = only at additive-defined sites (the `gate_site_noadd` sensitivity row) |
| `gate` | `none` | `archive` | `apply` | entity relabel only: `archive` applies the gate iff the AHoJ run had `holo_overlap_mode=gate`; `apply` always, with the archive's thresholds (2.4 defaults 2 / 20% / 30%) |
| `volume_column` | `volume` | `volume` | `volume` | the column copied into `volume` before curation; a missing column raises; a 2.4 archive without `volume_mode` raises |
| `volume_nan_to_zero` | 1 | 1 | 1 | the v2c rule (NaN volume with `#resis3 > 0` → 0.0), written for PyVOL; kept on 2.4 to match the pilot |
| `model_row` | `AF_model` | `AF_model` | `AF_model` | the query's own model row: role `M` **and** `structure == 'AF_model'` (AHoJ before PR #143), or, since #143 (model rows name their model, `AF-Q15858-F1-model_v6`), the role-`M` row whose `UNPs` are all query `UNPs`; UniWeave adds neighbour-UniProt rows with role `M` (`AF_<acc>` before #143, a non-query `UNPs` since) |
| `protect_labels` | `U` | `U * -` | `U * -` | labels no relabel touches; 2.1 keeps the set the release was made with |
| `delta_anchor` | query | query | **model** | the query-summary deltas are \|candidate − anchor\| (§5, item 8) |
| `rog_anchor` | query | query | **model** | the ±60% RoG window (item 13); a NaN anchor is an `Error`, not a silent drop of every candidate |
| `mapped_denominator` | member pocket | member pocket | **locked union** | the 75% `%mapped` cut divides by the query's `#resis3`, which for a locked set is the union (item 14; ST_100 makes the two equal) |
| `chain_axes_meaningful` | yes | yes | **no** | the model's `pocket_dist`/`tm_score`/`rmsd` are 0/1/0 by construction (item 7); the chain verdicts are reported as `N/A (model is anchor)` |

Configs: `configs/run_config_afc_2.4.json` and `configs/run_config_holo_2.4.json` set `run_type` and
leave every rule `null` (profile). `configs/run_config_full_run_3.json` is unchanged and runs `holo-2.1`
exactly as before (its `add_query_row: 1` equals the profile).

## 3. Apo/holo labels per path

**holo-2.1 (`v2c_ions`).** `relabel_apoholo_by_ions`, unchanged: rebuild `H` / `H*` / `A` from
`query_ligs` and `nonquery_ligs`, drop single-atom ions from the candidate ligands unless the query
ligand is itself an ion (`query_ion_disables_filter`), mark demoted holo rows `A**`.

**holo-2.4 (`v2c_ions_demote_only`).** AHoJ 2.4 already applied its holo-overlap gate, and
gate-rejected entities stay listed in `nonquery_ligs`. Rebuilding the labels from the ligand columns
would therefore promote gate-rejected pockets back to `H*` and *undo the gate* (item 2). With
`demote_only`, a candidate may only move from `H`/`H*` to `A**` (an ion-only holo call, the v2c
demotion); nothing is promoted. `relabel_mode v2c_ions` on a `holo_overlap_mode=gate` archive raises
unless `allow_undo_gate=1` is given for a deliberate comparison (§7 uses it to reproduce the pilot's
`hq1_ahoj_*` columns). On a `report_only` run there is no gate to undo, and `v2c_ions` is legitimate.

**afc-2.4 (`entity_gate_site`).** The query has no ligand, so `query_ligs` lists residue names and the
v2c ion filter would always be on (item 3). Labels are rebuilt from `ligands.json` (schema 2) for
observed candidate pockets (role `C`, label not protected, `#resis3 > 0`), as `relabel_rows` did in the
pilot, with three grades (§3c(d) of the brief):

| grade | label | condition | pooled as |
|---|---|---|---|
| holo | `H*` | a qualifying entity passes the gate (≥ 2 shared residues, ≥ 20% of the locked set, ≥ 30% of the entity's contacts) | holo |
| apo-contact | `A*` | an entity qualifies but fails the gate (AHoJ's `A*`) | apo |
| apo | `A**` | entities present, none qualifies (an ion at a non-ion site; an additive under `additive_rule site`) | apo |
| apo | `A` | nothing detected | apo |

**The label vocabulary (every path, AHOJ-020).** `run_types.APO_LABELS` (`A`, `A-`, `A*`, `A**`) and
`run_types.HOLO_LABELS` (`H`, `H*`) are the only place the pooled sets are spelled out; every active
filter reads them (`batch_job_functions`, `stats_core.dp_classify_new_point`,
`parsing_functions_DBv2.split_dataframe`). `U` is known and pooled nowhere. AHoJ `develop` from
`431b220` on writes `A-` for a query-ligand hit the holo-overlap gate demoted; 2.4.0 and 2.5.1 wrote
`A` for the same pocket. `A-` is apo, no relabel changes it (`demote_only` touches only `H`/`H*`), it
stays `A-` in `global_results_plots.csv`, and the summary row counts it in `n_apo_query_demoted`. Any
other candidate label, for example `Hweak`/`H*weak` from `--holo_overlap_mode weak_label`, joins
neither set: the job prints a `[WARN]`, the summary row carries `n_unknown_label` and `unknown_labels`
(`Hweak:3;X:1`), and `run_manifest.json` totals both under `labels`.

`entity_qualifies`: `all` (every entity), `nonion` (ions never), `site` (single-atom ions count only at
ion-defined sites; v2c's per-job rule made per-site), plus `additive_rule`. The ion list is
`parsing_functions_DBv2.EXCLUDED_LIGANDS_TOTAL_LE1`; the additive list is `run_types.ADDITIVES_DEFAULT`
(the pilot's stand-in for the BioLiP artifact list; `--additives_file` replaces it). Whether a site is
ion- or additive-defined comes from `sites.tsv` (`members`, the defining ligand instances as v2c job
keys, or an optional `ligands` column). `ion_rule site` without a sites table is an `Error`; a site
missing from the table is an `Error` for that job.

## 4. Appendix A, site by site

Line numbers are those of Appendix A (parse-ahoj `2519b31`). "unchanged" means the legacy behaviour is
kept on that path.

| # | site | holo-2.1 | holo-2.4 | afc-2.4 |
|--:|---|---|---|---|
| 1 | `.tgz` member name (`batch_job_functions.py:492`) | `run_types.read_member` accepts `./global_results.csv` and `global_results.csv` on every path; also for `script_parameters.json`, `ligands.json`, `residue_mappings.zip` | same | same |
| 2 | `relabel_apoholo_by_ions` rebuilds H/H*/A (`parsing_functions_DBv2.py:1811-1817`) | unchanged (`v2c_ions`) | `v2c_ions_demote_only`: never undoes the gate; `v2c_ions` raises on a gate run without `allow_undo_gate` | not used; `entity_gate_site` from `ligands.json` |
| 3 | ion filter keyed on the query row (`:1784-1788`) | unchanged (`query_disables`) | unchanged | `ion_rule site`: ions count only at ion-defined sites (from `sites.tsv`) |
| 4 | protects `U` only (`:1681`) | unchanged (`U`; frozen for fidelity) | `U`, `*`, `-` | `U`, `*`, `-` |
| 5 | `add_query_row` (`batch_job_functions.py:607-619`) | 1 | 1 | **0**, enforced |
| 6 | first role-`M` row (`:557`, `:648-692`; `stats_core.py`; `kde_core.py`) | `select_model_rows`: role `M` and `structure == 'AF_model'` (v2c's model row is `AF_model`, so identical) | same; since AHoJ PR #143 (develop, 2.6.0) the row whose `UNPs` are all query `UNPs` | same; UniWeave neighbour rows never reach the classifier or the summaries |
| 7 | model chain metrics 0/0/1/0 (`batch_process_database.py:400-401`) | n/a | n/a | chain classifiers still run; `chain_overall_classification` and `chain_consensus_class` become `N/A (model is anchor)`, `chain_axes_meaningful = 0`, `model_pocket_rms` blank |
| 8 | query-summary deltas (`parsing_functions_DBv2.py:1440-1445`) | \|candidate − query\| | same | deltas from the **model**; column names unchanged (the detached metrics table joins on them), `delta_anchor = model` recorded |
| 9 | NaN→0 volume rule (`batch_job_functions.py:547-549`) | unchanged (PyVOL) | `volume_column` explicit (default `volume` = the run's `volume_mode`, convex); rule kept; `volume_ahoj` keeps the original when another column is chosen | same as holo-2.4 |
| 10 | model summaries (`:327-393`) | unchanged | unchanged | `model_pocket_rms` blank (≡ 0), `query_resolution` blank, `query_exp_method` blank (was `-`) |
| 11 | ligand harvest (`:867-874`) | unchanged | unchanged | `query_ligands` blank for `POI_type set`; `candidate_ligands` from holo rows only (gate-rejected entities stay in `nonquery_ligs`) |
| 12 | summary row (`:925-1040`) | + provenance and `rule_*` columns, `query_mode/anchor/model` = None | + the query triple | + the query triple, `site_uid`, `site_ion_defined`, `site_additive_defined`, `site_n_members` |
| 13 | ±60% RoG window on the query row (`parsing_functions_DBv2.py:1296-1308`) | unchanged | unchanged | anchored on the model (`rog_anchor = model`); a NaN anchor raises instead of dropping everything |
| 14 | 75% mapped vs the query's `#resis3` (`:1324-1327`) | unchanged | unchanged | denominator = the locked union (`mapped_denominator` recorded); ST_100 sets make member pocket and union equal |
| 15 | lightweight collectors (`batch_job_functions.py:1293-1358`, `:1437-1484`) | `do_collect_model_pLDDT`: `select_model_rows`, records `ahoj_version`, `ahoj_schema` and the query triple; `do_collect_apoholo_metric_stats`: the full profile (relabel, volume) and provenance | same | same |
| 16 | `base.split('-')` (`:1182-1183`) | `parse_job_key`: holo keys | holo keys, including arm 1's case disambiguator: keys equal case-insensitively (chains `2Q`/`2q`) end in `.` plus one `u`/`l` per case-differing letter (`6xhv-2Q-MG-203.u`, `.l`, `.ul`, …; `ahoj-db-pipeline` `queries.disambiguate_keys`). The full key stays the job name and `JobKey.key`; the suffix is `JobKey.case_suffix` and is not part of `pdb`/`chain`/`lig`/`resi`. Any other suffix (`.2`, `.U`) is a `RunTypeError`. Two separators (2026-10-12): `-` (v2c, `1abc-A-ATP-501`) and `_` (kaleidoscope's AHoJ-schema export, its design §3.9: `3cqv_A_HEM_601`, `1abc_B_ZN_-5`, `6xhv_2Q_MG_203.u`; `-` is ambiguous there because residue numbers can be negative, which the hyphen form writes as `1abc-B-ZN--5`). One key uses one separator; mixed keys are a `RunTypeError`. `<pdb>` is 4 characters starting with a digit or the extended `pdb_` + 8. A UniProt accession starts with a letter, so neither form is ever an afc key, and a holo key is refused under `key_format afc` (`1abc_A_100_5`, CCD code `100`, would otherwise pass for `<UniProt>_<threshold>_<n>`). Nothing downstream splits the key: summary rows carry it as `job_name`, and the mirrored `plots`/`statistics` directories (`closest_*.json`) are named from the archive path | afc keys → `site_uid`, `site_uniprot`, `key_kind`; anything else `RunTypeError` |
| 17 | one pass per job key (`parse_AHoJ3_DBv2b.py:117-140`) | unchanged: a run directory holds one pass. The pilot's three passes live in three directories (`runs/af1`, `af2`, `hq1`) and are parsed as three runs | same | same |
| 18 | `--job_list` cut at `:` (`batch_process_database.py:62-70`) | unchanged: job keys never contain `:`; the afc key form uses `_` (`P46883_ST_100_14`); a `site_uid` with `:` is not a key | same | same |
| 19 | default paths (`:491, 501`; config `:63-65`) | `--root_data_dir` (or `--bundle`) and `--summary_dir` are required; the v2c and cloud-folder fallbacks are gone. Archives directly under the root (the pilot's flat layout) are read as one pseudo-batch | same | same; a bundle sets both from `run.yaml` |

## 5. What changed in `global_results.csv` between 2.1 and 2.4, as parse-ahoj sees it

52 columns → 78. New: the gate columns (`holo_contact_resis`, `holo_overlap_resis`,
`entity_contact_coverage`, `query_pocket_coverage`, `holo_overlap_entity`, `holo_assignment_detail`),
`#atoms3`, identity and UniWeave columns, `rsa_avg`, the volume family (`volume2`, `volume_seed`,
`volume_largest`, `volume_total`, `volume_mean`, `vol_convex`), `pocket_lig`, `pocket_pep`,
`allostery_pep`, `pocket_aln_atoms`, `pocket_rmsf_top_contribution`, `pocket_rmsf_localization`,
`pocket_lddt_map`, and the query triple. Gone: `#resis1`, `#resis2`, `chains2`, `chains4`, `solvent`.
`asa`/`asa_avg` are blank unless `get_asa` was on. The classifier uses none of the removed columns;
`pocket_lddt_map` is reported, not classified on (brief §3b).

Estimators (brief §5): `pocket_rms` is method 2 (mapped Kabsch) by default and reproduces method 1 at a
fixed anchor; `volume` is the convex estimator and is **not** PyVOL's number (ρ 0.666); `RoG` and
`sasa` are near-identical. So `holo-2.4` numbers are comparable with v2c on three axes and not on
volume, and `afc-2.4` numbers are distances from the model on every axis. Nothing in parse-ahoj hides
that: the `rule_volume_column`, `ahoj_volume_mode`, `delta_anchor` and `rog_anchor` columns travel with
every row.

### 5a. The output names of AHoJ 2.6.0

AHoJ 2.6.0 (AHoJ2 PR #146; its `documentation/output_names.md`) renames every output column and JSON
key and changes no value: `apoholo_assignment` -> `label`, `#resis3` -> `n_observed`, `RoG` -> `rog`,
`UNPs` -> `accessions`, `pocket_rms` -> `pocket_rmsd`, `tm_score_i` -> `tm_score_candidate`, ... (46
result columns); `ligands.json` goes to schema 3 (`pocket_index` -> `pocket`, entity `overlap_resis` ->
`n_overlap_residues`, `query_pocket_coverage` -> `query_cov_pct`, `entity_contact_coverage` ->
`entity_cov_pct`, `label` -> `resn`, ...). `script_parameters.json` renames nothing and gains
`output_schema`: 2 for the new names, 1 under `--legacy_names 1` (the old names), absent before 2.6.0.

parse-ahoj keeps the names before 2.6.0 everywhere: code, configs (`columns_nozero`, the classifier
metrics), and everything it writes (`global_results_plots.csv`, the summary columns, the statistics JSON
keys, the plot file names). Decided by the author, 2026-10-12: the old names minimise churn and keep
every output that AHoJ-webapp and ahoj-alphafold read unchanged. One step renames a 2.6.0 job back as
it is read:

- `run_types.load_global_results` reads `global_results.csv` and `script_parameters.json` in one pass and
  calls `ahoj_names.results_to_internal`. `output_schema` decides, checked against the header. Without
  `output_schema` the header decides: the two name sets are disjoint, so the header is never ambiguous.
  A header that contradicts `output_schema`, or mixes both sets, raises `RunTypeError`, which makes the
  job an `Error`. The schema read is kept in `df.attrs['ahoj_output_schema']`.
- `run_types.load_ligands_json` calls `ahoj_names.ligands_to_internal`: schema 3 becomes schema 2. A
  document without `schema_version` is judged by its pocket key, and AHoJ 2.1's bare list is left alone.
- Every live reader goes through these two: `do_kde_plots_and_classification`, `do_collect_model_pLDDT`,
  `do_collect_apoholo_metric_stats` (the last two read `global_results.csv` themselves before), and the
  standalone `conformational_spectrum_api.py` `main` (named `test_conformational_spectrum_api.py` in parse-ahoj).

The maps in `ahoj_names.py` are copies of AHoJ2's `output_names.py` at `0e280af`.
`tests/test_ahoj_names.py::test_vendored_maps_equal_ahoj2_output_names` compares them when `AHOJ2_DIR`
names an AHoJ2 checkout. Only the files parse-ahoj reads are mapped. `summarized_data.csv`,
`residue_level_info.csv`, `volume.csv`, `pocket_info.json` and `allostery.json` are not read on the live
path.

2.6.0 also carries develop's PR #143: model rows name their model (`AF-Q15858-F1-model_v6`) instead of
`AF_model`, so `select_model_rows` now finds the query's own model by its `UNPs` (§2, `model_row`).
Before that change, a develop or 2.6.0 job found no model row and was `Discarded` ("No model (role M)
row") on every path.

## 6. SIFTS artifacts on each path

| need | holo-2.1 | holo-2.4 | afc-2.4 |
|---|---|---|---|
| the query pocket in UniProt numbering | `residue_mappings.zip` → `poi_dict_unp.json` (`run_types.load_poi_dict_unp`; keyed `A_A_ARG_51` → `P12345_51`) | same file, same format (`documentation/output_readme.md` of AHoJ2) | same; the key is `U_SET_1` and the values are the locked set |
| candidate residue mapping | `cndt_dict2.json` / `cndt_nested_UNPi_PDBi.json` in the same zip | same | same |
| chain → UniProt dictionary (legacy `process_results_AHoJ.py`) | was a hard-coded `uniprot_segments_observed_dict.bin`; now `sifts_access.load_chain_uniprot(sifts_dir)` → `ahoj_sifts.load_chain_uniprot`, with the directory from `AHOJ_SIFTS_DIR` or `<workdir>/SIFTS` | same | same |
| residue-level flags (`sifts_mutation_like`, observed) | not needed | not needed | `sifts_access.map_pdb_residues(work_dir, pdb, tokens)` → `ahoj_sifts.map_pdb_residues` on the run's staged cache, offline. Not wired into a collector yet; the pilot needed none |

Rules kept: no new SIFTS parsing in parse-ahoj and no new hard-coded pickle paths. `ahoj_sifts` is
AHoJ2's package (develop `9e58cec`, AHoJ 2.5.1, `documentation/modularization.md` §7 and §9.3),
installed from the commit the run pinned; `sifts_access` raises `SiftsUnavailable` with the install line
when it is absent. The archive artifacts are the same on both AHoJ versions, so
`do_collect_uniprot_residues` works on both with only the key parsing changed.

## 7. Fidelity and regression (run 2026-09-17, py311, this PC)

`python analysis/run_types_fidelity/check_run_types.py {legacy|afc|hq1} --out DIR`. Read-only on the
data folders. The classifier config is `run_config_full_run_3.json` with plots and per-entry files off.

| check | input | reference | result |
|---|---|---|---|
| **legacy** (`holo-2.1`) | 100 v2c archives: the first 50 by name of batches `000` and `300` under `June_2024\v2c\output_fix` | the full_run_3 release (`query_summaries_do_kde_plots_and_classification.csv`) | **73 of 73** classified jobs reproduce `P(holo)`, `n_apo`, `n_holo` and the class exactly; the other 27 are `Discarded` here and absent from the release (0 in the release but not here; 0 here but not in the release) |
| **afc-2.4** | the pilot's 99 complete `af2` archives (`af_centric_pilot\runs\af2`, rc 0) with `sites.tsv` built from the pilot site list | `pilot_sites_collected.csv`, columns `af2_gate_site_*` (what `summarise` in `af_centric_rerun.py` reads) | **99 of 99** sites match `n_apo`, `n_holo`, `P(holo)`, overlap and class; `add_query_row=1` on an afc archive → `Error` |
| **holo-2.4 smoke** | the pilot's 98 `hq1` archives (2.4.0 on the rep job's holo query, gate on) | none for the profile default; `hq1_ahoj_*` for the v2c-relabel override | profile default: 98 of 98 `Success` (64 Holo, 34 Apo), `holo-2.4` detected, volume `convex`. Override `relabel_mode=v2c_ions allow_undo_gate=1`: **98 of 98** reproduce the pilot's `hq1_ahoj_*` (`n_apo`, `n_holo`, `P(holo)`). Gate kept vs undone on the same 98 jobs: 9,210 vs 9,701 holo rows, 6,601 vs 6,110 apo rows; `P(holo)` differs at 45 jobs |
| **holo-2.4 on AHoJ develop** (2026-09-23) | arm 1's calibration repeat `holo-cal-biolip-500-dev7fd6825`: 496 jobs on `develop` 7fd6825 (`script_version 2.5.1`, gate), 189 of them large ribosome entries; `--root_data_dir` + `--run_type holo-2.4` (`--bundle` was refused then, §8 item 9) | the same 496 jobs on 2.4.0 (`holo-cal-biolip-500`), same profile | 485 `Success`, 10 `Discarded`, 1 `Error` on both (`6xhv-2Q-MG-203.u`: the `.u` case suffix did not parse then; next row). 10,901 `A-` rows at 183 jobs pooled as apo (AHOJ-020): 0.38% of apo on the ordinary entries, 18.9% on the ribosome ones (Mg2+ sites). All 183 reproduce 2.4.0's `n_apo`, `n_holo`, `P(holo)` and class **exactly**. Without `A-` in the apo set: 0 of 183, `P(holo)` moved at 151, three class flips (Holo instead of Apo). No unknown label in either run |
| **holo-2.4 through `--bundle`** (2026-09-27) | the same two bundles, each given as `--bundle` (develop's `ahoj.version 2.5.1+develop.7fd6825` → `holo-2.4`, §8 item 9); case-suffixed keys accepted (§4 item 16) | 2.4.0 against develop, job by job | 486 `Success`, 10 `Discarded` (the same 10), 0 `Error` on both. `6xhv-2Q-MG-203.u` is 347 / 160 / 0.209, Apo, on both; 85 of its apo rows are `A-` on develop. **486 of 486** jobs have the same `n_apo`, `n_holo`, `P(holo)` and class on both versions (251 Apo, 234 Holo, 1 Ambiguous). Develop pools 10,986 `A-` rows at 184 jobs (the 183 above plus the `.u` job) |

`tests/test_run_types.py` (41 tests, `pytest tests/test_run_types.py`) repeats the end-to-end checks on
seven fixture archives under `tests/fixtures/` (8–50 KB each): two v2c jobs against the release, two af2
sites against the pilot (one ion-defined; `ion_rule nonion` gives the pilot's `gate_nonion` counts 6/0
there), two hq1 jobs against the pilot's v2c-relabel values, one develop job with an `A-` row against its
2.4.0 values, plus keys (arm 1's case suffixes `.u`, `.l`, `.ul`, and the develop job renamed with
each), detection, the AHoJ-minor table for archives and bundles, profiles, sites (members with the
suffix), relabel grades, protections, `demote_only`, model-row selection, the label vocabulary and the
unknown-label report, the bundle reader, and the batch driver on two one-job bundles (afc 2.4.0; holo
`2.5.1+develop.7fd6825` with a `.u` key). On this PC pytest needs `--basetemp <dir>` (the default temp is sandboxed).
The whole suite passes at 88 when `tests/test_create_plots_for_all_entries.py` and `tests/test_integration.py`
are ignored: both import `create_plots_for_all_entries`, a module that is not in the repository at `main`
`2519b31` either (only its two READMEs are), so their collection error predates this branch.

## 8. Open rules for the author (recorded, not settled)

Each has a default in the profile so a run is reproducible today; each is a one-key override.

1. **Ions (afc-2.4).** Default `ion_rule site`: single-atom ions count only at ion-defined sites (the
   pilot's `gate_site`; v2c's per-job rule made per-site). Alternatives measured in the brief:
   `nonion` leaves 10.3% of sites without a holo state and pulls AF-Holo to 56.1%; `all` counts every
   entity (AHoJ's own labels). Recommendation: keep `site`.
2. **Additives (all paths).** Default `additive_rule v2c`: crystallisation additives count as holo,
   as in v2c (9.9% of the pilot's curated holo rows are holo only through additives). `site` excludes
   them except at additive-defined sites (moves 705 structures, AF-Holo 57.2%). The list is the
   pilot's stand-in (`ADDITIVES_DEFAULT`); a production run should name the BioLiP artifact list with
   `--additives_file`. Recommendation: keep `v2c` for the main run and report `site` as a sensitivity row.
3. **Free amino acids (afc-2.4).** AHoJ 2.4.0 drops canonical amino acids from the detected entities
   of a `set` query, so they never reach `ligands.json` and no relabel can recover them (1,712
   detection sites, 2.0%, have a free amino acid among their defining ligands). This is an AHoJ2 fix
   (brief §3c(a)), not a parse-ahoj rule; until it lands, those sites lose that evidence silently in
   AHoJ, and parse-ahoj cannot see it. Recommendation: block a production afc run on the fix.
4. **Gate vs `report_only` (holo-2.4).** The profile follows the archive (`ahoj_holo_overlap_mode` is
   recorded) and never undoes a gate. A v2c-comparable holo-centric build would run AHoJ with
   `--holo_overlap_mode report_only` and then `relabel_mode v2c_ions` applies unchanged; a gated run
   keeps AHoJ's labels with `v2c_ions_demote_only`. On the pilot's holo pass the gate moved 491 of
   9,701 holo rows to apo and changed `P(holo)` at 45 of 98 jobs (§7). Recommendation: decide with the
   AHoJ profile in `ahoj-db-pipeline`; parse-ahoj needs no change either way.
5. **Volume estimator (holo-2.4 and afc-2.4).** Default `volume_column volume`, which is whatever
   `volume_mode` the AHoJ run used (convex by default; not PyVOL's number). `volume2` is available when
   the run computed a second mode (the pilot's hq1 stored `dual_probe` there). A PyVOL mode needs an
   AHoJ change (brief §5). The NaN→0 rule stays on (it reproduces the pilot). Recommendation: name the
   column in the run's config and say in Methods that volume is not comparable with v2c.
6. **The 75% mapped cut on locked sets (afc-2.4).** The denominator is the union set. For ST_100 sets
   member pocket and union coincide; for BioLiP-derived sets (`PROGRAM.md` §3 B) they may not.
   Recommendation: keep, and revisit if B is chosen.
7. **`sites.tsv` `members` format.** parse-ahoj parses the defining ligand codes from members written
   as v2c job keys. BioLiP-derived members will not have that shape; the bundle should then carry a
   `ligands` column (accepted already). Proposed interface change for `ahoj-db-pipeline`: add
   `ligands` (`;`-joined CCD codes of the defining instances) to `sites.tsv` in `run-bundle.md`.
   State on 2026-09-27: the BioLiP harvest's `sites_st100.tsv` (`biolip-2026-04-30`, 12,976 sites) fills
   `ligands` on every row, and its members are the harvest's instance keys, the same strings as the holo
   job keys: 355 of 39,049 carry the case suffix (§4 item 16), for example `P60489:ST_100:42` with
   `6xhv-2Q-MG-203.u` and `9o3j-2x-MG-101.l`. The ST_100 import (`afc-st100-detection-2026q4`) has no
   `ligands` column and no suffixed member. Either way the members parse and keep the full key.
   Members in kaleidoscope's `_` form (`3cqv_A_HEM_601`, `1abc_B_ZN_-5`) parse the same way (§4 item 16).
8. **Weak labels (holo-2.4).** An AHoJ run with `--holo_overlap_mode weak_label` writes `Hweak` and
   `H*weak`, which AHoJ routes to holo. parse-ahoj pools them in neither set and reports them
   (AHOJ-020). The arm-1 profiles hold `gate` (still `TODO` against `report_only`); neither writes weak
   labels. Recommendation: a `weak_label` run first needs a decision whether weak positives are holo,
   apo or excluded, and then an entry in `HOLO_LABELS` or `APO_LABELS`.
9. **AHoJ 2.5 bundles. Decided by the author, 2026-09-27: 2.5.x takes the 2.4 profiles, through an
   explicit list.** Before, `detect_run_type` mapped any archive of AHoJ 2.4 or later to `holo-2.4` /
   `afc-2.4`, while `read_run_yaml` wanted a profile named after the bundle's `ahoj.version` minor, so
   `--bundle` on arm 1's develop repeat (`ahoj.version 2.5.1+develop.7fd6825`) raised. Now
   `run_types.AHOJ_MINORS` (`2.1` → `holo-2.1`; `2.4` and `2.5` → `holo-2.4` / `afc-2.4`) is the one
   table both read, through `run_type_for`. A minor not in it (2.2, 2.3, 2.6 and later) is a
   `RunTypeError` for the archive and for the bundle until its archives have been checked and the
   minor added. This is stricter than the old `>= 2.4`. Profile names stay `holo-2.4` / `afc-2.4`; the
   summary row's `ahoj_version` says 2.5.1. The evidence: 60 archive pairs of the two calibration runs
   (2.4.0 against develop 7fd6825, `script_version 2.5.1`) have the same `global_results.csv` columns
   and archive members; `script_parameters.json` gains only `config`. The one label change, `A-`, is apo
   (AHOJ-020), and `--bundle` on the whole develop repeat gives the §7 result.
   **2.6.x added, 2026-10-12 (the author):** 2.6.0 renames every output name and changes no value (§5a).
   Evidence: the two jobs of `tests/fixtures/dev0e280af/` (AHoJ2 `0e280af`, the 2.6.0 names) classify
   and write byte for byte as their `--legacy_names 1` twins, and every older fixture rewritten under the
   2.6.0 names classifies as it does under its own (`tests/test_ahoj_names.py`).

## 9. What each run type keeps or overrides from `configs/run_config_full_run_3.json`

| block | keys | holo-2.1 | holo-2.4 | afc-2.4 |
|---|---|---|---|---|
| candidate filters | `bndgrsds_prcnt_seqres/obs 75`, `resolution_soft null`, `drop_no_resolution 1`, `full_pockets_aligned 0`, `pocket_len_resis3 0`, `rog_diff_thrs_query 0.6`, `rog_diff_thrs_mean null`, `pocket_dist null`, `dist_outlier_thr 20` | kept | kept | kept (RoG window and the 75% cut are anchored on the model / the union; §4 items 13, 14) |
| minimums | `min_apo 0`, `min_holo 0`, `min_apoholo 1`, `keep_no_model 0` | kept | kept | kept |
| curation | `columns_nozero`, `relabel_apoholo 1`, `add_query_row 1`, `filter_candidates 1` | kept | `relabel_apoholo 1` now means `v2c_ions_demote_only`; `add_query_row 1` kept | `relabel_apoholo 1` means `entity_gate_site`; **`add_query_row` → 0** |
| KDE | `kde_extend true`, `margin_cap_scale 1.0`, `bw_mode pooled`, `bw_scale 0.75`, `prior_mode equal` | kept | kept | kept |
| metrics | `pocket_class_metrics [pocket_rms, sasa, volume, RoG]`, `chain_class_metrics [pocket_dist, tm_score, rmsd]`, `separation_weights` | kept | kept (`volume` = the run's `volume_mode`) | kept; chain verdicts marked `N/A (model is anchor)`; `pocket_lddt_map` not added (brief §3b) |
| query summary | `query_summary_deltas 1`, `query_summary_metrics`, `query_summary_stats`, `separate_metric_summary 1`, `collect_query_meta 1`, `query_meta_role Q` | kept | kept | kept; deltas are from the model (`delta_anchor`), `query_resolution`/`query_exp_method` blank, `query_ligands` blank, `candidate_ligands` holo rows only |
| plots and files | `general_violins … save_relabeled_csv`, zip options | kept | kept | kept |
| paths | `root_data_dir`, `root_mirror_dir`, `summary_dir` | must be given (no defaults) | given, or from `--bundle` | from `--bundle` (`output/archives`, `sites.tsv`) plus `--summary_dir` |
| new keys | `run_type`, `bundle`, `sites_tsv`, `relabel_mode`, `ion_rule`, `additive_rule`, `gate`, `volume_column`, `volume_nan_to_zero`, `model_row`, `key_format`, `additives_file`, `allow_undo_gate` | all `null`/`auto`/0: the profile | same | same |

## 10. Files

- `run_types.py`: profiles, detection, archive access, job keys, sites table, entity relabel, model-row
  selection, bundle reader, provenance columns.
- `ahoj_names.py`: the AHoJ 2.6.0 output names renamed to parse-ahoj's on reading (§5a).
- `sifts_access.py`: the adapter to AHoJ2's `ahoj_sifts`.
- `batch_job_functions.py`: the profile wired into `do_kde_plots_and_classification`,
  `extract_model_pLDDT_and_query_meta`, `create_query_summary_row`, `do_collect_uniprot_residues`,
  `do_collect_model_pLDDT`, `do_collect_apoholo_metric_stats`.
- `batch_process_database.py`: the new flags, `--bundle`, the flat layout, required paths, the sites
  table in every worker, the manifest's run-type record.
- `parsing_functions_DBv2.py`: `relabel_apoholo_by_ions(demote_only=…)`.
- `process_results_AHoJ.py`: the hard-coded SIFTS pickle replaced by `sifts_access`.
- `configs/run_config_afc_2.4.json`, `configs/run_config_holo_2.4.json`.
- `analysis/run_types_fidelity/check_run_types.py` and `af_centric_pilot_sites.csv`.
- `tests/test_run_types.py`, `tests/test_ahoj_names.py`, `tests/fixtures/` (`dev0e280af/` and
  `dev0e280af_legacy/`: the same two AHoJ2 `0e280af` jobs with the 2.6.0 names and with `--legacy_names 1`).

Not done: wiring `sifts_access.map_pdb_residues` into a collector (no consumer yet); a `hq1`-style
holo-2.4 reference beyond the pilot (there is no released holo-2.4 build to compare with); the
`pymol_snapshots.py` and webapp changes named in `AFDB_V4_SOURCES.md` §6.1 (another branch owns them).
