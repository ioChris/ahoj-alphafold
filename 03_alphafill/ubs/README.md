# AlphaFill Unified Workflow

This folder contains the rebuilt AlphaFill analysis workflow that is centered on
the current unified AHoJ/AF analysis table rather than the older UBS-only
intermediate files.

Recommended local data layout:

- Keep raw AlphaFill assets in the existing locations:
  - `work_dir/alphafill/jsons`
  - `work_dir/alphafill/alphafill_analysis`
  - `work_dir/merge_ahoj_alphafill`
- Save all new derived AlphaFill-unified products in a dedicated top-level
  directory:
  - `work_dir/alphafill_unified`

This mirrors the approach used for `afdb_dataset`:

- raw/source-like products stay in their established folders
- the new pipeline gets a clean, dedicated output root for reproducible runs

Current scripts:

- `build_ubs_alphafill_analysis_tables.py`
  - Reads the unified AlphaFill merge outputs
  - Filters to UBS rows
  - Builds clean matched/unmatched UBS analysis tables
  - Adds QA columns that distinguish any AlphaFill match from same-UniProt-only
    matches

- `analyze_ubs_alphafill_unified.py`
  - Reads the rebuilt UBS AlphaFill tables
  - Applies identity filtering to AlphaFill long rows
  - Aggregates surviving transplants back to UBS level
  - Writes clean per-UBS summaries, bin summaries, class summaries, and
    correlation/coverage tables for downstream interpretation
  - Uses the database-facing verbal confidence thresholds from
    `parse-ahoj/kde_plot.py`
  - Stores both full and merged label columns in the per-UBS tables:
    - full: `prob_level_db`, `density_level_db`, `separation_level_db`
    - merged: `prob_level`, `density_level`, `separation_level`

- `plot_ubs_alphafill_unified.py`
  - Reads the rebuilt analysis tables
  - Generates a compact first-pass plot set
  - Keeps plotting separate from the data-table generation step

- `compare_ubs_alphafill_scopes.py`
  - Reads both `same_uniprot` and `any` analysis outputs
  - Generates direct scope-comparison plots and top-correlation summaries
  - Highlights the coverage-versus-fit-quality tradeoff when cross-UniProt
    AlphaFill matches are included

- `plot_legacy_style_distributions.py`
  - Recreates the most useful legacy AlphaFill distribution plots from the
    rebuilt unified per-UBS tables
  - Focuses on clash-score histograms, split half-violins, clash-quantile
    support panels, and worst-tail clash enrichment versus separation
  - Produces the plots separately for both `same_uniprot` and `any`
  - Supports `--level_mode merged|full` so plots can either:
    - use merged display bins (`very_low + low -> low`, `high + very_high -> high`)
    - or use the full database-facing verbal bins directly

- `plot_clash_ridgelines.py`
  - Builds ridgeline views of AlphaFill clash-score distributions for apo-like
    and holo-like pockets
  - Stacks apo ridges above holo ridges
  - Supports both the 5 full database separation labels and 5 separation
    quantiles for side-by-side comparison

Database-facing verbal confidence thresholds

The original AHoJ confidence-bar plotting code in
`parse-ahoj/kde_plot.py` defines a shared unitized confidence scale
`u in [0, 1]` with these verbal bands:

- `Very low`: `0.00 - 0.20`
- `Low`: `0.20 - 0.30`
- `Moderate`: `0.30 - 0.60`
- `High`: `0.60 - 0.85`
- `Very high`: `0.85 - 1.00`

For probabilities, the database code uses a sigmoid-style remapping of the
winning probability before applying those verbal bands:

- function: `u = (sigmoid(k * logit(p)) - 0.5) / 0.5`
- default gain: `k = 1.64`
- source: `parse-ahoj/kde_plot.py`, helpers `_unitize_probability()`,
  `_verbal_from_unit()`, and `verbose_probability_label()`

The same code documents these approximate probability cutoffs in sigmoid mode:

- `Very low`: `0.5000 - 0.5615`
- `Low`: `0.5615 - 0.5933`
- `Moderate`: `0.5933 - 0.6996`
- `High`: `0.6996 - 0.8223`
- `Very high`: `0.8223 - 1.0000`

These are the user-facing verbal thresholds shown in the rendered database
confidence plots. If AlphaFill-side verbal labels are used, they should ideally
be aligned to this scheme or merged from it consistently, for example:

- `low` <- `Very low` + `Low`
- `moderate` <- `Moderate`
- `high` <- `High` + `Very high`

The rebuilt AlphaFill workflow now follows exactly that pattern:

- per-UBS analysis tables keep both the full database labels and merged labels
- merged labels are used by default in the compact plot scripts
- legacy-style distribution plots can switch between merged and full bins with
  `--level_mode`
