# AlphaFill Unified Entry-Level Workflow

This folder contains the entry-level counterpart to the rebuilt
`alphafill_unified` UBS workflow.

Design

- Source of truth remains the unified AlphaFill merge outputs:
  - `work_dir/merge_ahoj_alphafill/v6/merged_ahoj_alphafill_summary.csv`
  - `work_dir/merge_ahoj_alphafill/v6/merged_ahoj_alphafill_long.csv`
- This branch works on:
  - `level == "entry"`
  - `threshold == "UNCLUSTERED"`
- Outputs are written to a separate work directory so entry-level and UBS-level
  analyses do not get mixed:
  - `work_dir/alphafill_unified_entries`

Current script

- `build_entry_alphafill_analysis_tables.py`
  - Reads the unified AlphaFill merge outputs
  - Filters to unclustered entry rows
  - Builds clean matched/unmatched entry analysis tables
  - Adds the same AlphaFill QA columns used in the UBS branch:
    - any AlphaFill match
    - same-UniProt-only match
    - cross-UniProt-only match

- `analyze_entry_alphafill_unified.py`
  - Reads the rebuilt entry-level summary and matched-long tables
  - Aggregates AlphaFill transplants back to one row per entry
  - Writes per-identity entry summary tables plus class/bin/correlation summaries
  - Defaults to `threshold == "UNCLUSTERED"`

- `plot_entry_alphafill_metric_correlations.py`
  - Reads the rebuilt per-entry analysis tables
  - Generates heatmaps, pair-matrices, separation panels, and support-grid panels
  - Writes both `.png` and `.svg`
  - Supports the same `--holo_plus_one` evidence correction used in the UBS branch

- `plot_entry_alphafill_unified.py`
  - Entry-level counterpart to the compact UBS plotting script
  - Generates match-rate overviews, class match-rate summaries, and focused unclustered heatmaps
  - Useful for directly comparing unclustered entry behavior to the clustered UBS branch

- `plot_entry_ridgelines.py`
  - Entry-level counterpart to the UBS ridgeline/violin workflow
  - Reads `per_entry_summary_*` tables from `analysis_v1`
  - Generates ridgelines, full violins, and vertical half-violins
  - Supports percentile-based or fixed metric caps and dashed/box half-violin summary styles
  - Default tail handling is now percentile-based (`metric_cap_mode=quantile`, `metric_cap_quantile=0.975`)
  - Writes `.png` and `.svg` for the vertical half-violin exports

- `confidence_labeling.py`
  - Shared database-facing verbal confidence thresholds copied from the UBS branch
  - Keeps entry-level probability/density/separation labeling aligned with AHoJ

Notes

- The current `alphafill_unified` folder remains the authoritative UBS branch.
- This folder is intentionally parallel rather than merged, so downstream entry
  analyses can evolve independently without complicating the UBS scripts.
- The first downstream outputs now live under:
  - `work_dir/alphafill_unified_entries/run1/analysis_v1`
  - `work_dir/alphafill_unified_entries/run1/analysis_v1/correlation_panels_v1`
  - `work_dir/alphafill_unified_entries/run1/analysis_v1/ridgeline_plots`
