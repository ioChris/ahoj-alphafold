# P2Rank AFDB Analysis

Analysis and visualization scripts for P2Rank predictions run on AFDB models with simulated ligand centers derived from AHoJ-DB UBS definitions.

Scripts:

- `analyze_p2rank_ubs_run4.py`
  Builds a merged per-pocket table from P2Rank `cases/*.csv` files and AHoJ ST_100 UBS metrics.
- `plot_p2rank_ubs_run4.py`
  Creates a first plot set from the merged tables and grouped success summaries.
- `analyze_p2rank_descriptor_correlations.py`
  Adds AHoJ pocket descriptor variability features to the ST_100 P2Rank table and computes correlations with P2Rank success.
- `confidence_labeling.py`
  Shared database-aligned verbal confidence binning helpers.

Default outputs:

- `work_dir/P2Rank/ubs_run_4/analysis_v1/tables`
- `work_dir/P2Rank/ubs_run_4/analysis_v1/plots`

## Descriptor Correlations

`analyze_p2rank_descriptor_correlations.py` is a follow-up analysis for comparing P2Rank success against AHoJ pocket descriptor metrics used by the apo/holo classifier.

The script focuses on the ST_100 UBS table produced by `analyze_p2rank_ubs_run4.py`. It uses the ST_100 union-geometry table to expand each UBS into its member AHoJ `job_name` values, joins those jobs to AHoJ descriptor summary tables, aggregates descriptor features back to the UBS level, and computes Pearson and Spearman correlations against the P2Rank success columns.

Default descriptor sources:

- Per-query descriptor summary: `AHOJ_DB/v2c/Summaries/parsing_v3c5_DBv2b_561batches_511845jobs_80seqres_Noneobs_4.5res_0fullpcktaln_sums0/query_summaries_rog_DQ5.csv`
- Apo/holo min/max summaries for exact range features: `AHOJ_DB/v2c/Summaries/parsing_v3c3_DBv2b_561batches_518883jobs_80seqres_Noneobs_4.5res_0fullpcktaln/{apo_summaries.csv,holo_summaries.csv}`

Primary descriptor metrics:

- `pocket_rms`
- `volume`
- `sasa`
- `RoG`

Main outputs:

- `tables/p2rank_st100_descriptor_features_per_ubs.csv`
- `tables/p2rank_st100_merged_with_descriptor_features.csv`
- `tables/p2rank_descriptor_correlations.csv`
- `tables/p2rank_descriptor_quantile_success.csv`
- `tables/p2rank_descriptor_manifest.json`
- `plots/descriptor_metrics/descriptor_top_spearman_success_tns.png`
- `plots/descriptor_metrics/descriptor_range_spearman_success_tns.png`
- `plots/descriptor_metrics/descriptor_range_deciles_success_tns.png`

Example:

```bash
# Run from this directory (04_p2rank/analysis/) after configuring common/paths.ini.
# Default input/output paths come from common/config.py.
python analyze_p2rank_descriptor_correlations.py
```

Useful overrides:

```bash
python analyze_p2rank_descriptor_correlations.py \
  --primary_success success_dca4_sites \
  --out_dir /path/to/work_dir/P2Rank/ubs_run_4/analysis_v1
```

Range features named like `volume_all_range_mean_across_jobs` are computed per AHoJ query as `max(apo/holo max) - min(apo/holo min)`, then averaged across all member jobs in a UBS. Summary features named like `volume_std_dev_mean_across_jobs` are taken from `query_summaries_rog_DQ5.csv` and then aggregated across member jobs.

## Query Metric Block Correlations

`analyze_p2rank_query_metric_block_correlations.py` is a broader sweep over the contiguous query-summary metric block from `local_flexibility` through `rmsd_coefficient_of_variation`. It expands each ST_100 UBS into member AHoJ jobs, joins the metric block from `query_summaries_rog_DQ5.csv`, aggregates each metric with mean/median/max across member jobs, derives confidence-interval widths, and computes Pearson/Spearman correlations against all available P2Rank success columns.

Main outputs:

- `tables/p2rank_st100_query_metric_block_features_per_ubs.csv`
- `tables/p2rank_st100_merged_with_query_metric_block_features.csv`
- `tables/p2rank_query_metric_block_correlations.csv`
- `tables/p2rank_query_metric_block_primary_mean_spearman_matrix.csv`
- `plots/query_metric_block/query_metric_block_top_spearman_success_tns.png`
- `plots/query_metric_block/query_metric_block_metric_subscore_heatmap_success_tns.png`

Example:

```bash
python analyze_p2rank_query_metric_block_correlations.py
```
