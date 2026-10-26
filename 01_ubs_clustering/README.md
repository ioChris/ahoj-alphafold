# 01 — UBS clustering & master analysis table

**Analysis 2.** Cluster AHoJ entries into **Unified Binding Sites (UBS)** by Jaccard
similarity of their UniProt residue sets, at thresholds `ST_100 / ST_070 / ST_050 / ST_030 /
ST_LJ` (`UNCLUSTERED` = per-entry), then assemble the master analysis table consumed by every
downstream stage.

Paths come from [`../common/config.py`](../common/config.py) (configure `common/paths.ini`).

| Script | Role | Key inputs → outputs |
|--------|------|----------------------|
| [`unification/`](unification/) `run_ubs_unification.py` | **Binding-site unification.** Per UniProt, cluster the binding sites by Jaccard similarity of their residue sets into the per-threshold UBS tables. | `QUERY_SUMMARIES_CLASS_CSV` → `UBS_ROOT_DIR/<ACC>/ST_*.tsv` |
| `classification_aggregation_and_interface_ubs_clustering_r.py` | **Main clustering + aggregation.** Parse per-job apo/holo classification, cluster into UBS per threshold, aggregate to one row per (UBS, threshold). | `FILTERED_SURVIVORS_CSV`, `QUERY_SUMMARIES_CLASS_CSV`, `UBS_ROOT_DIR` → `UBS_AGG_DIR/<run>/ubs_allthresholds_aggregated_classifications(_filtered).csv` (+ manifest, run log) |
| `build_analysis_unified_levels.py` | **Master-table builder.** Merge unclustered survivors with the clustered UBS aggregates into one analysis-ready table with dual evidence columns. | `FILTERED_SURVIVORS_ZIP`, `UBS_AGG_FILTERED_ZIP` → `UNIFIED_RUN_DIR/analysis_unified_levels.csv` (+ QC, manifest) |
| `ubs_processing_pipeline_with_afdb_filter.py` | Pre-filter UBS rows by AFDB model presence; compute winning class + density/separation labels; record excluded UBS. | UBS table → `WORK_DIR/afdb_ubs_sets/...` |
| `classification_aggregation_per_ubs_clusters.py` | Library of per-UBS aggregation helper functions (with a legacy usage example). ⚑ Possibly superseded by / bundled into the main script — confirm before relying on it. | — |

`analysis_unified_levels.csv` is the central table read by `plotting/`, `03_alphafill/`, and
`04_p2rank/dataset/`.
