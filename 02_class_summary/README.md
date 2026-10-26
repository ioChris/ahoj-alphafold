# 02 — class composition / prevalence summaries

**Analysis 3.** Summarize how the AFDB pocket apo/holo **class composition shifts with
clustering degree** (holo-excess at entry level moderates toward parity as redundancy is
reduced), plus site-size and classification-flow summaries. All scripts read the master
`analysis_unified_levels` table or the UBS aggregated-classifications table via
[`../common/config.py`](../common/config.py).

| Script | Role |
|--------|------|
| `prevalence_sweep_r1.py` | **Prevalence sweep** (current): apo/holo prevalence and counts across Jaccard thresholds; multi-panel figures + per-threshold summary CSVs. Supersedes the archived `prevalence_sweep.py`. |
| `site_size_vs_threshold_summary_v2d.py` | Pocket/site size (`resis3_unit`) across thresholds; emits raw, main-text, and SI summary tables. |
| `sankey_plot_06.py` | Multi-threshold / multi-mode Sankey diagrams of apo/holo classification flow (density → separation → probability → class). |
| `fine-grain_sankey.py` | Fine-grained, alpha-blended Sankey over the same class hierarchy (legacy v1.2 subset input). |
| `parse_initial_classification_aggregations_per_ubs.py` | QC / inspection of the aggregated classifications (counts by apo/holo/tie/no-data). |
| `analyze_aggregate_classifications_per_ubs.py` | Interactive Plotly inspection of UBS classification levels (legacy v1.2 subset input). |

See [`prevalence_plots_description.md`](prevalence_plots_description.md) for a detailed,
caption-ready description of the prevalence plot family (notation, modes, interpretation).
