# plotting — main & SI manuscript figures (classification / prevalence)

Figure scripts for the classification and prevalence analyses (stages 1–3). Each reads the
master `analysis_unified_levels` table and writes into the figure output directories, all
resolved from [`../common/config.py`](../common/config.py) (`ANALYSIS_UNIFIED_LEVELS_CSV`,
`PLOTS_DIR`, `SI_PLOTS_DIR`, `PRESENTATION_DIR`). Run any script standalone (e.g. in Spyder).

File names keep the figure labels of earlier drafts. The submitted manuscript's numbering is in
the figure map of the [top-level README](../README.md#script--figure--table-map); the last
column below repeats it.

| Script | Draws | Manuscript |
|--------|-------|------------|
| `manuscript_v4/` | composers of the multi-panel figures and Table 1 (see [its README](manuscript_v4/README.md)) | Table 1, Figures 2–4, Suppl. Figures S7–S9 |
| `figure1/make_figure1.py` | redundancy-aware analysis-set overview | Figure 1 |
| `figure1_main_REV4.py` | pocket-level state vs evidence across thresholds | Figure 2a–c |
| `figure2_main_REV1.py` | class counts / evidence-ratio / pLDDT hexbin grid | Figure 2d–e |
| `Figure3A_ridgeline_REV2.py` | apo/holo probability ridgelines vs evidence ratio | Figure 2f |
| `figure3C_bars_REV18.py` | density / separation / overlap metric panels | Figure 4e |
| `figure5_plddt_reliability.py` | pocket pLDDT vs conformational diversity | Figure 5 |
| `figureS1_apoholo_pctdiff.py` | apo-vs-holo difference of standard descriptors | Suppl. Figure S1 |
| `figureS1_supplementary.py` | evidence-weighted hexbin grid | Suppl. Figure S2 |
| `figureS2_supplementary_REV4.py` | ratio-only multi-threshold summaries | Suppl. Figure S3 |
| `Figure3B_sankey_REV1.py` | classification-flow Sankey (needs `plotly` + `kaleido`) | — (earlier draft) |
| `figureS3_*.py` | pLDDT-q10 vs evidence ratio and its robustness checks | — (earlier draft; figure dropped) |
| `plot_ratio_vs_separation.py` | evidence ratio vs separation | — (diagnostic) |
