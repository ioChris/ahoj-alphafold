# 05_docking/analysis — ingest & correlate docking results

Run **after** the colleague returns `docking_results.csv`
([output contract](../schemas/docking_results.schema.md)).

- **`ingest_docking_results.py`** — joins results to the ST_100 geometry per-pocket table on
  `site_uid`, derives docking outcomes (affinity goodness, redock success ≤ cutoff,
  dockability), and computes **per-ligand-group** correlations of each outcome against
  `pocket_class` / `pocket_p_holo` / `pocket_separation_pair` / `model_pocket_plddt`, both
  zero-order and **size-controlled** (partial Spearman controlling `union_rg`). Also emits
  success-by-class means.

  Outputs (under `DOCKING_ANALYSIS_DIR`): `docking_merged_per_pocket.csv`,
  `docking_correlations.csv`, `docking_success_by_class.csv`, `docking_ingest_manifest.json`.

## To promote to a manuscript panel
Mirror the P2Rank treatment (see [`../../04_p2rank/analysis/`](../../04_p2rank/analysis/)):
- controlled-pairs + CMH by size quantile (adapt `analyze_p2rank_controlled_pairs.py`);
- **control ligand heavy-atom count** for groups 1–2 (Vina score scales with ligand size) —
  and lean on group 3 (fixed probe) where ligand identity is constant;
- cross-analysis panels: docking outcome vs AlphaFill `afill_clash_score`/`afill_local_rmsd`
  and vs P2Rank `success_tns` on shared pockets;
- figures styled like the P2Rank panels (success-by-class bars, vs probability, vs
  separation stratified by class, size-controlled panel).

See [`../docs/DOCKING_DESIGN.md`](../docs/DOCKING_DESIGN.md) §8–§9 for the analysis &
confound plan.
