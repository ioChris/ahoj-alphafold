# 03 — AlphaFill ligand-transplant comparison

**Analysis 4.** Compare AFDB apo/holo pocket classes against **AlphaFill** ligand-transplant
fit (clash score/count, local/global RMSD), stratified by how well-separated the experimental
apo/holo states are. AF-Apo pockets fit transplanted ligands worse (more clash, higher RMSD)
mainly when the experimental apo/holo states are well separated.

Paths come from [`../common/config.py`](../common/config.py).

### Data-prep pipeline (this directory)
Run in order:

1. `alphafill_download_jsons_v2.py` — download AlphaFill JSONs for the project UniProts
   (`ANALYSIS_UNIFIED_LEVELS_CSV` → `ALPHAFILL_JSONS_DIR`).
2. `alphafill_parse_jsons_to_table_v3.py` — parse JSONs into a flat transplant table
   (→ `ALPHAFILL_DIR/alphafill_analysis/v2/alphafill_parsed.*`).
3. `merge_ahoj_alphafill_v6.py` — job-to-job merge of the unified AHoJ table with AlphaFill
   transplants (→ `MERGED_AHOJ_ALPHAFILL_{SUMMARY,LONG}_CSV`).

### Analysis branches
- [`ubs/`](ubs/) — **UBS-level** branch (build → analyze → plot). The manuscript AlphaFill
  figure is assembled here. See [`ubs/README.md`](ubs/README.md).
- [`entries/`](entries/) — **entry-level** (`UNCLUSTERED`) counterpart, kept parallel so the
  two levels can evolve independently. See [`entries/README.md`](entries/README.md).

Each branch keeps its own small `confidence_labeling.py` (verbal confidence-band helper,
vendored from the AHoJ-DB methodology) so it runs standalone; see
[`../common/README.md`](../common/README.md).
