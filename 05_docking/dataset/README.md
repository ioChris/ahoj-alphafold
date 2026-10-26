# 05_docking/dataset — the docking adapter (build the handover bundle)

Converts the ST_100 AFDB **geometry** table (the same one the P2Rank stage builds) into a
docking handover bundle. Paths from [`../../common/config.py`](../../common/config.py).

Run order:

1. **`sample_pilot_pockets.py`** *(pilot only)* — stratified sample of pockets across
   `union_rg` (size) × `pocket_class` (AF-Apo/AF-Holo) × `pocket_separation_pair`
   (low/mid/high), requiring an available group-1 ligand. Balanced across cells so
   hypothesis-critical cells (AF-Apo × high separation) are populated.
   → `DOCKING_PILOT_DIR/pilot_pockets.csv` (+ cell counts, manifest).

2. **`build_docking_manifest.py`** — the core adapter. For each pocket builds the docking
   box (center = `union_center_com`, size = RoG-derived `rg` mode or per-axis `bbox` mode)
   and emits one row per (pocket × ligand) across the selected `--groups`, plus
   `receptors.csv` and `ligands.csv`.
   → `docking_manifest.csv` (+ sidecar). Input contract:
   [`../schemas/docking_manifest.schema.md`](../schemas/docking_manifest.schema.md).

   Key flags: `--groups 1,2,3`, `--pilot_site_uids <csv>`, `--box_mode rg|bbox`,
   `--g1_all_ligands`, `--expect_reference_poses`, box tuning
   (`--k_rg --ligand_pad --padding --min_box --max_box`), engine suggestions
   (`--exhaustiveness --num_modes --seed`).

3. **`resolve_ligands.py`** — fetch RCSB *ideal* SDFs for CCDs (groups 1–2) and embed the
   probe SMILES to 3D (group 3, RDKit); writes a default probe library if none exists.
   → `<bundle>/ligands/{ccd,probes}/*.sdf` + `ligand_resolution.csv`.

Helpers / scaffolds:

- **`parse_ahoj_job_ligands.py`** — parses AHoJ job keys `{pdb}_{chain}_{CCD}_{resnum}` into
  query-holo ligand records (group 1). Importable (`extract_query_ligands`,
  `build_query_ligand_table`) and a QC CLI. *Confirm the key convention on a sample of your
  current `job_names` — it is inferred from `merge_ahoj_alphafill_v6.py`.*
- **`build_reference_poses.py`** — **SCAFFOLD** to place experimental ligands into the AF
  frame for redock RMSD (fetch PDB + CA superposition implemented; SDF export + general
  residue mapping left as validated TODOs). Optional; the pilot runs without it.

Inputs consumed (via config): `AFDB_ST100_DATASET_GEOM_CSV` (pockets + geometry + class
metadata + `rep_job`/`job_names`), `MERGED_AHOJ_ALPHAFILL_LONG_CSV` (group 2),
`DOCKING_PROBE_LIBRARY_CSV` (group 3), `AFDB_DIR` (receptor models, only for `bbox` mode /
receptor prep on the colleague's side).
