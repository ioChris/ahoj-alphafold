# Redock RMSD (run_3, G1 query-holo jobs)

Builds an **experimental reference pose** for every group-1 (query-holo) docking job and
computes a **symmetry-corrected, in-place** heavy-atom RMSD between the docked top pose
and that reference — i.e. how well Uni-Dock reproduced the crystallographic pose when it
docked the cognate ligand into the **AlphaFold** model. `redock_success = RMSD ≤ 2 Å`.

This is the functional complement to affinity / `pose_in_pocket`: it feeds
`05_docking/runner/analyze_docking_pilot.py --redock_csv`, joined to the manifest/results
by `job_id`.

## Why a run_3-specific pipeline (vs the pilot `runner/` scaffold)

The pilot `runner/build_reference_poses_local.py` + `compute_redock_rmsd.py` read a
handover bundle with pre-computed `escort_*` columns and perceived ligand bonds by
inter-atomic distance. Two things break that at full scale; this package fixes them:

- **Incomplete crystal ligands** (disordered atoms) → distance-based perception can't
  produce a clean subgraph. Here the experimental ligand is mapped onto the **CCD
  component** graph **by atom name** (PDB atom ids == `_chem_comp_atom.atom_id`), so
  missing atoms are simply absent. RMSD is taken over the *observed* atoms only.
- **Raw docked PDBQT** (no bond orders, only the top pose, packed `>>> {job_id}` inside
  `out/shard_*.tar.gz`) → reconstructed **in process** from the Meeko `REMARK SMILES`
  every pose carries (`meeko.PDBQTMolecule` + `RDKitMolCreate`), no `mk_export` subprocess.

RMSD = minimum over the CCD template's graph automorphisms of the in-place RMSD over the
observed atoms — verified identical to `spyrmsd.symmrmsd(minimize=False)`.

## Pipeline

```
# env: ahoj-docking  (rdkit, gemmi, spyrmsd, meeko, pandas)
# ALWAYS export KMP_DUPLICATE_LIB_OK=TRUE (else the OpenMP DLLs native-crash on Windows)

# 1. escort table (local, no network) — G1 jobs + receptor chain + pocket residues
python build_escort_table.py

# 2. fetch source structures (~10-20 GB) + CCD components (to a LOCAL cache, not OneDrive)
python fetch_structures.py --workers 12

# 3. compute redock RMSD  ->  run3/reference_poses/redock_rmsd.csv
python compute_redock.py --workers 16
```

`--limit_pdbs N`, `--job_ids_file`, and a per-subset `--escort_csv` support validation runs.

## Key facts (measured, run_3)

- G1 = 73,685 jobs; **69,395 have a docked pose** (4,290 errored docking jobs have none).
- **43,382 unique source PDBs**, 23,700 CCDs. Receptor chain from
  `QUERY_SUMMARIES_CLASS_CSV` (`ligand_ref == job_name`, **100 % coverage**,
  `prot_chains`); all G1 pockets are single-chain. Do **not** auto-detect the chain by
  sequence — many sources are large assemblies (e.g. 7uvz is a 53-chain ribosome) and the
  job-key chain is the *ligand* chain, not the protein one.
- Superposition is recomputed on the **v6** AF model (what docking used), so the reference
  lands in the exact docked frame — no re-superposition at RMSD time.

## Outputs (`run3/reference_poses/`)

| file | content |
|---|---|
| `escort_g1.csv` | per-job inputs (uniprot, af model, src pdb/chain/ccd/resnum, prot_chain, pocket positions, has_pose) |
| `redock_rmsd.csv` | **`job_id, redock_rmsd_top, redock_rmsd_best, redock_success`** + QC (`status`, `n_ref_atoms`, `ref_completeness`, `superpose_rmsd`, `superpose_mode`, `n_pocket_pairs`, `affinity_top`) |
| `*_manifest.json` | provenance + status/quality summaries |

`redock_rmsd_best == redock_rmsd_top` (the HPC run stored only the top pose per job).
Filter analysis on `status == "ok"` and, for a strict set, `superpose_mode == "pocket"`
(a `wholechain` fallback means < 3 pocket CA pairs mapped and the frame is less reliable).
