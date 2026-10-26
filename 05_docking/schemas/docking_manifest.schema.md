# `docking_manifest.csv` — input contract (column dictionary)

One row per **docking job** = one (pocket × ligand) pair. Produced by
`05_docking/dataset/build_docking_manifest.py`. This is what the colleague's runner consumes.

## Identity & join keys
| Column | Type | Notes |
|---|---|---|
| `job_id` | str | Unique per row. **Echo back unchanged** in results. Format: `{site_uid}__g{group}__{ligand_id}` (sanitized). |
| `site_uid` | str | Pocket / UBS id. Primary key back into the AHoJ-AlphaFold per-pocket table. |
| `site_uniprot` | str | UniProt accession = receptor id. |
| `threshold` | str | Clustering threshold (fixed `ST_100` for now); carried for provenance. |

## Receptor
| Column | Type | Notes |
|---|---|---|
| `afdb_filename` | str | e.g. `AF-P12345-F1-model_v4.cif.gz`. Highest local model version. |
| `af_model_len` | int | Residues indexed in the model (QA). |

## Ligand
| Column | Type | Notes |
|---|---|---|
| `ligand_group` | int | `1`=query holo (AHoJ), `2`=AlphaFill cognate, `3`=probe. |
| `ligand_id` | str | CCD (groups 1–2) or probe id (group 3). |
| `ligand_ref` | str | Provenance: source job key `pdb_chain_ccd_resnum` (g1), AlphaFill `reconstructed_key` (g2), or `probe:<SMILES>` (g3). |
| `ligand_smiles` | str | SMILES if resolved (may be empty; SDF is authoritative). |
| `ligand_sdf` | str | Relative path to the prepared 3D ligand, e.g. `ligands/ccd/7I1.sdf`. |
| `ligand_n_heavy_atoms` | int | Heavy-atom count if resolved (size covariate; may be empty). |

## Box (from pocket geometry)
| Column | Type | Notes |
|---|---|---|
| `center_x` / `center_y` / `center_z` | float | Box center (Å) = pocket-residue center-of-mass (`union_center_com_*`). |
| `size_x` / `size_y` / `size_z` | float | Box size (Å). Isotropic in `rg` mode; per-axis in `bbox` mode. |
| `box_mode` | str | `rg` or `bbox` — how the size was derived. |
| `pocket_rg` | float | `union_rg` (pocket size covariate). |
| `pocket_n_residues` | int | `union_n_residues_found` (pocket size covariate). |
| `pocket_chain_resi` | str | `union_chain_resi`, e.g. `"A_50 A_51 ..."` (for in-pocket checks). |

## Suggested engine settings (override allowed, but record actual)
| Column | Type | Notes |
|---|---|---|
| `exhaustiveness` | int | Vina exhaustiveness (default 8; 16–32 for the pilot is fine). |
| `num_modes` | int | Poses to output (default 9). |
| `seed` | int | RNG seed for reproducibility. |
| `has_reference_pose` | bool | True if `reference_poses/{site_uid}__{ligand_id}.sdf` exists (enables redock RMSD). |

## Carried classification metadata (for our analysis; runner can ignore)
These let a returned results file be analyzed even standalone. They are **not** needed to
dock. `pocket_class`, `pocket_p_apo`, `pocket_p_holo`, `pocket_separation_pair`,
`pocket_score`, `model_pocket_plddt`, `n_apo_avg`, `n_holo_avg`, `n_unp_pockets`.

---

### Sidecar `docking_manifest_manifest.json`
Records: input CSVs + sha256, adapter git commit, `box_mode` and box params, ligand-group
selection, row/pocket/receptor/ligand counts, and (for the pilot) the stratification spec.
