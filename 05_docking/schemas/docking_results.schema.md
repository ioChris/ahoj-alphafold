# `docking_results.csv` — output contract (what we need back)

**One row per input `job_id`.** Every job in the manifest must appear here, including
failures (with `status="error"`). Consumed by `05_docking/analysis/ingest_docking_results.py`.

## Required — identity (must match the manifest)
| Column | Type | Notes |
|---|---|---|
| `job_id` | str | Unchanged from the manifest. Join key. |
| `site_uid` | str | Unchanged. |
| `ligand_group` | int | Unchanged. |
| `ligand_id` | str | Unchanged. |

## Required — docking outcome
| Column | Type | Notes |
|---|---|---|
| `status` | str | `ok` \| `error` \| `skipped` (+ reason in `error`). |
| `error` | str | Empty unless `status!=ok`. |
| `vina_affinity_top` | float | Best (most negative) affinity, kcal/mol. Empty if error. |
| `vina_affinities` | str | All mode affinities, e.g. `"-8.1;-7.9;-7.2"` (semicolon-sep). |
| `n_modes` | int | Number of poses returned. |

## Required — localization / pose recovery
| Column | Type | Notes |
|---|---|---|
| `pose_in_pocket` | bool | Top pose has ≥3 heavy atoms within 4 Å of `pocket_chain_resi`. (We can recompute if you return poses.) |
| `redock_rmsd_top` | float | Heavy-atom, symmetry-corrected RMSD of top pose to reference. Empty if `has_reference_pose=False`. |
| `redock_rmsd_best` | float | Best RMSD over returned modes. Empty if no reference. |

## Required — reproducibility
| Column | Type | Notes |
|---|---|---|
| `engine` | str | `vina` \| `unidock` \| `smina` \| `gnina` \| ... |
| `engine_version` | str | e.g. `vina 1.2.5`. |
| `scoring_function` | str | `vina` \| `vinardo` \| `ad4` \| `cnn(gnina)` ... |
| `seed` | int | Actual seed used. |
| `exhaustiveness` | int | Actual value. |
| `size_x`/`size_y`/`size_z` | float | Actual box used (echo, in case overridden). |
| `runtime_s` | float | Wall-clock seconds for the job. |

## Optional — extras (welcome)
| Column | Type | Notes |
|---|---|---|
| `pose_path` | str | Relative path to top-pose SDF, e.g. `poses/{job_id}.sdf`. |
| `cnn_score` / `cnn_affinity` | float | If gnina rescoring is used. |
| `ligand_n_heavy_atoms` | int | Confirmed heavy-atom count of the docked ligand. |
| `receptor_prep` / `ligand_prep` | str | Prep policy tags (protonation, trimming). |

---

### Sidecar `docking_results_manifest.json` (required)
`engine`, `engine_version`, `scoring_function`, protonation policy, receptor-trim policy,
box mode, default exhaustiveness/num_modes/seed, run date, hardware (CPU/GPU), and any
deviations from the manifest suggestions. This is our provenance record for the paper's
Methods / Data Availability.

### Notes
- Keep affinity sign convention native to the engine (Vina: negative = better).
- If you return `poses/`, we recompute `pose_in_pocket` and (with reference poses) RMSD on
  our side, so those columns can be left empty if inconvenient — but `vina_affinity_top`,
  `status`, and the reproducibility columns are always required.
