# `05_docking/hpc/` — GPU full-run scaffold (Aurum)

Launch-ready scaffold for the **full ~100k–760k-job docking run** on **Aurum** GPU nodes,
using **Uni-Dock** (Vina scoring family — comparable to the CPU pilot). Read
[`HPC_RUN_PLAN.md`](HPC_RUN_PLAN.md) first: it has the scale, runtime/storage **estimates**,
and the sharded/packed-output architecture that protects the shared-filesystem inode budget.

## Files
| File | Role |
|---|---|
| `HPC_RUN_PLAN.md` | plan + runtime/storage estimates + open decisions |
| `docking.def` | Apptainer image (Uni-Dock + vina + rdkit + meeko + gemmi + spyrmsd) |
| `dock_shard.py` | dock ONE shard of pockets (1 grid × all ligands/pocket), pack → 1 bundle |
| `dock_array.sbatch` | SLURM array task = one shard on 1 L40S (gen-d) |
| `submit_waves.sh` | controlled wave submission (≤2000/array, ≤50 concurrent) |
| `collect_results.py` | un-tar shard bundles → one `docking_results.parquet` for analysis |

## Flow
```
build docking.sif (docking.def, on a gen-d node)
        │
prep receptors+ligands PDBQT once  ──►  stage to <project dir>/inputs/ (NFS)
        │
NSHARDS=<N> WAVE=400 CONC=40 ./submit_waves.sh      # array of shards, each ≤4 h on 1 GPU
        │   each task: node-local $SCRATCH → unidock per pocket → pack → tar to out/
        ▼
collect_results.py --in out/ --out docking_results.parquet
        │
05_docking/runner/analyze_docking_pilot.py  (join classification + P2Rank + AlphaFill)
```

## Before you scale out
1. **Access:** an account on the cluster (SSH key) from its administrators.
2. **Container:** build `docking.sif` on a `--constraint=gen-d` interactive node.
3. **Prep:** reuse `05_docking/runner/prep_receptors.py` + `prep_ligands.py` to build the
   PDBQT caches (they're engine-independent), then rsync to the project dir.
4. **Calibrate:** submit ONE shard over ~2,000 stratified pockets; measure dockings/GPU-hour
   and Uni-Dock↔Vina agreement (turns the estimate range into a firm number and sizes NSHARDS
   so each task ≈ 3–4 h).
5. **Scale:** `submit_waves.sh`; collect; analyse.

## Notes
- Reuses the pilot runner's parsers (`../runner/prep_common.py`) — keep both folders together.
- Engine is swappable (`--engine unidock|vina`) so the calibration cross-checks the CPU pilot.
- All outputs are **packed per shard** — never one file per job (NFS inode protection).
- Scripts target the cluster and are validated by the calibration shard, not on this
  workstation (no local GPU). Cluster-specific hostnames/paths are marked for the private repo.
