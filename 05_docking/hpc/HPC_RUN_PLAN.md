# Full docking run on HPC (Aurum GPU) — plan, estimates & scaffold

Prepared 2026-07-03. Companion to the CPU pilot (`05_docking/runner/`). This plans the
**full ~100k–760k-job GPU docking run** on **Aurum** (IOCB HPC), with runtime/storage
estimates and a launch-ready SLURM scaffold. Cluster facts below are from the cluster's documentation (2026-06 snapshot); verify
live before submitting.

> **Status:** written before the run as its plan. The calibration (§8) and the full run were
> carried out with these scripts; the estimates below are the pre-run ones.

---

## 1. Scale of the full run

From the current ST_100 dataset (`afdb_st_100_dataset_with_geometry.csv`):

| Quantity | Count |
|---|---|
| Pockets (site_uid) | **82,130** |
| Unique receptors (UniProt AFDB models) | **9,985** |
| Unique group-1 CCDs (query-holo) | 18,356 |
| **G1** query-holo dockings (rep/pocket, ion-filtered) | **~92,000** |
| G1 if *all* query members docked | 178,998 |
| **G2** AlphaFill-cognate (~16% of pockets) | **~13,000** |
| **G3** probe library (**8 probes** × all pockets) | **657,040** |

Probe set (fixed, ligand-controlled dockability): benzene, imidazole, phenol,
benzamidine, indole, acetaminophen, ibuprofen, caffeine.

**SCOPE (decided 2026-07-03):** dock **all distinct cognate ligands** (G1 all members) +
G2; **drop G3 probes** (optional stratified add-on only). Cognate dock/no-dock is the real
functional signal — more interesting than random probes.

`query_ligands_group1_members.csv` has 178,998 member rows but only **91,683 unique
(pocket × CCD)** pairs — 93% of pockets have a single cognate CCD, and docking the same CCD
from different source PDBs into the same pocket is redundant (identical receptor+box+ligand).
So the docking unit is **unique (pocket × CCD)**; the extra members still matter for **redock
RMSD** (each source PDB gives its own experimental reference pose → a docked pose can be scored
against several).

| Scope | Docking jobs | Note |
|---|---|---|
| **G1 all-members (unique pocket×CCD)** | **91,683** (~67,800 after ion filter) | the real cognate test |
| **G2 AlphaFill cognate** | ~13,000 | |
| **Chosen total (G1+G2, no probes)** | **~105k** (~81k dockable) | ~7× smaller than the probe plan |
| ~~G3 probes × all pockets~~ | ~~657,040~~ | dropped |
| optional stratified G3 (8 × ~5k pockets) | ~40,000 | if a ligand-identity control is wanted |

---

## 2. Runtime estimate (Uni-Dock on L40S)

**Engine:** Uni-Dock (GPU, Vina scoring family — same as the pilot's Vina baseline, so
scores are comparable). **Aurum gen-d** node = AMD EPYC 9654 (192 cores) + **2–4× NVIDIA
L40S (48 GB, Ada)** — a strong docking GPU.

**Cost model.** Time per pocket = one affinity-map (grid) build (amortised over all ligands
sharing that pocket's box) + per-ligand search. Because we cross-dock **many receptor-boxes
with few ligands each**, the run is partly **grid-bound** (unlike a 1-target virtual screen).
Assumptions: grid ≈ 2 s/pocket-box on L40S (our boxes are large, 16–30 Å, median 24 Å);
per-dock ≈ 0.2 s (small rigid probe) to 0.6–2.5 s (CCD, size-dependent). **Batch all of a
pocket's ligands (G1+G2+G3) against its single grid.**

| Scenario | Jobs | GPU-hours | Wall on 1 node (4× L40S) | Wall on 16 GPUs |
|---|---|---|---|---|
| **G1 all-members + G2 (chosen)** | **~105k** | **60–120** | **15–30 h** | **4–8 h** |
| + optional stratified G3 probes | ~145k | 75–140 | 19–35 h | 5–9 h |
| ~~G1+G2+G3 all probes~~ | ~~762k~~ | ~~100–192~~ | ~~25–48 h~~ | ~~6–12 h~~ |

**Sensitivity.** The range is dominated by (a) per-dock time (ligand flexibility — large
cofactors are 10–50× a probe) and (b) grid time (box size). The CPU pilot confirmed the
flexibility tail is real: at Vina exh 8, ligands >45 heavy atoms took 10–20 min *each* on
1 CPU core — on GPU these collapse to seconds, but they still dominate relative cost.
**→ Run a calibration shard first** (§6): 1× L40S, ~2,000 pockets, all groups, measure real
throughput and confirm Uni-Dock↔Vina agreement before committing the full submission.

**Scheduling shape.** Aurum prefers **short jobs (≤4 h)** and limits ~2,000 array
elements/user with ≤~50 concurrent (etiquette). So shard into **array tasks of ≤4 h each**
(≈150–400 pockets/task depending on group mix), submitted in controlled waves. With ~30–50
GPUs in flight, the full run completes in **well under a day of wall-clock**.

---

## 3. Storage estimate

Sizes are trivial for Aurum's 4 TB `$HOME`; the **binding constraint is inode/metadata**
(the wiki guideline is **< 1 million files** on NFS, and small-file I/O on NFS is discouraged).

| Item | Size |
|---|---|
| AFDB receptor models (9,985 × ~93 KB cif.gz) | ~0.9 GB |
| Receptor PDBQT (9,985 × ~360 KB) | ~3.5 GB |
| Ligand PDBQT (18k CCDs + probes) | ~0.15 GB |
| Manifests / reference poses | ~1 GB |
| **Pose SDFs — all 9 modes (~105k × 45 KB)** *(keeping all modes, per decision)* | **~4.7 GB** |
| Results table (parquet, ~105k rows) | ~0.1 GB |
| **Total (all 9 modes, packed)** | **~10 GB** |

For the chosen ~105k-job scope, keeping all 9 modes is only ~5 GB of poses — so **retain all
modes** (needed for redock-RMSD-per-member anyway). Fits easily on Aurum `$HOME` or a local
`D:` copy; no sharing/upload required. (With probes it would have been ~40 GB — moot now.)

**The file-count trap.** One SDF + one log per job = **~1.5 million files** → exceeds the
inode guideline and cripples NFS. **Mandatory design:**
1. Every array task writes to **node-local NVMe** (`$SCRATCH` = `/scratch/slurm_jobs/$USER/$SLURM_JOB_ID`; request `--constraint=gen-d&scratch` — gen-d L40S nodes carry the generic `scratch` feature at ~1.7 TB, **not** the gen-a-only `scratch2TB` tag).
2. Within a task, write **one packed bundle**: a single concatenated multi-SDF (`poses_<shard>.sdf.gz`) + one `results_<shard>.parquet`, **not** per-job files.
3. **tar** the bundle and copy the tarball back to `<project dir>/…` (or `$HOME`) at task end; scratch is purged.
Result: ~500–2,000 output files total (one bundle per shard) instead of 1.5 M.

---

## 4. Environment on Aurum

Anaconda/Miniconda are **unsupported** (licensing). Two supported routes; **container is preferred** for reproducibility across GPU generations:

**A. Apptainer container (recommended)** — `docking.def` in this folder builds a CUDA image
with `unidock`, `vina`, `rdkit`, `meeko`, `gemmi`, `spyrmsd`. Build on a gen-d interactive
node (login node blocks compiles), store in `<software dir>/`:
```bash
salloc --constraint=gen-d --gres=gpu:1 --time=01:00:00 --mem=32G
apptainer build <software dir>/docking.sif 05_docking/hpc/docking.def
apptainer exec --nv <software dir>/docking.sif unidock --version
```

**B. Miniforge + mamba** (if containers are inconvenient):
```bash
mamba create -p <software dir>/conda-envs/docking -c conda-forge \
    python=3.11 rdkit meeko gemmi numpy pandas spyrmsd
# Uni-Dock: conda-forge `unidock`, or build from github.com/dptech-corp/Uni-Dock
```

---

## 5. Job architecture

```
inputs (NFS, staged once)                per-shard on NODE-LOCAL scratch
────────────────────────                 ──────────────────────────────
docking_manifest.parquet   ─┐            $SCRATCH/
receptors/*.cif.gz          ├─ rsync ─►   ├─ prep receptor+ligand PDBQT (cached)
ligands/*.sdf               │            ├─ unidock: per pocket, 1 grid × all ligands
shards.csv (pocket→shard)  ─┘            ├─ parse → results rows + top poses
                                          └─ pack → poses_<shard>.sdf.gz + results_<shard>.parquet
                                                     └─ tar ─► <project dir>/out/
```

- **Shard = a contiguan block of pockets** (all their G1/G2/G3 ligands together, so each
  pocket's grid is built once). `dock_shard.py` takes `--shard i --nshards N`.
- **SLURM array** (`dock_array.sbatch`): `--array=0-(N-1)%50`, `--constraint=gen-d
  --gres=gpu:1 --time=04:00:00`, request node-local scratch.
- **Resumable:** a shard whose output tarball already exists is skipped → safe re-submit.
- **Idempotent packing:** no per-job files ever touch NFS.

Concrete files in this folder: `docking.def`, `dock_shard.py`, `dock_array.sbatch`,
`submit_waves.sh`, `README.md`.

---

## 6. Recommended sequence

1. **Access & stage.** Get an account on the cluster (SSH key) from its administrators. `rsync` the manifest + receptor cif.gz + ligand SDFs to `<project dir>/ahoj_dock/`.
2. **Build container** (§4A) on a gen-d node.
3. **Calibration shard.** Submit ONE array task over ~2,000 stratified pockets (all groups) on
   1× L40S. Measure: dockings/GPU-hour, VRAM, output size/shard, and Uni-Dock↔pilot-Vina
   affinity agreement on shared jobs. This turns the §2 range into a firm number.
4. **Scale out.** Set shard size so each task ≈ 3–4 h; submit in waves (`submit_waves.sh`,
   ≤50 concurrent). Full run: hours-to-≈1-day wall.
5. **Collect.** Un-tar shard bundles, concatenate `results_*.parquet`, run
   `05_docking/runner/analyze_docking_pilot.py` (+ P2Rank/AlphaFill joins) at full scale.

---

## 7. Decisions (resolved 2026-07-03)

- **Probes (G3): dropped** from the primary run (they were 86% of compute). Optional stratified
  add-on (8 probes × ~5k pockets ≈ 40k jobs) only if a ligand-identity control is later wanted.
- **G1 = all distinct cognate ligands** (all members → unique pocket×CCD, ~92k), **not** just the
  rep. Cognate dock/no-dock is the primary functional signal. Extra same-CCD members feed
  **redock RMSD** (one experimental reference pose per source PDB).
- **Poses: keep all 9 modes** (~5 GB at this scale; needed for per-member RMSD). No sharing/upload.
- **Cluster:** Aurum gen-d (L40S) target; **biogpu** (1× RTX 4090) only for a quick calibration.
- **Public/private split:** plan + generic scaffold stay public here; cluster-specific
  hostnames/paths/account contacts → private repo per the project's doc-split convention.

### Still to do on the cluster (the actual deployment)
1. ~~**Calibration shard** on Aurum~~ — **DONE 2026-07-04** (see §8).
2. Full run when the **new classification manifest** is final (the adapter re-emits it; scaffold
   unchanged). `dock_shard.py` docks unique (pocket×CCD); build reference poses per member.

---

## 8. Calibration results (2026-07-04, verified on Aurum L40S)

Ran the pilot's 1,941 jobs as a 4-shard array on 4× L40S (Uni-Dock, `search_mode=balance`,
9 modes). This turns the §2 range into firm numbers:

| Metric | Result |
|---|---|
| **Throughput** | **~2,105 dockings/GPU-hour** (0.585 jobs/s/GPU; 1,941 jobs in 3,320 GPU-s) |
| **Peak VRAM** | **589–681 MiB** of 48 GB (~1.4% — VRAM is a non-constraint; grid+all-ligands batch is tiny) |
| **Success** | 1,826/1,941 ok (94%); the 115 errors are the non-dockable inorganics/metals (as in the pilot) |
| **pose_in_pocket** | 93.2% of successful docks land in the manifest pocket |
| **Packed output** | ~280 KB/shard → **~60 MB for the full ~105k-job run** (top-pose geometry + all-9 affinities) |

**Uni-Dock ↔ pilot-Vina agreement** (1,826 shared jobs): **Spearman ρ = 0.953**; on the
97.9% bulk (both engines in a sane range) **Pearson = 0.958, RMSD = 0.60 kcal/mol**; median
bias **+0.07 kcal/mol**; **dockable agreement 94.1%**, pose_in_pocket 95.5%. The only
divergence is ~1.6% of jobs — large flexible ligands where Uni-Dock returns a positive clash
energy vs Vina's negative — the pilot's dockability tail, flaggable as `affinity_top > 0`.

**Firm full-run projection (~105k jobs, G1 all-members + G2):** **~45–60 GPU-hours**
(below the §2 estimate). Recommended **NSHARDS ≈ 50** (each ~1,600 pockets ≈ ~1 h) — short
tasks backfill well, which matters: **all L40S GPUs were allocated at submit time** (the
calibration queued ~10 min). Submit via `submit_waves.sh` (≤50 concurrent).

**Verified-live cluster facts / scaffold fixes:** gen-d L40S nodes (d051–d100) carry feature
`scratch` (Gres `scratch:1698G`), **not** `scratch2TB` → `dock_array.sbatch` uses
`--constraint=gen-d&scratch`. Image build needs `--mksquashfs-args "-processors 1"` (mksquashfs
"Bug in orderer" on 192-core nodes). apptainer 1.5.1; Uni-Dock v1.1.3; `$SCRATCH` auto-exported.
