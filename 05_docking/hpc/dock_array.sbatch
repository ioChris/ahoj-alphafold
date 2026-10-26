#!/bin/bash
# SLURM array job: one task = one shard of pockets, docked on 1 L40S (Aurum gen-d).
# Submit via submit_waves.sh (controls concurrency & waves). Automatic partition
# selection on Aurum3 -> do NOT hard-code a partition; drive it with constraints.
#
#SBATCH --job-name=ahoj-dock
#SBATCH --constraint=gen-d&scratch        # L40S node + node-local NVMe scratch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00                    # short tier (<=4h) schedules fastest
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# (array range + %concurrency are set on the command line by submit_waves.sh)

set -euo pipefail

# ---- configuration (override via `--export` / environment) ---------------------
# Constraint note: Aurum gen-d L40S nodes carry the generic `scratch` feature
# (Gres=scratch:1698G), NOT `scratch2TB` (that tag is gen-a only) — verified live
# 2026-07-03 via scontrol. Requesting `gen-d&scratch2TB` matches zero nodes.
PROJECT=${PROJECT:-$HOME/ahoj_dock}              # inputs + output bundles
IMAGE=${IMAGE:-$HOME/docking.sif}                # apptainer image (built from docking.def)
CODE=${CODE:-$PROJECT/code/05_docking/hpc}        # this folder, rsynced to the cluster
MANIFEST=${MANIFEST:-$PROJECT/inputs/docking_manifest.parquet}   # .csv also accepted by dock_shard.py
NSHARDS=${NSHARDS:?set NSHARDS in the environment}
# --------------------------------------------------------------------------------

SHARD=${SLURM_ARRAY_TASK_ID}
SCRATCH=${SCRATCH:-/scratch/slurm_jobs/$USER/$SLURM_JOB_ID}
mkdir -p "$SCRATCH" "$PROJECT/out" logs

echo "shard $SHARD/$NSHARDS on $(hostname); GPU:"; nvidia-smi -L || true

# Optional peak-VRAM sampler (off by default; set SAMPLE_VRAM=1 for calibration).
if [ "${SAMPLE_VRAM:-0}" = "1" ]; then
    ( peak=0; while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1);
        [ -n "$u" ] && [ "$u" -gt "$peak" ] && peak=$u; echo "$peak" > "$SCRATCH/.vram_peak"; sleep 5; done ) &
    VRAM_PID=$!
fi

# Stage read-only inputs to node-local scratch once (fast NVMe; avoids NFS small-file I/O).
# receptors/ligands PDBQT are pre-prepped (see HPC_RUN_PLAN); rsync only what a task needs
# is possible, but for simplicity stage the shared caches (a few GB) once per node.
srun apptainer exec --nv \
    --bind "$PROJECT":"$PROJECT" \
    --bind "$SCRATCH":"$SCRATCH" \
    "$IMAGE" \
    python "$CODE/dock_shard.py" \
        --manifest "$MANIFEST" \
        --shard "$SHARD" --nshards "$NSHARDS" \
        --receptors_pdbqt "$PROJECT/inputs/receptors_pdbqt" \
        --receptors_pdb  "$PROJECT/inputs/receptors_pdb" \
        --ligands_pdbqt  "$PROJECT/inputs/ligands_pdbqt" \
        --scratch "$SCRATCH" \
        --out "$PROJECT/out" \
        --engine unidock --search_mode balance --num_modes 9 --seed 42

if [ -n "${VRAM_PID:-}" ]; then kill "$VRAM_PID" 2>/dev/null || true;
    echo "PEAK_VRAM_MiB=$(cat "$SCRATCH/.vram_peak" 2>/dev/null || echo NA)"; fi

echo "shard $SHARD done"
