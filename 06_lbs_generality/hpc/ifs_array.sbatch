#!/bin/bash
# SLURM array job: one task = one shard of receptors, IF-SitePred on 1 L40S (Aurum gen-d).
# Automatic partition selection on Aurum3 -> do NOT name a partition; drive via constraints.
# Submit: sbatch --array=0-<N-1>%<conc> --export=ALL,NSHARDS=<N> ifs_array.sbatch
#
#SBATCH --job-name=ahoj-ifs
# CPU-only: ESM-IF1 (142M) runs on CPU in ~6 s/structure; fair-esm 2.0 + torch 2.11
# has a device bug on cuda (ESM-IF1 GVP encoder). No --gres/--constraint -> the
# auto-partition plugin picks a CPU partition (schedules faster than the GPU queue).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00                    # short tier (<=4h) schedules fastest
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-$HOME/ahoj_lbs}
IMAGE=${IMAGE:-$PROJECT/ifsitepred.sif}
CODE=${CODE:-$PROJECT/code/runners}            # ifs_batch_stage{1,2}.py rsynced here
RECEPTORS=${RECEPTORS:-$PROJECT/inputs/receptors.txt}
AFDB=${AFDB:-$PROJECT/inputs/afdb}             # AF-<UNP>-F1-model_v6.cif.gz
WORK=${WORK:-$PROJECT/work}                    # holds input/ predictions/ xyz/ (shared)
MODELS=${MODELS:-/opt/binding-sites/models}    # 40 LightGBMs, baked into the SIF
NSHARDS=${NSHARDS:?set NSHARDS in the environment (e.g. --export=ALL,NSHARDS=40)}

SHARD=${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORK/input" "$WORK/predictions" "$WORK/xyz" logs
echo "shard $SHARD/$NSHARDS on $(hostname) [$(nproc) cpus]"

# Activate envs by sourcing conda (NOT `mamba run` — it cannot create its lockfile in
# the read-only container). ACT is prepended to each in-container python invocation.
ACT=". /opt/conda/etc/profile.d/conda.sh &&"

# Stage 1 — ESM-IF1 residues on CPU (converts cif.gz -> chain-A pdb on the fly via --afdb_dir).
srun apptainer exec --bind "$PROJECT":"$PROJECT" "$IMAGE" bash -c \
    "$ACT conda activate esm_env && python '$CODE/ifs_batch_stage1.py' \
        --repo_dir '$WORK' --receptor_list '$RECEPTORS' \
        --input_dir '$WORK/input' --pred_dir '$WORK/predictions' \
        --models_dir '$MODELS' --afdb_dir '$AFDB' --device cpu --threads 8 \
        --shard $SHARD --nshards $NSHARDS"

# Stage 2 — PyMOL 1.5A grid + DBSCAN -> ranked 3D centres (CPU).
srun apptainer exec --bind "$PROJECT":"$PROJECT" "$IMAGE" bash -c \
    "$ACT conda activate pymol_env && python '$CODE/ifs_batch_stage2.py' \
        --repo_dir '$WORK' --receptor_list '$RECEPTORS' \
        --input_dir '$WORK/input' --pred_dir '$WORK/predictions' --xyz_dir '$WORK/xyz' \
        --shard $SHARD --nshards $NSHARDS"

echo "shard $SHARD done"
