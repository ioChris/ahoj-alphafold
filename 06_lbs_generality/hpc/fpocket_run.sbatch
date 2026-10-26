#!/bin/bash
# fpocket over the full ST_100 receptor set on ONE fat CPU node (parallel), then parse.
# Submit: sbatch --export=ALL,RECEPTORS=$HOME/ahoj_lbs/inputs/receptors_full.txt fpocket_run.sbatch
#
#SBATCH --job-name=ahoj-fpocket
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fpocket-%j.out
#SBATCH --error=logs/fpocket-%j.err

set -euo pipefail
PROJECT=${PROJECT:-$HOME/ahoj_lbs}
IMAGE=${IMAGE:-$PROJECT/fpocket.sif}
AFDB=${AFDB:-$PROJECT/inputs/afdb}
RECEPTORS=${RECEPTORS:-$PROJECT/inputs/receptors_full.txt}
CODE=${CODE:-$PROJECT/code/runners}
# RAW_OUT: override for a re-run on another model set, so the earlier CSV is kept
# (AFDB v4 re-alignment: RAW_OUT=$PROJECT/fpocket_raw_full_afdbv4.csv, AFDB=$PROJECT/inputs/afdb_v4).
RAW_OUT=${RAW_OUT:-$PROJECT/fpocket_raw_full.csv}
SCR=${SCRATCH:-/scratch/slurm_jobs/$USER/$SLURM_JOB_ID}
OUT="$SCR/fpocket_out"
mkdir -p "$OUT" logs
echo "fpocket on $(hostname) [$(nproc) cpus], $(grep -c . "$RECEPTORS") receptors"

srun apptainer exec --bind "$PROJECT":"$PROJECT" --bind "$SCR":"$SCR" "$IMAGE" \
    bash "$PROJECT/code/hpc/fpocket_node.sh" \
        "$AFDB" "$RECEPTORS" "$OUT" "$CODE" "$RAW_OUT" 60

echo "fpocket done -> $RAW_OUT"
