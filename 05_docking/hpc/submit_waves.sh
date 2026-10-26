#!/bin/bash
# Controlled wave submission of the docking array on Aurum.
#
# Aurum etiquette: <=~2000 array elements/user and <=~50 concurrent. We therefore
# submit the NSHARDS tasks in waves, each an array with a %concurrency cap, chaining
# waves with --dependency so total queue pressure stays bounded. Re-running skips
# shards whose output tarball already exists (dock_shard.py is resumable), so a
# failed/interrupted wave is safe to resubmit.
#
# Usage:  NSHARDS=800 WAVE=400 CONC=40 ./submit_waves.sh
#   NSHARDS  total shards (pick so each ~= 3-4 h; calibrate first, see HPC_RUN_PLAN §6)
#   WAVE     array elements per wave (<= 2000)
#   CONC     max concurrent tasks per wave (<= 50)
set -euo pipefail

NSHARDS=${NSHARDS:?}
WAVE=${WAVE:-400}
CONC=${CONC:-40}
export NSHARDS

mkdir -p logs
prev=""
start=0
while [ "$start" -lt "$NSHARDS" ]; do
    end=$(( start + WAVE - 1 ))
    [ "$end" -ge "$NSHARDS" ] && end=$(( NSHARDS - 1 ))
    dep=()
    [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
    jid=$(sbatch --parsable "${dep[@]}" \
                 --array="${start}-${end}%${CONC}" \
                 dock_array.sbatch)
    echo "submitted wave ${start}-${end} (%${CONC}) as job $jid ${dep[*]:-}"
    prev="$jid"
    start=$(( end + 1 ))
done
echo "all waves queued; final job id $prev"
echo "monitor:  squeue -u \$USER ;  ls \$PROJECT/out | wc -l"
