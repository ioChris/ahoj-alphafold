#!/bin/bash
# Build the tiny fpocket image (CPU, no GPU needed).
#SBATCH --job-name=fp-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/fpbuild-%j.out
#SBATCH --error=logs/fpbuild-%j.err
set -euo pipefail
cd "$HOME/ahoj_lbs/code/hpc"
apptainer build --fakeroot --mksquashfs-args "-processors 1" "$HOME/ahoj_lbs/fpocket.sif" fpocket.def
apptainer exec "$HOME/ahoj_lbs/fpocket.sif" bash -c 'fpocket -ver 2>&1 | head -1; python -c "import pandas,gemmi,numpy; print(\"deps ok\")"' || echo "WARN test failed"
echo "fpocket.sif: $(ls -lh $HOME/ahoj_lbs/fpocket.sif)"
