#!/bin/bash
# Build the IF-SitePred apptainer image as a batch job on a gen-d node.
# (Login node blocks compilers; gen-d matches the run architecture; net via login NAT.)
# Submit: sbatch build.sbatch  ->  produces $HOME/ahoj_lbs/ifsitepred.sif
#
#SBATCH --job-name=ifs-build
#SBATCH --constraint=gen-d&scratch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err

set -euo pipefail
cd "$HOME/ahoj_lbs/code/hpc"
echo "build on $(hostname) starting $(date)"
apptainer build --fakeroot --mksquashfs-args "-processors 1" \
    "$HOME/ahoj_lbs/ifsitepred.sif" ifsitepred.def
echo "== post-build import verification (non-fatal; CUDA deferred to a GPU node) =="
SIF="$HOME/ahoj_lbs/ifsitepred.sif"
apptainer exec "$SIF" bash -c '. /opt/conda/etc/profile.d/conda.sh && conda activate esm_env && python -c "import esm,torch,torch_geometric,biotite,gemmi,lightgbm; from biotite.structure import filter_backbone; print(\"esm_env imports OK; torch\", torch.__version__)"' || echo "WARN: esm_env import test failed"
apptainer exec "$SIF" bash -c '. /opt/conda/etc/profile.d/conda.sh && conda activate pymol_env && python -c "from pymol import cmd; import pandas, sklearn; print(\"pymol_env imports OK\")"' || echo "WARN: pymol_env import test failed"
echo "build done $(date); image: $(ls -lh "$SIF")"
