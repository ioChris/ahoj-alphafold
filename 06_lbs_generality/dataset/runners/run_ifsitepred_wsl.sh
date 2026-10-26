#!/usr/bin/env bash
# =============================================================================
# IF-SitePred batch orchestrator (WSL2 / Linux).
# -----------------------------------------------------------------------------
# 1) convert AFDB .cif.gz -> chain-A .pdb into <repo>/input/ (gemmi, one process)
# 2) Stage 1 (ESM-IF1 -> residues): NSHARDS parallel workers in the ifs_esm env
# 3) Stage 2 (DBSCAN -> centres):    NSHARDS parallel workers in the ifs_pymol env
# Idempotent at every step. The two batch workers (ifs_batch_stage{1,2}.py) are
# the same ones the Aurum GPU array job calls per shard.
#
# Usage: run_ifsitepred_wsl.sh <receptor_list.txt> <afdb_dir> <repo_dir> [nshards] [threads]
# =============================================================================
set -u
LIST="${1:?receptor_list}"; AFDB="${2:?afdb_dir}"; REPO="${3:?repo_dir}"
NSHARDS="${4:-4}"; THREADS="${5:-6}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MM="$HOME/bin/micromamba"
RUNNERS="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$(dirname "$REPO")/ifs_logs"          # persistent (ext4), NOT /tmp which WSL wipes
mkdir -p "$REPO/input" "$REPO/predictions" "$REPO/xyz" "$LOGDIR"
total=$(grep -cve '^[[:space:]]*$' "$LIST")
echo "[ifsitepred] $total receptors | nshards=$NSHARDS threads=$THREADS | repo=$REPO"

echo "[1/3 convert] cif.gz -> chain-A pdb ..."
"$MM" run -n lbs python - "$LIST" "$AFDB" "$REPO/input" <<'PY'
import sys, gemmi
from pathlib import Path
list_f, afdb, inp = sys.argv[1:4]; inp = Path(inp); afdb = Path(afdb)
n_ok = n_skip = n_miss = n_err = 0
for line in open(list_f):
    fn = line.strip()
    if not fn: continue
    t = fn[:-7] if fn.endswith(".cif.gz") else fn
    out = inp / f"{t}.pdb"
    if out.exists(): n_skip += 1; continue
    src = afdb / (t + ".cif.gz")
    if not src.exists(): n_miss += 1; continue
    try:
        st = gemmi.read_structure(str(src)); st.setup_entities()
        out.write_text(st.make_pdb_string()); n_ok += 1
    except Exception as e:
        n_err += 1; print("  [conv err]", t, e)
print(f"  converted={n_ok} skipped={n_skip} missing={n_miss} err={n_err}")
PY
echo "  input pdbs: $(ls "$REPO/input"/*.pdb 2>/dev/null | wc -l)"

run_shards(){ # $1=env $2=script $3=extra
  local env="$1" script="$2"; shift 2; local pids=()
  # Use `micromamba run` (full env activation) — direct env-python without activation
  # deadlocks torch's threading. Keep NSHARDS=1 to avoid the concurrent-run cache lock.
  for s in $(seq 0 $((NSHARDS-1))); do
    "$MM" run -n "$env" python "$RUNNERS/$script" --repo_dir "$REPO" --receptor_list "$LIST" \
      --shard "$s" --nshards "$NSHARDS" "$@" > "$LOGDIR/${script%.py}_$s.log" 2>&1 &
    pids+=($!)
    sleep 8
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; return $rc
}

echo "[2/3 stage1] ESM-IF1 residues ($NSHARDS x ${THREADS}thr) ..."
run_shards ifs_esm ifs_batch_stage1.py --threads "$THREADS"
echo "  residue files: $(ls "$REPO/predictions"/*/predicted_residues.json 2>/dev/null | wc -l)"

echo "[3/3 stage2] PyMOL/DBSCAN centres ($NSHARDS shards) ..."
run_shards ifs_pymol ifs_batch_stage2.py
echo "  centre files: $(ls "$REPO/predictions"/*/centre_rank_1.xyz 2>/dev/null | wc -l)"
echo "[ALL DONE]"
