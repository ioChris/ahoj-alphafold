#!/usr/bin/env bash
# =============================================================================
# fpocket batch runner (WSL2 / Linux) for the LBS-generality stage.
# -----------------------------------------------------------------------------
# fpocket has no Windows build, so we run it under WSL2 (Ubuntu) from a
# micromamba conda-forge env, then PARSE the outputs on Windows with
# `run_fpocket.py`. fpocket 4 reads mmCIF directly, so we only gunzip the
# AFDB `.cif.gz` (no pdb conversion). Outputs go to a NON-OneDrive scratch so
# thousands of small files never sync to the cloud.
#
# Usage:
#   run_fpocket_wsl.sh <receptor_list.txt> <afdb_dir> <out_root> [env_name] [limit]
#     receptor_list.txt : one afdb_filename per line (AF-<UNP>-F1-model_v6.cif.gz)
#     afdb_dir          : dir holding the .cif.gz models (WSL path, e.g. /mnt/c/...)
#     out_root          : where to write per-receptor <stem>_out/ dirs
#     env_name          : micromamba env with fpocket (default: lbs)
#     limit             : optional cap on number processed (smoke test)
#
# Idempotent: skips a receptor whose <stem>_out/ already exists. Safe to re-run.
# =============================================================================
set -u

LIST="${1:?receptor_list.txt}"; AFDB="${2:?afdb_dir}"; OUT="${3:?out_root}"
ENVN="${4:-lbs}"; LIMIT="${5:-0}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
MM="$HOME/bin/micromamba run -n $ENVN"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

total=$(grep -cve '^[[:space:]]*$' "$LIST")
i=0; ok=0; skip=0; fail=0
echo "[fpocket] $total receptors -> $OUT (env=$ENVN)"
while IFS= read -r fn; do
  fn="${fn%$'\r'}"                      # strip CRLF from Windows-written lists
  [ -z "$fn" ] && continue
  i=$((i+1))
  [ "$LIMIT" -gt 0 ] && [ "$i" -gt "$LIMIT" ] && { i=$((i-1)); break; }
  stem="${fn%.cif.gz}"
  if [ -d "$OUT/${stem}_out" ]; then skip=$((skip+1)); continue; fi
  src="$AFDB/$fn"
  if [ ! -f "$src" ]; then echo "[MISS] $fn"; fail=$((fail+1)); continue; fi
  cp "$src" "$TMP/" && ( cd "$TMP" && gunzip -f "$fn" )
  if $MM fpocket -f "$TMP/${stem}.cif" >/dev/null 2>&1 && [ -d "$TMP/${stem}_out" ]; then
    rm -rf "$OUT/${stem}_out"; mv "$TMP/${stem}_out" "$OUT/"; ok=$((ok+1))
  else
    fail=$((fail+1)); echo "[FAIL] $fn"
  fi
  rm -f "$TMP/${stem}.cif" "$TMP/${fn}"
  if [ $((i % 100)) -eq 0 ]; then echo "  ... $i/$total (ok=$ok skip=$skip fail=$fail)"; fi
done < "$LIST"
echo "[DONE] processed=$i ok=$ok skipped=$skip failed=$fail  out=$OUT"
