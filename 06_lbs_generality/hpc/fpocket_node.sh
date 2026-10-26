#!/usr/bin/env bash
# Runs INSIDE the fpocket container on one fat node: parallel fpocket over all
# receptors (to node-local scratch), then parse -> raw-pockets CSV.
# Args: <afdb_dir> <receptor_list> <out_dir> <code_dir> <raw_out_csv> [nparallel]
set -u
AFDB="$1"; RECEPTORS="$2"; OUT="$3"; CODE="$4"; RAW_OUT="$5"; NPAR="${6:-60}"
mkdir -p "$OUT"

run_one() {
  fn="$1"; t="${fn%.cif.gz}"
  [ -d "$OUT/${t}_out" ] && return 0
  tmp=$(mktemp -d)
  if gunzip -c "$AFDB/$fn" > "$tmp/$t.cif" 2>/dev/null; then
    fpocket -f "$tmp/$t.cif" >/dev/null 2>&1 && mv "$tmp/${t}_out" "$OUT/" 2>/dev/null
  fi
  rm -rf "$tmp"
}
export -f run_one
export AFDB OUT

tr -d '\r' < "$RECEPTORS" | grep -v '^[[:space:]]*$' | xargs -P "$NPAR" -I{} bash -c 'run_one "$@"' _ {}
echo "fpocket _out dirs: $(ls -d "$OUT"/*_out 2>/dev/null | wc -l)"

# receptors.csv (afdb_filename column) for the parser
{ echo afdb_filename; tr -d '\r' < "$RECEPTORS" | grep -v '^[[:space:]]*$'; } > "$OUT/../receptors.csv"
python "$CODE/run_fpocket.py" --fpocket_out_root "$OUT" \
    --receptors_csv "$OUT/../receptors.csv" --out_csv "$RAW_OUT"
