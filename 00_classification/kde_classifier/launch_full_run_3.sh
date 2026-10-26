#!/usr/bin/env bash
#
# launch_full_run_3.sh - reproduce (or resume) the full_run_3 apo/holo
# reclassification on this machine, via the resumable batch launcher.
#
#   ./launch_full_run_3.sh [config.json]
#
# Defaults to the machine-local full-run config. Safe to re-run: the launcher
# resumes only unfinished batches (a .batch_complete sentinel marks done ones),
# so a kill + re-invoke picks up where it left off. See docs/RUN_launcher.md.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
CFG="${1:-$REPO/.local_tests/run_config_full_run_3.local.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/log/full_run_3_${STAMP}.log"
mkdir -p "$REPO/log"

{
  echo "=================================================================="
  echo "full_run_3 launch  |  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host        : $(hostname)"
  echo "git commit  : $(git rev-parse HEAD 2>/dev/null || echo '?')"
  echo "interpreter : $PY  ($("$PY" --version 2>&1))"
  echo "config      : $CFG"
  echo "log         : $LOG"
  echo "=================================================================="
  echo
  echo ">>> init  (write batches.tsv if absent)"
  "$PY" run_full_batch_launcher.py init --config "$CFG"
  echo
  echo ">>> run   (resumable; processes pending, non-skipped batches)"
  "$PY" run_full_batch_launcher.py run --config "$CFG"
  echo
  echo ">>> status"
  "$PY" run_full_batch_launcher.py status --config "$CFG" || true
  echo
  echo ">>> loop returned at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "    when all machines are done, stitch summaries with:"
  echo "      $PY run_full_batch_launcher.py concat --config $CFG"
} 2>&1 | tee -a "$LOG"
