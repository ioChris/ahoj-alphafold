# -*- coding: utf-8 -*-
"""
Collect P2Rank's own predicted pockets into a raw-pockets table for re-scoring.
===============================================================================

To put P2Rank on the SAME harmonized heavy-atom yardstick as the other LBS-
generality tools, we score P2Rank's *own* predicted pocket centres with
`score/compute_lbs_success.py` instead of using its native (SAS-inflated)
`dca4rank`. This helper concatenates the per-receptor P2Rank prediction CSVs
(`<eval_dir>/predictions/<afdb_filename>_predictions.csv`) into one raw-pockets
table keyed by `afdb_filename`, ready for `normalize_predictions.py`.

P2Rank prediction columns (note leading spaces, stripped here):
  name, rank, score, probability, sas_points, surf_atoms, center_x/y/z, residue_ids, ...

    python collect_p2rank_raw_pockets.py \
        --eval_dir <.../UBS2_run3_dev_clean/eval_predict_st_100_..._CENTER_explicit> \
        --receptor_site_map <LBS_DIR>/receptor_site_map.csv \
        --out_csv <LBS_DIR>/p2rank/raw_pockets_run3.csv

Base env (pandas only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate P2Rank per-receptor prediction CSVs into a raw-pockets table.")
    p.add_argument("--eval_dir", type=Path, required=True,
                   help="P2Rank eval dir containing a predictions/ subfolder.")
    p.add_argument("--receptor_site_map", type=Path, default=cfg.LBS_DIR / "receptor_site_map.csv")
    p.add_argument("--out_csv", type=Path, default=cfg.LBS_DIR / "p2rank" / "raw_pockets_run3.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = args.eval_dir / "predictions"
    rsm = pd.read_csv(args.receptor_site_map, low_memory=False)
    receptors = sorted(rsm["afdb_filename"].dropna().astype(str).str.strip().unique())
    print(f"[INFO] {len(receptors)} receptors from {args.receptor_site_map.name}")

    frames = []
    n_ok = n_missing = n_empty = 0
    for i, fn in enumerate(receptors, 1):
        pf = pred_dir / f"{fn}_predictions.csv"
        if not pf.is_file():
            n_missing += 1
            continue
        try:
            d = pd.read_csv(pf)
        except Exception:
            n_empty += 1
            continue
        d.columns = [c.strip() for c in d.columns]
        if d.empty or "center_x" not in d.columns:
            n_empty += 1
            continue
        d = d[["rank", "score", "probability", "center_x", "center_y", "center_z"]].copy()
        d["afdb_filename"] = fn
        frames.append(d)
        n_ok += 1
        if i % 2000 == 0:
            print(f"  ... {i}/{len(receptors)} receptors read")

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["rank", "score", "probability", "center_x", "center_y", "center_z", "afdb_filename"])
    raw = raw.rename(columns={"rank": "pocket_rank", "probability": "score_prob"})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    print(f"[SAVE] {args.out_csv}  ({len(raw)} pocket rows over {raw['afdb_filename'].nunique()} receptors)")
    print(f"[STATS] receptors_with_predictions={n_ok}  missing_pred_file={n_missing}  empty/unreadable={n_empty}")


if __name__ == "__main__":
    main()
