# -*- coding: utf-8 -*-
"""
Generic adapter: any per-receptor pocket table -> normalized predictions.csv.
=============================================================================

Most predictors emit, per receptor (AFDB model), a table of ranked pockets with
a centre (and often lining residues). This helper broadcasts those pockets onto
every ST_100 `site_uid` hosted on that receptor (via `receptor_site_map.csv` from
build_lbs_inputs.py) and writes the normalized `predictions.csv`
(schema: ../../schemas/predictions.schema.md) that `score/compute_lbs_success.py`
consumes.

Input: a "raw pockets" CSV with, at minimum, columns for the receptor key
(afdb_filename OR site_uniprot), a rank, and centre x/y/z. Column names are
mapped via CLI flags, so a per-tool adapter is usually a one-liner around this.

    python normalize_predictions.py \
        --raw_pockets fpocket_raw.csv --receptor_site_map <...>/receptor_site_map.csv \
        --receptor_col afdb_filename --rank_col pocket_rank \
        --x_col center_x --y_col center_y --z_col center_z \
        --score_col score --residues_col pocket_residues \
        --tool fpocket --out_csv <...>/fpocket/predictions.csv

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


def normalize(raw: pd.DataFrame, receptor_site_map: pd.DataFrame, *, receptor_col: str,
              rank_col: str, x_col: str, y_col: str, z_col: str,
              score_col: str | None, residues_col: str | None, tool: str) -> pd.DataFrame:
    r = raw.copy()
    r[receptor_col] = r[receptor_col].astype(str).str.strip()
    rsm = receptor_site_map.copy()
    join_key = "afdb_filename" if receptor_col == "afdb_filename" else "site_uniprot"
    rsm[join_key] = rsm[join_key].astype(str).str.strip()

    keep = {receptor_col: join_key, rank_col: "pocket_rank",
            x_col: "center_x", y_col: "center_y", z_col: "center_z"}
    if score_col and score_col in r.columns:
        keep[score_col] = "score"
    if residues_col and residues_col in r.columns:
        keep[residues_col] = "pocket_residues"
    r = r[list(keep)].rename(columns=keep)

    # broadcast each receptor's pockets to every site_uid on that receptor
    out = rsm[[join_key, "site_uid"]].merge(r, on=join_key, how="inner")
    out["tool"] = tool
    cols = ["site_uid", "pocket_rank", "center_x", "center_y", "center_z"]
    for extra in ("score", "pocket_residues", "tool"):
        if extra in out.columns:
            cols.append(extra)
    return out[cols]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize a per-receptor pocket table into predictions.csv.")
    p.add_argument("--raw_pockets", type=Path, required=True)
    p.add_argument("--receptor_site_map", type=Path, default=cfg.LBS_DIR / "receptor_site_map.csv")
    p.add_argument("--receptor_col", default="afdb_filename", choices=["afdb_filename", "site_uniprot"])
    p.add_argument("--rank_col", default="pocket_rank")
    p.add_argument("--x_col", default="center_x")
    p.add_argument("--y_col", default="center_y")
    p.add_argument("--z_col", default="center_z")
    p.add_argument("--score_col", default="score")
    p.add_argument("--residues_col", default="pocket_residues")
    p.add_argument("--tool", required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.raw_pockets, low_memory=False)
    rsm = pd.read_csv(args.receptor_site_map, low_memory=False)
    out = normalize(raw, rsm, receptor_col=args.receptor_col, rank_col=args.rank_col,
                    x_col=args.x_col, y_col=args.y_col, z_col=args.z_col,
                    score_col=args.score_col, residues_col=args.residues_col, tool=args.tool)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[SAVE] {args.out_csv}  ({len(out)} pocket rows over {out['site_uid'].nunique()} sites)")


if __name__ == "__main__":
    main()
