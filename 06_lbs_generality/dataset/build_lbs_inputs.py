# -*- coding: utf-8 -*-
"""
Build the predictor input lists for the LBS-generality stage.
=============================================================

From the ST_100 geometry table, emit:
  * `receptors.csv` — one row per unique AFDB model to run each predictor on
    (site_uid is many-to-one onto a UniProt model, so a receptor hosts several
    pockets). Columns: afdb_filename, site_uniprot, model_path, n_sites, site_uids.
  * `receptor_site_map.csv` — long (afdb_filename, site_uid) so adapters can
    broadcast each predicted pocket to every site hosted on that receptor.

Optionally restrict to a stratified pilot via `--pilot_site_uids` (a CSV with a
`site_uid` column; reuse `05_docking/dataset/sample_pilot_pockets.py` to draw one
across size × class × separation — the SAME stratification as the docking pilot).

Base env (pandas only). Paths from common/config.py.

    python build_lbs_inputs.py                                  # full ST_100
    python build_lbs_inputs.py --pilot_site_uids <pilot.csv>    # pilot subset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build receptor / site lists for the LBS-generality predictors.")
    p.add_argument("--geometry_csv", type=Path, default=cfg.AFDB_ST100_DATASET_GEOM_CSV)
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--pilot_site_uids", type=Path, default=None,
                   help="Optional CSV with a site_uid column to restrict to a pilot subset.")
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or cfg.LBS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.geometry_csv, low_memory=False)
    df = df[df["threshold"] == args.threshold].drop_duplicates("site_uid").copy()
    df["site_uid"] = df["site_uid"].astype(str).str.strip()

    if args.pilot_site_uids is not None:
        pilot = pd.read_csv(args.pilot_site_uids)
        keep = set(pilot["site_uid"].astype(str).str.strip())
        df = df[df["site_uid"].isin(keep)].copy()
        print(f"[PILOT] restricted to {len(df)} pockets from {args.pilot_site_uids}")

    site_map = df[["afdb_filename", "site_uid", "site_uniprot"]].copy()
    site_map.to_csv(out_dir / "receptor_site_map.csv", index=False)

    receptors = (
        site_map.groupby("afdb_filename")
        .agg(site_uniprot=("site_uniprot", "first"),
             n_sites=("site_uid", "size"),
             site_uids=("site_uid", lambda s: " ".join(sorted(s))))
        .reset_index()
    )
    receptors["model_path"] = receptors["afdb_filename"].map(lambda f: str(args.afdb_dir / f))
    receptors["model_exists"] = receptors["model_path"].map(lambda p: Path(p).is_file())
    receptors.to_csv(out_dir / "receptors.csv", index=False)

    print(f"[SAVE] {out_dir / 'receptors.csv'}  ({len(receptors)} receptors, "
          f"{int(receptors['model_exists'].sum())} present on disk)")
    print(f"[SAVE] {out_dir / 'receptor_site_map.csv'}  ({len(site_map)} pockets)")


if __name__ == "__main__":
    main()
