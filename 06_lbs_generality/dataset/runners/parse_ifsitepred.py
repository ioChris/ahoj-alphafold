# -*- coding: utf-8 -*-
"""
IF-SitePred adapter (recommended predictor #2) -> raw pockets table.
====================================================================

IF-SitePred (Carbery et al., J. Cheminform. 2024; github.com/annacarbery/
binding-sites, BSD-3) predicts binding sites from ESM-IF1 inverse-folding
embeddings: 40 LightGBM classifiers (unanimous vote) label binding residues,
a pseudo-atom point cloud around them is DBSCAN-clustered, and up to 3 sites are
emitted RANKED BY CLUSTER SIZE. Per target it writes, under
`predictions/<target>/`:
  centre_rank_{0,1,2}.xyz  -- the site CENTRE (a single 'PS x y z' point)
  site_rank_{0,1,2}.xyz    -- all cluster points ('PS x y z' per line); the point
                              count is the cluster size, used here as the score.

This parser turns a tree of `predictions/<target>/` dirs into the per-receptor
"raw pockets" table (afdb_filename, pocket_rank, center_x/y/z, score,
n_site_points) that `normalize_predictions.py --tool ifsitepred` then broadcasts
to each ST_100 site_uid, and `score/compute_lbs_success.py` scores on the SAME
harmonized heavy-atom DCA/TNS metric as P2Rank and fpocket.

Target naming convention (set when preparing inputs): <target> == the AFDB model
filename stem, e.g. AF-<UNP>-F1-model_v6  ->  afdb_filename AF-<UNP>-F1-model_v6.cif.gz.

Ranking note: IF-SitePred ranks implicitly by file index (rank_0 = largest
cluster = best). pocket_rank = index+1; score = number of cluster points
(a continuous proxy). Base env (pandas only).

    python parse_ifsitepred.py --predictions_root <.../predictions> \
        --out_csv <LBS_DIR>/ifsitepred/ifsitepred_raw.csv
"""

from __future__ import annotations

import argparse
import re
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


def _read_xyz_points(path: Path) -> list[tuple[float, float, float]]:
    """Return [(x,y,z), ...] from an .xyz file; robust to a 2-line header.

    Accepts any line whose last three whitespace tokens parse as floats
    (covers 'PS x y z' point-cloud lines and standard xyz 'ELEM x y z')."""
    pts: list[tuple[float, float, float]] = []
    if not path.is_file():
        return pts
    for line in path.read_text(errors="ignore").splitlines():
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            x, y, z = float(toks[-3]), float(toks[-2]), float(toks[-1])
        except ValueError:
            continue
        pts.append((x, y, z))
    return pts


def _centroid(pts: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    if not pts:
        return None
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


def parse_one_target(target_dir: Path, afdb_filename: str) -> pd.DataFrame:
    """Parse one predictions/<target>/ dir into ranked raw-pocket rows."""
    rows = []
    ranks = sorted(int(m.group(1))
                   for f in target_dir.glob("centre_rank_*.xyz")
                   for m in [re.search(r"centre_rank_(\d+)\.xyz", f.name)] if m)
    for r in ranks:
        centre_pts = _read_xyz_points(target_dir / f"centre_rank_{r}.xyz")
        centre = _centroid(centre_pts)          # centre file is one point; centroid is a no-op safety net
        if centre is None:
            continue
        site_pts = _read_xyz_points(target_dir / f"site_rank_{r}.xyz")
        rows.append({
            "afdb_filename": afdb_filename,
            "pocket_rank": r,                   # IF-SitePred files are 1-indexed: rank_1 = best (largest cluster)
            "center_x": centre[0], "center_y": centre[1], "center_z": centre[2],
            "score": len(site_pts) if site_pts else float("nan"),   # cluster size = confidence proxy
            "n_site_points": len(site_pts),
        })
    return pd.DataFrame(rows)


def target_to_afdb_filename(target: str) -> str:
    """<target> stem -> afdb_filename. AF stems already carry model_vN; add .cif.gz."""
    return target if target.endswith(".cif.gz") else f"{target}.cif.gz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse IF-SitePred predictions/<target>/ dirs into a raw pockets table.")
    p.add_argument("--predictions_root", type=Path, required=True,
                   help="Directory containing per-target subdirs with centre_rank_*.xyz.")
    p.add_argument("--out_csv", type=Path, default=cfg.LBS_DIR / "ifsitepred" / "ifsitepred_raw.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    n_targets = n_empty = 0
    for target_dir in sorted(p for p in args.predictions_root.iterdir() if p.is_dir()):
        n_targets += 1
        df = parse_one_target(target_dir, target_to_afdb_filename(target_dir.name))
        if df.empty:
            n_empty += 1
            continue
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["afdb_filename", "pocket_rank", "center_x", "center_y", "center_z", "score", "n_site_points"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    print(f"[SAVE] {args.out_csv}  ({len(raw)} pockets over {raw['afdb_filename'].nunique()} receptors)")
    print(f"[STATS] targets={n_targets}  targets_with_no_site={n_empty}")
    print("[NEXT] normalize_predictions.py --raw_pockets "
          f"{args.out_csv} --tool ifsitepred --out_csv {args.out_csv.parent/'predictions.csv'}")


if __name__ == "__main__":
    main()
