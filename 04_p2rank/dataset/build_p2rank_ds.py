# -*- coding: utf-8 -*-
"""
Write the P2Rank dataset pair for an exported ST_100 subset.

`export_p2rank_subsets.py` writes `st_100_p2rank_<kind>_geom.csv`. P2Rank's
`eval-predict` with explicit sites needs two more files beside it:

  * `<stem>_INPUT.csv` -- the explicit-sites table (format `ahoj_ubs`): columns
    site_uniprots, site_uid, site_recipe, threshold, afdb_filename, chain_resi,
    center_x, center_y, center_z, one row per site, in the export's row order, CRLF;
  * `<stem>.ds` -- the dataset file: a blank line, the two PARAM lines naming the
    explicit-sites format and file, a blank line, then the absolute path of every
    model (unique `afdb_filename`, in order of first appearance), CRLF line ends.

The July 2026 run_3 files were written by hand; this script reproduces them byte
for byte (`--check_against <dir>` compares) so that a re-run on another model
directory differs only in the model paths and the geometry.

    python build_p2rank_ds.py --subset_csv <...>/st_100_p2rank_intersection_geom.csv --afdb_dir <models>
"""

from __future__ import annotations

import argparse
import filecmp
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


INPUT_COLUMNS = ["site_uniprots", "site_uid", "site_recipe", "threshold", "afdb_filename",
                 "chain_resi", "center_x", "center_y", "center_z"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write the P2Rank .ds and explicit-sites INPUT.csv for an exported subset.")
    p.add_argument("--subset_csv", type=Path, required=True,
                   help="st_100_p2rank_<kind>_geom.csv written by export_p2rank_subsets.py")
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    p.add_argument("--out_dir", type=Path, default=None, help="Default: beside --subset_csv.")
    p.add_argument("--check_against", type=Path, default=None,
                   help="Directory holding a reference .ds / _INPUT.csv of the same stem; report byte identity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.subset_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.subset_csv.stem

    # Everything is kept as text so coordinates are passed through as exported.
    df = pd.read_csv(args.subset_csv, dtype=str, keep_default_na=False, low_memory=False)
    df = df.rename(columns={"site_uniprot": "site_uniprots"})
    input_csv = out_dir / f"{stem}_INPUT.csv"
    df[INPUT_COLUMNS].to_csv(input_csv, index=False, lineterminator="\r\n")

    models = list(dict.fromkeys(df["afdb_filename"]))
    missing = [m for m in models if not (args.afdb_dir / m).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} models not found under {args.afdb_dir}, e.g. {missing[:3]}")
    ds = out_dir / f"{stem}.ds"
    lines = ["", "PARAM.EXPLICIT_SITES_FORMAT=ahoj_ubs", f"PARAM.EXPLICIT_SITES_FILE={input_csv.name}", ""]
    lines += [str(args.afdb_dir / m) for m in models]
    ds.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))

    print(f"[SAVE] {input_csv}  ({len(df)} sites)")
    print(f"[SAVE] {ds}  ({len(models)} models)")

    if args.check_against is not None:
        for f in (input_csv, ds):
            same = filecmp.cmp(f, args.check_against / f.name, shallow=False)
            print(f"[CHECK] {f.name}: {'byte-identical' if same else 'DIFFERS'} vs {args.check_against}")


if __name__ == "__main__":
    main()
