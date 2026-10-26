# -*- coding: utf-8 -*-
"""
Draw a stratified PILOT sample of ST_100 pockets for the first docking run.

Strata (crossed), per DOCKING_DESIGN.md §10:
  - pocket size       : union_rg, binned into tertiles (small/medium/large)
  - apo/holo class    : AF-Apo vs AF-Holo (from pocket_class)
  - apo/holo separation: pocket_separation_pair, binned low/mid/high (tertiles)

Allocation is *balanced* across cells (equal target per cell, capped by
availability, remainder redistributed) so small but hypothesis-critical cells
(e.g. AF-Apo x high-separation) are populated -- not proportional, which would
starve them.

Only pockets with a usable center and (by default) an available GROUP-1 query
ligand are eligible, so every sampled pocket is dockable for the main group.

Output: a CSV with a `site_uid` column (+ strata) suitable for
`build_docking_manifest.py --pilot_site_uids`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
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

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # allow sibling imports
from parse_ahoj_job_ligands import build_query_ligand_table, is_dockable_ccd


DEFAULT_INPUT = cfg.AFDB_ST100_DATASET_GEOM_CSV
DEFAULT_OUTPUT_DIR = cfg.DOCKING_PILOT_DIR


def _class_bin(v: object) -> str:
    s = str(v).lower()
    if "holo" in s:
        return "AF-Holo"
    if "apo" in s:
        return "AF-Apo"
    return "other"


def _qbin(series: pd.Series, n: int, labels: list[str]) -> pd.Series:
    """Quantile-bin; fall back to rank-based cut if too many ties for qcut."""
    s = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(s, q=n, labels=labels, duplicates="drop")
    except Exception:
        ranks = s.rank(method="first")
        return pd.cut(ranks, bins=n, labels=labels)


def eligible_pockets(df: pd.DataFrame, require_g1: bool) -> pd.DataFrame:
    df = df.copy()
    df = df[df["union_chain_resi"].fillna("").astype(str).str.strip().ne("")]
    df["union_rg"] = pd.to_numeric(df.get("union_rg"), errors="coerce")
    df = df[df["union_rg"].notna()]
    df["class_bin"] = df["pocket_class"].map(_class_bin)
    df = df[df["class_bin"].isin(["AF-Apo", "AF-Holo"])]
    df["pocket_separation_pair"] = pd.to_numeric(df.get("pocket_separation_pair"), errors="coerce")
    df = df[df["pocket_separation_pair"].notna()]
    if require_g1:
        # eligible = has at least one DOCKABLE query ligand (any member job), so
        # pockets whose representative ligand is an ion but that have a dockable
        # member still qualify (pair with build_docking_manifest --g1_all_ligands).
        g1 = build_query_ligand_table(df, include_members=True)
        g1 = g1[g1["ligand_id"].map(is_dockable_ccd)]
        have = set(g1["site_uid"].astype(str))
        df = df[df["site_uid"].astype(str).isin(have)]
    return df


def stratified_sample(df: pd.DataFrame, target_n: int, n_size_bins: int, n_sep_bins: int,
                      seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    size_labels = {3: ["small", "medium", "large"], 4: ["q1", "q2", "q3", "q4"]}.get(
        n_size_bins, [f"s{i+1}" for i in range(n_size_bins)])
    sep_labels = {3: ["low", "mid", "high"]}.get(n_sep_bins, [f"sep{i+1}" for i in range(n_sep_bins)])

    df = df.copy()
    df["size_bin"] = _qbin(df["union_rg"], n_size_bins, size_labels)
    df["sep_bin"] = _qbin(df["pocket_separation_pair"], n_sep_bins, sep_labels)
    df = df[df["size_bin"].notna() & df["sep_bin"].notna()]

    cells = list(df.groupby(["size_bin", "class_bin", "sep_bin"], observed=True))
    n_cells = len(cells)
    if n_cells == 0:
        return df.head(0)

    base = target_n // n_cells
    picks = []
    deficit = 0
    surplus_cells = []
    for key, cell in cells:
        take = min(base, len(cell))
        if take < base:
            deficit += (base - take)
        else:
            surplus_cells.append((key, cell, take))
        if take > 0:
            picks.append(cell.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    # redistribute the deficit across cells that still have spare rows
    already = pd.concat(picks) if picks else df.head(0)
    already_idx = set(already.index)
    if deficit > 0:
        spare = df[~df.index.isin(already_idx)]
        spare_cells = [c for _, c in spare.groupby(["size_bin", "class_bin", "sep_bin"], observed=True)]
        # round-robin one at a time so redistribution stays balanced
        pools = [c.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))) for c in spare_cells if len(c)]
        added = []
        i = 0
        while deficit > 0 and pools:
            pool = pools[i % len(pools)]
            if len(pool):
                added.append(pool.iloc[[0]])
                pools[i % len(pools)] = pool.iloc[1:]
                deficit -= 1
            if all(len(p) == 0 for p in pools):
                break
            i += 1
        if added:
            already = pd.concat([already] + added)

    keep_cols = ["site_uid", "site_uniprot", "size_bin", "class_bin", "sep_bin",
                 "union_rg", "pocket_separation_pair", "pocket_class",
                 "pocket_p_holo", "model_pocket_plddt"]
    keep_cols = [c for c in keep_cols if c in already.columns]
    return already[keep_cols].sort_values(["size_bin", "class_bin", "sep_bin"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified pilot sampler for docking.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target_n", type=int, default=1500)
    p.add_argument("--n_size_bins", type=int, default=3)
    p.add_argument("--n_sep_bins", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_require_g1", action="store_true",
                   help="Do not require a group-1 query ligand (sample dockability-only pockets too).")
    p.add_argument("--outfile_name", default="pilot_pockets.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    elig = eligible_pockets(df, require_g1=not args.no_require_g1)
    sample = stratified_sample(elig, args.target_n, args.n_size_bins, args.n_sep_bins, args.seed)

    out_csv = args.output_dir / args.outfile_name
    sample.to_csv(out_csv, index=False)

    cell_counts = (sample.groupby(["size_bin", "class_bin", "sep_bin"], observed=True)
                   .size().rename("n").reset_index())
    cell_counts.to_csv(args.output_dir / f"{Path(args.outfile_name).stem}_cell_counts.csv", index=False)

    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_csv": str(args.input_csv),
        "output_csv": str(out_csv),
        "target_n": args.target_n,
        "sampled_n": int(len(sample)),
        "eligible_n": int(len(elig)),
        "require_g1": (not args.no_require_g1),
        "n_size_bins": args.n_size_bins,
        "n_sep_bins": args.n_sep_bins,
        "seed": args.seed,
        "class_balance": sample["class_bin"].value_counts().to_dict() if len(sample) else {},
    }
    with open(args.output_dir / f"{Path(args.outfile_name).stem}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVE] {out_csv}")
    print(f"[SAMPLED] {len(sample)} / target {args.target_n}  (eligible {len(elig)})")
    print(f"[CLASS ] {meta['class_balance']}")
    print("[CELLS ] wrote cell_counts.csv")


if __name__ == "__main__":
    main()
