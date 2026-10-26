# -*- coding: utf-8 -*-
"""
Ligand heavy-atom counts from the docked top poses of a Uni-Dock run.

`analyze_docking_pilot.py --ligand_heavy_csv` takes a `ligand_id,n_heavy_atoms,source` table as the
ligand-size covariate. For run_3 it was derived from the docked poses (source = `pdbqt_pose`): each
shard bundle `out/shard_*.tar.gz` holds `poses_*.sdf.gz`, a text file of `>>> {job_id}` blocks, each
the top-pose PDBQT. The count is the number of ATOM/HETATM records whose AutoDock atom type is not a
hydrogen (H, HD, HS). One row per CCD code, sorted; a code seen with more than one count (it should
not happen: the prepared ligand is the same file for every job) keeps the most frequent one.

The July table had no committed generator; `--check_against` reports whether this script reproduces it.

    python ligand_heavy_atoms_from_poses.py --shards_dir <run>/out --out_csv <run>/analysis/ccd_heavy_atoms.csv
"""
from __future__ import annotations

import argparse
import collections
import gzip
import tarfile
from pathlib import Path

import pandas as pd

HYDROGEN_TYPES = {"H", "HD", "HS"}


def counts_from_shard(tar_path: Path, acc: dict) -> None:
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.name.startswith("poses_"):
                continue
            text = gzip.decompress(tar.extractfile(member).read()).decode("utf-8", "replace")
            ligand, n = None, 0
            for line in text.splitlines():
                if line.startswith(">>> "):
                    if ligand is not None and n:
                        acc[ligand][n] += 1
                    ligand, n = line[4:].strip().rsplit("__", 1)[-1], 0
                elif line.startswith(("ATOM", "HETATM")):
                    if line.split()[-1] not in HYDROGEN_TYPES:
                        n += 1
                elif line.startswith("ENDMDL"):
                    if ligand is not None and n:
                        acc[ligand][n] += 1
                    ligand, n = None, 0
            if ligand is not None and n:
                acc[ligand][n] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--check_against", type=Path, default=None)
    a = ap.parse_args()

    acc: dict = collections.defaultdict(collections.Counter)
    shards = sorted(a.shards_dir.glob("shard_*.tar.gz"))
    for s in shards:
        counts_from_shard(s, acc)
    rows = [(lig, c.most_common(1)[0][0], "pdbqt_pose") for lig, c in sorted(acc.items())]
    out = pd.DataFrame(rows, columns=["ligand_id", "n_heavy_atoms", "source"])
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    amb = sum(1 for c in acc.values() if len(c) > 1)
    print(f"[SAVE] {a.out_csv}  ({len(out)} ligands from {len(shards)} shards; ligands with more than one count: {amb})")
    if a.check_against is not None:
        ref = pd.read_csv(a.check_against, dtype={"ligand_id": str}, keep_default_na=False)
        m = out.astype({"ligand_id": str}).merge(ref, on="ligand_id", how="outer", suffixes=("", "_ref"), indicator=True)
        both = m[m._merge == "both"]
        print(f"[CHECK] ligands here {len(out)}, reference {len(ref)}, in both {len(both)}, "
              f"same count {int((both.n_heavy_atoms == both.n_heavy_atoms_ref).sum())}, "
              f"only here {int((m._merge == 'left_only').sum())}, only in reference {int((m._merge == 'right_only').sum())}")


if __name__ == "__main__":
    main()
