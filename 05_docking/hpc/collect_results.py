# -*- coding: utf-8 -*-
"""
Collect per-shard docking bundles into one results table (+ optional concatenated poses).

Reads `shard_*.tar.gz` produced by dock_shard.py from `--in`, concatenates every
`results_*.parquet`, and writes one `docking_results.parquet` (+ CSV) plus a coverage
report (shards found, jobs ok/error, missing shards). Pose bundles are left tarred;
pass --extract_poses to also concatenate the `poses_*.sdf.gz` blocks.

Usage:
    python collect_results.py --in $PROJECT/out \
        --out docking_results.parquet --nshards 800
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nshards", type=int, default=0, help="expected shard count (coverage check)")
    ap.add_argument("--extract_poses", action="store_true")
    args = ap.parse_args()

    tars = sorted(args.indir.glob("shard_*.tar.gz"))
    frames, seen = [], set()
    poses_out = args.out.with_name("poses_all.sdf.gz") if args.extract_poses else None
    pf = __import__("gzip").open(poses_out, "wt") if poses_out else None
    for t in tars:
        with tarfile.open(t, "r:gz") as tar:
            for mem in tar.getmembers():
                if mem.name.startswith("results_") and mem.name.endswith(".parquet"):
                    frames.append(pd.read_parquet(io.BytesIO(tar.extractfile(mem).read())))
                    seen.add(t.name)
                if pf and mem.name.startswith("poses_") and mem.name.endswith(".sdf.gz"):
                    pf.write(__import__("gzip").decompress(tar.extractfile(mem).read()).decode())
    if pf:
        pf.close()

    if not frames:
        raise SystemExit(f"no results parquet found in {args.indir}")
    res = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(args.out, index=False)
    res.to_csv(args.out.with_suffix(".csv"), index=False)

    nok = int((res.status == "ok").sum())
    print(f"[COLLECT] shards={len(seen)} jobs={len(res)} ok={nok} error={len(res)-nok}")
    print(f"[SAVE] {args.out}")
    if args.nshards:
        missing = [i for i in range(args.nshards)
                   if not (args.indir / f"shard_{i:05d}-of-{args.nshards:05d}.tar.gz").exists()]
        if missing:
            print(f"[WARN] {len(missing)} shards missing (resubmit): {missing[:20]}{'...' if len(missing) > 20 else ''}")
        else:
            print("[OK] all shards present")


if __name__ == "__main__":
    main()
