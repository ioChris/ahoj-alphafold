# -*- coding: utf-8 -*-
"""
Robustness check (CROSS-CUTTING — defends the apo/holo classification, not just redock):
are APO experimental structures less completely OBSERVED than HOLO ones at the binding
pocket (and at the chain), which would make the pocket-level apo/holo comparison unfair?

Motivation: whole-chain apo/holo-vs-AlphaFold comparisons (e.g. Saldano et al. 2022) can be
biased because ligand binding tends to order flexible regions, so holo depositions are often
more completely observed and align to more of the full-length model. AHoJ classifies at the
POCKET level, so the relevant test is whether the *pocket* residues are equally observed in
the apo vs holo experimental structures that feed the classification.

Reads AHoJ per-query outputs (`output_fix/<batch>/<job>.tgz` -> apo/holo_filtered_sorted_results.csv),
which record per matched structure: `apoholo_assignment` (A/H), `%unobserved_rsd` (unobserved
pocket residues), `UNPcov_obs` (observed chain coverage), `resolution`. Compares apo vs holo
paired within each query (controls for protein identity).

Result (random sample, seed 0): pocket residues are ~fully observed in BOTH classes
(median %unobserved 0% vs 0%), chain coverage ~equal (~99%), with only a minor holo
resolution edge — i.e. the pocket-level classification is NOT confounded by differential
observation completeness.

By default this samples over *all* AHoJ queries; pass `--survivors` to restrict to the analysis-set
survivors (`cfg.FILTERED_SURVIVORS_CSV`, key `job_name`) for an analysis-set-matched N — that run writes
separate `*_survivors.json` / `_survivors_sample.csv` so the default result is not overwritten.

Env: `ahoj-docking` (pandas). Run with PYTHONIOENCODING=utf-8 on Windows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import io
import json
import os
import random
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

_METRICS = ["%unobserved_rsd", "UNPcov_obs", "resolution", "%mapped_obs", "pocket_len"]


def _read(t, suffix):
    m = next((x for x in t.getnames() if x.endswith(suffix)), None)
    if not m:
        return None
    try:
        return pd.read_csv(io.StringIO(t.extractfile(m).read().decode("utf-8", "ignore")))
    except Exception:
        return None


def _summ(df):
    if df is None or len(df) == 0:
        return {"n": 0}
    out = {"n": len(df)}
    for c in _METRICS:
        if c in df.columns:
            out[c] = float(pd.to_numeric(df[c], errors="coerce").mean())
    return out


def worker(path):
    try:
        with tarfile.open(path, "r:gz") as t:
            apo = _read(t, "apo_filtered_sorted_results.csv")
            holo = _read(t, "holo_filtered_sorted_results.csv")
    except Exception:
        return None
    a, h = _summ(apo), _summ(holo)
    return {"query": os.path.basename(path)[:-4], "n_apo": a["n"], "n_holo": h["n"],
            "apo_unobs": a.get("%unobserved_rsd"), "holo_unobs": h.get("%unobserved_rsd"),
            "apo_unpcov": a.get("UNPcov_obs"), "holo_unpcov": h.get("UNPcov_obs"),
            "apo_res": a.get("resolution"), "holo_res": h.get("resolution"),
            "apo_plen": a.get("pocket_len"), "holo_plen": h.get("pocket_len")}


def parse_args():
    p = argparse.ArgumentParser(description="apo vs holo pocket/chain observation completeness (AHoJ output_fix).")
    p.add_argument("--outfix", type=Path, default=cfg.AHOJ_OUTPUT_FIX_DIR)
    p.add_argument("--n_sample", type=int, default=12000, help="random query .tgz to sample (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out_dir", type=Path, default=cfg.DOCKING_RUN3_DIR / "reference_poses")
    p.add_argument("--survivors", action="store_true",
                   help="restrict to the analysis-set survivors (cfg.FILTERED_SURVIVORS_CSV; key job_name "
                        "== the <job>.tgz basename) for an analysis-set-matched N; writes separate *_survivors "
                        "outputs so the default random-draw result is not overwritten")
    p.add_argument("--survivors_csv", type=Path, default=None,
                   help="explicit filtered_survivors.csv path (implies --survivors)")
    return p.parse_args()


def _med(s):
    return float(pd.to_numeric(s, errors="coerce").median())


def main():
    args = parse_args()
    all_paths = []
    for b in sorted(os.listdir(args.outfix)):
        bd = args.outfix / b
        if bd.is_dir():
            all_paths += glob.glob(str(bd / "*.tgz"))

    # Optional: restrict to the analysis-set survivors (job_name == <job>.tgz basename) so the
    # completeness check is matched to the exact pockets that enter the manuscript, rather than a
    # random draw over all AHoJ queries. Default off -> behaviour/outputs unchanged.
    surv_csv = args.survivors_csv or (cfg.FILTERED_SURVIVORS_CSV if args.survivors else None)
    survivor_set = None
    if surv_csv is not None:
        survivor_set = set(pd.read_csv(surv_csv, usecols=["job_name"])["job_name"].astype(str))
        n_before = len(all_paths)
        all_paths = [p for p in all_paths if os.path.basename(p)[:-4] in survivor_set]
        print(f"[SURVIVORS] {surv_csv.name}: {len(all_paths)} of {n_before} archives match "
              f"{len(survivor_set)} survivor job_names")
        if not all_paths:
            raise SystemExit("[SURVIVORS] no archives matched — check that job_name matches the .tgz basename")

    if args.n_sample and args.n_sample < len(all_paths):
        random.seed(args.seed)
        paths = random.sample(all_paths, args.n_sample)
    else:
        paths = all_paths
    print(f"[SAMPLE] {len(paths)} of {len(all_paths)} query archives (seed {args.seed})")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, p) for p in paths]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                rows.append(r)
            if i % 3000 == 0:
                print(f"  {i}/{len(paths)}")
    df = pd.DataFrame(rows)
    both = df[(df.n_apo > 0) & (df.n_holo > 0)].copy()

    def frac_apo_worse():
        a = pd.to_numeric(both.apo_unobs, errors="coerce"); h = pd.to_numeric(both.holo_unobs, errors="coerce")
        m = a.notna() & h.notna()
        return float((a[m] > h[m]).mean()) if m.any() else float("nan")

    summary = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "survivor_filter": (str(surv_csv) if surv_csv is not None else None),
        "n_survivor_job_names": (len(survivor_set) if survivor_set is not None else None),
        "n_sample": len(paths), "n_queries": int(len(df)), "n_paired": int(len(both)),
        "n_holo_only": int(((df.n_apo == 0) & (df.n_holo > 0)).sum()),
        "n_apo_only": int(((df.n_apo > 0) & (df.n_holo == 0)).sum()),
        "paired_pocket_pct_unobserved": {"apo": _med(both.apo_unobs), "holo": _med(both.holo_unobs),
                                         "frac_queries_apo_worse": frac_apo_worse()},
        "paired_chain_coverage_obs": {"apo": _med(both.apo_unpcov), "holo": _med(both.holo_unpcov)},
        "paired_resolution": {"apo": _med(both.apo_res), "holo": _med(both.holo_res)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_survivors" if survivor_set is not None else ""
    df.to_csv(args.out_dir / f"apo_holo_completeness{suffix}_sample.csv", index=False)
    (args.out_dir / f"apo_holo_completeness{suffix}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nqueries={len(df)}  paired(both apo&holo)={len(both)}  holo-only={summary['n_holo_only']}  apo-only={summary['n_apo_only']}")
    print("PAIRED within-query (median of per-query means):")
    print(f"  %unobserved pocket resid  apo {_med(both.apo_unobs):.2f}%  holo {_med(both.holo_unobs):.2f}%  (apo worse in {frac_apo_worse()*100:.0f}% of queries)")
    print(f"  chain coverage (observed) apo {_med(both.apo_unpcov):.1f}%  holo {_med(both.holo_unpcov):.1f}%")
    print(f"  resolution (A)            apo {_med(both.apo_res):.2f}   holo {_med(both.holo_res):.2f}")
    print(f"[SAVE] {args.out_dir / f'apo_holo_completeness{suffix}.json'}")


if __name__ == "__main__":
    main()
