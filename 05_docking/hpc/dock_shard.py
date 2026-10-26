# -*- coding: utf-8 -*-
"""
GPU docking of ONE shard of pockets with Uni-Dock (Aurum gen-d / L40S).

A *shard* is a deterministic slice of pockets (`--shard i --nshards N`). For each
pocket we run Uni-Dock ONCE per pocket-box, batching all of that pocket's ligands
(G1 rep + G2 AlphaFill + G3 probes) through `--gpu_batch`, so the receptor affinity
grid is built once and amortised over every ligand at that box.

Outputs are written to **node-local scratch** and packed into ONE bundle per shard
(`results_<shard>.parquet` + `poses_<shard>.sdf.gz`), then tarred — never one file
per job (protects the shared NFS inode budget). Resumable: a shard whose output
tarball already exists is skipped.

This is the HPC analogue of `05_docking/runner/run_docking.py`; it reuses that
package's PDBQT parsing / pose-in-pocket helpers. Engine is swappable
(`--engine unidock|vina`) so the calibration shard can cross-check Uni-Dock vs Vina.

NOTE: targets the cluster; validate with the calibration shard (see HPC_RUN_PLAN §6).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# reuse the pilot runner's parsing/geometry helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from prep_common import (  # noqa: E402
    parse_pocket_resi, parse_vina_pdbqt, count_pose_in_pocket,
    receptor_pocket_heavy_coords, safe_token,
)


def shard_pockets(manifest: pd.DataFrame, shard: int, nshards: int) -> List[str]:
    """Deterministic contiguous split of the unique pockets across shards."""
    pockets = sorted(manifest["site_uid"].astype(str).unique())
    return [p for k, p in enumerate(pockets) if k % nshards == shard]


def run_unidock(unidock: str, receptor: Path, box, ligands: List[Path], outdir: Path,
                num_modes: int, seed: int, search_mode: str, timeout: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [unidock, "--receptor", str(receptor),
           "--center_x", f"{box[0]}", "--center_y", f"{box[1]}", "--center_z", f"{box[2]}",
           "--size_x", f"{box[3]}", "--size_y", f"{box[4]}", "--size_z", f"{box[5]}",
           "--search_mode", search_mode, "--num_modes", str(num_modes), "--seed", str(seed),
           "--dir", str(outdir), "--gpu_batch", *[str(x) for x in ligands]]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def run_vina(vina: str, receptor: Path, box, ligand: Path, out_pdbqt: Path,
             num_modes: int, seed: int, exhaustiveness: int, timeout: int) -> None:
    cmd = [vina, "--receptor", str(receptor), "--ligand", str(ligand),
           "--center_x", f"{box[0]}", "--center_y", f"{box[1]}", "--center_z", f"{box[2]}",
           "--size_x", f"{box[3]}", "--size_y", f"{box[4]}", "--size_z", f"{box[5]}",
           "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
           "--seed", str(seed), "--out", str(out_pdbqt)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="parquet/csv with all jobs")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--receptors_pdbqt", type=Path, required=True)
    ap.add_argument("--receptors_pdb", type=Path, required=True, help="chain-A PDBs for pose_in_pocket")
    ap.add_argument("--ligands_pdbqt", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True, help="node-local $SCRATCH work area")
    ap.add_argument("--out", type=Path, required=True, help="shared dir for the packed bundle tarball")
    ap.add_argument("--engine", choices=["unidock", "vina"], default="unidock")
    ap.add_argument("--unidock", default="unidock")
    ap.add_argument("--vina", default="vina")
    ap.add_argument("--search_mode", default="balance", help="unidock: fast|balance|detail")
    ap.add_argument("--exhaustiveness", type=int, default=8, help="vina engine only")
    ap.add_argument("--num_modes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pocket_timeout", type=int, default=1800)
    ap.add_argument("--pocket_cutoff", type=float, default=4.0)
    ap.add_argument("--pocket_min_atoms", type=int, default=3)
    args = ap.parse_args()

    tag = f"{args.shard:05d}-of-{args.nshards:05d}"
    out_tar = args.out / f"shard_{tag}.tar.gz"
    if out_tar.exists():
        print(f"[SKIP] {out_tar} exists"); return

    m = pd.read_parquet(args.manifest) if str(args.manifest).endswith(".parquet") \
        else pd.read_csv(args.manifest, low_memory=False)
    my_pockets = set(shard_pockets(m, args.shard, args.nshards))
    sub = m[m["site_uid"].astype(str).isin(my_pockets)].copy()
    print(f"[SHARD {tag}] pockets={len(my_pockets)} jobs={len(sub)} engine={args.engine} host={os.uname().nodename if hasattr(os,'uname') else '?'}")

    work = args.scratch / f"shard_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    poses_path = work / f"poses_{tag}.sdf.gz"
    rows: List[dict] = []
    t0 = time.time()

    with gzip.open(poses_path, "wt") as poses_fh:
        for site_uid, g in sub.groupby("site_uid"):
            uni = str(g.iloc[0]["site_uniprot"])
            rec_pdbqt = args.receptors_pdbqt / f"{safe_token(uni)}.pdbqt"
            rec_pdb = args.receptors_pdb / f"{uni}.pdb"
            r0 = g.iloc[0]
            box = (r0.center_x, r0.center_y, r0.center_z, r0.size_x, r0.size_y, r0.size_z)
            pocket = parse_pocket_resi(r0.get("pocket_chain_resi", ""))
            pcoords = (receptor_pocket_heavy_coords(rec_pdb, pocket)
                       if pocket and rec_pdb.exists() else np.empty((0, 3)))

            # map ligand_id -> its prepared pdbqt; dedup so one grid docks all
            ligs = {str(row.ligand_id): args.ligands_pdbqt / f"{safe_token(str(row.ligand_id))}.pdbqt"
                    for _, row in g.iterrows()}
            ligs = {lid: p for lid, p in ligs.items() if p.exists()}
            outdir = work / safe_token(site_uid)

            try:
                if not rec_pdbqt.exists() or not ligs:
                    raise FileNotFoundError("missing receptor or all ligands")
                if args.engine == "unidock":
                    run_unidock(args.unidock, rec_pdbqt, box, list(ligs.values()), outdir,
                                args.num_modes, args.seed, args.search_mode, args.pocket_timeout)
                    out_files = {lid: outdir / f"{Path(p).stem}_out.pdbqt" for lid, p in ligs.items()}
                else:
                    outdir.mkdir(parents=True, exist_ok=True)
                    out_files = {}
                    for lid, p in ligs.items():
                        op = outdir / f"{safe_token(lid)}_out.pdbqt"
                        run_vina(args.vina, rec_pdbqt, box, p, op, args.num_modes, args.seed,
                                 args.exhaustiveness, args.pocket_timeout)
                        out_files[lid] = op
            except Exception as exc:
                for _, row in g.iterrows():
                    rows.append(_err_row(row, f"{type(exc).__name__}: {exc}", args))
                continue

            for _, row in g.iterrows():
                lid = str(row.ligand_id)
                op = out_files.get(lid)
                if op is None or not op.exists() or op.stat().st_size == 0:
                    rows.append(_err_row(row, "no engine output", args)); continue
                affs, models = parse_vina_pdbqt(op)
                if not affs:
                    rows.append(_err_row(row, "no modes parsed", args)); continue
                n_in = count_pose_in_pocket(models[0], pcoords, args.pocket_cutoff) if models and pcoords.size else 0
                rows.append({
                    "job_id": row.job_id, "site_uid": site_uid, "ligand_group": int(row.ligand_group),
                    "ligand_id": lid, "status": "ok", "error": "",
                    "affinity_top": float(affs[0]), "affinities": ";".join(f"{a:.3f}" for a in affs),
                    "n_modes": len(affs), "pose_in_pocket": bool(n_in >= args.pocket_min_atoms),
                    "n_pose_atoms_in_pocket": int(n_in),
                    "engine": args.engine, "search_mode": args.search_mode, "seed": args.seed,
                    "size_x": row.size_x, "size_y": row.size_y, "size_z": row.size_z,
                })
                # append the top pose (raw PDBQT block) into the concatenated bundle
                poses_fh.write(f">>> {row.job_id}\n")
                poses_fh.write(_first_model(op))

    res = pd.DataFrame(rows)
    res_path = work / f"results_{tag}.parquet"
    res.to_parquet(res_path, index=False)
    meta = {"shard": args.shard, "nshards": args.nshards, "pockets": len(my_pockets),
            "jobs": len(sub), "ok": int((res.status == "ok").sum()) if len(res) else 0,
            "wall_s": round(time.time() - t0, 1), "engine": args.engine}
    (work / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for f in (res_path, poses_path, work / f"meta_{tag}.json"):
            tar.add(f, arcname=f.name)
    print(f"[DONE {tag}] ok={meta['ok']}/{meta['jobs']} wall={meta['wall_s']}s -> {out_tar}")


def _err_row(row, msg, args) -> dict:
    return {"job_id": row.job_id, "site_uid": row.site_uid, "ligand_group": int(row.ligand_group),
            "ligand_id": str(row.ligand_id), "status": "error", "error": msg,
            "affinity_top": None, "affinities": "", "n_modes": 0, "pose_in_pocket": None,
            "n_pose_atoms_in_pocket": None, "engine": args.engine, "search_mode": args.search_mode,
            "seed": args.seed, "size_x": row.size_x, "size_y": row.size_y, "size_z": row.size_z}


def _first_model(pdbqt: Path) -> str:
    """Return the first MODEL block (top pose) as text for the concatenated bundle."""
    lines, inblk, started = [], False, False
    for ln in open(pdbqt):
        if ln.startswith("MODEL"):
            if started:
                break
            inblk, started = True, True
        lines.append(ln)
        if ln.startswith("ENDMDL"):
            break
    return "".join(lines) if lines else open(pdbqt).read()


if __name__ == "__main__":
    main()
