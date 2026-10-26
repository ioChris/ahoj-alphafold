# -*- coding: utf-8 -*-
"""
Run the box-constrained Vina docking batch over the docking manifest (CPU).

For each manifest row (one pocket x ligand job):
  1. dock the prepared ligand PDBQT into the prepared receptor PDBQT using the
     manifest box (center/size), Vina (exhaustiveness/num_modes/seed);
  2. parse all mode affinities;
  3. compute `pose_in_pocket` (>=3 top-pose heavy atoms within 4 A of the manifest
     pocket residues) from the receptor PDB;
  4. export all poses to SDF (poses/<job_id>.sdf) for downstream RMSD/QC.

Writes one small JSON per job to `<workdir>/job_results/` (parallel-safe, fully
resumable), then aggregates into `<out>/docking_results.csv` +
`docking_results_manifest.json` per 05_docking/schemas/docking_results.schema.md.
Redock RMSD is computed separately (compute_redock_rmsd.py) once reference poses
exist, and merged at analysis time.

Usage:
    python run_docking.py --manifest <manifest.csv> --bundle <bundle> \
        --workdir <workdir> --vina <vina.exe> --out <results_dir> \
        [--exhaustiveness 8] [--workers 26] [--cpu 1] [--limit N] [--aggregate-only]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from prep_common import (
    export_pose_sdf, parse_pocket_resi, parse_vina_pdbqt, count_pose_in_pocket,
    receptor_pocket_heavy_coords, run_vina, safe_token,
)


def _meeko_script(name: str) -> str:
    import shutil
    exe = shutil.which(name)
    if exe:
        return exe
    cand = Path(os.sys.executable).parent / "Scripts" / f"{name}.exe"
    return str(cand) if cand.exists() else name


def _vina_version(vina_bin: str) -> str:
    try:
        out = subprocess.run([vina_bin, "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.stdout else "unknown"
    except Exception:
        return "unknown"


def dock_one(row: dict, paths: dict, params: dict) -> dict:
    jid = str(row["job_id"])
    sjid = safe_token(jid)
    res = {
        "job_id": jid, "site_uid": row["site_uid"], "ligand_group": int(row["ligand_group"]),
        "ligand_id": row["ligand_id"], "status": "error", "error": "",
        "vina_affinity_top": None, "vina_affinities": "", "n_modes": 0,
        "pose_in_pocket": "", "redock_rmsd_top": "", "redock_rmsd_best": "",
        "engine": "vina", "engine_version": params["vina_version"],
        "scoring_function": "vina", "seed": int(row.get("seed", params["seed"])),
        "exhaustiveness": params["exhaustiveness"],
        "size_x": row["size_x"], "size_y": row["size_y"], "size_z": row["size_z"],
        "runtime_s": None, "pose_path": "",
        "ligand_n_heavy_atoms": row.get("ligand_n_heavy_atoms", ""),
        "receptor_prep": params["receptor_prep"], "ligand_prep": params["ligand_prep"],
    }
    done_json = paths["job_results"] / f"{sjid}.json"
    if done_json.exists():
        try:
            return json.loads(done_json.read_text())
        except Exception:
            pass  # corrupt -> redo

    uni = str(row["site_uniprot"])
    rec_pdbqt = paths["receptors_pdbqt"] / f"{safe_token(uni)}.pdbqt"
    rec_pdb = paths["receptors_pdb"] / f"{uni}.pdb"
    lig_pdbqt = paths["ligands_pdbqt"] / f"{safe_token(str(row['ligand_id']))}.pdbqt"
    out_pdbqt = paths["vina_out"] / f"{sjid}.pdbqt"

    if not rec_pdbqt.exists():
        res["error"] = f"missing receptor pdbqt ({uni})"; _save(done_json, res); return res
    if not lig_pdbqt.exists():
        res["error"] = f"missing ligand pdbqt ({row['ligand_id']})"; _save(done_json, res); return res

    t0 = _now()
    ok, msg = run_vina(
        params["vina_bin"], rec_pdbqt, lig_pdbqt, out_pdbqt,
        (row["center_x"], row["center_y"], row["center_z"]),
        (row["size_x"], row["size_y"], row["size_z"]),
        params["exhaustiveness"], int(row.get("num_modes", 9)),
        int(row.get("seed", params["seed"])), cpu=params["cpu"], timeout=params["timeout"],
    )
    res["runtime_s"] = round(_now() - t0, 2)
    if not ok:
        res["error"] = msg; _save(done_json, res); return res

    affs, models = parse_vina_pdbqt(out_pdbqt)
    if not affs:
        res["error"] = "no modes parsed"; _save(done_json, res); return res
    res["vina_affinity_top"] = float(affs[0])
    res["vina_affinities"] = ";".join(f"{a:.3f}" for a in affs)
    res["n_modes"] = len(affs)

    # pose_in_pocket from top model
    try:
        pocket = parse_pocket_resi(row.get("pocket_chain_resi", ""))
        if pocket and rec_pdb.exists() and models:
            pcoords = receptor_pocket_heavy_coords(rec_pdb, pocket)
            n_in = count_pose_in_pocket(models[0], pcoords, cutoff=params["pocket_cutoff"])
            res["pose_in_pocket"] = bool(n_in >= params["pocket_min_atoms"])
            res["n_pose_atoms_in_pocket"] = int(n_in)
    except Exception as exc:
        res["error"] = f"pose_in_pocket: {type(exc).__name__}: {exc}"

    # export poses to SDF (all modes)
    if params["export_sdf"]:
        sdf = paths["poses"] / f"{sjid}.sdf"
        ok_e, _ = export_pose_sdf(out_pdbqt, sdf, params["mk_export"])
        if ok_e:
            res["pose_path"] = f"poses/{sjid}.sdf"

    res["status"] = "ok"
    if not params["keep_pdbqt"]:
        try:
            out_pdbqt.unlink()
        except Exception:
            pass
    _save(done_json, res)
    return res


def _now() -> float:
    import time
    return time.time()


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


RESULT_COLS = [
    "job_id", "site_uid", "ligand_group", "ligand_id", "status", "error",
    "vina_affinity_top", "vina_affinities", "n_modes",
    "pose_in_pocket", "n_pose_atoms_in_pocket", "redock_rmsd_top", "redock_rmsd_best",
    "engine", "engine_version", "scoring_function", "seed", "exhaustiveness",
    "size_x", "size_y", "size_z", "runtime_s", "pose_path",
    "ligand_n_heavy_atoms", "receptor_prep", "ligand_prep",
]


def aggregate(job_results_dir: Path, out_dir: Path, manifest: pd.DataFrame, params: dict) -> pd.DataFrame:
    rows = []
    for jf in job_results_dir.glob("*.json"):
        try:
            rows.append(json.loads(jf.read_text()))
        except Exception:
            continue
    res = pd.DataFrame(rows)
    # ensure every manifest job appears (missing -> error row)
    have = set(res["job_id"]) if len(res) else set()
    miss = manifest[~manifest["job_id"].isin(have)]
    if len(miss):
        extra = pd.DataFrame({
            "job_id": miss["job_id"], "site_uid": miss["site_uid"],
            "ligand_group": miss["ligand_group"], "ligand_id": miss["ligand_id"],
            "status": "error", "error": "not run",
        })
        res = pd.concat([res, extra], ignore_index=True)
    for c in RESULT_COLS:
        if c not in res.columns:
            res[c] = ""
    res = res[RESULT_COLS + [c for c in res.columns if c not in RESULT_COLS]]
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "docking_results.csv", index=False)

    nok = int((res["status"] == "ok").sum())
    manifest_sidecar = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "engine": "vina", "engine_version": params["vina_version"], "scoring_function": "vina",
        "exhaustiveness": params["exhaustiveness"], "num_modes_default": 9, "seed_default": params["seed"],
        "cpu_per_job": params["cpu"], "workers": params["workers"],
        "receptor_prep": params["receptor_prep"], "ligand_prep": params["ligand_prep"],
        "protonation_policy": "ligand: as-provided in RCSB ideal SDF (Meeko default typing); "
                              "receptor: Meeko mk_prepare_receptor -p (polar-H/typing)",
        "box_mode": "manifest (rg) verbatim; no re-centering/resizing",
        "pose_in_pocket_def": f">= {params['pocket_min_atoms']} heavy atoms within "
                              f"{params['pocket_cutoff']} A of manifest pocket residues",
        "hardware": f"CPU-only; {platform.processor() or platform.machine()}; "
                    f"{os.cpu_count()} logical cores; {platform.system()} {platform.release()}",
        "n_jobs": int(len(res)), "n_ok": nok, "n_error": int((res["status"] == "error").sum()),
        "deviation_from_manifest": "exhaustiveness overridden 16 -> 8 for the CPU pilot "
                                   "(Vina default; recorded per handover contract). Boxes unchanged.",
    }
    with open(out_dir / "docking_results_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest_sidecar, fh, indent=2)
    print(f"[AGG] {len(res)} jobs  ok={nok}  error={int((res['status']=='error').sum())} "
          f"-> {out_dir/'docking_results.csv'}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--vina", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--cpu", type=int, default=1, help="Vina --cpu per job.")
    ap.add_argument("--num_modes", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--pocket_cutoff", type=float, default=4.0)
    ap.add_argument("--pocket_min_atoms", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_export_sdf", action="store_true")
    ap.add_argument("--keep_pdbqt", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--prep_ligands_csv", type=Path, default=None,
                    help="prep_ligands.csv to backfill ligand_n_heavy_atoms.")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest, low_memory=False)
    if args.limit:
        m = m.head(args.limit)
    # backfill heavy-atom counts if the manifest lacks them
    if args.prep_ligands_csv and Path(args.prep_ligands_csv).exists():
        pl = pd.read_csv(args.prep_ligands_csv)[["ligand_id", "n_heavy_atoms"]]
        m = m.drop(columns=[c for c in ("n_heavy_atoms",) if c in m.columns])
        m = m.merge(pl, on="ligand_id", how="left")
        if "ligand_n_heavy_atoms" not in m.columns:
            m["ligand_n_heavy_atoms"] = pd.NA
        m["ligand_n_heavy_atoms"] = pd.to_numeric(m["ligand_n_heavy_atoms"], errors="coerce").fillna(m["n_heavy_atoms"])

    paths = {
        "receptors_pdbqt": args.workdir / "receptors_pdbqt",
        "receptors_pdb": args.workdir / "receptors_pdb",
        "ligands_pdbqt": args.workdir / "ligands_pdbqt",
        "vina_out": args.workdir / "vina_out",
        "poses": args.out / "poses",
        "job_results": args.workdir / "job_results",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    params = {
        "vina_bin": args.vina, "vina_version": _vina_version(args.vina),
        "exhaustiveness": args.exhaustiveness, "seed": args.seed, "cpu": args.cpu,
        "timeout": args.timeout, "workers": args.workers,
        "pocket_cutoff": args.pocket_cutoff, "pocket_min_atoms": args.pocket_min_atoms,
        "export_sdf": not args.no_export_sdf, "keep_pdbqt": args.keep_pdbqt,
        "mk_export": _meeko_script("mk_export"),
        "receptor_prep": "gemmi chainA -> Meeko rigid PDBQT (no pLDDT trim)",
        "ligand_prep": "Meeko from RCSB ideal SDF (as-provided protonation)",
    }

    if args.aggregate_only:
        aggregate(paths["job_results"], args.out, m, params)
        return

    # big-ligand-first ordering so the slow nucleotide jobs overlap the fast tail
    if "ligand_n_heavy_atoms" in m.columns:
        m["_h"] = pd.to_numeric(m["ligand_n_heavy_atoms"], errors="coerce").fillna(0)
        m = m.sort_values("_h", ascending=False)

    jobs = m.to_dict("records")
    n = len(jobs)
    done0 = len(list(paths["job_results"].glob("*.json")))
    print(f"[DOCK] {n} jobs  workers={args.workers}  cpu/job={args.cpu}  exh={args.exhaustiveness}  "
          f"vina={params['vina_version']}  already_done={done0}")

    ndone = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(dock_one, row, paths, params): row["job_id"] for row in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            ndone += 1
            if ndone % 25 == 0 or ndone == n:
                total = len(list(paths["job_results"].glob("*.json")))
                print(f"  {ndone}/{n}  (results on disk={total})  last={r['job_id']} "
                      f"status={r['status']} aff={r.get('vina_affinity_top')}")

    aggregate(paths["job_results"], args.out, m, params)


if __name__ == "__main__":
    main()
