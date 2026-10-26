# -*- coding: utf-8 -*-
"""
Compute run_3 redock RMSD (Analysis 6, G1 query-holo jobs).

For every G1 job with a docked pose:
  reference pose = experimental ligand superposed into the AF (v6) frame
                   (gemmi seq-align -> pocket-CA superposition on the escort receptor
                    chain; ligand mapped onto the CCD component template BY ATOM NAME);
  docked pose    = top Uni-Dock pose, reconstructed from the PDBQT `REMARK SMILES`
                   (meeko), mapped onto the same template by graph isomorphism;
  redock_rmsd    = symmetry-corrected, in-place RMSD over the observed atoms.

redock_success = redock_rmsd_top <= 2 A. Only the top pose is stored by the HPC run,
so redock_rmsd_best == redock_rmsd_top.

Parallelised by SOURCE PDB (each experimental structure is parsed once, then all of
its jobs are scored). Output: <out_dir>/redock_rmsd.csv (+ redock_rmsd_manifest.json).
Join back to the manifest/results by job_id.

Env: `ahoj-docking`. Run with KMP_DUPLICATE_LIB_OK=TRUE (redock_core sets it on import).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --- repo bootstrap ---------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import redock_core as rc  # noqa: E402

RUN3_DIR = cfg.DOCKING_RUN3_DIR

# per-worker caches (separate process each)
_TMPL_CACHE: Dict[str, Optional[rc.Template]] = {}
_AF_CACHE: Dict[str, object] = {}
_PATHS: Dict[str, Path] = {}


def _init(structures_dir: str, ccd_dir: str, afdb_dir: str):
    _PATHS["structures"] = Path(structures_dir)
    _PATHS["ccd"] = Path(ccd_dir)
    _PATHS["afdb"] = Path(afdb_dir)


def _template(ccd: str) -> Optional[rc.Template]:
    if ccd not in _TMPL_CACHE:
        cif = _PATHS["ccd"] / f"{ccd}.cif"
        _TMPL_CACHE[ccd] = rc.load_ccd_template(cif) if cif.exists() else None
    return _TMPL_CACHE[ccd]


def _af_model(afdb_filename: str, uniprot: str):
    key = afdb_filename or uniprot
    if key not in _AF_CACHE:
        p = _PATHS["afdb"] / afdb_filename
        if not p.exists():
            # fall back to highest-version model for this UniProt
            hits = sorted(_PATHS["afdb"].glob(f"AF-{uniprot}-F1-model_v*.cif.gz"))
            p = hits[-1] if hits else None
        _AF_CACHE[key] = rc.read_model(p) if p and p.exists() else None
    return _AF_CACHE[key]


def _blank_row(job: dict, status: str) -> dict:
    return {"job_id": job["job_id"], "site_uid": job.get("site_uid", ""),
            "ligand_id": job["ligand_id"], "src_pdb": job["src_pdb"], "status": status,
            "redock_rmsd_top": None, "redock_rmsd_best": None, "redock_success": None,
            "n_ref_atoms": None, "n_template_atoms": None, "ref_completeness": None,
            "superpose_rmsd": None, "superpose_mode": "", "n_pocket_pairs": None,
            "n_autos": None, "affinity_top": None}


def process_pdb(task) -> List[dict]:
    """Score every job that comes from one source PDB (parsed once)."""
    src_pdb, jobs, blocks = task
    out: List[dict] = []
    cif = _PATHS["structures"] / f"{src_pdb}.cif.gz"
    if not cif.exists():
        return [_blank_row(j, "no_pdb") for j in jobs]
    try:
        qmodel = rc.read_model(cif)
    except Exception as exc:  # noqa: BLE001
        return [_blank_row(j, f"pdb_read_fail:{type(exc).__name__}") for j in jobs]
    qchains = {c.name: c for c in qmodel}

    for job in jobs:
        row = _blank_row(job, "ok")
        try:
            block = blocks.get(job["job_id"])
            if not block:
                row["status"] = "no_pose"; out.append(row); continue
            row["affinity_top"] = rc.docked_affinity(block)

            tmpl = _template(str(job["ligand_id"]))
            if tmpl is None:
                row["status"] = "no_template"; out.append(row); continue
            row["n_template_atoms"] = tmpl.n

            af = _af_model(str(job.get("afdb_filename", "")), str(job["site_uniprot"]))
            if af is None:
                row["status"] = "no_af"; out.append(row); continue
            afc = next((c for c in af if c.name == "A"), None)
            pc = qchains.get(str(job["prot_chain"]))
            if afc is None or pc is None:
                row["status"] = "no_chain"; out.append(row); continue

            positions = json.loads(job["pocket_positions"]) if job.get("pocket_positions") else []
            T, srmsd, npair, mode = rc.pocket_superpose(afc, pc, positions)
            row.update(superpose_rmsd=round(srmsd, 3), superpose_mode=mode, n_pocket_pairs=npair)

            exp = rc.extract_ligand_named(qmodel, str(job["src_lig_chain"]),
                                          str(job["ligand_id"]), int(job["src_resnum"]))
            if not exp:
                row["status"] = "no_lig"; out.append(row); continue
            exp_af = [(nm, el, np.array([(v := T.apply(pos)).x, v.y, v.z])) for nm, el, pos in exp]
            ref_tmpl, obs = rc.exp_coords_template_order(exp_af, tmpl)
            row["n_ref_atoms"] = int(obs.sum())
            row["ref_completeness"] = round(float(obs.sum()) / tmpl.n, 3) if tmpl.n else None

            dmol = rc.docked_mol_from_block(block)
            if dmol is None:
                row["status"] = "docked_fail"; out.append(row); continue
            dock_tmpl = rc.map_docked_to_template(dmol, tmpl)
            if dock_tmpl is None:
                row["status"] = "map_fail"; out.append(row); continue

            rmsd, nobs = rc.symmetry_rmsd_observed(dock_tmpl, ref_tmpl, obs, tmpl.autos)
            row["n_autos"] = len(tmpl.autos)
            if rmsd is None:
                row["status"] = "rmsd_fail"; out.append(row); continue
            row.update(redock_rmsd_top=round(rmsd, 3), redock_rmsd_best=round(rmsd, 3),
                       redock_success=bool(rmsd <= 2.0), status="ok")
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"error:{type(exc).__name__}"
        out.append(row)
    return out


def load_pose_index(out_dir: Path, shards: List[Path]) -> Dict[str, str]:
    """job_id -> raw PDBQT block, from all shard tarballs."""
    idx: Dict[str, str] = {}
    for sh in shards:
        with tarfile.open(sh, "r:gz") as tar:
            mem = next((m for m in tar.getmembers() if m.name.startswith("poses_")), None)
            if mem is None:
                continue
            text = gzip.decompress(tar.extractfile(mem).read()).decode("utf-8", "ignore")
        for jid, block in rc.iter_pose_blocks(text):
            idx[jid] = block
    return idx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute run_3 redock RMSD (G1).")
    p.add_argument("--escort_csv", type=Path, default=RUN3_DIR / "reference_poses" / "escort_g1.csv")
    p.add_argument("--shards_dir", type=Path, default=RUN3_DIR / "out")
    p.add_argument("--cache_dir", type=Path, default=cfg.DOCKING_REDOCK_CACHE_DIR)
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    p.add_argument("--out_dir", type=Path, default=RUN3_DIR / "reference_poses")
    p.add_argument("--out_name", default="redock_rmsd.csv")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--limit_pdbs", type=int, default=0, help="score only the first N source PDBs (subset)")
    p.add_argument("--job_ids_file", type=Path, default=None, help="restrict to job_ids in this file (one per line)")
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(args.escort_csv, low_memory=False)
    e = e[e["has_pose"]] if "has_pose" in e.columns else e
    if args.job_ids_file and args.job_ids_file.exists():
        keep = {ln.strip() for ln in args.job_ids_file.read_text().splitlines() if ln.strip()}
        e = e[e["job_id"].isin(keep)]
    print(f"[JOBS] {len(e)} with-pose G1 jobs across {e['src_pdb'].nunique()} source PDBs")

    shards = sorted(args.shards_dir.glob("shard_*.tar.gz"))
    print(f"[POSES] indexing {len(shards)} shards ...")
    t0 = time.time()
    pose_idx = load_pose_index(args.out_dir, shards)
    print(f"[POSES] {len(pose_idx)} pose blocks  ({time.time()-t0:.0f}s)")

    jobs_by_pdb: Dict[str, List[dict]] = defaultdict(list)
    for r in e.to_dict("records"):
        jobs_by_pdb[str(r["src_pdb"])].append(r)
    pdbs = sorted(jobs_by_pdb)
    if args.limit_pdbs:
        pdbs = pdbs[:args.limit_pdbs]
    tasks = [(pdb, jobs_by_pdb[pdb], {j["job_id"]: pose_idx[j["job_id"]]
              for j in jobs_by_pdb[pdb] if j["job_id"] in pose_idx}) for pdb in pdbs]
    print(f"[RUN] {len(tasks)} PDB tasks  workers={args.workers}")

    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.cache_dir / "structures"),
                                       str(args.cache_dir / "ccd"), str(args.afdb_dir))) as ex:
        futs = [ex.submit(process_pdb, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 1000 == 0 or i == len(tasks):
                nok = sum(1 for r in rows if r["status"] == "ok")
                print(f"  {i}/{len(tasks)} pdbs  rows={len(rows)}  ok={nok}  ({time.time()-t0:.0f}s)")

    # `rmsd_status` (not `status`) is what analyze_docking_pilot.py --redock_csv expects.
    df = pd.DataFrame(rows).rename(columns={"status": "rmsd_status"})
    out_csv = args.out_dir / args.out_name
    df.to_csv(out_csv, index=False)

    ok = df[df["rmsd_status"] == "ok"]
    succ = float((pd.to_numeric(ok["redock_rmsd_top"], errors="coerce") <= 2.0).mean()) if len(ok) else None
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "escort_csv": str(args.escort_csv), "n_jobs": int(len(df)),
        "status_counts": df["rmsd_status"].value_counts().to_dict(),
        "n_scored": int(len(ok)),
        "median_rmsd": float(pd.to_numeric(ok["redock_rmsd_top"], errors="coerce").median()) if len(ok) else None,
        "success_rate_2A": succ,
        "median_ref_completeness": float(pd.to_numeric(ok["ref_completeness"], errors="coerce").median()) if len(ok) else None,
        "median_superpose_rmsd": float(pd.to_numeric(ok["superpose_rmsd"], errors="coerce").median()) if len(ok) else None,
    }
    with open(args.out_dir / "redock_rmsd_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[SAVE] {out_csv}")
    print(f"[DONE] scored={meta['n_scored']}/{meta['n_jobs']}  median RMSD={meta['median_rmsd']}  "
          f"success<=2A={None if succ is None else round(succ,3)}")
    print(f"       status: {meta['status_counts']}")


if __name__ == "__main__":
    main()
