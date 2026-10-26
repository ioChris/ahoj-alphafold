# -*- coding: utf-8 -*-
"""
Prepare all unique ligands (CCD SDF -> PDBQT via Meeko) and record heavy-atom counts.

Reads the ligand SDFs from the handover bundle (`ligands/ccd/*.sdf`), writes one
PDBQT per ligand into `<workdir>/ligands_pdbqt/`, and a status table
`<workdir>/prep_ligands.csv` (ligand_id, n_heavy_atoms, status, message). Parallel
and resumable (skips ligands whose PDBQT already exists).

Usage:
    python prep_ligands.py --bundle <bundle> --workdir <workdir> [--workers N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from prep_common import prepare_ligand, ligand_heavy_atoms, safe_token


def _meeko_script(name: str) -> str:
    # env Scripts dir on Windows; fall back to bare name (PATH).
    import shutil
    exe = shutil.which(name)
    if exe:
        return exe
    cand = Path(os.sys.executable).parent / "Scripts" / f"{name}.exe"
    return str(cand) if cand.exists() else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--mk_prepare_ligand", default=_meeko_script("mk_prepare_ligand"))
    args = ap.parse_args()

    lig_dir = args.bundle / "ligands" / "ccd"
    out_dir = args.workdir / "ligands_pdbqt"
    out_dir.mkdir(parents=True, exist_ok=True)

    sdfs = sorted(lig_dir.glob("*.sdf"))
    print(f"[LIGANDS] {len(sdfs)} SDFs  workers={args.workers}  mk={args.mk_prepare_ligand}")

    def work(sdf: Path) -> dict:
        lid = sdf.stem
        pdbqt = out_dir / f"{safe_token(lid)}.pdbqt"
        nheavy = ligand_heavy_atoms(sdf)
        if pdbqt.exists() and pdbqt.stat().st_size > 0:
            return {"ligand_id": lid, "n_heavy_atoms": nheavy, "status": "ok", "message": "cached"}
        ok, msg = prepare_ligand(sdf, pdbqt, args.mk_prepare_ligand)
        return {"ligand_id": lid, "n_heavy_atoms": nheavy,
                "status": "ok" if ok else "error", "message": msg}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s for s in sdfs}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 100 == 0 or i == len(sdfs):
                nok = sum(r["status"] == "ok" for r in rows)
                print(f"  {i}/{len(sdfs)}  ok={nok}")

    df = pd.DataFrame(rows).sort_values("ligand_id")
    df.to_csv(args.workdir / "prep_ligands.csv", index=False)
    nok = int((df["status"] == "ok").sum())
    nerr = int((df["status"] == "error").sum())
    print(f"[DONE] ligands ok={nok} error={nerr} -> {args.workdir/'prep_ligands.csv'}")
    if nerr:
        print(df[df["status"] == "error"][["ligand_id", "message"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
