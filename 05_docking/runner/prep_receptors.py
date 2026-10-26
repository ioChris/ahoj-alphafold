# -*- coding: utf-8 -*-
"""
Prepare all receptors (AFDB `.cif.gz` -> rigid PDBQT) for docking.

For each AF model in the bundle's `receptors/` dir: gemmi extracts chain-A
amino-acid residues to a clean PDB, then Meeko builds a rigid receptor PDBQT.
Writes `<workdir>/receptors_pdbqt/<uniprot>.pdbqt` (+ the intermediate PDB in
`<workdir>/receptors_pdb/`) and a status table `<workdir>/prep_receptors.csv`.
Parallel and resumable.

Usage:
    python prep_receptors.py --bundle <bundle> --workdir <workdir> [--workers N]
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from prep_common import extract_receptor_pdb, prepare_receptor, safe_token


def _meeko_script(name: str) -> str:
    import shutil
    exe = shutil.which(name)
    if exe:
        return exe
    cand = Path(os.sys.executable).parent / "Scripts" / f"{name}.exe"
    return str(cand) if cand.exists() else name


def uniprot_from_filename(fn: str) -> str:
    m = re.match(r"AF-([^-]+)-F1-model_v", fn)
    return m.group(1) if m else Path(fn).stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--mk_prepare_receptor", default=_meeko_script("mk_prepare_receptor"))
    args = ap.parse_args()

    rec_dir = args.bundle / "receptors"
    pdb_dir = args.workdir / "receptors_pdb"
    pdbqt_dir = args.workdir / "receptors_pdbqt"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    cifs = sorted(rec_dir.glob("AF-*-F1-model_v*.cif.gz"))
    print(f"[RECEPTORS] {len(cifs)} models  workers={args.workers}")

    def work(cif: Path) -> dict:
        uni = uniprot_from_filename(cif.name)
        pdb = pdb_dir / f"{uni}.pdb"
        pdbqt = pdbqt_dir / f"{safe_token(uni)}.pdbqt"
        if pdbqt.exists() and pdbqt.stat().st_size > 0:
            return {"site_uniprot": uni, "afdb_filename": cif.name, "n_res": None,
                    "status": "ok", "message": "cached"}
        try:
            nres = extract_receptor_pdb(cif, pdb)
        except Exception as exc:
            return {"site_uniprot": uni, "afdb_filename": cif.name, "n_res": None,
                    "status": "error", "message": f"gemmi: {type(exc).__name__}: {exc}"}
        ok, msg = prepare_receptor(pdb, pdbqt, args.mk_prepare_receptor)
        return {"site_uniprot": uni, "afdb_filename": cif.name, "n_res": nres,
                "status": "ok" if ok else "error", "message": msg}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c for c in cifs}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(cifs):
                nok = sum(r["status"] == "ok" for r in rows)
                print(f"  {i}/{len(cifs)}  ok={nok}")

    df = pd.DataFrame(rows).sort_values("site_uniprot")
    df.to_csv(args.workdir / "prep_receptors.csv", index=False)
    nok = int((df["status"] == "ok").sum())
    nerr = int((df["status"] == "error").sum())
    print(f"[DONE] receptors ok={nok} error={nerr} -> {args.workdir/'prep_receptors.csv'}")
    if nerr:
        print(df[df["status"] == "error"][["site_uniprot", "message"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
