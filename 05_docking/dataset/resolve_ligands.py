# -*- coding: utf-8 -*-
"""
Resolve docking ligands to 3D files for the handover bundle.

- Groups 1 & 2 (CCD codes): fetch the RCSB *ideal* SDF
      https://files.rcsb.org/ligands/download/{CCD}_ideal.sdf
  and save to  <out_dir>/ccd/{CCD}.sdf .
- Group 3 (probes): embed each probe SMILES to 3D (RDKit ETKDG + MMFF) and save
  to  <out_dir>/probes/{probe_id}.sdf .

`out_dir` defaults to `<manifest_dir>/ligands`, so the paths line up with the
`ligand_sdf` column the manifest writes (`ligands/ccd/...`, `ligands/probes/...`).

RDKit is optional: without it, CCD SDFs are still fetched as-is (no SMILES /
heavy-atom count computed) and probe embedding is skipped with a warning.
Everything here runs on YOUR machine (network + RDKit); it is not executed in
the assistant session.

A resolution report (`ligand_resolution.csv`) records success/failure and
heavy-atom counts, which can be merged back to fill `ligand_n_heavy_atoms`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

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


DEFAULT_LIGANDS_CSV = cfg.DOCKING_MANIFEST_DIR / "ligands.csv"
DEFAULT_OUT_DIR = cfg.DOCKING_MANIFEST_DIR / "ligands"
DEFAULT_PROBE_CSV = cfg.DOCKING_PROBE_LIBRARY_CSV
RCSB_IDEAL_SDF = "https://files.rcsb.org/ligands/download/{ccd}_ideal.sdf"

# A small, size-varied default probe set (edit freely). Spans ~6 -> ~20 heavy atoms.
DEFAULT_PROBES = [
    ("benzene", "c1ccccc1"),
    ("imidazole", "c1c[nH]cn1"),
    ("phenol", "Oc1ccccc1"),
    ("benzamidine", "NC(=N)c1ccccc1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]


def _try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except Exception:
        return None, None


def write_default_probe_library(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(DEFAULT_PROBES, columns=["probe_id", "smiles"]).to_csv(path, index=False)
    print(f"[PROBE] wrote default probe library -> {path}")


def fetch_ccd_sdf(ccd: str, out_dir: Path, overwrite: bool = False, timeout: int = 30) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ccd}.sdf"
    rec = {"ligand_id": ccd, "kind": "ccd", "status": "ok", "path": str(out_path),
           "n_heavy_atoms": "", "error": ""}
    if out_path.exists() and not overwrite and out_path.stat().st_size > 0:
        rec["status"] = "cached"
    else:
        url = RCSB_IDEAL_SDF.format(ccd=ccd)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            if not data or b"$$$$" not in data:
                raise ValueError("empty or non-SDF payload")
            out_path.write_bytes(data)
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            return rec
    # heavy-atom count via RDKit if available
    Chem, _ = _try_rdkit()
    if Chem is not None:
        try:
            mol = next(iter(Chem.SDMolSupplier(str(out_path), removeHs=True)), None)
            if mol is not None:
                rec["n_heavy_atoms"] = mol.GetNumHeavyAtoms()
        except Exception:
            pass
    return rec


def embed_probe_sdf(probe_id: str, smiles: str, out_dir: Path, overwrite: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{probe_id}.sdf"
    rec = {"ligand_id": probe_id, "kind": "probe", "status": "ok", "path": str(out_path),
           "n_heavy_atoms": "", "error": ""}
    if out_path.exists() and not overwrite and out_path.stat().st_size > 0:
        rec["status"] = "cached"
        return rec
    Chem, AllChem = _try_rdkit()
    if Chem is None:
        rec["status"] = "error"
        rec["error"] = "rdkit not available; cannot embed probe SMILES"
        return rec
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("unparseable SMILES")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("embedding failed")
        AllChem.MMFFOptimizeMolecule(mol)
        rec["n_heavy_atoms"] = Chem.RemoveHs(mol).GetNumHeavyAtoms()
        w = Chem.SDWriter(str(out_path))
        w.write(mol)
        w.close()
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve docking ligands (CCD SDFs + probe embeddings).")
    p.add_argument("--ligands_csv", type=Path, default=DEFAULT_LIGANDS_CSV,
                   help="ligands.csv produced by build_docking_manifest.py (cols: ligand_group, ligand_id, ligand_smiles).")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--probe_csv", type=Path, default=DEFAULT_PROBE_CSV)
    p.add_argument("--write_default_probes", action="store_true",
                   help="(Re)write the built-in default probe library to --probe_csv and exit-less continue.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sleep", type=float, default=0.1, help="Seconds between CCD fetches (be polite to RCSB).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ensure a probe library exists
    if args.write_default_probes or not Path(args.probe_csv).exists():
        write_default_probe_library(args.probe_csv)

    if not Path(args.ligands_csv).exists():
        print(f"[WARN] {args.ligands_csv} not found; only ensured the probe library. "
              f"Run build_docking_manifest.py first to get ligands.csv.")
        return

    ligands = pd.read_csv(args.ligands_csv)
    probe_map = {}
    if Path(args.probe_csv).exists():
        pl = pd.read_csv(args.probe_csv)
        probe_map = dict(zip(pl["probe_id"].astype(str), pl["smiles"].astype(str)))

    reports = []
    ccds = sorted(set(ligands.loc[ligands["ligand_group"].isin([1, 2]), "ligand_id"].astype(str)))
    print(f"[CCD ] resolving {len(ccds)} unique CCDs -> {args.out_dir/'ccd'}")
    for i, ccd in enumerate(ccds, 1):
        reports.append(fetch_ccd_sdf(ccd, args.out_dir / "ccd", overwrite=args.overwrite))
        if args.sleep:
            time.sleep(args.sleep)
        if i % 50 == 0:
            print(f"  {i}/{len(ccds)}")

    probes = sorted(set(ligands.loc[ligands["ligand_group"] == 3, "ligand_id"].astype(str)))
    if probes:
        print(f"[PRB ] embedding {len(probes)} probes -> {args.out_dir/'probes'}")
        for pid in probes:
            smi = probe_map.get(pid, "")
            if not smi:
                # fall back to SMILES carried in ligands.csv
                hit = ligands.loc[(ligands["ligand_group"] == 3) & (ligands["ligand_id"].astype(str) == pid), "ligand_smiles"]
                smi = str(hit.iloc[0]) if len(hit) else ""
            reports.append(embed_probe_sdf(pid, smi, args.out_dir / "probes", overwrite=args.overwrite))

    rep_df = pd.DataFrame(reports)
    rep_csv = args.out_dir / "ligand_resolution.csv"
    rep_df.to_csv(rep_csv, index=False)

    n_ok = int((rep_df["status"].isin(["ok", "cached"])).sum()) if len(rep_df) else 0
    n_err = int((rep_df["status"] == "error").sum()) if len(rep_df) else 0
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ligands_csv": str(args.ligands_csv),
        "out_dir": str(args.out_dir),
        "probe_csv": str(args.probe_csv),
        "n_ccds": len(ccds), "n_probes": len(probes),
        "resolved_ok": n_ok, "resolution_errors": n_err,
        "rdkit_available": _try_rdkit()[0] is not None,
    }
    with open(args.out_dir / "ligand_resolution_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVE] {rep_csv}")
    print(f"[DONE] resolved_ok={n_ok} errors={n_err}  (rdkit={meta['rdkit_available']})")
    if n_err:
        print("       Some ligands failed (ions/covalent/sugars/obsolete CCDs are common); see the report.")


if __name__ == "__main__":
    main()
