# -*- coding: utf-8 -*-
"""
Build reference ligand poses in the AF model frame (for redock RMSD, group 1).

For each query-holo ligand we place the *experimental* ligand into the AF model
coordinate frame, so a docked pose (which is produced in the AF frame) can be
compared to it (pose-recovery RMSD).

Method (validated on a handful of real cases — see caveats):
  1. Correspond AF chain A <-> the PDB RECEPTOR chain by SEQUENCE alignment
     (gemmi.align_string_sequences) — robust to apo/holo conformational change,
     unlike spatial matching which needs a good global fit first.
  2. POCKET-REGION superposition: fit only the pocket residues (uniprot_residues,
     UniProt numbering == AF numbering) via superpose_positions. This isolates
     in-pocket pose accuracy from global domain motion. Falls back to a
     whole-chain CaP superposition if < min_pairs pocket residues map.
  3. Apply the transform to the experimental ligand atoms and write them as a PDB
     in the AF frame -> DOCKING_REFPOSES_DIR/<site_uid>__<CCD>.pdb .

Inputs: the group-1 table from parse_ahoj_job_ligands.py *with receptor info
attached* (needs columns: site_uid, site_uniprot, ligand_id, ligand_ref,
src_pdb, src_chain, src_resnum, prot_chains, uniprot_residues). Run in an env
with gemmi (e.g. the py311 analysis env).

QC (per job, written to reference_poses_report.csv): n_pocket_pairs, local_rmsd,
lig_centroid_to_pocket_A (ligand centroid vs pocket-CA centroid in the AF frame),
mode, status. Filter/spot-check borderline rows (high rmsd / large centroid dist).

CAVEATS (be honest):
  - Output is PDB, not SDF: symmetry-corrected redock RMSD (rdkit/spyrmsd) is
    computed downstream against the docked pose; atom names are preserved.
  - Multi-chain receptors: the first prot_chains token is used (the ST_100 set is
    monomeric single-chain, so this is normally moot).
  - Group 2 (AlphaFill) reference poses are already in the AF frame (the transplant
    coords) — extract them from the AlphaFill *filled* CIF instead; not handled here.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import glob
import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe(s) -> str:
    """Filesystem-safe token (site_uid has colons, illegal on Windows)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

RCSB_CIF = "https://files.rcsb.org/download/{pdb}.cif"
DEFAULT_INPUT = cfg.DOCKING_MANIFEST_DIR / "query_ligands_group1.csv"
DEFAULT_OUT_DIR = cfg.DOCKING_REFPOSES_DIR


def _np(p) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def find_af_cif(uniprot: str, afdb_dir: Path) -> Optional[Path]:
    hits = glob.glob(str(afdb_dir / f"AF-{uniprot}-F1-model_v*.cif.gz"))
    if not hits:
        return None
    def ver(p):
        try:
            return int(Path(p).name.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            return -1
    return Path(max(hits, key=ver))


def fetch_pdb_cif(pdb: str, cache_dir: Path, timeout: int = 60) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb.lower()}.cif"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        urllib.request.urlretrieve(RCSB_CIF.format(pdb=pdb.lower()), out)
        return out
    except Exception as exc:
        print(f"[MISS] {pdb}: {exc}")
        return None


def _read_model(path: Path):
    import gemmi
    return gemmi.read_structure(str(path))[0]  # gemmi reads .gz by extension


def seq_correspondence(afc, pc) -> Dict[int, object]:
    """Map AF residue seqid (== UniProt position) -> PDB residue, by sequence alignment."""
    import gemmi
    ap, pp = afc.get_polymer(), pc.get_polymer()
    ar, pr = list(ap), list(pp)
    s1 = gemmi.one_letter_code(ap.extract_sequence())
    s2 = gemmi.one_letter_code(pp.extract_sequence())
    res = gemmi.align_string_sequences(list(s1), list(s2), [])
    g1, g2 = res.add_gaps(s1, 1), res.add_gaps(s2, 2)
    mp: Dict[int, object] = {}
    i = j = 0
    for c1, c2 in zip(g1, g2):
        if c1 != "-" and c2 != "-" and i < len(ar) and j < len(pr):
            mp[ar[i].seqid.num] = pr[j]
        if c1 != "-":
            i += 1
        if c2 != "-":
            j += 1
    return mp


def pocket_superpose(afc, pc, positions: List[int], min_pairs: int = 3):
    """Return (transform, rmsd, n_pairs, mode). Pocket-region fit with whole-chain fallback."""
    import gemmi
    mp = seq_correspondence(afc, pc)
    af_ca = {r.seqid.num: r for r in afc if r.find_atom("CA", "*")}
    A, B = [], []
    for p in positions:
        pr = mp.get(p)
        if p in af_ca and pr is not None and pr.find_atom("CA", "*"):
            A.append(af_ca[p].find_atom("CA", "*").pos)
            B.append(pr.find_atom("CA", "*").pos)
    if len(A) >= min_pairs:
        sup = gemmi.superpose_positions(A, B)
        return sup.transform, float(sup.rmsd), len(A), "pocket"
    # fallback: whole-chain CaP
    sup = gemmi.calculate_superposition(afc.get_polymer(), pc.get_polymer(),
                                        gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP)
    return sup.transform, float(sup.rmsd), int(sup.count), "wholechain(fallback)"


def extract_ligand_atoms(pdb_model, chain: str, ccd: str, resnum: int) -> List[Tuple[str, str, object]]:
    """Return [(atom_name, element, pos)] heavy atoms of the target ligand (PDB coords)."""
    out = []
    for c in pdb_model:
        if c.name != chain:
            continue
        for r in c:
            if r.name.upper() == ccd.upper() and r.seqid.num == resnum:
                for a in r:
                    el = str(a.element.name) if hasattr(a.element, "name") else str(a.element)
                    if el != "H":
                        out.append((a.name, el, a.pos))
    return out


def write_ligand_pdb(atoms, ccd: str, transform, out_path: Path) -> np.ndarray:
    """Apply transform, write ligand as PDB in the AF frame, return transformed xyz array."""
    import gemmi
    st = gemmi.Structure()
    st.name = ccd
    model = gemmi.Model("1")
    chain = gemmi.Chain("L")
    res = gemmi.Residue()
    res.name = ccd[:3]
    res.seqid = gemmi.SeqId("1")
    res.het_flag = "H"
    xyz = []
    for name, el, pos in atoms:
        v = transform.apply(pos)
        a = gemmi.Atom()
        a.name = name
        a.element = gemmi.Element(el)
        a.pos = gemmi.Position(v.x, v.y, v.z)
        res.add_atom(a)
        xyz.append([v.x, v.y, v.z])
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_path))
    return np.array(xyz, dtype=float)


def build_one(row, afdb_dir: Path, cache: Path, out_dir: Path) -> dict:
    import pandas as pd
    qc = {"site_uid": row["site_uid"], "ligand_id": row["ligand_id"], "ligand_ref": row["ligand_ref"],
          "status": "ok", "mode": "", "n_pocket_pairs": 0, "local_rmsd": None,
          "lig_centroid_to_pocket_A": None, "pose_pdb": "", "error": ""}
    try:
        prot_chains = str(row.get("prot_chains", "")).split()
        ur = row.get("uniprot_residues", "")
        if not prot_chains or not isinstance(ur, str) or not ur.strip():
            qc["status"] = "skipped"; qc["error"] = "missing prot_chains/uniprot_residues"; return qc
        prot_chain = prot_chains[0]
        positions = sorted({int(x.split("_")[-1]) for x in ast.literal_eval(ur)})

        af_cif = find_af_cif(str(row["site_uniprot"]), afdb_dir)
        if af_cif is None:
            qc["status"] = "error"; qc["error"] = "AF model not found"; return qc
        pdb_cif = fetch_pdb_cif(str(row["src_pdb"]), cache)
        if pdb_cif is None:
            qc["status"] = "error"; qc["error"] = "PDB fetch failed"; return qc

        af = _read_model(af_cif); pdb = _read_model(pdb_cif)
        afc = next((c for c in af if c.name == "A"), None)
        pc = next((c for c in pdb if c.name == prot_chain), None)
        if afc is None or pc is None:
            qc["status"] = "error"; qc["error"] = f"chain missing (AF A / PDB {prot_chain})"; return qc

        T, rmsd, npair, mode = pocket_superpose(afc, pc, positions)
        qc["mode"] = mode; qc["n_pocket_pairs"] = npair; qc["local_rmsd"] = round(rmsd, 3)

        atoms = extract_ligand_atoms(pdb, str(row["src_chain"]), str(row["ligand_id"]), int(row["src_resnum"]))
        if not atoms:
            qc["status"] = "error"; qc["error"] = "ligand atoms not found"; return qc

        out_pdb = out_dir / f"{_safe(row['site_uid'])}__{_safe(row['ligand_id'])}.pdb"
        lig_xyz = write_ligand_pdb(atoms, str(row["ligand_id"]), T, out_pdb)
        qc["pose_pdb"] = out_pdb.name

        # QC: ligand centroid vs pocket-CA centroid (both in AF frame)
        af_ca = {r.seqid.num: r for r in afc if r.find_atom("CA", "*")}
        pocket_xyz = np.array([_np(af_ca[p].find_atom("CA", "*").pos) for p in positions if p in af_ca])
        if len(pocket_xyz) and len(lig_xyz):
            qc["lig_centroid_to_pocket_A"] = round(float(np.linalg.norm(lig_xyz.mean(0) - pocket_xyz.mean(0))), 2)
    except Exception as exc:
        qc["status"] = "error"; qc["error"] = f"{type(exc).__name__}: {exc}"
    return qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build group-1 reference poses (experimental ligand in AF frame).")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT,
                   help="query_ligands_group1.csv WITH receptor info (from parse_ahoj_job_ligands.py).")
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pdb_cache", type=Path, default=DEFAULT_OUT_DIR / "_pdb_cache")
    p.add_argument("--limit", type=int, default=0, help="Process only first N (0 = all).")
    p.add_argument("--site_uids", type=Path, default=None, help="Optional CSV with a site_uid column to restrict.")
    return p.parse_args()


def main() -> None:
    import pandas as pd
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    need = {"site_uid", "site_uniprot", "ligand_id", "src_pdb", "src_chain", "src_resnum", "prot_chains", "uniprot_residues"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"input missing columns {missing}; run parse_ahoj_job_ligands.py with receptor attach first.")
    if args.site_uids and Path(args.site_uids).exists():
        keep = set(pd.read_csv(args.site_uids)["site_uid"].astype(str))
        df = df[df["site_uid"].astype(str).isin(keep)]
    if args.limit:
        df = df.head(args.limit)

    reports = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        reports.append(build_one(row, args.afdb_dir, args.pdb_cache, args.out_dir))
        if i % 25 == 0:
            print(f"  {i}/{len(df)}")

    rep = pd.DataFrame(reports)
    rep_csv = args.out_dir / "reference_poses_report.csv"
    rep.to_csv(rep_csv, index=False)
    ok = int((rep["status"] == "ok").sum()) if len(rep) else 0
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_csv": str(args.input_csv), "out_dir": str(args.out_dir),
        "n_jobs": int(len(rep)), "ok": ok,
        "median_local_rmsd": float(np.nanmedian(pd.to_numeric(rep["local_rmsd"], errors="coerce"))) if len(rep) else None,
        "wholechain_fallbacks": int((rep["mode"] == "wholechain(fallback)").sum()) if len(rep) else 0,
    }
    with open(args.out_dir / "reference_poses_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[SAVE] {rep_csv}")
    print(f"[DONE] ok={ok}/{len(rep)}  median local rmsd={meta['median_local_rmsd']}  fallbacks={meta['wholechain_fallbacks']}")


if __name__ == "__main__":
    main()
