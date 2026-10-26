# -*- coding: utf-8 -*-
"""
Build group-1 reference ligand poses in the AF-model frame, from the LOCAL escort
experimental-holo structures in the handover bundle (no RCSB fetch).

For each group-1 job whose escort bound ligand IS the docked CCD
(escort_lig_id == ligand_id):
  1. sequence-correspond AF chain A <-> the escort receptor chain;
  2. POCKET-REGION superpose on the escort UniProt residues (== AF numbering);
  3. transform the experimental ligand heavy atoms into the AF frame;
  4. map them onto the CCD ideal-SDF template graph and write a reference SDF
     (template order) -> ref_poses/<job_id>.sdf  (for symmetry-corrected redock RMSD).

Mirrors the validated 05_docking/dataset/build_reference_poses.py, but reads the
bundled `experimental_holo/*.cif.gz` + escort columns instead of fetching PDBs, and
emits SDF (template) rather than PDB so downstream RMSD is graph-based.

Usage:
    python build_reference_poses_local.py --manifest <handover_manifest.csv> \
        --bundle <bundle> --out_dir <ref_poses_dir> [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prep_common import find_af_cif, safe_token
from rmsd_utils import build_reference_conformer


# ---- gemmi helpers (self-contained copy of the validated superposition logic) ----
def _read_model(path: Path):
    import gemmi
    return gemmi.read_structure(str(path))[0]


def seq_correspondence(afc, pc) -> Dict[int, object]:
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
    sup = gemmi.calculate_superposition(afc.get_polymer(), pc.get_polymer(),
                                        gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP)
    return sup.transform, float(sup.rmsd), int(sup.count), "wholechain(fallback)"


def extract_ligand_atoms(pdb_model, chain: str, ccd: str, resnum: int):
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


def _positions_from_uniprot_residues(ur: str) -> List[int]:
    """'[\"P123_16\", ...]' -> [16, ...] (last _ token)."""
    try:
        toks = ast.literal_eval(ur) if ur.strip().startswith("[") else ur.split()
    except Exception:
        toks = str(ur).split()
    out = set()
    for t in toks:
        try:
            out.add(int(str(t).split("_")[-1]))
        except Exception:
            continue
    return sorted(out)


def build_one(row: dict, bundle: str, out_dir: str) -> dict:
    from rdkit import Chem
    bundle = Path(bundle)
    out_dir = Path(out_dir)
    qc = {"job_id": row["job_id"], "site_uid": row["site_uid"], "ligand_id": row["ligand_id"],
          "status": "ok", "mode": "", "n_pocket_pairs": 0, "local_rmsd": None,
          "map_status": "", "ref_sdf": "", "error": ""}
    try:
        af_cif = find_af_cif(str(row["site_uniprot"]), bundle / "receptors")
        if af_cif is None:
            qc.update(status="error", error="AF model not found"); return qc
        esc_rel = str(row.get("escort_cif_gz", "")).strip()
        if not esc_rel:
            qc.update(status="skipped", error="no escort"); return qc
        esc_cif = bundle / esc_rel
        if not esc_cif.exists():
            qc.update(status="error", error=f"escort missing {esc_rel}"); return qc

        prot_chain = str(row.get("escort_prot_chains", "")).split()[0] if str(row.get("escort_prot_chains", "")).split() else "A"
        lig_chain = str(row.get("escort_lig_chain", prot_chain))
        ccd = str(row["ligand_id"])
        resnum = int(row["escort_lig_resnum"])
        positions = _positions_from_uniprot_residues(str(row.get("escort_uniprot_residues", "")))
        if not positions:
            qc.update(status="skipped", error="no pocket residues"); return qc

        af = _read_model(af_cif)
        pdb = _read_model(esc_cif)
        afc = next((c for c in af if c.name == "A"), None)
        pc = next((c for c in pdb if c.name == prot_chain), None)
        if afc is None or pc is None:
            qc.update(status="error", error=f"chain missing (AF A / PDB {prot_chain})"); return qc

        T, rmsd, npair, mode = pocket_superpose(afc, pc, positions)
        qc.update(mode=mode, n_pocket_pairs=npair, local_rmsd=round(rmsd, 3))

        atoms = extract_ligand_atoms(pdb, lig_chain, ccd, resnum)
        if not atoms:
            qc.update(status="error", error="ligand atoms not found"); return qc
        elements = [el for _, el, _ in atoms]
        af_coords = np.array([[ (v:=T.apply(pos)).x, v.y, v.z ] for _, _, pos in atoms], dtype=float)

        tmpl_sdf = bundle / "ligands" / "ccd" / f"{ccd}.sdf"
        template = Chem.SDMolSupplier(str(tmpl_sdf), removeHs=True)[0] if tmpl_sdf.exists() else None
        if template is None:
            qc.update(status="error", error="template sdf missing/unreadable"); return qc

        ref_mol, mstat = build_reference_conformer(template, elements, af_coords)
        qc["map_status"] = mstat
        if ref_mol is None:
            qc.update(status="error", error=f"map:{mstat}"); return qc

        out_dir.mkdir(parents=True, exist_ok=True)
        out_sdf = out_dir / f"{safe_token(row['job_id'])}.sdf"
        w = Chem.SDWriter(str(out_sdf))
        ref_mol.SetProp("_Name", f"{row['job_id']}|ref|{ccd}")
        w.write(ref_mol)
        w.close()
        qc["ref_sdf"] = out_sdf.name
    except Exception as exc:
        qc.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return qc


def _worker(args):
    row, bundle, out_dir = args
    return build_one(row, bundle, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="handover manifest with escort_* columns")
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest, low_memory=False)
    g1 = m[m["ligand_group"] == 1].copy()
    g1 = g1[g1["escort_lig_id"].astype(str) == g1["ligand_id"].astype(str)]  # docked CCD == bound CCD
    if args.limit:
        g1 = g1.head(args.limit)
    print(f"[REFPOSES] {len(g1)} group-1 jobs with matching escort ligand  workers={args.workers}")

    tasks = [(r, str(args.bundle), str(args.out_dir)) for r in g1.to_dict("records")]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            reports.append(fut.result())
            if i % 100 == 0 or i == len(tasks):
                nok = sum(r["status"] == "ok" for r in reports)
                print(f"  {i}/{len(tasks)}  ok={nok}")

    rep = pd.DataFrame(reports)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.out_dir / "reference_poses_report.csv", index=False)
    ok = int((rep["status"] == "ok").sum()) if len(rep) else 0
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "n_jobs": int(len(rep)), "ok": ok,
        "median_local_rmsd": float(np.nanmedian(pd.to_numeric(rep["local_rmsd"], errors="coerce"))) if len(rep) else None,
        "wholechain_fallbacks": int((rep["mode"] == "wholechain(fallback)").sum()) if len(rep) else 0,
        "map_failures": int((rep["map_status"].isin(["no_isomorphism", "partial_isomorphism", "ref_raw_build_failed"])).sum()) if len(rep) else 0,
    }
    with open(args.out_dir / "reference_poses_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[DONE] ok={ok}/{len(rep)}  median local rmsd={meta['median_local_rmsd']}  "
          f"fallbacks={meta['wholechain_fallbacks']}  map_fail={meta['map_failures']}")


if __name__ == "__main__":
    main()
