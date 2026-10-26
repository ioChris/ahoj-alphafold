# -*- coding: utf-8 -*-
"""
Add PDB-numbered pocket residues (`escort_pdb_residues`) to the escort table.

AHoJ's binding-residue set == `uniprot_residues` (validated: it equals AHoJ's
`num_binding_residues` exactly). The authoritative PDB-numbered list lives inside
the 40 GB Main_data TAR (per-query global_results), which is impractical to random-
access. A structural recompute (4.5 A contacts) OVER-counts vs AHoJ's set, so it is
NOT used. Instead we express AHoJ's *canonical* set in PDB numbering by mapping each
UniProt pocket position to its PDB residue via a sequence alignment of the AF model
chain A (UniProt-numbered) to the PDB receptor chain. This reproduces AHoJ's exact
residues (count-matched by construction), only re-expressed in PDB indices.

Run in an env with gemmi (e.g. py311). Reuses the validated alignment helpers from
build_reference_poses.py. Adds columns `escort_pdb_residues` (e.g. "A_45 A_67 ...")
and `escort_pdb_residues_n` (mapped count) to experimental_structures.csv.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # allow sibling imports
from build_reference_poses import seq_correspondence, find_af_cif, _read_model


def _parse_ur(val: str) -> Tuple[str, List[int]]:
    """'[\"P02185_40\", ...]' -> (accession, [positions])."""
    try:
        toks = ast.literal_eval(str(val))
    except Exception:
        return "", []
    if not isinstance(toks, list) or not toks:
        return "", []
    acc = str(toks[0]).rsplit("_", 1)[0]
    pos = []
    for t in toks:
        try:
            pos.append(int(str(t).rsplit("_", 1)[1]))
        except Exception:
            continue
    return acc, sorted(set(pos))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map AHoJ pocket residues to PDB numbering (escort_pdb_residues).")
    p.add_argument("--escort_csv", type=Path, default=cfg.DOCKING_PILOT_DIR / "experimental_structures.csv")
    p.add_argument("--holo_dir", type=Path, default=None, help="Dir with <pdb>.cif.gz (default: <escort_csv parent>/experimental_holo).")
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    holo_dir = args.holo_dir or (args.escort_csv.parent / "experimental_holo")
    e = pd.read_csv(args.escort_csv, low_memory=False)

    cache: Dict[Tuple[str, str, str], Dict[int, object]] = {}   # (acc, pdb, chain) -> {uni_pos: pdb_res}
    pdb_res_col, n_col = [], []
    n_ok = n_partial = n_fail = 0

    for r in e.itertuples(index=False):
        ur = getattr(r, "escort_uniprot_residues", "")
        pdb = str(getattr(r, "escort_pdb_id", "")).lower()
        chain = str(getattr(r, "escort_prot_chains", "")).split()[0] if getattr(r, "escort_prot_chains", "") else ""
        acc, positions = _parse_ur(ur)
        if not (acc and positions and pdb and chain):
            pdb_res_col.append(""); n_col.append(0); n_fail += 1
            continue
        key = (acc, pdb, chain)
        if key not in cache:
            mp: Dict[int, object] = {}
            try:
                af_cif = find_af_cif(acc, args.afdb_dir)
                pdb_cif = holo_dir / f"{pdb}.cif.gz"
                if af_cif and pdb_cif.exists():
                    af = _read_model(af_cif); pm = _read_model(pdb_cif)
                    afc = next((c for c in af if c.name == "A"), None)
                    pc = next((c for c in pm if c.name == chain), None)
                    if afc and pc:
                        corr = seq_correspondence(afc, pc)  # AF seqid (==UniProt pos) -> PDB residue
                        mp = {p: corr[p].seqid.num for p in positions if p in corr}
            except Exception:
                mp = {}
            cache[key] = mp
        mp = cache[key]
        toks = [f"{chain}_{mp[p]}" for p in positions if p in mp]
        pdb_res_col.append(" ".join(toks)); n_col.append(len(toks))
        if len(toks) == len(positions):
            n_ok += 1
        elif toks:
            n_partial += 1
        else:
            n_fail += 1

    e["escort_pdb_residues"] = pdb_res_col
    e["escort_pdb_residues_n"] = n_col
    e.to_csv(args.escort_csv, index=False)

    uni_n = e["escort_uniprot_residues"].map(lambda v: len(_parse_ur(v)[1]))
    exact = int((e["escort_pdb_residues_n"] == uni_n).sum())
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "escort_csv": str(args.escort_csv), "rows": int(len(e)),
        "fully_mapped": n_ok, "partial": n_partial, "unmapped": n_fail,
        "exact_count_match_vs_uniprot": exact,
    }
    with open(args.escort_csv.parent / "escort_pdb_residues_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[SAVE] {args.escort_csv}  (+escort_pdb_residues)")
    print(f"[MAP ] fully={n_ok} partial={n_partial} unmapped={n_fail} | exact-count==uniprot: {exact}/{len(e)}")


if __name__ == "__main__":
    main()
