# -*- coding: utf-8 -*-
"""
Attach AUTHORITATIVE PDB-numbered pocket residues to the escort table by parsing
AHoJ's own per-job output (no re-mapping).

Each AHoJ query has a `.tgz` under output_fix/<batch>/<job>.tgz containing
`pocket_residues.csv`, whose QUERY row (chain == the query PDB id) lists the
binding residues already in PDB numbering, e.g.:

    chain,query_ligand,mapped_binding_residues
    101m,A_HEM_155,A_39 A_42 A_43 ...          <- query, PDB numbering (what we want)
    AF_P02185_as_101mA,A_HEM_155,40 43 44 ...  <- AF/UniProt numbering (== uniprot_residues)
    102m,pocket_1,...                          <- result structures (ignored)

This is the source the user pointed to; it supersedes the alignment-mapped
`escort_pdb_residues`. We keep the mapped values only as a fallback when a job's
.tgz is absent. Adds/updates: `escort_pdb_residues` (authoritative where found),
`escort_pdb_residues_n`, `escort_pdb_residues_source` ('ahoj' | 'mapped' | '').

Base env is fine (tarfile + pandas; no gemmi).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Optional

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


def _job_name(pdb: str, lig_chain: str, lig_id: str, lig_resnum) -> str:
    try:
        rn = str(int(float(lig_resnum)))
    except Exception:
        rn = str(lig_resnum).strip()
    return f"{str(pdb).lower()}-{str(lig_chain).strip()}-{str(lig_id).upper().strip()}-{rn}"


def build_tgz_index(outfix: Path) -> Dict[str, Path]:
    """job_name -> .tgz path, over all batches."""
    idx: Dict[str, Path] = {}
    for batch in outfix.iterdir():
        if batch.is_dir():
            for f in batch.glob("*.tgz"):
                idx[f.stem] = f
    return idx


def query_pdb_residues(tgz: Path, pdb: str) -> Optional[str]:
    """Return the QUERY row's mapped_binding_residues (PDB numbering) or None."""
    try:
        with tarfile.open(tgz, "r:gz") as t:
            member = next((m for m in t.getnames() if m.endswith("pocket_residues.csv")), None)
            if member is None:
                return None
            data = t.extractfile(member).read().decode("utf-8", "ignore")
    except Exception:
        return None
    try:
        df = pd.read_csv(io.StringIO(data))
    except Exception:
        return None
    if "chain" not in df.columns or "mapped_binding_residues" not in df.columns:
        return None
    # query row: chain equals the query PDB id (not 'AF_...', not a result pdb)
    q = df[df["chain"].astype(str).str.lower() == str(pdb).lower()]
    if q.empty:
        return None
    val = str(q.iloc[0]["mapped_binding_residues"]).strip()
    return val or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach authoritative PDB pocket residues from AHoJ .tgz output.")
    p.add_argument("--escort_csv", type=Path, default=cfg.DOCKING_PILOT_DIR / "experimental_structures.csv")
    p.add_argument("--outfix", type=Path, default=cfg.AHOJ_OUTPUT_FIX_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.require("AHOJ_OUTPUT_FIX_DIR") if hasattr(cfg, "require") else None
    e = pd.read_csv(args.escort_csv, low_memory=False)

    print(f"[IDX ] indexing .tgz under {args.outfix} ...")
    idx = build_tgz_index(args.outfix)
    print(f"[IDX ] {len(idx)} job archives found")

    mapped_fallback = e["escort_pdb_residues"] if "escort_pdb_residues" in e.columns else pd.Series([""] * len(e))
    cache: Dict[str, Optional[str]] = {}
    out_res, out_n, out_src = [], [], []
    n_ahoj = n_fallback = n_none = 0

    for i, r in enumerate(e.itertuples(index=False)):
        pdb = str(getattr(r, "escort_pdb_id", "")).lower()
        jn = _job_name(pdb, getattr(r, "escort_lig_chain", ""), getattr(r, "escort_lig_id", ""), getattr(r, "escort_lig_resnum", ""))
        if jn not in cache:
            tgz = idx.get(jn)
            cache[jn] = query_pdb_residues(tgz, pdb) if tgz else None
        res = cache[jn]
        if res:
            out_res.append(res); out_n.append(len(res.split())); out_src.append("ahoj"); n_ahoj += 1
        else:
            fb = str(mapped_fallback.iloc[i]) if i < len(mapped_fallback) else ""
            if fb and fb.lower() != "nan":
                out_res.append(fb); out_n.append(len(fb.split())); out_src.append("mapped"); n_fallback += 1
            else:
                out_res.append(""); out_n.append(0); out_src.append(""); n_none += 1

    e["escort_pdb_residues"] = out_res
    e["escort_pdb_residues_n"] = out_n
    e["escort_pdb_residues_source"] = out_src
    e.to_csv(args.escort_csv, index=False)

    uni_n = e["escort_uniprot_residues"].map(
        lambda v: len(str(v).split(",")) if isinstance(v, str) and v.strip().startswith("[") else 0)
    exact = int((e["escort_pdb_residues_n"] == uni_n).sum())
    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "escort_csv": str(args.escort_csv), "outfix": str(args.outfix),
        "rows": int(len(e)), "from_ahoj": n_ahoj, "from_mapped_fallback": n_fallback, "none": n_none,
        "exact_count_match_vs_uniprot": exact,
    }
    with open(args.escort_csv.parent / "escort_pdb_residues_ahoj_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[SAVE] {args.escort_csv}")
    print(f"[DONE] authoritative(ahoj)={n_ahoj} fallback(mapped)={n_fallback} none={n_none} | exact==uniprot: {exact}/{len(e)}")


if __name__ == "__main__":
    main()
