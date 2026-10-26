# -*- coding: utf-8 -*-
"""
Parse AHoJ job keys into query-holo ligand records (docking ligand GROUP 1).

An AHoJ job key encodes a query holo binding event as::

    {pdb}_{chain}_{CCD}_{resnum}          e.g.  4b11_c_7i1_999

so the 3rd token is the ligand CCD to dock, and (pdb, chain, resnum) locate the
*experimental* ligand pose (used later for redocking RMSD; see
``build_reference_poses.py``). This is the same key format AlphaFill's
``reconstructed_key`` uses and that ``03_alphafill/merge_ahoj_alphafill_v6.py``
matches on.

The ST_100 dataset carries these keys in two columns:
  - ``rep_job``   : the representative job key for the pocket (single key)
  - ``job_names`` : all member job keys (semicolon-separated)

This module is import-friendly (use ``extract_query_ligands`` /
``parse_job_key``) and also runnable as a small QC CLI that writes a
per-pocket query-ligand table from the geometry dataset.

NOTE: the ``{pdb}_{chain}_{CCD}_{resnum}`` convention is inferred from
``normalize_job_key`` in ``merge_ahoj_alphafill_v6.py`` (its docstring example
``4b11_c_7i1_999 -> 4B11-C-7I1-999``). Confirm against a sample of your current
``job_names`` before the full run; the parser is defensive about token counts.

IMPORTANT — what ``chain`` / ``resnum`` refer to: they are the *ligand's* PDB
``auth_asym_id`` / ``auth_seq_id``, NOT the receptor protein chain. This is fixed
by the AlphaFill match: ``merge_ahoj_alphafill_v6.py`` matches these keys against
AlphaFill's ``reconstructed_key = f"{pdb_id}_{ligand_pdb_auth_asym_id}_
{compound_id}_{ligand_pdb_auth_seq_id}"``, so the tokens are the ligand's. Under
author numbering the ligand usually shares its protein chain's id, but not always.
Consequences: (a) docking is unaffected (AFDB receptor is chain A, box from
AF-mapped residues); (b) extracting the experimental ligand by (chain, CCD,
resnum) is correct; (c) when superposing the source PDB *protein* onto the AF
model for reference poses, do NOT assume this chain is the protein chain — use a
sequence / SIFTS alignment (see ``build_reference_poses.py``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional

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


DEFAULT_INPUT = cfg.AFDB_ST100_DATASET_GEOM_CSV
DEFAULT_OUTPUT_DIR = cfg.DOCKING_MANIFEST_DIR


# Ligands that are biologically real but NOT meaningfully dockable with Vina-family
# scoring: monatomic ions (a lone ion is a coordination problem, not docking) and
# ubiquitous buffer / cryoprotectant / crystallization additives. Filtered out of
# groups 1-2 by default (see build_docking_manifest.py --keep_all_ligands to disable).
# NOTE: sugars (GLC/BGC/MAN/BMA/NAG...) are intentionally NOT excluded here — they are
# arguably dockable; flag them separately if you want to review them.
MONOATOMIC_IONS = {
    "ZN", "MG", "CA", "NA", "K", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CD",
    "HG", "CL", "BR", "IOD", "F", "LI", "SR", "BA", "CS", "RB", "AL", "AU", "AG",
    "PT", "PB", "TL", "GA", "MO", "W", "V", "CR", "YB", "GD", "SM", "EU", "Y1",
    "3NI", "3CO", "4MO", "6MO", "OS", "RU", "RH", "IR", "PD", "RE", "SB", "SE", "BE",
}
COMMON_ADDITIVES = {
    "HOH", "DOD", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "12P",
    "DMS", "ACT", "FMT", "SO4", "PO4", "PI", "NO3", "CO3", "MPD", "TRS", "EPE",
    "BME", "DTT", "IPA", "MES", "CIT", "FLC", "TLA", "ACY", "IMD", "NH4", "MOH",
    "BEZ", "ACN", "CCN", "EOH", "PEP", "SCN", "AZI", "BCT", "OXL", "TAR", "MLI",
    "MLA", "GAI", "URE", "SIN", "SGM", "PG0", "PGO", "1BO", "BU3", "DIO",
}
DEFAULT_EXCLUDE_CCDS = MONOATOMIC_IONS | COMMON_ADDITIVES


def is_dockable_ccd(ccd: str, exclude: set | None = None) -> bool:
    """False for monatomic ions / common additives (undockable with Vina)."""
    if not ccd:
        return False
    return str(ccd).upper() not in (exclude if exclude is not None else DEFAULT_EXCLUDE_CCDS)


def _is_intlike(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    if tok[0] in "+-":
        tok = tok[1:]
    return tok.isdigit()


def parse_job_key(job_key: str) -> Optional[Dict[str, str]]:
    """Parse ``{pdb}_{chain}_{CCD}_{resnum}`` into its parts.

    Returns a dict with keys ``job_key, pdb, chain, ccd, resnum`` or ``None`` if
    the token does not look like a job key (e.g. missing an integer resnum).
    Robust to a pdb id that itself contains ``_`` (rare) by anchoring on the
    trailing ``CCD`` and integer ``resnum``.
    """
    if job_key is None:
        return None
    s = str(job_key).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    # Stored AHoJ job keys are HYPHEN-separated ({pdb}-{chain}-{CCD}-{resnum},
    # e.g. '7uw1-A-OIY-3007'; pdb lowercase, chain/CCD uppercase). The raw /
    # AlphaFill form uses underscores ('4b11_c_7i1_999'). Auto-detect: prefer the
    # delimiter that yields >=4 tokens ending in an integer resnum.
    for delim in ("-", "_"):
        if delim not in s:
            continue
        toks = s.split(delim)
        if len(toks) < 4 or not _is_intlike(toks[-1]):
            continue
        resnum = toks[-1].strip()
        ccd = toks[-2].strip().upper()
        chain = toks[-3].strip()
        pdb = delim.join(toks[:-3]).strip().lower()
        if pdb and chain and ccd:
            return {"job_key": s, "pdb": pdb, "chain": chain, "ccd": ccd, "resnum": resnum}
    return None


def _split_job_names(job_names: object) -> List[str]:
    if job_names is None or (isinstance(job_names, float) and pd.isna(job_names)):
        return []
    s = str(job_names).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    return [p.strip() for p in s.split(";") if p and p.strip()]


def extract_query_ligands(
    rep_job: object,
    job_names: object,
    include_members: bool = False,
) -> List[Dict[str, str]]:
    """Return query-holo ligand records for one pocket.

    By default returns only the representative job's ligand (one primary Group-1
    ligand per pocket, keeping job counts sane). With ``include_members=True``
    returns every *unique* (pdb, chain, ccd, resnum) across member jobs, with the
    representative first and an ``is_rep`` flag.
    """
    out: List[Dict[str, str]] = []
    seen = set()

    rep = parse_job_key(rep_job)
    if rep is not None:
        rep["is_rep"] = True
        out.append(rep)
        seen.add((rep["pdb"], rep["chain"], rep["ccd"], rep["resnum"]))

    if include_members:
        for jk in _split_job_names(job_names):
            rec = parse_job_key(jk)
            if rec is None:
                continue
            key = (rec["pdb"], rec["chain"], rec["ccd"], rec["resnum"])
            if key in seen:
                continue
            seen.add(key)
            rec["is_rep"] = False
            out.append(rec)

    return out


def build_query_ligand_table(df: pd.DataFrame, include_members: bool = False) -> pd.DataFrame:
    """Explode the geometry dataset into one row per (pocket × query ligand)."""
    rows: List[Dict[str, object]] = []
    for r in df.itertuples(index=False):
        site_uid = getattr(r, "site_uid", "")
        site_uniprot = getattr(r, "site_uniprot", "")
        rep_job = getattr(r, "rep_job", "")
        job_names = getattr(r, "job_names", "")
        ligs = extract_query_ligands(rep_job, job_names, include_members=include_members)
        for lig in ligs:
            rows.append({
                "site_uid": site_uid,
                "site_uniprot": site_uniprot,
                "ligand_group": 1,
                "ligand_id": lig["ccd"],
                "ligand_ref": lig["job_key"],
                "src_pdb": lig["pdb"],
                "src_chain": lig["chain"],
                "src_resnum": lig["resnum"],
                "is_rep": lig["is_rep"],
            })
    cols = ["site_uid", "site_uniprot", "ligand_group", "ligand_id", "ligand_ref",
            "src_pdb", "src_chain", "src_resnum", "is_rep"]
    return pd.DataFrame(rows, columns=cols)


def load_receptor_map(query_summaries_csv: Path) -> pd.DataFrame:
    """Per-job RECEPTOR info from QUERY_SUMMARIES_CLASS_CSV.

    Provides the receptor protein chain (`prot_chains`) — distinct from the
    job-key *ligand* chain (`lig_chain`), e.g. job `4b1m-E-FRU-1` has ligand
    chain E but protein chain B — plus the pocket residues in UniProt numbering
    (`uniprot_residues`, == AF numbering), keyed by `job_name` (our job key).
    Consumed by build_reference_poses.py so superposition uses the real receptor
    chain and an exact pocket-residue correspondence (no guessing).
    """
    want = {"job_name", "pdb", "lig_chain", "lig_id", "prot_chains",
            "prot_chains_num", "num_binding_residues", "uniprot_residues"}
    return pd.read_csv(query_summaries_csv, usecols=lambda c: c in want, low_memory=False)


def attach_receptor_info(g1_df: pd.DataFrame, query_summaries_csv: Path) -> pd.DataFrame:
    """Left-join receptor chain / pocket residues onto group-1 records.

    Join: our `ligand_ref` (the job key) == `job_name` in QUERY_SUMMARIES_CLASS.
    """
    if g1_df.empty:
        return g1_df
    qs = load_receptor_map(query_summaries_csv).rename(columns={
        "job_name": "ligand_ref", "pdb": "qs_pdb", "lig_chain": "qs_lig_chain", "lig_id": "qs_lig_id",
    })
    return g1_df.merge(qs, on="ligand_ref", how="left")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse AHoJ job keys into query-holo ligand records (group 1).")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--include_members", action="store_true",
                   help="Include all unique member-job ligands, not just the representative.")
    p.add_argument("--query_summaries", type=Path, default=cfg.QUERY_SUMMARIES_CLASS_CSV,
                   help="QUERY_SUMMARIES_CLASS csv: attaches receptor chain (prot_chains) + uniprot_residues.")
    p.add_argument("--no_attach_receptor", action="store_true",
                   help="Do not join receptor info from --query_summaries.")
    p.add_argument("--outfile_name", default="query_ligands_group1.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    tbl = build_query_ligand_table(df, include_members=args.include_members)

    n_receptor = 0
    if not args.no_attach_receptor:
        if Path(args.query_summaries).exists():
            tbl = attach_receptor_info(tbl, args.query_summaries)
            n_receptor = int(tbl["prot_chains"].notna().sum()) if "prot_chains" in tbl.columns else 0
        else:
            print(f"[WARN] query_summaries not found ({args.query_summaries}); receptor chain not attached.")

    out_csv = args.output_dir / args.outfile_name
    tbl.to_csv(out_csv, index=False)

    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_csv": str(args.input_csv),
        "output_csv": str(out_csv),
        "include_members": bool(args.include_members),
        "rows": int(len(tbl)),
        "pockets_with_ligand": int(tbl["site_uid"].nunique()) if len(tbl) else 0,
        "unique_ccds": int(tbl["ligand_id"].nunique()) if len(tbl) else 0,
        "rows_with_receptor_chain": n_receptor,
    }
    with open(args.output_dir / f"{Path(args.outfile_name).stem}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVE] {out_csv}")
    print(f"[ROWS] {len(tbl)}  [POCKETS] {meta['pockets_with_ligand']}  [CCDs] {meta['unique_ccds']}")


if __name__ == "__main__":
    main()
