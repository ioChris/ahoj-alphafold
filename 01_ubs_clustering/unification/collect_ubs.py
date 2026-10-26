# -*- coding: utf-8 -*-
"""
Collect the per-UniProt Unified Binding Site (UBS) tables written by
`run_ubs_unification.py` into one flat TSV.

Usage
-----
    python collect_ubs.py UBS_ROOT --threshold ST_100 --out sites.tsv [--input ubs_input.csv]

`UBS_ROOT` is the `--out` directory of the unification run: it holds one
sub-directory per UniProt accession with `ST_XXX.tsv` / `ST_LJXXX.tsv` tables.

Output columns
--------------
site_uid              ``<ACC>:<ST_XXX>:<ubs_id>``
uniprot               accession
residues              ``+``-joined UniProt indices of ``total_residues`` (the union), ascending
members               ``entry_ids`` (job names) ``;``-joined, in the table's order
n_members             number of members
multimeric_status_all the status codes present among the members (space-separated)
ligands               (only with ``--input``) the members' ``ligands`` values, ``;``-joined in member order

The module is importable: ``collect_ubs(root, threshold="ST_100", input_csv=None)``
returns the same table as a ``pandas.DataFrame``.
"""

import argparse
import ast
import glob
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["site_uid", "uniprot", "residues", "members", "n_members", "multimeric_status_all"]
LJ_LABEL = "LJ"


def normalize_threshold_label(threshold) -> str:
    """
    Accept ``ST_100``, ``100``, ``1.0``, ``0.7``, ``ST_LJ`` / ``LJ`` and return the
    table label: ``ST_XXX`` (3-digit percentage) or ``LJ`` (the large-jump table,
    whose file name varies per UniProt).
    """
    s = str(threshold).strip()
    if s.upper() in {"LJ", "ST_LJ"}:
        return LJ_LABEL
    if s.upper().startswith("ST_"):
        body = s[3:]
        if body.upper().startswith("LJ"):
            return LJ_LABEL if len(body) == 2 else f"ST_LJ{body[2:]}"
        return f"ST_{int(body):03d}"
    value = float(s)
    if value <= 1.0:
        value *= 100.0
    return f"ST_{int(round(value)):03d}"


def _parse_list(cell):
    """Parse a Python-list-literal cell of the TSV (``entry_ids``, ``total_residues``)."""
    if isinstance(cell, (list, tuple)):
        return list(cell)
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [s]
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return [val]


def residue_indices(tokens):
    """``['P0CG47_10', 'P0CG47_2']`` -> ``[2, 10]`` (UniProt indices, ascending)."""
    idx = []
    for tok in tokens:
        tok = str(tok)
        try:
            idx.append(int(tok.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            logger.warning("cannot parse UniProt index from residue token %r", tok)
    return sorted(set(idx))


def _table_path(uniprot_dir: str, label: str):
    """Path of the requested threshold table in a per-UniProt directory, or None."""
    if label == LJ_LABEL:
        hits = sorted(glob.glob(os.path.join(uniprot_dir, "ST_LJ*.tsv")))
        if len(hits) > 1:
            logger.warning("%s: several large-jump tables, using %s", uniprot_dir, os.path.basename(hits[0]))
        return hits[0] if hits else None
    path = os.path.join(uniprot_dir, f"{label}.tsv")
    return path if os.path.isfile(path) else None


def load_ligand_map(input_csv: str) -> dict:
    """``job_name -> ligands`` from the unification input CSV (missing values become '')."""
    df = pd.read_csv(input_csv, usecols=["job_name", "ligands"], dtype=str, keep_default_na=False)
    return dict(zip(df["job_name"], df["ligands"]))


def collect_ubs(root: str, threshold="ST_100", input_csv: str | None = None) -> pd.DataFrame:
    """
    Walk ``root/<ACC>/`` and return one row per UBS of the requested threshold table.

    Parameters
    ----------
    root : str
        Output root of ``run_ubs_unification.py``.
    threshold : str or float
        ``ST_100`` (default), ``100``, ``1.0``, ``0.7`` ... or ``LJ`` for each
        accession's large-jump table.
    input_csv : str or None
        The unification input CSV; when given, a ``ligands`` column is added.
    """
    label = normalize_threshold_label(threshold)
    ligand_map = load_ligand_map(input_csv) if input_csv else None

    rows = []
    n_dirs = n_hit = 0
    for name in sorted(os.listdir(root)):
        uniprot_dir = os.path.join(root, name)
        if not os.path.isdir(uniprot_dir):
            continue
        n_dirs += 1
        path = _table_path(uniprot_dir, label)
        if path is None:
            continue
        n_hit += 1
        table_label = os.path.splitext(os.path.basename(path))[0]  # e.g. ST_100 or ST_LJ072
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            logger.warning("%s is empty", path)
            continue
        for r in df.itertuples(index=False):
            members = [str(m) for m in _parse_list(r.entry_ids)]
            res = residue_indices(_parse_list(r.total_residues))
            row = {
                "site_uid": f"{name}:{table_label}:{int(r.ubs_id)}",
                "uniprot": name,
                "residues": "+".join(map(str, res)),
                "members": ";".join(members),
                "n_members": int(r.num_entries) if str(r.num_entries).strip() else len(members),
                "multimeric_status_all": str(r.multimeric_status_all).strip(),
            }
            if ligand_map is not None:
                row["ligands"] = ";".join(ligand_map.get(m, "") for m in members)
            rows.append(row)

    logger.info("%s: %d UniProt directories, %d with a %s table, %d sites",
                root, n_dirs, n_hit, label, len(rows))
    columns = OUTPUT_COLUMNS + (["ligands"] if ligand_map is not None else [])
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        key = out["site_uid"].str.rsplit(":", n=1).str[1].astype(int)
        out = out.assign(_k=key).sort_values(["uniprot", "_k"], kind="stable").drop(columns="_k")
        out = out.reset_index(drop=True)
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="collect_ubs.py",
        description="Collect per-UniProt UBS tables of one threshold into a single TSV.",
    )
    p.add_argument("root", metavar="DIR", help="output root of run_ubs_unification.py")
    p.add_argument("--threshold", default="ST_100", metavar="LABEL",
                   help="table to collect: ST_100 (default), ST_070, 0.7, 70, or LJ for the large-jump tables")
    p.add_argument("--out", default=None, metavar="TSV", help="output TSV (default: stdout)")
    p.add_argument("--input", default=None, metavar="CSV",
                   help="the unification input CSV; adds a 'ligands' column (members' ligands, ';'-joined)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = collect_ubs(args.root, threshold=args.threshold, input_csv=args.input)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.out, sep="\t", index=False, lineterminator="\n")
        logger.info("wrote %d sites to %s", len(df), args.out)
    else:
        df.to_csv(sys.stdout, sep="\t", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
