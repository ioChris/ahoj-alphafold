# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:24:06 2025

@author: Christos P. Feidakis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute union/intersection residue centers for each UBS from AlphaFold DB models.

- Input CSV must contain:
    - 'site_uniprots' (UniProt accession for the UBS' AF model)
    - 'union_residues' (list of positions; separators: , ; space | supports ranges like 45-53)
    - 'intersection_residues' (same as above)

- Output CSV gets appended columns:
    union_center_x, union_center_y, union_center_z,
    intersection_center_x, intersection_center_y, intersection_center_z

Usage:
    python compute_ubs_centers.py \
        --input_csv /path/to/ubs_filtered_annotated.csv \
        --afdb_dir /path/to/AFDB_DOWNLOAD_DIR \
        --output_dir /path/to/OUTPUT_DIR \
        [--center_mode geometric|com] \
        [--atom_subset all|heavy|CA] \
        [--uniprot_col site_uniprots] \
        [--union_col union_residues] \
        [--inter_col intersection_residues] \
        [--fail_policy keep|nan]
"""




import os
from pathlib import Path
import argparse
import ast
import glob
import math
from typing import Iterable, List, Optional, Tuple, Union, Dict
import json, sys, datetime

import gemmi
import numpy as np
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


def parse_args():
    p = argparse.ArgumentParser(description="Compute UBS centers (union/intersection) from AF-DB models.")
    p.add_argument("--input_csv",   type=Path,      default=None)
    p.add_argument("--afdb_dir",    type=Path,      default=None,           help="Directory with AF-DB .cif.gz files")
    p.add_argument("--output_dir",  type=Path,      default=None)
    p.add_argument("--center_mode", choices=["geometric", "com"],   default="geometric", help="geometric mean (default) or center-of-mass (com)")
    p.add_argument("--atom_subset", choices=["all", "heavy", "CA"], default="heavy", help="Atoms to use for center calculation")

    p.add_argument("--uniprot_col",                 default="site_uniprots")
    p.add_argument("--union_col",                   default="union_residues")
    p.add_argument("--inter_col",                   default="intersection_residues")
    p.add_argument("--fail_policy", choices=["keep", "nan"], default="nan", help="If residues missing -> 'keep' (skip that set) or fill NaNs")
    p.add_argument("--outfile_name",                default=None,           help="Optional explicit output filename; default adds _with_centers to input name")

    # Debug controls
    p.add_argument("--debug",       action="store_true",                    help="Verbose per-row debug prints")
    p.add_argument("--debug_uniprots",              default="",             help="Comma-separated UniProts to debug (exact, include isoform dashes)")
    p.add_argument("--debug_limit_rows", type=int,  default=5,              help="Max rows to debug per UniProt (0 = no limit)")
    p.add_argument("--debug_unp_limit",  type=int,  default=0,              help="Max UniProts to process for testing purposes (0 = no limit)")
    return p.parse_args()


# ----------------------- residue list parsing -----------------------

def get_cwd_prefix(levels=3):
    # levels=3 -> root + 2 dirs; levels=2 -> root + 1 dir, etc.
    return str(Path(*Path.cwd().parts[:levels]))


def _safe_literal_list(val: str) -> Optional[List]:
    try:
        obj = ast.literal_eval(val)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return None


def parse_residues_for_exact_uniprot(
    val: Union[str, List, float, int],
    current_uniprot: str
) -> List[int]:
    """
    Accepts:
      - Python-list-like string: "['O60882_182','O60882_183', ...]" (or with isoforms e.g. 'O60885-2_89')
      - Actual list: ['O60882_182', ...]
      - Fallback plain formats: "182 183 184", "182-186, 206, 208-209"

    Returns sorted unique integer residue positions that match *exactly* current_uniprot (incl. isoform dashes).
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []

    # If it's already a list, use that; else try to parse a list-like string.
    if isinstance(val, list):
        items = val
    else:
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        maybe_list = _safe_literal_list(s)
        if maybe_list is not None:
            items = maybe_list
        else:
            # Fallback: plain numeric tokens/ranges without UniProt prefix
            return _parse_plain_positions(s)

    out: List[int] = []
    cu = current_uniprot.strip()
    for it in items:
        if it is None:
            continue
        tok = str(it).strip()
        if not tok:
            continue
        if "_" in tok:
            left, right = tok.rsplit("_", 1)  # exact match, keep isoform dashes
            if left != cu:
                continue
            try:
                out.append(int(right))
            except ValueError:
                continue
        else:
            # Accept bare numbers too
            try:
                out.append(int(tok))
            except ValueError:
                continue

    return sorted(set(out))


def _parse_plain_positions(s: str) -> List[int]:
    for sep in [",", ";", "|"]:
        s = s.replace(sep, " ")
    toks = [t.strip() for t in s.split() if t.strip()]
    out: List[int] = []
    for tok in toks:
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
                lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                out.extend(range(lo, hi + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(tok))
            except ValueError:
                continue
    return sorted(set(out))


# ----------------------- AF model handling (gemmi) -----------------------

class AFModel:
    """
    Lightweight accessor for an AlphaFold monomer model (single polymer chain).
    Maps UniProt positions (1-based) -> atom coordinates (optionally filtered).
    """
    def __init__(self, cif_path: Path, atom_subset: str = "heavy"):
        self.cif_path = cif_path
        self.atom_subset = atom_subset
        self.struct = gemmi.read_structure(str(cif_path))  # reads .gz transparently
        self.model = self.struct[0]  # first model
        self.chain = self._pick_protein_chain(self.model)
        if self.chain is None:
            raise RuntimeError(f"No protein chain found in {self.cif_path}")
        self.pos_to_atoms: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        self._index_chain()

    @staticmethod
    def _pick_protein_chain(model: gemmi.Model) -> Optional[gemmi.Chain]:
        # Prefer chain "A", else first polymer chain
        for ch in model:
            if ch.name == "A":
                return ch
        for ch in model:
            if ch.is_polymer():
                return ch
        return None

    @staticmethod
    def _is_heavy(atom: gemmi.Atom) -> bool:
        try:
            el = gemmi.Element(atom.element)
        except Exception:
            return True
        return el != gemmi.Element("H")

    def _index_chain(self):
        #print("Trying to index chain")
        for res in self.chain:
            #if res.het_flag != " ":
            #    continue

            # Try auth_seq_id first; fall back to label_seq_id (common in AF-DB)
            pos = res.seqid.num     # integer position (AF models use UniProt-like numbering)
            #print(f"{pos} residue positions found with res.seqid.num")

            if pos is None or pos == 0:
                try:
                    pos = int(res.get_label_seq_id())  # gemmi exposes label_seq_id here
                    print(f"label_seq_id={pos}")
                    if pos is None or pos == 0:
                        print('No positions found with get_label_seq_id')
                except Exception:
                    pos = None

            if pos is None:
                continue

            bucket: List[Tuple[np.ndarray, float]] = []
            for atom in res:
                name = atom.name.strip()
                if self.atom_subset == "CA" and name != "CA":
                    continue
                if self.atom_subset == "heavy" and not self._is_heavy(atom):
                    continue
                coord = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                if np.any(~np.isfinite(coord)):
                    continue
                # simple weights: 1.0 for geometric; for COM we store element masses
                try:
                    el = gemmi.Element(atom.element)
                    mass = float(el.weight) if el and el.weight > 0 else 1.0
                except Exception:
                    mass = 1.0
                bucket.append((coord, mass))
            if bucket:
                self.pos_to_atoms.setdefault(pos, []).extend(bucket)

    def get_center(self, positions: Iterable[int], mode: str = "geometric") -> Optional[np.ndarray]:
        coords: List[np.ndarray] = []
        weights: List[float] = []

        for pos in positions:
            atoms = self.pos_to_atoms.get(pos)
            if not atoms:
                continue
            for c, m in atoms:
                coords.append(c)
                weights.append(m)

        if not coords:
            return None

        arr = np.vstack(coords)
        if mode == "geometric":
            return arr.mean(axis=0)

        # center-of-mass
        w = np.asarray(weights, dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            return arr.mean(axis=0)
        return (arr * w[:, None]).sum(axis=0) / w_sum


# ----------------------- workflow -----------------------

def find_af_cif_for_uniprot(uniprot: str, afdb_dir: Path) -> Optional[Path]:
    # Do NOT normalize; include isoform dashes in the pattern
    pattern = str(afdb_dir / f"AF-{uniprot}-F1-model_v*.cif.gz")
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    def _verkey(p: str) -> Tuple[int, str]:
        base = os.path.basename(p)
        try:
            v = int(base.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            v = -1
        return (v, base)
    hits.sort(key=_verkey, reverse=True)
    return Path(hits[0])


def compute_centers_for_csv(
    df: pd.DataFrame,
    afdb_dir: Path,
    uniprot_col: str,
    union_col: str,
    inter_col: str,
    atom_subset: str,
    center_mode: str,
    fail_policy: str = "nan",
    debug: bool = False,
    debug_uniprots: Optional[set] = None,
    debug_limit_rows: int = 5,
    unp_limit: int | None = None,
) -> pd.DataFrame:

    # Prepare output columns (chain_resi + coordinates)
    for col in [
        "union_chain_resi",
        "union_center_x", "union_center_y", "union_center_z",
        "intersection_chain_resi",
        "intersection_center_x", "intersection_center_y", "intersection_center_z", "af_model_len", "n_ubs_total"
    ]:
        if col not in df.columns:
            # Text columns (chain_resi) → empty string; numeric coords → NaN
            df[col] = "" if col.endswith("chain_resi") else np.nan

    # New per-row metadata columns
    for col in ["afdb_filename"]:
        if col not in df.columns:
            df[col] = ""


    cache: Dict[str, Optional[AFModel]] = {}
    cache_fname: Dict[str, str] = {}

    # Optionally cap number of UniProts to process
    if unp_limit:
        if unp_limit > 0:
            order  = pd.unique(df[uniprot_col].astype(str))   # preserves first appearance order
            keep   = set(order[:unp_limit])
            df_sub = df[df[uniprot_col].astype(str).isin(keep)]
            groups = df_sub.groupby(uniprot_col).groups
        else:
            groups = df.groupby(uniprot_col).groups
    else:
        groups = df.groupby(uniprot_col).groups

    for uniprot, sub_ix in groups.items():
        u = str(uniprot).strip()
        if not u:
            continue

        # Load/cache AF model for the *exact* UniProt (isoform dashes allowed)
        if u not in cache:
            cif = find_af_cif_for_uniprot(u, afdb_dir)
            if cif is None:
                print(f"[MISS] No AF model for {u}")
                cache[u] = None
                cache_fname[u] = ""
            else:
                try:
                    cache[u] = AFModel(cif, atom_subset=atom_subset)
                    cache_fname[u] = Path(cif).name
                    chain_name = cache[u].chain.name if cache[u] else "?"
                    pos_keys = sorted(cache[u].pos_to_atoms.keys())
                    if pos_keys:
                        print(f"[LOAD] {u}: file='{cif.name}', chain='{chain_name}', "
                              f"pos_range={pos_keys[0]}..{pos_keys[-1]} (n={len(pos_keys)})")
                    else:
                        print(f"[LOAD] {u}: file='{cif.name}', chain='{chain_name}', no positions indexed")
                except Exception as e:
                    print(f"[ERR ] Failed to load {u}: {e}")
                    cache[u] = None
                    cache_fname[u] = ""

        model = cache[u]

        # ✅ Store AF model length for all rows of this UniProt
        if model is not None:
            df.loc[sub_ix, "af_model_len"] = len(model.pos_to_atoms)
            # Store per-UniProt metadata on all its rows
            df.loc[sub_ix, "n_ubs_total"]  = len(sub_ix)
            df.loc[sub_ix, "afdb_filename"] = cache_fname.get(u, "")

        n_set_u = 0
        n_set_i = 0
        n_rows  = len(sub_ix)

        # Debug control for this UniProt
        do_debug_u = debug or (debug_uniprots and u in debug_uniprots)
        row_debug_count = 0

        for ix in sub_ix:
            if model is None:
                continue

            union_list = parse_residues_for_exact_uniprot(df.at[ix, union_col], u) if union_col in df.columns else []
            inter_list = parse_residues_for_exact_uniprot(df.at[ix, inter_col], u) if inter_col in df.columns else []

            chain_name = model.chain.name if (model and model.chain) else "A"

            # Keep only positions that actually exist in the model
            avail_positions = model.pos_to_atoms.keys()
            union_list_in_model = [p for p in union_list if p in avail_positions]
            inter_list_in_model = [p for p in inter_list if p in avail_positions]

            # Build chain_resi strings (use whatever separator you prefer: "|" or " ")
            union_chain_resi_str = " ".join(f"{chain_name}_{p}" for p in union_list_in_model)
            inter_chain_resi_str = " ".join(f"{chain_name}_{p}" for p in inter_list_in_model)

            # Assign
            df.at[ix, "union_chain_resi"] = union_chain_resi_str
            df.at[ix, "intersection_chain_resi"] = inter_chain_resi_str

            n_u_have = sum(1 for p in union_list if p in avail_positions)
            n_i_have = sum(1 for p in inter_list if p in avail_positions)

            # Compute centers as before, but use the *_in_model lists for safety
            center_u = model.get_center(union_list_in_model, mode=center_mode) if union_list_in_model else None
            center_i = model.get_center(inter_list_in_model, mode=center_mode) if inter_list_in_model else None

            if center_u is not None:
                df.at[ix, "union_center_x"] = round(float(center_u[0]), 3)
                df.at[ix, "union_center_y"] = round(float(center_u[1]), 3)
                df.at[ix, "union_center_z"] = round(float(center_u[2]), 3)
                n_set_u += 1
            if center_i is not None:
                df.at[ix, "intersection_center_x"] = round(float(center_i[0]), 3)
                df.at[ix, "intersection_center_y"] = round(float(center_i[1]), 3)
                df.at[ix, "intersection_center_z"] = round(float(center_i[2]), 3)
                n_set_i += 1

            if do_debug_u and (debug_limit_rows == 0 or row_debug_count < debug_limit_rows):
                row_debug_count += 1
                chain_name = model.chain.name if model and model.chain else "?"
                # Human-readable "selection statement"
                sel_stmt_u = (f"chain {chain_name}; residues {','.join(map(str, union_list))}; "
                              f"atoms={atom_subset}; mode={center_mode}")
                sel_stmt_i = (f"chain {chain_name}; residues {','.join(map(str, inter_list))}; "
                              f"atoms={atom_subset}; mode={center_mode}")

                print(f"[DBG ] {u} row={ix} UNION idx={union_list} (n={len(union_list)}; in_model={n_u_have})")
                print(f"[DBG ] selection (union): {sel_stmt_u}")
                print(f"[DBG ] {u} row={ix} INTER  idx={inter_list} (n={len(inter_list)}; in_model={n_i_have})")
                print(f"[DBG ] selection (inter): {sel_stmt_i}")

                if not union_list and df.at[ix, union_col]:
                    print(f"[WARN] union_list empty; raw='{df.at[ix, union_col]}'")
                if not inter_list and df.at[ix, inter_col]:
                    print(f"[WARN] inter_list empty; raw='{df.at[ix, inter_col]}'")

        print(f"[AF  ] {u}: rows={n_rows}, union_centers_set={n_set_u}, intersection_centers_set={n_set_i}")

    return df


def main():
    args = parse_args()

    # Set final Paths & Ensure Path types (covers any indirect string injection)
    # LEGACY prep util (pre-afdb_dataset pipeline): anchor under configured roots.
    INPUT_CSV_PATH      = str(cfg.WORK_DIR / "afdb_ubs_sets" / "sub_df_ST_LJ_ubs_filtered_annotated.csv")
    OUTPUT_DIR          = str(cfg.WORK_DIR / "afdb_ubs_sets" / "ubs_centers")
    AFDB_DOWNLOAD_DIR   = str(cfg.AFDB_DIR)

    args.input_csv  = args.input_csv  or Path(INPUT_CSV_PATH)
    args.afdb_dir   = args.afdb_dir   or Path(AFDB_DOWNLOAD_DIR)
    args.output_dir = args.output_dir or Path(OUTPUT_DIR)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save resolved run parameters ---
    params = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "script": Path(sys.argv[0]).name,
        "input_csv": str(args.input_csv),
        "afdb_dir": str(args.afdb_dir),
        "output_dir": str(args.output_dir),
        "outfile_name": args.outfile_name or f"{Path(args.input_csv).stem}_with_centers_nUNP{str(args.debug_unp_limit)}.csv",
        "center_mode": args.center_mode,
        "atom_subset": args.atom_subset,
        "uniprot_col": args.uniprot_col,
        "union_col": args.union_col,
        "inter_col": args.inter_col,
        "fail_policy": args.fail_policy,
        "debug": bool(args.debug),
        "debug_uniprots": args.debug_uniprots,
        "debug_limit_rows": int(args.debug_limit_rows),
        "debug_unp_limit": int(args.debug_unp_limit),
    }
    params_path = args.output_dir / f"{Path(args.input_csv).stem}_with_centers_run_params.json"
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    print(f"[PARAMS] Saved run parameters to {params_path}")

    # Load the CSV
    df = pd.read_csv(args.input_csv)
    if args.uniprot_col not in df.columns:
        raise SystemExit(f"Missing required column '{args.uniprot_col}' in input CSV.")

    debug_uniprots = set([u.strip() for u in args.debug_uniprots.split(",") if u.strip()]) if args.debug_uniprots else None

    # Do the processing
    out_df = compute_centers_for_csv(
        df=df,
        afdb_dir=args.afdb_dir,
        uniprot_col=args.uniprot_col,
        union_col=args.union_col,
        inter_col=args.inter_col,
        atom_subset=args.atom_subset,
        center_mode=args.center_mode,
        fail_policy=args.fail_policy,
        debug=args.debug,
        debug_uniprots=debug_uniprots,
        debug_limit_rows=args.debug_limit_rows,
        unp_limit=args.debug_unp_limit,
    )

    if args.outfile_name:
        out_path = args.output_dir / args.outfile_name
    else:
        stem = Path(args.input_csv).stem
        out_path = args.output_dir / f"{stem}_with_centers_nUNP{str(args.debug_unp_limit)}.csv"

    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()




