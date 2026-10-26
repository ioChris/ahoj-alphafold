# -*- coding: utf-8 -*-
"""
Append AFDB residue selections, centers, and radius of gyration to an AFDB
dataset CSV built from ST_100 UBS rows.

Expected input columns:
  - site_uniprot
  - union_residues
  - intersection_residues

Expected residue format:
  - UniProt-coordinate tokens such as "P12345_50"

Appended output columns:
  - union_chain_resi / intersection_chain_resi
  - union_center_geom_x/y/z
  - intersection_center_geom_x/y/z
  - union_center_com_x/y/z
  - intersection_center_com_x/y/z
  - union_rg
  - intersection_rg
  - union_n_atoms / intersection_n_atoms
  - union_n_residues_found / intersection_n_residues_found
  - af_model_len
  - afdb_filename
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import glob
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

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


DEFAULT_INPUT = cfg.AFDB_ST100_DATASET_CSV
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
DEFAULT_AFDB_DIR = cfg.AFDB_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute AFDB geometry for pocket residue sets.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--afdb_dir", type=Path, default=DEFAULT_AFDB_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--atom_subset", choices=["all", "heavy", "CA"], default="heavy")
    p.add_argument("--outfile_name", default=None)
    p.add_argument("--debug_unp_limit", type=int, default=0)
    return p.parse_args()


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
    current_uniprot: str,
) -> List[int]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []

    if isinstance(val, list):
        items = val
    else:
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        maybe_list = _safe_literal_list(s)
        items = maybe_list if maybe_list is not None else []
        if maybe_list is None:
            return []

    out: List[int] = []
    current = current_uniprot.strip()
    for item in items:
        tok = str(item).strip()
        if not tok:
            continue
        if "_" in tok:
            left, right = tok.rsplit("_", 1)
            if left != current:
                continue
            try:
                out.append(int(right))
            except ValueError:
                continue
        else:
            try:
                out.append(int(tok))
            except ValueError:
                continue
    return sorted(set(out))


class AFModel:
    def __init__(self, cif_path: Path, atom_subset: str = "heavy"):
        self.cif_path = cif_path
        self.atom_subset = atom_subset
        self.struct = gemmi.read_structure(str(cif_path))
        self.model = self.struct[0]
        self.chain = self._pick_protein_chain(self.model)
        if self.chain is None:
            raise RuntimeError(f"No protein chain found in {self.cif_path}")
        self.pos_to_atoms: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        self._index_chain()

    @staticmethod
    def _pick_protein_chain(model: gemmi.Model) -> Optional[gemmi.Chain]:
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

    def _index_chain(self) -> None:
        for res in self.chain:
            pos = res.seqid.num
            if pos is None or pos == 0:
                try:
                    pos = int(res.get_label_seq_id())
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
                try:
                    el = gemmi.Element(atom.element)
                    mass = float(el.weight) if el and el.weight > 0 else 1.0
                except Exception:
                    mass = 1.0
                bucket.append((coord, mass))

            if bucket:
                self.pos_to_atoms.setdefault(pos, []).extend(bucket)

    def get_atoms_for_positions(self, positions: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        coords: List[np.ndarray] = []
        masses: List[float] = []
        for pos in positions:
            atoms = self.pos_to_atoms.get(pos)
            if not atoms:
                continue
            for coord, mass in atoms:
                coords.append(coord)
                masses.append(mass)
        if not coords:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(coords), np.asarray(masses, dtype=float)


def geometric_center(coords: np.ndarray) -> Optional[np.ndarray]:
    return coords.mean(axis=0) if len(coords) else None


def center_of_mass(coords: np.ndarray, masses: np.ndarray) -> Optional[np.ndarray]:
    if not len(coords):
        return None
    total = masses.sum()
    if total <= 0:
        return geometric_center(coords)
    return (coords * masses[:, None]).sum(axis=0) / total


def radius_of_gyration(coords: np.ndarray, masses: np.ndarray) -> Optional[float]:
    if not len(coords):
        return None
    com = center_of_mass(coords, masses)
    if com is None:
        return None
    sq = ((coords - com) ** 2).sum(axis=1)
    total = masses.sum()
    if total <= 0:
        return float(np.sqrt(sq.mean()))
    return float(np.sqrt((masses * sq).sum() / total))


def find_af_cif_for_uniprot(uniprot: str, afdb_dir: Path) -> Optional[Path]:
    pattern = str(afdb_dir / f"AF-{uniprot}-F1-model_v*.cif.gz")
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None

    def _verkey(path_str: str) -> Tuple[int, str]:
        base = Path(path_str).name
        try:
            version = int(base.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            version = -1
        return (version, base)

    hits.sort(key=_verkey, reverse=True)
    return Path(hits[0])


def append_geometry(df: pd.DataFrame, afdb_dir: Path, atom_subset: str, debug_unp_limit: int = 0) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "union_chain_resi",
        "intersection_chain_resi",
        "union_center_geom_x", "union_center_geom_y", "union_center_geom_z",
        "intersection_center_geom_x", "intersection_center_geom_y", "intersection_center_geom_z",
        "union_center_com_x", "union_center_com_y", "union_center_com_z",
        "intersection_center_com_x", "intersection_center_com_y", "intersection_center_com_z",
        "union_rg", "intersection_rg",
        "union_n_atoms", "intersection_n_atoms",
        "union_n_residues_found", "intersection_n_residues_found",
        "af_model_len",
        "afdb_filename",
    ]:
        out[col] = "" if col.endswith("chain_resi") or col == "afdb_filename" else np.nan

    cache: Dict[str, Optional[AFModel]] = {}
    uniprots = list(pd.unique(out["site_uniprot"].astype(str)))
    if debug_unp_limit > 0:
        uniprots = uniprots[:debug_unp_limit]

    for uniprot in uniprots:
        rows = out.index[out["site_uniprot"].astype(str) == uniprot].tolist()
        cif_path = find_af_cif_for_uniprot(uniprot, afdb_dir)
        if cif_path is None:
            print(f"[MISS] {uniprot}")
            continue

        try:
            model = AFModel(cif_path, atom_subset=atom_subset)
        except Exception as exc:
            print(f"[ERR ] {uniprot}: {exc}")
            continue

        cache[uniprot] = model
        chain_name = model.chain.name if model.chain is not None else "A"
        out.loc[rows, "af_model_len"] = len(model.pos_to_atoms)
        out.loc[rows, "afdb_filename"] = cif_path.name

        for ix in rows:
            for prefix, col in (
                ("union", "union_residues"),
                ("intersection", "intersection_residues"),
            ):
                positions = parse_residues_for_exact_uniprot(out.at[ix, col], uniprot)
                positions_found = [p for p in positions if p in model.pos_to_atoms]
                coords, masses = model.get_atoms_for_positions(positions_found)

                out.at[ix, f"{prefix}_chain_resi"] = " ".join(f"{chain_name}_{p}" for p in positions_found)
                out.at[ix, f"{prefix}_n_residues_found"] = len(positions_found)
                out.at[ix, f"{prefix}_n_atoms"] = len(coords)

                gcenter = geometric_center(coords)
                if gcenter is not None:
                    out.at[ix, f"{prefix}_center_geom_x"] = round(float(gcenter[0]), 3)
                    out.at[ix, f"{prefix}_center_geom_y"] = round(float(gcenter[1]), 3)
                    out.at[ix, f"{prefix}_center_geom_z"] = round(float(gcenter[2]), 3)

                com = center_of_mass(coords, masses)
                if com is not None:
                    out.at[ix, f"{prefix}_center_com_x"] = round(float(com[0]), 3)
                    out.at[ix, f"{prefix}_center_com_y"] = round(float(com[1]), 3)
                    out.at[ix, f"{prefix}_center_com_z"] = round(float(com[2]), 3)

                rg = radius_of_gyration(coords, masses)
                if rg is not None:
                    out.at[ix, f"{prefix}_rg"] = round(float(rg), 3)

        print(f"[AF  ] {uniprot}: rows={len(rows)} file={cif_path.name}")

    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    out_df = append_geometry(
        df=df,
        afdb_dir=args.afdb_dir,
        atom_subset=args.atom_subset,
        debug_unp_limit=args.debug_unp_limit,
    )

    out_name = args.outfile_name or f"{args.input_csv.stem}_with_geometry.csv"
    out_csv = args.output_dir / out_name
    out_json = args.output_dir / f"{Path(out_name).stem}_run_params.json"

    out_df.to_csv(out_csv, index=False)
    run_meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_csv": str(args.input_csv),
        "afdb_dir": str(args.afdb_dir),
        "output_csv": str(out_csv),
        "atom_subset": args.atom_subset,
        "debug_unp_limit": args.debug_unp_limit,
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2)

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_json}")


if __name__ == "__main__":
    main()
