# -*- coding: utf-8 -*-
"""
fpocket adapter (recommended predictor #1) -> raw pockets table.
================================================================

SCAFFOLD. Runs fpocket on each AFDB receptor and parses its output into a
per-receptor "raw pockets" table (afdb_filename, pocket_rank, center_x/y/z,
score, pocket_residues), which `normalize_predictions.py` then turns into the
normalized `predictions.csv` for `score/compute_lbs_success.py`.

WHY fpocket:  unsupervised Voronoi alpha-sphere geometry (MIT, github.com/Discngine/
fpocket) — a different algorithm class from P2Rank's supervised model, so a shared
holo>apo gradient across the two is a strong "not P2Rank-specific" statement.
mmCIF-native since v4 (gunzip the .cif.gz), seconds/structure, ranked by
Pocket/Druggability Score, and emits both pocket centres and lining residues.

RUNTIME NOTE (Windows):  fpocket has no native Windows build — run under WSL2
(Ubuntu) or a conda-forge `fpocket` env / Linux box. This script therefore ships
the PARSER as tested logic and the fpocket INVOCATION as an optional subprocess
call (`--run`), so on Windows you can point `--fpocket_out_dir` at output produced
under WSL and only parse here.

fpocket output layout (per input `<name>.pdb`):
  <name>_out/
    <name>_info.txt                 # per-pocket descriptors (Score, Druggability Score, ...)
    <name>_out.pdb                  # STP alpha-sphere dummy atoms per pocket
    pockets/pocket<k>_vert.pqr      # alpha-sphere centres (+ radii) for pocket k
    pockets/pocket<k>_atm.pdb       # lining protein atoms of pocket k

Pocket CENTRE = centroid of that pocket's alpha-sphere vertices (pocket<k>_vert.pqr).
Pocket RANK   = fpocket's own pocket order (pocket1 = rank 1); RESIDUES parsed from
pocket<k>_atm.pdb (chain_resi tokens, e.g. "A_50").

Status: PARSER implemented (pure text parsing; no fpocket needed to parse). The
`--run` path (subprocess to the `fpocket` binary + cif->pdb handling) is a
documented TODO — validate on a handful of receptors before the pilot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


def parse_vert_pqr_centroid(pqr_path: Path) -> tuple[float, float, float] | None:
    """Centroid of the alpha-sphere vertices in a pocket<k>_vert.pqr file."""
    xs, ys, zs = [], [], []
    with pqr_path.open() as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                # PQR columns are whitespace-tolerant; take fields -5..-3 as x,y,z
                parts = line.split()
                try:
                    x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
                except (ValueError, IndexError):
                    # fixed-width fallback (PDB-style columns 31-54)
                    try:
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    except ValueError:
                        continue
                xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))


def parse_atm_residues(atm_pdb: Path) -> str:
    """Lining residues of a pocket as 'A_50 A_51 ...' from pocket<k>_atm.pdb."""
    seen = []
    seenset = set()
    with atm_pdb.open() as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip() or "A"
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                tok = f"{chain}_{resnum}"
                if tok not in seenset:
                    seenset.add(tok); seen.append(tok)
    return " ".join(seen)


def parse_info_scores(info_txt: Path) -> dict:
    """{pocket_index: {'score': float, 'druggability': float}} from *_info.txt."""
    out: dict[int, dict] = {}
    cur = None
    if not info_txt.is_file():
        return out
    for line in info_txt.read_text(errors="ignore").splitlines():
        m = re.match(r"\s*Pocket\s+(\d+)\s*:", line)
        if m:
            cur = int(m.group(1)); out[cur] = {}
            continue
        if cur is not None:
            ms = re.search(r"Score\s*:\s*([-\d.eE]+)", line)
            if ms and "Druggability" not in line:
                out[cur]["score"] = float(ms.group(1))
            md = re.search(r"Druggability Score\s*:\s*([-\d.eE]+)", line)
            if md:
                out[cur]["druggability"] = float(md.group(1))
    return out


def parse_one_receptor(out_dir: Path, afdb_filename: str, rank_by: str = "fpocket") -> pd.DataFrame:
    """Parse a single `<name>_out/` directory into raw-pocket rows."""
    pockets_dir = out_dir / "pockets"
    info = parse_info_scores(next(out_dir.glob("*_info.txt"), out_dir / "missing_info.txt"))
    rows = []
    if pockets_dir.is_dir():
        for vert in sorted(pockets_dir.glob("pocket*_vert.pqr"),
                           key=lambda p: int(re.search(r"pocket(\d+)_", p.name).group(1))):
            k = int(re.search(r"pocket(\d+)_", vert.name).group(1))
            centroid = parse_vert_pqr_centroid(vert)
            if centroid is None:
                continue
            atm = pockets_dir / f"pocket{k}_atm.pdb"
            residues = parse_atm_residues(atm) if atm.is_file() else ""
            sc = info.get(k, {})
            rows.append({
                "afdb_filename": afdb_filename,
                "pocket_rank": k,  # fpocket rank1 = best (Pocket Score-ranked)
                "center_x": centroid[0], "center_y": centroid[1], "center_z": centroid[2],
                "score": sc.get("druggability", sc.get("score", np.nan)),
                "pocket_residues": residues,
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse fpocket outputs into a raw pockets table.")
    p.add_argument("--fpocket_out_root", type=Path, required=True,
                   help="Directory containing per-receptor `<name>_out/` folders (produced under WSL/Linux).")
    p.add_argument("--receptors_csv", type=Path, default=cfg.LBS_RECEPTORS_CSV,
                   help="receptors.csv from build_lbs_inputs.py (maps <name>_out -> afdb_filename).")
    p.add_argument("--out_csv", type=Path, default=cfg.LBS_DIR / "fpocket" / "fpocket_raw.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    receptors = pd.read_csv(args.receptors_csv)
    frames = []
    for out_dir in sorted(args.fpocket_out_root.glob("*_out")):
        # `<name>_out` where <name> is the receptor file stem (strip .cif.gz/.pdb)
        stem = out_dir.name[:-4]
        matches = receptors[receptors["afdb_filename"].astype(str).str.startswith(stem.split(".")[0])]
        afdb_filename = matches["afdb_filename"].iloc[0] if len(matches) else f"{stem}.cif.gz"
        frames.append(parse_one_receptor(out_dir, afdb_filename))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["afdb_filename", "pocket_rank", "center_x", "center_y", "center_z", "score", "pocket_residues"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_csv, index=False)
    print(f"[SAVE] {args.out_csv}  ({len(raw)} pockets over {raw['afdb_filename'].nunique()} receptors)")
    print("[NEXT] normalize_predictions.py --raw_pockets "
          f"{args.out_csv} --tool fpocket --out_csv {args.out_csv.parent/'predictions.csv'}")


if __name__ == "__main__":
    main()
