# -*- coding: utf-8 -*-
"""
Precompute the per-pocket GROUND-TRUTH ATOM store for the LBS-generality metric.
===============================================================================

For every ST_100 pocket, extract the heavy-atom coordinates of its union
binding-site residues (`union_chain_resi`, e.g. "A_50 A_51 ...") from the AF
model, and write a compact ragged store consumed by
`score/compute_lbs_success.py`. The DCA metric is centre-to-nearest-ground-truth-
atom, which reproduces P2Rank's own `closest_pocket_dist` to < 1e-4 Angstrom, so
this store is the shared, tool-independent yardstick for every predictor.

Run ONCE per classification round, in an env with gemmi (the repo's analysis /
py311 env -- NOT base conda). Mirrors the AF-model reading in
`04_p2rank/dataset/compute_afdb_pocket_geometry.py`.

Output (`--out_npz`, default WORK_DIR/lbs_generality/ground_truth_atoms.npz):
  site_uids (str[N])  starts (int[N])  counts (int[N])  coords (float32[M,3])
where site s occupies coords[starts[s] : starts[s]+counts[s]].

WHY HEAVY ATOMS (and how to match native P2Rank exactly).  The heavy-atom store
reproduces P2Rank's `closest_pocket_dist` exactly and is a touch STRICTER than
P2Rank's native `dca4rank`, because the P2Rank eval also added the site's SAS
surface points to the ground truth (`site_eval_sas_pts_as_atoms = true`; confirmed
in the eval `params.txt`, `ligSize = 97.2 atoms/site` ~= our 98.1). This is fine and
defensible for the cross-tool panel (one yardstick for every tool, gradient
preserved). TODO OPTION `--sas_expand`: to reproduce native P2Rank *exactly*, append
solvent-accessible surface points around the union residues (freesasa / a probe
sampler) to each site's atom set before saving; the scorer then needs no change.

    python build_ground_truth_atoms.py            # full ST_100 (~82k pockets)
    python build_ground_truth_atoms.py --limit 200  # smoke test
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

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

import gemmi  # analysis / py311 env only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the per-pocket ground-truth atom store for LBS scoring.")
    p.add_argument("--geometry_csv", type=Path, default=cfg.AFDB_ST100_DATASET_GEOM_CSV)
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--resi_col", default="union_chain_resi",
                   help="AF-model residue-selection column (default union_chain_resi).")
    p.add_argument("--out_npz", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Process only the first N pockets (smoke test).")
    return p.parse_args()


def parse_positions(sel: str) -> list[int]:
    """'A_50 A_51 ...' -> [50, 51, ...] (chain letter ignored; AF models are chain A)."""
    return [int(m.group(1)) for m in re.finditer(r"[A-Za-z0-9]+_(\d+)", str(sel))]


def find_af_cif(uniprot: str, afdb_dir: Path, hint: str | None) -> Path | None:
    if hint:
        p = afdb_dir / hint
        if p.is_file():
            return p
    hits = sorted(glob.glob(str(afdb_dir / f"AF-{uniprot}-F1-model_v*.cif.gz")))
    if not hits:
        return None
    def verkey(s: str):
        try:
            return int(Path(s).name.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            return -1
    hits.sort(key=verkey, reverse=True)
    return Path(hits[0])


def heavy_atoms_by_position(cif_path: Path) -> dict:
    """{seqid.num: (k,3) heavy-atom coords} for the protein chain (A, else first polymer)."""
    st = gemmi.read_structure(str(cif_path))
    model = st[0]
    chain = None
    for ch in model:
        if ch.name == "A":
            chain = ch
            break
    if chain is None:
        for ch in model:
            if ch.is_polymer():
                chain = ch
                break
    if chain is None:
        return {}
    out: dict[int, np.ndarray] = {}
    H = gemmi.Element("H")
    for res in chain:
        pos = res.seqid.num
        arr = [[a.pos.x, a.pos.y, a.pos.z] for a in res if a.element != H]
        if arr:
            out[pos] = np.asarray(arr, dtype=np.float32)
    return out


def main() -> None:
    args = parse_args()
    out_npz = args.out_npz or (cfg.WORK_DIR / "lbs_generality" / "ground_truth_atoms.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.geometry_csv, low_memory=False)
    df = df[df["threshold"] == args.threshold].drop_duplicates("site_uid").copy()
    if args.limit > 0:
        df = df.head(args.limit)
    df["site_uid"] = df["site_uid"].astype(str).str.strip()

    site_uids: list[str] = []
    starts: list[int] = []
    counts: list[int] = []
    coords_chunks: list[np.ndarray] = []
    total = 0
    model_cache: dict[str, dict] = {}
    n_miss_model = n_empty = 0

    for row in df.itertuples(index=False):
        uid = row.site_uid
        uniprot = str(row.site_uniprot)
        hint = getattr(row, "afdb_filename", None)
        cif = find_af_cif(uniprot, args.afdb_dir, hint if isinstance(hint, str) else None)
        if cif is None:
            n_miss_model += 1
            continue
        key = cif.name
        if key not in model_cache:
            try:
                model_cache[key] = heavy_atoms_by_position(cif)
            except Exception as exc:
                print(f"[ERR ] {uniprot}: {exc}")
                model_cache[key] = {}
        pos_map = model_cache[key]
        positions = parse_positions(getattr(row, args.resi_col))
        atoms = [pos_map[p] for p in positions if p in pos_map]
        if not atoms:
            n_empty += 1
            continue
        stacked = np.vstack(atoms).astype(np.float32)
        site_uids.append(uid)
        starts.append(total)
        counts.append(len(stacked))
        coords_chunks.append(stacked)
        total += len(stacked)

    coords = np.vstack(coords_chunks) if coords_chunks else np.empty((0, 3), dtype=np.float32)
    np.savez_compressed(
        out_npz,
        site_uids=np.asarray(site_uids, dtype=object),
        starts=np.asarray(starts, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        coords=coords,
    )
    print(f"[SAVE] {out_npz}")
    print(f"[STATS] sites={len(site_uids)}  atoms={total}  missing_model={n_miss_model}  empty_residues={n_empty}")


if __name__ == "__main__":
    main()
