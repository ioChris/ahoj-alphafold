# -*- coding: utf-8 -*-
"""
Build the docking handover manifest from the ST_100 AFDB geometry dataset.

This is the ADAPTER we run on our side. It converts the per-pocket geometry
table (the same one the P2Rank stage produces) into a per-(pocket x ligand)
docking task list plus receptor/ligand lists, ready to hand to a colleague who
runs the batch (AutoDock Vina / Uni-Dock). See:
  - 05_docking/docs/DOCKING_DESIGN.md      (methods summary; full design is private)
  - 05_docking/schemas/docking_manifest.schema.md  (output column contract)
  - (private) ahoj-af-project/docking/HANDOVER_GUIDE.md  (how the colleague runs it)

Ligand groups (see DOCKING_DESIGN.md §2):
  1  query holo ligands from AHoJ (rep_job / job_names)         -- MAIN
  2  AlphaFill cognate CCDs (merged AHoJ<->AlphaFill long table) -- cross-check
  3  probe library (fixed drug-like set)                         -- dockability control

Box (see DOCKING_DESIGN.md §4):
  center = union_center_com_{x,y,z}  (fallback: union_center_geom_*)
  size   = rg mode (default, CSV-only)  or  bbox mode (re-opens the AF model)
The box is per-POCKET (identical across ligands of a pocket), which is what we
want for the probe group and is fine generally.

Re-run after a new classification round: nothing here is hard-coded to a run;
just regenerate AFDB_ST100_DATASET_GEOM_CSV (+ merged AlphaFill) and re-run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # allow sibling imports
from parse_ahoj_job_ligands import build_query_ligand_table, is_dockable_ccd


DEFAULT_INPUT = cfg.AFDB_ST100_DATASET_GEOM_CSV
DEFAULT_ALPHAFILL_LONG = cfg.MERGED_AHOJ_ALPHAFILL_LONG_CSV
DEFAULT_PROBE_CSV = cfg.DOCKING_PROBE_LIBRARY_CSV
DEFAULT_OUTPUT_DIR = cfg.DOCKING_MANIFEST_DIR

# Metadata carried from the geometry table into the manifest (analysis-side; the
# runner ignores these but they let a returned results file be analyzed standalone).
CARRY_COLS = [
    "threshold", "pocket_class", "pocket_p_apo", "pocket_p_holo",
    "pocket_separation_pair", "pocket_score", "model_pocket_plddt",
    "n_apo_avg", "n_holo_avg", "n_unp_pockets",
    "union_rg", "union_n_residues_found", "af_model_len", "afdb_filename",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitize(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(token)).strip("-")


def _git_commit() -> Optional[str]:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except Exception:
        return False


def pocket_center(row: pd.Series, center_kind: str) -> Optional[Tuple[float, float, float]]:
    order = ["com", "geom"] if center_kind == "com" else ["geom", "com"]
    for kind in order:
        c = (row.get(f"union_center_{kind}_x"), row.get(f"union_center_{kind}_y"), row.get(f"union_center_{kind}_z"))
        if all(_finite(v) for v in c):
            return (float(c[0]), float(c[1]), float(c[2]))
    return None


def box_size_from_rg(rg: float, k_rg: float, ligand_pad: float,
                     min_box: float, max_box: float) -> float:
    """Isotropic box edge from pocket RoG: ``2*k_rg*rg`` (a residue-cloud diameter
    proxy) plus a ligand-tumbling allowance, clamped to [min_box, max_box]. Does
    NOT add the bbox ``padding`` on top (that double-counted and pinned most
    pockets at max_box); ``padding`` is used only by bbox mode."""
    if not _finite(rg):
        rg = 0.0
    size = 2.0 * k_rg * float(rg) + ligand_pad
    return float(min(max(size, min_box), max_box))


# --- optional bbox mode: compact gemmi loader (only used with --box_mode bbox) ---
def _parse_chain_resi(chain_resi: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for tok in str(chain_resi).split():
        tok = tok.strip()
        if not tok or "_" not in tok:
            continue
        ch, num = tok.rsplit("_", 1)
        try:
            out.append((ch, int(num)))
        except ValueError:
            continue
    return out


def pocket_bbox_size(cif_path: Path, chain_resi: str, padding: float,
                     min_box: float, max_box: float) -> Optional[Tuple[float, float, float]]:
    """Per-axis box from the heavy-atom bounding box of pocket residues."""
    try:
        import gemmi
    except Exception:
        return None
    want = _parse_chain_resi(chain_resi)
    if not want:
        return None
    want_set = {(c, n) for c, n in want}
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception:
        return None
    model = st[0]
    coords: List[Tuple[float, float, float]] = []
    for ch in model:
        for res in ch:
            key = (ch.name, res.seqid.num)
            if key not in want_set:
                continue
            for atom in res:
                el = str(atom.element.name) if hasattr(atom.element, "name") else str(atom.element)
                if el == "H":
                    continue
                p = atom.pos
                if all(math.isfinite(v) for v in (p.x, p.y, p.z)):
                    coords.append((p.x, p.y, p.z))
    if not coords:
        return None
    arr = np.asarray(coords, dtype=float)
    span = arr.max(axis=0) - arr.min(axis=0) + 2.0 * padding
    return tuple(float(min(max(s, min_box), max_box)) for s in span)


# ---------------------------------------------------------------------------
# Ligand-group builders -> long table [site_uid, ligand_group, ligand_id, ligand_ref, ligand_smiles]
# ---------------------------------------------------------------------------
def group1_ligands(df: pd.DataFrame, include_members: bool) -> pd.DataFrame:
    tbl = build_query_ligand_table(df, include_members=include_members)
    if tbl.empty:
        return pd.DataFrame(columns=["site_uid", "ligand_group", "ligand_id", "ligand_ref", "ligand_smiles"])
    tbl = tbl[["site_uid", "ligand_group", "ligand_id", "ligand_ref"]].copy()
    tbl["ligand_smiles"] = ""
    return tbl


def group2_ligands(alphafill_long: Path, keep_site_uids: set) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["site_uid", "ligand_group", "ligand_id", "ligand_ref", "ligand_smiles"])
    if not Path(alphafill_long).exists():
        print(f"[G2 ] AlphaFill long table not found ({alphafill_long}); skipping group 2.")
        return empty
    af = pd.read_csv(alphafill_long, low_memory=False)
    need = {"unit_uid", "afill_compound_id"}
    if not need.issubset(af.columns):
        print(f"[G2 ] AlphaFill long table missing {need - set(af.columns)}; skipping group 2.")
        return empty
    if "level" in af.columns:
        af = af[af["level"].astype(str).str.lower() == "ubs"]
    if "threshold" in af.columns:
        af = af[af["threshold"].astype(str) == "ST_100"]
    af = af.rename(columns={"unit_uid": "site_uid"})
    af = af[af["site_uid"].isin(keep_site_uids)].copy()
    af["ligand_id"] = af["afill_compound_id"].astype(str).str.upper().str.strip()
    af = af[af["ligand_id"].ne("") & af["ligand_id"].ne("NAN")]
    if af.empty:
        return empty
    # Rank candidate transplants per (pocket, CCD) by best (lowest) clash then local_rmsd.
    for c in ("afill_clash_score", "afill_local_rmsd"):
        if c in af.columns:
            af[c] = pd.to_numeric(af[c], errors="coerce")
        else:
            af[c] = np.nan
    ref_col = "afill_matched_job" if "afill_matched_job" in af.columns else None
    af = af.sort_values(["site_uid", "ligand_id", "afill_clash_score", "afill_local_rmsd"])
    best = af.drop_duplicates(["site_uid", "ligand_id"], keep="first")
    out = pd.DataFrame({
        "site_uid": best["site_uid"].values,
        "ligand_group": 2,
        "ligand_id": best["ligand_id"].values,
        "ligand_ref": best[ref_col].astype(str).values if ref_col else "alphafill",
        "ligand_smiles": "",
    })
    return out


def group3_ligands(probe_csv: Path, keep_site_uids: set) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["site_uid", "ligand_group", "ligand_id", "ligand_ref", "ligand_smiles"])
    if not Path(probe_csv).exists():
        print(f"[G3 ] Probe library not found ({probe_csv}); run resolve_ligands.py first or skip group 3.")
        return empty
    probes = pd.read_csv(probe_csv)
    if not {"probe_id", "smiles"}.issubset(probes.columns):
        print("[G3 ] Probe library needs columns probe_id,smiles; skipping group 3.")
        return empty
    rows = []
    for site_uid in keep_site_uids:
        for pr in probes.itertuples(index=False):
            rows.append({
                "site_uid": site_uid,
                "ligand_group": 3,
                "ligand_id": str(pr.probe_id),
                "ligand_ref": f"probe:{pr.smiles}",
                "ligand_smiles": str(pr.smiles),
            })
    return pd.DataFrame(rows, columns=["site_uid", "ligand_group", "ligand_id", "ligand_ref", "ligand_smiles"])


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def build_manifest(df: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Keep pockets with a usable center + residue selection (mirrors export_p2rank_subsets).
    df = df.copy()
    df = df[df["union_chain_resi"].fillna("").astype(str).str.strip().ne("")]
    df["_center"] = df.apply(lambda r: pocket_center(r, args.center_kind), axis=1)
    df = df[df["_center"].notna()].copy()

    if args.pilot_site_uids:
        keep = set(pd.read_csv(args.pilot_site_uids)["site_uid"].astype(str))
        df = df[df["site_uid"].astype(str).isin(keep)].copy()
        print(f"[PILOT] restricted to {len(df)} pockets from {args.pilot_site_uids}")

    keep_site_uids = set(df["site_uid"].astype(str))

    # --- per-pocket box geometry (one box per pocket, shared across its ligands) ---
    box_rows = []
    for idx in df.index:
        row = df.loc[idx]
        site_uid = str(row["site_uid"])
        center = row["_center"]
        rg = row.get("union_rg", np.nan)
        if args.box_mode == "bbox":
            cif = cfg.AFDB_DIR / str(row.get("afdb_filename", "")) if str(row.get("afdb_filename", "")) else None
            size = None
            if cif and cif.exists():
                size = pocket_bbox_size(cif, str(row["union_chain_resi"]), args.padding, args.min_box, args.max_box)
            if size is None:  # fallback to rg
                s = box_size_from_rg(rg, args.k_rg, args.ligand_pad, args.min_box, args.max_box)
                size = (s, s, s)
                mode = "rg(fallback)"
            else:
                mode = "bbox"
        else:
            s = box_size_from_rg(rg, args.k_rg, args.ligand_pad, args.min_box, args.max_box)
            size = (s, s, s)
            mode = "rg"
        box_rows.append({
            "site_uid": site_uid,
            "center_x": round(center[0], 3), "center_y": round(center[1], 3), "center_z": round(center[2], 3),
            "size_x": round(size[0], 3), "size_y": round(size[1], 3), "size_z": round(size[2], 3),
            "box_mode": mode,
            "pocket_chain_resi": str(row["union_chain_resi"]),
        })
    box_df = pd.DataFrame(box_rows)

    # --- ligand groups ---
    groups = [int(g) for g in str(args.groups).split(",") if g.strip()]
    lig_frames = []
    if 1 in groups:
        lig_frames.append(group1_ligands(df, include_members=args.g1_all_ligands))
    if 2 in groups:
        lig_frames.append(group2_ligands(args.alphafill_long, keep_site_uids))
    if 3 in groups:
        lig_frames.append(group3_ligands(args.probe_csv, keep_site_uids))
    ligands_long = pd.concat(lig_frames, ignore_index=True) if lig_frames else pd.DataFrame(
        columns=["site_uid", "ligand_group", "ligand_id", "ligand_ref", "ligand_smiles"])
    ligands_long = ligands_long[ligands_long["site_uid"].astype(str).isin(keep_site_uids)].copy()

    # drop undockable ligands (monatomic ions / buffers) from groups 1-2 unless overridden
    n_dropped_undockable = 0
    if not getattr(args, "keep_all_ligands", False) and len(ligands_long):
        keep_mask = (ligands_long["ligand_group"] == 3) | ligands_long["ligand_id"].map(is_dockable_ccd)
        n_dropped_undockable = int((~keep_mask).sum())
        ligands_long = ligands_long[keep_mask].copy()
        if n_dropped_undockable:
            print(f"[FILT] dropped {n_dropped_undockable} undockable ligand jobs (ions/buffers) "
                  f"from groups 1-2 (use --keep_all_ligands to keep them).")
    build_manifest._n_dropped_undockable = n_dropped_undockable  # surfaced in the sidecar

    # --- assemble per-job manifest ---
    meta_cols = [c for c in (["site_uid", "site_uniprot"] + CARRY_COLS) if c in df.columns]
    meta = df[meta_cols].drop_duplicates("site_uid").copy()

    man = ligands_long.merge(box_df, on="site_uid", how="left").merge(meta, on="site_uid", how="left")

    # ligand file path + reference-pose expectation
    def _lig_sdf(row):
        if int(row["ligand_group"]) == 3:
            return f"ligands/probes/{_sanitize(row['ligand_id'])}.sdf"
        return f"ligands/ccd/{_sanitize(row['ligand_id'])}.sdf"

    man["ligand_sdf"] = man.apply(_lig_sdf, axis=1)
    man["ligand_n_heavy_atoms"] = ""  # filled by resolve_ligands.py if desired
    man["has_reference_pose"] = man["ligand_group"].isin([1, 2]) if args.expect_reference_poses else False
    man["exhaustiveness"] = args.exhaustiveness
    man["num_modes"] = args.num_modes
    man["seed"] = args.seed
    man["pocket_rg"] = pd.to_numeric(man.get("union_rg"), errors="coerce")
    man["pocket_n_residues"] = pd.to_numeric(man.get("union_n_residues_found"), errors="coerce")

    man["job_id"] = (man["site_uid"].map(_sanitize) + "__g" + man["ligand_group"].astype(str)
                     + "__" + man["ligand_id"].map(_sanitize))
    # keep job_id unique (a pocket could in principle repeat a CCD across groups -> group in id already)
    man = man.drop_duplicates("job_id").reset_index(drop=True)

    # column order: identity/receptor/ligand/box first, then carried metadata
    front = ["job_id", "site_uid", "site_uniprot", "threshold", "afdb_filename", "af_model_len",
             "ligand_group", "ligand_id", "ligand_ref", "ligand_smiles", "ligand_sdf", "ligand_n_heavy_atoms",
             "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "box_mode",
             "pocket_rg", "pocket_n_residues", "pocket_chain_resi",
             "exhaustiveness", "num_modes", "seed", "has_reference_pose"]
    rest = [c for c in man.columns if c not in front]
    man = man[[c for c in front if c in man.columns] + rest]

    # --- receptors / ligands side tables ---
    receptors = (man[["site_uniprot", "afdb_filename", "af_model_len"]]
                 .drop_duplicates("site_uniprot").sort_values("site_uniprot").reset_index(drop=True))
    receptors["model_path_hint"] = receptors["afdb_filename"].map(lambda f: str(cfg.AFDB_DIR / str(f)) if str(f) else "")

    ligands = (man[["ligand_group", "ligand_id", "ligand_smiles", "ligand_sdf"]]
               .drop_duplicates(["ligand_group", "ligand_id"]).sort_values(["ligand_group", "ligand_id"])
               .reset_index(drop=True))

    return man, receptors, ligands


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the docking handover manifest from the ST_100 geometry dataset.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--alphafill_long", type=Path, default=DEFAULT_ALPHAFILL_LONG)
    p.add_argument("--probe_csv", type=Path, default=DEFAULT_PROBE_CSV)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--groups", default="1", help="Comma-separated ligand groups to include, e.g. '1,2,3'.")
    p.add_argument("--g1_all_ligands", action="store_true",
                   help="Group 1: include all unique member-job ligands, not just the representative.")
    p.add_argument("--keep_all_ligands", action="store_true",
                   help="Keep undockable ligands (monatomic ions / buffers) in groups 1-2 (default: drop).")
    p.add_argument("--pilot_site_uids", type=Path, default=None,
                   help="Optional CSV with a 'site_uid' column to restrict to a pilot subset.")
    # box
    p.add_argument("--box_mode", choices=["rg", "bbox"], default="rg")
    p.add_argument("--center_kind", choices=["com", "geom"], default="com")
    p.add_argument("--k_rg", type=float, default=1.0)
    p.add_argument("--ligand_pad", type=float, default=8.0, help="Å added for ligand extent (rg mode).")
    p.add_argument("--padding", type=float, default=4.0, help="Å padding each side.")
    p.add_argument("--min_box", type=float, default=16.0)
    p.add_argument("--max_box", type=float, default=30.0)
    # engine suggestions
    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--num_modes", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expect_reference_poses", action="store_true",
                   help="Mark groups 1-2 has_reference_pose=True (set once poses are generated).")
    p.add_argument("--outfile_name", default="docking_manifest.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, low_memory=False)
    man, receptors, ligands = build_manifest(df, args)

    out_csv = args.output_dir / args.outfile_name
    man.to_csv(out_csv, index=False)
    receptors.to_csv(args.output_dir / "receptors.csv", index=False)
    ligands.to_csv(args.output_dir / "ligands.csv", index=False)

    meta = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "adapter_git_commit": _git_commit(),
        "inputs": {
            "geometry_csv": str(args.input_csv),
            "alphafill_long": str(args.alphafill_long),
            "probe_csv": str(args.probe_csv),
            "pilot_site_uids": str(args.pilot_site_uids) if args.pilot_site_uids else None,
        },
        "params": {
            "groups": args.groups, "g1_all_ligands": bool(args.g1_all_ligands),
            "keep_all_ligands": bool(args.keep_all_ligands),
            "box_mode": args.box_mode, "center_kind": args.center_kind,
            "k_rg": args.k_rg, "ligand_pad": args.ligand_pad, "padding": args.padding,
            "min_box": args.min_box, "max_box": args.max_box,
            "exhaustiveness": args.exhaustiveness, "num_modes": args.num_modes, "seed": args.seed,
            "expect_reference_poses": bool(args.expect_reference_poses),
        },
        "counts": {
            "jobs": int(len(man)),
            "dropped_undockable_ligand_jobs": getattr(build_manifest, "_n_dropped_undockable", 0),
            "pockets": int(man["site_uid"].nunique()) if len(man) else 0,
            "receptors": int(len(receptors)),
            "unique_ligands": int(len(ligands)),
            "jobs_by_group": {str(g): int((man["ligand_group"] == g).sum()) for g in sorted(man["ligand_group"].unique())} if len(man) else {},
        },
    }
    with open(args.output_dir / f"{Path(args.outfile_name).stem}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {args.output_dir / 'receptors.csv'}  ({len(receptors)} receptors)")
    print(f"[SAVE] {args.output_dir / 'ligands.csv'}  ({len(ligands)} unique ligands)")
    print(f"[JOBS] {len(man)}  by group: {meta['counts']['jobs_by_group']}")


if __name__ == "__main__":
    main()
