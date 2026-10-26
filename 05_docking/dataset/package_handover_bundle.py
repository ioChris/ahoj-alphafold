# -*- coding: utf-8 -*-
"""
Assemble a single, self-contained, share-ready docking handover bundle.

Takes a built manifest dir (from build_docking_manifest.py, e.g. the pilot) and
produces ONE directory the colleague can be given (OneDrive share / zip):

    <out_dir>/
      docking_manifest.csv        # LABEL-STRIPPED copy (docking-only columns)
      receptors.csv, ligands.csv
      receptors/AF-*.cif.gz       # copied from AFDB_DIR (self-contained)
      ligands/ccd/*.sdf           # copied resolved ligands
      ligands/probes/*.sdf        # if present
      reference_poses/            # if present (redock RMSD)
      docs/                       # HANDOVER_GUIDE, design summary, schemas, env yml
      BUNDLE_README.md            # auto-generated: what's inside + how to use
      package_manifest.json

Why label-stripped: the colleague only needs receptor + box + ligand to dock; the
apo/holo classification columns are our unpublished results and are re-joined
privately by job_id/site_uid at ingest. This keeps the shared copy "non-secret +
necessary" and avoids label-peeking bias.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd


def _copy(src, dst, retries: int = 6, wait: float = 0.7):
    """shutil.copy2 resilient to transient OneDrive/AV file locks (WinError 32)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError:
            if i == retries - 1:
                print(f"[LOCK] giving up on {src} after {retries} tries")
                raise
            time.sleep(wait)


def _retry(fn, retries: int = 8, wait: float = 0.8):
    """Retry a write callable through transient OneDrive/AV locks."""
    for i in range(retries):
        try:
            return fn()
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(wait)

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Unpublished classification / evidence columns to DROP from the shared manifest.
LABEL_COLS = [
    "pocket_class", "pocket_p_apo", "pocket_p_holo", "pocket_separation_pair",
    "pocket_score", "model_pocket_plddt", "n_apo_avg", "n_holo_avg", "n_unp_pockets",
]


def find_af_cif(uniprot: str, afdb_dir: Path):
    hits = glob.glob(str(afdb_dir / f"AF-{uniprot}-F1-model_v*.cif.gz"))
    if not hits:
        return None
    def ver(p):
        try:
            return int(Path(p).name.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            return -1
    return Path(max(hits, key=ver))


def copytree_if(src: Path, dst: Path, skip_existing: bool = False) -> int:
    if not src.exists():
        return 0
    if skip_existing and dst.exists() and any(dst.rglob("*")):
        return sum(1 for p in dst.rglob("*") if p.is_file())
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            _copy(p, dst / rel)
            n += 1
    return n


def write_readme(out_dir: Path, counts: dict) -> None:
    txt = f"""# Docking pilot handover bundle

**Self-contained, share-ready.** Everything needed to run the docking pilot; contains
**no unpublished classification labels** (the manifest is docking-only — receptor + box +
ligand). Results are re-joined to our apo/holo analysis on our side by `job_id`/`site_uid`.

> This is a **pilot (v0) on current data** for pipeline validation. The scientific pilot
> reruns on the new classification round — treat these specific numbers as provisional.

## Contents
- `docking_manifest.csv` — {counts['jobs']} jobs over {counts['pockets']} pockets / {counts['receptors']} receptors.
  Ligand groups: 1 = query holo (main), 2 = AlphaFill cognate ({counts['g2_pockets']} pockets), 3 = probe (if present).
- `receptors.csv`, `ligands.csv` — lookup tables.
- `receptors/` — {counts['receptor_files']} AlphaFold-DB models (`AF-*-F1-model_v*.cif.gz`).
- `experimental_holo/` — {counts['exp_structures']} experimental holo structures (`<pdb>.cif.gz`) that
  **escort** the AF models. `experimental_structures.csv` + the manifest's `escort_*` columns link each
  job to its PDB, receptor chain (`escort_prot_chains`), bound-ligand location (`escort_lig_*`), and the
  **pocket residue set both ways**: `escort_uniprot_residues` (UniProt numbering = the AF pocket
  definition) and `escort_pdb_residues` (same AHoJ binding residues in the PDB's own numbering,
  e.g. "A_45 A_67 ..."). All pockets are single-chain / non-interface, so each escort is a clean
  single-chain receptor.
- `ligands/ccd/` — {counts['ccd_sdf']} resolved CCD ligand SDFs (3D).
- `ligands/probes/` — {counts['probe_sdf']} probe SDFs (group 3){' — PENDING (needs rdkit env)' if counts['probe_sdf'] == 0 else ''}.
- `reference_poses/` — {counts['refpose_files']} reference poses for redock RMSD{' — PENDING (optional)' if counts['refpose_files'] == 0 else ''}.
- `docs/` — start with `HANDOVER_GUIDE.md`; I/O contracts in `docking_*.schema.md`; env in `environment.docking.yml`.

## How to use (short)
1. Read `docs/HANDOVER_GUIDE.md` (env, runbook, and the exact I/O contract; has an AI-agent spec).
2. Dry-run ~5 jobs first, validate against `docs/docking_results.schema.md`.
3. Dock each manifest row (box center/size given); return `docking_results.csv` (one row per `job_id`)
   + top-pose SDFs + a `docking_results_manifest.json` recording engine/version/seed.

## Engine
AutoDock Vina 1.2.x (reproducible baseline) or Uni-Dock (GPU, same scoring family) — your call.

_Generated {counts['timestamp']}._
"""
    (out_dir / "BUNDLE_README.md").write_text(txt, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble a self-contained docking handover bundle.")
    p.add_argument("--manifest_dir", type=Path, default=cfg.DOCKING_PILOT_DIR,
                   help="Dir with docking_manifest.csv + receptors.csv + ligands.csv + ligands/.")
    p.add_argument("--out_dir", type=Path, default=cfg.DOCKING_HANDOVER_DIR)
    p.add_argument("--afdb_dir", type=Path, default=cfg.AFDB_DIR)
    p.add_argument("--docs", default="", help="Comma-separated doc files to copy into docs/.")
    p.add_argument("--no_copy_receptors", action="store_true", help="Skip copying AF models (share by UniProt instead).")
    p.add_argument("--skip_existing_data", action="store_true",
                   help="Don't re-copy receptors/ligands/experimental_holo if already present (refresh text files only).")
    p.add_argument("--keep_labels", action="store_true", help="Do NOT strip classification columns (default: strip).")
    p.add_argument("--zip", action="store_true", help="Also write <out_dir>.zip.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    md = args.manifest_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(md / "docking_manifest.csv", low_memory=False)
    if not args.keep_labels:
        man = man.drop(columns=[c for c in LABEL_COLS if c in man.columns])
    # link each job to its experimental escort structure (if fetched)
    exp_csv = md / "experimental_structures.csv"
    if exp_csv.exists():
        esc = pd.read_csv(exp_csv, low_memory=False)
        keep = [c for c in ["job_id", "escort_pdb_id", "escort_prot_chains", "escort_lig_chain",
                            "escort_lig_id", "escort_lig_resnum", "escort_uniprot_residues",
                            "escort_pdb_residues", "escort_pdb_residues_n", "escort_pdb_residues_source",
                            "escort_cif_gz", "escort_status"]
                if c in esc.columns]
        man = man.merge(esc[keep], on="job_id", how="left")
    # bundle-relative pointer to each job's receptor (no absolute disk paths)
    if "afdb_filename" in man.columns:
        man["receptor_file"] = "receptors/" + man["afdb_filename"].astype(str)
    _retry(lambda: man.to_csv(out / "docking_manifest.csv", index=False))
    # ligands.csv: paths already relative -> copy as-is
    if (md / "ligands.csv").exists():
        _copy(md / "ligands.csv", out / "ligands.csv")
    # receptors.csv: rewrite model_path_hint as bundle-relative (strip absolute disk path)
    if (md / "receptors.csv").exists():
        rec = pd.read_csv(md / "receptors.csv")
        if "afdb_filename" in rec.columns:
            rec["model_path_hint"] = "receptors/" + rec["afdb_filename"].astype(str)
        _retry(lambda: rec.to_csv(out / "receptors.csv", index=False))

    sk = args.skip_existing_data
    ccd_n = copytree_if(md / "ligands" / "ccd", out / "ligands" / "ccd", sk)
    probe_n = copytree_if(md / "ligands" / "probes", out / "ligands" / "probes", sk)
    ref_n = copytree_if(md / "reference_poses", out / "reference_poses", sk)
    exp_n = copytree_if(md / "experimental_holo", out / "experimental_holo", sk)
    if exp_csv.exists():
        _copy(exp_csv, out / "experimental_structures.csv")
    if (md / "ligands" / "probe_library.csv").exists():
        _copy(md / "ligands" / "probe_library.csv", out / "ligands" / "probe_library.csv")

    rec_n = 0
    if not args.no_copy_receptors:
        rdst = out / "receptors"
        rdst.mkdir(parents=True, exist_ok=True)
        if args.skip_existing_data and any(rdst.glob("*.cif.gz")):
            rec_n = len(list(rdst.glob("*.cif.gz")))
        else:
            for u in pd.read_csv(md / "receptors.csv")["site_uniprot"].astype(str).unique():
                cif = find_af_cif(u, args.afdb_dir)
                if cif:
                    _copy(cif, rdst / cif.name)
                    rec_n += 1

    docs_dir = out / "docs"
    doc_n = 0
    for d in [x.strip() for x in args.docs.split(",") if x.strip()]:
        dp = Path(d)
        if dp.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            _copy(dp, docs_dir / dp.name)
            doc_n += 1
        else:
            print(f"[WARN] doc not found, skipped: {dp}")

    counts = {
        "timestamp": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "jobs": int(len(man)), "pockets": int(man["site_uid"].nunique()),
        "receptors": int(man["site_uniprot"].nunique()),
        "g2_pockets": int(man.loc[man["ligand_group"] == 2, "site_uid"].nunique()),
        "receptor_files": rec_n, "ccd_sdf": ccd_n, "probe_sdf": probe_n,
        "refpose_files": ref_n, "exp_structures": exp_n, "docs": doc_n,
        "labels_stripped": (not args.keep_labels),
    }
    _retry(lambda: write_readme(out, counts))
    def _wjson():
        with open(out / "package_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(counts, fh, indent=2)
    _retry(_wjson)

    if args.zip:
        zip_path = out.parent / f"{out.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(out.parent))
        print(f"[ZIP ] {zip_path}")

    print(f"[SAVE] bundle -> {out}")
    print(f"[DATA] jobs={counts['jobs']} receptors={rec_n} ccd_sdf={ccd_n} probes={probe_n} refposes={ref_n} docs={doc_n}")
    print(f"[MANIFEST] labels_stripped={counts['labels_stripped']} cols={len(man.columns)}")


if __name__ == "__main__":
    main()
