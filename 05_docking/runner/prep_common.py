# -*- coding: utf-8 -*-
"""
Shared helpers for the CPU docking runner (AutoDock Vina baseline).

Self-contained: depends only on gemmi + rdkit + numpy (the `ahoj-docking` env)
and the Meeko CLI (`mk_prepare_receptor`, `mk_prepare_ligand`, `mk_export`) + a
Vina binary. Nothing here imports the wider AHoJ-AlphaFold `common.config`, so the
whole `runner/` folder runs standalone from the handover bundle.

Design notes
------------
* Receptor prep: gemmi reads the AFDB `.cif.gz`, keeps chain A amino-acid residues
  only (AFDB monomers are single-chain, no waters/ligands), writes a clean PDB,
  then Meeko builds the rigid receptor PDBQT. We do NOT trim low-pLDDT regions in
  the pilot (recorded as a prep-policy flag).
* Ligand prep: Meeko on the provided 3D CCD SDF (protonation left as-provided by
  the RCSB ideal coordinates; recorded as prep policy).
* Boxes come verbatim from the manifest (`center_*`, `size_*`); the runner never
  re-centres or resizes (handover contract).
* `pose_in_pocket`: top pose has >= `min_atoms` heavy atoms within `cutoff` A of any
  heavy atom of the manifest pocket residues (`pocket_chain_resi`).
"""

from __future__ import annotations

import glob
import gzip
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# filesystem helpers
# --------------------------------------------------------------------------- #
def safe_token(s) -> str:
    """Filesystem-safe token (site_uid / job_id contain ':' which is illegal on Windows)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def find_af_cif(uniprot: str, receptors_dir: Path) -> Optional[Path]:
    """Highest-version AF model for a UniProt in a receptors dir (bundle or AFDB)."""
    hits = glob.glob(str(receptors_dir / f"AF-{uniprot}-F1-model_v*.cif.gz"))
    if not hits:
        return None

    def ver(p: str) -> int:
        try:
            return int(Path(p).name.split("model_v", 1)[1].split(".", 1)[0])
        except Exception:
            return -1

    return Path(max(hits, key=ver))


# --------------------------------------------------------------------------- #
# receptor extraction (gemmi)  -> clean chain-A protein PDB
# --------------------------------------------------------------------------- #
def extract_receptor_pdb(cif_gz: Path, out_pdb: Path, chain: str = "A") -> int:
    """Write chain-`chain` amino-acid residues of an AFDB model to a plain PDB.

    Returns the number of residues written. Raises on read failure.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_gz))
    st.setup_entities()
    model = st[0]
    out = gemmi.Structure()
    out.spacegroup_hm = "P 1"
    nm = gemmi.Model("1")
    newchain = gemmi.Chain(chain)
    nres = 0
    # AFDB monomer: first (only) chain is the protein; be defensive and match name too.
    for ch in model:
        if ch.name not in (chain, ""):  # AFDB uses 'A'
            # still allow if it's the only chain
            if len(list(model)) > 1:
                continue
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info and info.is_amino_acid():
                newchain.add_residue(res)
                nres += 1
        break  # single-chain models only
    nm.add_chain(newchain)
    out.add_model(nm)
    out.setup_entities()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(out.make_pdb_string())
    return nres


# --------------------------------------------------------------------------- #
# Meeko CLI wrappers
# --------------------------------------------------------------------------- #
def prepare_receptor(pdb_in: Path, pdbqt_out: Path, mk_prepare_receptor: str,
                     timeout: int = 300) -> Tuple[bool, str]:
    """Meeko rigid receptor PDBQT. Returns (ok, message)."""
    pdbqt_out.parent.mkdir(parents=True, exist_ok=True)
    # Meeko appends .pdbqt to the -o stem.
    stem = str(pdbqt_out.with_suffix(""))
    cmd = [mk_prepare_receptor, "--read_pdb", str(pdb_in), "-o", stem, "-p"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "meeko-receptor-timeout"
    if pdbqt_out.exists() and pdbqt_out.stat().st_size > 0:
        return True, "ok"
    return False, f"meeko-receptor-fail: {(r.stderr or r.stdout)[-300:].strip()}"


def prepare_ligand(sdf_in: Path, pdbqt_out: Path, mk_prepare_ligand: str,
                   timeout: int = 120) -> Tuple[bool, str]:
    """Meeko ligand PDBQT from a 3D SDF. Returns (ok, message)."""
    pdbqt_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [mk_prepare_ligand, "-i", str(sdf_in), "-o", str(pdbqt_out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "meeko-ligand-timeout"
    if pdbqt_out.exists() and pdbqt_out.stat().st_size > 0:
        return True, "ok"
    return False, f"meeko-ligand-fail: {(r.stderr or r.stdout)[-300:].strip()}"


def export_pose_sdf(pdbqt_in: Path, sdf_out: Path, mk_export: str,
                    timeout: int = 120) -> Tuple[bool, str]:
    """Convert a docked pose PDBQT (multi-MODEL) to SDF, restoring bond orders.

    Meeko's mk_export reads the SMILES REMARK written by mk_prepare_ligand, so the
    exported SDF has correct bonds/orders for downstream symmetry-corrected RMSD.
    """
    sdf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [mk_export, str(pdbqt_in), "-s", str(sdf_out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "mk_export-timeout"
    if sdf_out.exists() and sdf_out.stat().st_size > 0:
        return True, "ok"
    return False, f"mk_export-fail: {(r.stderr or r.stdout)[-300:].strip()}"


# --------------------------------------------------------------------------- #
# ligand descriptors (size covariate)
# --------------------------------------------------------------------------- #
def ligand_heavy_atoms(sdf: Path) -> Optional[int]:
    from rdkit import Chem

    try:
        supp = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True)
        m = next((x for x in supp if x is not None), None)
        if m is None:
            m = next((x for x in Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
                      if x is not None), None)
        return int(m.GetNumHeavyAtoms()) if m is not None else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# pocket geometry / pose localisation
# --------------------------------------------------------------------------- #
def parse_pocket_resi(chain_resi: str) -> List[Tuple[str, int]]:
    """'A_16 A_17 ...' -> [('A',16),('A',17),...]"""
    out: List[Tuple[str, int]] = []
    for tok in str(chain_resi).split():
        if "_" in tok:
            ch, num = tok.rsplit("_", 1)
            try:
                out.append((ch, int(num)))
            except ValueError:
                continue
    return out


def receptor_pocket_heavy_coords(pdb_path: Path, pocket: Sequence[Tuple[str, int]]) -> np.ndarray:
    """Heavy-atom coordinates (Nx3) of the pocket residues in a receptor PDB."""
    want = set(pocket)
    coords: List[List[float]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = line[76:78].strip() or line[12:16].strip()[0:1]
            if elem == "H":
                continue
            ch = line[21].strip()
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            if (ch, resnum) in want:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=float) if coords else np.empty((0, 3))


def parse_vina_pdbqt(pdbqt_path: Path) -> Tuple[List[float], List[np.ndarray]]:
    """Parse a Vina output PDBQT.

    Returns (affinities, models) where `affinities[i]` is the Vina score of MODEL i+1
    (from the `REMARK VINA RESULT:` line) and `models[i]` is that model's heavy-atom
    coordinates (Kx3), excluding AutoDock H types (H/HD).
    """
    affinities: List[float] = []
    models: List[np.ndarray] = []
    cur: List[List[float]] = []
    in_model = False
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                in_model = True
                cur = []
            elif line.startswith("REMARK VINA RESULT"):
                try:
                    affinities.append(float(line.split(":")[1].split()[0]))
                except Exception:
                    affinities.append(float("nan"))
            elif line.startswith(("ATOM", "HETATM")) and in_model:
                adtype = line[77:79].strip()
                if adtype in ("H", "HD"):
                    continue
                try:
                    cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    continue
            elif line.startswith("ENDMDL"):
                models.append(np.array(cur, dtype=float) if cur else np.empty((0, 3)))
                in_model = False
    # single-model files without MODEL/ENDMDL wrappers
    if not models and cur:
        models.append(np.array(cur, dtype=float))
    return affinities, models


def count_pose_in_pocket(pose_coords: np.ndarray, pocket_coords: np.ndarray,
                         cutoff: float = 4.0) -> int:
    """Number of pose heavy atoms within `cutoff` A of any pocket heavy atom."""
    if pose_coords.size == 0 or pocket_coords.size == 0:
        return 0
    # pairwise min distance per pose atom (vectorised, memory-light in chunks)
    n = 0
    for a in pose_coords:
        d = np.sqrt(((pocket_coords - a) ** 2).sum(axis=1))
        if d.min() <= cutoff:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Vina invocation
# --------------------------------------------------------------------------- #
def run_vina(vina_bin: str, receptor_pdbqt: Path, ligand_pdbqt: Path, out_pdbqt: Path,
             center: Tuple[float, float, float], size: Tuple[float, float, float],
             exhaustiveness: int, num_modes: int, seed: int, cpu: int = 1,
             timeout: int = 1200) -> Tuple[bool, str]:
    """Run one Vina docking. Returns (ok, message)."""
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        vina_bin,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]}", "--center_y", f"{center[1]}", "--center_z", f"{center[2]}",
        "--size_x", f"{size[0]}", "--size_y", f"{size[1]}", "--size_z", f"{size[2]}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        "--cpu", str(cpu),
        "--out", str(out_pdbqt),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"vina-timeout>{timeout}s"
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        return True, "ok"
    return False, f"vina-fail(rc={r.returncode}): {(r.stderr or r.stdout)[-300:].strip()}"
