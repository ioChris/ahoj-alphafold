# -*- coding: utf-8 -*-
"""
Core helpers for run_3 redock-RMSD (Analysis 6, G1 query-holo jobs).

Places the experimental bound ligand into the AF (v6) model frame and computes a
symmetry-corrected, IN-PLACE (no re-superposition) heavy-atom RMSD against the
docked top pose. Robust to the two things that broke the pilot scaffold at scale:

  * incomplete crystal ligands  -> map the experimental atoms onto the CCD
    *component* graph BY ATOM NAME (PDB atom ids == CCD `_chem_comp_atom.atom_id`),
    so missing/disordered atoms are simply absent, never a failed graph perception;
  * raw docked PDBQT (no bond orders) -> reconstruct the bonded RDKit mol IN
    PROCESS from the `REMARK SMILES` Meeko embeds in every pose (meeko API, no
    mk_export subprocess), then map it onto the same CCD template by isomorphism.

RMSD is the minimum over the template's graph automorphisms of the in-place RMSD
over the experimentally observed atoms (verified identical to
`spyrmsd.symmrmsd(minimize=False)`).

Self-contained: rdkit + gemmi + meeko + numpy (the `ahoj-docking` env). Importing
this module sets KMP_DUPLICATE_LIB_OK (the rdkit/gemmi/numpy OpenMP DLLs otherwise
native-crash on Windows with 0xC06D007F).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import gemmi
from rdkit import Chem

# ------------------------------------------------------------------ CCD template
_ORDER = {
    "SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE,
    "AROM": Chem.BondType.AROMATIC, "QUAD": Chem.BondType.QUADRUPLE,
    "1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE,
}


@dataclass
class Template:
    """A CCD chemical-component graph: the canonical frame for one ligand id."""
    ccd: str
    mol: Chem.Mol                 # heavy-atom RDKit mol, atoms in component order
    name2idx: Dict[str, int]      # CCD atom_id -> template index
    autos: List[np.ndarray]       # graph automorphisms (permutations), for symmetry
    n: int


def _elem_title(sym: str) -> str:
    sym = sym.strip()
    return sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()


def load_ccd_template(cif_path: Path, max_autos: int = 200000) -> Optional[Template]:
    """Build a `Template` from an RCSB CCD component cif (files.rcsb.org/ligands/...)."""
    try:
        blk = gemmi.cif.read(str(cif_path)).sole_block()
    except Exception:
        return None
    ccd = str(blk.name).lstrip("comp_") or Path(cif_path).stem
    names = list(blk.find_loop("_chem_comp_atom.atom_id"))
    elems = list(blk.find_loop("_chem_comp_atom.type_symbol"))
    if not names or len(names) != len(elems):
        return None
    # D (deuterium) / T (tritium) are hydrogen isotopes: filter them out with H
    # (RDKit's Atom('D') raises "Element 'D' not found").
    keep = [i for i, e in enumerate(elems) if e.upper() not in ("H", "D", "T")]
    if not keep:
        return None
    name2idx = {names[i].strip('"'): k for k, i in enumerate(keep)}
    try:
        rw = Chem.RWMol()
        for i in keep:
            rw.AddAtom(Chem.Atom(_elem_title(elems[i])))
        b1 = [x.strip('"') for x in blk.find_loop("_chem_comp_bond.atom_id_1")]
        b2 = [x.strip('"') for x in blk.find_loop("_chem_comp_bond.atom_id_2")]
        bo = list(blk.find_loop("_chem_comp_bond.value_order")) or ["SING"] * len(b1)
        for a, b, o in zip(b1, b2, bo):
            ia, ib = name2idx.get(a), name2idx.get(b)
            if ia is not None and ib is not None:
                rw.AddBond(ia, ib, _ORDER.get(str(o).upper(), Chem.BondType.SINGLE))
        mol = rw.GetMol()
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE, catchErrors=True)
        autos = _automorphisms(mol, max_autos)
    except Exception:
        return None
    return Template(ccd=ccd, mol=mol, name2idx=name2idx, autos=autos, n=mol.GetNumAtoms())


def _generic(mol: Chem.Mol) -> Chem.Mol:
    """Connectivity-only copy (all bonds SINGLE, no aromatic/charge/H) for isomorphism."""
    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0); a.SetIsAromatic(False); a.SetNoImplicit(True); a.SetNumExplicitHs(0)
    g = rw.GetMol()
    Chem.SanitizeMol(g, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE, catchErrors=True)
    return g


def _automorphisms(mol: Chem.Mol, max_matches: int) -> List[np.ndarray]:
    g = _generic(mol)
    ms = g.GetSubstructMatches(g, uniquify=False, maxMatches=max_matches)
    return [np.asarray(m, dtype=int) for m in ms] if ms else [np.arange(mol.GetNumAtoms())]


# --------------------------------------------------------------- superposition
def read_model(path: Path):
    return gemmi.read_structure(str(path))[0]


def seq_correspondence(afc, pc) -> Dict[int, object]:
    """AF chain seqid (== UniProt position) -> corresponding query-chain residue."""
    ap, pp = afc.get_polymer(), pc.get_polymer()
    ar, pr = list(ap), list(pp)
    s1 = gemmi.one_letter_code(ap.extract_sequence())
    s2 = gemmi.one_letter_code(pp.extract_sequence())
    res = gemmi.align_string_sequences(list(s1), list(s2), [])
    g1, g2 = res.add_gaps(s1, 1), res.add_gaps(s2, 2)
    mp: Dict[int, object] = {}
    i = j = 0
    for c1, c2 in zip(g1, g2):
        if c1 != "-" and c2 != "-" and i < len(ar) and j < len(pr):
            mp[ar[i].seqid.num] = pr[j]
        if c1 != "-":
            i += 1
        if c2 != "-":
            j += 1
    return mp


def pocket_superpose(afc, pc, positions: Sequence[int], min_pairs: int = 3):
    """Superpose the query protein chain onto the AF chain on pocket-residue CA pairs.

    Returns (transform, rmsd, n_pairs, mode). The transform maps QUERY coords -> AF
    frame (apply to the experimental ligand). Falls back to whole-chain CA if the
    pocket has too few mappable CA pairs.
    """
    mp = seq_correspondence(afc, pc)
    af_ca = {r.seqid.num: r for r in afc if r.find_atom("CA", "*")}
    A, B = [], []
    for p in positions:
        pr = mp.get(p)
        if p in af_ca and pr is not None and pr.find_atom("CA", "*"):
            A.append(af_ca[p].find_atom("CA", "*").pos)
            B.append(pr.find_atom("CA", "*").pos)
    if len(A) >= min_pairs:
        sup = gemmi.superpose_positions(A, B)      # transform maps B(query) -> A(AF)
        return sup.transform, float(sup.rmsd), len(A), "pocket"
    sup = gemmi.calculate_superposition(afc.get_polymer(), pc.get_polymer(),
                                        gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP)
    return sup.transform, float(sup.rmsd), int(sup.count), "wholechain"


def extract_ligand_named(model, chain: str, ccd: str, resnum: int,
                         altloc_pref: str = "A") -> List[Tuple[str, str, object]]:
    """Heavy atoms of one ligand instance as (atom_name, element, gemmi.Position).

    Keeps the highest-priority altloc per atom name (blank or `altloc_pref` first).
    """
    best: Dict[str, Tuple[int, str, str, object]] = {}
    for c in model:
        if c.name != chain:
            continue
        for r in c:
            if r.name.upper() != ccd.upper() or r.seqid.num != resnum:
                continue
            for a in r:
                el = str(a.element.name)
                if el == "H":
                    continue
                al = a.altloc or ""
                rank = 0 if al in ("", "\x00") else (1 if al.upper() == altloc_pref else 2)
                cur = best.get(a.name)
                if cur is None or rank < cur[0]:
                    best[a.name] = (rank, a.name, el, a.pos)
    return [(nm, el, pos) for _, nm, el, pos in best.values()]


# --------------------------------------------------------------- docked pose
_MEEKO = None


def _meeko():
    global _MEEKO
    if _MEEKO is None:
        from meeko import PDBQTMolecule, RDKitMolCreate
        _MEEKO = (PDBQTMolecule, RDKitMolCreate)
    return _MEEKO


def docked_mol_from_block(pdbqt_block: str) -> Optional[Chem.Mol]:
    """Reconstruct the bonded, heavy-atom docked mol (one conformer) from a raw PDBQT
    MODEL block via Meeko's embedded `REMARK SMILES`. Returns None on failure."""
    PDBQTMolecule, RDKitMolCreate = _meeko()
    try:
        pmol = PDBQTMolecule(pdbqt_block, skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pmol)
        if not mols or mols[0] is None:
            return None
        m = mols[0]
        # Strip ALL hydrogens: RemoveHs keeps "non-removable" H's (imine/amidine/
        # guanidine =N-H etc.), which would inflate the atom count and break the
        # heavy-atom graph match against the CCD template.
        try:
            return Chem.RemoveAllHs(m)
        except Exception:
            return Chem.RemoveHs(m)
    except Exception:
        return None


def docked_affinity(pdbqt_block: str) -> Optional[float]:
    m = re.search(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+\.\d+)", pdbqt_block)
    return float(m.group(1)) if m else None


def map_docked_to_template(dock_mol: Chem.Mol, template: Template) -> Optional[np.ndarray]:
    """Docked-pose coords in template atom order (via graph isomorphism). None if the
    docked graph is not isomorphic to the complete CCD template."""
    if dock_mol.GetNumAtoms() != template.n:
        return None
    match = _generic(dock_mol).GetSubstructMatch(_generic(template.mol))  # tmpl i -> dock idx
    if not match or len(match) != template.n:
        return None
    dpos = dock_mol.GetConformer().GetPositions()
    return np.asarray([dpos[match[i]] for i in range(template.n)], dtype=float)


# --------------------------------------------------------------- RMSD
def exp_coords_template_order(exp_named: Sequence[Tuple[str, str, np.ndarray]],
                             template: Template) -> Tuple[np.ndarray, np.ndarray]:
    """Experimental (AF-frame) coords laid out in template order + observed mask."""
    coords = np.full((template.n, 3), np.nan)
    obs = np.zeros(template.n, dtype=bool)
    for name, _el, xyz in exp_named:
        j = template.name2idx.get(name)
        if j is not None:
            coords[j] = xyz
            obs[j] = True
    return coords, obs


def symmetry_rmsd_observed(dock_tmpl: np.ndarray, ref_tmpl: np.ndarray,
                           observed: np.ndarray, autos: Sequence[np.ndarray],
                           min_atoms: int = 3) -> Tuple[Optional[float], int]:
    """min over automorphisms t of in-place RMSD( dock[i], ref[t[i]] ) over observed t[i]."""
    nobs = int(observed.sum())
    if nobs < min_atoms or dock_tmpl is None:
        return None, nobs
    best = np.inf
    for t in autos:
        om = observed[t]
        if om.sum() < min_atoms:
            continue
        d = dock_tmpl[om] - ref_tmpl[t][om]
        r = float(np.sqrt((d * d).sum(axis=1).mean()))
        if r < best:
            best = r
    return (None if not np.isfinite(best) else best), nobs


# --------------------------------------------------------------- pose store
def iter_pose_blocks(poses_text: str):
    """Yield (job_id, pdbqt_block) from a concatenated `>>> {job_id}` poses stream."""
    job_id = None
    buf: List[str] = []
    for line in poses_text.splitlines(keepends=True):
        if line.startswith(">>> "):
            if job_id is not None:
                yield job_id, "".join(buf)
            job_id = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    if job_id is not None:
        yield job_id, "".join(buf)
