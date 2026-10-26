# -*- coding: utf-8 -*-
"""
Symmetry-corrected, in-place (no re-alignment) redock RMSD utilities.

The docked pose and the reference pose are the SAME CCD molecule but may have
different atom orders (Meeko/mk_export canonicalises from SMILES; the reference is
built on the ideal-SDF template) and internal symmetry (equivalent phosphate O's,
ring flips, etc.). We therefore match the two molecules by GRAPH ISOMORPHISM on a
connectivity-only ("generic") graph — every isomorphic mapping is enumerated, which
simultaneously covers the atom-order difference AND molecular symmetry — and take
the minimum RMSD over all mappings, WITHOUT superposing (both poses already live in
the AF-model frame). This is the conventional cross-docking pose-recovery metric.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def generic_query(mol: Chem.Mol) -> Chem.Mol:
    """Connectivity-only copy: all bonds SINGLE, charges/H cleared — so isomorphism
    matching ignores bond order / protonation differences between the two sources."""
    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE, catchErrors=True)
    return m


def mol_from_coords(elements: Sequence[str], coords: np.ndarray) -> Optional[Chem.Mol]:
    """Build an RDKit mol (heavy atoms only) from elements + 3D coords, perceiving
    connectivity by distance. Returns None on failure."""
    try:
        rw = Chem.RWMol()
        conf = Chem.Conformer(len(elements))
        for i, el in enumerate(elements):
            idx = rw.AddAtom(Chem.Atom(el))
            conf.SetAtomPosition(idx, tuple(float(x) for x in coords[i]))
        m = rw.GetMol()
        m.AddConformer(conf, assignId=True)
        rdDetermineBonds.DetermineConnectivity(m)
        return m
    except Exception:
        return None


def _conf_coords(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    c = mol.GetConformer(conf_id)
    return np.array([list(c.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)


def build_reference_conformer(template: Chem.Mol, elements: Sequence[str],
                              af_coords: np.ndarray) -> Tuple[Optional[Chem.Mol], str]:
    """Map experimental heavy-atom coords (in AF frame) onto the CCD template graph.

    Returns (template_with_conformer, status). Status 'ok' | reason. The returned mol
    is the template with a single conformer holding the experimental coords in
    TEMPLATE atom order.
    """
    tmpl = Chem.Mol(template)
    tmpl = Chem.RemoveHs(tmpl) if tmpl.GetNumAtoms() != tmpl.GetNumHeavyAtoms() else tmpl
    ref_raw = mol_from_coords(elements, af_coords)
    if ref_raw is None:
        return None, "ref_raw_build_failed"
    gt, gr = generic_query(tmpl), generic_query(ref_raw)
    # map template atom i -> ref_raw atom index
    match = gr.GetSubstructMatch(gt)
    if not match or len(match) != tmpl.GetNumAtoms():
        # try the reverse direction, then invert
        rev = gt.GetSubstructMatch(gr)
        if rev and len(rev) == ref_raw.GetNumAtoms():
            inv = {rev[j]: j for j in range(len(rev))}
            match = tuple(inv.get(i, -1) for i in range(tmpl.GetNumAtoms()))
            if -1 in match:
                return None, "partial_isomorphism"
        else:
            return None, "no_isomorphism"
    conf = Chem.Conformer(tmpl.GetNumAtoms())
    for i in range(tmpl.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(float(x) for x in af_coords[match[i]]))
    tmpl.RemoveAllConformers()
    tmpl.AddConformer(conf, assignId=True)
    return tmpl, "ok"


def symmetry_rmsd_inplace(docked: Chem.Mol, reference: Chem.Mol,
                          max_matches: int = 5000) -> Tuple[Optional[float], Optional[float], int, str]:
    """In-place (no alignment) symmetry-corrected RMSD between a multi-conformer docked
    mol and a single-conformer reference mol (same molecule, possibly different order).

    Returns (rmsd_top, rmsd_best, n_isomorphisms, status). rmsd_top uses conformer 0
    (the top Vina mode); rmsd_best is the minimum over all docked conformers.
    """
    try:
        dq, rq = generic_query(Chem.RemoveHs(docked)), generic_query(Chem.RemoveHs(reference))
    except Exception:
        dq, rq = generic_query(docked), generic_query(reference)
    # all mappings ref-atom -> docked-atom (covers order + symmetry)
    matches = dq.GetSubstructMatches(rq, uniquify=False, maxMatches=max_matches)
    if not matches:
        return None, None, 0, "no_isomorphism"
    ref_xyz = _conf_coords(reference, reference.GetConformer().GetId())
    d_docked = Chem.RemoveHs(docked)
    n_conf = d_docked.GetNumConformers()
    if n_conf == 0:
        return None, None, len(matches), "no_docked_conformer"
    per_conf = []
    for cid in range(n_conf):
        dx = _conf_coords(d_docked, d_docked.GetConformer(cid).GetId())
        best = np.inf
        for match in matches:  # match[i] = docked atom for reference atom i
            idx = np.array(match, dtype=int)
            diff = dx[idx] - ref_xyz
            best = min(best, float(np.sqrt((diff * diff).sum(axis=1).mean())))
        per_conf.append(best)
    return per_conf[0], float(np.min(per_conf)), len(matches), "ok"
