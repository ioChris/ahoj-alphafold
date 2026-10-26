# -*- coding: utf-8 -*-
"""Does occlusion by full-length AlphaFold model segments bias the model's pocket metrics, and so the
AF-Apo / AF-Holo calls?  (Zhang et al. 2023 JCIM, zhang2023jcim, for the docking precedent.)  Backs
Supplementary Note 1 ("The model's own context", and the incomplete-pocket shares of apo and holo conformers).

THE ASYMMETRY. An AFDB model is full length: termini, loops and extra domains are all modelled. A PDB
structure carries only the residues of its construct that were observed. AHoJ 2.1.0, which made the v2c
archives, measured every pocket on a selection of the pocket residues inside the loaded object:

  * SASA   cmd.get_area(pocket) with dot_solvent 1 and `flag ignore, hetatm and not polymer`. PyMOL
           assesses accessibility in the context of the object, so every non-pocket atom of the object
           can occlude the pocket. The AF object is the whole model (one copy per query chain of the
           pocket, superposed and merged); a PDB candidate is its pocket chains plus everything within
           15 A of them (crop_structure, byres).
  * volume PyVOL pymol_pocket_cmdline(protein=<pocket selection>): PyVOL saves ONLY the selection
           (pyvol/pymol_interface.py, cmd.save(prot_file, protein)) -- other atoms cannot enter.
  * RoG    rgyrate(<pocket selection>) -- the selection only.
  * pocket_rms  cmd.align(pocket, query pocket, cycles=0, transform=0) -- the two selections only.

So only SASA can be lowered by a model segment that the experimental structures lack. This script
measures by how much, for every analysis-set entry, and what it does to the calls.

UNMATCHED RESIDUES. A model residue outside the pocket and within 10 A of it (any atom; beyond ~6.4 A
it cannot change the pocket's SASA at all) is classed from SIFTS (uniprot_segments_observed and
pdb_chain_uniprot, local copy of 2026-09-13) against the entry's curated experimental conformers -- the
query plus every apo and holo candidate the classifier used -- on the chains that carry the pocket:

    q      not observed in the query chain the model copy was superposed on
    maj    observed in fewer than half of the conformers (query included)   <- the primary definition
    none   observed in no conformer at all
    pl50   pLDDT < 50 (a model-only definition, no experimental data)
    pl70   pLDDT < 70

For each definition the model's pocket SASA is recomputed with those residues ignored (PyMOL flag
ignore, the same mechanism the SASA uses for ligands), in the v2.1.0 frame (archived US-align matrix
re-applied, which reproduces the archived value exactly), and the entry is re-scored by the released
classifier (kde_api.classify_point_kde_r3, full_run_3 settings) with only the model's SASA
replaced. Volume, RoG and pocket RMSD are left as published, which the list above shows is correct.

    python af_model_occlusion.py sifts                                  -> compact SIFTS index
    python af_model_occlusion.py collect [--limit N] [--sample N] [--procs P]  -> per-entry CSV
    python af_model_occlusion.py verify                                 -> volume and RoG ignore the rest
    python af_model_occlusion.py summarise                              -> the Supplementary Note 1 numbers

`collect` needs PyMOL (a Python 3.11 env with PyMOL 3.1, where the classification code imports cleanly) and
prints its fidelity line: recomputed model SASA equal to the archived value, and recomputed P(holo) equal to
the released one. `summarise` needs no PyMOL. Set AHOJ_AF_MODEL_SET=afdbv4 for the detection and docking
tables the manuscript quotes (the version 4 models; Supplementary Note 4).

All files go to the manuscript-analyses folder (ma_paths.OUT). The per-entry CSV (~238k rows) is
regenerable; the summary CSV is the published table.
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ahoj_job  # job archives under the names before AHoJ 2.6.0

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = str(P.WD)
UNI = str(P.cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
DB = str(P.cfg.AHOJ_DB_DIR)
MERGED = str(P.cfg.CLASSIFICATION_MERGED_DIR)
SUMMARY = str(P.cfg.CLASSIFICATION_SUMMARIES_CSV)
PARAMS = os.path.join(DB, "Summaries", "apoholo_metric_stats", "full_run_3", "parameters.json")
ROOT = str(P.cfg.AHOJ_OUTPUT_FIX_DIR)
STORE = str(P.anchor("afdb_store_dir", "AHOJ_AFDB_STORE_DIR"))   # AFDB model store with v4/ and v3/ subfolders
SIFTS_DIR = str(P.anchor("sifts_dir", "AHOJ_SIFTS_DIR"))         # SIFTS 2026-09-13 flat files

OUT_DIR = str(P.OUT)
SIFTS_NPZ = os.path.join(OUT_DIR, "sifts_index_2026-09-13.npz")
ENTRIES_CSV = os.path.join(OUT_DIR, "occlusion_entries.csv")
CSV_DIR = OUT_DIR

METRICS = ["pocket_rms", "sasa", "volume", "RoG"]
KW = dict(kde_extend=True, margin_cap_scale=1.0, bw_mode="pooled", bw_scale=0.75, prior_mode="equal")
DEFS = ["q", "maj", "none", "pl50", "pl70"]
NEAR = 10.0       # residues considered at all (A, any atom to any pocket atom)
CTX = 12.0        # SASA context narrowing: atoms beyond this from the pocket are ignored (exact, see collect)
SHELLS = (4.0, 5.0)
CHECK_EVERY = 50  # every Nth entry also gets the SASA in the full, un-narrowed context

_ARGS = None
_SIFTS = None
_PM = None


# ============================================================================================ SIFTS
def _read_sifts(name):
    path = os.path.join(SIFTS_DIR, name)
    with open(path, encoding="utf-8") as fh:
        stamp = fh.readline().strip()
    d = pd.read_csv(path, sep="\t", skiprows=1, usecols=["PDB", "CHAIN", "SP_PRIMARY", "SP_BEG", "SP_END"],
                    dtype={"PDB": str, "CHAIN": str, "SP_PRIMARY": str}, keep_default_na=False)
    d["SP_BEG"] = pd.to_numeric(d.SP_BEG, errors="coerce")
    d["SP_END"] = pd.to_numeric(d.SP_END, errors="coerce")
    d = d.dropna(subset=["SP_BEG", "SP_END"])
    d["key"] = d.PDB.str.lower() + "|" + d.CHAIN + "|" + d.SP_PRIMARY
    d = d.sort_values(["key", "SP_BEG"], kind="stable")
    keys, first = np.unique(d.key.to_numpy(), return_index=True)
    return stamp, keys, first, d.SP_BEG.to_numpy(np.int32), d.SP_END.to_numpy(np.int32), len(d)


def build_sifts():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    for tag, name in (("obs", "uniprot_segments_observed.tsv"), ("sqr", "pdb_chain_uniprot.tsv")):
        stamp, keys, first, beg, end, n = _read_sifts(name)
        print(f"{name}: {stamp}; {n:,} segments, {len(keys):,} (pdb, chain, UniProt) keys")
        out[f"{tag}_keys"] = keys.astype("S")
        out[f"{tag}_first"] = np.append(first, len(beg)).astype(np.int64)
        out[f"{tag}_beg"], out[f"{tag}_end"] = beg, end
        out[f"{tag}_stamp"] = np.array(stamp)
    np.savez(SIFTS_NPZ, **out)
    print(f"wrote {SIFTS_NPZ}")


class Sifts:
    def __init__(self, path=SIFTS_NPZ):
        z = np.load(path)
        self.t = {tag: (z[f"{tag}_keys"], z[f"{tag}_first"], z[f"{tag}_beg"], z[f"{tag}_end"]) for tag in ("obs", "sqr")}
        self.cache = {}

    def seg(self, tag, pdb, chain, acc):
        """-> (beg, end) arrays of UniProt segments for one chain of one PDB entry and one accession."""
        k = (tag, pdb, chain, acc)
        if k in self.cache:
            return self.cache[k]
        keys, first, beg, end = self.t[tag]
        kb = f"{pdb.lower()}|{chain}|{acc}".encode()
        i = int(np.searchsorted(keys, kb))
        if i < len(keys) and keys[i] == kb:
            r = (beg[first[i]:first[i + 1]], end[first[i]:first[i + 1]])
        else:
            r = (np.zeros(0, np.int32), np.zeros(0, np.int32))
        self.cache[k] = r
        return r

    def covered(self, tag, pdb, chains, acc, pos):
        """Boolean over `pos`: position inside a segment of any of `chains`."""
        hit = np.zeros(len(pos), bool)
        for c in chains:
            b, e = self.seg(tag, pdb, c, acc)
            if len(b):
                hit |= ((pos[:, None] >= b[None, :]) & (pos[:, None] <= e[None, :])).any(axis=1)
        return hit


# ============================================================================================ archive
def _read_members(tgz, names):
    out = {}
    with tarfile.open(tgz, "r:gz") as tar:
        have = set(tar.getnames())
        for n in names:
            for c in (f"./{n}", n):
                if c in have:
                    out[n] = tar.extractfile(c).read()
                    break
    return out


def _matrix(lines):
    """v2.1.0 tmalign4b parse: the three rows after the rotation-matrix header -> 4x4 row-major."""
    m, row = [], 0
    for ln in lines:
        if 4 >= row > 0:
            if row >= 2:
                v = list(map(float, ln.split()))
                m.extend(v[2:5])
                m.append(v[1])
            row += 1
        elif ln.lower().startswith("------ the rotation matrix"):
            row = 1
    return m + [0, 0, 0, 1] if len(m) == 12 else None


def _init():
    global _ARGS, _SIFTS
    with open(PARAMS, encoding="utf-8") as fh:
        _ARGS = SimpleNamespace(**json.load(fh))
    _SIFTS = Sifts()


def _curate(df):
    """Classifier's own curation (balanced_classification._load): -> df_c, apo, holo, model row."""
    from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions
    from parse_AHoJ3_DBv2b import replace_zero_with_nan
    a = _ARGS
    if int(a.relabel_apoholo) == 1:
        df, _ = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=None, verbose=False)
    df = replace_zero_with_nan(df, a.columns_nozero)
    s = pd.to_numeric(df["#resis3"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v)
    df_q, df_af = df[df["role"] == "Q"], df[df["role"] == "M"]
    df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=a.full_pockets_aligned,
                              bndgrsds_prcnt_seqres=a.bndgrsds_prcnt_seqres,
                              bndgrsds_prcnt_obs=a.bndgrsds_prcnt_obs, resolution_soft=a.resolution_soft,
                              rog_diff_threshold_deltaQ=a.rog_diff_thrs_query,
                              rog_diff_threshold_mean=a.rog_diff_thrs_mean, pocket_distance=a.pocket_dist,
                              pocket_len_resis3=a.pocket_len_resis3,
                              drop_no_resolution=a.drop_no_resolution)
    df_c = pd.concat([df_q, df_f[df_f["role"] == "C"]], ignore_index=True)  # add_query_row = 1
    apo = df_c[df_c["apoholo_assignment"].isin(["A", "A*", "A**"])]
    holo = df_c[df_c["apoholo_assignment"].isin(["H", "H*"])]
    return df_c, apo, holo, df_af


def _classify(df_c, apo, holo, point):
    from kde_api import classify_point_kde_r3
    o = classify_point_kde_r3(df_c, apo, holo, point, METRICS, **KW)[0]
    return o["overall"]["avg_P(holo)"], o["overall"]["overall_classification"], o["sasa"]["P(holo)"]


# ============================================================================================ PyMOL
_GLOBAL_CMD = False   # `verify` needs the global pymol.cmd, because PyVOL drives that one and not pymol2


def _pymol():
    global _PM
    if _PM is None:
        if _GLOBAL_CMD:
            import pymol
            pymol.finish_launching(["pymol", "-qc"])
            _PM = SimpleNamespace(cmd=pymol.cmd)
        else:
            import pymol2
            _PM = pymol2.PyMOL()
            _PM.start()
        _PM.cmd.feedback("disable", "all", "everything")
    return _PM.cmd


def _model_path(acc):
    for sub, ver in (("v4", "v4"), ("v3", "v3")):
        p = os.path.join(STORE, sub, f"AF-{acc}-F1-model_{ver}.cif.gz")
        if os.path.isfile(p):
            return p
    return None


def _sasa(cmd, ignore_extra=None):
    """Model pocket SASA as v2.1.0 get_sasa measured it, in the narrowed context `ctx` (set by caller).
    `ignore_extra`: selection of residues to leave out (flag ignore), e.g. unmatched model segments."""
    cmd.flag("ignore", "all", "clear")
    cmd.flag("ignore", "hetatm and not polymer", "set")
    cmd.flag("ignore", "AFm and not ctx", "set")
    if ignore_extra:
        cmd.flag("ignore", ignore_extra, "set")
    cmd.set("dot_solvent", 1)
    return round(cmd.get_area("mp"), 2)


def _resi_sele(pairs):
    """[(segi, resv), ...] -> PyMOL selection over the merged model object."""
    by = {}
    for s, r in pairs:
        by.setdefault(s, []).append(int(r))
    return " or ".join(f"(AFm and segi {s} and resi {'+'.join(map(str, sorted(v)))})" for s, v in by.items())


# ============================================================================================ one entry
def process(job, loaded):
    """job = (job_name, tgz, index). `loaded`: per-worker set of accessions already loaded as src_<acc>."""
    job_name, tgz, idx = job
    rec = {"job_name": job_name}
    mem = _read_members(tgz, ["global_results.csv", "pocket_selections.csv", "script_parameters.json"])
    if "global_results.csv" not in mem:
        return {**rec, "status": "no_global_results"}
    df = ahoj_job.legacy_results(pd.read_csv(io.BytesIO(mem["global_results.csv"]), low_memory=False),
                                 mem.get("script_parameters.json"))
    df_c, apo, holo, df_af = _curate(df)
    if df_af.empty:
        return {**rec, "status": "no_model_row"}
    m = df_af.iloc[:1].copy()
    rec.update({"n_conf": len(df_c), "n_apo": len(apo), "n_holo_pool": len(holo),
                "sasa_arch": pd.to_numeric(m["sasa"], errors="coerce").iloc[0]})
    if len(apo) < 1 or len(holo) < 1:
        return {**rec, "status": "empty_class"}
    mats = str(m["aln_matrix"].iloc[0]).split() if pd.notna(m["aln_matrix"].iloc[0]) else []
    aflines = [ln for ln in mem.get("pocket_selections.csv", b"").decode("utf-8", "replace").splitlines()
               if ln.startswith("AF_")]
    if not mats or not aflines:
        return {**rec, "status": "no_matrix_or_selection"}
    copies = []
    for mn in mats:
        g = re.match(r"aln_AF-(.+?)-F1-model-v\d_to_(\w{4})(.+)\.txt$", mn)
        if not g:
            return {**rec, "status": f"bad_matrix_name:{mn}"}
        copies.append((g.group(1), g.group(2), g.group(3), mn))
    lines = []
    for ln in aflines:
        key, expr = ln.split(",", 1)
        g = re.match(r"AF_(.+?)_as_(\w{4})(.+?)\.", key)
        rs = re.search(r"resi ([0-9+\-]+)", expr)
        if not g or not rs:
            return {**rec, "status": "bad_af_selection"}
        lines.append((g.group(1), g.group(3), rs.group(1)))
    q_pdb = copies[0][1]
    accs = sorted({c[0] for c in copies})
    rec.update({"acc": ";".join(accs), "query_pdb": q_pdb, "n_copies": len(copies), "n_af_lines": len(lines),
                "n_acc": len(accs)})
    mat_bytes = _read_members(tgz, [f"matrices/{c[3]}" for c in copies])

    cmd = _pymol()
    for obj in cmd.get_names("objects"):
        if not obj.startswith("src_"):
            cmd.delete(obj)
    cmd.delete("mp")
    parts = []
    for i, (acc, pdb, qc, mn) in enumerate(copies):
        src = "src_" + re.sub(r"\W", "_", acc)
        if acc not in loaded:
            p = _model_path(acc)
            if p is None:
                return {**rec, "status": "no_model_file"}
            cmd.load(p, src)
            loaded.add(acc)
        raw = mat_bytes.get(f"matrices/{mn}")
        mat = _matrix(raw.decode("utf-8", "replace").splitlines()) if raw else None
        if mat is None:
            return {**rec, "status": "no_matrix_file"}
        name = f"c{i}"
        cmd.create(name, src)
        cmd.alter(f"{name} and chain A", f"chain='{qc}'")
        cmd.alter(name, f"segi='c{i}'")
        cmd.transform_object(name, mat, state=0, homogenous=1)
        parts.append(name)
    if len(parts) > 1:
        cmd.create("AFm", " or ".join(parts))
        for pn in parts:
            cmd.delete(pn)
    else:
        cmd.set_name(parts[0], "AFm")
    for acc, qc, resi in lines:     # v2.1.0: '<object> and chain <p_chain> and resi <resis>', merged
        cmd.select("mp", f"AFm and chain {qc} and resi {resi}", merge=1)
    n_mp = cmd.count_atoms("mp")
    if n_mp == 0:
        return {**rec, "status": "empty_model_pocket"}
    cmd.select("ctx", f"AFm and (mp expand {CTX})")

    # ---- pocket and near atoms
    P, N = [], []
    cmd.iterate_state(1, "mp", "P.append((segi, resv, x, y, z, b))", space={"P": P})
    cmd.iterate_state(1, f"(AFm and not mp) within {NEAR} of mp", "N.append((segi, resv, x, y, z, b))",
                      space={"N": N})
    pk = pd.DataFrame(P, columns=["segi", "resv", "x", "y", "z", "b"])
    rec["pocket_n_res"] = int(pk.groupby(["segi", "resv"]).ngroups)
    rec["pocket_plddt"] = float(pk.groupby(["segi", "resv"]).b.first().mean())
    seg_info = {f"c{i}": c for i, c in enumerate(copies)}
    pk_pos = {sg: np.array(sorted(g.resv.unique()), int) for sg, g in pk.groupby("segi")}
    # SIFTS drift guard. The archives were mapped with the 2024 SIFTS; the local copy is 2026-09-13, in which a
    # chain can map to another accession or an entry can be missing. The pocket itself is the test: the query
    # observes every pocket position (it defines them), and a curated candidate at least 75% of them
    # (%mapped_obs >= 75). A structure failing its test under the 2026 file is left out of the coverage count.
    q_ok = {}
    for sg, pos in pk_pos.items():
        acc, pdb, qc, _ = seg_info[sg]
        q_ok[sg] = float(_SIFTS.covered("obs", pdb, [qc], acc, pos).mean())
    rec["sifts_pocket_obs_frac"] = float(np.mean(list(q_ok.values())))
    rec["q_valid"] = bool(all(v >= 0.9 for v in q_ok.values()))

    near = pd.DataFrame(N, columns=["segi", "resv", "x", "y", "z", "b"])
    if len(near):
        from scipy.spatial import cKDTree
        dist, nn = cKDTree(pk[["x", "y", "z"]].to_numpy()).query(near[["x", "y", "z"]].to_numpy(), k=1)
        near["d"] = dist
        near["pk_segi"] = pk.segi.to_numpy()[nn]
        near = near.sort_values("d")
        res = near.groupby(["segi", "resv"], as_index=False).agg(d=("d", "min"), plddt=("b", "first"),
                                                                 pk_segi=("pk_segi", "first"))
    else:
        res = pd.DataFrame(columns=["segi", "resv", "d", "plddt", "pk_segi"])
    # a residue of one model copy lying on the pocket part of ANOTHER copy: the superposed full-length
    # monomers of a multi-chain pocket reaching into each other (not something a single model can do)
    res["cross"] = res.segi != res.pk_segi

    # ---- experimental coverage of the near residues, per model copy
    conf = [(str(r.structure).lower(), [c for c in str(r.chains3).split("-") if c and c != "nan"],
             str(r.apoholo_assignment)) for r in df_c.itertuples()]
    is_apo = np.array([a in ("A", "A*", "A**") for _, _, a in conf])
    valid = {}
    for acc in sorted({seg_info[sg][0] for sg in pk_pos}):
        pos = np.unique(np.concatenate([pk_pos[sg] for sg in pk_pos if seg_info[sg][0] == acc]))
        valid[acc] = np.array([_SIFTS.covered("obs", cp, cch, acc, pos).mean() >= 0.75 for cp, cch, _ in conf],
                              dtype=bool)
    rec["n_conf_valid"] = int(min(v.sum() for v in valid.values()))
    rec["n_apo_valid"] = int(min((v & is_apo).sum() for v in valid.values()))
    rec["n_holo_valid"] = int(min((v & ~is_apo).sum() for v in valid.values()))
    res["q_obs"] = False
    res["q_sqr"] = False
    res["n_obs"] = 0
    res["n_obs_apo"] = 0
    res["n_sqr"] = 0
    res["n_valid"] = 0
    res["n_valid_apo"] = 0
    for sg, sub in res.groupby("segi"):
        acc, pdb, qc, _ = seg_info[sg]
        pos = sub.resv.to_numpy(int)
        res.loc[sub.index, "q_obs"] = _SIFTS.covered("obs", pdb, [qc], acc, pos)
        res.loc[sub.index, "q_sqr"] = _SIFTS.covered("sqr", pdb, [qc], acc, pos)
        n_obs = np.zeros(len(pos), int)
        n_obs_apo = np.zeros(len(pos), int)
        n_sqr = np.zeros(len(pos), int)
        for k, (cp, cch, _) in enumerate(conf):
            if not valid[acc][k]:
                continue
            o = _SIFTS.covered("obs", cp, cch, acc, pos)
            n_obs += o
            if is_apo[k]:
                n_obs_apo += o
            n_sqr += _SIFTS.covered("sqr", cp, cch, acc, pos)
        res.loc[sub.index, "n_obs"] = n_obs
        res.loc[sub.index, "n_obs_apo"] = n_obs_apo
        res.loc[sub.index, "n_sqr"] = n_sqr
        res.loc[sub.index, "n_valid"] = int(valid[acc].sum())
        res.loc[sub.index, "n_valid_apo"] = int((valid[acc] & is_apo).sum())
    nv = res.n_valid.where(res.n_valid > 0)
    res["f_obs"] = res.n_obs / nv
    res["f_sqr"] = res.n_sqr / nv
    na_ = res.n_valid_apo.where(res.n_valid_apo > 0)
    nh_ = (res.n_valid - res.n_valid_apo).where(res.n_valid - res.n_valid_apo > 0)
    res["f_obs_apo"] = res.n_obs_apo / na_
    res["f_obs_holo"] = (res.n_obs - res.n_obs_apo) / nh_
    flag = {"q": ~res.q_obs.astype(bool), "maj": res.f_obs < 0.5, "none": res.n_obs == 0,
            "pl50": res.plddt < 50, "pl70": res.plddt < 70}
    # a definition is undefined where its reference cannot be read under the 2026 SIFTS
    undefined = {"q": not rec["q_valid"], "maj": rec["n_conf_valid"] == 0, "none": rec["n_conf_valid"] == 0,
                 "pl50": False, "pl70": False}
    rec["n_near10"] = len(res)
    for sh in SHELLS:
        w = res.d <= sh
        t = f"{sh:.0f}"
        rec[f"n{t}_all"] = int(w.sum())
        for dn, fl in flag.items():
            rec[f"n{t}_{dn}"] = np.nan if undefined[dn] else int((w & fl).sum())
        # the maj-unmatched residues split: outside most constructs vs inside a construct but disordered
        rec[f"n{t}_maj_outside_construct"] = (np.nan if undefined["maj"] else
                                              int((w & flag["maj"] & (res.f_sqr < 0.5)).sum()))
        rec[f"n{t}_q_outside_construct"] = (np.nan if undefined["q"] else
                                            int((w & flag["q"] & ~res.q_sqr.astype(bool)).sum()))
        rec[f"n{t}_maj_cross"] = np.nan if undefined["maj"] else int((w & flag["maj"] & res.cross).sum())
        # class-specific coverage: missing from most APO conformers, or from most HOLO conformers
        rec[f"n{t}_unobs_apo_major"] = int((w & (res.f_obs_apo < 0.5)).sum()) if rec["n_apo_valid"] else np.nan
        rec[f"n{t}_unobs_holo_major"] = int((w & (res.f_obs_holo < 0.5)).sum()) if rec["n_holo_valid"] else np.nan
    for dn, fl in flag.items():
        rec[f"n10_{dn}"] = np.nan if undefined[dn] else int(fl.sum())
        rec[f"dmin_{dn}"] = float(res.d[fl].min()) if (fl.any() and not undefined[dn]) else np.nan
    for sh in SHELLS:
        w = (res.d <= sh) & flag["maj"]
        rec[f"plddt_mean{sh:.0f}_maj"] = float(res.plddt[w].mean()) if w.any() else np.nan
    w = (res.d <= 5) & flag["maj"] & (not undefined["maj"])
    rec["res5_maj"] = " ".join(f"{s}:{int(r)}:{p:.0f}:{d:.1f}" for s, r, p, d in
                               res.loc[w, ["segi", "resv", "plddt", "d"]].itertuples(index=False))

    # ---- SASA: archived frame, narrowed context; each definition leaves its residues out
    rec["sasa_full"] = _sasa(cmd)
    if idx % CHECK_EVERY == 0:
        cmd.flag("ignore", "all", "clear")
        cmd.flag("ignore", "hetatm and not polymer", "set")
        rec["sasa_full_ctx"] = round(cmd.get_area("mp"), 2)
    for dn, fl in flag.items():
        sub = res[fl]
        if undefined[dn]:
            rec[f"sasa_{dn}"] = np.nan
        else:
            rec[f"sasa_{dn}"] = _sasa(cmd, _resi_sele(zip(sub.segi, sub.resv))) if len(sub) else rec["sasa_full"]

    # ---- experimental SASA range (the classifier's own curated values)
    for tag, g in (("apo", apo), ("holo", holo)):
        v = pd.to_numeric(g["sasa"], errors="coerce").dropna()
        rec[f"{tag}_sasa_min"] = float(v.min()) if len(v) else np.nan
        rec[f"{tag}_sasa_med"] = float(v.median()) if len(v) else np.nan
        rec[f"{tag}_sasa_max"] = float(v.max()) if len(v) else np.nan
        mo = pd.to_numeric(g["%mapped_obs"], errors="coerce")
        rec[f"{tag}_pocket_incomplete_frac"] = float((mo < 100).mean()) if len(mo) else np.nan

    # ---- re-score: released point, then the model's SASA replaced per definition
    point = m.copy()
    for k in METRICS:
        point[k] = pd.to_numeric(point[k], errors="coerce")
    ph, cl, ps = _classify(df_c, apo, holo, point)
    rec.update({"p_holo_full": ph, "cls_full": cl, "p_sasa_full": ps})
    done = {}
    for dn in DEFS:
        if undefined[dn]:
            rec.update({f"sasa_{dn}_cf": np.nan, f"p_holo_{dn}": np.nan, f"cls_{dn}": "", f"p_sasa_{dn}": np.nan})
            continue
        s_new = rec["sasa_arch"] + (rec[f"sasa_{dn}"] - rec["sasa_full"])   # exact when sasa_full == archived
        rec[f"sasa_{dn}_cf"] = round(s_new, 2)
        if abs(s_new - rec["sasa_arch"]) < 0.005:
            rec.update({f"p_holo_{dn}": ph, f"cls_{dn}": cl, f"p_sasa_{dn}": ps})
            continue
        key = round(s_new, 2)
        if key not in done:
            pt = point.copy()
            pt["sasa"] = key
            done[key] = _classify(df_c, apo, holo, pt)
        rec.update(dict(zip([f"p_holo_{dn}", f"cls_{dn}", f"p_sasa_{dn}"], done[key])))
    rec["status"] = "ok"
    return rec


def _work(group):
    """group = list of jobs of one accession key; models stay loaded for the whole group."""
    if _ARGS is None:
        _init()
    cmd = _pymol()
    loaded = set()
    out = []
    for job in group:
        t0 = time.time()
        try:
            rec = process(job, loaded)
        except Exception as e:  # noqa: BLE001
            rec = {"job_name": job[0], "status": f"error:{type(e).__name__}: {str(e)[:200]}"}
        rec["t_s"] = round(time.time() - t0, 3)
        out.append(rec)
    cmd.delete("all")
    return out


def job_groups(limit=None, sample=None, seed=0):
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job", "query_path", "uniprots_key"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].reset_index(drop=True)
    if sample:
        uni = uni.sample(n=sample, random_state=seed).reset_index(drop=True)
    elif limit:
        uni = uni.iloc[:limit]
    uni["batch"] = uni.query_path.astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-2]
    uni["tgz"] = [os.path.join(ROOT, b, f"{j}.tgz") for b, j in zip(uni.batch, uni.rep_job)]
    uni["idx"] = np.arange(len(uni))
    groups = [list(zip(g.rep_job, g.tgz, g.idx)) for _, g in uni.groupby("uniprots_key", sort=False)]
    # split very large accession groups so no single worker carries one for hours
    out = []
    for g in groups:
        for i in range(0, len(g), 400):
            out.append(g[i:i + 400])
    out.sort(key=len, reverse=True)
    return out, len(uni)


def collect(limit, sample, procs, out=ENTRIES_CSV):
    os.makedirs(OUT_DIR, exist_ok=True)
    groups, n = job_groups(limit, sample)
    print(f"entries: {n:,} in {len(groups):,} accession groups; procs={procs}", flush=True)
    t0 = time.time()
    rows, done = [], 0
    with Pool(procs) as pool:
        for recs in pool.imap_unordered(_work, groups, chunksize=1):
            rows.extend(recs)
            done += len(recs)
            if done // 10000 != (done - len(recs)) // 10000:
                print(f"  {done:,} done  {time.time() - t0:7.0f} s", flush=True)
                pd.DataFrame(rows).to_csv(out + ".partial", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    if os.path.exists(out + ".partial"):
        os.remove(out + ".partial")
    print(f"wrote {out}  rows={len(d):,}  {time.time() - t0:.0f} s")
    print(d.status.value_counts().head(20).to_string())
    ok = d[d.status == "ok"]
    if len(ok):
        eq = (ok.sasa_full - ok.sasa_arch).abs() < 0.005
        print(f"\nfidelity: recomputed model SASA equals the archived value for {100 * eq.mean():.3f}% of "
              f"{len(ok):,} entries (max |diff| {(ok.sasa_full - ok.sasa_arch).abs().max():.2f} A^2)")
        if "sasa_full_ctx" in ok:
            c = ok[ok.sasa_full_ctx.notna()]
            print(f"          narrowed context equals full context for {100 * ((c.sasa_full_ctx - c.sasa_full).abs() < 0.005).mean():.3f}% "
                  f"of {len(c):,} checked entries")
        s = pd.read_csv(SUMMARY, usecols=["job_name", "pocket_avg_P(holo)"], low_memory=False)
        x = ok.merge(s, on="job_name", how="left")
        fp = (x.p_holo_full.round(3) - x["pocket_avg_P(holo)"]).abs() < 0.0015
        print(f"          recomputed P(holo) equals the released value for {100 * fp.mean():.3f}% of {len(x):,}")
        print(f"          median time per entry {ok.t_s.median():.2f} s")


# ============================================================================================ verify
def _rgyrate_v21(cmd, sele):
    """v2.1.0 pml_functions.rgyrate: mass-weighted radius of gyration of the selection, 2 dp."""
    model = cmd.get_model(sele).atom
    x = np.array([a.coord for a in model])
    m = np.array([a.get_mass() for a in model])
    c = (x * m[:, None]).sum(0) / m.sum()
    return round(float(np.sqrt((m * ((x - c) ** 2).sum(1)).sum() / m.sum())), 2)


def _pyvol_v21(sele):
    """v2.1.0 get_pyvol_pocket_volume: PyVOL on the selection, min_volume 200, volume parsed from stdout."""
    import contextlib
    import shutil
    import tempfile
    if "float_" not in np.__dict__:          # PyVOL 1.7.8 still calls np.float_, removed in NumPy 2
        np.float_ = np.float64
    from pyvol.pymol_interface import pymol_pocket_cmdline
    tmp, cwd, buf = tempfile.mkdtemp(), os.getcwd(), io.StringIO()
    try:
        os.chdir(tmp)
        with contextlib.redirect_stdout(buf):
            pymol_pocket_cmdline(protein=sele, min_volume=200)
    except Exception:  # noqa: BLE001  (v2.1.0 sets the model's volume to 0 on failure)
        return 0.0
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    mm = re.search(r"Volume: (\d+\.?\d*) A\^3", buf.getvalue())
    return float(mm.group(1)) if mm else None


def verify(entries=ENTRIES_CSV, n_top=8, n_rand=6, seed=1):
    """The code says only SASA can see a non-pocket atom; this shows it on real entries. Each model pocket
    is rebuilt exactly as `collect` builds it, measured, then EVERY atom outside the pocket is deleted and
    the pocket measured again: PyVOL volume and RoG must not move, SASA must."""
    global _GLOBAL_CMD
    _GLOBAL_CMD = True
    os.environ["PATH"] = os.path.join(os.path.dirname(sys.executable), "bin") + os.pathsep \
        + os.path.join(os.path.dirname(sys.executable), "Library", "bin") + os.pathsep + os.environ["PATH"]  # msms
    _init()
    d = pd.read_csv(entries, low_memory=False)
    d = d[d.status == "ok"].copy()
    d["dsasa_maj"] = d.sasa_maj - d.sasa_full
    single, multi = d[d.n_copies == 1], d[d.n_copies > 1]
    pick = pd.concat([single.nlargest(n_top, "dsasa_maj"), multi.nlargest(n_top // 2, "dsasa_maj"),
                      d.sample(n_rand, random_state=seed)]).drop_duplicates("job_name")
    uni = pd.read_csv(UNI, usecols=["threshold", "rep_job", "query_path"], low_memory=False)
    uni = uni[uni.threshold == "UNCLUSTERED"].set_index("rep_job")
    cmd = _pymol()
    print("=" * 100)
    print("af_model_occlusion.py verify -- model pocket measured with the whole model present, then with every "
          "atom outside the pocket deleted")
    print("=" * 100)
    print(f"{'entry':24s}{'copies':>7s}{'vol arch':>10s}{'vol whole':>10s}{'vol pocket':>11s}{'RoG arch':>9s}"
          f"{'RoG whole':>10s}{'RoG pocket':>11s}{'SASA whole':>11s}{'SASA pocket':>12s}")
    same_v, same_r, n = 0, 0, 0
    for r in pick.itertuples():
        qp = str(uni.loc[r.job_name, "query_path"]).replace("\\", "/").split("/")
        job = (r.job_name, os.path.join(ROOT, qp[-2], f"{r.job_name}.tgz"), 1)
        for obj in cmd.get_names("objects"):
            cmd.delete(obj)
        rec = process(job, set())
        if rec.get("status") != "ok":
            print(f"{r.job_name:24s} {rec.get('status')}")
            continue
        gr = ahoj_job.read_global_results(job[1])
        arch = gr[gr.role == "M"].iloc[0]
        cmd.flag("ignore", "all", "clear")
        cmd.flag("ignore", "hetatm and not polymer", "set")
        cmd.set("dot_solvent", 1)
        s0, r0, v0 = round(cmd.get_area("mp"), 2), _rgyrate_v21(cmd, "mp"), _pyvol_v21("mp")
        cmd.remove("AFm and not mp")
        s1, r1, v1 = round(cmd.get_area("mp"), 2), _rgyrate_v21(cmd, "mp"), _pyvol_v21("mp")
        v0 = np.nan if v0 is None else v0      # PyVOL found no pocket: v2.1.0 leaves the value blank
        v1 = np.nan if v1 is None else v1
        n += 1
        same_v += int(v0 == v1 or (np.isnan(v0) and np.isnan(v1)))
        same_r += int(r0 == r1)
        print(f"{r.job_name:24s}{r.n_copies:>7d}{arch.volume:>10.1f}{v0:>10.1f}{v1:>11.1f}{arch.RoG:>9.2f}{r0:>10.2f}"
              f"{r1:>11.2f}{s0:>11.2f}{s1:>12.2f}")
    print(f"\nPyVOL volume unchanged (a blank, no pocket found, stays blank) when every non-pocket atom is deleted: "
          f"{same_v} of {n}; RoG unchanged: "
          f"{same_r} of {n}; SASA rises in every case shown")


# ============================================================================================ summarise
LEVELS =["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030"]
DEF_LABEL = {"q": "unobserved in the query chain", "maj": "observed in < half of the conformers",
             "none": "observed in no conformer", "pl50": "pLDDT < 50", "pl70": "pLDDT < 70"}
CLS_MAP = {"Apo": "apo", "Holo": "holo", "Ambiguous": "tie"}


def _pct(x):
    x = np.asarray(x, dtype=float)
    return 100.0 * float(np.nanmean(x)) if len(x) else np.nan


def _cls(p):
    """Site class from the mean member P(holo), as balanced_classification._cls: holo above 0.5."""
    p = np.round(np.asarray(p, dtype=float), 6)
    return np.where(np.isnan(p), "", np.where(p > 0.5, "holo", np.where(p < 0.5, "apo", "tie")))


def _q(x, qs=(0.25, 0.5, 0.75)):
    x = pd.Series(x).dropna()
    return [float(x.quantile(q)) for q in qs] if len(x) else [np.nan] * len(qs)


ZHANG = [("P00533", "EGFR"), ("P15056", "BRAF"), ("P07333", "CSF1R"), ("P10721", "KIT"), ("P00797", "RENI")]


def _ranges(col, top=4):
    """Most frequent model residues among the 'maj' residues within 5 A, as 'resv(count)'."""
    c = pd.Series([int(t.split(":")[1]) for v in col.dropna() for t in str(v).split()]).value_counts()
    return ", ".join(f"{r}({n})" for r, n in c.head(top).items()) if len(c) else "-"


def _entries(path):
    d = pd.read_csv(path, low_memory=False)
    print(f"entries in {os.path.basename(path)}: {len(d):,}")
    print("  status: " + ", ".join(f"{k} {v:,}" for k, v in d.status.value_counts().items()))
    d = d[d.status == "ok"].copy()
    u = pd.read_csv(UNI, usecols=["threshold", "rep_job", "pocket_class", "pocket_p_holo", "uniprots_key",
                                  "is_interface"], low_memory=False)
    u = u[u.threshold == "UNCLUSTERED"].rename(columns={"rep_job": "job_name", "pocket_class": "rel_cls",
                                                        "pocket_p_holo": "rel_p_holo"})
    d = d.merge(u.drop(columns="threshold"), on="job_name", how="left")
    d["multi"] = d.n_copies > 1
    for dn in DEFS:
        d[f"cls_{dn}"] = d[f"cls_{dn}"].map(CLS_MAP).fillna("")
        d[f"dsasa_{dn}"] = d[f"sasa_{dn}"] - d.sasa_full
    d["cls_full"] = d.cls_full.map(CLS_MAP).fillna("")
    return d


def _block_prevalence(d, rows):
    print("\n=== 1. how often does a model residue that the experiments lack sit on the pocket? ===")
    print("  a residue counts if any of its atoms is within 4 A (or 5 A) of a pocket atom; the entry's "
          "conformers are the query plus every curated apo and holo candidate")
    groups = [("all entries", d), ("single-copy model (one chain carries the pocket)", d[~d.multi]),
              ("multi-copy model (pocket spans chains)", d[d.multi]),
              ("released AF-Apo", d[d.rel_cls == "apo"]), ("released AF-Holo", d[d.rel_cls == "holo"])]
    for dn in DEFS:
        print(f"\n  definition '{dn}': {DEF_LABEL[dn]}")
        for lab, g in groups:
            ok = g[f"n4_{dn}"].notna()
            p4 = _pct(g.loc[ok, f"n4_{dn}"] > 0)
            p5 = _pct(g.loc[ok, f"n5_{dn}"] > 0)
            n4 = int((g.loc[ok, f"n4_{dn}"] > 0).sum())
            print(f"    {lab:50s} defined {int(ok.sum()):>7,}; within 4 A {n4:>6,} ({p4:5.2f}%), within 5 A {p5:5.2f}%")
            rows.append({"block": "prevalence", "definition": dn, "group": lab, "n": int(ok.sum()),
                         "n_within4": n4, "pct_within4": p4, "pct_within5": p5})
    for lab, g in groups[:3]:
        w = g[g.n4_maj > 0]
        med = _q(w.n4_maj)
        print(f"\n  '{lab}', entries with a 'maj' residue within 4 A (n = {len(w):,}): residues per entry "
              f"median {med[1]:.0f} (IQR {med[0]:.0f}-{med[2]:.0f}); mean pLDDT of those residues median "
              f"{np.nanmedian(w.plddt_mean4_maj):.1f}; share of those residues outside most constructs "
              f"{100 * w.n4_maj_outside_construct.sum() / max(w.n4_maj.sum(), 1):.1f}%; from another model "
              f"copy {100 * w.n4_maj_cross.sum() / max(w.n4_maj.sum(), 1):.1f}%")
    # pLDDT of the unmatched residues: well-predicted segments the crystals lack, or disordered ones?
    w = d[(d.n4_maj > 0) & ~d.multi]
    print(f"\n  single-copy entries with a 'maj' residue within 4 A: mean pLDDT of those residues >= 70 in "
          f"{_pct(w.plddt_mean4_maj >= 70):.1f}%, < 50 in {_pct(w.plddt_mean4_maj < 50):.1f}%")
    print(f"  overlap of definitions (single-copy, within 4 A): 'maj' and pLDDT < 70 both "
          f"{_pct(((w.n4_pl70 > 0))):.1f}% of the 'maj' entries; pLDDT < 70 entries also 'maj' "
          f"{_pct((d.loc[(d.n4_pl70 > 0) & ~d.multi, 'n4_maj'] > 0)):.1f}%")


def _block_sasa(d, rows):
    print("\n=== 2. how much pocket SASA do those residues take from the model? ===")
    print("  dSASA = model pocket SASA with the definition's residues ignored minus as published (>= 0)")
    for dn in DEFS:
        for lab, g in (("all", d), ("single-copy", d[~d.multi]), ("multi-copy", d[d.multi])):
            ok = g[f"dsasa_{dn}"].notna()
            ds, s0 = g.loc[ok, f"dsasa_{dn}"], g.loc[ok, "sasa_full"]
            hit = ds > 0.005
            rel = 100 * ds[hit] / s0[hit].where(s0[hit] > 0)
            qa, qr = _q(ds[hit]), _q(rel)
            p10 = _pct(100 * ds / s0.where(s0 > 0) > 10)
            print(f"  {dn:5s} {lab:12s} entries {int(ok.sum()):>7,}: SASA changes for {int(hit.sum()):>6,} "
                  f"({_pct(hit):5.2f}%); among them median +{qa[1]:.1f} A^2 (IQR {qa[0]:.1f}-{qa[2]:.1f}), "
                  f"+{qr[1]:.1f}% (IQR {qr[0]:.1f}-{qr[2]:.1f}); > 10% of the pocket's SASA in {p10:.2f}% of all")
            rows.append({"block": "dsasa", "definition": dn, "group": lab, "n": int(ok.sum()),
                         "n_changed": int(hit.sum()), "pct_changed": _pct(hit), "median_dsasa": qa[1],
                         "median_rel_pct": qr[1], "pct_over10pct": p10})
    # below the experimental range
    lo = d[["apo_sasa_min", "holo_sasa_min"]].min(axis=1)
    hi = d[["apo_sasa_max", "holo_sasa_max"]].max(axis=1)
    print("\n  model SASA against the experimental range (min and max over the curated apo and holo "
          "conformers, query included):")
    for dn in ["full"] + DEFS:
        s = d.sasa_arch if dn == "full" else d[f"sasa_{dn}_cf"]
        ok = s.notna() & lo.notna()
        below, above = (s < lo)[ok], (s > hi)[ok]
        print(f"    {dn:5s} n = {int(ok.sum()):,}: below the range {_pct(below):5.2f}%, above {_pct(above):5.2f}%")
        rows.append({"block": "range", "definition": dn, "group": "all", "n": int(ok.sum()),
                     "pct_below": _pct(below), "pct_above": _pct(above)})
    w = d[d.dsasa_maj > 0.005]
    ok = w.sasa_maj_cf.notna()
    lo_w, hi_w = lo[w.index], hi[w.index]
    print(f"    among the {len(w):,} entries whose SASA the 'maj' residues lower: below the range "
          f"{_pct((w.sasa_arch < lo_w)[ok]):.1f}% as published, {_pct((w.sasa_maj_cf < lo_w)[ok]):.1f}% without them; "
          f"above {_pct((w.sasa_arch > hi_w)[ok]):.1f}% and {_pct((w.sasa_maj_cf > hi_w)[ok]):.1f}%")
    # direction: at the sites concerned, is the apo or the holo state the more exposed one?
    ok = w.apo_sasa_med.notna() & w.holo_sasa_med.notna()
    print(f"    at those entries the holo median SASA exceeds the apo median in {_pct((w.holo_sasa_med > w.apo_sasa_med)[ok]):.1f}% "
          f"(all entries: {_pct((d.holo_sasa_med > d.apo_sasa_med)[d.apo_sasa_med.notna() & d.holo_sasa_med.notna()]):.1f}%)")


def _block_calls(d, rows):
    print("\n=== 3. what does it do to the calls? (entry level; the model's SASA replaced, all else as released) ===")
    fid = (d.p_holo_full.round(3) - d.rel_p_holo).abs() < 0.0015
    print(f"  fidelity: recomputed P(holo) equals the released value for {_pct(fid):.3f}% of {len(d):,}; "
          f"recomputed class equals the released class for {_pct(d.cls_full == d.rel_cls):.3f}%")
    for dn in DEFS:
        for lab, g in (("all", d), ("single-copy", d[~d.multi]), ("multi-copy", d[d.multi])):
            ok = (g[f"cls_{dn}"] != "") & g.cls_full.isin(["apo", "holo"]) & g[f"cls_{dn}"].isin(["apo", "holo", "tie"])
            a, b = g.loc[ok, "cls_full"], g.loc[ok, f"cls_{dn}"]
            h2a, a2h = int(((a == "holo") & (b == "apo")).sum()), int(((a == "apo") & (b == "holo")).sum())
            tie = int((b == "tie").sum())
            dp = (g.loc[ok, f"p_holo_{dn}"] - g.loc[ok, "p_holo_full"])
            share0 = _pct(a == "holo")
            share1 = _pct(b[b != "tie"] == "holo")
            print(f"  {dn:5s} {lab:12s} n = {int(ok.sum()):>7,}: calls changed {h2a + a2h:>6,} ({100 * (h2a + a2h) / max(ok.sum(), 1):.2f}%; "
                  f"holo -> apo {h2a:,}, apo -> holo {a2h:,}, -> tie {tie:,}); AF-Holo share {share0:.2f}% -> {share1:.2f}%; "
                  f"mean dP(holo) {dp.mean():+.4f}, |dP| > 0.05 in {_pct(dp.abs() > 0.05):.2f}%")
            rows.append({"block": "calls", "definition": dn, "group": lab, "n": int(ok.sum()),
                         "n_changed": h2a + a2h, "pct_changed": 100 * (h2a + a2h) / max(ok.sum(), 1),
                         "holo_to_apo": h2a, "apo_to_holo": a2h, "to_tie": tie, "holo_share_released": share0,
                         "holo_share_corrected": share1, "mean_dp": dp.mean()})
    # changed calls: how close to the boundary were they?
    ch = d[(d.cls_maj != d.cls_full) & d.cls_maj.isin(["apo", "holo"]) & d.cls_full.isin(["apo", "holo"])]
    print(f"  'maj' changed calls (n = {len(ch):,}): median |P(holo) - 0.5| as released {np.median(np.abs(ch.p_holo_full - 0.5)):.3f} "
          f"(all entries {np.median(np.abs(d.p_holo_full - 0.5)):.3f}); SASA's own P(holo) moves by median "
          f"{np.nanmedian(np.abs(ch.p_sasa_maj - ch.p_sasa_full)):.3f}")


def _site_frame(d):
    """Per clustering level and site: mean member P(holo), released and corrected per definition."""
    u = pd.read_csv(UNI, usecols=["threshold", "unit_uid", "job_names", "pocket_class", "pocket_p_holo"],
                    low_memory=False)
    u = u[u.threshold.isin(LEVELS)].copy()
    ex = u[["threshold", "unit_uid", "job_names"]].assign(job_name=u.job_names.str.split(";")).explode("job_name")
    cols = ["job_name", "p_holo_full"] + [f"p_holo_{dn}" for dn in DEFS] + ["n4_maj", "n4_pl70", "n4_q", "n4_none",
                                                                          "dsasa_maj", "multi"]
    ex = ex.drop(columns="job_names").merge(d[cols], on="job_name", how="left")
    ex["occ_maj4"] = ex.n4_maj > 0
    ex["occ_pl70_4"] = ex.n4_pl70 > 0
    ex["occ_q4"] = ex.n4_q > 0
    ex["occ_none4"] = ex.n4_none > 0
    agg = {"n_mem": ("job_name", "size"), "n_ok": ("p_holo_full", "count"), "full_p": ("p_holo_full", "mean"),
           "occ_maj4": ("occ_maj4", "max"), "occ_maj4_frac": ("occ_maj4", "mean"), "occ_pl70_4": ("occ_pl70_4", "max"),
           "occ_q4": ("occ_q4", "max"), "occ_none4": ("occ_none4", "max"), "multi": ("multi", "max")}
    for dn in DEFS:
        agg[f"p_{dn}"] = (f"p_holo_{dn}", "mean")
        agg[f"n_{dn}"] = (f"p_holo_{dn}", "count")
    s = ex.groupby(["threshold", "unit_uid"], sort=False).agg(**agg).reset_index()
    s = u.drop(columns="job_names").merge(s, on=["threshold", "unit_uid"], how="left")
    s["covered"] = s.n_ok == s.n_mem
    for dn in DEFS:
        changed = (s[f"p_{dn}"] - s.full_p).abs() > 1e-9
        ok = s.covered & (s[f"n_{dn}"] == s.n_mem)
        # unchanged sites keep the released label, so the tie rule cannot manufacture a flip
        s[f"cls_{dn}"] = np.where(~ok, "", np.where(changed, _cls(s[f"p_{dn}"]), s.pocket_class))
    return s


def _block_sites(s, rows):
    print("\n=== 4. AF-Holo share at every clustering level, released and with the model's SASA corrected ===")
    print("  (a site's P(holo) is the mean over its members; a site none of whose members changes keeps its label; ties excluded)")
    for lv in LEVELS:
        g = s[s.threshold == lv]
        fid = _pct(((g.full_p.round(3) - g.pocket_p_holo).abs() < 0.0015)[g.covered])
        rd = g.pocket_class.isin(["apo", "holo"]) & g.covered.fillna(False).astype(bool)
        line = (f"  {lv:12s} sites {len(g):>7,}, all members read {int(rd.sum()):,} (rebuilt = released {fid:.2f}%): "
                f"released {_pct(g.pocket_class[rd] == 'holo'):.2f}%")
        rec = {"block": "sites", "level": lv, "n": int(rd.sum()), "holo_share_released": _pct(g.pocket_class[rd] == "holo")}
        for dn in DEFS:
            c = g[f"cls_{dn}"]
            ok = c.isin(["apo", "holo"]) & rd     # released and corrected on ONE site set
            line += (f" | {dn} {_pct(c[ok] == 'holo'):.2f}% vs {_pct(g.pocket_class[ok] == 'holo'):.2f}% "
                     f"({_pct((c != g.pocket_class)[ok]):.2f}% changed)")
            rec[f"n_{dn}"] = int(ok.sum())
            rec[f"holo_share_{dn}"] = _pct(c[ok] == "holo")
            rec[f"holo_share_released_same_set_{dn}"] = _pct(g.pocket_class[ok] == "holo")
            rec[f"changed_pct_{dn}"] = _pct((c != g.pocket_class)[ok])
        rec["occluded_maj4_pct"] = _pct(g.occ_maj4.fillna(False).astype(bool))
        print(line)
        rows.append(rec)
    g = s[s.threshold == "ST_100"]
    print(f"  ST_100 sites with a 'maj' residue within 4 A in any member: {int(g.occ_maj4.fillna(False).sum()):,} "
          f"({_pct(g.occ_maj4.fillna(False).astype(bool)):.2f}%); pLDDT < 70 residue within 4 A: "
          f"{int(g.occ_pl70_4.fillna(False).sum()):,} ({_pct(g.occ_pl70_4.fillna(False).astype(bool)):.2f}%)")


def _block_downstream(s, rows):
    """Figure 4's estimator (separation_shape.Gap) on the detection and docking sets, with ONE set of
    cluster-bootstrap resamples per task so every difference below carries its own interval."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import separation_shape as S
    print("\n=== 5. detection and docking (ST_100; model set "
          f"{os.environ.get('AHOJ_AF_MODEL_SET', '') or 'July v6 (AHOJ_AF_MODEL_SET unset)'}) ===")
    lbs = S.load_lbs()
    st = s[s.threshold == "ST_100"].drop(columns=["threshold", "pocket_class", "pocket_p_holo"])
    x = lbs.merge(st, on="unit_uid", how="left")
    print(f"  detection pockets {len(x):,}; joined to the occlusion table {int(x.covered.eq(True).sum()):,}; "
          f"multi-copy members {int(x.multi.eq(True).sum()):,}")
    x = x[x.covered.eq(True)].copy()
    for c in ("occ_maj4", "occ_pl70_4", "occ_q4", "occ_none4"):
        x[c] = x[c].eq(True)
    x["cls_rel"] = x.cls
    for cl in ("apo", "holo"):
        g = x[x.cls == cl]
        print(f"  AF-{cl:4s} n = {len(g):,}: 'maj' residue within 4 A {_pct(g.occ_maj4):.2f}%, pLDDT < 70 within 4 A "
              f"{_pct(g.occ_pl70_4):.2f}%, unobserved in the query {_pct(g.occ_q4):.2f}%, in no conformer {_pct(g.occ_none4):.2f}%")
    dk = S.load_docking(x)
    tasks = [("consensus", x)] + [(t, x) for t in S.PRED] + [("docking", dk)]
    flags = [("maj4", "occ_maj4"), ("pl70_4", "occ_pl70_4"), ("q4", "occ_q4"), ("none4", "occ_none4")]
    for task, fr in tasks:
        g0 = S.Gap(fr.assign(cls=fr.cls_rel), task)
        n = len(g0.d)
        zero = np.zeros(n, dtype=int)
        p0, b0 = g0.point(zero, 1)[0], g0.boot(zero, 1)[:, 0]
        rt = g0.rates(zero, 1, raw=True)[0]
        print(f"\n  --- {task}: n = {n:,} pockets ({g0.d.site_uniprot.nunique():,} UniProt clusters); raw rates AF-Apo "
              f"{rt[0]:.1f}, AF-Holo {rt[1]:.1f} (gap {rt[1] - rt[0]:+.1f} pp) ---")
        S.show("size-standardized gap, all pockets", p0, b0, n, width=58)
        lo, hi = S.ci(b0)
        rows.append({"block": "gap", "task": task, "variant": "released, all", "n": n, "gap": p0, "lo": lo, "hi": hi})
        for tag, col in flags:
            f_ = g0.d[col].to_numpy(bool)
            cell = np.where(f_, -1, 0)
            p1, b1 = g0.point(cell, 1)[0], g0.boot(cell, 1)[:, 0]
            S.show(f"  without the {int(f_.sum()):,} '{tag}' pockets", p1, b1, int((~f_).sum()), width=58)
            S.show(f"    difference (without minus all)", p1 - p0, b1 - b0, width=58)
            lo, hi = S.ci(b1)
            lo2, hi2 = S.ci(b1 - b0)
            rows.append({"block": "gap", "task": task, "variant": f"released, without {tag}", "n": int((~f_).sum()),
                         "gap": p1, "lo": lo, "hi": hi, "diff": p1 - p0, "diff_lo": lo2, "diff_hi": hi2})
            # are the flagged pockets harder to find or dock, in either class?
            cc = np.where(f_, 1, 0)
            r = g0.rates(cc, 2, raw=True)
            if tag in ("maj4", "pl70_4"):
                print(f"      raw success, unflagged | flagged: AF-Apo {r[0, 0]:.1f} | {r[1, 0]:.1f}, "
                      f"AF-Holo {r[0, 1]:.1f} | {r[1, 1]:.1f}")
        # corrected labels on the same row set and the same resamples
        for dn in ("maj", "q", "pl70"):
            lab = fr[f"cls_{dn}"].where(fr[f"cls_{dn}"].isin(["apo", "holo"]), fr.cls_rel)
            g1 = S.Gap(fr.assign(cls=lab), task)
            assert len(g1.d) == n
            g1._boot_ix = g0._indices()
            p1, b1 = g1.point(zero, 1)[0], g1.boot(zero, 1)[:, 0]
            nch = int((g1.d.cls.to_numpy() != g0.d.cls.to_numpy()).sum())
            S.show(f"  corrected '{dn}' labels ({nch:,} pockets change class)", p1, b1, n, width=58)
            S.show(f"    difference (corrected minus released)", p1 - p0, b1 - b0, width=58)
            lo, hi = S.ci(b1)
            lo2, hi2 = S.ci(b1 - b0)
            rows.append({"block": "gap", "task": task, "variant": f"corrected {dn}", "n": n, "gap": p1, "lo": lo,
                         "hi": hi, "diff": p1 - p0, "diff_lo": lo2, "diff_hi": hi2, "n_changed": nch})
        # separation gradient without the occluded pockets
        c = S.codes_from(g0.d["sep"], S.TIERS)
        cm = np.where(g0.d["occ_maj4"].to_numpy(bool), -1, c)
        pa, ba = g0.point(c, 5), g0.boot(c, 5)
        pm, bm = g0.point(cm, 5), g0.boot(cm, 5)
        print("      by separation tier, all | without 'maj4':  " + "  ".join(
            f"{t} {pa[i]:+.1f} | {pm[i]:+.1f}" for i, t in enumerate(S.TIERS)))
        S.show("      very high minus very low, all", pa[4] - pa[0], ba[:, 4] - ba[:, 0], width=58)
        S.show("      very high minus very low, without 'maj4'", pm[4] - pm[0], bm[:, 4] - bm[:, 0], width=58)
        for i, t in enumerate(S.TIERS):
            lo, hi = S.ci(ba[:, i])
            lo2, hi2 = S.ci(bm[:, i])
            rows.append({"block": "gap_tier", "task": task, "variant": t, "n": int((c == i).sum()), "gap": pa[i],
                         "lo": lo, "hi": hi, "gap_without_maj4": pm[i], "lo_without": lo2, "hi_without": hi2})


def summarise(entries=ENTRIES_CSV):
    print("=" * 100)
    print("af_model_occlusion.py summarise -- occlusion of the model pocket by segments the experiments lack")
    print("=" * 100)
    d = _entries(entries)
    eq = (d.sasa_full - d.sasa_arch).abs() < 0.005
    print(f"\n=== 0. fidelity and coverage ===\n  entries {len(d):,}; recomputed model SASA equals the archived value "
          f"for {_pct(eq):.3f}% (max |diff| {(d.sasa_full - d.sasa_arch).abs().max():.2f} A^2)")
    dd = (d.sasa_full - d.sasa_arch).abs()
    big = d[dd > 0.0105]
    print(f"  of the {int((~eq).sum()):,} that differ, {int(((dd >= 0.005) & (dd <= 0.0105)).sum()):,} by 0.01 A^2 (rounding, "
          f"PyMOL 3.0 then, 3.1 now) and {len(big):,} by more ({big.job_name.str[:4].nunique()} PDB entries, accessions "
          f"{', '.join(sorted(big.acc.unique()))}); the corrected SASA is the archived value plus the recomputed change")
    c = d[d.sasa_full_ctx.notna()]
    print(f"  narrowed SASA context equals the full context for {_pct((c.sasa_full_ctx - c.sasa_full).abs() < 0.005):.3f}% "
          f"of {len(c):,} checked entries")
    print(f"  SIFTS 2026-09-13 guard: query readable for {_pct(d.q_valid):.2f}% of entries; conformers readable "
          f"{100 * d.n_conf_valid.sum() / d.n_conf.sum():.2f}% of {int(d.n_conf.sum()):,}; entries with none readable "
          f"{int((d.n_conf_valid == 0).sum()):,}")
    print(f"  multi-copy models (pocket spans chains): {int(d.multi.sum()):,} entries ({_pct(d.multi):.2f}%); "
          f"of them with more than one accession {int((d.n_acc > 1).sum()):,}; with more model copies than pocket "
          f"selections {int((d.n_copies > d.n_af_lines).sum()):,}")
    with open(PARAMS, encoding="utf-8") as fh:
        par = json.load(fh)
    t = par["rog_diff_thrs_query"]
    print(f"  curation (read on the way, not an occlusion result): rog_diff_thrs_query = {t} in parameters.json, applied by "
          f"sort_dataframe_v2c as query_rog * (1 +/- {t}), a RELATIVE window of +/-{100 * t:.0f}%, not an absolute one "
          f"in Angstrom; %mapped_obs >= {par['bndgrsds_prcnt_obs']} and %mapped_sqr >= {par['bndgrsds_prcnt_seqres']}")
    rows = []
    _block_prevalence(d, rows)
    _block_sasa(d, rows)
    _block_calls(d, rows)
    print("\n=== 3b. class-specific construct coverage near the pocket (experimental side only) ===")
    ok = d.n4_unobs_apo_major.notna() & d.n4_unobs_holo_major.notna()
    a_only = (d.n4_unobs_apo_major > d.n4_unobs_holo_major)[ok]
    h_only = (d.n4_unobs_holo_major > d.n4_unobs_apo_major)[ok]
    print(f"  entries with both classes readable {int(ok.sum()):,}: more residues within 4 A missing from most APO "
          f"conformers than from most HOLO ones {_pct(a_only):.2f}%, the reverse {_pct(h_only):.2f}%")
    for cl in ("apo", "holo"):
        v = d[f"{cl}_pocket_incomplete_frac"]
        print(f"  {cl} conformers with an incomplete pocket (%mapped_obs < 100): mean over entries {_pct(v):.2f}%")
    print("\n=== 3c. the targets zhang2023jcim names for occluding full-length segments ===")
    for acc, name in ZHANG:
        g = d[d.acc == acc]
        if not len(g):
            print(f"  {name:6s} {acc}: not in the analysis set")
            continue
        ch = (g.cls_maj != g.cls_full) & g.cls_maj.isin(["apo", "holo"])
        w = g[g.n4_maj > 0]
        print(f"  {name:6s} {acc}: {len(g):,} entries; 'maj' residue within 4 A {_pct(g.n4_maj > 0):.1f}%, pLDDT < 70 within 4 A "
              f"{_pct(g.n4_pl70 > 0):.1f}%; median dSASA where 'maj' lowers it {np.nanmedian(g.dsasa_maj[g.dsasa_maj > 0.005]) if (g.dsasa_maj > 0.005).any() else 0:.1f} A^2; "
              f"released AF-Holo {_pct(g.cls_full == 'holo'):.1f}% -> {_pct(g.cls_maj[g.cls_maj.isin(['apo', 'holo'])] == 'holo'):.1f}%, "
              f"calls changed {int(ch.sum()):,}; residue ranges within 5 A: {_ranges(w.res5_maj)}")
    print("\n=== 3d. where the changed calls are: concentration by accession ('maj') ===")
    ch = d[(d.cls_maj != d.cls_full) & d.cls_maj.isin(["apo", "holo"]) & d.cls_full.isin(["apo", "holo"])]
    per = ch.groupby("acc").size().sort_values(ascending=False)
    n_acc = d.acc.nunique()
    top = per.head(max(1, int(round(0.01 * n_acc))))
    print(f"  {len(ch):,} changed calls in {len(per):,} of {n_acc:,} accessions; the top 1% of accessions ({len(top):,}) carry "
          f"{100 * top.sum() / len(ch):.1f}% of them, the top 20 accessions {100 * per.head(20).sum() / len(ch):.1f}%")
    print("  top 20: accession, entries, changed (holo -> apo / apo -> holo), most frequent 'maj' residues within 5 A")
    for acc, k in per.head(20).items():
        g = d[d.acc == acc]
        c = ch[ch.acc == acc]
        print(f"    {acc:12s} {len(g):>5,} entries, {k:>4,} changed ({int((c.cls_full == 'holo').sum()):,} / "
              f"{int((c.cls_full == 'apo').sum()):,}); {_ranges(g[g.n4_maj > 0].res5_maj)}")
    s = _site_frame(d)
    _block_sites(s, rows)
    _block_downstream(s, rows)
    out = pd.DataFrame(rows)
    sfx = "_afdbv4" if os.environ.get("AHOJ_AF_MODEL_SET", "").strip().lower() == "afdbv4" else ""
    path = os.path.join(CSV_DIR, f"af_model_occlusion_summary{sfx}.csv")
    out.to_csv(path, index=False)
    print(f"\n[OK] -> {path}")


# ============================================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["sifts", "collect", "verify", "summarise"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default=ENTRIES_CSV)
    a = ap.parse_args()
    if a.mode == "sifts":
        build_sifts()
    elif a.mode == "collect":
        collect(a.limit, a.sample, a.procs, a.out)
    elif a.mode == "verify":
        verify(a.out)
    else:
        summarise(a.out)
