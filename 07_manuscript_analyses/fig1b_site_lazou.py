# -*- coding: utf-8 -*-
"""Figure 1b / SI Figure S9 showcase entry, AHoJ-DB 2yiw A I46 1354 (MAPK14 = p38alpha, UniProt Q16539).
Backs the Supplementary Figure S9 caption (panel d: Met198 of the AFDB model 1.3 A from Lazou's closed
reference and 7.7 A from the open one; the 148 apo + 117 holo states) and main Figure 1b.

Question: is its pocket the p38alpha lipid-binding site of Diskin, Engelberg & Livnah 2008 (`diskin2008lipid`),
which Lazou et al. 2024 (`lazou2024remember`) study as their "MAPK" case, and where does the AlphaFold DB
model sit on Lazou's open/closed axis?

    python fig1b_site_lazou.py > out.txt      (needs gemmi and pandas; imports the classification code)

Inputs, all read-only:
  * the per-job AHoJ archive <ahoj_output_fix_dir>/093/2yiw-A-I46-1354.tgz (common/paths.ini):
    pocket residues, SIFTS mapping, per-structure metrics and pocket ligands;
  * the full_run_3 classification summary (n_apo, n_holo) and the entry's classification_kde_pocket.json
    under <example_entries_dir>/093/2yiw-A-I46-1354/statistics (the per-metric P(apo) printed in Figure 1b);
  * the aligned chains of the same 2024-05-27 AHoJ run, kept in the webserver copy of the entry
    (<webserver_results_dir>/<job>/structure_files), and the deposited 2YIW mmCIF (struct_site), in the
    redocking structure cache (cfg.DOCKING_REDOCK_CACHE_DIR/structures);
  * the AlphaFold DB v4 model the entry was classified with (<afdb_store_dir>/v4).

Curated labels: the classifier's own curation (relabel -> zero-to-NaN -> volume rule -> sort_dataframe_v2c ->
role C), called in the same order as ensemble_containment.process, with the full_run_3 parameters. The script
asserts that it reproduces the released n_apo and n_holo.

Lazou's measure (PNAS 2024, Table 1 and Results): for the MAPK case the moving segment is the Met198 side chain,
and each structure is placed by the all-atom RMSD of that segment to the closed reference (2ZB1 chain A) and to
the open reference (2NPQ chain A, bound to beta-octyl glucoside, BOG), after superposition with the moving segment
excluded. Here the superposition uses the C-alpha atoms of every residue the two chains share except UniProt 198
(an iterative Kabsch fit, 5 cycles, pairs deviating by more than 2 A rejected, never below half the pairs), and
the RMSD is over the eight Met198 atoms. A structure is "closed-like" when it is nearer the closed reference.
Lazou's third category ("other", near neither reference) needs the threshold of their SI, which was not read,
so it is not reproduced. Residues are matched in UniProt numbering: each chain's offset is the mode of
(PDB - UniProt) over its SIFTS-mapped pocket residues, and residue 198 must be MET (or MSE) after the offset.

Outputs: stdout, and fig1b_site_lazou_met198.csv (one row per chain: the model, the query and the 264 curated
candidates), in the manuscript-analyses folder (ma_paths.OUT).
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import hashlib
import io
import json
import os
import sys
import tarfile
import zipfile
from collections import Counter

import gemmi
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ahoj_job  # noqa: E402  (job archives under the names before AHoJ 2.6.0)

JOB, BATCH, UNP = "2yiw-A-I46-1354", "093", "Q16539"
QUERY_PDB, QUERY_CHAIN, QUERY_LIG, QUERY_LIG_NUM = "2yiw", "A", "I46", 1354
OPEN_REF, CLOSED_REF = ("2npq", "A"), ("2zb1", "A")      # Lazou Table 1, MAPK row
APO_SHOWN = ("5xyy", "A")                                 # SI Figure S9d, the representative apo structure
MOVING = 198                                              # Lazou's moving segment for MAPK (Met198)
NEIGHBOUR = 197                                           # Trp197, drawn beside Met198 in SI Figure S9d
MET_ATOMS = ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"]
TRP_ATOMS = ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]

SUMMARY = str(P.cfg.CLASSIFICATION_SUMMARIES_CSV)
KDE_JSON = os.path.join(str(P.cfg.EXAMPLE_ENTRIES_DIR), BATCH, JOB, "statistics", "classification_kde_pocket.json")
STRUCT_DIR = os.path.join(str(P.anchor("webserver_results_dir", "AHOJ_WEBSERVER_RESULTS_DIR")), JOB,
                          "structure_files")
DEPOSITED_QUERY = os.path.join(str(P.cfg.DOCKING_REDOCK_CACHE_DIR), "structures", f"{QUERY_PDB}.cif.gz")
AF_MODEL = os.path.join(str(P.anchor("afdb_store_dir", "AHOJ_AFDB_STORE_DIR")), "v4", f"AF-{UNP}-F1-model_v4.cif.gz")

OUT_CSV = P.out("fig1b_site_lazou_met198.csv")


def output_fix_dir():
    return str(P.cfg.AHOJ_OUTPUT_FIX_DIR)


TGZ = os.path.join(output_fix_dir(), BATCH, f"{JOB}.tgz")


# ---------------------------------------------------------------------------------------------------------------
# archive readers
def read_archive():
    with tarfile.open(TGZ, "r:gz") as tar:
        def text(name):
            return tar.extractfile(name).read().decode("utf-8")
        params = json.loads(text("./script_parameters.json"))
        glob = ahoj_job.legacy_results(pd.read_csv(io.StringIO(text("./global_results.csv"))), params)
        pres = text("./pocket_residues.csv")
        z = zipfile.ZipFile(io.BytesIO(tar.extractfile("./residue_mappings.zip").read()))
        poi = json.loads(z.read("poi_dict_unp.json"))
        nested = json.loads(z.read("cndt_nested_UNPi_PDBi.json"))
    return glob, pres, params, poi, nested


def curated_labels(glob):
    """The classifier's curation, same calls and order as ensemble_containment.process."""
    from ensemble_containment import _load_args
    from parsing_functions_DBv2 import sort_dataframe_v2c, relabel_apoholo_by_ions
    from parse_AHoJ3_DBv2b import replace_zero_with_nan
    a = _load_args()
    df = glob.copy()
    if int(a.relabel_apoholo) == 1:
        df, _ = relabel_apoholo_by_ions(df, builtin="total_le1", save_dir=None, verbose=False)
    df = replace_zero_with_nan(df, a.columns_nozero)
    s = pd.to_numeric(df["#resis3"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = np.where(s.notna() & (s != 0) & v.isna(), 0.0, v)
    df_f = sort_dataframe_v2c(df, sorting=0, full_pockets_aligned=a.full_pockets_aligned,
                              bndgrsds_prcnt_seqres=a.bndgrsds_prcnt_seqres,
                              bndgrsds_prcnt_obs=a.bndgrsds_prcnt_obs, resolution_soft=a.resolution_soft,
                              rog_diff_threshold_deltaQ=a.rog_diff_thrs_query,
                              rog_diff_threshold_mean=a.rog_diff_thrs_mean, pocket_distance=a.pocket_dist,
                              pocket_len_resis3=a.pocket_len_resis3, drop_no_resolution=a.drop_no_resolution)
    df_c = df_f[df_f["role"] == "C"].copy()
    df_c["cls"] = np.where(df_c["apoholo_assignment"].isin(["A", "A*", "A**"]), "apo",
                           np.where(df_c["apoholo_assignment"].isin(["H", "H*"]), "holo", "other"))
    return df_c


# ---------------------------------------------------------------------------------------------------------------
# structures
def load(path, chain=None, keep_ligands=False):
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    if not keep_ligands:
        st.remove_ligands_and_waters()
    model = st[0]
    if chain is None:
        return model
    for ch in model:
        if ch.name == chain:
            return ch
    raise KeyError(f"{path}: chain {chain} not found")


def residues_by_unp(chain, offset):
    """UniProt number -> residue (polymer residues only; the first residue wins on an insertion code)."""
    out = {}
    for r in chain:
        if r.het_flag == "H" and r.name != "MSE":
            continue
        u = r.seqid.num - offset
        out.setdefault(u, r)
    return out


def atom_xyz(res, names):
    got = {}
    for a in res:
        nm = "SD" if (res.name == "MSE" and a.name == "SE") else a.name
        got.setdefault(nm, a.pos)
    if not all(n in got for n in names):
        return None
    return np.array([[got[n].x, got[n].y, got[n].z] for n in names])


def ca_pairs(mob, tgt, exclude):
    keys = sorted(k for k in mob if k in tgt and k not in exclude)
    p, q = [], []
    for k in keys:
        a, b = mob[k].find_atom("CA", "*"), tgt[k].find_atom("CA", "*")
        if a is not None and b is not None:
            p.append([a.pos.x, a.pos.y, a.pos.z])
            q.append([b.pos.x, b.pos.y, b.pos.z])
    return np.array(p), np.array(q)


def kabsch(p, q):
    pc, qc = p.mean(0), q.mean(0)
    h = (p - pc).T @ (q - qc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, qc - pc @ rot.T


def fit(p, q, cycles=5, cutoff=2.0):
    """Iterative Kabsch: refit on the pairs within `cutoff` A, at most `cycles` times, never below half."""
    mask = np.ones(len(p), bool)
    for _ in range(cycles + 1):
        rot, t = kabsch(p[mask], q[mask])
        dev = np.linalg.norm(p @ rot.T + t - q, axis=1)
        new = dev < cutoff
        if new.sum() < 0.5 * len(p) or (new == mask).all():
            break
        mask = new
    rot, t = kabsch(p[mask], q[mask])
    rms = float(np.sqrt(((p[mask] @ rot.T + t - q[mask]) ** 2).sum(1).mean()))
    return rot, t, int(mask.sum()), rms


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def offset_of(nested, pdb, chain):
    if pdb == QUERY_PDB and chain == QUERY_CHAIN:
        return 0
    m = nested.get(pdb, {}).get(chain, {}).get(UNP, {})
    offs = Counter(int(p) - int(u) for u, p in m.items() if p not in (None, "null", ""))
    return offs.most_common(1)[0][0] if offs else None


# ---------------------------------------------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 220, "display.max_columns", 30)
    glob, pres, params, poi, nested = read_archive()
    print(f"archive: {TGZ}")
    # the aligned chains come from the webserver copy of the entry: prove it is the archive's run
    with tarfile.open(TGZ, "r:gz") as tar:
        for name in ("global_results.csv", "pocket_residues.csv"):
            a = hashlib.md5(tar.extractfile(f"./{name}").read()).hexdigest()
            b = hashlib.md5(open(os.path.join(os.path.dirname(STRUCT_DIR), name), "rb").read()).hexdigest()
            assert a == b, f"{name}: the webserver copy is not the archive's run"
    print("webserver copy of the entry: global_results.csv and pocket_residues.csv have the archive's md5 (same run)")
    print(f"AHoJ {params['script_version']}, run {params['date']}, afdb_version {params['afdb_version']}, "
          f"ligand scan radius {params['lig_scan_radius']} A")

    # ---- 1. the entry and its classification
    print("\n=== 1. entry and classification (full_run_3) ===")
    s = pd.read_csv(SUMMARY, low_memory=False)
    row = s[s.job_name == JOB].iloc[0]
    print(f"{JOB}: n_apo {row.n_apo}, n_holo {row.n_holo} candidates; the query is holo by construction, so the "
          f"figure's experimental states are {row.n_apo} apo + {row.n_holo + 1} holo = {row.n_apo + row.n_holo + 1}")
    kde = json.load(open(KDE_JSON, encoding="utf-8"))
    for m in ("sasa", "pocket_rms", "volume", "RoG"):
        print(f"  P(apo) {m:10s} {kde[m]['P(apo)']:.3f}  (model value {kde[m]['model_value']})")
    print(f"  overall P(apo) {kde['overall']['avg_P(apo)']:.3f}, class {kde['overall']['overall_classification']};"
          f" summary pocket_avg_P(apo) {row['pocket_avg_P(apo)']}, model pocket pLDDT {row.model_pocket_pLDDT_avg}")

    # ---- 2. pocket residues against the deposited struct_site record
    print("\n=== 2. pocket residues ===")
    line = [ln for ln in pres.splitlines() if ln.startswith(f"{QUERY_PDB},")][0]
    pdb_res = [int(x.split("_")[1]) for x in line.split(",")[2].split()]
    unp_map = {int(k.split("_")[-1]): int(v.split("_")[1]) for k, v in poi[f"{QUERY_CHAIN}_{QUERY_LIG}_{QUERY_LIG_NUM}"].items()}
    names = {int(k.split("_")[-1]): k.split("_")[2] for k in poi[f"{QUERY_CHAIN}_{QUERY_LIG}_{QUERY_LIG_NUM}"]}
    assert all(unp_map[p] == p for p in pdb_res), "PDB and UniProt numbering differ"
    print(f"AHoJ pocket, {len(pdb_res)} residues (PDB author numbering = UniProt {UNP} numbering for all "
          f"{len(pdb_res)}): " + " ".join(f"{gemmi.find_tabulated_residue(names[p]).one_letter_code.upper()}{p}"
                                          for p in pdb_res))
    doc = gemmi.cif.read(DEPOSITED_QUERY)
    blk = doc.sole_block()
    sites = blk.find("_struct_site.", ["id", "pdbx_auth_comp_id", "pdbx_auth_seq_id"])
    sid = [r[0] for r in sites if r[1] == QUERY_LIG and int(r[2]) == QUERY_LIG_NUM][0]
    gen = blk.find("_struct_site_gen.", ["site_id", "auth_comp_id", "auth_seq_id", "symmetry"])
    site = [(r[1], int(r[2])) for r in gen if r[0] == sid and r[1] != "HOH"]
    n_wat = sum(1 for r in gen if r[0] == sid and r[1] == "HOH")
    site_res = [n for _, n in site]
    print(f"deposited 2YIW struct_site {sid} (I46 {QUERY_LIG_NUM}): {len(site_res)} residues + {n_wat} water: "
          + " ".join(f"{gemmi.find_tabulated_residue(c).one_letter_code.upper()}{n}" for c, n in site))
    extra = sorted(set(pdb_res) - set(site_res))
    missing = sorted(set(site_res) - set(pdb_res))
    print(f"struct_site residues inside the AHoJ pocket: {len(set(site_res) & set(pdb_res))} of {len(site_res)}; "
          f"AHoJ-only: {' '.join(f'{names[p][0]}{names[p][1:].lower()}{p}' for p in extra)}; "
          f"struct_site-only: {missing if missing else 'none'}")
    # nearest heavy-atom distance of each residue to I46, in the deposited 2YIW
    st = gemmi.read_structure(DEPOSITED_QUERY)
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    ch = [c for c in st[0] if c.name == QUERY_CHAIN][0]
    lig = [r for r in ch if r.name == QUERY_LIG and r.seqid.num == QUERY_LIG_NUM][0]
    lig_xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig])
    prot = {r.seqid.num: r for r in ch if r.het_flag != "H"}

    def mind(num):
        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in prot[num]])
        return float(np.min(np.linalg.norm(xyz[:, None] - lig_xyz[None], axis=2)))
    d = {p: mind(p) for p in pdb_res}
    print("nearest heavy-atom distance to I46 (A): " + ", ".join(f"{p} {d[p]:.2f}" for p in pdb_res))
    print(f"  struct_site residues: {min(d[p] for p in site_res):.2f} to {max(d[p] for p in site_res):.2f} A; "
          f"AHoJ-only residues: {min(d[p] for p in extra):.2f} to {max(d[p] for p in extra):.2f} A "
          f"(AHoJ scans {params['lig_scan_radius']} A)")
    for num in (NEIGHBOUR, MOVING):
        print(f"  {prot[num].name}{num} in the holo query: nearest heavy atom {mind(num):.2f} A from I46 "
              f"-> {'inside' if num in pdb_res else 'outside'} the pocket definition")

    # ---- 3. ligands in the pocket across the entry
    print("\n=== 3. what binds the pocket across the entry (uncurated candidates, as AHoJ reports them) ===")
    cand = glob[glob.role == "C"].copy()
    lab = cand.apoholo_assignment.value_counts().to_dict()
    print(f"candidate chains {len(cand)}: " + ", ".join(f"{k} {v}" for k, v in sorted(lab.items())))
    holo = cand[cand.apoholo_assignment.isin(["H", "H*"])]
    codes = Counter()
    per_chain = []
    for _, r in holo.iterrows():
        cc = set()
        for col in ("query_ligs", "nonquery_ligs"):
            if isinstance(r[col], str) and r[col].strip():
                cc.update(r[col].strip().split("-"))
        codes.update(cc)
        per_chain.append(cc)
    print(f"holo chains {len(holo)}; ligand codes in the pocket (chains carrying each): "
          + ", ".join(f"{k} {v}" for k, v in codes.most_common()))
    bog = holo[holo.nonquery_ligs.fillna("").str.split("-").apply(lambda x: "BOG" in x)]
    print(f"chains with BOG (n-octyl-beta-D-glucopyranoside) in the pocket: {len(bog)} of {len(holo)} holo chains, "
          f"in {bog.structure.nunique()} PDB entries")
    for pdb, chn in (OPEN_REF, ("2fst", "X"), CLOSED_REF, APO_SHOWN):
        r = glob[(glob.structure == pdb) & (glob.chains3 == chn)].iloc[0]
        print(f"  {pdb.upper()} {chn}: label {r.apoholo_assignment}, pocket ligands {r.nonquery_ligs if isinstance(r.nonquery_ligs, str) else '-'}, "
              f"volume {r.volume}, SASA {r.sasa}, RoG {r.RoG}, pocket RMSD to 2YIW {r.pocket_rms}")
    m = glob[glob.role == "M"].iloc[0]
    print(f"  AlphaFold model: volume {m.volume}, SASA {m.sasa}, RoG {m.RoG}, pocket RMSD to 2YIW {m.pocket_rms}")

    # ---- 4. curated labels
    print("\n=== 4. curated labels (the classifier's curation) ===")
    cur = curated_labels(glob)
    n_apo, n_holo = int((cur.cls == "apo").sum()), int((cur.cls == "holo").sum())
    print(f"curated candidates {len(cur)}: apo {n_apo}, holo {n_holo} (released: {row.n_apo} / {row.n_holo})")
    assert (n_apo, n_holo) == (row.n_apo, row.n_holo), "curation does not reproduce the released counts"
    dropped = sorted(set(zip(cand.structure, cand.chains3)) - set(zip(cur.structure, cur.chains3)))
    print(f"dropped by curation: {len(dropped)} " + ", ".join(
        f"{p.upper()} {c} ({glob[(glob.structure == p) & (glob.chains3 == c)].exp_method.iloc[0]})" for p, c in dropped))

    # ---- 5. Lazou's measure: Met198 all-atom RMSD to the open and the closed reference
    print("\n=== 5. Met198 (Lazou's moving segment) and Trp197: all-atom RMSD to 2NPQ A (open) and 2ZB1 A (closed) ===")

    def chain_path(pdb, chn):
        if pdb == QUERY_PDB and chn == QUERY_CHAIN:
            return os.path.join(STRUCT_DIR, f"query_{QUERY_PDB}.cif.gz")
        return os.path.join(STRUCT_DIR, f"cndt_{pdb}{chn}_aligned_to_{QUERY_PDB}{QUERY_CHAIN}.cif.gz")

    def unp_residues(pdb, chn):
        off = 0 if pdb == "AF" else offset_of(nested, pdb, chn)
        if off is None:
            return None, None
        c = load(AF_MODEL, "A") if pdb == "AF" else load(chain_path(pdb, chn), chn)
        return residues_by_unp(c, off), off

    odd = []
    for pdb, chains in nested.items():
        for chn, d in chains.items():
            m = d.get(UNP, {})
            unmapped = sum(1 for p in m.values() if p in (None, "null", ""))
            off = offset_of(nested, pdb, chn)
            if off or unmapped:
                odd.append(f"{pdb.upper()} {chn} (offset {off:+d}, {unmapped} unmapped)")
    print(f"candidate chains whose SIFTS numbering is not UniProt's at every pocket residue: {len(odd)} -> "
          + ", ".join(odd))
    refs = {}
    for tag, (pdb, chn) in (("open", OPEN_REF), ("closed", CLOSED_REF)):
        refs[tag] = unp_residues(pdb, chn)[0]
    rows = []
    items = [("AF", "A", "model", "AlphaFold DB v4")] + [(QUERY_PDB, QUERY_CHAIN, "holo", "query")] + \
            [(r.structure, r.chains3, r.cls, r.apoholo_assignment) for r in cur.itertuples()]
    for pdb, chn, cls, note in items:
        res, off = unp_residues(pdb, chn)
        rec = {"structure": pdb, "chain": chn, "class": cls, "label": note, "offset": off}
        if res is None or MOVING not in res or res[MOVING].name not in ("MET", "MSE"):
            rec["status"] = "no Met198"
            rows.append(rec)
            continue
        rec["status"] = "ok"
        for tag, ref in refs.items():
            p, q = ca_pairs(res, ref, exclude={MOVING})
            rot, t, n_used, rms = fit(p, q)
            rec[f"fit_n_{tag}"], rec[f"fit_rms_{tag}"] = n_used, round(rms, 3)
            for num, atoms, key in ((MOVING, MET_ATOMS, "met198"), (NEIGHBOUR, TRP_ATOMS, "trp197")):
                a, b = (atom_xyz(res[num], atoms) if num in res else None), atom_xyz(ref[num], atoms)
                rec[f"{key}_rmsd_{tag}"] = round(rmsd(a @ rot.T + t, b), 3) if a is not None and b is not None else np.nan
        rows.append(rec)
    tab = pd.DataFrame(rows)
    assert not tab.duplicated(["structure", "chain"]).any(), "a chain appears twice"
    for key in ("met198", "trp197"):
        c, o = tab[f"{key}_rmsd_closed"], tab[f"{key}_rmsd_open"]
        tab[f"{key}_nearer"] = np.where(c.isna() | o.isna(), "incomplete", np.where(c < o, "closed", "open"))
    tab.loc[tab.status != "ok", ["met198_nearer", "trp197_nearer"]] = np.nan
    ok = tab[tab.status == "ok"]
    tab.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(tab)} rows: the model, the query and {len(cur)} curated candidates)")
    nm = tab[tab.status != "ok"]
    print(f"structures without a usable Met198 (no residue 198 or not MET after the offset): {len(nm)}"
          + ("" if nm.empty else " -> " + ", ".join(f"{r.structure.upper()} {r.chain}" for r in nm.itertuples())))
    for pdb, chn, what in ((OPEN_REF[0], OPEN_REF[1], "open reference"), (CLOSED_REF[0], CLOSED_REF[1], "closed reference"),
                           ("AF", "A", "AlphaFold DB v4 model"), (QUERY_PDB, QUERY_CHAIN, "holo query, I46 bound"),
                           (APO_SHOWN[0], APO_SHOWN[1], "SI Fig. S10d apo structure")):
        r = ok[(ok.structure == pdb) & (ok.chain == chn)].iloc[0]
        print(f"  {pdb.upper():4s} {chn} ({what}): Met198 {r.met198_rmsd_closed:.2f} A to closed, "
              f"{r.met198_rmsd_open:.2f} A to open -> {r.met198_nearer}; Trp197 {r.trp197_rmsd_closed:.2f} / "
              f"{r.trp197_rmsd_open:.2f} A -> {r.trp197_nearer}   [fit: {int(r.fit_n_closed)} / {int(r.fit_n_open)} CA]")
    cand_ok = ok[ok["class"].isin(["apo", "holo"]) & (ok.label != "query")]
    ct = pd.crosstab(cand_ok["class"], cand_ok.met198_nearer, margins=True)
    print("\ncurated candidates, AHoJ class x nearer Lazou reference (Met198):")
    print(ct.to_string())
    ct2 = pd.crosstab(cand_ok["class"], cand_ok.trp197_nearer, margins=True)
    print("\ncurated candidates, AHoJ class x nearer reference (Trp197):")
    print(ct2.to_string())
    both = cand_ok[(cand_ok.met198_nearer != "incomplete")]
    for c in ("apo", "holo"):
        sub = both[both["class"] == c]
        k = int((sub.met198_nearer == ("closed" if c == "apo" else "open")).sum())
        print(f"  {c}: {k} of {len(sub)} ({100 * k / len(sub):.1f}%) are nearer the "
              f"{'closed' if c == 'apo' else 'open'} reference")
    n_closed = int((both.met198_nearer == "closed").sum())
    print(f"  all curated candidates with a complete Met198: {len(both)}; closed-like {n_closed} "
          f"({100 * n_closed / len(both):.1f}%), open-like {len(both) - n_closed}")
    for c in ("apo", "holo"):
        sub = both[both["class"] == c]
        print(f"  {c} chains, median Met198 RMSD: {sub.met198_rmsd_closed.median():.2f} A to closed, "
              f"{sub.met198_rmsd_open.median():.2f} A to open")
    agree = int(((both["class"] == "apo") & (both.met198_nearer == "closed")).sum()
                + ((both["class"] == "holo") & (both.met198_nearer == "open")).sum())
    print(f"  AHoJ occupancy label and nearer Met198 reference agree for {agree} of {len(both)} chains "
          f"({100 * agree / len(both):.1f}%)")
    # sensitivity: the binary rule has no "other" class; count the chains far from BOTH references
    for cut in (2.0, 3.0):
        far = both[(both.met198_rmsd_closed > cut) & (both.met198_rmsd_open > cut)]
        print(f"  ad hoc 'near neither' (Met198 more than {cut:.0f} A from each reference): {len(far)} of {len(both)} "
              f"(apo {int((far['class'] == 'apo').sum())}, holo {int((far['class'] == 'holo').sum())}); "
              f"within {cut:.0f} A of closed {int((both.met198_rmsd_closed <= cut).sum())}, "
              f"of open {int((both.met198_rmsd_open <= cut).sum())}")
    # holo but closed-like: which ligands?
    hc = both[(both["class"] == "holo") & (both.met198_nearer == "closed")]
    lig_of = {(r.structure, r.chains3): "-".join(sorted(set(
        (r.query_ligs if isinstance(r.query_ligs, str) else "").split("-")
        + (r.nonquery_ligs if isinstance(r.nonquery_ligs, str) else "").split("-")) - {""}))
        for r in cur.itertuples()}
    print(f"  holo chains nearer the closed reference: {len(hc)} -> "
          + ", ".join(f"{r.structure.upper()} {r.chain} [{lig_of[(r.structure, r.chain)]}]" for r in hc.itertuples()))
    ao = both[(both["class"] == "apo") & (both.met198_nearer == "open")]
    print(f"  apo chains nearer the open reference: {len(ao)}")
    bog_ok = both.merge(bog[["structure", "chains3"]].rename(columns={"chains3": "chain"}), on=["structure", "chain"])
    print(f"  BOG-bound chains: {int((bog_ok.met198_nearer == 'open').sum())} of {len(bog_ok)} nearer the open reference "
          f"(median Met198 RMSD {bog_ok.met198_rmsd_open.median():.2f} A to open, "
          f"{bog_ok.met198_rmsd_closed.median():.2f} A to closed)")
    oo = ok[(ok.structure == OPEN_REF[0]) & (ok.chain == OPEN_REF[1])].iloc[0]
    print(f"  sanity: the open reference against itself {oo.met198_rmsd_open:.2f} A; against the closed reference "
          f"{oo.met198_rmsd_closed:.2f} A (the open-closed separation of Met198)")

    # ---- 6. occlusion: fitted onto the holo query, how close do Trp197 / Met198 come to where I46 sits?
    print("\n=== 6. occlusion of the I46 position (each chain superposed on 2YIW A, same fit) ===")
    q_res = residues_by_unp(load(chain_path(QUERY_PDB, QUERY_CHAIN), QUERY_CHAIN), 0)
    # the deposited 2YIW and AHoJ's saved query share one frame; I46 coordinates come from the deposited file
    dq = residues_by_unp(ch, 0)
    p, q = ca_pairs(dq, q_res, exclude=set())
    assert rmsd(p, q) < 1e-3, "deposited 2YIW and the saved query are not in one frame"

    def occl(res):
        p, q = ca_pairs(res, q_res, exclude={NEIGHBOUR, MOVING})
        rot, t, _, _ = fit(p, q)
        out = {}
        for num, atoms in ((MOVING, ["CB", "CG", "SD", "CE"]), (NEIGHBOUR, TRP_ATOMS[4:])):
            xyz = atom_xyz(res[num], atoms) @ rot.T + t
            out[num] = float(np.min(np.linalg.norm(xyz[:, None] - lig_xyz[None], axis=2)))
        return out
    for pdb, chn, what in (("AF", "A", "AlphaFold DB v4 model"), (CLOSED_REF[0], CLOSED_REF[1], "closed reference"),
                           (APO_SHOWN[0], APO_SHOWN[1], "SI Fig. S10d apo"), (OPEN_REF[0], OPEN_REF[1], "open reference"),
                           (QUERY_PDB, QUERY_CHAIN, "holo query")):
        o = occl(unp_residues(pdb, chn)[0])
        print(f"  {pdb.upper():4s} {chn} ({what}): nearest side-chain atom to an I46 atom: Met198 {o[MOVING]:.2f} A, "
              f"Trp197 {o[NEIGHBOUR]:.2f} A")
    # does BOG in 2NPQ sit where I46 sits?
    npq = load(chain_path(*OPEN_REF), OPEN_REF[1], keep_ligands=True)
    npq_res = residues_by_unp(npq, offset_of(nested, *OPEN_REF))
    p, q = ca_pairs(npq_res, q_res, exclude={NEIGHBOUR, MOVING})
    rot, t, _, _ = fit(p, q)
    lig_c = lig_xyz.mean(0)
    for r in npq:
        if r.name == "BOG":
            xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r]) @ rot.T + t
            dmin = np.min(np.linalg.norm(xyz[:, None] - lig_xyz[None], axis=2), axis=0)
            print(f"  2NPQ BOG {r.seqid.num}: centroid {np.linalg.norm(xyz.mean(0) - lig_c):.2f} A from the I46 centroid; "
                  f"{int((dmin < 2.0).sum())} of {len(lig_xyz)} I46 atoms within 2 A of a BOG atom")

    # the model: same file as the run's, and its pLDDT at the moving residues
    af = load(AF_MODEL, "A")
    afr = residues_by_unp(af, 0)
    run_model = residues_by_unp(load(os.path.join(STRUCT_DIR, f"model_AF-{UNP}-F1-model-v4_aligned_to_"
                                                  f"{QUERY_PDB}{QUERY_CHAIN}.cif.gz"), "A"), 0)
    p, q = ca_pairs(afr, run_model, exclude=set())
    rot, t = kabsch(p, q)
    print(f"\nAlphaFold DB v4 store model vs the run's aligned model: {len(p)} CA, RMSD {rmsd(p @ rot.T + t, q):.3f} A "
          f"after superposition (the same model)")
    atoms = [a.b_iso for n in pdb_res for a in afr[n]]
    print(f"AlphaFold model pLDDT: Trp197 {afr[NEIGHBOUR][0].b_iso:.2f}, Met198 {afr[MOVING][0].b_iso:.2f}; "
          f"pocket residues {np.mean([afr[n][0].b_iso for n in pdb_res]):.2f} per residue, "
          f"{np.mean(atoms):.2f} per atom (the pipeline's pocket pLDDT is {row.model_pocket_pLDDT_avg})")


if __name__ == "__main__":
    main()
