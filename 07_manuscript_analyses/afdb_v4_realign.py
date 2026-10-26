# -*- coding: utf-8 -*-
"""Detection and docking re-run on the AlphaFold DB v4 models: validation gates and the v4 / v6 comparison.

Supplementary Note 4 ("Model version: the results do not depend on it"): its table of detection, docking and
redocking rates and gaps under version 4 and version 6 is afdb_v4_realign_sensitivity.csv (`sensitivity`), and
its other numbers (the 2,746 coordinate-identical accessions and their exact gate, the 488 sites on 82 accessions
without a v6 model, the 78 sequence-changed accessions with 847 sites, 63.3% of residues and 338 sites, the
docking calibration noise floor) are printed by `report`. Nothing here writes into the v4 store, the staged model
folder or any v6 output; it reads the `_afdbv4` tables next to the July (v6) ones and writes its tables to
<manuscript_analyses_dir>.

    python afdb_v4_realign.py geometry   -> gate 1 (geometry on carried-over models equals July's), coverage of the
                                            detection set, residue coverage of the sequence-changed accessions
    python afdb_v4_realign.py p2rank     -> gate 1 for P2Rank (identical models give July's per-site result), gate 3
                                            (the harmonized distance equals P2Rank's closestPocketDist), class rates
    python afdb_v4_realign.py fpocket | ifsitepred -> gate 1 on the raw pockets of identical models, class rates, agreement
    python afdb_v4_realign.py manifest   -> the docking manifest on the v4 geometry against July's
    python afdb_v4_realign.py calibration -> gate 1 for the docking: calibration jobs on identical models against July
    python afdb_v4_realign.py sensitivity -> v4 against v6 on the common set: class rates, gaps and their differences with
                                            cluster-bootstrap intervals, by model category and separation tier
    python afdb_v4_realign.py seqchanged -> sites on sequence-changed models: does v6 hold the same residues (py311 env, gemmi)
    python afdb_v4_realign.py downstream -> separation-tier and balanced-label gaps on v4 next to v6 (reads saved printouts)
    python afdb_v4_realign.py report     -> every quoted number (runs each finished section)

The per-accession model identity (identical / moved / different_atoms) is afdb_v4_v6_accessions.csv, written by
afdb_v4_v6_diff.py. "Identical" there means every polymer atom has the same coordinates and B-factor in both
files, so on those accessions every deterministic step must reproduce July exactly.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ma_paths as P  # noqa: E402
import afdb_v4_sources as S  # noqa: E402  (paths)

ACC_CSV = P.out("afdb_v4_v6_accessions.csv")                                   # afdb_v4_v6_diff.py worklist
GEOM_V6 = os.path.join(S.WORK, "afdb_dataset", "st100_run3", "afdb_st_100_dataset_with_geometry.csv")
GEOM_V4 = os.path.join(S.WORK, "afdb_dataset", "st100_run3_afdbv4", "afdb_st_100_dataset_with_geometry.csv")
COVERAGE_CSV = P.out("afdb_v4_realign_residue_coverage.csv")                  # committed: sequence-changed accessions

GEOM_NUM = ["union_center_geom_x", "union_center_geom_y", "union_center_geom_z",
            "union_center_com_x", "union_center_com_y", "union_center_com_z", "union_rg",
            "intersection_center_geom_x", "intersection_center_geom_y", "intersection_center_geom_z",
            "intersection_center_com_x", "intersection_center_com_y", "intersection_center_com_z", "intersection_rg",
            "union_n_atoms", "intersection_n_atoms", "union_n_residues_found", "intersection_n_residues_found",
            "af_model_len"]
GEOM_TXT = ["union_chain_resi", "intersection_chain_resi"]
KEEP = ["site_uid", "site_uniprot", "n_union", "pocket_class", "afdb_filename"] + GEOM_NUM + GEOM_TXT


def _load_geometry():
    v6 = pd.read_csv(GEOM_V6, usecols=KEEP, low_memory=False)
    v4 = pd.read_csv(GEOM_V4, usecols=KEEP, low_memory=False)
    assert list(v6.site_uid) == list(v4.site_uid), "the two geometry tables must hold the same sites in the same order"
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category
    v4["category"] = v4.site_uniprot.map(cat).fillna("no_v6_model")
    return v4, v6


def summarise_geometry(write=True):
    v4, v6 = _load_geometry()
    has4, has6 = v4.afdb_filename.notna(), v6.afdb_filename.notna()
    print("== Geometry on the v4 models (section: geometry) ==")
    print(f"sites in the selection: {len(v4):,} on {v4.site_uniprot.nunique():,} accessions")
    print(f"sites with a model, v6 (July): {int(has6.sum()):,} on {v6[has6].site_uniprot.nunique():,} accessions")
    print(f"sites with a model, v4: {int(has4.sum()):,} on {v4[has4].site_uniprot.nunique():,} accessions")
    ret = has4 & ~has6
    print(f"sites returning (v4 model, no v6 model): {int(ret.sum()):,} on {v4[ret].site_uniprot.nunique():,} accessions")
    print("model file versions, v4 run: " + ", ".join(
        f"{k} {n:,}" for k, n in v4[has4].drop_duplicates("site_uniprot").afdb_filename.str.extract(r"model_(v\d+)")[0]
        .value_counts().items()))
    for c in ["identical", "moved", "different_atoms", "no_v6_model"]:
        m = v4.category == c
        print(f"  category {c}: {v4[m].site_uniprot.nunique():,} accessions, {int(m.sum()):,} sites")

    # gate 1: on identical models the geometry table must equal July's, cell for cell
    idn = (v4.category == "identical").values
    a, b = v4[idn], v6[idn]
    num_bad = int((~((a[GEOM_NUM].values == b[GEOM_NUM].values) |
                     (a[GEOM_NUM].isna().values & b[GEOM_NUM].isna().values))).any(axis=1).sum())
    txt_bad = int((a[GEOM_TXT].fillna("").values != b[GEOM_TXT].fillna("").values).any(axis=1).sum())
    print(f"gate 1, identical models: {int(idn.sum()):,} sites; rows with any numeric difference {num_bad:,}; "
          f"rows with a residue-list difference {txt_bad:,}")

    # moved models: same residues, how far the centres and Rg go (different frames, so only Rg is frame-free)
    mv = (v4.category == "moved").values
    a, b = v4[mv], v6[mv]
    same_res = int((a.union_chain_resi.fillna("").values == b.union_chain_resi.fillna("").values).sum())
    drg = (a.union_rg.values - b.union_rg.values)
    print(f"moved models: {int(mv.sum()):,} sites; identical union residue list {same_res:,}; "
          f"|d union_rg| median {np.nanmedian(np.abs(drg)):.3f} A, 95th pct {np.nanpercentile(np.abs(drg), 95):.3f} A, "
          f"max {np.nanmax(np.abs(drg)):.3f} A")

    # sequence-changed accessions: residue coverage of the union set under each version
    sq = (v4.category == "different_atoms").values
    a, b = v4[sq], v6[sq]
    cov = pd.DataFrame({"acc": a.site_uniprot.values, "n_union": a.n_union.values,
                        "found_v4": a.union_n_residues_found.values, "found_v6": b.union_n_residues_found.values})
    per = cov.groupby("acc").agg(n_sites=("n_union", "size"), n_union=("n_union", "sum"),
                                 found_v4=("found_v4", "sum"), found_v6=("found_v6", "sum")).reset_index()
    per["complete_v4"] = per.found_v4 == per.n_union
    per["complete_v6"] = per.found_v6 == per.n_union
    print(f"sequence-changed models: {len(per):,} accessions, {len(cov):,} sites; union residues {int(per.n_union.sum()):,}; "
          f"found on v4 {int(per.found_v4.sum()):,}, on v6 {int(per.found_v6.sum()):,}")
    print(f"  accessions with every union residue found: v4 {int(per.complete_v4.sum()):,}, v6 {int(per.complete_v6.sum()):,}; "
          f"sites complete: v4 {int((cov.found_v4 == cov.n_union).sum()):,}, v6 {int((cov.found_v6 == cov.n_union).sum()):,}")
    # found is not the same as right: a residue number can exist in a changed sequence and be another residue
    allv4 = v4[has4]
    print(f"all sites with a v4 model: every union residue found {int((allv4.union_n_residues_found == allv4.n_union).sum()):,} "
          f"of {len(allv4):,}")
    allv6 = v6[has6]
    print(f"all sites with a v6 model: every union residue found {int((allv6.union_n_residues_found == allv6.n_union).sum()):,} "
          f"of {len(allv6):,}")
    if write:
        per.to_csv(COVERAGE_CSV, index=False, lineterminator="\n")
        print(f"[saved] {os.path.basename(COVERAGE_CSV)}")


P2R_EVAL = str(P.anchor("p2rank_dev_eval_dir", "AHOJ_P2RANK_DEV_EVAL_DIR"))   # the P2Rank dev build's test_output/
P2R_RUNS = {"SAS points as atoms": ("UBS2_run3_afdbv4", "UBS2_run3_dev_clean"),
            "SAS points off": ("UBS2_run3_afdbv4_sasoff", "UBS2_run3_sasoff")}
_P2R_TABLE = os.path.join("analysis_v1", "tables", "p2rank_st100_merged_per_pocket.csv")
P2R_NATIVE = {"v4": os.path.join(S.WORK, "P2Rank", "ubs_run_5_afdbv4", _P2R_TABLE),
              "v6": os.path.join(S.WORK, "P2Rank", "ubs_run_5", _P2R_TABLE)}
LBS = os.path.join(S.WORK, "lbs_generality")
P2R_HARM = {"v4": os.path.join(LBS, "p2rank_afdbv4", "lbs_p2rank_st100_merged_per_pocket.csv"),
            "v6": os.path.join(LBS, "p2rank", "lbs_p2rank_st100_merged_per_pocket.csv")}


def _observed(subdir):
    f = os.path.join(P2R_EVAL, subdir, "eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit", "cases",
                     "observed_sites.csv")
    o = pd.read_csv(f)
    o.columns = [c.strip() for c in o.columns]
    o["site"] = o["site"].astype(str).str.strip()
    return o.set_index("site")


def _class_rates(df, col, label):
    for ver, d in df.items():
        r = {c: 100 * d.loc[d.pocket_class == c, col].mean() for c in ("apo", "holo")}
        d = d[d[col].notna()]
        print(f"  {label}, {ver}: n {len(d):,}; AF-Apo {r['apo']:.2f}%, AF-Holo {r['holo']:.2f}%, "
              f"all {100 * d[col].mean():.2f}%, gap {r['holo'] - r['apo']:.2f} points")


def summarise_p2rank():
    print("== P2Rank on the v4 models (section: p2rank) ==")
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category.to_dict()
    for label, (new, july) in P2R_RUNS.items():
        x, y = _observed(new), _observed(july)
        ids = [u for u in y.index if cat.get(u.split(":")[0]) == "identical"]
        same = {c: int((x.loc[ids, c].astype(str).str.strip() == y.loc[ids, c].astype(str).str.strip()).sum())
                for c in ("dca4rank", "closestPocketDist")}
        print(f"gate 1, {label}: sites {len(x):,} (July {len(y):,}); identical-model sites {len(ids):,}; "
              f"same dca4rank {same['dca4rank']:,}; same closestPocketDist {same['closestPocketDist']:,}")
    cols = ["pocket_class", "success_tns", "closest_pocket_dist"]
    nat = {v: pd.read_csv(f, usecols=["unit_uid"] + cols, low_memory=False) for v, f in P2R_NATIVE.items()}
    har = {v: pd.read_csv(f, usecols=["site_uid"] + cols, low_memory=False) for v, f in P2R_HARM.items()}
    _class_rates(nat, "success_tns", "native TNS (SAS points as atoms)")
    _class_rates(har, "success_tns", "harmonized TNS (centre to union heavy atom, 4 A)")
    # gate 3: the harmonized scorer's nearest-atom distance against P2Rank's own closestPocketDist
    for ver in ("v4", "v6"):
        m = nat[ver].merge(har[ver], left_on="unit_uid", right_on="site_uid", suffixes=("_nat", "_har"))
        both = m.closest_pocket_dist_nat.notna() & m.closest_pocket_dist_har.notna()
        d = (m.closest_pocket_dist_nat - m.closest_pocket_dist_har)[both].abs()
        print(f"gate 3, {ver}: sites with both distances {int(both.sum()):,}; |difference| median {d.median():.2e} A, "
              f"max {d.max():.2e} A; above 1e-3 A {int((d > 1e-3).sum()):,}")
    # common set, paired agreement of the binary outcome
    c = har["v4"].merge(har["v6"], on="site_uid", suffixes=("_v4", "_v6"))
    c = c[c.success_tns_v4.notna() & c.success_tns_v6.notna()].copy()      # v6 leaves the sites without a v6 model unscored
    c["success_tns_v4"], c["success_tns_v6"] = c.success_tns_v4.astype(bool), c.success_tns_v6.astype(bool)
    agree = (c.success_tns_v4 == c.success_tns_v6)
    for k in ("apo", "holo"):
        mk = c.pocket_class_v4 == k
        print(f"  common set, {k}: {int(mk.sum()):,} sites; v4 {100 * c.success_tns_v4[mk].mean():.2f}%, "
              f"v6 {100 * c.success_tns_v6[mk].mean():.2f}%")
    c["category"] = c.site_uid.str.split(":").str[0].map(cat)
    print(f"harmonized TNS on the common set: {len(c):,} sites; same outcome {int(agree.sum()):,} ({100 * agree.mean():.2f}%)")
    for k in ("identical", "moved", "different_atoms"):
        mk = c.category == k
        print(f"  {k}: {int(mk.sum()):,} sites; same outcome {100 * agree[mk].mean():.2f}%; "
              f"v4 {100 * c.success_tns_v4[mk].mean():.2f}%, v6 {100 * c.success_tns_v6[mk].mean():.2f}%")


RAW_V6 = str(P.anchor("lbs_raw_v6_dir", "AHOJ_LBS_RAW_V6_DIR"))   # the July (v6) raw pocket lists as pulled from the cluster
LBS_TOOLS = {   # tool: (raw pockets v4, raw pockets v6 as pulled from the cluster, merged table v4, merged table v6)
    "fpocket": (os.path.join(LBS, "fpocket", "full_afdbv4", "fpocket_raw_full_afdbv4.csv"),
                os.path.join(RAW_V6, "fpocket_raw_full.csv"),
                os.path.join(LBS, "fpocket", "full_afdbv4", "lbs_fpocket_st100_merged_per_pocket.csv"),
                os.path.join(LBS, "fpocket", "full", "lbs_fpocket_st100_merged_per_pocket.csv")),
    "ifsitepred": (os.path.join(LBS, "ifsitepred", "full_afdbv4", "ifsitepred_raw_full_afdbv4.csv"),
                   os.path.join(RAW_V6, "ifsitepred_raw_full.csv"),
                   os.path.join(LBS, "ifsitepred", "full_afdbv4", "lbs_ifsitepred_st100_merged_per_pocket.csv"),
                   os.path.join(LBS, "ifsitepred", "full", "lbs_ifsitepred_st100_merged_per_pocket.csv")),
}


def _paired(har, cat):
    """Class rates on each version, then the paired agreement on the sites scored under both."""
    _class_rates(har, "success_tns", "harmonized TNS")
    c = har["v4"].merge(har["v6"], on="site_uid", suffixes=("_v4", "_v6"))
    c = c[c.success_tns_v4.notna() & c.success_tns_v6.notna()].copy()
    c["success_tns_v4"], c["success_tns_v6"] = c.success_tns_v4.astype(bool), c.success_tns_v6.astype(bool)
    agree = (c.success_tns_v4 == c.success_tns_v6)
    c["category"] = c.site_uid.str.split(":").str[0].map(cat)
    for k in ("apo", "holo"):
        mk = c.pocket_class_v4 == k
        print(f"  common set, {k}: {int(mk.sum()):,} sites; v4 {100 * c.success_tns_v4[mk].mean():.2f}%, "
              f"v6 {100 * c.success_tns_v6[mk].mean():.2f}%")
    print(f"  common set: {len(c):,} sites; same outcome {int(agree.sum()):,} ({100 * agree.mean():.2f}%)")
    for k in ("identical", "moved", "different_atoms"):
        mk = c.category == k
        print(f"    {k}: {int(mk.sum()):,} sites; same outcome {100 * agree[mk].mean():.2f}%; "
              f"v4 {100 * c.success_tns_v4[mk].mean():.2f}%, v6 {100 * c.success_tns_v6[mk].mean():.2f}%")


def summarise_lbs(tool):
    raw4_f, raw6_f, m4, m6 = LBS_TOOLS[tool]
    if not (os.path.isfile(raw4_f) and os.path.isfile(m4)):
        print(f"== {tool} on the v4 models: not run yet ==")
        return
    print(f"== {tool} on the v4 models (section: {tool}) ==")
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category.to_dict()
    raw = {}
    for v, f in (("v4", raw4_f), ("v6", raw6_f)):
        d = pd.read_csv(f, usecols=["afdb_filename", "pocket_rank", "center_x", "center_y", "center_z", "score"])
        d["acc"] = d.afdb_filename.str.split("-").str[1]
        raw[v] = d
        print(f"raw pockets, {v}: {len(d):,} over {d.acc.nunique():,} receptors")
    # gate 1: on identical models the ranked pocket list is the same
    a = raw["v4"][raw["v4"].acc.map(cat) == "identical"].sort_values(["acc", "pocket_rank"]).reset_index(drop=True)
    b = raw["v6"][raw["v6"].acc.map(cat) == "identical"].sort_values(["acc", "pocket_rank"]).reset_index(drop=True)
    print(f"gate 1: identical-model receptors with pockets, v4 {a.acc.nunique():,}, v6 {b.acc.nunique():,}; "
          f"pockets v4 {len(a):,}, v6 {len(b):,}")
    j = a.merge(b, on=["acc", "pocket_rank"], how="outer", suffixes=("_v4", "_v6"), indicator=True)
    both = j[j._merge == "both"]
    dist = np.sqrt(sum((both[f"center_{k}_v4"] - both[f"center_{k}_v6"]) ** 2 for k in "xyz"))
    exact = int(((dist == 0) & (both.score_v4 == both.score_v6)).sum())
    print(f"  pockets matched by receptor and rank {len(both):,}; unmatched {int((j._merge != 'both').sum()):,}; "
          f"same centre and score exactly {exact:,}; centre shift above 1e-3 A {int((dist > 1e-3).sum()):,} "
          f"(max {dist.max():.3g} A)")
    rec_bad = both.assign(bad=(dist > 1e-3).values).groupby("acc").bad.any()
    unm = set(j[j._merge != "both"].acc)
    print(f"  receptors with any pocket shifted above 1e-3 A or unmatched: {int((rec_bad | rec_bad.index.isin(unm)).sum()):,} "
          f"of {both.acc.nunique():,}")
    cols = ["site_uid", "pocket_class", "success_tns"]
    _paired({"v4": pd.read_csv(m4, usecols=cols, low_memory=False), "v6": pd.read_csv(m6, usecols=cols, low_memory=False)}, cat)


DOCK = os.path.join(S.WORK, "docking")


def summarise_manifest():
    """The docking manifest on the v4 geometry against July's (both written by build_docking_manifest.py in the
    ahoj-docking env; in the py311 env pandas reads the sodium CCD code NA as missing and drops 91 group-2 jobs)."""
    print("== Docking manifest on the v4 models (section: manifest) ==")
    m = {v: pd.read_csv(os.path.join(DOCK, d, "manifest", "docking_manifest.csv"), low_memory=False,
                        usecols=["job_id", "site_uid", "ligand_group", "ligand_id", "afdb_filename"], keep_default_na=False)
         for v, d in (("v4", "run3_afdbv4"), ("v6", "run3"))}
    for v, d in m.items():
        g = d.ligand_group.astype(int).value_counts()
        print(f"{v}: jobs {len(d):,} (group 1 {g[1]:,}, group 2 {g[2]:,}); pockets {d.site_uid.nunique():,}; "
              f"receptors {d.afdb_filename.nunique():,}; ligand CCD codes {d.ligand_id.nunique():,}")
    a, b = set(m["v4"].job_id), set(m["v6"].job_id)
    only = m["v4"][m["v4"].job_id.isin(a - b)]
    g6 = pd.read_csv(GEOM_V6, usecols=["site_uid", "afdb_filename"])
    nov6 = set(g6[g6.afdb_filename.isna()].site_uid)
    print(f"jobs in both {len(a & b):,}; v6 only {len(b - a):,}; v4 only {len(a - b):,} on {only.site_uid.nunique():,} sites, "
          f"of which on sites without a v6 model {int(only.site_uid.isin(nov6).sum()):,}")
    print(f"CCD codes new in v4: {len(set(m['v4'].ligand_id) - set(m['v6'].ligand_id)):,}; "
          f"jobs with an empty ligand id (sodium, CCD code NA read as missing; they cannot dock): "
          f"v4 {int((m['v4'].ligand_id == '').sum()):,}, v6 {int((m['v6'].ligand_id == '').sum()):,}")


CALIB_CSV = os.path.join(DOCK, "run3_afdbv4", "calibration", "results", "docking_results_calib_afdbv4.csv")
CALIB_REP2_CSV = CALIB_CSV.replace("_afdbv4.csv", "_afdbv4_rep2.csv")
JULY_RESULTS = os.path.join(DOCK, "run3", "results", "docking_results_run3.csv")


def summarise_calibration():
    """Gate 1 for the docking: the calibration jobs sit on identical models, so July's result is the expectation."""
    if not os.path.isfile(CALIB_CSV):
        print("== Docking calibration on the v4 models: not run yet ==")
        return
    print("== Docking calibration on the v4 models (section: calibration) ==")
    c = pd.read_csv(CALIB_CSV, low_memory=False)
    j = pd.read_csv(JULY_RESULTS, low_memory=False)
    m = c.merge(j, on="job_id", suffixes=("_v4", "_v6"))
    print(f"calibration jobs {len(c):,} on {c.site_uid.nunique():,} pockets; found in July's results {len(m):,}; "
          f"ok v4 {int((c.status == 'ok').sum()):,}, ok July (same jobs) {int((m.status_v6 == 'ok').sum()):,}; "
          f"same status {int((m.status_v4 == m.status_v6).sum()):,}")
    ok = m[(m.status_v4 == "ok") & (m.status_v6 == "ok")]
    d = (ok.affinity_top_v4 - ok.affinity_top_v6)
    print(f"both ok {len(ok):,}; affinity_top equal {int((d == 0).sum()):,} ({100 * (d == 0).mean():.2f}%); "
          f"|difference| <= 0.1 {int((d.abs() <= 0.1).sum()):,}, > 0.5 {int((d.abs() > 0.5).sum()):,}, "
          f"> 1.0 {int((d.abs() > 1.0).sum()):,} kcal/mol; mean difference {d.mean():+.4f}, median {d.median():+.4f}, "
          f"max |difference| {d.abs().max():.2f}")
    print(f"  all nine mode affinities equal {int((ok.affinities_v4.astype(str) == ok.affinities_v6.astype(str)).sum()):,}; "
          f"pose_in_pocket equal {int((ok.pose_in_pocket_v4 == ok.pose_in_pocket_v6).sum()):,}; "
          f"box equal {int(((ok.size_x_v4 == ok.size_x_v6)).sum()):,}")
    dock4 = ok.pose_in_pocket_v4.astype(bool) & (ok.affinity_top_v4 <= -6)
    dock6 = ok.pose_in_pocket_v6.astype(bool) & (ok.affinity_top_v6 <= -6)
    print(f"  dockable (pose in pocket and affinity <= -6): v4 {100 * dock4.mean():.2f}%, July {100 * dock6.mean():.2f}%; "
          f"same call {int((dock4 == dock6).sum()):,} of {len(ok):,}")
    from scipy.stats import spearmanr
    print(f"  Spearman of affinity_top, v4 against July: {spearmanr(ok.affinity_top_v4, ok.affinity_top_v6)[0]:.4f}")
    # control: the same calibration run twice on the same day. Uni-Dock on a GPU is not bit-reproducible at a fixed
    # seed, so the drift against July has to be read against this run-to-run noise.
    if os.path.isfile(CALIB_REP2_CSV):
        r = pd.read_csv(CALIB_REP2_CSV, low_memory=False)
        for label, x, y in (("run 1 against run 2, same day", c, r), ("run 2 against July", r, j)):
            q = x.merge(y, on="job_id", suffixes=("_a", "_b"))
            q = q[(q.status_a == "ok") & (q.status_b == "ok")]
            e = (q.affinity_top_a - q.affinity_top_b)
            print(f"  {label}: both ok {len(q):,}; affinity_top equal {int((e == 0).sum()):,} ({100 * (e == 0).mean():.2f}%); "
                  f"|difference| <= 0.1 {int((e.abs() <= 0.1).sum()):,}, > 0.5 {int((e.abs() > 0.5).sum()):,}, "
                  f"> 1.0 {int((e.abs() > 1.0).sum()):,}")


PREP_DIR = os.path.join(DOCK, "run3_afdbv4", "prep")
CALIB_MANIFEST = os.path.join(DOCK, "run3_afdbv4", "calibration", "docking_manifest_calib.csv")


def summarise_prep():
    """Counts that no other section prints: P2Rank pockets, the receptor and ligand
    preparation on the HPC cluster (tables pulled into run3_afdbv4/prep), July's docking status counts, the S6 re-score."""
    print("== Preparation counts (section: prep) ==")
    for ver, sub in (("v4", "p2rank_afdbv4"), ("v6", "p2rank")):
        f = os.path.join(LBS, sub, "raw_pockets_run3.csv")
        if os.path.isfile(f):
            r = pd.read_csv(f, usecols=["afdb_filename"])
            print(f"P2Rank raw pockets, {ver}: {len(r):,} over {r.afdb_filename.nunique():,} receptors")
    f4, f6 = os.path.join(PREP_DIR, "prep_receptors_afdbv4.csv"), os.path.join(PREP_DIR, "prep_receptors_july_v6.csv")
    if os.path.isfile(f4):
        cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category
        a, b = pd.read_csv(f4), pd.read_csv(f6)
        a["category"] = a.site_uniprot.map(cat).fillna("no_v6_model")
        ea, eb = set(a[a.status == "error"].site_uniprot), set(b[b.status == "error"].site_uniprot)
        print(f"receptors prepared, v4: {len(a):,}; Meeko failures {len(ea):,} (all recovered with OpenBabel); July: {len(b):,} "
              f"prepared, {len(eb):,} Meeko failures")
        byc = a[a.status == "error"].category.value_counts()
        print("  v4 Meeko failures by model category: " + ", ".join(f"{k} {int(v):,}" for k, v in byc.items())
              + f"; on identical models also failed in July {len(ea & eb & set(a[a.category == 'identical'].site_uniprot)):,}; "
              f"not on identical models {int(len(ea) - byc.get('identical', 0)):,}")
        for ver, f in (("v4", "prep_ligands_afdbv4.csv"), ("July", "prep_ligands_july_v6.csv")):
            g = pd.read_csv(os.path.join(PREP_DIR, f), keep_default_na=False)
            print(f"ligands prepared, {ver}: ok {int((g.status == 'ok').sum()):,}, error {int((g.status == 'error').sum()):,}")
    if os.path.isfile(CALIB_MANIFEST):
        c = pd.read_csv(CALIB_MANIFEST, usecols=["site_uniprot", "site_uid"])
        print(f"calibration manifest: {len(c):,} jobs, {c.site_uid.nunique():,} pockets, {c.site_uniprot.nunique():,} receptors")
    j = pd.read_csv(JULY_RESULTS, usecols=["status"])
    print(f"July docking results: {len(j):,} jobs, ok {int((j.status == 'ok').sum()):,}, error {int((j.status == 'error').sum()):,}")
    v4res = os.path.join(DOCK, "run3_afdbv4", "results", "docking_results_run3_afdbv4.csv")
    if os.path.isfile(v4res):
        import json
        import tarfile
        r = pd.read_csv(v4res, usecols=["status"])
        print(f"v4 docking results: {len(r):,} jobs, ok {int((r.status == 'ok').sum()):,}, error {int((r.status == 'error').sum()):,}")
        for ver, run in (("v4", "run3_afdbv4"), ("July", "run3")):
            wall = []
            for name in sorted(os.listdir(os.path.join(DOCK, run, "out"))):
                with tarfile.open(os.path.join(DOCK, run, "out", name)) as t:
                    m = [x for x in t.getmembers() if x.name.startswith("meta_")][0]
                    wall.append(json.load(t.extractfile(m))["wall_s"])
            print(f"  shard wall time inside the container, {ver}: {len(wall)} shards, {sum(wall) / 3600:.1f} GPU-h, "
                  f"{min(wall) / 60:.0f} to {max(wall) / 60:.0f} min per shard")
            h = pd.read_csv(os.path.join(DOCK, run, "analysis", "ccd_heavy_atoms.csv"))
            print(f"  ligand heavy-atom table, {ver}: {len(h):,} ligands")
    t4 = os.path.join(LBS, "analysis_v1_afdbv4", "tables", "lbs_size_adaptive_tolerance_long.csv")
    if os.path.isfile(t4):
        a = pd.read_csv(t4)
        b = pd.read_csv(os.path.join(LBS, "analysis_v1", "tables", "lbs_size_adaptive_tolerance_long.csv"))
        m = a.merge(b, on=["tool", "scoring", "population"], suffixes=("_v4", "_v6"))
        m = m[m.population == "full"]
        for r in m.itertuples(index=False):
            print(f"  size-adaptive re-score, {r.tool} {r.scoring}: apo {r.apo_tns_v4:.1f} | {r.apo_tns_v6:.1f}, "
                  f"holo {r.holo_tns_v4:.1f} | {r.holo_tns_v6:.1f}, gap {r.gap_holo_minus_apo_v4:.1f} | {r.gap_holo_minus_apo_v6:.1f}")
        dg = (m.gap_holo_minus_apo_v4 - m.gap_holo_minus_apo_v6).abs()
        dr = pd.concat([(m.apo_tns_v4 - m.apo_tns_v6).abs(), (m.holo_tns_v4 - m.holo_tns_v6).abs()])
        print(f"  largest rate difference {dr.max():.1f} points, largest gap difference {dg.max():.1f} points")


SEQCH_CSV = P.out("afdb_v4_realign_seqchanged_sites.csv")                   # committed


def build_seqchanged():
    """For every site on a sequence-changed model: does the v6 file hold the same amino acid as the v4 file at every
    union residue number? The sites were defined on the 2022 UniProt sequence (the v4 one) and both runs map them by
    residue number, so where v6 differs the July run scored another set of residues. Needs gemmi (py311 env)."""
    import gemmi
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category
    g = pd.read_csv(GEOM_V4, usecols=["site_uid", "site_uniprot", "union_chain_resi"], low_memory=False)
    g = g[g.site_uniprot.map(cat) == "different_atoms"]
    root = S.STORE

    def residues(path):
        st = gemmi.read_structure(path)
        return {r.seqid.num: (r.name, r[0].b_iso) for r in st[0][0]}
    rows = []
    for acc, grp in g.groupby("site_uniprot"):
        m4 = residues(os.path.join(root, "v4", f"AF-{acc}-F1-model_v4.cif.gz"))
        m6 = residues(os.path.join(root, "v6", f"AF-{acc}-F1-model_v6.cif.gz"))
        for r in grp.itertuples(index=False):
            pos = [int(t.split("_")[1]) for t in str(r.union_chain_resi).split()]
            same = sum(1 for q in pos if q in m4 and q in m6 and m4[q][0] == m6[q][0])
            rows.append({"site_uid": r.site_uid, "acc": acc, "n_union_found_v4": len(pos), "n_same_residue_in_v6": same,
                         "pocket_plddt_v4": round(float(np.mean([m4[q][1] for q in pos if q in m4])), 2),
                         "pocket_plddt_v6": round(float(np.mean([m6[q][1] for q in pos if q in m6])), 2)})
    pd.DataFrame(rows).to_csv(SEQCH_CSV, index=False, lineterminator="\n")
    print(f"[saved] {os.path.basename(SEQCH_CSV)} ({len(rows):,} sites)")


def summarise_seqchanged():
    if not os.path.isfile(SEQCH_CSV):
        print("== Sequence-changed models: run `seqchanged` in the py311 env first ==")
        return
    print("== Sites on sequence-changed models (section: seqchanged) ==")
    s = pd.read_csv(SEQCH_CSV)
    s["all_same"] = s.n_same_residue_in_v6 == s.n_union_found_v4
    print(f"sites {len(s):,} on {s.acc.nunique():,} accessions; union residues {int(s.n_union_found_v4.sum()):,}, with the same amino "
          f"acid at the same number in v6 {int(s.n_same_residue_in_v6.sum()):,} "
          f"({100 * s.n_same_residue_in_v6.sum() / s.n_union_found_v4.sum():.1f}%); sites where every residue matches "
          f"{int(s.all_same.sum()):,} ({100 * s.all_same.mean():.1f}%), where at least one differs {int((~s.all_same).sum()):,}")
    print(f"mean pocket pLDDT: v4 {s.pocket_plddt_v4.mean():.1f}, v6 {s.pocket_plddt_v6.mean():.1f}")
    d = _sens_frame().merge(s[["site_uid", "all_same"]], on="site_uid")
    for flag, lab in ((False, "v6 holds other residues at the site numbers"), (True, "same residues in v6")):
        g = d[d.all_same == flag]
        print(f"  {lab}: {len(g):,} scored sites")
        for m in ("p2rank", "fpocket", "ifsitepred", "consensus"):
            print(f"    {m:<10s} v4 {100 * g[m + '_v4'].mean():.1f}%, v6 {100 * g[m + '_v6'].mean():.1f}%, same outcome "
                  f"{100 * (g[m + '_v4'].round() == g[m + '_v6'].round()).mean():.1f}%")


DOWN_DIR = os.path.join(str(P.OUT), "afdb_v4_realign")   # printouts of the manuscript-side scripts under the switch


def summarise_downstream():
    """Numbers of the manuscript-side analyses on v4 next to v6. They are computed by separation_shape.py and
    balanced_classification.py `sites` (run with AHOJ_AF_MODEL_SET=afdbv4 and unset); this section only reads their
    saved printouts and tables from <manuscript_analyses_dir>/afdb_v4_realign/ (separation_shape_{afdbv4,v6}.txt, the
    stdout of separation_shape.py under each setting, and balanced_sites_csv/, balanced_classification.py `sites
    --csv-dir` under afdbv4) and echoes the lines it quotes."""
    txt = {v: os.path.join(DOWN_DIR, f"separation_shape_{v}.txt") for v in ("afdbv4", "v6")}
    if not all(os.path.isfile(f) for f in txt.values()):
        print("== Downstream on v4: separation_shape printouts not there yet ==")
        return
    print("== Manuscript-side analyses on v4 against v6 (section: downstream) ==")
    for ver, f in txt.items():
        lab = "v4" if ver == "afdbv4" else "v6"
        for line in open(f, encoding="utf-8", errors="replace"):
            s = line.rstrip()
            if s.startswith(("LBS ST_100 pockets", "  docking set:", "  consensus    v.low", "  docking      v.low")) or \
                    (s.startswith(("  consensus        ", "  docking          ")) and "[" in s and s.count("[") == 1):
                print(f"  separation_shape, {lab}: {s.strip()}")
    q4 = os.path.join(LBS, "quality_gated_afdbv4", "pooled_overall_ratio_vs_class.csv")
    if os.path.isfile(q4):       # SI Figure S7 (analyze_lbs_ratio_vs_class.py): what predicts success, pooled over the four tasks
        a, b = pd.read_csv(q4), pd.read_csv(os.path.join(LBS, "quality_gated", "pooled_overall_ratio_vs_class.csv"))
        m = a.merge(b, on=["regime", "gate"], suffixes=("_v4", "_v6"))
        for r in m.itertuples(index=False):
            print(f"  ratio vs class, {r.regime} / {r.gate}: n {int(r.n_pockets_v4):,} | {int(r.n_pockets_v6):,}; Spearman with success, "
                  f"evidence ratio {r.rho_log2_ratio_v4:.3f} | {r.rho_log2_ratio_v6:.3f}, p(holo) {r.rho_p_holo_v4:.3f} | "
                  f"{r.rho_p_holo_v6:.3f}, class {r.rho_is_holo_v4:.3f} | {r.rho_is_holo_v6:.3f}; class surpasses ratio "
                  f"{r.surpasses_ratio_is_holo_v4} | {r.surpasses_ratio_is_holo_v6}")
    g4 = os.path.join(DOWN_DIR, "balanced_sites_csv", "balanced_downstream_gaps.csv")
    if os.path.isfile(g4):
        a, b = pd.read_csv(g4), pd.read_csv(P.out("balanced_downstream_gaps.csv"))
        k = ["task", "labels", "estimate", "stratum"]
        m = a.merge(b, on=k, suffixes=("_v4", "_v6"))
        sel = m[((m.stratum == "all") & m.estimate.isin(["raw", "size_std"])) |
                ((m.estimate == "size_std_span") & (m.stratum == "released separation")) |
                ((m.task == "consensus") & (m.estimate == "size_std") & m.stratum.str.startswith("released separation"))]
        for r in sel.itertuples(index=False):
            print(f"  balanced sites, {r.task}, {r.labels} labels, {r.estimate}, {r.stratum}: n {int(r.n_v4):,} | {int(r.n_v6):,}; "
                  f"gap v4 {r.gap_v4:.2f} [{r.lo_v4:.2f}, {r.hi_v4:.2f}], v6 {r.gap_v6:.2f} [{r.lo_v6:.2f}, {r.hi_v6:.2f}]")


B = 400                      # as separation_shape.Gap
SEED = 20260908
TIERS = ["very_low", "low", "moderate", "high", "very_high"]
SENS_CSV = P.out("afdb_v4_realign_sensitivity.csv")                       # committed


def _sens_frame():
    """One row per site scored under both versions by all three detection tools; outcome columns <tool>_<version>."""
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category.to_dict()
    files = {"p2rank": P2R_HARM, "fpocket": {"v4": LBS_TOOLS["fpocket"][2], "v6": LBS_TOOLS["fpocket"][3]},
             "ifsitepred": {"v4": LBS_TOOLS["ifsitepred"][2], "v6": LBS_TOOLS["ifsitepred"][3]}}
    base = pd.read_csv(files["p2rank"]["v4"], low_memory=False,
                       usecols=["site_uid", "site_uniprot", "pocket_class", "separation_level_db"])
    for tool, f in files.items():
        for ver in ("v4", "v6"):
            t = pd.read_csv(f[ver], usecols=["site_uid", "success_tns"], low_memory=False)
            base = base.merge(t.rename(columns={"success_tns": f"{tool}_{ver}"}), on="site_uid", how="left")
    cols = [f"{t}_{v}" for t in files for v in ("v4", "v6")]
    d = base[base[cols].notna().all(axis=1) & base.pocket_class.isin(["apo", "holo"])].copy()
    d[cols] = d[cols].astype(float)
    for ver in ("v4", "v6"):
        d[f"consensus_{ver}"] = d[[f"{t}_{ver}" for t in files]].mean(axis=1)      # as Figure 3: mean of the three outcomes
    d["category"] = d.site_uniprot.map(cat)
    d["sep"] = d.separation_level_db.astype(str).str.strip().str.lower()
    return d


def _sens_rows(d, methods, strata):
    """Rates, gaps and the v4 - v6 differences with cluster-bootstrap intervals (clusters = UniProt accessions; one
    set of resamples for every method, stratum and version, so every difference is paired)."""
    rng = np.random.default_rng(SEED)
    codes, _ = pd.factorize(d.site_uniprot)
    nc = codes.max() + 1
    draws = rng.integers(0, nc, size=(B, nc))
    W = np.zeros((B, nc))
    for b in range(B):
        W[b] = np.bincount(draws[b], minlength=nc)
    ish = (d.pocket_class.values == "holo")
    rows = []
    for sname, mask in strata:
        m = mask.values if hasattr(mask, "values") else mask
        per = {}
        for k, cm in (("apo", m & ~ish), ("holo", m & ish)):
            per[k, "n"] = np.bincount(codes[cm], minlength=nc).astype(float)
        for meth in methods:
            y4, y6 = d[f"{meth}_v4"].values, d[f"{meth}_v6"].values
            s = {}
            for k, cm in (("apo", m & ~ish), ("holo", m & ish)):
                s[k, "v4"] = np.bincount(codes[cm], weights=y4[cm], minlength=nc)
                s[k, "v6"] = np.bincount(codes[cm], weights=y6[cm], minlength=nc)

            def rate(k, v, w=None):
                if w is None:
                    return 100 * s[k, v].sum() / per[k, "n"].sum()
                return 100 * (w @ s[k, v]) / (w @ per[k, "n"])
            pt = {(k, v): rate(k, v) for k in ("apo", "holo") for v in ("v4", "v6")}
            bt = {(k, v): rate(k, v, W) for k in ("apo", "holo") for v in ("v4", "v6")}
            dgap = (bt["holo", "v4"] - bt["apo", "v4"]) - (bt["holo", "v6"] - bt["apo", "v6"])
            same = (np.round(y4[m]) == np.round(y6[m])) if meth == "consensus" else (y4[m] == y6[m])
            row = {"method": meth, "stratum": sname, "n_apo": int(per["apo", "n"].sum()), "n_holo": int(per["holo", "n"].sum()),
                   "apo_v4": pt["apo", "v4"], "apo_v6": pt["apo", "v6"], "holo_v4": pt["holo", "v4"], "holo_v6": pt["holo", "v6"],
                   "gap_v4": pt["holo", "v4"] - pt["apo", "v4"], "gap_v6": pt["holo", "v6"] - pt["apo", "v6"]}
            row["dgap"] = row["gap_v4"] - row["gap_v6"]
            row["dgap_lo"], row["dgap_hi"] = np.nanpercentile(dgap, [2.5, 97.5])
            for k in ("apo", "holo"):
                dd = bt[k, "v4"] - bt[k, "v6"]
                row[f"d{k}"] = pt[k, "v4"] - pt[k, "v6"]
                row[f"d{k}_lo"], row[f"d{k}_hi"] = np.nanpercentile(dd, [2.5, 97.5])
            row["same_outcome_pct"] = 100 * same.mean()
            rows.append(row)
    return pd.DataFrame(rows)


DOCK_MERGED = {"v4": os.path.join(DOCK, "run3_afdbv4", "analysis", "docking_run3_merged.csv"),
               "v6": os.path.join(DOCK, "run3", "analysis", "docking_run3_merged.csv")}


def _dock_frame():
    """One row per pocket docked under both versions: the dockable fraction of its cognate (group 1) jobs with a pose,
    dockable = pose in pocket and affinity_top <= -6 (as separation_shape.load_docking)."""
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category.to_dict()
    per = {}
    for ver, f in DOCK_MERGED.items():
        dk = pd.read_csv(f, low_memory=False, usecols=["site_uid", "ligand_group", "status", "affinity_top", "pose_in_pocket"])
        dk = dk[(dk.ligand_group == 1) & (dk.status.astype(str).str.lower() == "ok")].copy()
        inpk = dk.pose_in_pocket.astype(str).str.lower().isin(["true", "1", "yes"])
        dk["ok"] = (inpk & (pd.to_numeric(dk.affinity_top, errors="coerce") <= -6)).astype(float)
        per[ver] = dk.groupby("site_uid").ok.mean().rename(f"docking_{ver}")
        print(f"  docking, {ver}: group-1 jobs with a pose {len(dk):,} on {dk.site_uid.nunique():,} pockets; "
              f"job-level dockable {100 * dk.ok.mean():.2f}%")
    base = pd.read_csv(P2R_HARM["v4"], low_memory=False,
                       usecols=["site_uid", "site_uniprot", "pocket_class", "separation_level_db"])
    d = base.merge(per["v4"], on="site_uid").merge(per["v6"], on="site_uid")
    d = d[d.pocket_class.isin(["apo", "holo"])].copy()
    d["category"] = d.site_uniprot.map(cat)
    d["sep"] = d.separation_level_db.astype(str).str.strip().str.lower()
    return d


REDOCK = {"v4": os.path.join(DOCK, "run3_afdbv4", "reference_poses", "redock_rmsd.csv"),
          "v6": os.path.join(DOCK, "run3", "reference_poses", "redock_rmsd.csv")}


def _redock_frame():
    """One row per cognate docking job scored in a good frame under both versions (rmsd_status ok, pocket
    superposition, superposition RMSD at most 2 A; as summarize_redock.py). Outcome: redock RMSD at most 2 A."""
    cat = pd.read_csv(ACC_CSV, usecols=["acc", "category"]).set_index("acc").category.to_dict()
    per = {}
    for ver, f in REDOCK.items():
        r = pd.read_csv(f, low_memory=False)
        n_all, n_ok = len(r), int((r.rmsd_status == "ok").sum())
        r = r[(r.rmsd_status == "ok") & (r.superpose_mode == "pocket") &
              (pd.to_numeric(r.superpose_rmsd, errors="coerce") <= 2.0)].copy()
        r[f"redock_{ver}"] = r.redock_success.astype(str).str.lower().isin(["true", "1"]).astype(float)
        print(f"  redock, {ver}: jobs with a pose {n_all:,}; scored {n_ok:,}; good frame {len(r):,}; at most 2 A "
              f"{100 * r[f'redock_{ver}'].mean():.2f}%; median RMSD {r.redock_rmsd_top.median():.2f} A; "
              f"median superposition RMSD {pd.to_numeric(r.superpose_rmsd).median():.3f} A")
        per[ver] = r[["job_id", "site_uid", f"redock_{ver}", "redock_rmsd_top"]].rename(
            columns={"redock_rmsd_top": f"rmsd_{ver}"})
    d = per["v4"].merge(per["v6"].drop(columns="site_uid"), on="job_id")
    base = pd.read_csv(P2R_HARM["v4"], low_memory=False,
                       usecols=["site_uid", "site_uniprot", "pocket_class", "separation_level_db"])
    d = d.merge(base, on="site_uid")
    d = d[d.pocket_class.isin(["apo", "holo"])].copy()
    d["category"] = d.site_uniprot.map(cat)
    d["sep"] = d.separation_level_db.astype(str).str.strip().str.lower()
    return d


def _print_sens(out):
    for r in out.itertuples(index=False):
        print(f"  {r.method:<10s} {r.stratum:<22s} n {r.n_apo:>6,}/{r.n_holo:>6,}  apo {r.apo_v4:5.2f} | {r.apo_v6:5.2f}  "
              f"holo {r.holo_v4:5.2f} | {r.holo_v6:5.2f}  gap {r.gap_v4:5.2f} | {r.gap_v6:5.2f}  "
              f"d gap {r.dgap:+5.2f} [{r.dgap_lo:+5.2f}, {r.dgap_hi:+5.2f}]  d apo {r.dapo:+5.2f} [{r.dapo_lo:+5.2f}, {r.dapo_hi:+5.2f}]  "
              f"d holo {r.dholo:+5.2f} [{r.dholo_lo:+5.2f}, {r.dholo_hi:+5.2f}]  same {r.same_outcome_pct:6.2f}%")


def _strata(d):
    st = [("all", pd.Series(True, index=d.index))]
    st += [(f"model {c}", d.category == c) for c in ("identical", "moved", "different_atoms")]
    return st + [(f"separation {t}", d.sep == t) for t in TIERS]


def summarise_sensitivity(write=True):
    if not all(os.path.isfile(LBS_TOOLS[t][2]) for t in LBS_TOOLS):
        print("== Model-version sensitivity: detection tools not all scored yet ==")
        return
    print("== Model-version sensitivity of the detection results (section: sensitivity) ==")
    d = _sens_frame()
    print(f"common set: {len(d):,} sites on {d.site_uniprot.nunique():,} accessions (AF-Apo {int((d.pocket_class == 'apo').sum()):,}, "
          f"AF-Holo {int((d.pocket_class == 'holo').sum()):,}); cluster bootstrap by accession, B = {B}, seed {SEED}, "
          f"resamples shared between versions")
    strata = [("all", pd.Series(True, index=d.index))]
    strata += [(f"model {c}", d.category == c) for c in ("identical", "moved", "different_atoms")]
    strata += [(f"separation {t}", d.sep == t) for t in TIERS]
    out = _sens_rows(d, ["p2rank", "fpocket", "ifsitepred", "consensus"], strata)
    for name, mask in strata:
        print(f"  stratum {name}: {int(mask.sum()):,} sites")
    _print_sens(out)
    if all(os.path.isfile(f) for f in DOCK_MERGED.values()):
        print("-- docking (cognate ligands, pocket level) --")
        k = _dock_frame()
        print(f"  common set: {len(k):,} pockets on {k.site_uniprot.nunique():,} accessions (AF-Apo "
              f"{int((k.pocket_class == 'apo').sum()):,}, AF-Holo {int((k.pocket_class == 'holo').sum()):,})")
        dstrata = _strata(k)
        for name, mask in dstrata:
            print(f"  stratum {name}: {int(mask.sum()):,} pockets")
        dout = _sens_rows(k, ["docking"], dstrata)
        _print_sens(dout)
        idn = k[k.category == "identical"]
        print(f"  identical models: pockets with a different dockable fraction {int((idn.docking_v4 != idn.docking_v6).sum()):,} "
              f"of {len(idn):,} ({100 * (idn.docking_v4 != idn.docking_v6).mean():.2f}%), run-to-run noise of Uni-Dock")
        out = pd.concat([out, dout], ignore_index=True)
    if all(os.path.isfile(f) for f in REDOCK.values()):
        print("-- redock (cognate ligands, job level, good frame) --")
        q = _redock_frame()
        print(f"  common set: {len(q):,} jobs on {q.site_uid.nunique():,} pockets and {q.site_uniprot.nunique():,} accessions "
              f"(AF-Apo {int((q.pocket_class == 'apo').sum()):,}, AF-Holo {int((q.pocket_class == 'holo').sum()):,})")
        rstrata = _strata(q)
        for name, mask in rstrata:
            print(f"  stratum {name}: {int(mask.sum()):,} jobs")
        rout = _sens_rows(q, ["redock"], rstrata)
        _print_sens(rout)
        idn = q[q.category == "identical"]
        dr = (idn.rmsd_v4 - idn.rmsd_v6).abs()
        print(f"  identical models: jobs with a different outcome {int((idn.redock_v4 != idn.redock_v6).sum()):,} of {len(idn):,} "
              f"({100 * (idn.redock_v4 != idn.redock_v6).mean():.2f}%); RMSD equal to 0.01 A {100 * (dr <= 0.01).mean():.2f}%, "
              f"differing by more than 1 A {100 * (dr > 1).mean():.2f}% (run-to-run noise of Uni-Dock)")
        out = pd.concat([out, rout], ignore_index=True)
    if write:
        out.round(3).to_csv(SENS_CSV, index=False, lineterminator="\n")
        print(f"[saved] {os.path.basename(SENS_CSV)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("geometry")
    sub.add_parser("p2rank")
    sub.add_parser("fpocket")
    sub.add_parser("ifsitepred")
    sub.add_parser("manifest")
    sub.add_parser("calibration")
    sub.add_parser("prep")
    sub.add_parser("sensitivity")
    sub.add_parser("downstream")
    sub.add_parser("seqchanged")
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "geometry":
        summarise_geometry()
    elif a.cmd == "p2rank":
        summarise_p2rank()
    elif a.cmd in LBS_TOOLS:
        summarise_lbs(a.cmd)
    elif a.cmd == "manifest":
        summarise_manifest()
    elif a.cmd == "calibration":
        summarise_calibration()
    elif a.cmd == "prep":
        summarise_prep()
    elif a.cmd == "sensitivity":
        summarise_sensitivity()
    elif a.cmd == "downstream":
        summarise_downstream()
    elif a.cmd == "seqchanged":
        build_seqchanged()
        summarise_seqchanged()
    elif a.cmd == "report":
        summarise_geometry(write=False)
        summarise_p2rank()
        for tool in LBS_TOOLS:
            summarise_lbs(tool)
        summarise_manifest()
        summarise_calibration()
        summarise_prep()
        summarise_sensitivity(write=False)
        summarise_seqchanged()
        summarise_downstream()


if __name__ == "__main__":
    main()
