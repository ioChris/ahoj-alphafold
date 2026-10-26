# -*- coding: utf-8 -*-
"""
Fidelity and regression checks for the run-type paths (docs/RUN_TYPES.md section 7).

    python analysis/run_types_fidelity/check_run_types.py legacy --out DIR [--n 50] [--procs 8]
    python analysis/run_types_fidelity/check_run_types.py afc    --out DIR [--procs 8]
    python analysis/run_types_fidelity/check_run_types.py hq1    --out DIR [--procs 8]

legacy  holo-2.1 on a fixed sample of AHoJ-DB v2c archives (the first --n archives, sorted by name, of
        batches 000 and 300). Re-classifying them must reproduce the released P(holo), n_apo and n_holo
        (query_summaries_do_kde_plots_and_classification.csv of full_run_3) exactly.
afc     afc-2.4 on the pilot's af2 pass (AHoJ 2.4.0, model as query, mapped RMSD). Must reproduce the
        pilot's gate_site classification (pilot_sites_collected.csv, columns af2_gate_site_*), which
        `summarise` in ahoj-af-project's af_centric_rerun.py turns into the brief's numbers.
hq1     holo-2.4 smoke test on the pilot's hq1 pass (AHoJ 2.4.0 on the rep job's holo query). Reports
        the outcome counts under the profile default (v2c_ions_demote_only), and reproduces the pilot's
        hq1_ahoj_* columns under the v2c relabel (relabel_mode v2c_ions, which the profile refuses on a
        gate run; the check passes it explicitly and reports the override).

Everything is read-only on the data folders; outputs go to --out. The interpreter is the repo's py311.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

# Repo-wide path anchors (common/config.py of the ahoj-alphafold repo; set them with environment
# variables or common/paths.ini). The loop finds the repo root however deeply this file is nested.
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

import run_types as rt  # noqa: E402

# -- read-only inputs (common/config.py anchors; override with the environment variables)
V2C_ROOT = os.environ.get("AHOJ_V2C_ROOT", str(cfg.AHOJ_OUTPUT_FIX_DIR))
PILOT_RUNS = os.environ.get("AHOJ_PILOT_RUNS", "<UNSET:AHOJ_PILOT_RUNS>")   # AHoJ 2.4 pilot runs (af2/, hq1/)
RELEASED = os.environ.get("AHOJ_RELEASED_SUMMARY", str(cfg.CLASSIFICATION_SUMMARIES_CSV))  # full_run_3 merged summary
PILOT_COLLECTED = os.environ.get("AHOJ_PILOT_COLLECTED",
                                 str(cfg.WORK_DIR / "af_centric_rerun" / "pilot_sites_collected.csv"))
PILOT_SITES = os.environ.get("AHOJ_PILOT_SITES",
                             os.path.join(HERE, "af_centric_pilot_sites.csv"))   # copied from ahoj-af-project main
CONFIG = os.path.join(REPO, "configs", "run_config_full_run_3.json")

LEGACY_BATCHES = ("000", "300")


def base_args(out_dir, **over):
    """The full_run_3 classifier config with plots and per-entry files off (numbers only)."""
    with open(CONFIG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.update(dict(general_violins=0, dp_curves=0, kde_curves=0, kde_violins=0, conf_bars=0,
                    save_statistics=0, save_filtered_df=0, save_per_method_json=0, save_relabeled_csv=0,
                    zip_plots_dir=0, zip_stats_dir=0, keep_no_model=0,
                    root_mirror_dir=os.path.join(out_dir, "mirror"),
                    run_type="auto", bundle=None, key_format=None, allow_undo_gate=0, sites_tsv=None, relabel_mode=None, ion_rule=None,
                    additive_rule=None, gate=None, volume_column=None, volume_nan_to_zero=None,
                    model_row=None, additives_file=None))
    cfg["add_query_row"] = None            # the profile decides
    cfg.update(over)
    return SimpleNamespace(**cfg)


_ARGS = None


def _init(args, sites_path):
    global _ARGS
    _ARGS = args
    if sites_path:
        rt.set_sites(rt.load_sites_table(sites_path))


def _one(task):
    job, path = task
    from batch_job_functions import do_kde_plots_and_classification
    t0 = time.time()
    status, detail = do_kde_plots_and_classification(_ARGS, job, path)
    rec = {"job_name": job, "status": status, "seconds": round(time.time() - t0, 2)}
    if status == "Success":
        row = detail[4].iloc[0].to_dict()
        rec.update(row)
    else:
        rec["reason"] = str(detail[1]) if len(detail) > 1 else ""
    return rec


def run_jobs(tasks, args, procs, sites_path=None):
    if procs <= 1:
        _init(args, sites_path)
        return pd.DataFrame([_one(t) for t in tasks])
    with Pool(procs, initializer=_init, initargs=(args, sites_path)) as pool:
        return pd.DataFrame(pool.map(_one, tasks))


def _close(a, b, tol=1e-6):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return ((a - b).abs() < tol) | (a.isna() & b.isna())


# =============================================================================== legacy
def cmd_legacy(a):
    os.makedirs(a.out, exist_ok=True)
    tasks = []
    for b in LEGACY_BATCHES:
        fs = sorted(glob.glob(os.path.join(V2C_ROOT, b, "*.tgz")))[: a.n]
        tasks += [(os.path.basename(f)[:-4], f) for f in fs]
    print(f"legacy: {len(tasks)} v2c archives from batches {LEGACY_BATCHES} ({a.n} each, sorted by name)")
    res = run_jobs(tasks, base_args(a.out), a.procs)
    res.to_csv(os.path.join(a.out, "legacy_results.csv"), index=False)
    rel = pd.read_csv(RELEASED, usecols=["job_name", "n_apo", "n_holo", "pocket_avg_P(holo)",
                                         "pocket_overall_classification"], low_memory=False)
    rel = rel[rel.job_name.isin(res.job_name)]
    m = res.merge(rel, on="job_name", how="left", suffixes=("", "_rel"))
    print(f"  outcomes here: {res.status.value_counts().to_dict()}")
    print(f"  run types detected: {res.get('run_type', pd.Series(dtype=object)).dropna().unique().tolist()}")
    ok = m.status.eq("Success") & m["pocket_avg_P(holo)_rel"].notna()
    same_p = _close(m.loc[ok, "pocket_avg_P(holo)"], m.loc[ok, "pocket_avg_P(holo)_rel"])
    same_a = m.loc[ok, "n_apo"].astype(float).eq(m.loc[ok, "n_apo_rel"].astype(float))
    same_h = m.loc[ok, "n_holo"].astype(float).eq(m.loc[ok, "n_holo_rel"].astype(float))
    same_c = m.loc[ok, "pocket_overall_classification"].astype(str).eq(m.loc[ok, "pocket_overall_classification_rel"].astype(str))
    n = int(ok.sum())
    print(f"  jobs classified here AND in the release: {n}")
    print(f"  P(holo) identical: {int(same_p.sum())} of {n}; n_apo: {int(same_a.sum())}; n_holo: {int(same_h.sum())}; "
          f"class: {int(same_c.sum())}")
    all_same = same_p & same_a & same_h & same_c
    print(f"  LEGACY FIDELITY: {int(all_same.sum())} of {n} reproduced exactly")
    not_here = m[(m.status != "Success") & m["pocket_avg_P(holo)_rel"].notna()]
    not_rel = m[(m.status == "Success") & m["pocket_avg_P(holo)_rel"].isna()]
    print(f"  in the release but not classified here: {len(not_here)}; classified here but not in the release: {len(not_rel)}")
    for t in not_here.head(5).itertuples():
        print(f"    {t.job_name}: {t.status} {getattr(t, 'reason', '')}")
    bad = m[ok][~all_same.values]
    for t in bad.head(10).itertuples():
        print(f"    MISMATCH {t.job_name}: P {t._asdict().get('pocket_avg_P(holo)')} vs {t._asdict().get('pocket_avg_P(holo)_rel')}, "
              f"n_apo {t.n_apo} vs {t.n_apo_rel}, n_holo {t.n_holo} vs {t.n_holo_rel}")
    m.to_csv(os.path.join(a.out, "legacy_vs_release.csv"), index=False)
    return int(all_same.sum()), n


# =============================================================================== afc
def pilot_sites_tsv(out_dir):
    """sites.tsv (run-bundle.md columns) from the pilot's site list; members = the v2c job names."""
    s = pd.read_csv(PILOT_SITES, dtype=str, keep_default_na=False)
    t = pd.DataFrame({"site_uid": s.site_uid, "uniprot": s.site_uniprot, "residues": s.resi,
                      "source": "st100-v2c", "members": s.job_names.str.replace(";", ";", regex=False),
                      "n_members": s.members_total, "anchorable": 1, "anchor_note": ""})
    p = os.path.join(out_dir, "sites.tsv")
    t.to_csv(p, sep="\t", index=False)
    return p


def _pilot_expected():
    c = pd.read_csv(PILOT_COLLECTED, low_memory=False)
    keep = ["key", "site_uid", "site_ion", "site_add", "passes", "af2_gate_site_n_apo", "af2_gate_site_n_holo",
            "af2_gate_site_p_holo", "af2_gate_site_overlap", "af2_gate_site_mode_overlap", "af2_gate_site_cls",
            "hq1_ahoj_n_apo", "hq1_ahoj_n_holo", "hq1_ahoj_p_holo", "hq1_ahoj_cls"]
    return c[[k for k in keep if k in c.columns]]


def cmd_afc(a):
    os.makedirs(a.out, exist_ok=True)
    sites = pilot_sites_tsv(a.out)
    fs = sorted(glob.glob(os.path.join(PILOT_RUNS, "af2", "*.tgz")))
    # a killed run can leave a partial .tgz: count an archive only with rc 0 metadata (pilot _complete rule)
    def complete(f):
        meta = os.path.join(PILOT_RUNS, "af2", "_meta", os.path.basename(f)[:-4] + ".json")
        return os.path.isfile(meta) and str(json.load(open(meta, encoding="utf-8")).get("rc")) == "0"
    tasks = [(os.path.basename(f)[:-4], f) for f in fs if complete(f)]
    print(f"afc: {len(tasks)} complete af2 archives of {len(fs)}")
    args = base_args(a.out, sites_tsv=sites)
    res = run_jobs(tasks, args, a.procs, sites_path=sites)
    res.to_csv(os.path.join(a.out, "afc_results.csv"), index=False)
    print(f"  outcomes: {res.status.value_counts().to_dict()}")
    print(f"  run types detected: {res.get('run_type', pd.Series(dtype=object)).dropna().unique().tolist()}; "
          f"rules: {res.get('rule_relabel_mode', pd.Series(dtype=object)).dropna().unique().tolist()} "
          f"{res.get('rule_ion_rule', pd.Series(dtype=object)).dropna().unique().tolist()} "
          f"gate {res.get('rule_gate_thresholds', pd.Series(dtype=object)).dropna().unique().tolist()} "
          f"add_query_row {res.get('rule_add_query_row', pd.Series(dtype=object)).dropna().unique().tolist()}")
    for t in res[res.status != "Success"].itertuples():
        print(f"    {t.job_name}: {t.status} {getattr(t, 'reason', '')}")
    exp = _pilot_expected()
    m = res.merge(exp, left_on="job_name", right_on="key", how="left")
    has_exp = m.af2_gate_site_n_apo.notna()
    ok = m.status.eq("Success")
    same_a = m.n_apo.astype(float).eq(m.af2_gate_site_n_apo.astype(float))
    same_h = m.n_holo.astype(float).eq(m.af2_gate_site_n_holo.astype(float))
    n_counts = int((has_exp & ok).sum())
    print(f"  sites with pilot counts and a Success here: {n_counts}; n_apo identical {int((same_a & has_exp & ok).sum())}, "
          f"n_holo identical {int((same_h & has_exp & ok).sum())}")
    has_p = m.af2_gate_site_p_holo.notna() & ok
    same_p = _close(m["pocket_avg_P(holo)"], m.af2_gate_site_p_holo) & has_p
    same_ov = _close(m["pocket_overall_overlap"], m.af2_gate_site_overlap, 1e-6) & has_p
    same_cls = m["pocket_overall_classification"].astype(str).str.lower().eq(m.af2_gate_site_cls.astype(str).str.lower()) & has_p
    print(f"  sites the pilot classified: {int(has_p.sum())}; P(holo) identical {int(same_p.sum())}, overlap identical "
          f"{int(same_ov.sum())}, class identical {int(same_cls.sum())}")
    full = same_a & same_h & same_p & same_cls
    print(f"  AFC-2.4 FIDELITY (gate_site): {int(full.sum())} of {int(has_p.sum())} classified sites match the pilot "
          f"(n_apo, n_holo, P(holo), class)")
    # sites the pilot could not classify (no apo or no holo): counts must still agree
    no_p = has_exp & ok & m.af2_gate_site_p_holo.isna()
    print(f"  sites the pilot left unclassified (one class empty): {int(no_p.sum())}; counts identical there "
          f"{int((same_a & same_h & no_p).sum())}; parse-ahoj outcome on them: "
          f"{m.loc[no_p, 'pocket_overall_classification'].astype(str).value_counts().to_dict()}")
    for t in m[has_exp & ok & ~(same_a & same_h)].head(10).itertuples():
        print(f"    COUNT MISMATCH {t.job_name}: n_apo {t.n_apo} vs {t.af2_gate_site_n_apo}, n_holo {t.n_holo} vs {t.af2_gate_site_n_holo}")
    for t in m[has_p & ~(same_p & same_cls)].head(10).itertuples():
        print(f"    P MISMATCH {t.job_name}: P {t._asdict().get('pocket_avg_P(holo)')} vs {t.af2_gate_site_p_holo}")
    m.to_csv(os.path.join(a.out, "afc_vs_pilot.csv"), index=False)
    # loud-failure demonstration: add_query_row=1 on an afc archive must be an Error, not a number
    demo = run_jobs(tasks[:1], base_args(a.out, sites_tsv=sites, add_query_row=1), 1, sites_path=sites)
    print(f"  loud check: add_query_row=1 on {tasks[0][0]} -> {demo.status.iloc[0]}: {str(demo.reason.iloc[0])[:120]}")
    return int(full.sum()), int(has_p.sum())


# =============================================================================== hq1
def cmd_hq1(a):
    os.makedirs(a.out, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(PILOT_RUNS, "hq1", "*.tgz")))
    tasks = [(os.path.basename(f)[:-4], f) for f in fs]
    print(f"hq1: {len(tasks)} archives (AHoJ 2.4.0, holo query, gate on, v6 model as candidate)")
    # profile default: holo-2.4 = keep the gate labels, demote ion-only holo calls
    # the pilot named this holo pass by site key, so the profile's holo-key check is relaxed (key_format any)
    res = run_jobs(tasks, base_args(a.out, key_format="any"), a.procs)
    res.to_csv(os.path.join(a.out, "hq1_results_profile.csv"), index=False)
    print(f"  profile default (v2c_ions_demote_only): outcomes {res.status.value_counts().to_dict()}; run types "
          f"{res.get('run_type', pd.Series(dtype=object)).dropna().unique().tolist()}; volume column "
          f"{res.get('rule_volume_column', pd.Series(dtype=object)).dropna().unique().tolist()}; "
          f"AHoJ volume_mode {res.get('ahoj_volume_mode', pd.Series(dtype=object)).dropna().unique().tolist()}")
    for t in res[res.status == "Error"].head(5).itertuples():
        print(f"    {t.job_name}: {t.status} {getattr(t, 'reason', '')}")
    ok = res[res.status == "Success"]
    if len(ok):
        print(f"  classified {len(ok)}: {ok['pocket_overall_classification'].value_counts().to_dict()}; "
              f"chain verdicts kept meaningful: {ok.chain_axes_meaningful.unique().tolist()}; "
              f"mean n_apo {ok.n_apo.mean():.1f}, n_holo {ok.n_holo.mean():.1f}")
    # the pilot's hq1_ahoj_* columns used the FULL v2c relabel on the 2.4 labels (undoing the gate); the
    # profile refuses that on a gate run, so this is an explicit override, reported as such
    exp = _pilot_expected()
    res2 = run_jobs(tasks, base_args(a.out, key_format="any", relabel_mode="v2c_ions", allow_undo_gate=1), a.procs)
    res2.to_csv(os.path.join(a.out, "hq1_results_v2c_relabel.csv"), index=False)
    m = res2.merge(exp, left_on="job_name", right_on="key", how="left")
    okk = m.status.eq("Success") & m.hq1_ahoj_p_holo.notna()
    same = (_close(m["pocket_avg_P(holo)"], m.hq1_ahoj_p_holo) & m.n_apo.astype(float).eq(m.hq1_ahoj_n_apo.astype(float))
            & m.n_holo.astype(float).eq(m.hq1_ahoj_n_holo.astype(float))) & okk
    print(f"  override relabel_mode=v2c_ions (the pilot's hq1_ahoj rule): outcomes {res2.status.value_counts().to_dict()}; "
          f"HQ1 FIDELITY vs pilot hq1_ahoj_*: {int(same.sum())} of {int(okk.sum())} (n_apo, n_holo, P(holo))")
    for t in m[okk & ~same].head(8).itertuples():
        print(f"    MISMATCH {t.job_name}: P {t._asdict().get('pocket_avg_P(holo)')} vs {t.hq1_ahoj_p_holo}, "
              f"n_apo {t.n_apo} vs {t.hq1_ahoj_n_apo}, n_holo {t.n_holo} vs {t.hq1_ahoj_n_holo}")
    # how much the gate changes on this pass: profile default vs the v2c relabel
    j = res[["job_name", "n_apo", "n_holo", "pocket_avg_P(holo)"]].merge(
        res2[["job_name", "n_apo", "n_holo", "pocket_avg_P(holo)"]], on="job_name", suffixes=("_gate", "_v2c"))
    j = j.dropna()
    print(f"  gate kept vs undone on {len(j)} jobs: holo rows {j.n_holo_gate.sum():.0f} vs {j.n_holo_v2c.sum():.0f}, "
          f"apo rows {j.n_apo_gate.sum():.0f} vs {j.n_apo_v2c.sum():.0f}; P(holo) differs at "
          f"{int((~_close(j['pocket_avg_P(holo)_gate'], j['pocket_avg_P(holo)_v2c'], 1e-9)).sum())} jobs")
    m.to_csv(os.path.join(a.out, "hq1_vs_pilot.csv"), index=False)
    return int(same.sum()), int(okk.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("legacy", cmd_legacy), ("afc", cmd_afc), ("hq1", cmd_hq1)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--procs", type=int, default=8)
        if name == "legacy":
            p.add_argument("--n", type=int, default=50, help="archives per batch")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
