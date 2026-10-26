# -*- coding: utf-8 -*-
"""
Reference figure + coverage table for the AlphaFill<->redock comparison (run_3, G1).

Panels:
  A  COVERAGE  — how many pockets have redock only / both / AlphaFill only (the overlap
                 the per-pocket link is computed on).
  B  LINK      — pockets with a clean AlphaFill transplant redock better (2x enrichment;
                 clash<->redock Spearman rho ~0.18).
  C  ABUNDANCE — redock holo>apo gap is constant across structural-evidence terciles
                 (not a study-intensity artifact).
  D  COMPLETENESS — apo vs holo experimental structures equally observed at the pocket
                 (classification-robustness control; from apo_holo_completeness.json).

Writes: <out_dir>/alphafill_redock_link.{png,pdf,svg} + coverage_alphafill_redock.csv.
Env: `plotting` (matplotlib/seaborn/pandas). Run with KMP_DUPLICATE_LIB_OK=TRUE.
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

RUN3 = cfg.DOCKING_RUN3_DIR
APO, HOLO = "#4C72B0", "#C44E52"     # colour-blind-safe blue/red
BOTH, RONLY, AONLY = "#55A868", "#8172B3", "#CCB974"


def spear(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 20:
        return float("nan")
    ra = pd.Series(a[m]).rank().to_numpy(float, copy=True)
    rb = pd.Series(b[m]).rank().to_numpy(float, copy=True)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redock_csv", type=Path, default=RUN3 / "reference_poses" / "redock_rmsd.csv")
    ap.add_argument("--escort_csv", type=Path, default=RUN3 / "reference_poses" / "escort_g1.csv")
    ap.add_argument("--manifest_csv", type=Path, default=RUN3 / "manifest" / "docking_manifest.csv")
    ap.add_argument("--alphafill_csv", type=Path, default=cfg.MERGED_AHOJ_ALPHAFILL_LONG_CSV)
    ap.add_argument("--completeness_json", type=Path, default=RUN3 / "reference_poses" / "apo_holo_completeness.json")
    ap.add_argument("--out_dir", type=Path, default=RUN3 / "reference_poses")
    args = ap.parse_args()

    # --- redock pockets (good-frame) ---
    rk = pd.read_csv(args.redock_csv, usecols=["site_uid", "rmsd_status", "superpose_mode",
                                               "redock_rmsd_top", "n_template_atoms"], low_memory=False)
    rk = rk[(rk.rmsd_status == "ok") & (rk.superpose_mode == "pocket")].copy()
    rk["r"] = pd.to_numeric(rk.redock_rmsd_top, errors="coerce")
    A = rk.groupby("site_uid").agg(redock_best=("r", "min"), lig=("n_template_atoms", "median")).reset_index()
    man = pd.read_csv(args.manifest_csv, usecols=["site_uid", "ligand_group", "pocket_class",
                                                  "pocket_p_holo", "n_apo_avg", "n_holo_avg"], low_memory=False)
    g1_pockets = set(man[man.ligand_group == 1]["site_uid"])
    man = man.drop_duplicates("site_uid")
    A = A.merge(man[["site_uid", "pocket_class", "pocket_p_holo", "n_apo_avg", "n_holo_avg"]], on="site_uid", how="left")
    A["evid"] = pd.to_numeric(A.n_apo_avg, errors="coerce") + pd.to_numeric(A.n_holo_avg, errors="coerce")

    # --- AlphaFill ST_100 pockets (same_uniprot, identity >= 0.30) ---
    af = pd.read_csv(args.alphafill_csv, usecols=["unit_uid", "threshold", "afill_local_rmsd",
                                                  "afill_clash_score", "afill_cross_uniprot",
                                                  "afill_mapping_align_identity"], low_memory=False)
    af = af[(af.threshold == "ST_100") & (pd.to_numeric(af.afill_mapping_align_identity, errors="coerce") >= 0.30)
            & (af.afill_cross_uniprot == False)]  # noqa: E712
    B = af.groupby("unit_uid").agg(afill_local=("afill_local_rmsd", "min"),
                                   afill_clash=("afill_clash_score", "min")).reset_index().rename(columns={"unit_uid": "site_uid"})

    setA, setB = set(A.site_uid), set(B.site_uid)
    both = setA & setB
    cov = {
        "redock_scored_pockets": len(setA),
        "alphafill_ST100_same_uniprot_pockets": len(setB),
        "both": len(both),
        "redock_only": len(setA - setB),
        "alphafill_only": len(setB - setA),
        "union": len(setA | setB),
        "g1_docking_pockets_universe": len(g1_pockets),
        "pct_of_redock_with_alphafill": round(100 * len(both) / max(1, len(setA)), 1),
        "pct_of_alphafill_with_redock": round(100 * len(both) / max(1, len(setB)), 1),
    }
    pd.DataFrame([cov]).to_csv(args.out_dir / "coverage_alphafill_redock.csv", index=False)
    print("[COVERAGE]", json.dumps(cov, indent=2))

    J = A.merge(B, on="site_uid", how="inner")
    J["clean"] = pd.to_numeric(J.afill_local, errors="coerce") <= 1.5
    rho_clash = spear(J.redock_best, pd.to_numeric(J.afill_clash, errors="coerce"))

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.2))

    # A) coverage
    a = ax[0, 0]
    labels = ["redock\nonly", "both", "AlphaFill\nonly"]
    vals = [cov["redock_only"], cov["both"], cov["alphafill_only"]]
    a.bar(labels, vals, color=[RONLY, BOTH, AONLY])
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    a.set_ylabel("pockets"); a.set_title("A  Coverage / overlap (ST_100, same-UniProt)")
    a.text(0.98, 0.95, f"overlap = {cov['pct_of_redock_with_alphafill']}% of redock\n"
                       f"= {cov['pct_of_alphafill_with_redock']}% of AlphaFill",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5, color="0.3")

    # B) link
    b = ax[0, 1]
    grp = J.groupby("clean")["redock_best"].apply(lambda s: (s <= 3).mean() * 100)
    order = [True, False]
    bx = b.bar(["clean transplant\n(local RMSD ≤1.5 Å)", "poor transplant\n(>1.5 Å)"],
               [grp.get(True, np.nan), grp.get(False, np.nan)], color=["#55A868", "#B0B0B0"])
    for rect, key in zip(bx, order):
        n = int((J.clean == key).sum())
        b.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
               f"{rect.get_height():.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    b.set_ylabel("redock ≤3 Å success (%)")
    b.set_ylim(0, max(grp.get(True, 0), grp.get(False, 0)) * 1.35)
    b.set_title(f"B  Per-pocket link  (clash↔redock ρ={rho_clash:+.2f}, n={len(J):,})")

    # C) abundance control
    c = ax[1, 0]
    A2 = A.dropna(subset=["evid"]).copy()
    A2["T"] = pd.qcut(A2.evid, 3, labels=["low", "mid", "high"], duplicates="drop")
    ts = ["low", "mid", "high"]; x = np.arange(len(ts)); w = 0.38
    apo_s = [(A2[(A2["T"] == t) & (A2.pocket_class == "apo")].redock_best <= 2).mean() * 100 for t in ts]
    holo_s = [(A2[(A2["T"] == t) & (A2.pocket_class == "holo")].redock_best <= 2).mean() * 100 for t in ts]
    c.bar(x - w / 2, apo_s, w, label="apo", color=APO)
    c.bar(x + w / 2, holo_s, w, label="holo", color=HOLO)
    c.set_xticks(x); c.set_xticklabels([f"{t}\nevidence" for t in ts])
    c.set_ylabel("redock ≤2 Å success (%)"); c.set_title("C  Abundance control (gap constant)")
    c.set_ylim(0, max(max(apo_s), max(holo_s)) * 1.28)
    c.legend(frameon=False, fontsize=8, loc="upper left")
    for xi, (av, hv) in enumerate(zip(apo_s, holo_s)):
        c.text(xi, max(av, hv), f"+{hv-av:.1f}pp", ha="center", va="bottom", fontsize=8, color="0.3")

    # D) completeness
    d = ax[1, 1]
    comp = json.loads(Path(args.completeness_json).read_text()) if args.completeness_json.exists() else {}
    pu = comp.get("paired_pocket_pct_unobserved", {}); cc = comp.get("paired_chain_coverage_obs", {})
    metrics = ["% unobserved\npocket resid", "chain coverage\nobserved (%)"]
    apo_v = [pu.get("apo", 0), cc.get("apo", np.nan)]; holo_v = [pu.get("holo", 0), cc.get("holo", np.nan)]
    x2 = np.arange(len(metrics))
    d.bar(x2 - w / 2, apo_v, w, label="apo", color=APO)
    d.bar(x2 + w / 2, holo_v, w, label="holo", color=HOLO)
    d.set_xticks(x2); d.set_xticklabels(metrics); d.set_ylim(0, 118); d.set_ylabel("%")
    d.set_title("D  Pocket observation completeness"); d.legend(frameon=False, fontsize=8, loc="center left")
    for xi, (av, hv) in enumerate(zip(apo_v, holo_v)):
        d.text(xi - w / 2, av, f"{av:.1f}", ha="center", va="bottom", fontsize=8)
        d.text(xi + w / 2, hv, f"{hv:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("AlphaFill transplant ↔ redock: convergence, per-pocket link, and confound controls (run_3, G1)",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"alphafill_redock_link.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {args.out_dir / 'alphafill_redock_link.png'}  (+pdf/svg)  + coverage_alphafill_redock.csv")


if __name__ == "__main__":
    main()
