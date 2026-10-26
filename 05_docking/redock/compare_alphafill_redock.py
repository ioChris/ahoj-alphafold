# -*- coding: utf-8 -*-
"""
AlphaFill transplant vs redock RMSD: convergence, per-pocket quality link, and robustness.

Two methodologically independent ways of placing a cognate ligand into an AF model at the
SAME experimentally-defined (BioLiP/AHoJ) binding site:
  * AlphaFill  — transplant a homologous holo ligand by superposition (quality: local RMSD,
                 clash score);
  * redock     — physics-based docking of the cognate ligand (quality: pose RMSD to crystal).

We do NOT equate their absolute success numbers (AlphaFill coverage is donor-availability;
redock success is pose-accuracy). We report:
  (1) CONVERGENCE  — both show holo > apo;
  (2) PER-POCKET LINK — do pockets that accept a clean transplant also redock better?
      (redock best RMSD vs AlphaFill transplant quality, same-UniProt and cross-UniProt scopes);
  (3) ABUNDANCE CONTROL — the redock holo>apo gap vs structural-evidence abundance
      (is 'holo' just 'well-studied'?), by evidence tercile + partial Spearman.

Companion completeness control: `check_apo_holo_completeness.py`.
AlphaFill scope matches the corrected manuscript universe (T2-qualgate): same-UniProt +
mapping identity >= 0.30 (plus an any/cross-UniProt panel).

Env: `ahoj-docking` (pandas/numpy). Run with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
(numpy corrcoef/BLAS native-crashes on this box; Spearman is computed by explicit formula).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

RUN3 = cfg.DOCKING_RUN3_DIR


def spear(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 20:
        return float("nan"), int(m.sum())
    ra = pd.Series(a[m]).rank().to_numpy(float, copy=True)
    rb = pd.Series(b[m]).rank().to_numpy(float, copy=True)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (float((ra * rb).sum() / d) if d > 0 else float("nan")), int(m.sum())


def partial(x, y, z):
    """Partial Spearman x~y controlling a single covariate z (no BLAS)."""
    rxy = spear(x, y)[0]; rxz = spear(x, z)[0]; ryz = spear(y, z)[0]
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else float("nan")


def load_pockets(redock_csv, escort_csv, manifest_csv):
    rk = pd.read_csv(redock_csv, usecols=["site_uid", "rmsd_status", "superpose_mode",
                                          "redock_rmsd_top", "n_template_atoms"], low_memory=False)
    rk = rk[(rk.rmsd_status == "ok") & (rk.superpose_mode == "pocket")].copy()
    rk["r"] = pd.to_numeric(rk.redock_rmsd_top, errors="coerce")
    pk = rk.groupby("site_uid").agg(redock_best=("r", "min"), lig=("n_template_atoms", "median")).reset_index()
    man = pd.read_csv(manifest_csv, usecols=["site_uid", "pocket_class", "pocket_p_holo",
                                             "n_apo_avg", "n_holo_avg", "pocket_rg", "pocket_separation_pair"],
                      low_memory=False).drop_duplicates("site_uid")
    d = pk.merge(man, on="site_uid", how="left")
    d["evid"] = pd.to_numeric(d.n_apo_avg, errors="coerce") + pd.to_numeric(d.n_holo_avg, errors="coerce")
    d["succ2"] = d.redock_best <= 2
    # verbal apo/holo separation bins — the 5 DATABASE separation classes used by Fig 5
    # (bands on pocket_separation_pair in [0,1]): Very low 0-0.20 / Low 0.20-0.30 /
    # Moderate 0.30-0.60 / High 0.60-0.85 / Very high 0.85-1.00.
    d["sepbin"] = pd.cut(pd.to_numeric(d.pocket_separation_pair, errors="coerce"),
                         [-1e-9, 0.20, 0.30, 0.60, 0.85, 1.0001],
                         labels=["Very low", "Low", "Moderate", "High", "Very high"])
    return d


def af_by_scope(af_path, same_uniprot):
    af = pd.read_csv(af_path, usecols=["unit_uid", "afill_local_rmsd", "afill_clash_score",
                                       "afill_cross_uniprot", "afill_mapping_align_identity"], low_memory=False)
    af = af[pd.to_numeric(af.afill_mapping_align_identity, errors="coerce") >= 0.30]
    if same_uniprot:
        af = af[af.afill_cross_uniprot == False]  # noqa: E712
    g = af.groupby("unit_uid").agg(afill_local=("afill_local_rmsd", "min"),
                                   afill_clash=("afill_clash_score", "min"),
                                   n_transplant=("afill_local_rmsd", "size")).reset_index()
    return g.rename(columns={"unit_uid": "site_uid"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redock_csv", type=Path, default=RUN3 / "reference_poses" / "redock_rmsd.csv")
    ap.add_argument("--escort_csv", type=Path, default=RUN3 / "reference_poses" / "escort_g1.csv")
    ap.add_argument("--manifest_csv", type=Path, default=RUN3 / "manifest" / "docking_manifest.csv")
    ap.add_argument("--alphafill_csv", type=Path, default=cfg.MERGED_AHOJ_ALPHAFILL_LONG_CSV)
    ap.add_argument("--out_dir", type=Path, default=RUN3 / "reference_poses")
    args = ap.parse_args()

    d = load_pockets(args.redock_csv, args.escort_csv, args.manifest_csv)
    out = {"timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
           "n_redock_pockets": int(len(d))}

    def by_class(sub, col, le=None):
        r = {}
        for cls in ["apo", "holo", "tie"]:
            g = sub[sub.pocket_class == cls]
            if not len(g):
                continue
            r[cls] = {"n": int(len(g)), "median": round(float(pd.to_numeric(g[col], errors="coerce").median()), 3)}
            if le is not None:
                r[cls]["success_rate"] = round(float((pd.to_numeric(g[col], errors="coerce") <= le).mean()), 4)
        return r

    # (1) convergence — redock
    out["convergence_redock_rmsd_by_class"] = by_class(d, "redock_best", le=2.0)

    # (3) abundance control
    d["evidT"] = pd.qcut(d.evid, 3, labels=["low", "mid", "high"], duplicates="drop")
    ab = {}
    for t, g in d.groupby("evidT", observed=True):
        ga, gh = g[g.pocket_class == "apo"], g[g.pocket_class == "holo"]
        ab[str(t)] = {"n_struct_range": [float(g.evid.min()), float(g.evid.max())],
                      "apo_success_2A": round(float(ga.succ2.mean()), 4), "apo_n": int(len(ga)),
                      "holo_success_2A": round(float(gh.succ2.mean()), 4), "holo_n": int(len(gh)),
                      "gap_pp": round(float((gh.succ2.mean() - ga.succ2.mean()) * 100), 2)}
    out["abundance_control_by_evidence_tercile"] = ab
    out["partial_spearman_redock_vs_p_holo"] = {
        "zero_order": round(spear(d.redock_best, d.pocket_p_holo)[0], 3),
        "ctrl_evidence": round(partial(d.redock_best, d.pocket_p_holo, d.evid), 3),
        "ctrl_ligand_size": round(partial(d.redock_best, d.pocket_p_holo, d.lig), 3),
        "ctrl_pocket_rg": round(partial(d.redock_best, d.pocket_p_holo, d.pocket_rg), 3),
        "p_holo_vs_evidence": round(spear(d.evid, d.pocket_p_holo)[0], 3),
    }

    # (2b) SEPARATION-CONDITIONAL (verbal bins) — the remodeled-Fig-5 result: the AF-Apo
    # placement penalty concentrates where apo/holo states genuinely diverge, for BOTH modalities.
    def _med(s):
        return round(float(pd.to_numeric(s, errors="coerce").median()), 3)
    sep_redock = {}
    for b in ["Very low", "Low", "Moderate", "High", "Very high"]:
        g = d[d.sepbin == b]; ga, gh = g[g.pocket_class == "apo"], g[g.pocket_class == "holo"]
        sep_redock[b] = {"apo_n": int(len(ga)), "holo_n": int(len(gh)),
                         "apo_median_rmsd": _med(ga.redock_best), "holo_median_rmsd": _med(gh.redock_best),
                         "apo_success_2A": round(float(ga.succ2.mean()), 4), "holo_success_2A": round(float(gh.succ2.mean()), 4),
                         "gap_success_pp": round(float((gh.succ2.mean() - ga.succ2.mean()) * 100), 2),
                         "gap_median_rmsd_apo_minus_holo": round(_med(ga.redock_best) - _med(gh.redock_best), 3)}
    afsu = af_by_scope(args.alphafill_csv, True).merge(d[["site_uid", "pocket_class", "sepbin"]], on="site_uid", how="inner")
    sep_afill = {}
    for b in ["Very low", "Low", "Moderate", "High", "Very high"]:
        g = afsu[afsu.sepbin == b]; ga, gh = g[g.pocket_class == "apo"], g[g.pocket_class == "holo"]
        sep_afill[b] = {"apo_n": int(len(ga)), "holo_n": int(len(gh)),
                        "apo_median_local_rmsd": _med(ga.afill_local), "holo_median_local_rmsd": _med(gh.afill_local),
                        "apo_median_clash": _med(ga.afill_clash), "holo_median_clash": _med(gh.afill_clash)}
    out["separation_conditional_verbal_bins"] = {
        "cutoffs": "Very low 0-0.20 / Low 0.20-0.30 / Moderate 0.30-0.60 / High 0.60-0.85 / "
                   "Very high 0.85-1.00 (the 5 database separation classes; Fig 5 binning)",
        "redock": sep_redock, "alphafill_same_uniprot": sep_afill,
        "note": "apo penalty (gap) grows Low->High for both modalities; see remodeled Fig 5 "
                "(plot_remodeled_fig5.py). AlphaFill overlap = redock-scored pockets only.",
    }

    # (2) per-pocket link — join AlphaFill quality
    scopes = {}
    joined_out = None
    for scope, same in [("same_uniprot", True), ("any_incl_cross_uniprot", False)]:
        af = af_by_scope(args.alphafill_csv, same)
        j = d.merge(af, on="site_uid", how="inner")
        rL, nL = spear(j.redock_best, pd.to_numeric(j.afill_local, errors="coerce"))
        rC, _ = spear(j.redock_best, pd.to_numeric(j.afill_clash, errors="coerce"))
        j["aok"] = pd.to_numeric(j.afill_local, errors="coerce") <= 1.5
        scopes[scope] = {
            "n_pockets_both": int(len(j)),
            "spearman_redock_vs_afill_local_rmsd": round(rL, 3),
            "spearman_redock_vs_afill_clash": round(rC, 3),
            "redock_le3A_rate_clean_transplant": round(float((j[j.aok].redock_best <= 3).mean()), 4),
            "redock_le3A_rate_poor_transplant": round(float((j[~j.aok].redock_best <= 3).mean()), 4),
            "afill_transplant_quality_by_class": {
                "afill_local_rmsd": by_class(j, "afill_local"),
                "afill_clash_score": by_class(j, "afill_clash"),
            },
        }
        if same:
            joined_out = j
    out["per_pocket_link"] = scopes

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "alphafill_redock_comparison.json").write_text(json.dumps(out, indent=2))
    if joined_out is not None:
        joined_out.to_csv(args.out_dir / "alphafill_redock_pockets.csv", index=False)

    # console
    pr = out["partial_spearman_redock_vs_p_holo"]
    print(f"[REDOCK by class] {out['convergence_redock_rmsd_by_class']}")
    print(f"[ABUNDANCE] holo-apo gap by evidence tercile: " +
          ", ".join(f"{t}:{v['gap_pp']:+}pp" for t, v in ab.items()))
    print(f"[SEPARATION] redock holo-apo gap by verbal bin: " +
          ", ".join(f"{b}:{sep_redock[b]['gap_success_pp']:+}pp(medΔ{sep_redock[b]['gap_median_rmsd_apo_minus_holo']:+})"
                    for b in ["Very low", "Low", "Moderate", "High", "Very high"]))
    print(f"[PARTIAL] redock~p_holo zero={pr['zero_order']} |evid={pr['ctrl_evidence']} "
          f"|lig={pr['ctrl_ligand_size']} |Rg={pr['ctrl_pocket_rg']}; p_holo~evid={pr['p_holo_vs_evidence']}")
    for s, v in scopes.items():
        print(f"[LINK {s}] n={v['n_pockets_both']} rho(redock~afill_clash)={v['spearman_redock_vs_afill_clash']} "
              f"rho(redock~afill_local)={v['spearman_redock_vs_afill_local_rmsd']} "
              f"redock<=3A clean={v['redock_le3A_rate_clean_transplant']} poor={v['redock_le3A_rate_poor_transplant']}")
    print(f"[SAVE] {args.out_dir / 'alphafill_redock_comparison.json'}")


if __name__ == "__main__":
    main()
