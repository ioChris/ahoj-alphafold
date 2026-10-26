# -*- coding: utf-8 -*-
"""Is the evidence ratio just PDB study effort? Success against the ratio at fixed total evidence.

Backs Supplementary Table S11 (success against the apo/holo predictors with study effort controlled) and
the "total evidence, ratio fixed" column of Supplementary Table S12. The manuscript's claim is that the *balance* of apo and holo evidence predicts downstream success. A site
is holo-leaning both when its biology favours the bound state and when its bound state has simply been
studied more, so the balance is confounded with the *amount*:

  * if success tracks the TOTAL (n_apo + n_holo), the story is study effort -- well-characterised
    proteins are easier for every tool;
  * if success tracks the RATIO at fixed total, the story is state composition, which is the claim.

Nothing in the codebase separated these for the downstream outcomes. `balanced_classification.py`
answers it for the *label* (at fixed ratio the AF-Holo share moves 12.8 points across total-evidence
quartiles against 70.7 points across ratio bins at fixed total), and `size_ratio_std` bounds the class
gap at fixed size and fixed ratio -- but neither asks whether SUCCESS follows the amount.

What this adds: a third control regime beside Supplementary Figure S6's two.

    zero      partial rho(success, predictor)                       no controls
    ctrl      partial rho(success, predictor | pocket size)         the published regime
    ctrl+tot  partial rho(success, predictor | size, log10 total)   NEW
    mirror    partial rho(success, log10 total | size, log2 ratio)  NEW -- the amount's own effect

The mirror is the decisive one: if study effort were doing the work, success would correlate with the
total once the ratio is held fixed.

Machinery is reused, not reimplemented: the same loaders, gates, common-mask and partial-Spearman code
that draw SI Figure S7, so the `zero` and `ctrl` columns printed here must reproduce that figure's
published values. The script checks that and says so.

Usage (base conda python; the `plotting` env lacks pandas extras this needs):

    python evidence_amount_vs_ratio.py [--out <csv>]
    AHOJ_AF_MODEL_SET=afdbv4 python evidence_amount_vs_ratio.py      # the v4 re-run (the manuscript's)

--out defaults to P.OUT/evidence_amount_vs_ratio{_afdbv4}.csv; a bare file name is placed in P.OUT.

Intervals are a cluster bootstrap over UniProt accession (B = 400, seed 20260922), resampled once per
gate and shared by every regime so the regime differences are paired.
"""
from __future__ import annotations

import ma_paths as P  # roots; puts the classification code on sys.path

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC = P.REPO
sys.path.insert(0, str(PUBLIC))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, PUBLIC / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qg = _load("qg_parent", "05_docking/runner/analyze_quality_gated_ratio_vs_class.py")
s7 = _load("s7_lbs", "06_lbs_generality/analysis/analyze_lbs_ratio_vs_class.py")

TOTAL = "log10_total_evidence"
GATES = [("all", None), ("prob_mod+", qg.CONFIDENT), ("prob_high+", qg.HIGH)]
METHODS = ["fpocket", "ifsitepred", "p2rank", "docking"]
B, SEED = 400, 20260922


def method_data(tool: str):
    if tool == "docking":
        return qg.load_docking(s7.DOCKING_MERGED, group=1)
    return s7.load_lbs(s7.LBS_TABLES[tool], tool)


def frame(md):
    """Per-pocket frame with the total-evidence covariate and a cluster id attached."""
    g = md.g.copy()
    tot = pd.to_numeric(g["n_apo"], errors="coerce") + pd.to_numeric(g["n_holo"], errors="coerce")
    g[TOTAL] = np.log10(tot.where(tot > 0))
    pid = "site_uid" if "site_uid" in g.columns else "unit_uid"
    g["_clu"] = g[pid].astype(str).str.split(":").str[0]
    return g


def regimes(md):
    """(name, control columns) for the four regimes. `md.ctrl_cols` is the published size control:
    pocket rg for detection, rg + ligand heavy atoms for docking."""
    size = list(md.ctrl_cols)
    return [("zero", []), ("ctrl", size), ("ctrl+tot", size + [TOTAL])]


def rhos_for(sub, ycol, preds, ctrl):
    y = sub[ycol].to_numpy(float)
    P = sub[preds].to_numpy(float)
    C = sub[ctrl].to_numpy(float) if ctrl else np.empty((len(sub), 0))
    r, _, n = qg.partial_rhos(y, P, C)
    return np.asarray(r, float), n


# --------------------------------------------------------------------------- the fast bootstrap
# `qg.partial_rhos` re-ranks its whole matrix on every call, and the bootstrap needs four calls
# per resample (three regimes + the mirror) on up to 87k rows -- four full rank passes each time.
# Ranking is per-column and independent of which controls a regime uses, so one rank pass over the
# union of the columns serves all four: rank once, then slice the ranked columns per regime and go
# straight to `_partial_rhos_from_ranks`. Same estimator, same numbers, a quarter of the ranking.
def _plan(md):
    """Column layout for the union matrix, and each regime's slice into it.

    Union order: [y, RATIO, p_holo, is_holo, *size, TOTAL]. A regime is (name, n_pred, cols),
    where `cols` indexes the union in the order `_partial_rhos_from_ranks` expects.
    """
    size = list(md.ctrl_cols)
    union = [qg.RATIO, "p_holo", "is_holo"] + size + [TOTAL]     # predictors first, then controls
    at = {c: 1 + i for i, c in enumerate(union)}                 # +1: column 0 is y
    P = [at[c] for c in qg.PREDICTORS]
    S = [at[c] for c in size]
    T = at[TOTAL]
    plan = [("zero", 3, [0] + P),
            ("ctrl", 3, [0] + P + S),
            ("ctrl+tot", 3, [0] + P + S + [T]),
            # the mirror: predictor = total evidence, controls = size + the ratio
            ("mirror_total", 1, [0, T] + S + [at[qg.RATIO]])]
    return union, plan


def _rhos_all(M, plan):
    """All four regimes from ONE rank pass over the union matrix M."""
    Vr = qg._rank_cols(M)
    return {name: qg._partial_rhos_from_ranks(Vr[:, cols], n_pred, len(cols) - 1 - n_pred)
            for name, n_pred, cols in plan}


def run(out_csv: Path):
    sfx = "_afdbv4" if os.environ.get("AHOJ_AF_MODEL_SET", "").strip().lower() == "afdbv4" else ""
    print("=" * 96)
    print(f"  success vs the evidence ratio at fixed total evidence   [model set {sfx or 'July/v6'!r}]")
    print("=" * 96)

    rows = []
    rng = np.random.default_rng(SEED)
    for tool in METHODS:
        md = method_data(tool)
        g = frame(md)
        ycol = md.primary().col
        size = list(md.ctrl_cols)
        print(f"\n--- {tool}  (outcome {ycol}; size control {size}) ---")
        for gname, levels in GATES:
            gm = qg.gate_mask(g, levels, None)
            # one common mask across ALL regimes, so the regimes differ only in the control set
            need = [ycol, qg.RATIO, "p_holo", "is_holo", TOTAL] + size
            sub = g[gm & g[need].notna().all(axis=1)]
            if len(sub) < 200:
                print(f"  {gname:<11} n={len(sub):,}  (too few rows)")
                continue

            union, plan = _plan(md)
            M = np.column_stack([sub[ycol].to_numpy(float)] +
                                [sub[c].to_numpy(float) for c in union])
            pts = _rhos_all(M, plan)

            # gate: the published regimes must come out of the fast path unchanged
            for rname, ctrl in regimes(md):
                ref, _n = rhos_for(sub, ycol, qg.PREDICTORS, ctrl)
                if not np.allclose(ref, pts[rname], atol=1e-9, equal_nan=True):
                    raise AssertionError(f"fast path disagrees with qg.partial_rhos "
                                         f"for {tool}/{gname}/{rname}")

            # paired cluster bootstrap over UniProt: one rank pass per resample
            clus = sub["_clu"].to_numpy()
            uniq = np.unique(clus)
            idx_of = {c: np.flatnonzero(clus == c) for c in uniq}
            boot = {k: [] for k, _n, _c in plan}
            for _ in range(B):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                take = np.concatenate([idx_of[c] for c in pick])
                r = _rhos_all(M[take], plan)
                for k, v in r.items():
                    boot[k].append(v)

            def ci(arr, j=0):
                a = np.asarray(arr, float)[:, j]
                a = a[np.isfinite(a)]
                return (np.nan, np.nan) if a.size < 10 else (float(np.percentile(a, 2.5)),
                                                             float(np.percentile(a, 97.5)))

            print(f"  {gname:<11} n={len(sub):,}  clusters={len(uniq):,}")
            for j, p in enumerate(qg.PREDICTORS):
                bits = []
                for rname in ("zero", "ctrl", "ctrl+tot"):
                    lo, hi = ci(boot[rname], j)
                    bits.append(f"{rname} {pts[rname][j]:+.3f} [{lo:+.3f},{hi:+.3f}]")
                    rows.append({"model_set": sfx or "v6", "method": tool, "gate": gname,
                                 "n": len(sub), "clusters": len(uniq), "regime": rname,
                                 "predictor": p, "rho": pts[rname][j], "lo": lo, "hi": hi})
                print(f"    {p:<11} " + "   ".join(bits))
            lo, hi = ci(boot["mirror_total"], 0)
            print(f"    {'TOTAL|ratio':<11} mirror {pts['mirror_total'][0]:+.3f} [{lo:+.3f},{hi:+.3f}]"
                  "   <- the amount's own effect, ratio and size fixed")
            rows.append({"model_set": sfx or "v6", "method": tool, "gate": gname, "n": len(sub),
                         "clusters": len(uniq), "regime": "mirror", "predictor": TOTAL,
                         "rho": pts["mirror_total"][0], "lo": lo, "hi": hi})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] -> {out_csv}")

    # ---- the headline, pooled the way SI Figure S7 pools: Fisher-z mean over the four methods
    print("\n" + "=" * 96)
    print("  POOLED over the four methods (Fisher-z mean), ungated and gated")
    print("=" * 96)
    for gname, _ in GATES:
        print(f"\n  gate {gname}")
        for rname in ("zero", "ctrl", "ctrl+tot"):
            bits = []
            for p in qg.PREDICTORS:
                v = df[(df.gate == gname) & (df.regime == rname) & (df.predictor == p)].rho.to_numpy()
                v = v[np.isfinite(v)]
                bits.append(f"{p} {np.tanh(np.arctanh(v).mean()):+.3f}" if v.size else f"{p} n/a")
            print(f"    {rname:<9} " + "   ".join(bits))
        v = df[(df.gate == gname) & (df.regime == "mirror")].rho.to_numpy()
        v = v[np.isfinite(v)]
        if v.size:
            print(f"    {'mirror':<9} total evidence at fixed ratio and size "
                  f"{np.tanh(np.arctanh(v).mean()):+.3f}")
    return df


def main():
    global B
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"evidence_amount_vs_ratio{P.MODEL_SFX}.csv")
    ap.add_argument("--boot", type=int, default=B, help="bootstrap resamples (default 400)")
    a = ap.parse_args()
    B = a.boot
    run(Path(P.out(a.out)))


if __name__ == "__main__":
    main()
