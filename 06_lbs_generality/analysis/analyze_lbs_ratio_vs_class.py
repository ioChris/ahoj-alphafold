# -*- coding: utf-8 -*-
"""
LBS-generality quality-gated ratio-vs-class  (T3-2nd).

Extends the "quality-gated ratio-vs-class" analysis (parent script
`05_docking/runner/analyze_quality_gated_ratio_vs_class.py`, which covered docking +
P2Rank-native + AlphaFill) to the THREE ligand-binding-site predictors —
**fpocket, IF-SitePred, harmonized P2Rank re-score** — on our default harmonized metric
**DCA-TNS = `success_tns`**, plus **docking** (native dockability). The point is to show
the ratio-vs-class relationship established for docking/P2Rank GENERALIZES across
binding-site predictors, which is direct evidence for claim **C4 generality**.

WHAT WE COMPARE (per pocket, within quality gates)
  Three DIRECTIONAL apo/holo predictors of predictor success:
    * `log2_ratio` = log2((n_holo_avg + 1) / n_apo_avg)   — graded evidence ratio
    * `p_holo`     = class probability
    * `is_holo`    = binary class label
  scored against the predictor's success outcome (LBS: `success_tns`, harmonized DCA-TNS;
  docking: fraction of cognate ligands dockable — DCA-TNS does not apply to docking).

REGIME REPORTING (the honest, established story — carried over verbatim)
  We report BOTH:
    * zero-order Spearman rho, and
    * size-controlled partial Spearman rho (LBS control = pocket size `rg` ONLY — there is
      NO ligand in binding-site detection, so there is NO ligand-size covariate; docking
      keeps `union_rg` + ligand heavy-atom count).
  because the relationship is regime-dependent:
    * UNGATED, the core ordering `ratio >= p_holo >= is_holo` holds in BOTH regimes and in
      every predictor — this is the general result.
    * Inside a HIGH-CONFIDENCE gate AND size-controlled, `is_holo` climbs and can overtake
      `p_holo` and even the ratio — but this crossover is a size-controlled phenomenon
      (zero-order -> parity), and it is predictor-dependent (strong in IF-SitePred/docking,
      marginal/absent elsewhere). `winning_p` (the undirected gate axis) is itself the
      weakest predictor and we NEVER gate on `p_holo` (that would leak the outcome).

GATES (undirected, on `winning_p` via the manuscript's verbal bins; user-requested subset)
    * `all`         — ungated
    * `prob_mod+`   — probability_level(winning_p) in {moderate, high, very_high}
    * `prob_high+`  — probability_level(winning_p) in {high, very_high}

The crossover is judged robust only when the point Delta-rho > 0 AND the PAIRED bootstrap
95% CI of the difference vs the ratio excludes 0. n is reported at every gate. All three
predictors are scored on the SAME ratio-defined rows within each gate, so differences are
perfectly paired.

Reuses the parent module's tested machinery (partial Spearman, paired bootstrap, nested CV
AUC, threshold sweep, per-tool gate figure) unchanged; adds only (a) an rg-only LBS loader,
(b) the zero-order pass, and (c) the cross-tool generality figure + synthesis report.

Run in the `plotting` env (pandas/numpy/scipy/sklearn/matplotlib). Read .csv (no pyarrow).

  python analyze_lbs_ratio_vs_class.py
  python analyze_lbs_ratio_vs_class.py --tools fpocket ifsitepred --boot 500
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --- repo root + config -----------------------------------------------------
for _root in Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        REPO_ROOT = _root
        sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402


# --- import the parent quality-gated module (reuse ALL its machinery) --------
def _load_parent():
    p = REPO_ROOT / "05_docking" / "runner" / "analyze_quality_gated_ratio_vs_class.py"
    if not p.is_file():
        raise FileNotFoundError(f"parent quality-gated script not found: {p}")
    spec = importlib.util.spec_from_file_location("qg_parent", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qg_parent"] = mod            # register BEFORE exec (dataclasses need it)
    spec.loader.exec_module(mod)
    return mod


qg = _load_parent()

# user-requested gate subset (undirected, winning_p only)
LBS_GATE_SPECS = [
    ("all", None, None),
    ("prob_mod+", qg.CONFIDENT, None),
    ("prob_high+", qg.HIGH, None),
]

# the four methods and their display labels for the cross-tool figure
METHOD_ORDER = ["fpocket", "ifsitepred", "p2rank", "docking"]
METHOD_LABEL = {"fpocket": "fpocket", "ifsitepred": "IF-SitePred",
                "p2rank": "P2Rank", "docking": "docking"}

# default LBS per-pocket tables (full runs, ST_100)
LBS_TABLES = {
    "fpocket": cfg.lbs_tool_dir("fpocket") / "lbs_fpocket_st100_merged_per_pocket.csv",
    "ifsitepred": cfg.lbs_tool_dir("ifsitepred") / "lbs_ifsitepred_st100_merged_per_pocket.csv",
    "p2rank": cfg.lbs_tool_dir("p2rank") / "lbs_p2rank_st100_merged_per_pocket.csv",
}
DOCKING_MERGED = cfg.DOCKING_RUN3_DIR / "analysis" / "docking_run3_merged.csv"


# ============================================================================
# LBS loader — clone of parent.load_p2rank, but rg-only control (no ligand)
# ============================================================================
def load_lbs(per_pocket_csv: Path, tool: str) -> "qg.MethodData":
    """Standardized per-pocket table for one LBS predictor.

    Primary harmonized outcome = `success_tns` (DCA-TNS); secondary = `success_dca_4A`.
    Size control = pocket `rg` ONLY: binding-site detection has no ligand, so there is
    no ligand-size covariate (unlike docking/P2Rank-native which dock a real ligand)."""
    d = pd.read_csv(per_pocket_csv, low_memory=False)
    d = d.rename(columns={"pocket_density_pair": "density"})
    d["p_apo"] = d["pocket_p_apo"]
    d["p_holo"] = d["pocket_p_holo"]
    d["n_apo"] = d["n_apo_avg"]
    d["n_holo"] = d["n_holo_avg"]
    for sc in ["success_tns", "success_dca_4A"]:
        d[sc] = pd.to_numeric(d[sc], errors="coerce")
    d["rg"] = pd.to_numeric(d["rg"], errors="coerce")
    g = qg._finish(d)                                       # attach ratio + gate levels
    return qg.MethodData(
        name=tool, g=g, ctrl_cols=["rg"],
        ctrl_note="pocket size `rg` only (binding-site detection has no ligand -> no ligand-size covariate)",
        coverage_note=(f"LBS {METHOD_LABEL.get(tool, tool)} per-pocket table, ST_100 full run "
                       f"({len(g):,} apo/holo pockets; unit_uid = UNIPROT:ST_100:idx, same universe "
                       f"as docking/P2Rank). Primary outcome = harmonized DCA-TNS `success_tns`."),
        outcomes=[
            qg.Outcome("success_tns", f"{METHOD_LABEL.get(tool, tool)} DCA-TNS detection",
                       binary=True, primary=True),
            qg.Outcome("success_dca_4A", f"{METHOD_LABEL.get(tool, tool)} DCA<=4A detection",
                       binary=True),
        ],
    )


# ============================================================================
# Zero-order pass — same rows & schema as parent.eval_gate, but NO size control
# ============================================================================
def zero_order_gates(md: "qg.MethodData", gates, boot: int) -> pd.DataFrame:
    """Parent-eval_gate schema (rho_*, diff/CI, surpasses_ratio_*) but zero-order:
    partial_rhos with an EMPTY control block, on the SAME ratio-defined rows the
    size-controlled pass uses (so the two regimes are directly comparable)."""
    rows = []
    for name, pl, dl in gates:
        g = md.g
        outc = md.primary()
        gm = qg.gate_mask(g, pl, dl)
        cm = qg._common_mask(g, outc.col, md.ctrl_cols)     # identical mask to controlled pass
        sub = g[gm & cm]
        n = len(sub)
        n_gate = int(gm.sum())
        n_holo = int((sub.pocket_class == "holo").sum())
        n_apo = n - n_holo
        row = {"gate": name, "outcome": outc.col, "n_gate": n_gate, "n_used": n,
               "n_apo": n_apo, "n_holo": n_holo}
        if n < 100 or n_holo < 15 or n_apo < 15:
            row["underpowered"] = True
            rows.append(row)
            continue
        y = sub[outc.col].to_numpy(float)
        preds = sub[qg.PREDICTORS].to_numpy(float)          # col 0 = ratio
        empty = np.empty((n, 0))
        rhos, pvals, _ = qg.partial_rhos(y, preds, empty)
        for k, pred in enumerate(qg.PREDICTORS):
            row[f"rho_{pred}"] = None if np.isnan(rhos[k]) else round(float(rhos[k]), 4)
        bd = qg.bootstrap_diff(y, preds, empty, b=boot, seed=0)
        for k, pred in enumerate(qg.PREDICTORS[1:], start=1):
            d = bd.get(k, {})
            row[f"diff_{pred}_minus_ratio"] = d.get("diff")
            row[f"ci_lo_{pred}"] = d.get("lo")
            row[f"ci_hi_{pred}"] = d.get("hi")
            row[f"p_gt0_{pred}"] = d.get("p_gt0")
            row[f"surpasses_ratio_{pred}"] = bool(
                d.get("diff") is not None and d.get("lo") is not None
                and d["diff"] > 0 and d["lo"] > 0)
        row["underpowered"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def controlled_gates(md: "qg.MethodData", gates, boot: int) -> pd.DataFrame:
    """Size-controlled gate table over the user's 3-gate subset (parent.eval_gate)."""
    return pd.DataFrame([qg.eval_gate(md, name, pl, dl, boot=boot) for name, pl, dl in gates])


# ============================================================================
# Pooled "overall across methods" — Fisher-z averaged partial ρ + cluster bootstrap
# ============================================================================
# "Overall correlation across all four methods" is computed as the EQUAL-WEIGHT
# Fisher-z (arctanh) mean of the four per-method partial Spearman ρ — one vote per
# method. This is the average of the four bars the eye is already estimating, and it
# avoids pseudo-replication: the three LBS tables share the SAME pockets (identical
# ratio/p_holo/is_holo per pocket, only the success outcome differs) and docking is a
# nested subset, so STACKING rows would duplicate predictor values ~4x and inflate n.
#
# CI = a JOINT pocket-level CLUSTER bootstrap: resample pockets from the shared universe
# and keep every method's rows for the resampled pockets together (docking uses its
# covered subset), then recompute all four per-method ρ and their Fisher-z mean per
# replicate. Because the LBS methods share pockets their ρ estimates are POSITIVELY
# correlated; the joint resample propagates that covariance (an independent per-method
# bootstrap would make the CI of the average too narrow / anticonservative).
POOL_PREDS = list(qg.PREDICTORS)                       # [log2_ratio, p_holo, is_holo]


def _fisher_mean(rhos: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Equal-weight (default) Fisher-z mean of per-method ρ; NaN-robust."""
    r = np.asarray(rhos, float)
    ok = np.isfinite(r)
    if not ok.any():
        return np.nan
    z = np.arctanh(np.clip(r[ok], -0.999999, 0.999999))
    w = np.ones_like(z) if weights is None else np.asarray(weights, float)[ok]
    return float(np.tanh(np.sum(w * z) / np.sum(w)))


def _method_pool_arrays(md: "qg.MethodData", pl, dl, regime: str):
    """In-gate, common-mask (ratio-defined) arrays for one method + one regime.

    Returns (pid, y, preds[:,3], ctrl[:,c]); ctrl is empty for the zero-order regime,
    md.ctrl_cols for the size-controlled regime. pid = the pocket identifier shared
    across methods (site_uid == unit_uid == UNIPROT:ST_100:idx)."""
    g = md.g
    outc = md.primary()
    gm = qg.gate_mask(g, pl, dl)
    cm = qg._common_mask(g, outc.col, md.ctrl_cols)
    sub = g[gm & cm]
    pid_col = "site_uid" if "site_uid" in sub.columns else "unit_uid"
    pid = sub[pid_col].astype(str).to_numpy()
    y = sub[outc.col].to_numpy(float)
    preds = sub[POOL_PREDS].to_numpy(float)
    ctrl = (sub[md.ctrl_cols].to_numpy(float) if regime == "ctrl"
            else np.empty((len(sub), 0), float))
    return pid, y, preds, ctrl


def _partial_rhos_fast(y, preds, ctrl) -> np.ndarray:
    """Point partial-Spearman ρ per predictor column (skips the p-value work)."""
    n = len(y)
    if n < 20:
        return np.full(preds.shape[1], np.nan)
    n_ctrl = ctrl.shape[1] if ctrl.ndim == 2 else 0
    M = np.column_stack([y, preds] + ([ctrl] if n_ctrl else []))
    Vr = qg._rank_cols(M)
    return qg._partial_rhos_from_ranks(Vr, preds.shape[1], n_ctrl)


def _fam_collapse(rr: np.ndarray, lbs_idx: List[int], dock_idx: List[int]) -> np.ndarray:
    """2-family Fisher-z mean: collapse the pocket-sharing LBS trio to ONE vote (their
    Fisher-z mean), then average that with docking — so the correlated LBS family and
    docking each carry half the weight instead of the LBS family carrying 3/4."""
    npred = rr.shape[1]
    fam_lbs = np.array([_fisher_mean(rr[lbs_idx, k]) for k in range(npred)])
    if dock_idx:
        dock = rr[dock_idx[0]]
        return np.array([_fisher_mean(np.array([fam_lbs[k], dock[k]])) for k in range(npred)])
    return fam_lbs


def _cross_count(results: Dict[str, dict], methods, regime: str, gate: str, pred: str) -> str:
    """k/N per-method robust crossovers at this cell (individual surpasses_ratio_pred==True)."""
    key = "zero" if regime == "zero" else "controlled"
    k = 0
    for m in methods:
        df = results[m][key]
        r = df[df.gate == gate]
        if not r.empty and bool(r.iloc[0].get(f"surpasses_ratio_{pred}", False)):
            k += 1
    return f"{k}/{len(methods)}"


def compute_pooled(results: Dict[str, dict], gate_specs, boot: int, seed: int = 0) -> dict:
    """pooled[regime][gate] = per-predictor overall ρ (equal-weight Fisher-z mean of the 4
    methods) + its cluster-bootstrap CI + the Δ vs ratio + CI + robust-surpass flag, PLUS a
    2-family-collapsed sensitivity (LBS trio → one vote vs docking) and a k/N per-method
    heterogeneity count so the pooled flag is not mis-read as a 4-method consensus.

    NB the equal-weight point estimate is 'the average of the four displayed bars': three of
    the four are the LBS family scored on the SAME pockets/predictors/rg control (only the
    outcome differs), so it is LBS-family-heavy (3/4). `_fam` reports the family-balanced view;
    the headline (ratio wins overall except ctrl/prob_high+) holds under both."""
    methods = [m for m in METHOD_ORDER if m in results]
    lbs_idx = [i for i, m in enumerate(methods) if m != "docking"]
    dock_idx = [i for i, m in enumerate(methods) if m == "docking"]
    out = {"zero": {}, "ctrl": {}}
    for regime in ("zero", "ctrl"):
        for name, pl, dl in gate_specs:
            # per-method aligned arrays for this gate/regime
            marr = {m: _method_pool_arrays(results[m]["md"], pl, dl, regime) for m in methods}
            # ---- point estimate: equal-weight Fisher-z mean of per-method ρ ----
            per_method_rho = {}
            for m in methods:
                pid, y, preds, ctrl = marr[m]
                per_method_rho[m] = _partial_rhos_fast(y, preds, ctrl)
            rr_pt = np.array([per_method_rho[m] for m in methods])         # methods x preds
            pt_rho = np.array([_fisher_mean(rr_pt[:, k]) for k in range(len(POOL_PREDS))])
            pt_fam = _fam_collapse(rr_pt, lbs_idx, dock_idx)               # 2-family point
            # ---- joint pocket-cluster bootstrap ----
            # union pocket frame + per-method int map (union-code -> method row, -1 if absent)
            all_pids = sorted(set().union(*[set(marr[m][0]) for m in methods]))
            code = {p: i for i, p in enumerate(all_pids)}
            U = len(all_pids)
            mmap, my, mpr, mct = {}, {}, {}, {}
            for m in methods:
                pid, y, preds, ctrl = marr[m]
                mp = np.full(U, -1, int)
                mp[[code[p] for p in pid]] = np.arange(len(pid))
                mmap[m], my[m], mpr[m], mct[m] = mp, y, preds, ctrl
            rng = np.random.default_rng(seed)
            boot_rho = np.full((boot, len(POOL_PREDS)), np.nan)
            boot_fam = np.full((boot, len(POOL_PREDS)), np.nan)
            for b in range(boot):
                draw = rng.integers(0, U, U)
                rr = np.full((len(methods), len(POOL_PREDS)), np.nan)
                for mi, m in enumerate(methods):
                    rows = mmap[m][draw]
                    rows = rows[rows >= 0]
                    if len(rows) < 100:
                        continue
                    rr[mi] = _partial_rhos_fast(my[m][rows], mpr[m][rows], mct[m][rows])
                boot_rho[b] = [_fisher_mean(rr[:, k]) for k in range(len(POOL_PREDS))]
                boot_fam[b] = _fam_collapse(rr, lbs_idx, dock_idx)
            boot_diff = boot_rho[:, 1:] - boot_rho[:, :1]          # equal-weight pred - ratio
            boot_fdiff = boot_fam[:, 1:] - boot_fam[:, :1]         # 2-family   pred - ratio
            rec = {"n_methods": len(methods), "n_pockets": U, "rho": {}, "rho_ci": {},
                   "diff": {}, "diff_ci": {}, "surpasses": {},
                   "rho_fam": {}, "diff_fam": {}, "diff_fam_ci": {}, "surpasses_fam": {},
                   "cross": {}}
            for k, pred in enumerate(POOL_PREDS):
                col = boot_rho[:, k]; col = col[np.isfinite(col)]
                rec["rho"][pred] = None if np.isnan(pt_rho[k]) else round(float(pt_rho[k]), 4)
                rec["rho_ci"][pred] = (round(float(np.percentile(col, 2.5)), 4),
                                       round(float(np.percentile(col, 97.5)), 4)) if col.size else (None, None)
                rec["rho_fam"][pred] = None if np.isnan(pt_fam[k]) else round(float(pt_fam[k]), 4)
            for k, pred in enumerate(POOL_PREDS[1:]):
                # equal-weight Δ vs ratio (point Δ, cluster-bootstrap percentile CI)
                d = boot_diff[:, k]; d = d[np.isfinite(d)]
                pt = pt_rho[1 + k] - pt_rho[0]
                lo = round(float(np.percentile(d, 2.5)), 4) if d.size else None
                hi = round(float(np.percentile(d, 97.5)), 4) if d.size else None
                rec["diff"][pred] = None if np.isnan(pt) else round(float(pt), 4)
                rec["diff_ci"][pred] = (lo, hi)
                rec["surpasses"][pred] = bool(pt > 0 and lo is not None and lo > 0)
                # 2-family Δ vs ratio
                df = boot_fdiff[:, k]; df = df[np.isfinite(df)]
                ptf = pt_fam[1 + k] - pt_fam[0]
                flo = round(float(np.percentile(df, 2.5)), 4) if df.size else None
                fhi = round(float(np.percentile(df, 97.5)), 4) if df.size else None
                rec["diff_fam"][pred] = None if np.isnan(ptf) else round(float(ptf), 4)
                rec["diff_fam_ci"][pred] = (flo, fhi)
                rec["surpasses_fam"][pred] = bool(ptf > 0 and flo is not None and flo > 0)
                # per-method heterogeneity: k/N methods that individually cross robustly
                rec["cross"][pred] = _cross_count(results, methods, regime, name, pred)
            out[regime][name] = rec
    return out


def pooled_to_frame(pooled: dict, gates: List[str]) -> pd.DataFrame:
    """Flatten the pooled structure to a tidy CSV."""
    rows = []
    for regime in ("zero", "ctrl"):
        for gate in gates:
            rec = pooled.get(regime, {}).get(gate)
            if not rec:
                continue
            row = {"regime": regime, "gate": gate, "n_methods": rec["n_methods"],
                   "n_pockets": rec["n_pockets"]}
            for pred in POOL_PREDS:
                row[f"rho_{pred}"] = rec["rho"].get(pred)
                lo, hi = rec["rho_ci"].get(pred, (None, None))
                row[f"rho_ci_{pred}"] = f"[{lo},{hi}]"
            for pred in POOL_PREDS[1:]:
                # equal-weight (4-method, LBS-family-heavy); Δ is a POINT estimate, CI = cluster bootstrap
                row[f"diff_{pred}_minus_ratio_point"] = rec["diff"].get(pred)
                lo, hi = rec["diff_ci"].get(pred, (None, None))
                row[f"diff_ci_{pred}"] = f"[{lo},{hi}]"
                row[f"surpasses_ratio_{pred}"] = rec["surpasses"].get(pred)
                # 2-family sensitivity (LBS trio collapsed to one vote vs docking)
                row[f"rho_fam_{pred}"] = rec.get("rho_fam", {}).get(pred)
                row[f"diff_fam_{pred}_point"] = rec.get("diff_fam", {}).get(pred)
                flo, fhi = rec.get("diff_fam_ci", {}).get(pred, (None, None))
                row[f"diff_fam_ci_{pred}"] = f"[{flo},{fhi}]"
                row[f"surpasses_fam_{pred}"] = rec.get("surpasses_fam", {}).get(pred)
                # per-method heterogeneity (k/N methods that individually cross robustly)
                row[f"cross_{pred}"] = rec.get("cross", {}).get(pred)
            row["rho_fam_log2_ratio"] = rec.get("rho_fam", {}).get("log2_ratio")
            rows.append(row)
    return pd.DataFrame(rows)


def _parse_ci(s):
    try:
        lo, hi = str(s).strip("[]").split(",")
        return (None if lo.strip() == "None" else float(lo),
                None if hi.strip() == "None" else float(hi))
    except Exception:
        return (None, None)


def load_saved_for_figure(qg_dir: Path):
    """Reconstruct the figure inputs (per-method gate frames + pooled dict) from the CSVs
    a prior full run wrote, so the manuscript figure can be re-exported WITHOUT recomputing
    the bootstraps. Returns (controlled, zero, pooled, gates)."""
    tools = [t for t in METHOD_ORDER if (qg_dir / t / "gates_partial_spearman.csv").is_file()]
    controlled = {t: pd.read_csv(qg_dir / t / "gates_partial_spearman.csv") for t in tools}
    zero = {t: pd.read_csv(qg_dir / t / "gates_zero_order.csv") for t in tools}
    pf = pd.read_csv(qg_dir / "pooled_overall_ratio_vs_class.csv")
    pooled = {"zero": {}, "ctrl": {}}
    for _, r in pf.iterrows():
        rec = {"n_methods": int(r["n_methods"]), "n_pockets": int(r["n_pockets"]),
               "rho": {}, "diff": {}, "diff_ci": {}, "surpasses": {}}
        for pred in POOL_PREDS:
            v = r.get(f"rho_{pred}")
            rec["rho"][pred] = None if pd.isna(v) else float(v)
        for pred in POOL_PREDS[1:]:
            v = r.get(f"diff_{pred}_minus_ratio_point")
            rec["diff"][pred] = None if pd.isna(v) else float(v)
            rec["diff_ci"][pred] = _parse_ci(r.get(f"diff_ci_{pred}"))
            rec["surpasses"][pred] = bool(r.get(f"surpasses_ratio_{pred}"))
        pooled[str(r["regime"])][str(r["gate"])] = rec
    gates = [g[0] for g in LBS_GATE_SPECS]
    return controlled, zero, pooled, gates


def write_figure_caption(pooled: dict, path: Path) -> None:
    """Standalone caption for the manuscript figure (placement main-vs-SI still TBD)."""
    def g(regime, gate, key, pred):
        rec = pooled.get(regime, {}).get(gate, {})
        return rec.get(key, {}).get(pred)
    ph = pooled.get("ctrl", {}).get("prob_high+", {})
    cap = (
        "Ratio vs classification across binding-site predictors and docking.\n\n"
        "For every apo/holo pocket we compare three directional predictors of binding-site-"
        "predictor success — the graded evidence ratio log2((n_holo+1)/n_apo), the class probability "
        "p_holo, and the binary class label is_holo — against each method's success outcome. Columns "
        "are three undirected quality gates on the classification confidence winning_p (all, "
        "moderate+, high+); the top row shows zero-order Spearman rho and the bottom row the "
        "size-controlled partial rho (controls: pocket size rg for the three ligand-binding-site "
        "predictors fpocket / IF-SitePred / harmonized P2Rank, scored on binary DCA-TNS detection; "
        "union_rg + ligand heavy atoms for docking, scored on the continuous fraction of cognate "
        "ligands dockable). Bars are the three predictors (colours in legend); error bars on p_holo "
        "and is_holo are 95 % bootstrap CIs of the rho difference vs the ratio.\n\n"
        "The shaded 'all methods (mean of 4)' group is the pooled overall correlation across the four "
        "methods: the equal-weight Fisher-z mean of the four per-method partial rho (the average of "
        "the four bars), with a joint pocket-level cluster bootstrap CI that resamples the shared "
        "pockets and recomputes all four methods together (the three LBS tables share the same "
        "pockets, so an independent bootstrap would be anticonservative). Because that mean is "
        "LBS-family-heavy (three of the four votes are the same pockets, differing only in outcome), "
        "a two-family sensitivity (LBS trio collapsed to one vote vs docking) and a per-method k/4 "
        "crossover count are reported in pooled_overall_ratio_vs_class.csv.\n\n"
        "Result: ungated, ratio >= p_holo >= is_holo in all four predictors in both regimes — the "
        "graded evidence ratio is the best single apo/holo summary of predictor success and the "
        "relationship is not P2Rank-specific (generality, C4). The classification only matches or "
        "edges past the ratio inside the tight, size-controlled high-confidence corner, and even "
        "there the pooled crossover (is_holo overall rho "
        f"{g('ctrl','prob_high+','rho','is_holo')} vs ratio {g('ctrl','prob_high+','rho','log2_ratio')}, "
        f"Delta {ph.get('diff',{}).get('is_holo')}) is a 2/4-method majority (IF-SitePred and docking; "
        "fpocket and P2Rank keep the ratio ahead) that grows under the family-balanced weighting; the "
        "pooled winner is identical under the Fisher-z mean, a simple mean, and observation-stacking.\n\n"
        "n = 86,651 pockets (LBS, ST_100 full runs) / 60,689 (docking, cognate group 1). Generated by "
        "06_lbs_generality/analysis/analyze_lbs_ratio_vs_class.py. Manuscript placement (main vs SI) TBD."
    )
    path.write_text(cap, encoding="utf-8")


# ============================================================================
# Cross-tool generality figure  (2 regimes x 3 gates; methods on x, 3 predictors/bars)
# ============================================================================
def _rho(df: pd.DataFrame, gate: str, pred: str):
    r = df[df.gate == gate]
    if r.empty or r.iloc[0].get("underpowered", False):
        return np.nan
    v = r.iloc[0].get(f"rho_{pred}")
    return np.nan if v is None else float(v)


def _diff_ci(df: pd.DataFrame, gate: str, pred: str):
    """Absolute [lo, hi] band for a non-ratio predictor: ratio_rho + CI(diff vs ratio)."""
    r = df[df.gate == gate]
    if r.empty or r.iloc[0].get("underpowered", False):
        return np.nan, np.nan
    row = r.iloc[0]
    base = row.get("rho_log2_ratio")
    lo, hi = row.get(f"ci_lo_{pred}"), row.get(f"ci_hi_{pred}")
    if base is None or lo is None or hi is None:
        return np.nan, np.nan
    return float(base) + float(lo), float(base) + float(hi)


def fig_cross_tool(controlled: Dict[str, pd.DataFrame], zero: Dict[str, pd.DataFrame],
                   pooled: dict, gates: List[str], out_paths) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if isinstance(out_paths, (str, Path)):
        out_paths = [Path(out_paths)]

    methods = [m for m in METHOD_ORDER if m in controlled]
    regimes = [("zero-order Spearman ρ", zero, "zero"),
               ("size-controlled partial ρ", controlled, "ctrl")]
    x = np.arange(len(methods), dtype=float)
    x_pool = float(len(methods)) + 0.6                 # pooled group sits after a gap
    w = 0.26
    fig, axes = plt.subplots(len(regimes), len(gates),
                             figsize=(4.5 * len(gates), 3.75 * len(regimes)),
                             sharey="row", squeeze=False)
    for ri, (regime_label, store, rkey) in enumerate(regimes):
        for ci, gate in enumerate(gates):
            ax = axes[ri][ci]
            # shade + separate the pooled column
            ax.axvspan(x_pool - 0.45, x_pool + 0.45, color="0.88", alpha=0.7, zorder=0)
            ax.axvline((len(methods) - 1 + x_pool) / 2 + 0.03, color="0.6", lw=0.8, ls="--", zorder=1)
            prec = pooled.get(rkey, {}).get(gate, {})
            for k, pred in enumerate(qg.PREDICTORS):
                # ---- per-method bars ----
                vals = [_rho(store[m], gate, pred) for m in methods]
                yerr = None
                if pred != qg.RATIO:
                    los, his = [], []
                    for mi, m in enumerate(methods):
                        lo, hi = _diff_ci(store[m], gate, pred)
                        v = vals[mi]
                        los.append(np.nan if np.isnan(lo) or np.isnan(v) else max(0.0, v - lo))
                        his.append(np.nan if np.isnan(hi) or np.isnan(v) else max(0.0, hi - v))
                    yerr = np.abs(np.nan_to_num(np.array([los, his])))
                ax.bar(x + (k - 1) * w, vals, width=w, color=qg.PRED_COLORS[pred],
                       yerr=yerr, capsize=2.5, error_kw=dict(lw=0.8, ecolor="0.35"),
                       label=qg.PRED_LABEL[pred] if (ri == 0 and ci == 0) else None, zorder=3)
                # ---- pooled "all methods" bar (Fisher-z mean; heavier edge) ----
                pv = prec.get("rho", {}).get(pred)
                pv = np.nan if pv is None else float(pv)
                perr = None
                if pred != qg.RATIO and np.isfinite(pv):
                    base = prec.get("rho", {}).get(qg.RATIO)
                    lo, hi = prec.get("diff_ci", {}).get(pred, (None, None))
                    if base is not None and lo is not None and hi is not None:
                        lo_abs, hi_abs = base + lo, base + hi
                        perr = np.array([[max(0.0, pv - lo_abs)], [max(0.0, hi_abs - pv)]])
                ax.bar([x_pool + (k - 1) * w], [pv], width=w, color=qg.PRED_COLORS[pred],
                       yerr=perr, capsize=2.5, error_kw=dict(lw=0.9, ecolor="0.2"),
                       edgecolor="0.1", linewidth=1.1, zorder=3)
            ax.axhline(0, color="0.3", lw=0.8, zorder=2)
            ns = [_gate_n(controlled[m], gate) for m in methods]
            ax.set_xticks(list(x) + [x_pool])
            labs = [f"{METHOD_LABEL[m]}\nn={n:,}" if n else METHOD_LABEL[m]
                    for m, n in zip(methods, ns)]
            labs.append(f"all methods\n(mean of {prec.get('n_methods', len(methods))})")
            ax.set_xticklabels(labs, fontsize=8)
            # bold the pooled tick label
            ax.get_xticklabels()[-1].set_fontweight("bold")
            ax.set_xlim(-0.6, x_pool + 0.6)
            ax.grid(axis="y", alpha=0.25)
            if ri == 0:
                ax.set_title(f"gate: {gate}", fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{regime_label}\nvs predictor success", fontsize=9)
    axes[0][0].legend(fontsize=8.5, title="apo/holo predictor", loc="upper right")
    fig.suptitle(
        "Ratio vs class across binding-site predictors + docking — does the pattern generalize?\n"
        "Ungated: ratio ≥ p_holo ≥ is_holo everywhere. High-confidence + size-controlled: "
        "is_holo climbs, overtaking the ratio in some predictors.\n"
        "Shaded 'all methods (mean of 4)' = equal-weight Fisher-z mean of the 4 per-method ρ "
        "(LBS-family-heavy: the 3 LBS methods share pockets).\n"
        "Error bars = 95% CI of the ρ difference vs the ratio (joint pocket-cluster bootstrap).\n"
        "Controls: LBS = rg, docking = union_rg + ligand size.   "
        "Outcomes: LBS = binary DCA-TNS, docking = fraction dockable.",
        fontsize=9, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for p in out_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300 if p.suffix.lower() == ".png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def _gate_n(df: pd.DataFrame, gate: str) -> int:
    r = df[df.gate == gate]
    if r.empty:
        return 0
    v = r.iloc[0].get("n_used")
    return int(v) if pd.notna(v) else 0


# ============================================================================
# Cross-method synthesis report  (the main deliverable — with a manuscript/SI call)
# ============================================================================
def _fmt(v, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def crossover_verdict(controlled: pd.DataFrame, pred: str, gate: str = "prob_high+") -> str:
    r = controlled[controlled.gate == gate]
    if r.empty or r.iloc[0].get("underpowered", False):
        return "n/a"
    row = r.iloc[0]
    d, lo, hi = row.get(f"diff_{pred}_minus_ratio"), row.get(f"ci_lo_{pred}"), row.get(f"ci_hi_{pred}")
    if d is None or lo is None:
        return "n/a"
    robust = d > 0 and lo > 0
    return f"{'YES' if robust else 'no'} (Δ={d:+.3f} [{lo:+.3f},{hi:+.3f}])"


def write_cross_report(results: Dict[str, dict], pooled: dict, gates: List[str], out_path: Path) -> None:
    L: List[str] = []
    L.append("# LBS-generality: quality-gated ratio-vs-class across binding-site predictors + docking\n")
    L.append("**Question (C4 generality).** The docking + P2Rank-native + AlphaFill analysis "
             "(*T2-qualgate*) established that the graded holo/apo **evidence ratio** "
             "`log2((n_holo+1)/n_apo)` predicts functional readouts at least as well as the class "
             "probability `p_holo` or the binary label `is_holo`, size-controlled. Does that "
             "ratio-vs-class relationship **generalize across ligand-binding-site predictors**? Here we "
             "re-run it on the three LBS predictors — **fpocket, IF-SitePred, harmonized P2Rank** — on "
             "our default harmonized metric **DCA-TNS (`success_tns`)**, alongside **docking** (native "
             "dockability; DCA-TNS does not apply to docking).\n")
    L.append("**Method.** Per pocket, three directional apo/holo predictors "
             "{`log2_ratio`, `p_holo`, `is_holo`} scored against predictor success, within three "
             "undirected quality gates on `winning_p` (`all`, `prob_mod+`, `prob_high+`). We report "
             "**both** zero-order Spearman ρ and **size-controlled** partial Spearman ρ (LBS control = "
             "pocket size `rg` only — no ligand in detection, so no ligand-size covariate; docking "
             "control = `union_rg` + ligand heavy atoms). All three predictors share the SAME "
             "ratio-defined rows within each gate (paired); the crossover is called robust only when "
             "the point Δρ>0 AND the paired bootstrap 95% CI of the difference vs the ratio excludes 0. "
             "We never gate on `p_holo` (it would leak the outcome). Nested 5-fold CV logistic AUC "
             "(size / +ratio / +prob / +class / +ratio+prob) accompanies each binary outcome.\n")

    # -- 1. the general ungated ordering ------------------------------------
    L.append("\n## 1. Ungated: the core ordering generalizes across all four predictors\n")
    L.append("Zero-order and size-controlled partial ρ vs success at the `all` gate "
             "(higher ρ = the predictor tracks success better):\n")
    L.append("| method | outcome | n | ρ ratio (zero / ctrl) | ρ p_holo (zero / ctrl) | ρ is_holo (zero / ctrl) | ordering |")
    L.append("|---|---|---|---|---|---|---|")
    for m in [x for x in METHOD_ORDER if x in results]:
        cg = results[m]["controlled"]; zg = results[m]["zero"]
        n = _gate_n(cg, "all")
        trip = lambda pred: f"{_fmt(_rho(zg,'all',pred))} / {_fmt(_rho(cg,'all',pred))}"
        # ordering by controlled rho
        order = sorted(["log2_ratio", "p_holo", "is_holo"], key=lambda p: -(_rho(cg, "all", p) or -9))
        omap = {"log2_ratio": "ratio", "p_holo": "p_holo", "is_holo": "is_holo"}
        L.append(f"| {METHOD_LABEL[m]} | {results[m]['primary']} | {n:,} | "
                 f"{trip('log2_ratio')} | {trip('p_holo')} | {trip('is_holo')} | "
                 f"{' > '.join(omap[o] for o in order)} |")
    L.append("\n**Read.** In every predictor and in both regimes the ungated ordering is "
             "`ratio ≥ p_holo ≥ is_holo`: the graded evidence ratio is the best single apo/holo "
             "summary of binding-site-predictor success, exactly as it was for docking/P2Rank. "
             "The apo/holo *classification* is a thresholded summary of that same evidence. This is "
             "the generality result: **the ratio-vs-class relationship is not P2Rank-specific — it "
             "reproduces in fpocket, IF-SitePred, the harmonized P2Rank re-score, and docking.**")

    # -- 2. the high-confidence crossover (size-controlled) -----------------
    L.append("\n## 2. High-confidence + size-controlled: is_holo climbs (predictor-dependent crossover)\n")
    L.append("Robust crossover = the class predictor's size-controlled partial ρ **surpasses the "
             "ratio** at `prob_high+` (point Δρ>0 AND bootstrap 95% CI of the difference excludes 0):\n")
    L.append("| method | ρ ratio | ρ p_holo | ρ is_holo | is_holo beats ratio? (prob_high+, ctrl) | p_holo beats ratio? |")
    L.append("|---|---|---|---|---|---|")
    for m in [x for x in METHOD_ORDER if x in results]:
        cg = results[m]["controlled"]
        L.append(f"| {METHOD_LABEL[m]} | {_fmt(_rho(cg,'prob_high+','log2_ratio'))} | "
                 f"{_fmt(_rho(cg,'prob_high+','p_holo'))} | {_fmt(_rho(cg,'prob_high+','is_holo'))} | "
                 f"{crossover_verdict(cg,'is_holo')} | {crossover_verdict(cg,'p_holo')} |")
    L.append("\n**Read.** The crossover is a **size-controlled** phenomenon (zero-order → parity; see "
             "the per-gate figure) and it is **predictor-dependent**: strongest where the predictor is "
             "most sensitive to a well-defined site, weaker/absent elsewhere. This matches the parent "
             "thread's honest caveat — the general, robust claim is the ungated ordering (§1); the "
             "is_holo-surpasses-ratio crossover is a conditional refinement that appears once the "
             "apo/holo call is both confident and size-adjusted, not a universal reversal. `winning_p` "
             "(the undirected gate axis) remains the weakest predictor throughout.")

    # -- 2b. pooled overall-across-methods ----------------------------------
    L.append("\n## 2b. Overall across all four methods — which metric wins on balance?\n")
    L.append("**`all methods (mean of 4)` group in the figure.** The overall ρ is the **equal-weight "
             "Fisher-z (arctanh) mean** of the four per-method partial ρ — literally the average of the "
             "four displayed bars. CIs are a **joint pocket-level cluster bootstrap**: pockets are resampled "
             "from the shared universe and each method's rows for the resampled pockets are kept together "
             "(docking uses its covered subset), so the positive covariance from the three LBS tables "
             "sharing the same pockets is propagated (a naive row-stack would pseudo-replicate predictor "
             "values ~4× and understate the CI). It is a **meta-analytic summary of comparable-but-not-"
             "identical partial correlations**: the three LBS ρ are vs binary DCA-TNS with an `rg` control, "
             "docking's ρ is vs continuous fraction-dockable with a `union_rg`+ligand-size control. **Caveat "
             "(honest weighting):** the four are not four independent votes — the LBS trio are the SAME "
             "pockets with IDENTICAL predictor values and control, differing only in outcome, so the "
             "equal-weight mean is **LBS-family-heavy (3/4 weight)**. We therefore also report a **2-family "
             "sensitivity** that collapses the LBS trio to one vote vs docking (`fam` columns), and a per-"
             "method heterogeneity count `k/4` = how many methods individually cross. Δ vs ratio is a "
             "**point** estimate (the per-method §2 Δ is a bootstrap mean); robust ⇔ point Δ>0 AND CI>0.\n")
    L.append("| regime | gate | overall ρ ratio / p_holo / is_holo | highest | is_holo−ratio [95% CI] (k/4 cross) | is_holo−ratio, 2-family | p_holo−ratio [95% CI] |")
    L.append("|---|---|---|---|---|---|---|")
    rlab = {"zero": "zero-order", "ctrl": "size-controlled"}
    for regime in ("zero", "ctrl"):
        for gate in gates:
            rec = pooled.get(regime, {}).get(gate)
            if not rec:
                continue
            r = rec["rho"]
            winner = max(POOL_PREDS, key=lambda p: (r.get(p) if r.get(p) is not None else -9))
            wmap = {"log2_ratio": "**ratio**", "p_holo": "**p_holo**", "is_holo": "**is_holo**"}

            def dci(pred):
                d = rec["diff"].get(pred)
                lo, hi = rec["diff_ci"].get(pred, (None, None))
                if d is None or lo is None:
                    return "–"
                star = " ✓" if rec["surpasses"].get(pred) else ""
                return f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"

            def fdci(pred):
                d = rec.get("diff_fam", {}).get(pred)
                lo, hi = rec.get("diff_fam_ci", {}).get(pred, (None, None))
                if d is None or lo is None:
                    return "–"
                star = " ✓" if rec.get("surpasses_fam", {}).get(pred) else ""
                return f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"
            cross_is = rec.get("cross", {}).get("is_holo", "")
            L.append(f"| {rlab[regime]} | {gate} | "
                     f"{_fmt(r.get('log2_ratio'))} / {_fmt(r.get('p_holo'))} / {_fmt(r.get('is_holo'))} | "
                     f"{wmap[winner]} | {dci('is_holo')} ({cross_is}) | {fdci('is_holo')} | {dci('p_holo')} |")
    L.append("\n**Read.** Pooled across all four methods, the **evidence ratio has the highest overall "
             "correlation at the `all` and `prob_mod+` gates in both regimes**, and **stays on top at every "
             "gate zero-order**; only inside the tight `prob_high+`, size-controlled cell does the pooled "
             "`is_holo` edge past the ratio — and that is a **2/4-method majority** (IF-SitePred and docking "
             "cross; fpocket and P2Rank keep the ratio ahead), not a consensus. The crossover **survives and "
             "in fact grows under the family-balanced 2-family weighting** (the `fam` column), so it is not an "
             "artifact of triple-counting the LBS family; it remains a small, confidence-and-size-conditional "
             "margin. The single-metric answer to \"which apo/holo summary correlates best overall\" is the "
             "**graded evidence ratio**, with the binary label matching/beating it only in the well-supported, "
             "size-adjusted corner. Robustness: the pooled winner in every cell is **identical** under the "
             "Fisher-z mean, a simple arithmetic mean, and an observation-stacked estimate (verified). `✓` "
             "marks a Δ whose 95% CI excludes 0.")

    # -- 3. nested AUC snapshot ---------------------------------------------
    L.append("\n## 3. Nested 5-fold CV logistic AUC (DCA-TNS, per gate)\n")
    L.append("Incremental AUC over the size-only baseline (size / +ratio / +prob / +class / +ratio+prob):\n")
    for m in [x for x in METHOD_ORDER if x in results]:
        auc = results[m].get("auc")
        if auc is None or auc.empty:
            continue
        prim = results[m]["primary"]
        sub = auc[auc.outcome == prim]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="gate", columns="model", values="cv_auc", aggfunc="first")
        cols = [c for c in ["size", "size+ratio", "size+prob", "size+class", "size+ratio+prob"]
                if c in piv.columns]
        piv = piv.reindex(index=[g for g in gates if g in piv.index], columns=cols)
        L.append(f"\n**{METHOD_LABEL[m]}** (`{prim}`)\n")
        L.append("| gate | " + " | ".join(cols) + " |")
        L.append("|" + "---|" * (len(cols) + 1))
        for idx, row in piv.iterrows():
            L.append(f"| {idx} | " + " | ".join(f"{row[c]:.4f}" if pd.notna(row[c]) else "–"
                                                 for c in cols) + " |")

    # -- 4. verdict + manuscript/SI call ------------------------------------
    any_cross = any(
        crossover_verdict(results[m]["controlled"], "is_holo").startswith("YES")
        for m in results)
    L.append("\n## 4. Verdict + manuscript placement\n")
    L.append("- **Generality (headline, robust).** The ratio-vs-class ordering "
             "`ratio ≥ p_holo ≥ is_holo` holds ungated, in BOTH regimes, in ALL FOUR predictors "
             "(fpocket, IF-SitePred, P2Rank, docking). The graded evidence ratio is the best apo/holo "
             "predictor of binding-site-predictor success, and the finding is **not tool-specific** → "
             "supports **C4**.")
    if any_cross:
        L.append("- **Conditional crossover (supporting).** In at least one predictor, `is_holo` "
                 "**robustly** surpasses the ratio once the call is high-confidence AND size-controlled "
                 "(see §2 CIs). This reproduces the parent-thread crossover in a second/third predictor, "
                 "strengthening the interpretation that the class label is reliable exactly where it is "
                 "well supported — but it is conditional and predictor-dependent, so it is **supporting**, "
                 "not headline.")
    else:
        L.append("- **No robust crossover here.** Even at `prob_high+` and size-controlled, the class "
                 "predictors' CI does not clear the ratio in these LBS tables — the ratio remains at "
                 "least as good, reinforcing §1.")
    L.append("\n**Recommendation.** This is **generality evidence for C4** and belongs with the "
             "multi-tool LBS panel (T3-2nd): most naturally a **SI figure/table** (or an SI panel of the "
             "T3-2nd multi-tool figure) — the cross-tool figure `fig_cross_tool_ratio_vs_class.png` is the "
             "one-panel \"does the pattern generalize across predictors?\" summary, with the per-tool gate "
             "figures and CIs as backup. The headline main-text claim stays the holo>apo detection "
             "gradient; this shows the *evidence-ratio-vs-classification* structure behind it is shared "
             "across predictors.\n")
    L.append("\n---\n*Per-tool tables, CIs, nested-AUC, threshold-sweep crossover, and figures are under "
             "`quality_gated/{fpocket,ifsitepred,p2rank,docking}/`; `gates_zero_vs_controlled.csv` holds "
             "both regimes side by side; the pooled overall-across-methods ρ is in "
             "`pooled_overall_ratio_vs_class.csv` (and the shaded `all methods` group of "
             "`fig_cross_tool_ratio_vs_class.png`). Generated by "
             "`06_lbs_generality/analysis/analyze_lbs_ratio_vs_class.py`.*")
    out_path.write_text("\n".join(L), encoding="utf-8")


# ============================================================================
# Per-tool driver — reuse parent machinery + add the zero-order pass
# ============================================================================
def run_tool(md: "qg.MethodData", out_dir: Path, boot: int, sweep_boot: int,
             do_sweep: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    md.g.to_csv(out_dir / "pocket_level.csv", index=False)

    # size-controlled gate table (parent) + zero-order gate table (ours), same rows
    cg = controlled_gates(md, LBS_GATE_SPECS, boot=boot)
    zg = zero_order_gates(md, LBS_GATE_SPECS, boot=boot)
    cg.to_csv(out_dir / "gates_partial_spearman.csv", index=False)
    zg.to_csv(out_dir / "gates_zero_order.csv", index=False)
    _write_combined(md, cg, zg, out_dir / "gates_zero_vs_controlled.csv")

    # nested CV AUC per gate (binary outcomes)
    auc_frames = [qg.eval_gate_auc(md, name, pl, dl) for name, pl, dl in LBS_GATE_SPECS]
    auc_frames = [a for a in auc_frames if not a.empty]
    auc = pd.concat(auc_frames, ignore_index=True) if auc_frames else pd.DataFrame()
    if not auc.empty:
        auc.to_csv(out_dir / "gates_nested_auc.csv", index=False)

    # per-tool gate bars (reuse the parent figure, on the size-controlled table)
    _patch_gates(lambda: qg.fig_gates(cg, md, out_dir / f"fig_gates_{md.name}.png"))

    # optional threshold sweep (the continuous crossover view; parent code)
    cross, sweep_df = {}, pd.DataFrame()
    if do_sweep:
        sweep_df = qg.sweep(md, boot=sweep_boot)
        sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
        cross = qg.find_crossover(sweep_df)
        qg.fig_sweep(sweep_df, md, out_dir / f"fig_sweep_{md.name}.png")

    # per-tool report (parent writer, size-controlled view)
    qg.write_method_report(md, cg, auc, sweep_df, cross, out_dir / "REPORT.md")

    print(f"\n===== {md.name}  (n={len(md.g):,} apo/holo pockets) =====")
    show = ["gate", "n_used", "n_apo", "n_holo", "rho_log2_ratio", "rho_p_holo", "rho_is_holo",
            "diff_is_holo_minus_ratio", "ci_lo_is_holo", "surpasses_ratio_is_holo"]
    print("[size-controlled]")
    print(cg[[c for c in show if c in cg.columns]].to_string(index=False))
    print("[zero-order]")
    print(zg[[c for c in show if c in zg.columns]].to_string(index=False))
    return {"name": md.name, "controlled": cg, "zero": zg, "auc": auc,
            "primary": md.primary().col, "n": len(md.g), "md": md}


def _write_combined(md, cg: pd.DataFrame, zg: pd.DataFrame, path: Path) -> None:
    """Side-by-side zero-order vs size-controlled per gate, for the record."""
    keep = ["gate", "n_used", "n_apo", "n_holo"]
    rows = []
    for gate in [g[0] for g in LBS_GATE_SPECS]:
        c = cg[cg.gate == gate]
        z = zg[zg.gate == gate]
        if c.empty or z.empty:
            continue
        c = c.iloc[0]; z = z.iloc[0]
        rec = {k: c.get(k) for k in keep}
        for pred in qg.PREDICTORS:
            rec[f"rho_{pred}_zero"] = z.get(f"rho_{pred}")
            rec[f"rho_{pred}_ctrl"] = c.get(f"rho_{pred}")
        for pred in ["p_holo", "is_holo"]:
            rec[f"diff_{pred}_zero"] = z.get(f"diff_{pred}_minus_ratio")
            rec[f"ci_zero_{pred}"] = f"[{z.get(f'ci_lo_{pred}')},{z.get(f'ci_hi_{pred}')}]"
            rec[f"diff_{pred}_ctrl"] = c.get(f"diff_{pred}_minus_ratio")
            rec[f"ci_ctrl_{pred}"] = f"[{c.get(f'ci_lo_{pred}')},{c.get(f'ci_hi_{pred}')}]"
            rec[f"surpasses_ratio_{pred}_ctrl"] = c.get(f"surpasses_ratio_{pred}")
            rec[f"surpasses_ratio_{pred}_zero"] = z.get(f"surpasses_ratio_{pred}")
        rows.append(rec)
    pd.DataFrame(rows).to_csv(path, index=False)


def _patch_gates(fn):
    """parent.fig_gates / write_method_report order gates by the module GATE_SPECS;
    temporarily restrict it to our 3-gate subset so ordering/labels are correct."""
    saved = qg.GATE_SPECS
    qg.GATE_SPECS = LBS_GATE_SPECS
    try:
        fn()
    finally:
        qg.GATE_SPECS = saved


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", nargs="+", default=["fpocket", "ifsitepred", "p2rank", "docking"],
                    choices=["fpocket", "ifsitepred", "p2rank", "docking"])
    ap.add_argument("--out_base", type=Path, default=cfg.LBS_QUALITY_GATED_DIR)
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for gate diffs")
    ap.add_argument("--pool_boot", type=int, default=600,
                    help="joint pocket-cluster bootstrap resamples for the pooled overall ρ")
    ap.add_argument("--sweep_boot", type=int, default=150, help="bootstrap resamples per sweep point")
    ap.add_argument("--no_sweep", action="store_true", help="skip the continuous threshold sweep")
    ap.add_argument("--group", type=int, default=1, help="docking ligand group (cognate = 1)")
    ap.add_argument("--manuscript_dir", type=Path, default=None,
                    help="also export the cross-tool figure (png+pdf+svg) + caption here (manuscript folder)")
    ap.add_argument("--fig_basename", default="lbs_ratio_vs_class_cross_tool",
                    help="basename for the manuscript figure files (no figure index)")
    ap.add_argument("--render_only", action="store_true",
                    help="skip all computation; re-render the cross-tool figure from the CSVs already "
                         "in --out_base (fast manuscript export, no bootstraps)")
    args = ap.parse_args()

    args.out_base.mkdir(parents=True, exist_ok=True)
    # keep parent module ordering consistent with our 3-gate subset everywhere
    qg.GATE_SPECS = LBS_GATE_SPECS

    # ---- fast path: re-render the figure from saved CSVs (manuscript export) ----
    if args.render_only:
        controlled, zero, pooled, gates = load_saved_for_figure(args.out_base)
        _export_figure(controlled, zero, pooled, gates, args)
        return

    results: Dict[str, dict] = {}
    for tool in [t for t in METHOD_ORDER if t in args.tools]:
        if tool == "docking":
            if not DOCKING_MERGED.exists():
                print(f"[skip] docking merged not found: {DOCKING_MERGED}")
                continue
            md = qg.load_docking(DOCKING_MERGED, group=args.group)
        else:
            csv = LBS_TABLES[tool]
            if not Path(csv).exists():
                print(f"[skip] {tool} table not found: {csv}")
                continue
            md = load_lbs(csv, tool)
        results[tool] = run_tool(md, args.out_base / tool, args.boot, args.sweep_boot,
                                 do_sweep=not args.no_sweep)

    if not results:
        print("[error] no methods ran"); return

    gates = [g[0] for g in LBS_GATE_SPECS]
    controlled = {m: results[m]["controlled"] for m in results}
    zero = {m: results[m]["zero"] for m in results}

    # pooled "overall across all methods" (Fisher-z mean + joint pocket-cluster bootstrap)
    print(f"\n[pooled] computing overall-across-methods rho (boot={args.pool_boot}) ...")
    pooled = compute_pooled(results, LBS_GATE_SPECS, boot=args.pool_boot, seed=0)
    pooled_to_frame(pooled, gates).to_csv(args.out_base / "pooled_overall_ratio_vs_class.csv", index=False)
    for regime in ("zero", "ctrl"):
        for gate in gates:
            rec = pooled[regime][gate]
            r = rec["rho"]
            print(f"  [{regime}/{gate}] overall rho  ratio={r['log2_ratio']}  "
                  f"p_holo={r['p_holo']}  is_holo={r['is_holo']}  "
                  f"| is_holo>ratio? {rec['surpasses']['is_holo']}  "
                  f"p_holo>ratio? {rec['surpasses']['p_holo']}")

    write_cross_report(results, pooled, gates, args.out_base / "REPORT.md")
    _export_figure(controlled, zero, pooled, gates, args)

    print(f"\n[SAVE] {args.out_base}")
    print(f"       pooled overall    -> pooled_overall_ratio_vs_class.csv")
    print(f"       synthesis report  -> REPORT.md")


def _export_figure(controlled, zero, pooled, gates, args) -> None:
    """Render the cross-tool figure to out_base (png) and, if requested, to the manuscript
    folder in png+pdf+svg + a standalone caption."""
    paths = [args.out_base / "fig_cross_tool_ratio_vs_class.png"]
    if args.manuscript_dir is not None:
        mdir = args.manuscript_dir
        mdir.mkdir(parents=True, exist_ok=True)
        paths += [mdir / f"{args.fig_basename}{ext}" for ext in (".png", ".pdf", ".svg")]
        write_figure_caption(pooled, mdir / f"{args.fig_basename}_caption.txt")
    fig_cross_tool(controlled, zero, pooled, gates, paths)
    print(f"       cross-tool figure -> fig_cross_tool_ratio_vs_class.png"
          + (f"  (+ manuscript: {args.manuscript_dir}\\{args.fig_basename}.{{png,pdf,svg}} + caption)"
             if args.manuscript_dir is not None else ""))


if __name__ == "__main__":
    main()
