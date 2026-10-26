# -*- coding: utf-8 -*-
"""
QUALITY-GATED ratio-vs-class test  (docking + P2Rank + AlphaFill).

We established across all three functional readouts that the holo/apo **evidence
ratio** log2((n_holo_avg+1)/n_apo_avg) out-predicts the apo/holo **class label**
(`is_holo`) and the class probability (`p_holo`), size-controlled, and that the
classification looks largely redundant with the ratio.

BUT p_holo may be unreliable when the underlying evidence is THIN: a probability
estimated from few known states or low KDE model density (`pocket_density_pair`)
carries little information, whereas a confident call backed by high density should
be genuinely more meaningful.

HYPOTHESIS: restrict to HIGH-QUALITY pockets (high classification confidence
`winning_p` AND high density `pocket_density_pair`) and the class label / p_holo
should become a *better* predictor of functional outcome — possibly matching or
surpassing the evidence ratio.

WHAT THIS SCRIPT DOES
  * Builds a standardized per-pocket table for each method with the DIRECTIONAL
    predictors {log2_ratio, p_holo, is_holo}, the GATING confidence axes
    {winning_p, density=pocket_density_pair}, size controls, and functional outcomes.
  * Gates on VERBAL confidence levels (reusing 03_alphafill/ubs/confidence_labeling.py):
    probability_level(winning_p) and support_level(pocket_density_pair);
    moderate+ = {moderate,high,very_high}, high+ = {high,very_high};
    gates = each axis alone + the 2x2 prob x density combinations + the ungated set.
  * Within every gate, compares predictive power of the classification (is_holo, p_holo)
    vs the evidence ratio (log2_ratio) with TWO metrics, both reused from the sibling
    analyses (analyze_docking_by_confidence.redundancy / analyze_p2rank_ratio_vs_class):
       (a) size-controlled partial Spearman rho of each predictor vs the outcome;
       (b) nested 5-fold CV logistic AUC (size / +ratio / +prob / +class / +ratio+prob).
  * Because tightening a gate SHRINKS n, the crossover claim is judged on the
    *difference* rho(p_holo) - rho(log2_ratio) (and is_holo - log2_ratio) with a
    PAIRED bootstrap 95% CI, not bare point estimates. n is reported at every gate.
  * Q3 crossover sweep: sweeps a continuous threshold on winning_p, on density, and
    on a combined prob x density confidence score, reporting where (if anywhere) the
    classification robustly surpasses the ratio.

All three predictors are compared on the SAME pocket set within each gate (rows where
the ratio is defined, i.e. n_apo>0), so the difference is perfectly paired.

Run in the `plotting` env (pandas/numpy/scipy/sklearn/matplotlib). Read .csv (no pyarrow).

  python analyze_quality_gated_ratio_vs_class.py --method all
  python analyze_quality_gated_ratio_vs_class.py --method docking --boot 500
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as sstats
    from scipy.stats import rankdata
except Exception:                                                     # pragma: no cover
    sstats = None

# --- repo root + config -----------------------------------------------------
for _root in Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

# --- reuse the manuscript's EXACT verbal-bin definitions --------------------
for _r in Path(__file__).resolve().parents:
    _cl = _r / "03_alphafill" / "ubs" / "confidence_labeling.py"
    if _cl.is_file():
        sys.path.insert(0, str(_cl.parent))
        break
import confidence_labeling as cl  # noqa: E402

# ---------------------------------------------------------------------------
CONFIDENT = ("moderate", "high", "very_high")        # "moderate+"
HIGH = ("high", "very_high")                          # "high+"
PROB_K = 1.64                                         # confidence_labeling sigmoid k

# The three DIRECTIONAL apo/holo predictors being compared.
RATIO = "log2_ratio"
PREDICTORS = [RATIO, "p_holo", "is_holo"]
PRED_LABEL = {RATIO: "evidence ratio", "p_holo": "p_holo (class prob)", "is_holo": "class label"}
PRED_COLORS = {RATIO: "#7B3294", "p_holo": "#2C7FB8", "is_holo": "#C2A5CF"}
CLASS_COLORS = {"apo": "#2C7FB8", "holo": "#F28E2B"}

# gates, ordered from loosest (all) to tightest (high+ x high+)
GATE_SPECS = [
    ("all", None, None),
    ("prob_mod+", CONFIDENT, None),
    ("dens_mod+", None, CONFIDENT),
    ("prob_high+", HIGH, None),
    ("dens_high+", None, HIGH),
    ("prob_mod+_dens_mod+", CONFIDENT, CONFIDENT),
    ("prob_mod+_dens_high+", CONFIDENT, HIGH),
    ("prob_high+_dens_mod+", HIGH, CONFIDENT),
    ("prob_high+_dens_high+", HIGH, HIGH),
]


# ============================================================================
# Vectorized confidence unitizers (match confidence_labeling exactly)
# ============================================================================
def unit_prob_vec(p: np.ndarray, k: float = PROB_K) -> np.ndarray:
    """Vectorized confidence_labeling.unitize_probability(mode='sigmoid')."""
    p = np.clip(np.asarray(p, dtype=float), 0.5, 1.0 - 1e-12)
    logits = np.log(p / (1.0 - p))
    pp = 1.0 / (1.0 + np.exp(-k * logits))
    return np.clip((pp - 0.5) / 0.5, 0.0, 1.0)


def unit_dens_vec(d: np.ndarray) -> np.ndarray:
    """Vectorized confidence_labeling.unitize(src_min=0, src_max=1)."""
    return np.clip(np.asarray(d, dtype=float), 0.0, 1.0)


# ============================================================================
# Partial-Spearman machinery (residualize ranks on control ranks)
# ============================================================================
def _rank_cols(M: np.ndarray) -> np.ndarray:
    """Average ranks of each column (ties -> mean rank), C-fast via scipy."""
    return np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])


def _partial_rhos_from_ranks(Vr: np.ndarray, n_pred: int, n_ctrl: int) -> np.ndarray:
    """Vr columns = [y, pred_1..pred_p, ctrl_1..ctrl_c] already rank-transformed.
    Residualize y and each predictor on [1, ctrl-ranks]; return partial rho per pred."""
    n = Vr.shape[0]
    if n_ctrl:
        Z = np.column_stack([np.ones(n), Vr[:, 1 + n_pred:1 + n_pred + n_ctrl]])
    else:
        Z = np.ones((n, 1))
    target = Vr[:, :1 + n_pred]                                       # y + predictors
    beta, *_ = np.linalg.lstsq(Z, target, rcond=None)
    resid = target - Z @ beta
    yv = resid[:, 0]
    ystd = yv.std()
    out = np.full(n_pred, np.nan)
    if ystd == 0:
        return out
    for j in range(n_pred):
        pv = resid[:, 1 + j]
        if pv.std() == 0:
            continue
        out[j] = float(np.corrcoef(yv, pv)[0, 1])
    return out


def partial_rhos(y: np.ndarray, preds: np.ndarray, ctrl: np.ndarray):
    """Point-estimate partial Spearman rho for each predictor column.

    Returns (rhos[p], pvals[p], n). All predictors share the SAME rows (caller
    passes an already-cleaned, common-mask matrix) so the rhos are comparable."""
    n = len(y)
    n_pred = preds.shape[1]
    n_ctrl = ctrl.shape[1] if ctrl.ndim == 2 else 0
    if n < 20:
        return np.full(n_pred, np.nan), np.full(n_pred, np.nan), n
    M = np.column_stack([y, preds] + ([ctrl] if n_ctrl else []))
    Vr = _rank_cols(M)
    rhos = _partial_rhos_from_ranks(Vr, n_pred, n_ctrl)
    df = n - 2 - n_ctrl
    pvals = np.full(n_pred, np.nan)
    if sstats is not None and df > 0:
        for j, r in enumerate(rhos):
            if r is None or np.isnan(r) or abs(r) >= 1:
                continue
            t = abs(r) * np.sqrt(df / (1 - r * r))
            pvals[j] = float(2 * sstats.t.sf(t, df))
    return rhos, pvals, n


def bootstrap_diff(y: np.ndarray, preds: np.ndarray, ctrl: np.ndarray,
                   b: int, seed: int = 0):
    """Paired bootstrap of rho(pred_k) - rho(ratio) for every non-ratio predictor.

    preds column 0 MUST be the evidence ratio. Returns per-predictor dict with
    the point diff, 2.5/97.5 CI, and P(diff>0) across resamples."""
    n = len(y)
    n_pred = preds.shape[1]
    rng = np.random.default_rng(seed)
    diffs = np.full((b, n_pred - 1), np.nan)
    for i in range(b):
        idx = rng.integers(0, n, n)
        M = np.column_stack([y[idx], preds[idx]] + ([ctrl[idx]] if ctrl.ndim == 2 and ctrl.shape[1] else []))
        Vr = _rank_cols(M)
        r = _partial_rhos_from_ranks(Vr, n_pred, ctrl.shape[1] if ctrl.ndim == 2 else 0)
        diffs[i, :] = r[1:] - r[0]                                    # pred_k - ratio
    out = {}
    for k in range(1, n_pred):
        d = diffs[:, k - 1]
        d = d[~np.isnan(d)]
        if d.size < max(20, b // 5):
            out[k] = dict(diff=None, lo=None, hi=None, p_gt0=None, n_boot=int(d.size))
            continue
        out[k] = dict(diff=round(float(np.mean(d)), 4),
                      lo=round(float(np.percentile(d, 2.5)), 4),
                      hi=round(float(np.percentile(d, 97.5)), 4),
                      p_gt0=round(float(np.mean(d > 0)), 4),
                      n_boot=int(d.size))
    return out


# ============================================================================
# Nested cross-validated logistic AUC  (binary outcomes)
# ============================================================================
def nested_auc(g: pd.DataFrame, ycol: str, ctrl_cols: List[str], mask: pd.Series) -> pd.DataFrame:
    """size / +ratio / +prob / +class / +ratio+prob  AUC on a COMMON row mask."""
    rows = []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except Exception as e:                                            # pragma: no cover
        return pd.DataFrame([{"model": "ERROR", "predictors": repr(e), "cv_auc": None, "n": 0}])
    models = {
        "size": ctrl_cols,
        "size+ratio": ctrl_cols + [RATIO],
        "size+prob": ctrl_cols + ["p_holo"],
        "size+class": ctrl_cols + ["is_holo"],
        "size+ratio+prob": ctrl_cols + [RATIO, "p_holo"],
    }
    y = g.loc[mask, ycol].astype(int).values
    if len(np.unique(y)) < 2 or mask.sum() < 100:
        return pd.DataFrame(rows)
    for name, cols in models.items():
        cols = [c for c in cols if c in g.columns]
        X = StandardScaler().fit_transform(g.loc[mask, cols].values)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        try:
            proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv,
                                      method="predict_proba")[:, 1]
            auc = round(float(roc_auc_score(y, proba)), 4)
        except Exception:
            auc = None
        rows.append({"model": name, "predictors": "+".join(cols), "cv_auc": auc, "n": int(mask.sum())})
    return pd.DataFrame(rows)


# ============================================================================
# Method dataset descriptor + loaders  (standardized per-pocket table)
# ============================================================================
@dataclass
class Outcome:
    col: str            # column in the per-pocket frame holding a "goodness" (higher = better)
    label: str
    binary: bool = False
    primary: bool = False


@dataclass
class MethodData:
    name: str
    g: pd.DataFrame                 # per-pocket frame
    ctrl_cols: List[str]
    outcomes: List[Outcome]
    ctrl_note: str = ""
    coverage_note: str = ""

    def primary(self) -> Outcome:
        return next(o for o in self.outcomes if o.primary)


def _finish(g: pd.DataFrame) -> pd.DataFrame:
    """Attach the shared predictor + gating columns to any per-pocket frame that
    already carries pocket_class, p_apo, p_holo, density, n_apo, n_holo."""
    g = g[g["pocket_class"].isin(["apo", "holo"])].copy()
    for c in ["p_apo", "p_holo", "density", "n_apo", "n_holo"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    if "winning_p" not in g.columns:
        g["winning_p"] = g[["p_apo", "p_holo"]].max(axis=1)
    g["winning_p"] = pd.to_numeric(g["winning_p"], errors="coerce")
    g["is_holo"] = (g["pocket_class"] == "holo").astype(float)
    # canonical evidence ratio: +1 to HOLO only (query compensation); apo unoffset;
    # apo==0 -> undefined (NaN). Matches manuscript ratio_holo_to_apo_avg_plus1.
    g[RATIO] = np.log2((g["n_holo"] + 1.0) / g["n_apo"].replace(0, np.nan))
    # verbal gate levels (undirected confidence axes)
    g["prob_level"] = g["winning_p"].map(lambda v: cl.probability_level(v, merge=False))
    g["dens_level"] = g["density"].map(lambda v: cl.support_level(v, merge=False))
    # continuous unitized confidence + combined score (for the threshold sweep)
    g["u_prob"] = unit_prob_vec(g["winning_p"].values)
    g["u_dens"] = unit_dens_vec(g["density"].values)
    g["u_combined"] = g["u_prob"] * g["u_dens"]
    return g


def load_docking(merged_csv: Path, group: int = 1) -> MethodData:
    df = pd.read_csv(merged_csv, low_memory=False)
    ok = df[(df.get("status") == "ok") & (df["ligand_group"] == group)].copy()
    ok["aff"] = pd.to_numeric(ok["affinity_top"], errors="coerce")
    ok["dockable"] = pd.to_numeric(ok["dockable"], errors="coerce")
    ok["heavy"] = pd.to_numeric(ok.get("ligand_n_heavy_atoms"), errors="coerce")
    g = ok.groupby("site_uid").agg(
        pocket_class=("pocket_class", "first"),
        p_apo=("pocket_p_apo", "first"), p_holo=("pocket_p_holo", "first"),
        density=("pocket_density_pair", "first"),
        n_apo=("n_apo_avg", "first"), n_holo=("n_holo_avg", "first"),
        union_rg=("union_rg", "first"), heavy=("heavy", "median"),
        n_lig=("aff", "size"),
        frac_dockable=("dockable", "mean"), any_dockable=("dockable", "max"),
        median_aff=("aff", "median"),
    ).reset_index()
    g = _finish(g)
    g["aff_goodness"] = -pd.to_numeric(g["median_aff"], errors="coerce")   # more negative aff = better
    g["union_rg"] = pd.to_numeric(g["union_rg"], errors="coerce")
    return MethodData(
        name="docking", g=g, ctrl_cols=["union_rg", "heavy"],
        ctrl_note="pocket size `union_rg` + ligand size (median `ligand_n_heavy_atoms`)",
        coverage_note=f"G{group} query-holo cognate; one row per pocket ({len(g):,}).",
        outcomes=[
            Outcome("frac_dockable", "fraction of cognate ligands dockable", binary=False, primary=True),
            Outcome("any_dockable", "any cognate ligand dockable", binary=True),
            Outcome("aff_goodness", "top-pose affinity goodness (-kcal/mol)", binary=False),
        ],
    )


def load_p2rank(per_pocket_csv: Path) -> MethodData:
    d = pd.read_csv(per_pocket_csv, low_memory=False)
    d = d.rename(columns={"pocket_density_pair": "density"})
    d["p_apo"] = d["pocket_p_apo"]
    d["p_holo"] = d["pocket_p_holo"]
    d["n_apo"] = d["n_apo_avg"]
    d["n_holo"] = d["n_holo_avg"]
    for sc in ["success_tns", "success_dca_4A"]:
        d[sc] = pd.to_numeric(d[sc], errors="coerce")
    d["rg"] = pd.to_numeric(d["rg"], errors="coerce")
    d["ligand_atoms"] = pd.to_numeric(d["ligand_atoms"], errors="coerce")
    g = _finish(d)
    return MethodData(
        name="p2rank", g=g, ctrl_cols=["rg", "ligand_atoms"],
        ctrl_note="pocket size `rg` + ligand size `ligand_atoms`",
        coverage_note=f"per-pocket P2Rank table ({len(g):,}).",
        outcomes=[
            Outcome("success_tns", "P2Rank TNS detection", binary=True, primary=True),
            Outcome("success_dca_4A", "P2Rank DCA<=4A detection", binary=True),
        ],
    )


def load_alphafill(long_csv: Path, threshold: str = "100", scope: str = "same_uniprot",
                   min_identity: float = 0.30) -> MethodData:
    """Per-pocket AlphaFill transplant fit.

    IMPORTANT (fixed 2026-07-07): the v7 long table pools MULTIPLE UBS clustering
    schemes inside `unit_uid` — the numeric density thresholds :ST_030/050/070/100:
    AND a separate :ST_LJ###: scheme — and BOTH match scopes (same_uniprot =
    `afill_cross_uniprot==False`, plus cross-UniProt "any"). A bare ':ST_' substring
    pools all of them (~80k pockets), which mixes clustering thresholds (the same
    protein site counted 4-5x = pseudo-replication) and cross-sequence transplants.
    We restrict to ONE threshold (ST_100, the same universe docking & P2Rank use;
    `unit_uid` format identical to their `site_uid`) and the manuscript's headline
    scope (same_uniprot, alignment identity >= 0.30). That yields 17,551 apo/holo
    pockets — matching the manuscript's same_uniprot ge_0p30 UBS count exactly."""
    df = pd.read_csv(long_csv, low_memory=False)
    m = df["unit_uid"].astype(str).str.contains(fr":ST_{threshold}:", regex=True)
    m &= df["pocket_class"].isin(["apo", "holo"])
    ident = pd.to_numeric(df.get("afill_mapping_align_identity"), errors="coerce")
    m &= ident >= float(min_identity)
    is_cross = df["afill_cross_uniprot"].astype(str).str.lower().isin(["true", "1"])
    if scope == "same_uniprot":
        m &= ~is_cross                                               # donor = same UniProt
    st = df[m].copy()
    st["clash"] = pd.to_numeric(st["afill_clash_score"], errors="coerce")
    st["rmsd"] = pd.to_numeric(st["afill_local_rmsd"], errors="coerce")
    g = st.groupby("unit_uid").agg(
        pocket_class=("pocket_class", "first"),
        p_apo=("pocket_p_apo", "first"), p_holo=("pocket_p_holo", "first"),
        winning_p=("winning_p", "first"),
        density=("pocket_density_pair", "first"),
        n_apo=("n_apo_avg", "first"), n_holo=("n_holo_avg", "first"),
        site_size_unit=("site_size_unit", "first"),
        n_transplant=("clash", "size"),
        clash=("clash", "median"), rmsd=("rmsd", "median"),
    ).reset_index()
    g = _finish(g)
    g["site_size_unit"] = pd.to_numeric(g["site_size_unit"], errors="coerce")
    g["clash_goodness"] = -g["clash"]                                # fewer clashes = better
    g["rmsd_goodness"] = -g["rmsd"]                                  # lower local RMSD = better
    # median-split binary proxy (fixed global threshold) for the nested-AUC view
    med = g["clash"].median()
    g["good_fit"] = (g["clash"] <= med).astype(float)
    return MethodData(
        name="alphafill", g=g, ctrl_cols=["site_size_unit"],
        ctrl_note="pocket size proxy `site_size_unit` (no union_rg / ligand-size in the long table)",
        coverage_note=(f"AlphaFill ST_{threshold} {scope} (identity>={min_identity:g}) transplants "
                       f"aggregated per pocket ({len(g):,}); same UBS universe + unit_uid format as "
                       "docking/P2Rank. ~17% of docking pockets carry a transplant; weak outcome "
                       "(clash/local-RMSD is geometric fit, not binding) -> supporting evidence only."),
        outcomes=[
            Outcome("clash_goodness", "transplant fit goodness (-clash_score)", binary=False, primary=True),
            Outcome("good_fit", f"good fit (clash<=global median {med:.3f})", binary=True),
            Outcome("rmsd_goodness", "local geometry goodness (-local_rmsd)", binary=False),
        ],
    )


# ============================================================================
# Gate + outcome evaluation
# ============================================================================
def gate_mask(g: pd.DataFrame, prob_levels, dens_levels) -> pd.Series:
    m = pd.Series(True, index=g.index)
    if prob_levels is not None:
        m &= g["prob_level"].isin(prob_levels)
    if dens_levels is not None:
        m &= g["dens_level"].isin(dens_levels)
    return m


def _common_mask(g: pd.DataFrame, ycol: str, ctrl_cols: List[str]) -> pd.Series:
    """Rows valid for ALL predictors + outcome + controls (ratio-defined)."""
    cols = [ycol, RATIO, "p_holo", "is_holo"] + ctrl_cols
    return g[cols].notna().all(axis=1)


def eval_gate(md: MethodData, gate_name: str, prob_levels, dens_levels,
              boot: int) -> dict:
    """One gate x the PRIMARY outcome: partial rho for 3 predictors + bootstrap diffs."""
    g = md.g
    outc = md.primary()
    gm = gate_mask(g, prob_levels, dens_levels)
    cm = _common_mask(g, outc.col, md.ctrl_cols)
    sub = g[gm & cm]
    n = len(sub)
    n_gate = int(gm.sum())
    n_holo = int((sub.pocket_class == "holo").sum())
    n_apo = n - n_holo
    row = {"gate": gate_name, "outcome": outc.col, "n_gate": n_gate, "n_used": n,
           "n_apo": n_apo, "n_holo": n_holo}
    if n < 100 or n_holo < 15 or n_apo < 15:
        row["underpowered"] = True
        return row
    y = sub[outc.col].to_numpy(float)
    preds = sub[PREDICTORS].to_numpy(float)                          # col 0 = ratio
    ctrl = sub[md.ctrl_cols].to_numpy(float)
    rhos, pvals, _ = partial_rhos(y, preds, ctrl)
    for k, pred in enumerate(PREDICTORS):
        row[f"rho_{pred}"] = None if np.isnan(rhos[k]) else round(float(rhos[k]), 4)
        row[f"p_{pred}"] = None if np.isnan(pvals[k]) else _fmt_p(pvals[k])
    bd = bootstrap_diff(y, preds, ctrl, b=boot, seed=0)
    for k, pred in enumerate(PREDICTORS[1:], start=1):
        d = bd.get(k, {})
        row[f"diff_{pred}_minus_ratio"] = d.get("diff")
        row[f"ci_lo_{pred}"] = d.get("lo")
        row[f"ci_hi_{pred}"] = d.get("hi")
        row[f"p_gt0_{pred}"] = d.get("p_gt0")
        # robust surpass = point diff > 0 AND bootstrap CI excludes 0 on the positive side
        row[f"surpasses_ratio_{pred}"] = bool(
            d.get("diff") is not None and d.get("lo") is not None
            and d["diff"] > 0 and d["lo"] > 0)
    row["underpowered"] = False
    return row


def eval_gate_auc(md: MethodData, gate_name: str, prob_levels, dens_levels) -> pd.DataFrame:
    """Nested CV AUC per binary outcome within the gate."""
    frames = []
    for outc in md.outcomes:
        if not outc.binary:
            continue
        gm = gate_mask(md.g, prob_levels, dens_levels)
        cm = _common_mask(md.g, outc.col, md.ctrl_cols)
        mask = gm & cm
        if mask.sum() < 100:
            continue
        a = nested_auc(md.g, outc.col, md.ctrl_cols, mask)
        if a.empty:
            continue
        a.insert(0, "gate", gate_name)
        a.insert(1, "outcome", outc.col)
        frames.append(a)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================================
# Q3 threshold sweep  (find the crossover)
# ============================================================================
SWEEP_AXES = [
    ("winning_p", "winning_p", np.round(np.arange(0.50, 0.951, 0.05), 3)),
    ("density", "density", np.round(np.arange(0.00, 0.901, 0.10), 3)),
    ("combined", "u_combined", np.round(np.arange(0.00, 0.701, 0.10), 3)),
]


def sweep(md: MethodData, boot: int) -> pd.DataFrame:
    g = md.g
    outc = md.primary()
    cm = _common_mask(g, outc.col, md.ctrl_cols)
    base = g[cm]
    rows = []
    for axis_name, axis_col, grid in SWEEP_AXES:
        for t in grid:
            sub = base[base[axis_col] >= t]
            n = len(sub)
            n_holo = int((sub.pocket_class == "holo").sum())
            rec = {"axis": axis_name, "threshold": float(t), "n_used": n,
                   "n_holo": n_holo, "n_apo": n - n_holo}
            if n < 100 or n_holo < 15 or (n - n_holo) < 15:
                rec["underpowered"] = True
                rows.append(rec)
                continue
            y = sub[outc.col].to_numpy(float)
            preds = sub[PREDICTORS].to_numpy(float)
            ctrl = sub[md.ctrl_cols].to_numpy(float)
            rhos, _, _ = partial_rhos(y, preds, ctrl)
            for k, pred in enumerate(PREDICTORS):
                rec[f"rho_{pred}"] = None if np.isnan(rhos[k]) else round(float(rhos[k]), 4)
            bd = bootstrap_diff(y, preds, ctrl, b=boot, seed=0)
            for k, pred in enumerate(PREDICTORS[1:], start=1):
                d = bd.get(k, {})
                rec[f"diff_{pred}"] = d.get("diff")
                rec[f"ci_lo_{pred}"] = d.get("lo")
                rec[f"ci_hi_{pred}"] = d.get("hi")
                rec[f"surpasses_{pred}"] = bool(
                    d.get("diff") is not None and d.get("lo") is not None
                    and d["diff"] > 0 and d["lo"] > 0)
            rec["underpowered"] = False
            rows.append(rec)
    return pd.DataFrame(rows)


def find_crossover(sweep_df: pd.DataFrame) -> dict:
    """Smallest threshold per axis where p_holo (and/or is_holo) robustly surpasses
    the ratio (point diff>0 AND bootstrap CI lower bound>0)."""
    out = {}
    for axis in sweep_df.axis.unique():
        sub = sweep_df[(sweep_df.axis == axis) & (~sweep_df.get("underpowered", False))]
        info = {}
        for pred in ["p_holo", "is_holo"]:
            col = f"surpasses_{pred}"
            if col not in sub.columns:
                info[pred] = None
                continue
            hit = sub[sub[col] == True].sort_values("threshold")
            if hit.empty:
                info[pred] = None
            else:
                r = hit.iloc[0]
                info[pred] = {"threshold": float(r.threshold), "n_used": int(r.n_used),
                              "diff": r.get(f"diff_{pred}"), "ci_lo": r.get(f"ci_lo_{pred}"),
                              "ci_hi": r.get(f"ci_hi_{pred}")}
        out[axis] = info
    return out


# ============================================================================
# Figures
# ============================================================================
def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    if p == 0.0:
        return 0.0
    return float(f"{p:.1e}") if p < 1e-4 else round(float(p), 4)


def fig_gates(gate_df: pd.DataFrame, md: MethodData, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = gate_df[~gate_df.get("underpowered", False).fillna(False)].copy()
    order = [gs[0] for gs in GATE_SPECS if gs[0] in set(d.gate)]
    d = d.set_index("gate").reindex(order).reset_index()
    x = np.arange(len(d))
    w = 0.26
    fig, ax = plt.subplots(figsize=(1.15 * len(d) + 3, 5.2))
    for k, pred in enumerate(PREDICTORS):
        vals = [d[f"rho_{pred}"].iloc[i] if f"rho_{pred}" in d.columns else np.nan for i in range(len(d))]
        vals = [np.nan if v is None else float(v) for v in vals]
        yerr = None
        if pred != RATIO:
            lo = d.get(f"ci_lo_{pred}"); hi = d.get(f"ci_hi_{pred}")
            # error bars are the CI of the DIFF vs ratio, drawn around the predictor rho
            if lo is not None and hi is not None:
                base_ratio = [float(v) if v is not None else np.nan for v in d["rho_log2_ratio"]]
                lo_abs = [base_ratio[i] + (float(d[f'ci_lo_{pred}'].iloc[i]) if d[f'ci_lo_{pred}'].iloc[i] is not None else np.nan) for i in range(len(d))]
                hi_abs = [base_ratio[i] + (float(d[f'ci_hi_{pred}'].iloc[i]) if d[f'ci_hi_{pred}'].iloc[i] is not None else np.nan) for i in range(len(d))]
                yerr = np.array([[vals[i] - lo_abs[i] for i in range(len(d))],
                                 [hi_abs[i] - vals[i] for i in range(len(d))]])
                yerr = np.abs(np.nan_to_num(yerr))
        ax.bar(x + (k - 1) * w, vals, width=w, color=PRED_COLORS[pred],
               yerr=yerr, capsize=2.5, error_kw=dict(lw=0.8, ecolor="0.35"),
               label=PRED_LABEL[pred])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\nn={int(n):,}" for g, n in zip(d.gate, d.n_used)],
                       rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel(f"size-controlled partial ρ  vs  {md.primary().label}")
    ax.set_title(f"{md.name}: classification vs evidence ratio, by quality gate\n"
                 f"(error bars = 95% bootstrap CI of the ρ difference vs ratio; controls: {md.ctrl_note})",
                 fontsize=10)
    ax.legend(fontsize=8.5, title="apo/holo predictor")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def fig_sweep(sweep_df: pd.DataFrame, md: MethodData, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes_names = [a[0] for a in SWEEP_AXES]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, axis_name in zip(axes, axes_names):
        sub = sweep_df[(sweep_df.axis == axis_name) & (~sweep_df.get("underpowered", False).fillna(False))]
        if sub.empty:
            ax.set_title(f"{axis_name}: no powered thresholds"); continue
        for pred in PREDICTORS:
            ax.plot(sub.threshold, sub[f"rho_{pred}"].astype(float), "o-",
                    color=PRED_COLORS[pred], lw=2, ms=4, label=PRED_LABEL[pred])
        # mark robust crossover points
        for pred in ["p_holo", "is_holo"]:
            hit = sub[sub.get(f"surpasses_{pred}", False) == True]
            if not hit.empty:
                t0 = hit.sort_values("threshold").iloc[0]
                ax.axvline(float(t0.threshold), color=PRED_COLORS[pred], ls=":", lw=1.2, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(sub.threshold, sub.n_used, "-", color="0.6", lw=1, alpha=0.6)
        ax2.set_yscale("log"); ax2.set_ylabel("n pockets (log)", color="0.5", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7, colors="0.5")
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_xlabel(f"{axis_name} threshold (keep >= t)")
        ax.set_ylabel(f"partial ρ vs {md.primary().col}" if axis_name == axes_names[0] else "")
        ax.set_title(axis_name); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{md.name}: does the classification surpass the evidence ratio as the "
                 f"quality threshold rises?  (dotted line = robust crossover)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


# ============================================================================
# Report (per method)
# ============================================================================
def write_method_report(md: MethodData, gate_df: pd.DataFrame, auc_df: pd.DataFrame,
                        sweep_df: pd.DataFrame, cross: dict, path: Path) -> None:
    outc = md.primary()
    L = [f"# {md.name}: quality-gated ratio-vs-class  (per-pocket)\n",
         f"{md.coverage_note}\n",
         f"Primary outcome = **{outc.label}** (`{outc.col}`, higher = better). "
         f"Size controls = {md.ctrl_note}. Predictors compared: evidence ratio "
         f"`log2((n_holo+1)/n_apo)` vs class probability `p_holo` vs binary `is_holo`. "
         "All three are scored on the SAME ratio-defined pockets within each gate, so the "
         "difference is paired; 95% CIs are paired bootstraps of the ρ difference vs the ratio.\n"]

    L.append("\n## Gate table — size-controlled partial ρ vs the primary outcome\n")
    L.append("| gate | n (gate/used) | apo/holo | ρ ratio | ρ p_holo | Δ(p_holo−ratio) [95% CI] | ρ is_holo | Δ(is_holo−ratio) [95% CI] | class beats ratio? |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    order = [gs[0] for gs in GATE_SPECS if gs[0] in set(gate_df.gate)]
    gd = gate_df.set_index("gate").reindex(order).reset_index()
    for _, r in gd.iterrows():
        if r.get("underpowered", False) or pd.isna(r.get("rho_log2_ratio")):
            L.append(f"| {r['gate']} | {int(r.n_gate):,}/{int(r.n_used):,} | "
                     f"{int(r.n_apo)}/{int(r.n_holo)} | underpowered |||||||")
            continue
        def ci(pred):
            lo, hi = r.get(f"ci_lo_{pred}"), r.get(f"ci_hi_{pred}")
            dv = r.get(f"diff_{pred}_minus_ratio")
            return f"{dv:+.3f} [{lo:+.3f},{hi:+.3f}]" if dv is not None and lo is not None else "–"
        beats = []
        if r.get("surpasses_ratio_p_holo"): beats.append("p_holo")
        if r.get("surpasses_ratio_is_holo"): beats.append("is_holo")
        L.append(f"| {r['gate']} | {int(r.n_gate):,}/{int(r.n_used):,} | {int(r.n_apo)}/{int(r.n_holo)} | "
                 f"{r.rho_log2_ratio:+.3f} | {r.rho_p_holo:+.3f} | {ci('p_holo')} | "
                 f"{r.rho_is_holo:+.3f} | {ci('is_holo')} | {', '.join(beats) if beats else '—'} |")

    if auc_df is not None and not auc_df.empty:
        L.append("\n## Nested 5-fold CV logistic AUC (binary outcomes)\n")
        for oc in auc_df.outcome.unique():
            sub = auc_df[auc_df.outcome == oc]
            piv = sub.pivot_table(index="gate", columns="model", values="cv_auc", aggfunc="first")
            cols = [c for c in ["size", "size+ratio", "size+prob", "size+class", "size+ratio+prob"] if c in piv.columns]
            piv = piv.reindex(index=[g for g in order if g in piv.index], columns=cols)
            L.append(f"\n**{oc}**\n")
            L.append("| gate | " + " | ".join(cols) + " |")
            L.append("|" + "---|" * (len(cols) + 1))
            for idx, row in piv.iterrows():
                L.append(f"| {idx} | " + " | ".join(f"{row[c]:.4f}" if pd.notna(row[c]) else "–" for c in cols) + " |")

    L.append("\n## Q3 — threshold sweep crossover\n")
    L.append("Smallest quality threshold at which the classification **robustly** surpasses the "
             "ratio (point Δρ>0 AND bootstrap CI lower bound>0):\n")
    for axis, info in cross.items():
        parts = []
        for pred in ["p_holo", "is_holo"]:
            v = info.get(pred)
            if v is None:
                parts.append(f"{pred}: never")
            else:
                parts.append(f"{pred}: t≥{v['threshold']:g} (n={v['n_used']:,}, Δ={v['diff']:+.3f} "
                             f"[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}])")
        L.append(f"- **{axis}** — " + "; ".join(parts))

    L.append("\n## Reading\n")
    L.append("- If Δ(p_holo−ratio) stays **negative with CI excluding 0** across gates, the ratio "
             "remains the better predictor even among high-quality classifications — the classification "
             "is a thresholded summary of the evidence at every confidence level.")
    L.append("- If Δ turns **positive with CI>0** at some gate/threshold, the class label / p_holo "
             "genuinely surpasses the ratio once the call is well supported — a real quality-dependent "
             "crossover, candidate for a manuscript figure.")
    L.append("- Watch n: tightening the gate shrinks the sample; a positive point estimate whose CI "
             "still straddles 0 is **not** a crossover, just noise.")
    path.write_text("\n".join(L), encoding="utf-8")


# ============================================================================
# Driver
# ============================================================================
def run_method(md: MethodData, out_dir: Path, boot: int, sweep_boot: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    md.g.to_csv(out_dir / "pocket_level.csv", index=False)

    gate_rows, auc_frames = [], []
    for name, pl, dl in GATE_SPECS:
        gate_rows.append(eval_gate(md, name, pl, dl, boot=boot))
        a = eval_gate_auc(md, name, pl, dl)
        if not a.empty:
            auc_frames.append(a)
    gate_df = pd.DataFrame(gate_rows)
    auc_df = pd.concat(auc_frames, ignore_index=True) if auc_frames else pd.DataFrame()
    gate_df.to_csv(out_dir / "gates_partial_spearman.csv", index=False)
    if not auc_df.empty:
        auc_df.to_csv(out_dir / "gates_nested_auc.csv", index=False)

    sweep_df = sweep(md, boot=sweep_boot)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)
    cross = find_crossover(sweep_df)

    fig_gates(gate_df, md, out_dir / f"fig_gates_{md.name}.png")
    fig_sweep(sweep_df, md, out_dir / f"fig_sweep_{md.name}.png")
    write_method_report(md, gate_df, auc_df, sweep_df, cross, out_dir / "REPORT.md")

    print(f"\n===== {md.name} =====")
    show = [c for c in ["gate", "n_gate", "n_used", "n_apo", "n_holo", "rho_log2_ratio",
                        "rho_p_holo", "diff_p_holo_minus_ratio", "ci_lo_p_holo", "ci_hi_p_holo",
                        "surpasses_ratio_p_holo", "rho_is_holo", "diff_is_holo_minus_ratio",
                        "surpasses_ratio_is_holo"] if c in gate_df.columns]
    print(gate_df[show].to_string(index=False))
    print("[crossover]", {a: {k: (v if v is None else v.get("threshold")) for k, v in info.items()}
                          for a, info in cross.items()})
    return {"name": md.name, "gate_df": gate_df, "auc_df": auc_df, "sweep_df": sweep_df,
            "cross": cross, "primary": md.primary().col, "ctrl": md.ctrl_note,
            "n": len(md.g)}


def main() -> None:
    ap = argparse.ArgumentParser()
    dock_default = (cfg.DOCKING_RUN3_DIR / "analysis" / "docking_run3_merged.csv"
                    if hasattr(cfg, "DOCKING_RUN3_DIR") else None)
    ap.add_argument("--method", choices=["docking", "p2rank", "alphafill", "all"], default="all")
    ap.add_argument("--docking_csv", type=Path, default=dock_default)
    ap.add_argument("--p2rank_csv", type=Path, default=cfg.P2RANK_MERGED_PER_POCKET_CSV)
    ap.add_argument("--alphafill_csv", type=Path, default=cfg.MERGED_AHOJ_ALPHAFILL_LONG_CSV)
    ap.add_argument("--out_base", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for gate diffs")
    ap.add_argument("--sweep_boot", type=int, default=300, help="bootstrap resamples per sweep point")
    ap.add_argument("--group", type=int, default=1, help="docking ligand group")
    ap.add_argument("--afill_threshold", default="100",
                    help="AlphaFill UBS density threshold token (100=ST_100, matches docking/P2Rank)")
    ap.add_argument("--afill_scope", choices=["same_uniprot", "any"], default="same_uniprot",
                    help="AlphaFill match scope (manuscript headline = same_uniprot)")
    ap.add_argument("--afill_min_identity", type=float, default=0.30,
                    help="min AlphaFill alignment identity (manuscript = 0.30)")
    args = ap.parse_args()

    if args.out_base is None:
        # default: the docking run3 analysis dir / quality_gated
        base = args.p2rank_csv  # fallback anchor
        dock_anchor = args.docking_csv
        args.out_base = (dock_anchor.parent / "quality_gated") if dock_anchor else (base.parent / "quality_gated")
    args.out_base.mkdir(parents=True, exist_ok=True)

    methods = ["docking", "p2rank", "alphafill"] if args.method == "all" else [args.method]
    results = []
    for m in methods:
        if m == "docking":
            if args.docking_csv is None or not Path(args.docking_csv).exists():
                print(f"[skip] docking csv not found: {args.docking_csv}"); continue
            md = load_docking(args.docking_csv, group=args.group)
        elif m == "p2rank":
            md = load_p2rank(args.p2rank_csv)
        else:
            md = load_alphafill(args.alphafill_csv, threshold=args.afill_threshold,
                                scope=args.afill_scope, min_identity=args.afill_min_identity)
        results.append(run_method(md, args.out_base / m, args.boot, args.sweep_boot))

    # auto synthesis -> REPORT_auto.md so a hand-curated REPORT.md is never clobbered
    write_synthesis(results, args.out_base / "REPORT_auto.md")
    print(f"\n[SAVE] {args.out_base}")


def write_synthesis(results: List[dict], path: Path) -> None:
    if not results:
        return
    L = ["# Quality-gated ratio-vs-class — cross-method synthesis\n",
         "**Question.** On high-quality apo/holo calls (high classification confidence `winning_p` "
         "AND high KDE density `pocket_density_pair`), does the class label / `p_holo` become a better "
         "predictor of functional outcome than the evidence ratio `log2((n_holo+1)/n_apo)` — matching "
         "or surpassing it?\n",
         "**Method.** Per-pocket, size-controlled partial Spearman + nested 5-fold CV logistic AUC, "
         "within verbal-confidence gates (moderate+ / high+ on each of probability and density and "
         "their 2×2 combinations), plus a continuous threshold sweep. Crossover counts as robust only "
         "when the point Δρ>0 AND the paired bootstrap 95% CI of the difference excludes 0. n reported "
         "at every gate.\n"]
    L.append("\n## Per-method crossover verdict\n")
    L.append("| method | n pockets | primary outcome | any robust crossover (gate)? | any robust crossover (sweep)? |")
    L.append("|---|---|---|---|---|")
    any_cross = False
    for r in results:
        gd = r["gate_df"]
        gate_hit = []
        for pred in ["p_holo", "is_holo"]:
            col = f"surpasses_ratio_{pred}"
            if col in gd.columns:
                hits = gd[gd[col] == True]
                gate_hit += [f"{pred}@{g}" for g in hits.gate.tolist()]
        sweep_hit = []
        for axis, info in r["cross"].items():
            for pred in ["p_holo", "is_holo"]:
                if info.get(pred):
                    sweep_hit.append(f"{pred}@{axis}≥{info[pred]['threshold']:g}")
        if gate_hit or sweep_hit:
            any_cross = True
        L.append(f"| {r['name']} | {r['n']:,} | {r['primary']} | "
                 f"{', '.join(gate_hit) if gate_hit else 'none'} | "
                 f"{', '.join(sweep_hit) if sweep_hit else 'none'} |")

    L.append("\n## Recommendation\n")
    if any_cross:
        L.append("At least one method shows a **robust** quality-dependent crossover (classification "
                 "surpasses the ratio once the call is well supported). See per-method REPORTs for the "
                 "exact gate/threshold and effect size. If the crossover reproduces across methods and "
                 "the effect is material, this is a **candidate for a final manuscript analysis** "
                 "(\"the classification is reliable exactly where it is well supported\"). If it appears "
                 "in only one method or with small effect, keep it **supplementary**.")
    else:
        L.append("**No robust crossover in any method.** Even restricted to high-confidence, "
                 "high-density pockets, the continuous evidence ratio remains at least as good a "
                 "predictor of functional outcome as the class label or `p_holo` (the Δρ CI never "
                 "clears 0 on the positive side). This *strengthens* the main finding: the apo/holo "
                 "classification is a thresholded summary of the evidence at **every** quality level, "
                 "so the axis is best reported as graded evidence (ratio/probability), not a hard "
                 "binary. Recommendation: report as a **supplementary** robustness check supporting the "
                 "main ratio-vs-class result — it answers the obvious reviewer question (\"does the "
                 "label do better when it's confident?\") with a clean negative.")
    L.append("\nSee `docking/REPORT.md`, `p2rank/REPORT.md`, `alphafill/REPORT.md` and the "
             "`fig_gates_*` / `fig_sweep_*` figures for the full tables, CIs, and n at every gate.")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
