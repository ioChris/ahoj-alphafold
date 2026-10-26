# -*- coding: utf-8 -*-
"""
Which facilitator is stronger -- holo/apo EVIDENCE or the apo/holo CLASS? A 2x2 quadrant read.
==============================================================================================

The combined predictor + docking figure carries a contour surface per method: detection
(harmonized DCA/TNS) or dockability, smoothed over

    x = evidence balance   log2((n_holo_avg + 1) / n_apo_avg)      ["the ratio"]
    y = p_holo             the classifier's holo posterior         ["the class axis"]

Read by eye those surfaces look like BOTH axes help, with the ratio usually dominant -- but the
IF-SitePred surface looks more diagonal, which is what prompted this analysis. This script turns
the eyeball read into a number by collapsing each surface onto the four quadrants defined by the
two natural cut points (ratio = 0, p_holo = 0.5) and reading the pooled success rate in each:

        p_holo >= 0.5   |   Q2  apo-leaning evidence   |   Q1  holo-leaning evidence  |
        (AF-Holo)       |       + AF-Holo class        |       + AF-Holo class        |
                        |       DISCORDANT             |       concordant             |
        ----------------+------------------------------+------------------------------+
        p_holo <  0.5   |   Q3  apo-leaning evidence   |   Q4  holo-leaning evidence  |
        (AF-Apo)        |       + AF-Apo class         |       + AF-Apo class         |
                        |       concordant             |       DISCORDANT             |
                              log2 ratio < 0                  log2 ratio > 0

Estimands (all in percentage points of success rate)
----------------------------------------------------
    E_ratio  = 1/2 * [(Q1 - Q2) + (Q4 - Q3)]      main effect of the evidence balance
    E_class  = 1/2 * [(Q1 - Q4) + (Q2 - Q3)]      main effect of the classification
    inter    = (Q1 - Q2) - (Q4 - Q3)              interaction (are the two effects additive?)
    span     = Q1 - Q3       = E_ratio + E_class  concordant corner-to-corner spread
    PREF     = E_ratio - E_class = **Q4 - Q2**    <- the answer to "which one is stronger"
    theta    = atan2(E_class, E_ratio) in degrees  0 = purely ratio-driven (vertical contours)
                                                  90 = purely class-driven (horizontal contours)
                                                  45 = the two contribute equally

`PREF = Q4 - Q2` is an identity of the 2x2, not an approximation, and it is the whole point of the
figure: the two DISCORDANT quadrants are exactly where the two facilitators disagree, so whichever
corner scores higher names the stronger facilitator. A reader can therefore verify the headline by
comparing two numbers on the tile -- no model, no regression.

What this is NOT
----------------
  * NOT a replacement for `analyze_lbs_ratio_vs_class.py`. That script ranks the two as *continuous*
    predictors (size-controlled partial Spearman + nested-CV AUC) and is the inferential statement.
    This one is a descriptive, dichotomized summary whose only job is to be readable.
  * NOT scale-free. Any 2x2 compares "one step along x" with "one step along y", and the size of a
    step is set by WHERE the axis is cut. The sign cut (ratio = 0, p_holo = 0.5) is the interpretable
    one -- both are the substantive boundaries of their own scale -- but it splits the ratio far more
    unevenly than it splits p_holo. `--schemes` therefore also runs `median` (each axis at its own
    median => equal marginals, the scale-free comparison) and `quartile` (top vs bottom quartile of
    each axis => equal marginals AND a matched extreme contrast). Report the primary scheme only if
    the other two agree with it; they are printed side by side for exactly that reason.
  * NOT causal, and (raw) NOT size-adjusted. Pocket SIZE dominates detection in this dataset
    (rho ~ +0.61 with union_rg) and the evidence ratio carries size confounding, so the raw
    quadrant rates mix "more holo evidence" with "bigger pocket". Every estimand is therefore also
    computed under DIRECT STANDARDIZATION to the method's own `union_rg` decile distribution
    -- twice, by two different estimators (`adjust = size_std`, `size_resid`; see `Grid.rates`),
    both of which hold size composition fixed across the four quadrants. Quote raw AND adjusted.

Methods covered: the 3 LBS predictors on the identical 86,946-pocket set, their `consensus` (the
arithmetic mean of the three -- a derived summary, NOT a 4th independent method), and docking on its
own nested G1 population (60,893 pockets, readout = "any cognate ligand docked", a DIFFERENT
quantity with a DIFFERENT denominator -- never pool it with the detectors).

The y axis is `pocket_p_holo >= 0.5`, which IS the class boundary (99.97% agreement with
`pocket_class`; the 29 disagreements are exact-0.5 rounding, and the 296 `tie` pockets fall on the
holo side by construction). Using p_holo rather than the label keeps ties in and keeps the LBS and
docking tables on the same rule.

CIs: cluster bootstrap by `site_uniprot` -- pockets within a protein are correlated. One resample of
UniProt clusters is shared by ALL methods and ALL schemes, so every cross-method and cross-scheme
difference is PAIRED. Standardization weights are held fixed at their full-sample values (a fixed
standard population), which is what makes the adjusted estimand well defined.

Inputs  : work_dir/lbs_generality/p2rank/lbs_p2rank_st100_merged_per_pocket.csv
          work_dir/lbs_generality/{fpocket,ifsitepred}/full/lbs_*_st100_merged_per_pocket.csv
          work_dir/docking/run3/analysis/pocket_level_table.csv          (ligand_group == 1)
Outputs : work_dir/lbs_generality/analysis_v1/tables/quadrants_*.csv
          work_dir/lbs_generality/analysis_v1/tables/quadrants_REPORT_auto.md

Base env (pandas + numpy):
    python 06_lbs_generality/analysis/analyze_ratio_vs_class_quadrants.py [--B 1000]
"""
from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd

for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# sibling reuse: the per-tool table resolver and the cluster-bootstrap index builder already exist
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from analyze_fig6a_quality_gates import _cluster_index, _tool_table  # noqa: E402

PRED = ["P2Rank", "fpocket", "IF-SitePred"]
# `consensus` = arithmetic mean of the 3 predictors. A derived summary, NOT a 4th independent
# method -- never count it as a separate test when reasoning about multiplicity.
LBS_METHODS = PRED + ["consensus"]
METHODS = LBS_METHODS + ["docking"]
SCHEMES = ["sign", "median", "quartile"]
# raw           = the observable rate; what a user actually meets
# size_std      = DIRECT standardization (counterfactual rate at the population size distribution)
# size_resid    = INDIRECT standardization (success net of E[success | size]); the adjustment the
#                 contour surfaces use, and the one that re-estimates inside each bootstrap replicate
ADJUSTMENTS = ["raw", "size_std", "size_resid"]
SIZE_COL_DEFAULT = "union_rg"
# `union_n_residues_found` is the alternative: a residue COUNT that reads no coordinates, so it is
# immune to "you computed that from the AF model", and it is the stronger size proxy
# (rho +0.64 with detection success vs +0.59 for union_rg). It therefore removes MORE.
SIZE_COLS = ("union_rg", "union_n_residues_found")
N_SIZE_BINS = 10

# Q1..Q4 in the Cartesian sense: Q1 top-right, then counter-clockwise.
#   index -> (holo-leaning evidence?, holo-side class?)
QUAD_DEF = {0: (True, True), 1: (False, True), 2: (False, False), 3: (True, False)}
QUAD_LABEL = {
    0: "Q1 holo-evidence & AF-Holo (concordant)",
    1: "Q2 apo-evidence & AF-Holo (discordant)",
    2: "Q3 apo-evidence & AF-Apo (concordant)",
    3: "Q4 holo-evidence & AF-Apo (discordant)",
}


# --------------------------------------------------------------------------- data
def load_long() -> pd.DataFrame:
    """One row per (pocket x method): the stacked frame every estimator below reads.

    Stacking is what lets a single resample of UniProt clusters serve all five methods, which is
    what makes the cross-method contrasts paired.
    """
    keep = ["site_uid", "site_uniprot", "pocket_class", "pocket_p_holo", "winning_p", "union_rg",
            "union_n_residues_found", "model_pocket_plddt", "n_apo_avg", "n_holo_avg",
            "log2_ratio_holo_to_apo_avg_plus1", "success_tns"]
    lbs = pd.read_csv(_tool_table("p2rank"), low_memory=False)[keep].rename(
        columns={"log2_ratio_holo_to_apo_avg_plus1": "log2_ratio", "success_tns": "P2Rank"})
    for tool, name in [("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")]:
        t = pd.read_csv(_tool_table(tool), low_memory=False)[["site_uid", "success_tns"]]
        lbs = lbs.merge(t.rename(columns={"success_tns": name}), on="site_uid", how="left")

    # the 3 predictors are scored on an identical set (no prediction is scored as a MISS, never
    # dropped) -- assert rather than assume, because the whole comparison rests on it
    scored = {k: set(lbs.loc[lbs[k].notna(), "site_uid"]) for k in PRED}
    assert scored["P2Rank"] == scored["fpocket"] == scored["IF-SitePred"], "predictor sets differ"
    lbs = lbs[lbs["P2Rank"].notna()].copy()
    lbs["consensus"] = lbs[PRED].mean(axis=1)

    frames = []
    for m in LBS_METHODS:
        f = lbs[["site_uid", "site_uniprot", "pocket_class", "pocket_p_holo", "log2_ratio",
                 "union_rg", "union_n_residues_found", "model_pocket_plddt", "winning_p",
                 "n_apo_avg", "n_holo_avg"]].copy()
        f["success"] = lbs[m].to_numpy(float)
        f["method"], f["pop"] = m, "lbs"
        frames.append(f)

    # docking: own nested population (G1 = a cognate ligand exists), own readout, own denominator
    dk = pd.read_csv(cfg.DOCKING_RUN3_DIR / "analysis" / "pocket_level_table.csv", low_memory=False)
    dk = dk[dk["ligand_group"] == 1].copy()
    # `is_holo` is NaN for the 296 tie pockets; the p_holo >= 0.5 rule keeps them, so do NOT drop
    dk = dk.merge(lbs[["site_uid", "site_uniprot", "pocket_class", "log2_ratio",
                       "union_n_residues_found", "model_pocket_plddt", "winning_p",
                       "n_apo_avg", "n_holo_avg"]], on="site_uid", how="left")
    assert dk["log2_ratio"].notna().all(), "docking pockets missing from the LBS base table"
    dk["success"] = (dk["frac_dockable"] > 0).astype(float)   # "any cognate ligand docked"
    dk["method"], dk["pop"] = "docking", "docking"
    frames.append(dk[frames[0].columns])

    long = pd.concat(frames, ignore_index=True)
    long["cls"] = long["pocket_class"].str.lower()
    return long


def add_axes(long: pd.DataFrame, scheme: str) -> np.ndarray:
    """Quadrant index 0..3 per row for one cut scheme; -1 = excluded (quartile middles).

    Cut points are computed WITHIN each population (`pop`), because docking's G1 subset has its own
    marginal distributions -- a median borrowed from the LBS population would not be its median.
    """
    q = np.full(len(long), -1, dtype=np.int64)
    for pop, idx in long.groupby("pop", sort=False).indices.items():
        r = long["log2_ratio"].to_numpy(float)[idx]
        p = long["pocket_p_holo"].to_numpy(float)[idx]
        if scheme == "sign":
            # the substantive boundaries: ratio 0 = balanced evidence, p_holo 0.5 = the class line.
            # ratio == 0 (2,738 pockets) falls on the apo-leaning side; reported in the census.
            xr, xp = r > 0, p >= 0.5
            keep = np.ones(len(idx), bool)
        elif scheme == "median":
            xr, xp = r > np.nanmedian(r), p > np.nanmedian(p)
            keep = np.isfinite(r) & np.isfinite(p)
        elif scheme == "quartile":
            rlo, rhi = np.nanpercentile(r, [25, 75])
            plo, phi = np.nanpercentile(p, [25, 75])
            xr, xp = r >= rhi, p >= phi
            keep = ((r <= rlo) | (r >= rhi)) & ((p <= plo) | (p >= phi))
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        qq = np.where(xp, np.where(xr, 0, 1), np.where(xr, 3, 2))
        q[idx] = np.where(keep, qq, -1)
    return q


def size_bins(long: pd.DataFrame, size_col: str = SIZE_COL_DEFAULT, k: int = N_SIZE_BINS) -> np.ndarray:
    """Per-population size-quantile index; -1 where the size variable is missing.

    Quantiles are taken WITHIN each population, because docking's nested G1 subset has its own size
    distribution. A row with no size lands in no stratum and drops out of the adjusted passes only
    (it still counts in `raw`). `union_n_residues_found` is integer-valued and ties collapse
    quantile edges, so duplicate edges are removed rather than producing empty strata.
    """
    b = np.full(len(long), -1, dtype=np.int64)
    for pop, idx in long.groupby("pop", sort=False).indices.items():
        v = long[size_col].to_numpy(float)[idx]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        edges = np.unique(np.nanpercentile(v[ok], np.linspace(0, 100, k + 1)[1:-1]))
        b[idx[ok]] = np.searchsorted(edges, v[ok], side="right")
    return b


# ------------------------------------------------------------------- estimator core
class Grid:
    """Pre-baked cell codes so one bootstrap replicate is a handful of `np.bincount` calls.

    cell = ((method * n_scheme + scheme) * 4 + quadrant) * K + size_bin

    Everything downstream -- crude rates, BOTH size adjustments, the derived estimands -- is a
    reduction of the same (method, scheme, quadrant, size_bin) count/total pair, which is why a
    bootstrap replicate costs two `bincount`s rather than a groupby.

    A row with quadrant -1 (excluded by the quartile scheme) or size_bin -1 (missing size) is coded
    -1 and dropped from every pass.
    """

    def __init__(self, long: pd.DataFrame, schemes: list[str],
                 size_col: str = SIZE_COL_DEFAULT, k: int = N_SIZE_BINS):
        self.schemes = schemes
        self.methods = METHODS
        self.size_col = size_col
        mi = long["method"].map({m: i for i, m in enumerate(METHODS)}).to_numpy()
        zb = size_bins(long, size_col, k)
        self.K = int(zb.max()) + 1
        self.n_cells = len(METHODS) * len(schemes) * 4 * self.K
        code = np.full((len(schemes), len(long)), -1, dtype=np.int64)
        for si, s in enumerate(schemes):
            q = add_axes(long, s)
            ok = (q >= 0) & (zb >= 0)
            code[si, ok] = ((mi[ok] * len(schemes) + si) * 4 + q[ok]) * self.K + zb[ok]
        self.code = code.ravel()                                   # scheme-major, rows repeated
        self.succ = np.tile(long["success"].to_numpy(float), len(schemes))
        self.n_rows = len(long)                                    # stride of the scheme-major stack
        # DIRECT standardization needs a standard population. Weights are each method's own
        # size-stratum shares, held FIXED at their full-sample values (never resampled) -- that is
        # what makes the adjusted estimand well defined across bootstrap replicates.
        w = np.zeros((len(METHODS), self.K))
        for i, m in enumerate(METHODS):
            sel = (long["method"] == m).to_numpy() & (zb >= 0)
            w[i] = np.bincount(zb[sel], minlength=self.K)
        self.W = w / np.clip(w.sum(axis=1, keepdims=True), 1, None)

    def rates(self, row_take: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """(method, scheme, quadrant) rates in %, crude and under both size adjustments."""
        if row_take is None:
            code, succ = self.code, self.succ
        else:
            # expand a row-level bootstrap index across the scheme-major stacking
            take = np.concatenate([row_take + si * self.n_rows for si in range(len(self.schemes))])
            code, succ = self.code[take], self.succ[take]
        keep = code >= 0
        code, succ = code[keep], succ[keep]
        cnt = np.bincount(code, minlength=self.n_cells).astype(float)
        tot = np.bincount(code, weights=succ, minlength=self.n_cells)
        shape = (len(METHODS), len(self.schemes), 4, self.K)
        cnt, tot = cnt.reshape(shape), tot.reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_bin = 100.0 * tot / cnt                      # rate within (quadrant, size stratum)
        crude_n = cnt.sum(axis=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            crude = 100.0 * tot.sum(axis=3) / crude_n

        # --- DIRECT standardization: the counterfactual rate this quadrant would show if its
        # pockets had the method's overall size distribution. Weights renormalized over the strata
        # the quadrant actually populates.
        W = self.W[:, None, None, :]
        wm = np.where(np.isfinite(per_bin), W, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            direct = np.nansum(np.where(np.isfinite(per_bin), per_bin, 0.0) * wm, axis=3) / wm.sum(axis=3)

        # --- INDIRECT standardization: success net of what size alone predicts, i.e.
        #        grand + mean_quadrant( s - E[s | size] ),
        # which reduces to  grand + SUM_k p(k|quadrant) * (rate[q,k] - E[k]).
        # Unlike `direct` this re-estimates E[s|size] inside every bootstrap replicate, and it uses
        # the quadrant's OWN stratum weights, so it stays defined where a quadrant misses a stratum
        # entirely. It is the adjustment the contour surfaces use, because at contour resolution
        # direct standardization runs out of pockets per (cell x stratum).
        n_size = cnt.sum(axis=2)                                   # (method, scheme, size_bin)
        t_size = tot.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_size = 100.0 * t_size / n_size                     # E[s | size] for this sample
            grand = 100.0 * tot.sum(axis=(2, 3)) / cnt.sum(axis=(2, 3))
            share = cnt / np.where(crude_n[..., None] > 0, crude_n[..., None], np.nan)
            excess = np.where(np.isfinite(per_bin), per_bin - exp_size[:, :, None, :], 0.0)
            indirect = grand[..., None] + np.nansum(share * excess, axis=3)
        return {"raw": crude, "size_std": direct, "size_resid": indirect, "n": crude_n}


def estimands(q: np.ndarray) -> dict[str, np.ndarray]:
    """q[..., 4] -> the derived quantities. Works on point estimates and bootstrap stacks alike."""
    Q1, Q2, Q3, Q4 = (q[..., i] for i in range(4))
    e_ratio = 0.5 * ((Q1 - Q2) + (Q4 - Q3))
    e_class = 0.5 * ((Q1 - Q4) + (Q2 - Q3))
    return {
        "Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4,
        "E_ratio": e_ratio, "E_class": e_class,
        "interaction": (Q1 - Q2) - (Q4 - Q3),
        "span": Q1 - Q3,
        "PREF": e_ratio - e_class,            # == Q4 - Q2, exactly
        "theta_deg": np.degrees(np.arctan2(e_class, e_ratio)),
        # share of the total (additive) effect carried by the class axis; NaN if both effects ~0
        "class_share": np.where(np.abs(e_ratio) + np.abs(e_class) > 1e-9,
                                e_class / (np.abs(e_ratio) + np.abs(e_class)), np.nan),
    }


CI_KEYS = ["Q1", "Q2", "Q3", "Q4", "E_ratio", "E_class", "interaction", "span", "PREF",
           "theta_deg", "class_share"]


def _ragged(flat: np.ndarray, starts: np.ndarray, ends: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Vectorized `np.concatenate([flat[starts[s]:ends[s]] for s in sel])`.

    The list-comprehension form is the obvious way to gather bootstrap clusters and it is what the
    sibling scripts use; here it would run ~10k clusters x B replicates in pure Python, so it is
    written as a single fancy-index instead (same result, ~100x faster).
    """
    cnt = (ends[sel] - starts[sel]).astype(np.int64)
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, np.int64)
    cum = np.cumsum(cnt)
    idx = np.arange(total) - np.repeat(cum - cnt, cnt) + np.repeat(starts[sel], cnt)
    return flat[idx]


def bootstrap(long: pd.DataFrame, grid: Grid, B: int, seed: int = 20260820):
    """Cluster bootstrap by UniProt, shared across every method / scheme / adjustment."""
    rng = np.random.default_rng(seed)
    # `long` is stacked (one row per pocket x method), so resample at the UniProt level on the
    # PROTEIN -> POCKET -> ROW hierarchy: cluster index over unique pockets, then a CSR-style map
    # from each pocket to its (up to 5) stacked rows.
    pocket = long.drop_duplicates("site_uid").reset_index(drop=True)
    order, starts, ends, nC = _cluster_index(pocket)
    pid = pd.Categorical(long["site_uid"], categories=pocket["site_uid"]).codes.astype(np.int64)
    srt = np.argsort(pid, kind="stable")
    flat = srt
    off = np.searchsorted(pid[srt], np.arange(len(pocket) + 1), side="left")

    out = {a: {k: np.full((B, len(METHODS), len(grid.schemes)), np.nan) for k in CI_KEYS}
           for a in ADJUSTMENTS}
    for b in range(B):
        pick = rng.integers(0, nC, nC)
        pockets = _ragged(order, starts, ends, pick)
        rows = _ragged(flat, off[:-1], off[1:], pockets)
        r = grid.rates(rows)
        for a in ADJUSTMENTS:
            est = estimands(r[a])
            for k in CI_KEYS:
                out[a][k][b] = est[k]
    return out, nC


# --------------------------------------------------------------------------- tables
def quadrant_table(grid: Grid, obs, boot) -> pd.DataFrame:
    rows = []
    for mi, m in enumerate(METHODS):
        for si, s in enumerate(grid.schemes):
            for a in ADJUSTMENTS:
                est = estimands(obs[a])
                rec = {"method": m, "scheme": s, "adjust": a,
                       "size_col": grid.size_col if a != "raw" else "",
                       "n_size_bins": grid.K if a != "raw" else 0,
                       "population": "docking G1" if m == "docking" else "LBS shared",
                       "readout": "any cognate ligand docked" if m == "docking"
                                  else "harmonized DCA/TNS"}
                for qi in range(4):
                    rec[f"rate_Q{qi + 1}"] = est[f"Q{qi + 1}"][mi, si]
                    rec[f"n_Q{qi + 1}"] = int(obs["n"][mi, si, qi])
                for k in CI_KEYS[4:]:
                    v = est[k][mi, si]
                    rec[k] = v
                    if boot is not None:
                        d = boot[a][k][:, mi, si]
                        rec[f"{k}_lo"] = np.nanpercentile(d, 2.5)
                        rec[f"{k}_hi"] = np.nanpercentile(d, 97.5)
                if boot is not None:
                    d = boot[a]["PREF"][:, mi, si]
                    # two-sided bootstrap p for "the two facilitators are equally strong".
                    # A bootstrap p cannot resolve below 1/B, so censor there and read a value
                    # equal to 1/B as "p < 1/B" -- never report a bootstrap p of exactly 0.
                    rec["PREF_p_two_sided"] = max(
                        2 * min(np.nanmean(d > 0), np.nanmean(d < 0)), 1.0 / len(d))
                    rec["winner"] = ("ratio" if rec["PREF"] > 0 else "class") if (
                        rec.get("PREF_lo", 0) * rec.get("PREF_hi", 0) > 0) else "tie (CI spans 0)"
                rec["n_total"] = int(obs["n"][mi, si].sum())
                rows.append(rec)
    return pd.DataFrame(rows)


def method_contrasts(grid: Grid, obs, boot) -> pd.DataFrame:
    """Paired differences BETWEEN methods -- 'does IF-SitePred lean on the class more than P2Rank?'

    Paired because every method is recomputed on the same resampled UniProt clusters. For the LBS
    methods the pockets are literally identical; docking is a nested subset, so a docking-vs-LBS
    contrast additionally carries the population difference -- flagged in `comparable`.
    """
    rows = []
    for si, s in enumerate(grid.schemes):
        for a in ADJUSTMENTS:
            est = estimands(obs[a])
            for i, mi_ in enumerate(METHODS):
                for j, mj in enumerate(METHODS):
                    if j <= i:
                        continue
                    for k in ["PREF", "class_share", "theta_deg", "E_ratio", "E_class"]:
                        d = est[k][i, si] - est[k][j, si]
                        rec = {"scheme": s, "adjust": a, "metric": k,
                               "method_a": mi_, "method_b": mj, "delta": d,
                               "comparable": "same pockets" if "docking" not in (mi_, mj)
                                             else "nested population"}
                        if boot is not None:
                            bd = boot[a][k][:, i, si] - boot[a][k][:, j, si]
                            rec["lo"] = np.nanpercentile(bd, 2.5)
                            rec["hi"] = np.nanpercentile(bd, 97.5)
                            rec["p_two_sided"] = max(                      # censored at 1/B
                                2 * min(np.nanmean(bd > 0), np.nanmean(bd < 0)), 1.0 / len(bd))
                        rows.append(rec)
    return pd.DataFrame(rows)


def composition(long: pd.DataFrame, scheme: str = "sign") -> pd.DataFrame:
    """What each quadrant is MADE OF -- the table that explains the raw/adjusted difference.

    If Q4 (holo evidence, AF-Apo) simply holds bigger pockets than Q2, the raw PREF overstates the
    ratio. This is where a reader checks that, and it is why the adjusted passes exist.
    """
    q = add_axes(long, scheme)
    rows = []
    for pop in ["lbs", "docking"]:
        sub = long[(long["pop"] == pop) & (long["method"].isin(["P2Rank", "docking"]))].copy()
        sub["q"] = q[(long["pop"] == pop) & (long["method"].isin(["P2Rank", "docking"]))]
        sub = sub[sub["q"] >= 0]
        if sub.empty:
            continue
        for qi, g in sub.groupby("q"):
            rows.append({
                "population": pop, "scheme": scheme, "quadrant": f"Q{qi + 1}",
                "label": QUAD_LABEL[qi], "n": len(g), "n_uniprot": g["site_uniprot"].nunique(),
                "pct_of_population": 100 * len(g) / len(sub),
                "mean_union_rg": g["union_rg"].mean(), "median_union_rg": g["union_rg"].median(),
                "mean_union_residues": g["union_n_residues_found"].mean(),
                "mean_plddt": g["model_pocket_plddt"].mean(),
                "mean_log2_ratio": g["log2_ratio"].mean(),
                "mean_p_holo": g["pocket_p_holo"].mean(),
                "mean_winning_p": g["winning_p"].mean(),
                "mean_n_apo_avg": g["n_apo_avg"].mean(), "mean_n_holo_avg": g["n_holo_avg"].mean(),
                "pct_class_holo": 100 * g["cls"].eq("holo").mean(),
                "pct_class_tie": 100 * g["cls"].eq("tie").mean(),
            })
    return pd.DataFrame(rows)


def census(long: pd.DataFrame) -> pd.DataFrame:
    """Population bookkeeping + the boundary cases the sign cut has to make a call on."""
    rows = []
    for pop, g in long[long["method"].isin(["P2Rank", "docking"])].groupby("pop"):
        r, p = g["log2_ratio"].to_numpy(float), g["pocket_p_holo"].to_numpy(float)
        rows.append({
            "population": pop, "n_pockets": len(g), "n_uniprot": g["site_uniprot"].nunique(),
            "n_ratio_gt0": int((r > 0).sum()), "n_ratio_eq0": int((r == 0).sum()),
            "n_ratio_lt0": int((r < 0).sum()),
            "n_p_holo_ge_0p5": int((p >= 0.5).sum()),
            "median_log2_ratio": float(np.nanmedian(r)), "median_p_holo": float(np.nanmedian(p)),
            "n_class_apo": int(g["cls"].eq("apo").sum()), "n_class_holo": int(g["cls"].eq("holo").sum()),
            "n_class_tie": int(g["cls"].eq("tie").sum()),
            "pct_class_agrees_with_p_holo_cut": 100 * float(
                ((p >= 0.5) == g["cls"].isin(["holo", "tie"])).mean()),
            "spearman_ratio_vs_p_holo": float(pd.Series(r).corr(pd.Series(p), method="spearman")),
            "n_union_rg_missing": int(g["union_rg"].isna().sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
def write_report(path: _Path, qt: pd.DataFrame, cens: pd.DataFrame, comp: pd.DataFrame,
                 B: int, nC: int) -> None:
    prim = qt[(qt.scheme == "sign")].set_index(["method", "adjust"])
    L = ["# Ratio vs class: the 2x2 quadrant read (auto-generated)", "",
         f"Cluster bootstrap: B={B} over {nC:,} UniProt clusters, shared across methods/schemes.", "",
         "`PREF = E_ratio - E_class = Q4 - Q2` (pp). PREF > 0 => the evidence balance is the "
         "stronger facilitator; PREF < 0 => the classification is.", "",
         "## Primary scheme (`sign`: ratio = 0, p_holo = 0.5)", "",
         "| method | adj | Q1 | Q2 | Q3 | Q4 | E_ratio | E_class | PREF [95% CI] | theta | winner |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for a in ADJUSTMENTS:
            r = prim.loc[(m, a)]
            # --B 0 skips the bootstrap, so the CI / winner columns may not exist at all
            ci = (f"[{r.PREF_lo:+.1f}, {r.PREF_hi:+.1f}]" if "PREF_lo" in r.index else "(no bootstrap)")
            win = r.winner if "winner" in r.index else "-"
            L.append(f"| {m} | {a} | {r.rate_Q1:.1f} | {r.rate_Q2:.1f} | {r.rate_Q3:.1f} | "
                     f"{r.rate_Q4:.1f} | {r.E_ratio:+.1f} | {r.E_class:+.1f} | "
                     f"{r.PREF:+.1f} {ci} | {r.theta_deg:.0f} deg | {win} |")
    L += ["", "## Cut-scheme robustness (PREF, pp)", "",
          "| method | adj | " + " | ".join(SCHEMES) + " |", "|---|---|" + "---|" * len(SCHEMES)]
    for m in METHODS:
        for a in ADJUSTMENTS:
            cells = []
            for s in SCHEMES:
                sel = qt[(qt.method == m) & (qt.adjust == a) & (qt.scheme == s)]
                cells.append(f"{sel.PREF.iloc[0]:+.1f}" if len(sel) else "-")
            L.append(f"| {m} | {a} | " + " | ".join(cells) + " |")
    L += ["", "## Population census", "", cens.to_markdown(index=False, floatfmt=".3f"),
          "", "## Quadrant composition (`sign` scheme)", "",
          comp.to_markdown(index=False, floatfmt=".2f"), ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=None,
                    help="default: work_dir/lbs_generality/analysis_v1/tables")
    ap.add_argument("--B", type=int, default=1000, help="cluster-bootstrap replicates (0 = skip)")
    ap.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES)
    ap.add_argument("--size_col", default=SIZE_COL_DEFAULT, choices=SIZE_COLS,
                    help="size variable both adjustments condition on (default: union_rg)")
    ap.add_argument("--n_size_bins", type=int, default=N_SIZE_BINS,
                    help="quantile strata of --size_col (default 10; ties collapse edges)")
    ap.add_argument("--suffix", default="",
                    help="appended to every output filename, e.g. '_nres' for a size-proxy variant")
    a = ap.parse_args()

    cfg.require("WORK_DIR")
    outdir = _Path(a.outdir) if a.outdir else (cfg.LBS_ANALYSIS_DIR / "tables")
    outdir.mkdir(parents=True, exist_ok=True)

    long = load_long()
    npk = long.groupby("pop")["site_uid"].nunique()
    print(f"pockets: LBS {npk.get('lbs', 0):,} x {len(LBS_METHODS)} methods | "
          f"docking G1 {npk.get('docking', 0):,} | long rows {len(long):,}")

    grid = Grid(long, list(a.schemes), size_col=a.size_col, k=a.n_size_bins)
    print(f"size adjustment on {a.size_col} in {grid.K} strata "
          f"(size_std = direct standardization, size_resid = success net of E[success|size])")
    obs = grid.rates()
    boot, nC = (None, _cluster_index(long.drop_duplicates('site_uid'))[3])
    if a.B > 0:
        print(f"cluster bootstrap B={a.B} over {nC:,} UniProt clusters ...")
        boot, nC = bootstrap(long, grid, a.B)

    sfx = a.suffix
    qt = quadrant_table(grid, obs, boot)
    qt.to_csv(outdir / f"quadrants_summary{sfx}.csv", index=False)
    cens = census(long)
    cens.to_csv(outdir / f"quadrants_census{sfx}.csv", index=False)
    comp = pd.concat([composition(long, s) for s in a.schemes], ignore_index=True)
    comp.to_csv(outdir / f"quadrants_composition{sfx}.csv", index=False)
    mc = method_contrasts(grid, obs, boot)
    mc.to_csv(outdir / f"quadrants_method_contrasts{sfx}.csv", index=False)
    write_report(outdir / f"quadrants_REPORT_auto{sfx}.md", qt, cens, comp[comp.scheme == "sign"],
                 a.B, nC)

    pd.set_option("display.width", 220)
    show = ["method", "adjust", "rate_Q1", "rate_Q2", "rate_Q3", "rate_Q4",
            "E_ratio", "E_class", "PREF", "PREF_lo", "PREF_hi", "theta_deg"]
    show = [c for c in show if c in qt.columns]
    print("\n=== PRIMARY (sign cut: ratio = 0, p_holo = 0.5) ===")
    print(qt[qt.scheme == "sign"][show].round(2).to_string(index=False))
    print("\n=== PREF by cut scheme (pp; >0 = evidence ratio is the stronger facilitator) ===")
    print(qt.pivot_table(index=["method", "adjust"], columns="scheme", values="PREF")
            .reindex(METHODS, level=0).round(2).to_string())
    print("\nwrote ->", outdir)


if __name__ == "__main__":
    main()
