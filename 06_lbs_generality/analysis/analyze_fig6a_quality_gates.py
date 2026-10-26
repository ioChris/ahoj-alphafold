# -*- coding: utf-8 -*-
"""
Fig 6A under density / separation quality gates: does the winning-p divergence strengthen?
==========================================================================================

Main Figure 6 Panel A plots harmonized DCA/TNS detection success against the
winning-probability verbal bin (very_low..very_high), split by AlphaFold pocket class
(AF-Apo / All / AF-Holo), for the 3 LBS predictors. The headline tendency is a
DIVERGENCE: as `winning_p` rises, AF-Apo success falls and AF-Holo success rises.

This script asks whether that divergence strengthens or weakens once we additionally
require moderate+ (or any other cut of) **pocket density** and/or **separation** --
the two other verbal confidence descriptors carried through the manuscript.

Estimand
--------
    gap[bin] = holo_rate[bin] - apo_rate[bin]
    D        = gap[very_high] - gap[very_low]     <- "the tendency" / divergence amplitude
    dD       = D(gate) - D(ungated)               <- strengthens (+) / weakens (-)

D is a contrast between FOUR cells (very_low/very_high x apo/holo). It therefore responds
only to selection on SUCCESS *within* those cells; a gate's correlation with `winning_p` is
largely irrelevant to it by construction. What separates the two gates is the CLASS-ASYMMETRY
of that within-cell selection -- see `within_cell_selection()`, the mechanism table.

CIs: cluster bootstrap by `site_uniprot` (pockets within a protein are correlated). The gated
sets are NESTED in the ungated set, so dD uses a PAIRED bootstrap -- one resample of UniProt
clusters, both D's recomputed on it -> paired difference.

Gates use the verbal level columns already present in the per-pocket tables, produced by
`03_alphafill/ubs/confidence_labeling.py` (`support_level` for density/separation,
`probability_level` for winning_p). "moderate+" = {moderate, high, very_high} = the project's
own `confident_probs` constant. Verified fresh: 0/86,946 label mismatches.

CAUTION on interpretation (learned the hard way; see the module docstring of the SI caption):
  * `pocket_density_pair` is a peak-normalized KDE *concentration* (parse-ahoj `kde_core.py`),
    prior-free -- low density means DIFFUSE evidence, NOT little evidence. The pockets the
    density gate drops are structure-RICH.
  * Never quote the POOLED Spearman(winning_p, density) = -0.141: it is a class-averaging
    dilution of -0.267 (apo) / -0.094 (holo). Report within-class values.
  * **SEPARATION IS A SITE-SIZE PROXY -- do not interpret it as a detectability determinant.**
    AF-Apo success rises across separation bins in raw form (43.0 -> 53.3%, +10.3 pp; +20.0 pp in the
    wp=very_high subset), which invites "distinct apo/holo states make pockets more detectable". It
    does not survive: high-separation sites are LARGER (union_rg 6.28 -> 8.12 A; union residues
    8.6 -> 14.2) and size dominates detection (rho +0.610 union_rg / +0.650 residue count, vs
    separation's own +0.075). Standardized to the site-size distribution the association is ABOLISHED
    for P2Rank (+1.3 pp [-2.6, +4.8]) and IF-SitePred (+0.2 [-4.3, +4.9]) and negative only for
    fpocket (-15.6 [-18.9, -12.3]); with protein AND size held fixed it is +0.51 pp [-0.05, +1.06].
    The same attenuation hits the AF-Holo arm (+10.6 -> +1.9), so nothing here is apo-specific.
    Size is a CONFOUNDER, not a mediator: at ST_100 the union residue set IS the intersection set
    (verified identical in 100.000% of 82,130 rows, max |union_rg - intersection_rg| = 0.0), it is
    fixed by AHoJ-DB clustering BEFORE AlphaFold is involved, and the reversal reproduces on
    `union_n_residues_found` -- a residue COUNT that reads no coordinates at all.
  * Roughly a third of the headline divergence D is size composition: consensus D 33.61 -> 21.15
    under rg standardization (P2Rank 28.69 -> 15.51; fpocket 30.52 -> 18.14; IF-SitePred
    41.63 -> 29.80). It stays large and positive -- the result holds -- but report both.

Inputs  : work_dir/lbs_generality/p2rank/lbs_p2rank_st100_merged_per_pocket.csv
          work_dir/lbs_generality/{fpocket,ifsitepred}/full/lbs_*_st100_merged_per_pocket.csv
Outputs : work_dir/lbs_generality/analysis_v1/tables/fig6a_gates_*.csv

Base env (pandas + numpy):
    python 06_lbs_generality/analysis/analyze_fig6a_quality_gates.py [--B 1000]
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

PRED = ["P2Rank", "fpocket", "IF-SitePred"]
# NOTE: `consensus` is the arithmetic MEAN of the 3 predictors -- a derived summary, NOT an
# independent 4th method (bootstrap corr ~ +0.81 with its components). Never count it as a
# 4th test when reasoning about multiplicity.
METHODS = PRED + ["consensus"]
LEVELS5 = ["very_low", "low", "moderate", "high", "very_high"]
MODPLUS = {"moderate", "high", "very_high"}
HIGHPLUS = {"high", "very_high"}
# The manuscript gate family (moderate+ = the project's `confident_probs` constant). The bootstrap
# in run_gates() reports on THESE; the threshold ladder covers every other cut.
GATES = ["all", "dens_mod+", "sep_mod+", "both_mod+"]
# The stricter family, for the side-by-side reference grid. Same structure, cut one rung higher.
GATES_HIGH = ["all", "dens_high+", "sep_high+", "both_high+"]
GATES_ALL = GATES + GATES_HIGH[1:]          # 'all' is shared; don't emit it twice
CLASSES = ["apo", "holo", "all"]
RUNGS = {
    "low+": {"low", "moderate", "high", "very_high"},
    "moderate+": MODPLUS,
    "high+": {"high", "very_high"},
    "very_high+": {"very_high"},
}


def _tool_table(tool: str) -> _Path:
    """Resolve a scored per-pocket table (layout differs slightly per tool)."""
    cands = [
        cfg.lbs_tool_dir(tool) / f"lbs_{tool}_st100_merged_per_pocket.csv",
        cfg.LBS_DIR / tool / f"lbs_{tool}_st100_merged_per_pocket.csv",
        cfg.LBS_DIR / tool / cfg.LBS_FULL_SUBDIR / f"lbs_{tool}_st100_merged_per_pocket.csv",
        cfg.LBS_DIR / tool / "analysis_v1" / "tables" / f"lbs_{tool}_st100_merged_per_pocket.csv",
    ]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"no scored table for {tool!r}; tried:\n  " + "\n  ".join(map(str, cands)))


def load_base() -> pd.DataFrame:
    keep = [
        "site_uid", "site_uniprot", "pocket_class", "pocket_p_holo", "winning_p",
        "pocket_density_pair", "pocket_separation_pair", "union_rg",
        # size + model-quality controls: `union_n_residues_found` is a residue COUNT (no coordinates
        # read) and is therefore the size proxy immune to any "computed from the AF model" objection;
        # keep both so downstream size checks never have to re-merge from the p2rank table by hand
        "union_n_residues_found", "model_pocket_plddt",
        "n_apo_avg", "n_holo_avg", "log2_ratio_holo_to_apo_avg_plus1",
        "prob_level_db", "density_level_db", "separation_level_db", "success_tns",
    ]
    base = pd.read_csv(_tool_table("p2rank"), low_memory=False)[keep].rename(
        columns={"log2_ratio_holo_to_apo_avg_plus1": "log2_ratio", "success_tns": "P2Rank"})

    for tool, name in [("fpocket", "fpocket"), ("ifsitepred", "IF-SitePred")]:
        t = pd.read_csv(_tool_table(tool), low_memory=False)[["site_uid", "success_tns"]]
        base = base.merge(t.rename(columns={"success_tns": name}), on="site_uid", how="left")

    # the 3 predictors are scored on an identical set (a tool with no prediction is scored as a
    # MISS, never dropped) -- assert rather than assume
    scored = {k: set(base.loc[base[k].notna(), "site_uid"]) for k in PRED}
    assert scored["P2Rank"] == scored["fpocket"] == scored["IF-SitePred"], "predictor sets differ"

    base = base[base["P2Rank"].notna()].copy()
    base["consensus"] = base[PRED].mean(axis=1)
    base["cls"] = base["pocket_class"].str.lower()
    base["g_dens"] = base["density_level_db"].isin(MODPLUS)
    base["g_sep"] = base["separation_level_db"].isin(MODPLUS)
    base["g_both"] = base["g_dens"] & base["g_sep"]
    base["g_dens_high"] = base["density_level_db"].isin(HIGHPLUS)
    base["g_sep_high"] = base["separation_level_db"].isin(HIGHPLUS)
    base["g_both_high"] = base["g_dens_high"] & base["g_sep_high"]
    return base


def gate_mask(df: pd.DataFrame, gate: str) -> np.ndarray:
    if gate == "all":
        return np.ones(len(df), bool)
    return {"dens_mod+": df["g_dens"], "sep_mod+": df["g_sep"], "both_mod+": df["g_both"],
            "dens_high+": df["g_dens_high"], "sep_high+": df["g_sep_high"],
            "both_high+": df["g_both_high"]}[gate].to_numpy()


# ---------------------------------------------------------------------- point estimates
def bin_rates(base: pd.DataFrame, bin_col: str = "prob_level_db") -> pd.DataFrame:
    """Per-bin rates for BOTH gate families (moderate+ and high+); the reference grids draw from this."""
    rows = []
    for gate in GATES_ALL:
        sub = base[gate_mask(base, gate)]
        for m in METHODS:
            for lv in LEVELS5:
                s = sub[sub[bin_col] == lv]
                rec = {"gate": gate, "method": m, "bin": lv, "n_bin": len(s)}
                for cl in CLASSES:
                    ss = s if cl == "all" else s[s["cls"] == cl]
                    rec[f"rate_{cl}"] = 100 * ss[m].mean() if len(ss) else np.nan
                    rec[f"n_{cl}"] = len(ss)
                rec["gap"] = rec["rate_holo"] - rec["rate_apo"]
                rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- bootstrap machinery
def _cluster_index(base: pd.DataFrame):
    codes, _ = pd.factorize(base["site_uniprot"])
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    starts = np.searchsorted(cs, np.arange(cs.max() + 1), side="left")
    ends = np.searchsorted(cs, np.arange(cs.max() + 1), side="right")
    return order, starts, ends, len(starts)


def _cells(base: pd.DataFrame) -> np.ndarray:
    """cell = prob_bin*3 + class_code (apo=0, holo=1, tie=2) -> very_low apo/holo = 0/1;
    very_high apo/holo = 12/13."""
    b = base["prob_level_db"].map({lv: i for i, lv in enumerate(LEVELS5)}).to_numpy()
    c = base["cls"].map({"apo": 0, "holo": 1, "tie": 2}).to_numpy()
    return (b * 3 + c).astype(np.int64)


def _D_grid(succ, cell, masks):
    """D per (mask, method) for one already-resampled sample."""
    out = {}
    for gi, mk in enumerate(masks):
        c = cell[mk]
        cnt = np.bincount(c, minlength=15)
        for mi in range(len(METHODS)):
            tot = np.bincount(c, weights=succ[mi][mk], minlength=15)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = 100.0 * tot / cnt
            out[(gi, mi)] = ((r[13] - r[12]) - (r[1] - r[0]),   # D
                             r[12] - r[0],                       # apo slope (very_high - very_low)
                             r[13] - r[1])                       # holo slope
    return out


def _paired_boot(base, masks, B, seed):
    """Paired cluster bootstrap of dD (and of the apo/holo slope deltas) vs masks[0]=ungated."""
    rng = np.random.default_rng(seed)
    cell = _cells(base)
    succ = [base[m].to_numpy(float) for m in METHODS]
    obs = _D_grid(succ, cell, masks)
    order, starts, ends, nC = _cluster_index(base)
    keys = [(gi, mi) for gi in range(len(masks)) for mi in range(len(METHODS))]
    bD = {k: np.full(B, np.nan) for k in keys}
    bdD = {k: np.full(B, np.nan) for k in keys}
    bda = {k: np.full(B, np.nan) for k in keys}
    bdh = {k: np.full(B, np.nan) for k in keys}
    for b in range(B):
        pick = rng.integers(0, nC, nC)
        idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
        r = _D_grid([s[idx] for s in succ], cell[idx], [mk[idx] for mk in masks])
        for k in keys:
            bD[k][b] = r[k][0]
            bdD[k][b] = r[k][0] - r[(0, k[1])][0]
            bda[k][b] = r[k][1] - r[(0, k[1])][1]
            bdh[k][b] = r[k][2] - r[(0, k[1])][2]
    return obs, bD, bdD, bda, bdh


def run_gates(base: pd.DataFrame, B: int, seed: int = 12345) -> pd.DataFrame:
    masks = [gate_mask(base, g) for g in GATES]
    obs, bD, bdD, bda, bdh = _paired_boot(base, masks, B, seed)
    rows = []
    for gi, g in enumerate(GATES):
        for mi, m in enumerate(METHODS):
            D, aslope, hslope = obs[(gi, mi)]
            rows.append({
                "gate": g, "method": m, "n_kept": int(masks[gi].sum()),
                "D_divergence": D,
                "D_lo": np.nanpercentile(bD[(gi, mi)], 2.5),
                "D_hi": np.nanpercentile(bD[(gi, mi)], 97.5),
                "apo_slope": aslope, "holo_slope": hslope,
                "dD_vs_ungated": D - obs[(0, mi)][0],
                "dD_lo": np.nanpercentile(bdD[(gi, mi)], 2.5),
                "dD_hi": np.nanpercentile(bdD[(gi, mi)], 97.5),
                # NOTE: a two-sided p below 1/B reads as 0 -- report as p < 1/B, never as 0/1
                "dD_p_two_sided": 2 * min(np.nanmean(bdD[(gi, mi)] > 0), np.nanmean(bdD[(gi, mi)] < 0)),
                "n_clusters": _cluster_index(base)[3], "B": B,
            })
    return pd.DataFrame(rows)


def threshold_ladder(base: pd.DataFrame, B: int = 400, seed: int = 4242) -> pd.DataFrame:
    """dD (and its apo/holo decomposition) as a function of WHERE the support cut is placed.

    Answers two things: (a) is `moderate+` a cherry-pick? (b) does raising the cut strengthen
    the trend, and does it do so *through the apo arm*?
    """
    specs = [("all", "ungated", np.ones(len(base), bool))]
    for prop, col in [("density", "density_level_db"), ("separation", "separation_level_db")]:
        for rung, keep in RUNGS.items():
            specs.append((prop, rung, base[col].isin(keep).to_numpy()))
    masks = [s[2] for s in specs]
    obs, bD, bdD, bda, bdh = _paired_boot(base, masks, B, seed)

    # endpoint rates per class, for the apo-vs-holo decomposition
    rows = []
    for gi, (prop, rung, mk) in enumerate(specs):
        sub = base[mk]
        for mi, m in enumerate(METHODS):
            D, aslope, hslope = obs[(gi, mi)]
            rec = {
                "property": prop, "rung": rung, "method": m,
                "n_kept": int(mk.sum()), "pct_kept": 100 * mk.sum() / len(base),
                "D": D, "dD_vs_ungated": D - obs[(0, mi)][0],
                "dD_lo": np.nanpercentile(bdD[(gi, mi)], 2.5),
                "dD_hi": np.nanpercentile(bdD[(gi, mi)], 97.5),
                # how much of dD is the APO arm moving vs the HOLO arm moving
                "d_apo_slope": aslope - obs[(0, mi)][1],
                "d_apo_lo": np.nanpercentile(bda[(gi, mi)], 2.5),
                "d_apo_hi": np.nanpercentile(bda[(gi, mi)], 97.5),
                "d_holo_slope": hslope - obs[(0, mi)][2],
                "d_holo_lo": np.nanpercentile(bdh[(gi, mi)], 2.5),
                "d_holo_hi": np.nanpercentile(bdh[(gi, mi)], 97.5),
            }
            for lv in ["very_low", "very_high"]:
                for cl in ["apo", "holo"]:
                    s = sub[(sub["prob_level_db"] == lv) & (sub["cls"] == cl)]
                    rec[f"rate_{lv}_{cl}"] = 100 * s[m].mean() if len(s) else np.nan
                    rec[f"n_{lv}_{cl}"] = len(s)
            rows.append(rec)
    return pd.DataFrame(rows)


def divergence_by_support_bin(base: pd.DataFrame, B: int = 400, seed: int = 909) -> pd.DataFrame:
    """D computed WITHIN each DELIMITED support bin -- the interaction, not the gate.

    Gating (cumulative cuts) is the wrong lens for "how much does density matter": `moderate+`
    still holds ~91% of pockets, so the cumulative average barely moves. Slicing into DELIMITED
    bins (very_low | low | moderate | high | very_high) shows the effect at full strength: the
    winning-p divergence is nearly absent in the lowest density bin and full-strength in the
    highest. This is the "confidence amplifier" reading -- classification confidence only
    produces a detection divergence where there is evidence density beneath it.

    Not tautological: density carries NO divergence of its own (the marginal gap vs its own bins
    is flat), so this is an interaction with winning_p, not a main effect.
    """
    rows_spec = []
    for prop, col in [("density", "density_level_db"), ("separation", "separation_level_db")]:
        for lv in LEVELS5:
            rows_spec.append((prop, lv, base[col].eq(lv).to_numpy()))
    masks = [s[2] for s in rows_spec]

    rng = np.random.default_rng(seed)
    cell = _cells(base)
    succ = [base[m].to_numpy(float) for m in METHODS]
    obs = _D_grid(succ, cell, masks)
    order, starts, ends, nC = _cluster_index(base)
    keys = [(gi, mi) for gi in range(len(masks)) for mi in range(len(METHODS))]
    bD = {k: np.full(B, np.nan) for k in keys}
    for b in range(B):
        pick = rng.integers(0, nC, nC)
        idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
        r = _D_grid([s[idx] for s in succ], cell[idx], [mk[idx] for mk in masks])
        for k in keys:
            bD[k][b] = r[k][0]

    rows = []
    for gi, (prop, lv, mk) in enumerate(rows_spec):
        sub = base[mk]
        for mi, m in enumerate(METHODS):
            D = obs[(gi, mi)][0]
            rec = {"property": prop, "bin": lv, "method": m, "n_bin": int(mk.sum()),
                   "D": D,
                   "D_lo": np.nanpercentile(bD[(gi, mi)], 2.5),
                   "D_hi": np.nanpercentile(bD[(gi, mi)], 97.5)}
            for wl in ["very_low", "very_high"]:
                for cl in ["apo", "holo"]:
                    s = sub[(sub["prob_level_db"] == wl) & (sub["cls"] == cl)]
                    rec[f"rate_wp_{wl}_{cl}"] = 100 * s[m].mean() if len(s) else np.nan
                    rec[f"n_wp_{wl}_{cl}"] = len(s)
            rec["gap_wp_very_low"] = rec["rate_wp_very_low_holo"] - rec["rate_wp_very_low_apo"]
            rec["gap_wp_very_high"] = rec["rate_wp_very_high_holo"] - rec["rate_wp_very_high_apo"]
            rows.append(rec)
    return pd.DataFrame(rows)


def headline_contrasts(base: pd.DataFrame, B: int = 600, seed: int = 31337) -> pd.DataFrame:
    """Bottom-vs-top support bin: the headline contrast, UNPOOLED vs POOLED.

    Records a tested-and-rejected design choice. The intuition that pooling {very_low + low} against
    {high + very_high} would both fix the small bottom bin AND sharpen the contrast is WRONG on both
    counts, for both descriptors:

      * It HALVES the contrast. Density's effect is a THRESHOLD at very_low only (D = 10.2 there but
        already 33.9 at `low`), so pooling drags the bottom group up to 23.8 and the density contrast
        falls +28.4 -> +14.2 pp. Separation likewise: -13.9 -> -7.6 pp.
      * It barely buys precision. The bottom-group n rises 2.2x (3,545 -> 7,733) but the CI narrows
        only ~6% (20.8 -> 19.6 pp), because the bootstrap is CLUSTERED BY UNIPROT -- pockets within a
        protein are correlated, so extra pockets are not extra independent information.

    So the figure uses the UNPOOLED very_low-vs-high contrast. `high` (not `very_high`) is the top
    group for both descriptors: it is density's maximum-D bin and much better populated.
    """
    schemes = {
        "unpooled": {"lo": ["very_low"], "hi": ["high"]},
        "pooled": {"lo": ["very_low", "low"], "hi": ["high", "very_high"]},
    }
    specs = []
    for prop, col in [("density", "density_level_db"), ("separation", "separation_level_db")]:
        for scheme, sides in schemes.items():
            for side, bins in sides.items():
                specs.append((prop, scheme, side, base[col].isin(bins).to_numpy()))
    masks = [s[3] for s in specs]

    rng = np.random.default_rng(seed)
    cell = _cells(base)
    succ = [base[m].to_numpy(float) for m in METHODS]
    obs = _D_grid(succ, cell, masks)
    order, starts, ends, nC = _cluster_index(base)
    keys = [(gi, mi) for gi in range(len(masks)) for mi in range(len(METHODS))]
    boot = {k: np.full(B, np.nan) for k in keys}
    for b in range(B):
        pick = rng.integers(0, nC, nC)
        idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
        r = _D_grid([s[idx] for s in succ], cell[idx], [mk[idx] for mk in masks])
        for k in keys:
            boot[k][b] = r[k][0]

    gi_of = {(p, s, sd): i for i, (p, s, sd, _) in enumerate(specs)}
    rows = []
    for prop in ["density", "separation"]:
        for scheme in schemes:
            gl, gh = gi_of[(prop, scheme, "lo")], gi_of[(prop, scheme, "hi")]
            for mi, m in enumerate(METHODS):
                # PAIRED: one resample, both groups recomputed -> CI on the difference itself
                diff = boot[(gh, mi)] - boot[(gl, mi)]
                rows.append({
                    "property": prop, "scheme": scheme, "method": m,
                    "n_lo": int(masks[gl].sum()), "D_lo": obs[(gl, mi)][0],
                    "D_lo_lo": np.nanpercentile(boot[(gl, mi)], 2.5),
                    "D_lo_hi": np.nanpercentile(boot[(gl, mi)], 97.5),
                    "n_hi": int(masks[gh].sum()), "D_hi": obs[(gh, mi)][0],
                    "D_hi_lo": np.nanpercentile(boot[(gh, mi)], 2.5),
                    "D_hi_hi": np.nanpercentile(boot[(gh, mi)], 97.5),
                    "contrast": obs[(gh, mi)][0] - obs[(gl, mi)][0],
                    "contrast_lo": np.nanpercentile(diff, 2.5),
                    "contrast_hi": np.nanpercentile(diff, 97.5),
                })
    return pd.DataFrame(rows)


def gap_grid(base: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """The AF-Holo - AF-Apo gap on the full 2D (winning-p bin x support bin) grid.

    This is the "in conjunction with" view: it shows the whole interaction surface rather than a
    1D slice. Reading it: the gap grows left->right along winning_p (the Fig 6A tendency); the ROWS
    show what that growth depends on. For density the bottom row (very-low density) stays pale --
    the tendency does not exist there. For separation the top rows are compressed within the
    confident winning-p columns.

    DO NOT read that compression as separation improving AF-Apo detection (an earlier version of
    this docstring did, and was wrong on both clauses). It tracks POCKET SIZE: high-separation sites
    are simply larger (union_rg 6.28 -> 8.12 A, union residue count 8.6 -> 14.2 across separation
    bins) and size is the dominant success correlate (rho ~ +0.61-0.65 vs separation's own +0.075) --
    partly mechanically, since the DCA ground truth IS the union-residue heavy-atom cloud. Standardize
    on site size and the apparent AF-Apo lift is abolished: within the confident columns it goes
    +19.97 -> +2.08 pp (wp=very_high) and +27.50 -> +2.40 pp (wp=high). "Closes the gap" is also false
    marginally -- across separation bins the gap WIDENS (10.7 / 15.4 / 16.8 / 15.0 / 23.2), because
    AF-Holo rises faster (+22.8) than AF-Apo (+10.3). See the module docstring's separation note.

    Cells with fewer than `min_n` pockets in EITHER class are emitted with gap = NaN (too noisy to
    colour); the smallest raw cell on this grid is 44.
    """
    rows = []
    for prop, col in [("density", "density_level_db"), ("separation", "separation_level_db")]:
        for wl in LEVELS5:
            for sl in LEVELS5:
                cellm = base["prob_level_db"].eq(wl) & base[col].eq(sl)
                a = base[cellm & base.cls.eq("apo")]
                h = base[cellm & base.cls.eq("holo")]
                for m in METHODS:
                    ok = len(a) >= min_n and len(h) >= min_n
                    ra = 100 * a[m].mean() if len(a) else np.nan
                    rh = 100 * h[m].mean() if len(h) else np.nan
                    rows.append({"property": prop, "prob_bin": wl, "support_bin": sl, "method": m,
                                 "n_apo": len(a), "n_holo": len(h),
                                 "rate_apo": ra, "rate_holo": rh,
                                 "gap": (rh - ra) if ok else np.nan,
                                 "below_min_n": not ok})
    return pd.DataFrame(rows)


def delimited_bins(base: pd.DataFrame) -> pd.DataFrame:
    """Success vs DELIMITED support bins, within winning-p subsets.

    `wp_very_high` is the endpoint cell the figure's a/b panels show; `wp_high+` and `wp_moderate+`
    are progressively broader confident subsets (nested: very_high < high+ < moderate+); `wp_all` is
    the marginal (flat for density -- density carries no divergence itself).
    """
    subsets = {
        # cumulative (nested) subsets
        "wp_very_high": base["prob_level_db"].eq("very_high"),
        "wp_high+": base["prob_level_db"].isin({"high", "very_high"}),
        "wp_moderate+": base["prob_level_db"].isin(MODPLUS),
        "wp_all": pd.Series(True, index=base.index),
    }
    # DELIMITED winning-p bins (each pocket in exactly one). `wp_very_high` above is by construction
    # identical to `wp_bin_very_high`; both are emitted so the cumulative and delimited families can
    # be plotted from one table without the caller having to know which names are which.
    for lv in LEVELS5:
        subsets[f"wp_bin_{lv}"] = base["prob_level_db"].eq(lv)
    rows = []
    for prop, col in [("density", "density_level_db"), ("separation", "separation_level_db")]:
        for wname, wmask in subsets.items():
            for lv in LEVELS5:
                sub = base[wmask.to_numpy() & base[col].eq(lv).to_numpy()]
                for m in METHODS:
                    a, h = sub[sub.cls.eq("apo")], sub[sub.cls.eq("holo")]
                    ra = 100 * a[m].mean() if len(a) else np.nan
                    rh = 100 * h[m].mean() if len(h) else np.nan
                    rows.append({"property": prop, "wp_subset": wname, "bin": lv, "method": m,
                                 "n_apo": len(a), "n_holo": len(h),
                                 "rate_apo": ra, "rate_holo": rh, "gap": rh - ra})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- diagnostics
def composition(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    apo_m = base["cls"].eq("apo").to_numpy()
    holo_m = base["cls"].eq("holo").to_numpy()
    for gate in GATES_ALL:
        m = gate_mask(base, gate)
        sub = base[m]
        rec = {
            "gate": gate, "n": len(sub), "frac_kept": len(sub) / len(base),
            "pct_apo": 100 * sub["cls"].eq("apo").mean(),
            "pct_holo": 100 * sub["cls"].eq("holo").mean(),
            "retain_apo": 100 * (m & apo_m).sum() / apo_m.sum(),
            "retain_holo": 100 * (m & holo_m).sum() / holo_m.sum(),
            "mean_rg": sub["union_rg"].mean(),
            "mean_winning_p": sub["winning_p"].mean(),
            "mean_n_apo_avg": sub["n_apo_avg"].mean(),
            "mean_n_holo_avg": sub["n_holo_avg"].mean(),
        }
        for lv in LEVELS5:
            rec[f"retain_{lv}"] = 100 * sub["prob_level_db"].eq(lv).sum() / base["prob_level_db"].eq(lv).sum()
        rows.append(rec)
    return pd.DataFrame(rows)


def within_cell_selection(base: pd.DataFrame) -> pd.DataFrame:
    """THE mechanism table: for each endpoint cell, success of pockets the gate DROPS vs KEEPS.

    If dropped-minus-kept differs BY CLASS, the gap moves and D changes. This -- not any
    correlation with winning_p -- is what separates the density gate from the separation gate.
    Also carries the evidence depth of the dropped pockets, which is what licenses (or refutes)
    calling them 'diffuse' rather than 'sparse'.
    """
    rows = []
    for gate in ["dens_mod+", "sep_mod+", "both_mod+"]:
        gm = gate_mask(base, gate)
        for lv in ["very_low", "very_high"]:
            for cl in ["apo", "holo"]:
                cellm = (base["prob_level_db"] == lv) & (base["cls"] == cl)
                kept, dropped = base[cellm & gm], base[cellm & ~gm]
                for m in METHODS:
                    rows.append({
                        "gate": gate, "bin": lv, "cls": cl, "method": m,
                        "n_kept": len(kept), "n_dropped": len(dropped),
                        "pct_dropped": 100 * len(dropped) / max(len(kept) + len(dropped), 1),
                        "rate_kept": 100 * kept[m].mean() if len(kept) else np.nan,
                        "rate_dropped": 100 * dropped[m].mean() if len(dropped) else np.nan,
                        "dropped_minus_kept": (100 * dropped[m].mean() - 100 * kept[m].mean())
                        if len(kept) and len(dropped) else np.nan,
                        "n_apo_avg_kept": kept["n_apo_avg"].mean() if len(kept) else np.nan,
                        "n_apo_avg_dropped": dropped["n_apo_avg"].mean() if len(dropped) else np.nan,
                        "n_holo_avg_kept": kept["n_holo_avg"].mean() if len(kept) else np.nan,
                        "n_holo_avg_dropped": dropped["n_holo_avg"].mean() if len(dropped) else np.nan,
                    })
    return pd.DataFrame(rows)


def correlations(base: pd.DataFrame) -> pd.DataFrame:
    """Within-class Spearman of each support metric with winning_p and with success.

    The POOLED value is a class-averaging dilution and must not be quoted on its own.
    """
    rows = []
    for scope in ["pooled", "apo", "holo"]:
        s = base if scope == "pooled" else base[base["cls"] == scope]
        for prop in ["pocket_density_pair", "pocket_separation_pair"]:
            rec = {"scope": scope, "property": prop, "n": len(s),
                   "rho_with_winning_p": s[prop].corr(s["winning_p"], method="spearman"),
                   "rho_with_union_rg": s[prop].corr(s["union_rg"], method="spearman")}
            for m in METHODS:
                rec[f"rho_with_{m}"] = s[prop].corr(s[m], method="spearman")
            rows.append(rec)
    # the decisive one: within the very_high cell, density's association with success BY CLASS
    for cl in ["apo", "holo"]:
        s = base[(base["prob_level_db"] == "very_high") & (base["cls"] == cl)]
        rec = {"scope": f"very_high_{cl}", "property": "pocket_density_pair", "n": len(s),
               "rho_with_winning_p": s["pocket_density_pair"].corr(s["winning_p"], method="spearman"),
               "rho_with_union_rg": s["pocket_density_pair"].corr(s["union_rg"], method="spearman")}
        for m in METHODS:
            rec[f"rho_with_{m}"] = s["pocket_density_pair"].corr(s[m], method="spearman")
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=None, help="default: work_dir/lbs_generality/analysis_v1/tables")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap replicates for the gate table")
    ap.add_argument("--B_ladder", type=int, default=400, help="bootstrap replicates for the ladder")
    a = ap.parse_args()

    cfg.require("WORK_DIR")
    outdir = _Path(a.outdir) if a.outdir else (cfg.LBS_ANALYSIS_DIR / "tables")
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_base()
    print(f"scored pockets: {len(base):,} | apo {base.cls.eq('apo').sum():,} "
          f"holo {base.cls.eq('holo').sum():,} tie {base.cls.eq('tie').sum():,}")

    bin_rates(base).to_csv(outdir / "fig6a_gates_bin_rates.csv", index=False)
    composition(base).to_csv(outdir / "fig6a_gates_composition.csv", index=False)
    within_cell_selection(base).to_csv(outdir / "fig6a_gates_within_cell_selection.csv", index=False)
    correlations(base).to_csv(outdir / "fig6a_gates_correlations.csv", index=False)

    # secondary: swap the x-axis -- is the divergence carried by winning_p alone?
    alt = []
    for col, tag in [("density_level_db", "density"), ("separation_level_db", "separation"),
                     ("prob_level_db", "winning_p")]:
        t = bin_rates(base, bin_col=col)
        t = t[t["gate"] == "all"].copy()
        t["axis"] = tag
        alt.append(t)
    pd.concat(alt).to_csv(outdir / "fig6a_gates_altaxis_bin_rates.csv", index=False)

    print(f"gate table: paired cluster bootstrap B={a.B} ...")
    dv = run_gates(base, B=a.B)
    dv.to_csv(outdir / "fig6a_gates_divergence.csv", index=False)

    print(f"threshold ladder: paired cluster bootstrap B={a.B_ladder} ...")
    ld = threshold_ladder(base, B=a.B_ladder)
    ld.to_csv(outdir / "fig6a_gates_threshold_ladder.csv", index=False)

    delimited_bins(base).to_csv(outdir / "fig6a_gates_delimited_bins.csv", index=False)
    gap_grid(base).to_csv(outdir / "fig6a_gates_gap_grid.csv", index=False)

    print("headline contrasts (unpooled vs pooled) ...")
    hc = headline_contrasts(base)
    hc.to_csv(outdir / "fig6a_gates_headline_contrasts.csv", index=False)

    print(f"divergence within delimited support bins: cluster bootstrap B={a.B_ladder} ...")
    dsb = divergence_by_support_bin(base, B=a.B_ladder)
    dsb.to_csv(outdir / "fig6a_gates_divergence_by_support_bin.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== D = gap(very_high) - gap(very_low), pp ===")
    print(dv[["gate", "method", "n_kept", "D_divergence", "dD_vs_ungated", "dD_lo", "dD_hi"]]
          .round(2).to_string(index=False))
    print("\n=== THRESHOLD LADDER (consensus): dD and its apo/holo decomposition ===")
    print(ld[ld.method.eq("consensus")][["property", "rung", "pct_kept", "dD_vs_ungated", "dD_lo", "dD_hi",
                                         "d_apo_slope", "d_holo_slope",
                                         "rate_very_high_apo", "rate_very_high_holo"]]
          .round(2).to_string(index=False))
    print("\n=== DIVERGENCE WITHIN DELIMITED SUPPORT BINS (consensus) -- the interaction ===")
    print(dsb[dsb.method.eq("consensus")][["property", "bin", "n_bin", "D", "D_lo", "D_hi",
                                           "rate_wp_very_high_apo", "rate_wp_very_high_holo"]]
          .round(2).to_string(index=False))
    print("\n=== HEADLINE CONTRAST: pooling HALVES it and barely buys precision (consensus) ===")
    print(hc[hc.method.eq("consensus")][["property", "scheme", "n_lo", "D_lo", "n_hi", "D_hi",
                                         "contrast", "contrast_lo", "contrast_hi"]]
          .round(2).to_string(index=False))
    print("\nwrote ->", outdir)


if __name__ == "__main__":
    main()
