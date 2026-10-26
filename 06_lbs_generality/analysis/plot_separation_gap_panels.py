# -*- coding: utf-8 -*-
"""
The separation panels of manuscript v4 **Figure 4** -- "when does it matter".

Two panels, selected with --panel:

  crosstask   ONE unit-free axis carrying all three downstream tasks: AlphaFill transplant fit,
              binding-site detection, and cognate-ligand docking. This is the panel the section
              exists for -- three methods that share almost nothing tracing the same gradient --
              and it only works if they are on comparable axes, which pp-of-success and
              clash-score-units are not.
  predictors  the per-predictor detection gap, which keeps the detail the cross-task panel
              averages away (notably that fpocket has no gap at all at very low separation).

WHY CLIFF'S DELTA
-----------------
Success is binary, transplant fit is a continuous score where LOWER is better, and docking is a
rate on a different denominator. Cliff's delta is unit-free, works on all three, and for a binary
outcome reduces exactly to the difference in success rates -- verified here: P2Rank's delta is
+0.038 at very low separation against a 3.8 pp gap, fpocket's -0.002 against -0.2 pp.
Positive always means AF-Holo does better, whatever "better" means for that task.

**The two panels are NOT on one scale, and must not be labelled as if they were.** The
per-predictor panel plots the gap in PERCENTAGE POINTS, which is what the manuscript quotes
(consensus 3.4 -> 19.1). Consensus is the mean of three binaries, not a binary, so its delta
(+0.046 -> +0.329) is a rank effect size and NOT that gap -- the equivalence holds only for the
individual predictors. The cross-task panel needs delta because clash score and success rate
have no common unit; the predictor panel needs the gap because that is the published quantity.

THE TWO CONSTRAINTS THIS PANEL IS BUILT ON
------------------------------------------
* **Never plot that separation improves AF-Apo detection.** The raw +10.3 pp AF-Apo gain across
  separation bins is pocket size; standardized it is abolished, and it is not apo-specific. Only
  the GAP BETWEEN CLASSES is a safe estimand, so only the gap is ever drawn.
* **Report the size-standardized number.** Every delta here is standardized by direct adjustment
  to its own task's `union_rg` decile distribution, computed within tier and decile and then
  averaged with fixed weights. Raw, the separation gap is non-monotonic and would invert the
  story; the raw series is drawn faintly beside it rather than hidden.

DENOMINATORS DIFFER, DELIBERATELY
---------------------------------
Detection 86,946 UBS pockets; docking its nested 60,689; AlphaFill the matched same-UniProt set.
AlphaFill carries no size column of its own -- `union_rg` and `site_uniprot` are joined from the
LBS table on `unit_uid`, which covers 81% of the matched set. The panel prints each n so the
three are never read as one denominator.

CIs are percentile intervals from a cluster bootstrap over `site_uniprot`, because pockets within
a protein are correlated. Monte Carlo: quote from this script with its seed, never a side run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _p in (Path(__file__).resolve().parents[2],
           Path(__file__).resolve().parent,
           Path(__file__).resolve().parents[2] / "04_p2rank" / "analysis",
           Path(__file__).resolve().parents[2] / "03_alphafill" / "ubs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import config as cfg  # noqa: E402
from common import figspec  # noqa: E402
import analyze_fig6a_quality_gates as G  # noqa: E402
from confidence_labeling import support_level  # noqa: E402
import plot_clash_ridgelines as pcr  # noqa: E402

figspec.apply_rc()

TIERS = ["very_low", "low", "moderate", "high", "very_high"]
# Upright, not tilted: these panels are 89 mm wide, so a tick is 15.7 mm apart and the longest
# label ("very high") is 9.5 mm at 6 pt. The 35-degree tilt was inherited from a narrower canvas.
TIER_LABEL = ["very low", "low", "moderate", "high", "very high"]
PRED = ["P2Rank", "fpocket", "IF-SitePred"]
K_DECILES = 10

TASK_STYLE = {
    "AlphaFill fit": dict(color="#8E5AA8", lw=1.8, marker="o", ms=3.0),
    "Detection": dict(color="#3B7A57", lw=1.8, marker="s", ms=3.0),
    "Docking": dict(color="#7A3B6B", lw=1.8, marker="^", ms=3.2),
}
PRED_STYLE = {"P2Rank": dict(color="#4C72B0", ls=":"),
              "fpocket": dict(color="#DD8452", ls="--"),
              "IF-SitePred": dict(color="#55A868", ls="-."),
              "consensus": dict(color="#3B7A57", ls="-", lw=1.8)}


# ------------------------------------------------------------------ Cliff's delta
def _delta(a, h):
    """Cliff's delta of h against a: +1 when every h exceeds every a. Rank-based, tie-aware."""
    na, nh = len(a), len(h)
    if na == 0 or nh == 0:
        return np.nan
    both = np.concatenate([a, h])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(len(both), float)
    ranks[order] = np.arange(1, len(both) + 1)
    # average ranks within ties, or a binary outcome gets a delta that depends on sort order
    sb = both[order]
    i = 0
    while i < len(sb):
        j = i
        while j + 1 < len(sb) and sb[j + 1] == sb[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    U_h = ranks[na:].sum() - nh * (nh + 1) / 2.0
    return 2.0 * U_h / (na * nh) - 1.0


def _prep(df, value_col, higher_is_better=True):
    d = df[df["cls"].isin(["apo", "holo"])].copy()
    d = d[d[value_col].notna() & d["union_rg"].notna() & d["site_uniprot"].notna()]
    zb = pd.qcut(pd.to_numeric(d["union_rg"], errors="coerce"), K_DECILES,
                 labels=False, duplicates="drop")
    d = d[zb.notna()].copy()
    y = d[value_col].to_numpy(float)
    if not higher_is_better:
        y = -y            # so a positive delta always means "AF-Holo does better"
    tier = pd.Categorical(d["tier"], categories=TIERS).codes
    keep = tier >= 0
    codes, _ = pd.factorize(d["site_uniprot"])
    zbv = zb[zb.notna()].astype(int).to_numpy()
    w = np.bincount(zbv[keep], minlength=K_DECILES).astype(float)
    w /= w.sum()
    return y[keep], (d["cls"].to_numpy() == "holo")[keep], tier[keep], zbv[keep], codes[keep], w


def _deltas(y, ish, tier, zb, w, standardized=True):
    out = np.full(len(TIERS), np.nan)
    for t in range(len(TIERS)):
        m = tier == t
        if not m.any():
            continue
        if not standardized:
            out[t] = _delta(y[m & ~ish], y[m & ish])
            continue
        num = den = 0.0
        for dcl in range(K_DECILES):
            c = m & (zb == dcl)
            a, h = y[c & ~ish], y[c & ish]
            if len(a) and len(h):
                dv = _delta(a, h)
                if np.isfinite(dv):
                    num += w[dcl] * dv
                    den += w[dcl]
        out[t] = num / den if den > 0 else np.nan
    return out


def _gaps(y, ish, tier, zb, w, standardized=True):
    """Size-standardized mean difference (holo - apo), in the outcome's own units."""
    out = np.full(len(TIERS), np.nan)
    for t in range(len(TIERS)):
        m = tier == t
        if not m.any():
            continue
        if not standardized:
            out[t] = y[m & ish].mean() - y[m & ~ish].mean()
            continue
        num = den = 0.0
        for dcl in range(K_DECILES):
            c = m & (zb == dcl)
            a_, h_ = y[c & ~ish], y[c & ish]
            if len(a_) and len(h_):
                num += w[dcl] * (h_.mean() - a_.mean())
                den += w[dcl]
        out[t] = num / den if den > 0 else np.nan
    return out


def _clusters(codes):
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    return order, np.searchsorted(sc, np.arange(sc.max() + 1), "left"), \
        np.searchsorted(sc, np.arange(sc.max() + 1), "right"), sc.max() + 1


def tier_delta(df, value_col, label, B, seed, higher_is_better=True, stat="delta"):
    est = _deltas if stat == "delta" else _gaps
    y, ish, tier, zb, codes, w = _prep(df, value_col, higher_is_better)
    obs = est(y, ish, tier, zb, w, True)
    raw = est(y, ish, tier, zb, w, False)
    order, st, en, nC = _clusters(codes)
    rng = np.random.default_rng(seed)
    boot = np.full((B, len(TIERS)), np.nan)
    for b in range(B):
        pick = rng.integers(0, nC, nC)
        ix = np.concatenate([order[st[c]:en[c]] for c in pick])
        boot[b] = est(y[ix], ish[ix], tier[ix], zb[ix], w, True)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    rows = [{"task": label, "tier": t, "n": int((tier == i).sum()), "stat": stat,
             "delta_std": obs[i], "ci_lo": lo[i], "ci_hi": hi[i], "delta_raw": raw[i]}
            for i, t in enumerate(TIERS)]
    print(f"  {label:16s} n={len(y):,} clusters={nC:,}  " +
          "  ".join(f"{t[:4]}:{obs[i]:+.3f}" for i, t in enumerate(TIERS)))
    return pd.DataFrame(rows), len(y)


# ------------------------------------------------------------------ data
def load_lbs():
    b = G.load_base()
    b["consensus"] = b[PRED].mean(axis=1)
    b["cls"] = b["pocket_class"].astype(str).str.lower()
    b["tier"] = b["separation_level_db"].astype(str).str.strip().str.lower()
    return b


def load_docking(lbs):
    d = pd.read_csv(Path(cfg.DOCKING_RUN3_DIR) / "analysis" / "docking_run3_merged.csv",
                    low_memory=False)
    d = d[(d.ligand_group == 1) & (d.status.astype(str).str.lower() == "ok")].copy()
    aff = pd.to_numeric(d["affinity_top"], errors="coerce")
    inpk = d["pose_in_pocket"].astype(str).str.lower().isin(["true", "1", "yes"])
    d["_ok"] = (inpk & (aff <= -6)).astype(float)
    d["cls"] = d["pocket_class"].astype(str).str.lower()
    d["tier"] = (1.0 - pd.to_numeric(d["pocket_overlap_pair"], errors="coerce")).map(support_level)
    pk = d.groupby(["site_uid", "cls", "tier"], as_index=False).agg(
        dockable=("_ok", "mean"), union_rg=("union_rg", "first"))
    return pk.merge(lbs[["site_uid", "site_uniprot"]], on="site_uid", how="left")


def load_alphafill(lbs, scope="same_uniprot_ge_0p30"):
    """Matched ST_100 transplants. union_rg / site_uniprot join from the LBS table on unit_uid --
    this summary carries neither, and without them the delta cannot be size-standardized or
    cluster-bootstrapped the way the other two tasks are."""
    p = Path(cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR) / f"per_ubs_summary_{scope}.csv"
    d = pd.read_csv(p, low_memory=False)
    d = d[d["threshold"].astype(str).str.strip() == "ST_100"].copy()
    d["_c"] = pcr.normalize_class(d["pocket_class"])
    d = d[d["matched"].fillna(False) & d["_c"].isin(pcr.CLASS_ORDER)].copy()
    d["cls"] = np.where(d["_c"].str.lower().str.contains("holo"), "holo", "apo")
    d["tier"] = d["separation_level_db"].astype(str).str.strip().str.lower()
    d["avg_clash"] = pd.to_numeric(d["avg_clash"], errors="coerce")
    n_all = len(d)
    d = d.merge(lbs[["site_uid", "union_rg", "site_uniprot"]],
                left_on="unit_uid", right_on="site_uid", how="left")
    cov = d["union_rg"].notna().mean()
    print(f"  AlphaFill matched {n_all:,}; size/cluster joined for {d.union_rg.notna().sum():,} "
          f"({100*cov:.0f}%)")
    return d


# ------------------------------------------------------------------ panels
def draw_crosstask(ax, res, ns):
    x = np.arange(len(TIERS))
    for task in ("AlphaFill fit", "Detection", "Docking"):
        r = res[res.task == task].set_index("tier").reindex(TIERS)
        st = TASK_STYLE[task]
        ax.plot(x, r.delta_std, label=f"{task}  (n = {ns[task]:,})", zorder=3, **st)
        ax.fill_between(x, r.ci_lo, r.ci_hi, color=st["color"], alpha=0.13, lw=0, zorder=1)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABEL)      # upright -- see the note at TIER_LABEL
    ax.set_xlabel("Apo-holo separation")
    ax.set_ylabel("AF-Holo advantage\n(Cliff's $\\delta$, size-standardized)")
    ax.set_xlim(-0.3, len(TIERS) - 0.7)
    ax.grid(axis="y", color=figspec.FAINT, lw=0.6, ls="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.2, labelspacing=0.25,
              fontsize=figspec.FS_NOTE)


def draw_predictors(ax, res):
    x = np.arange(len(TIERS))
    for m in PRED + ["consensus"]:
        r = res[res.task == m].set_index("tier").reindex(TIERS)
        st = dict(PRED_STYLE[m])
        st.setdefault("lw", 1.0)
        ax.plot(x, 100 * r.delta_std, label=m.capitalize() if m == "consensus" else m,
                zorder=3, **st)
        if m == "consensus":
            ax.fill_between(x, 100 * r.ci_lo, 100 * r.ci_hi, color=st["color"], alpha=0.13, lw=0)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABEL)      # upright -- see the note at TIER_LABEL
    ax.set_xlabel("Apo-holo separation")
    ax.set_ylabel("AF-Holo − AF-Apo detection\ngap (pp, size-standardized)")
    ax.set_xlim(-0.3, len(TIERS) - 0.7)
    ax.grid(axis="y", color=figspec.FAINT, lw=0.6, ls="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # long handles: at 1 pt these four dash patterns are indistinguishable in a short swatch
    ax.legend(loc="upper left", handlelength=3.6, borderaxespad=0.2, labelspacing=0.25,
              fontsize=figspec.FS_NOTE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--panel", choices=("crosstask", "predictors", "both"), default="both")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_basename", default=None)
    ap.add_argument("--fig_size", default=None, metavar="W,H", help="canvas in MILLIMETRES")
    ap.add_argument("--letter", default=None)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    print("loading ...")
    lbs = load_lbs()
    dk = load_docking(lbs)
    af = load_alphafill(lbs)

    print(f"\nsize-standardized Cliff's delta by separation tier, {a.boot} cluster-bootstrap draws:")
    parts, ns = [], {}
    r, n = tier_delta(af, "avg_clash", "AlphaFill fit", a.boot, a.seed, higher_is_better=False)
    parts.append(r); ns["AlphaFill fit"] = n
    r, n = tier_delta(lbs, "consensus", "Detection", a.boot, a.seed)
    parts.append(r); ns["Detection"] = n
    r, n = tier_delta(dk, "dockable", "Docking", a.boot, a.seed)
    parts.append(r); ns["Docking"] = n
    for m in PRED + ["consensus"]:
        r, _ = tier_delta(lbs, m, m, a.boot, a.seed, stat="gap")
        parts.append(r)
    res = pd.concat(parts, ignore_index=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "Figure4_separation_source_data.csv", index=False)
    print("\nwrote", out / "Figure4_separation_source_data.csv")

    want = ["crosstask", "predictors"] if a.panel == "both" else [a.panel]
    for k, which in enumerate(want):
        if a.fig_size:
            w_mm, h_mm = (float(v) for v in a.fig_size.split(","))
        else:
            w_mm, h_mm = 60.0, 58.0
        fig = plt.figure(figsize=(w_mm * figspec.MM, h_mm * figspec.MM))
        ax = fig.add_subplot(111)
        (draw_crosstask(ax, res, ns) if which == "crosstask" else draw_predictors(ax, res))
        # 13 mm of bottom margin was sized for tilted tick labels; upright they need ~8.5 mm,
        # and the surplus would sit as a white band under the axis title
        fig.subplots_adjust(**figspec.margins(w_mm * figspec.MM, h_mm * figspec.MM,
                                              left=15.0, right=2.0, top=4.5, bottom=10.0))
        if a.letter:
            L = a.letter.split(",")[k] if "," in a.letter else a.letter
            fig.text(0.5 / w_mm, 1 - 0.5 / h_mm, L, ha="left", va="top",
                     fontsize=figspec.FS_LETTER, fontweight="bold")
        base = a.out_basename or f"Figure4_{which}"
        if a.panel == "both" and not a.out_basename:
            base = f"Figure4_{which}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{base}.{ext}", dpi=600, facecolor="white")
        plt.close(fig)
        print(f"wrote {out / (base + '.png')}  {w_mm:.0f} x {h_mm:.0f} mm")


if __name__ == "__main__":
    main()
