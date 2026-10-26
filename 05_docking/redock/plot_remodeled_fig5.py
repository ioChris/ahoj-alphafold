# -*- coding: utf-8 -*-
"""
Remodeled Figure 5 (draft) — the redock (docking) ligand-placement panel, in Figure 5's
separation binning, to sit alongside the manuscript's AlphaFill quad.

Manuscript Fig 5 (C3, entry-level AlphaFill) = a quad of AlphaFill metrics (avg clash score,
clash count, local RMSD, global RMSD) across the 5 database SEPARATION classes + an "All" group,
split apo vs holo. Redock has only an RMSD (docking produces no AlphaFill-style clash-with-receptor
metric), so its contribution is 1-2 RMSD panels in the SAME binning.

What the redock RMSD is (metric definition): heavy-atom RMSD between (a) the cognate ligand DOCKED
into the AFDB (AlphaFold) receptor and (b) the EXPERIMENTAL crystal ligand from the AHoJ query holo
structure, brought into the AF frame by superposing the query receptor onto the AF model on the
pocket-residue CA atoms. In-place, symmetry-corrected. It is cross-docking into the predicted model
scored against a crystal-derived reference pose (NOT classic self-redocking, NOT the crystal frame).

Separation classes (applied to `pocket_separation_pair` in [0,1], the same verbal bands the
classification/AlphaFill Fig 5 use): Very low 0-0.20 / Low 0.20-0.30 / Moderate 0.30-0.60 /
High 0.60-0.85 / Very high 0.85-1.00, plus "All" (pooled).

Panel A (context): AlphaFill local RMSD by the same bins (pocket-level, same-UniProt id>=0.30) — a
re-derivation for the parallel; the manuscript Fig 5 quad is entry-level and is left untouched.
Panels B/C: redock median RMSD and redock <=2 A success, the redock addition.

Env: `plotting`. Run with KMP_DUPLICATE_LIB_OK=TRUE.

v4 SI panels (2026-09-15). `--v4_panels_dir` additionally renders each panel as its own PNG in
the manuscript's grammar -- the AF-Apo / AF-Holo fills of the half-violin panels, lowercase tiers
with "overall" LAST behind a dotted divider, an n row, figspec type at a true 88 x 62 mm canvas --
so the SI composer can place them next to the docking-affinity violins without rescaling. The
legacy 1x3 figure is written unchanged either way.
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import MaxNLocator

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402
from common import figspec  # noqa: E402

RUN3 = cfg.DOCKING_RUN3_DIR
APO, HOLO = "#4C72B0", "#C44E52"
SEP_EDGES = [-1e-9, 0.20, 0.30, 0.60, 0.85, 1.0001]
SEP_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
ORDER = ["All"] + SEP_LABELS

# v4 panel style: the same fills, edge and class names as plot_clash_ridgelines, so the bar panels
# and the half-violin panel read as one figure. The pooled group goes LAST, as in Figure 4.
V4_FILL = {"apo": "#8DB9DC", "holo": "#F4BD84"}
V4_EDGE = "#4A4A4A"
V4_CLASS_EDGE = {"apo": "#2F6F9E", "holo": "#D8893A"}   # = plot_clash_ridgelines.CLASS_EDGE, for the n row
V4_LABEL = {"apo": "AF-Apo", "holo": "AF-Holo"}
V4_ORDER = SEP_LABELS + ["All"]
V4_TICKS = [b.lower() for b in SEP_LABELS] + ["overall"]
V4_FONTS = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]
V4_MONO = "DejaVu Sans Mono"


def sep5(s):
    return pd.cut(pd.to_numeric(s, errors="coerce"), SEP_EDGES, labels=SEP_LABELS)


def med_ci(vals, n_boot=1000, seed=0):
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy()
    if len(v) < 5:
        return np.nan, np.nan, np.nan, len(v)
    rng = np.random.default_rng(seed)
    b = np.median(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)


def prop_ci(mask):
    m = pd.Series(mask).dropna().to_numpy().astype(float)
    if len(m) < 5:
        return np.nan, np.nan, np.nan, len(m)
    p = m.mean(); se = np.sqrt(max(p * (1 - p), 1e-9) / len(m))
    return p * 100, (p - 1.96 * se) * 100, (p + 1.96 * se) * 100, len(m)


def _rows(df, valfn, valcol):
    """Return {(class, xlabel): (est, lo, hi, n)} for the 6 x-positions (All + 5 classes)."""
    out = {}
    for cls in ("apo", "holo"):
        g = df[df.pocket_class == cls]
        out[(cls, "All")] = valfn(g[valcol])
        for b in SEP_LABELS:
            out[(cls, b)] = valfn(g[g.sepbin == b][valcol])
    return out


def panel(ax, res, ylabel, title, letter=None, pct=False):
    x = np.arange(len(ORDER)); w = 0.38
    for off, cls, col in [(-w / 2, "apo", APO), (w / 2, "holo", HOLO)]:
        est = [res[(cls, b)][0] for b in ORDER]
        lo = [max(0, res[(cls, b)][0] - res[(cls, b)][1]) for b in ORDER]
        hi = [max(0, res[(cls, b)][2] - res[(cls, b)][0]) for b in ORDER]
        ax.bar(x + off, est, w, color=col, label=cls, yerr=[lo, hi], capsize=2.5,
               error_kw=dict(lw=0.9, ecolor="0.4"))
    ax.axvline(0.5, color="0.7", lw=0.8, ls="--")  # separate "All" from the graded classes
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={res[('apo',b)][3]+res[('holo',b)][3]:,})" for b in ORDER],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.margins(y=0.13)
    if letter:
        # panel letter as its own bold artist -- lowercase per the Nature figure guide, and
        # deliberately NOT inside set_title (which renders at title size, non-bold)
        ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="bottom")


def panel_v4(res, ylabel, letter, out, canvas_mm=(88.0, 62.0), integer_ticks=False):
    """One panel at its true composed size, in the manuscript v4 grammar (see module docstring).

    Margins in millimetres match plot_clash_ridgelines' exact-canvas branch (13 / 2 / 3 / 15), so
    the bar panels and the half-violin panel share an axes box when placed side by side.
    """
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": V4_FONTS,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    w_in, h_in = canvas_mm[0] * figspec.MM, canvas_mm[1] * figspec.MM
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.subplots_adjust(**figspec.margins(w_in, h_in, left=13.0, right=2.0, top=3.0, bottom=15.0))

    x = np.arange(len(V4_ORDER)); w = 0.36
    top = 0.0
    for off, cls in [(-w / 2, "apo"), (w / 2, "holo")]:
        est = [res[(cls, b)][0] for b in V4_ORDER]
        lo = [max(0, res[(cls, b)][0] - res[(cls, b)][1]) for b in V4_ORDER]
        hi = [max(0, res[(cls, b)][2] - res[(cls, b)][0]) for b in V4_ORDER]
        ax.bar(x + off, est, w, color=V4_FILL[cls], edgecolor=V4_EDGE, linewidth=0.8,
               label=V4_LABEL[cls], yerr=[lo, hi], capsize=1.6,
               error_kw=dict(lw=0.7, ecolor="0.35"), zorder=3)
        top = max(top, max(e + h for e, h in zip(est, hi) if np.isfinite(e)))
    i_all = len(SEP_LABELS)
    ax.axvline(i_all - 0.5, color="0.78", lw=0.8, ls=":", zorder=0)   # fence off the pooled group
    ax.set_xlim(-0.6, len(V4_ORDER) - 0.4)
    ax.set_ylim(0.0, top * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(V4_TICKS, fontsize=figspec.FS_TITLE)
    ax.get_xticklabels()[i_all].set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=19, labelcolor="0.15")   # room for the two-line n row
    ax.tick_params(axis="y", labelsize=figspec.FS_TICK)
    if integer_ticks:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_ylabel(ylabel, fontsize=figspec.FS_TITLE)
    ax.set_xlabel("Apo-holo separation", fontsize=figspec.FS_TITLE, labelpad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.grid(False, axis="x")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)

    # n row as on the half-violin panel (counts_stacked): apo above holo, centred on the tick, in
    # the class edge colours. Five-digit counts do not fit side by side at 88 mm.
    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    n_y = -0.025
    ax.text(-0.010, n_y, r"$n$", ha="right", va="top", fontsize=figspec.FS_TICK, color="0.30",
            transform=ax.transAxes, clip_on=False)
    for i, b in enumerate(V4_ORDER):
        for dy, cls in ((0.0, "apo"), (-0.062, "holo")):
            ax.text(i, n_y + dy, f"{res[(cls, b)][3]}", ha="center", va="top",
                    fontsize=figspec.FS_NOTE, color=V4_CLASS_EDGE[cls], fontfamily=V4_MONO,
                    transform=tr, clip_on=False)

    if letter:
        fig.text(0.5 / (w_in / figspec.MM), 1 - 0.5 / (h_in / figspec.MM), letter,
                 ha="left", va="top", fontsize=figspec.FS_LETTER, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redock_csv", type=Path, default=RUN3 / "reference_poses" / "redock_rmsd.csv")
    ap.add_argument("--manifest_csv", type=Path, default=RUN3 / "manifest" / "docking_manifest.csv")
    ap.add_argument("--alphafill_csv", type=Path, default=cfg.MERGED_AHOJ_ALPHAFILL_LONG_CSV)
    ap.add_argument("--out_dir", type=Path, default=RUN3 / "reference_poses")
    ap.add_argument("--v4_panels_dir", type=Path, default=None,
                    help="also write one 88x62 mm PNG per panel in the manuscript v4 style")
    ap.add_argument("--v4_letters", default="b,c,d",
                    help="panel letters for redock RMSD, redock success, AlphaFill local RMSD")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest_csv, usecols=["site_uid", "pocket_class", "pocket_separation_pair"],
                      low_memory=False).drop_duplicates("site_uid")
    man["sepbin"] = sep5(man.pocket_separation_pair)

    rk = pd.read_csv(args.redock_csv, usecols=["site_uid", "rmsd_status", "superpose_mode", "redock_rmsd_top"],
                     low_memory=False)
    rk = rk[(rk.rmsd_status == "ok") & (rk.superpose_mode == "pocket")].copy()
    rk["r"] = pd.to_numeric(rk.redock_rmsd_top, errors="coerce")
    R = rk.groupby("site_uid")["r"].min().reset_index().rename(columns={"r": "redock_best"}).merge(man, on="site_uid", how="left")
    R["succ2"] = R.redock_best <= 2

    af = pd.read_csv(args.alphafill_csv, usecols=["unit_uid", "threshold", "afill_local_rmsd",
                                                  "afill_cross_uniprot", "afill_mapping_align_identity"], low_memory=False)
    af = af[(af.threshold == "ST_100") & (pd.to_numeric(af.afill_mapping_align_identity, errors="coerce") >= 0.30)
            & (af.afill_cross_uniprot == False)]  # noqa: E712
    A = af.groupby("unit_uid")["afill_local_rmsd"].min().reset_index().rename(
        columns={"unit_uid": "site_uid", "afill_local_rmsd": "afill_local"}).merge(man, on="site_uid", how="inner")

    resA = _rows(A, med_ci, "afill_local")
    resB = _rows(R, med_ci, "redock_best")
    resC = _rows(R, lambda s: prop_ci(pd.to_numeric(s, errors="coerce") <= 2), "redock_best")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.9))
    panel(ax[0], resA, "AlphaFill local RMSD (Å)  ↓ better",
          "AlphaFill transplant  (pocket-level, for parallel)", letter="a")
    panel(ax[1], resB, "docked vs experimental\nligand RMSD (Å)  ↓ better",
          "Redock  (docked ligand vs crystal ligand, AF frame)", letter="b")
    panel(ax[2], resC, "redock success ≤2 Å (%)  ↑ better", "Redock success", letter="c")
    ax[0].legend(frameon=False, fontsize=9, title="AF-model class", loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out_dir / f"fig5_remodeled_placement.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for name, res in [("AlphaFill local RMSD", resA), ("Redock median RMSD", resB), ("Redock ≤2A %", resC)]:
        print(f"\n{name} (apo / holo, by All + 5 sep classes):")
        for b in ORDER:
            print(f"  {b:10s} apo {res[('apo',b)][0]:.3f} (n={res[('apo',b)][3]:,}) | holo {res[('holo',b)][0]:.3f} (n={res[('holo',b)][3]:,})")
    print(f"\n[SAVE] {args.out_dir / 'fig5_remodeled_placement.png'}")

    if args.v4_panels_dir:
        L = [s.strip() for s in args.v4_letters.split(",")]
        d = args.v4_panels_dir
        for res, ylabel, letter, slug, int_ticks in [
            (resB, "Redock RMSD (Å)", L[0], "redock_rmsd", False),
            (resC, "Redock success, ≤ 2 Å (%)", L[1], "redock_success", True),
            (resA, "AlphaFill local RMSD (Å)", L[2], "afill_local_rmsd", False),
        ]:
            p = panel_v4(res, ylabel, letter, d / f"panel_{letter}_{slug}.png", integer_ticks=int_ticks)
            print(f"[v4] wrote {p}")


if __name__ == "__main__":
    main()
