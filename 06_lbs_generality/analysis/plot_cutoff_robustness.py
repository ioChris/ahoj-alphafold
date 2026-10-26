# -*- coding: utf-8 -*-
"""
SI robustness figure: the AF-Holo>AF-Apo TNS gradient across DCA-tolerance choices.
===================================================================================

Turns the size-adaptive + floored re-scoring (rescore_size_adaptive_{tolerance,floor}.py)
into a supplementary figure + a consolidated SI table. Message: the holo>apo detection
gradient persists for EVERY predictor under EVERY meaningful distance criterion -- fixed 4 A,
size-adaptive k*Rg (k=0.5/1/2), and the floored max(4, 0.5*Rg) -- so it is not an artifact of
the 4 A cutoff. Columns = the three re-scorable tools; each panel shows apo & holo TNS across
the cutoff schemes with the gap shaded. Native P2Rank (SAS-eval, not re-scorable) is out of scope.

The **SI deliverable is the full-population figure** (1 x 3). A full + pilot variant (2 x 3) is
also written for INTERNAL reference only (the balanced pilot is a size-control check, not an SI
panel). `--manuscript_dir <dir>` also copies the SI figure + caption, prefixed by --manuscript_name
(default `SI_lbs_cutoff_robustness`), into that dir. Convention: SI figures live under
`<project>/manuscript/figures/SI/` (SI_ prefix, descriptive, NO figure number -- the final
`Figure Sx` is assigned at SI finalization), kept apart from the numbered main figures.

Inputs  : work_dir/lbs_generality/analysis_v1/tables/lbs_size_adaptive_{tolerance,floor}_long.csv
Outputs : work_dir/lbs_generality/analysis_v1/plots/lbs_cutoff_robustness.{png,pdf,svg}          (SI, full only)
          work_dir/lbs_generality/analysis_v1/plots/lbs_cutoff_robustness_full_pilot.png         (internal ref)
          work_dir/lbs_generality/analysis_v1/tables/lbs_cutoff_robustness_SI_table.csv          (full only)
          work_dir/lbs_generality/analysis_v1/tables/lbs_cutoff_robustness_full_pilot_table.csv  (internal ref)

Base env (pandas + numpy + matplotlib):
    python 06_lbs_generality/analysis/plot_cutoff_robustness.py [--manuscript_dir <path>]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Preferred manuscript fonts (Helvetica Neue / TeX Gyre Heros); later families are glyph
# fallbacks (Å, ·, − live in DejaVu Sans if the primary lacks them). Editable text in SVG/PDF.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["svg.fonttype"] = "none"   # keep <text> as text (selectable / restylable)
plt.rcParams["pdf.fonttype"] = 42       # embed TrueType so the PDF is portable + editable

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# cutoff schemes to show, in order, with pretty x labels. Keys are the `scoring` values;
# the second element says which long table to read them from.
SCHEMES = [
    ("fixed_4A",       "tol",  "4 Å\n(fixed)"),
    ("rg_k0.5",        "tol",  "0.5·Rg"),
    ("rg_k0.5_floor4", "floor", "max(4,\n0.5·Rg)"),
    ("rg_k1",          "tol",  "1.0·Rg"),
    ("rg_k2",          "tol",  "2.0·Rg"),
]
# Figure column labels carry NO per-tool qualifier: the harmonized heavy-atom DCA/TNS metric is
# applied identically to all three tools, so tagging only P2Rank "(harmonized)" would wrongly imply
# it was evaluated differently. The harmonized-vs-native distinction is kept in the caption's
# internal note + RESULTS.md / DECISIONS.md, not on the manuscript figure.
TOOLS = [("ifsitepred", "IF-SitePred"), ("p2rank", "P2Rank"), ("fpocket", "fpocket")]
POP_LABELS = {"full": "Full population", "pilot": "Balanced pilot (internal ref)"}

APO_C, HOLO_C, BAND_C = "#3B6FB6", "#D1495B", "#F2C14E"  # apo / holo / gap band

CAPTION = """\
Supplementary Figure S?. The AF-Holo > AF-Apo binding-site detection gradient is robust to the DCA distance tolerance.

Top-N-site (TNS) success on AlphaFold DB models for AF-Apo (blue) and AF-Holo (red) pockets, for three ligand-binding-site predictors with unrelated inductive biases: IF-SitePred (ESM-IF1 inverse-folding embeddings), P2Rank (supervised random forest) and fpocket (unsupervised Voronoi alpha-spheres). All three predictors are scored with a single, identical heavy-atom distance metric: the distance from each predicted pocket centre to the nearest union binding-site-residue heavy atom on the AlphaFold model, counted a success when a matching pocket is ranked within the number of pockets expected for that UniProt entry (TNS). Five distance criteria are compared: the fixed 4 Angstrom cutoff used in the main analysis; three per-site, size-adaptive cutoffs equal to 0.5x, 1.0x and 2.0x the pocket radius of gyration (Rg); and the floored criterion max(4 Angstrom, 0.5 x Rg) (grey band). The shaded region is the holo-minus-apo gap, with its size in percentage points annotated above each AF-Holo point. The gap remains positive for every predictor under every distance criterion, showing the gradient is not an artifact of the 4 Angstrom choice. Because the median Rg is about 7.5 Angstrom, absolute success increases with the cutoff and saturates at 2.0x Rg (median about 15 Angstrom), which compresses all gaps; the floored max(4 Angstrom, 0.5 x Rg) reproduces the fixed-4 Angstrom success levels while remaining size-normalized. n = 34,900 AF-Apo and 51,751 AF-Holo pockets (the full ST_100 population with a defined Rg). P2Rank's native surface-point (SAS) evaluation, which is not applicable to the other tools, is reported separately as a labelled reference and is not shown here.

---
[INTERNAL NOTE - not for the manuscript]
"P2Rank" in this figure = P2Rank's own predicted pocket centres re-scored with the shared heavy-atom DCA/TNS metric (the same yardstick used for IF-SitePred and fpocket); it was labelled "P2Rank (harmonized)" in internal tables. It is NOT P2Rank's native SAS-point evaluation (apo 66.1 / holo 84.8), which is looser, PDB-incomparable, and kept only as a labelled reference. The harmonization is applied identically to every tool, so no per-tool qualifier is shown on the figure. Figure = full population only; a full+pilot variant exists for internal reference (the balanced 1,494 pilot is a size control, not an SI panel).
"""


def load_long() -> pd.DataFrame:
    t = cfg.LBS_ANALYSIS_DIR / "tables"
    tol = pd.read_csv(t / "lbs_size_adaptive_tolerance_long.csv")
    fl = pd.read_csv(t / "lbs_size_adaptive_floor_long.csv")
    tol["src"], fl["src"] = "tol", "floor"
    keep = ["tool", "scoring", "population", "apo_tns", "apo_n", "holo_tns", "holo_n",
            "gap_holo_minus_apo", "median_cutoff_A", "src"]
    df = pd.concat([tol[keep], fl[keep]], ignore_index=True)
    wanted = {(s, src) for s, src, _ in SCHEMES}
    df = df[[(s, src) in wanted for s, src in zip(df["scoring"], df["src"])]].copy()
    return df


def _series(df, tool, pop, col):
    out = []
    for scoring, src, _ in SCHEMES:
        r = df[(df["tool"] == tool) & (df["population"] == pop)
               & (df["scoring"] == scoring) & (df["src"] == src)]
        out.append(np.nan if r.empty else float(r.iloc[0][col]))
    return np.array(out)


def render(df: pd.DataFrame, pops: list[str], out_stem: Path, exts: tuple[str, ...]) -> None:
    x = np.arange(len(SCHEMES))
    xlabels = [lab for _, _, lab in SCHEMES]
    nrows = len(pops)
    fig, axes = plt.subplots(nrows, len(TOOLS), figsize=(12.6, 3.9 * nrows + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    for i, pop in enumerate(pops):
        for j, (tool, tool_lab) in enumerate(TOOLS):
            ax = axes[i, j]
            apo, holo, gap = (_series(df, tool, pop, "apo_tns"), _series(df, tool, pop, "holo_tns"),
                              _series(df, tool, pop, "gap_holo_minus_apo"))
            ax.fill_between(x, apo, holo, color=BAND_C, alpha=0.35, zorder=1)
            ax.plot(x, apo, "-o", color=APO_C, lw=2, ms=6, zorder=3)
            ax.plot(x, holo, "-o", color=HOLO_C, lw=2, ms=6, zorder=3)
            for k in range(len(x)):
                if np.isfinite(gap[k]):
                    ax.annotate(f"+{gap[k]:.0f}", (x[k], holo[k]), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=8, color=HOLO_C, fontweight="bold")
            ax.axvspan(x[2] - 0.35, x[2] + 0.35, color="0.85", alpha=0.5, zorder=0)  # recommended scheme
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.grid(axis="y", ls=":", alpha=0.5)
            n_apo = int(np.nanmax(_series(df, tool, pop, "apo_n")))
            n_holo = int(np.nanmax(_series(df, tool, pop, "holo_n")))
            if i == 0:
                ax.set_title(tool_lab, fontsize=11, fontweight="bold", pad=8)
            # top-left corner is free in every panel (lines sit below ~90%); avoids the fixed-4 A
            # line at bottom-left.
            ax.text(0.03, 0.96, f"n = {n_apo:,} apo / {n_holo:,} holo", transform=ax.transAxes,
                    fontsize=7.5, color="0.35", va="top", ha="left")
            if j == 0:
                lab = "TNS success (%)" if nrows == 1 else f"{POP_LABELS[pop]}\n\nTNS success (%)"
                ax.set_ylabel(lab, fontsize=10)
    axes[0, 0].set_ylim(28, 100)

    handles = [plt.Line2D([], [], color=HOLO_C, marker="o", lw=2, label="AF-Holo"),
               plt.Line2D([], [], color=APO_C, marker="o", lw=2, label="AF-Apo"),
               Patch(facecolor=BAND_C, alpha=0.35, label="holo − apo gap (pp)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02 if nrows == 1 else 1.005), fontsize=10)
    fig.suptitle("Robustness of the AF-Holo > AF-Apo detection gradient to the DCA distance tolerance",
                 y=1.10 if nrows == 1 else 1.055, fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.05 if nrows == 1 else -0.02,
             "Per-site cutoff = k·Rg (size-adaptive) or max(4, 0.5·Rg) (floored); grey band marks the "
             "recommended max(4, 0.5·Rg).\nThe gradient stays positive for every tool and every cutoff.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if nrows == 1 else 0.98))
    for ext in exts:
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def si_table(df: pd.DataFrame, pops: list[str]) -> pd.DataFrame:
    si = df[df["population"].isin(pops)].copy()
    si["scheme"] = pd.Categorical(si["scoring"].map({s: lab.replace("\n", " ") for s, _, lab in SCHEMES}),
                                  categories=[lab.replace("\n", " ") for _, _, lab in SCHEMES], ordered=True)
    si["tool"] = pd.Categorical(si["tool"], categories=[t for t, _ in TOOLS], ordered=True)
    si["population"] = pd.Categorical(si["population"], categories=["full", "pilot"], ordered=True)
    si = si.sort_values(["population", "tool", "scheme"])
    return si[["population", "tool", "scheme", "median_cutoff_A", "apo_tns", "apo_n",
               "holo_tns", "holo_n", "gap_holo_minus_apo"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="SI robustness figure (full-population) + internal full+pilot variant.")
    ap.add_argument("--manuscript_dir", type=Path, default=None,
                    help="If given, also copy the full-population SI figure + caption into this dir, "
                         "prefixed by --manuscript_name. Convention: SI figures go under "
                         "<project>/manuscript/figures/SI/ (kept apart from the numbered main figures).")
    ap.add_argument("--manuscript_name", default="SI_lbs_cutoff_robustness",
                    help="File stem for the manuscript copy (SI_ prefix, descriptive, no figure number; "
                         "assign the final 'Figure Sx' at SI finalization).")
    args = ap.parse_args()

    plots_dir = cfg.LBS_ANALYSIS_DIR / "plots"
    tables_dir = cfg.LBS_ANALYSIS_DIR / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = load_long()

    # SI deliverable: full population only (1 x 3), all three formats.
    render(df, ["full"], plots_dir / "lbs_cutoff_robustness", ("png", "pdf", "svg"))
    # Internal reference: full + balanced pilot (2 x 3), png only.
    render(df, ["full", "pilot"], plots_dir / "lbs_cutoff_robustness_full_pilot", ("png",))

    si_full = si_table(df, ["full"])
    si_full.to_csv(tables_dir / "lbs_cutoff_robustness_SI_table.csv", index=False, encoding="utf-8-sig")
    si_table(df, ["full", "pilot"]).to_csv(
        tables_dir / "lbs_cutoff_robustness_full_pilot_table.csv", index=False, encoding="utf-8-sig")

    # caption text saved next to the figure (SI caption + a delimited internal note)
    (plots_dir / "lbs_cutoff_robustness_caption.txt").write_text(CAPTION, encoding="utf-8")

    copied = []
    if args.manuscript_dir is not None:
        args.manuscript_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            dst = args.manuscript_dir / f"{args.manuscript_name}.{ext}"
            shutil.copyfile(plots_dir / f"lbs_cutoff_robustness.{ext}", dst)
            copied.append(dst)
        cap_dst = args.manuscript_dir / f"{args.manuscript_name}_caption.txt"
        cap_dst.write_text(CAPTION, encoding="utf-8")
        copied.append(cap_dst)

    print(f"[SAVE] {plots_dir / 'lbs_cutoff_robustness.png'} (+ pdf, svg)  [SI, full population]")
    print(f"[SAVE] {plots_dir / 'lbs_cutoff_robustness_full_pilot.png'}  [internal ref]")
    print(f"[SAVE] {tables_dir / 'lbs_cutoff_robustness_SI_table.csv'}  [SI, full population]")
    print(f"[SAVE] {plots_dir / 'lbs_cutoff_robustness_caption.txt'}")
    for d in copied:
        print(f"[COPY] {d}")
    print(si_full.to_string(index=False))


if __name__ == "__main__":
    main()
