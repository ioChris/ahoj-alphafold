# -*- coding: utf-8 -*-
"""
Docking analog of manuscript **Figure 5**, produced with the *actual* Figure-5
plotting code — `03_alphafill/ubs/plot_clash_ridgelines.py::plot_split_half_violins_vertical`
(split half-violins by AF class across separation/probability bins, dashed
median+quartile lines, a "Worst 10%" tail subpanel, the exact Figure-5 fills
`#8DB9DC`/`#F4BD84`). We reuse that function verbatim; only the *data* changes.

The Figure-5 function is built for a NON-NEGATIVE "badness" metric (clash / RMSD,
where higher = worse, plotted worse-upward). Docking affinity is negative and
lower = better, so we feed it the monotonic transform

    aff_penalty = clip(affinity_top, floor, ceil) - floor     (>= 0)

which lands on exactly the same orientation (worse binders = larger penalty =
up, so "Worst 10%" = weakest binders — the correct docking reading), then
monkeypatch (i) the y-axis label and (ii) a tick formatter so the axis still
reads in real kcal/mol (tick value v  ->  v + floor).

Run in the `plotting` env. Output: <out_basename>_{separation,probability}.png/.svg.

v4 SI panel (2026-09-15). The manuscript v4 SI draws this panel the way Figure 4a/b
are drawn -- exact 88 x 62 mm canvas, "overall" LAST behind the dotted divider, the
class key inside the top-left -- through the flags below. Two of them fix what the
July render got wrong: the floor was a fixed -13 kcal/mol while only 0.7% of pockets
sit below -12, so two kcal/mol of empty axis sat under every violin; and the plotter's
8% headroom was less than the key needed, so the key sat on the tallest violin.
`--floor -12 --headroom 1.15` removes the dead band and clears the key. Every default
is the July behaviour, so the original fig5real_* files regenerate unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --- make the Figure-5 module and our per-pocket helper importable ---
_HERE = Path(__file__).resolve()
for _root in _HERE.parents:
    if (_root / "common" / "config.py").is_file():
        REPO = _root
        break
sys.path.insert(0, str(REPO))                              # common.figspec
sys.path.insert(0, str(REPO / "03_alphafill" / "ubs"))     # plot_clash_ridgelines + confidence_labeling
sys.path.insert(0, str(_HERE.parent))                      # analyze_docking_by_confidence.per_pocket

import plot_clash_ridgelines as pcr                        # noqa: E402  the real Figure-5 code
import analyze_docking_by_confidence as adbc               # noqa: E402  per_pocket() + bin definitions
from common import figspec                                 # noqa: E402

AFF_FLOOR = -13.0     # kcal/mol mapped to penalty 0 (best plotted binder)
AFF_CEIL = 1.0        # kcal/mol mapped to penalty 14 (worst; caps the degenerate tail)
PENALTY = "aff_penalty"


def build_docking_table(merged_csv: Path, group: int,
                        floor: float = AFF_FLOOR, ceil: float = AFF_CEIL) -> pd.DataFrame:
    df = pd.read_csv(merged_csv, low_memory=False)
    g = adbc.per_pocket(df, group)                         # per-pocket median_aff, class, sep_level, prob_level
    g = g[g.pocket_class.isin(["apo", "holo"])].copy()
    aff = pd.to_numeric(g["median_aff"], errors="coerce")
    g[PENALTY] = aff.clip(floor, ceil) - floor             # >= 0, higher = worse binder
    g["matched"] = True                                    # Figure-5 code filters on this AlphaFill flag
    return g


def _install_patches(floor: float = AFF_FLOOR, headroom: float = 1.0):
    """Monkeypatch the Figure-5 module so the penalty axis reads in kcal/mol.

    `headroom` stretches the violin axes' upper limit by that factor. The plotter sets
    (0, upper) exactly once, with upper = 1.08 x the capped maximum; the class key needs
    more than that when every violin runs to the top, as the affinity ones do.
    """
    _orig_label = pcr.metric_label

    def _label(metric):
        return "Top-pose affinity (kcal/mol)" if metric == PENALTY else _orig_label(metric)
    pcr.metric_label = _label

    _orig_subplots = pcr.plt.subplots

    def _subplots(*a, **k):
        fig, axes = _orig_subplots(*a, **k)
        try:                                               # axes = (violin ax, tail ax); relabel violin y
            ax = axes[0] if isinstance(axes, (tuple, list, np.ndarray)) else axes
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v + floor:.0f}"))
            if headroom != 1.0:
                _orig_set_ylim = ax.set_ylim

                def _set_ylim(*ya, **yk):
                    if len(ya) == 2 and not yk and ya[0] == 0.0:
                        return _orig_set_ylim(0.0, ya[1] * headroom)
                    return _orig_set_ylim(*ya, **yk)
                ax.set_ylim = _set_ylim
        except Exception:
            pass
        return fig, axes
    pcr.plt.subplots = _subplots
    return lambda: (setattr(pcr, "metric_label", _orig_label),
                    setattr(pcr.plt, "subplots", _orig_subplots))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--which", choices=["separation", "probability", "both"], default="both")
    ap.add_argument("--out_basename", default="fig5real")
    # --- v4 SI panel options; defaults reproduce the July render ---
    ap.add_argument("--all_position", choices=["first", "last"], default="first",
                    help="where the pooled group sits; Figure 4 puts it LAST behind a divider")
    ap.add_argument("--floor", type=float, default=AFF_FLOOR,
                    help="kcal/mol mapped to the axis bottom; affinities below it are clipped there")
    ap.add_argument("--ceil", type=float, default=AFF_CEIL)
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="stretch the violin axes' top by this factor (1.15 clears the class key)")
    ap.add_argument("--canvas_mm", default=None, metavar="W,H",
                    help="render at this exact canvas (Figure 4a/b are 88,62); implies no bbox trimming")
    ap.add_argument("--letter", default=None, help="panel letter drawn by the plotter, e.g. a")
    ap.add_argument("--no_title", action="store_true", help="no in-figure title (the caption carries it)")
    ap.add_argument("--stacked_counts", action="store_true",
                    help="n row as apo over holo (five-digit counts collide side by side at 88 mm)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.canvas_mm:
        w_mm, h_mm = (float(v) for v in args.canvas_mm.split(","))
        pcr.EXACT_CANVAS = True
        pcr.FIG_SIZE_IN = (w_mm * figspec.MM, h_mm * figspec.MM)

    g = build_docking_table(args.merged_csv, args.group, floor=args.floor, ceil=args.ceil)
    n_clipped = int((pd.to_numeric(g["median_aff"], errors="coerce") < args.floor).sum())
    print(f"[data] {len(g):,} apo/holo pockets; {n_clipped:,} ({n_clipped / len(g) * 100:.2f}%) "
          f"below the {args.floor:g} kcal/mol floor are drawn at the floor")

    jobs = [("sep_level", "separation"), ("prob_level", "probability")]
    if args.which != "both":
        jobs = [j for j in jobs if j[1] == args.which]

    restore = _install_patches(floor=args.floor, headroom=args.headroom)
    try:
        for level_col, name in jobs:
            title = "" if args.no_title else \
                f"Docking top-pose affinity by AF class across {name} bins  (G{args.group})"
            pcr.plot_split_half_violins_vertical(
                g, level_col=level_col, level_order=list(pcr.VERBAL_ORDER),
                outpath=args.out_dir / f"{args.out_basename}_{name}.png",
                title=title,
                metric=PENALTY,
                summary_style="dashed",            # Figure-5 uses dashed median/quartile lines
                include_all_group=True,
                all_group_position=args.all_position,
                letter=args.letter,
                counts_stacked=args.stacked_counts,
            )
            print(f"[saved] {args.out_basename}_{name}.png / .svg")
    finally:
        restore()
    print(f"[SAVE] {args.out_dir}")


if __name__ == "__main__":
    main()
