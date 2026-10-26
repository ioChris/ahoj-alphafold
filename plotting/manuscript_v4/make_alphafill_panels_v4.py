# -*- coding: utf-8 -*-
"""
Render the AlphaFill panels of manuscript v4 **Figure 4** -- "when does it matter".

Two panels, not the four of the old Figure 5: clash score and clash count. The two RMSD panels
(global, local) drop out of the main text; SI Figure S7 carries all four under the
permissive `any` scope, and decision O2 sends the main-text pair there as well.

    a  average clash count   old Fig 5a
    b  average clash score    old Fig 5c

Scope, as in v3 and unchanged: ST_100 UBSs with at least one same-UniProt AlphaFill match at
alignment identity >= 0.30. Groups are "All" plus the five apo-holo separation bins.

TWO TRAPS this script exists to avoid, both verified:

  * The old make_fig5_dashed.py named its intermediates panel_{LETTER}_{metric}.png. Windows is
    case-insensitive, so a stale panel_A_*.png from an uppercase run is silently OVERWRITTEN rather
    than duplicated, and nothing errors either way. This writes lowercase names into a v4-only
    directory and clears it first.
  * show_counts was pinned to the first tuple. If the metric order ever drifted, the n annotation
    would land on the wrong panel silently. Here the panel carrying it is named explicitly.

The panel letters are drawn by this composer rather than by the plotter, because
plot_split_half_violins_vertical has no letter argument -- unlike the Figure 2 parts, which draw
their own. Recorded here so the difference is deliberate rather than discovered later.
(The letters are now passed to the plotter as real 8 pt text, `letter=` below.)

Input: <cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR>/per_ubs_summary_<scope>.csv.

Usage (from the repository root; make_fig4_v4.py --render calls it with --out_dir <parts>):
    python plotting/manuscript_v4/make_alphafill_panels_v4.py --out_dir DIR
"""
import argparse
import os

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import FIG_DIR, PARTS_DIR, PUBLIC_REPO, cfg, figspec  # noqa: E402

# the plotter is imported at module level, from THIS repository's 03_alphafill/ubs
UBS = os.path.join(str(PUBLIC_REPO), "03_alphafill", "ubs")
for _p in (str(PUBLIC_REPO), UBS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_clash_ridgelines as pcr  # noqa: E402

ANALYSIS = Path(cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR)      # <work_dir>/alphafill_unified/run2/analysis_v1
V4 = Path(FIG_DIR)
PARTS = Path(PARTS_DIR) / "fig4_alphafill"

# (metric, panel letter, human title). Order IS the panel order.
# COUNT before SCORE, reversing the v3 order. The clash-SCORE violins run to the top of their
# axes at every separation bin, so the class key sat on top of the first pair of violins; the
# clash-COUNT panel has an empty upper-left corner and can host it. Swapped 2026-08-24.
PANELS = [
    ("avg_clash_count", "a", "Average clash count"),
    ("avg_clash",       "b", "Average clash score"),
]
COUNTS_ON = "a"          # which panel carries the n = (apo | holo) annotation -- named, not implied
LEGEND_ON = "a"          # one class key for the pair, not one per panel
THRESHOLD = "ST_100"
# "Overall" sits at the RIGHT end, matching the detection figure, where it is fenced off from the
# binned groups by a divider. At the left it read as the first separation bin.
ALL_POSITION = "last"
PANEL_W_MM, PANEL_H_MM = 88.0, 62.0   # each panel's true size in the composed figure
DPI = 600


def font(sz):
    for f in ("arialbd.ttf", "Arial_Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scope", default="same_uniprot_ge_0p30",
                    help="per-UBS summary suffix; same_uniprot_ge_0p30 is the manuscript headline "
                         "scope (the permissive `any` scope is SI Figure S7)")
    ap.add_argument("--out_dir", default=str(V4))
    ap.add_argument("--parts_dir", default=str(PARTS),
                    help="folder for the two single-panel intermediates (cleared first)")
    ap.add_argument("--out_basename", default="Figure4ab_alphafill_clash")
    a = ap.parse_args()

    src = ANALYSIS / f"per_ubs_summary_{a.scope}.csv"
    if not src.is_file():
        sys.exit(f"missing AlphaFill summary: {src}")
    per = pd.read_csv(src)
    per["threshold"] = per["threshold"].astype(str).str.strip()
    sub = per.loc[per["threshold"] == THRESHOLD].copy()
    print(f"{src.name}: {len(per):,} rows, {THRESHOLD} -> {len(sub):,}")

    # Clear stale intermediates -- an uppercase panel_A_*.png from an earlier run would be
    # silently overwritten on case-insensitive Windows rather than flagged. Remove the FILES, not
    # the directory: on a cloud-synced folder the sync client can hold a handle on the folder
    # and makes rmdir fail with WinError 5 while the files themselves delete fine.
    parts = Path(a.parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    for f in parts.glob("*"):
        if f.is_file():
            try:
                f.unlink()
            except OSError as e:
                print(f"  ! could not remove stale {f.name}: {e}")

    paths = []
    for metric, letter, _title in PANELS:
        outp = parts / f"panel_{letter}_{metric}.png"
        pcr.EXACT_CANVAS = True
        pcr.FIG_SIZE_IN = (PANEL_W_MM * figspec.MM, PANEL_H_MM * figspec.MM)
        pcr.plot_split_half_violins_vertical(
            sub, level_col="separation_level_db", level_order=pcr.VERBAL_ORDER,
            outpath=outp, title="",          # no per-panel title; the letter is added on assembly
            metric=metric, summary_style="dashed", show_whiskers=True,
            # counts and legend on ONE panel of the pair. The counts row still occupies its
            # space on the other panel, so both share a baseline -- printing them on one and
            # reclaiming the space on the other is what left the two panels at different heights.
            show_counts=(letter == COUNTS_ON),
            show_legend=(letter == LEGEND_ON),
            include_all_group=True, all_group_position=ALL_POSITION,
            letter=letter,          # real 8 pt text, not painted on afterwards
        )
        paths.append(outp)
        print("  wrote", outp.name)

    # placed, never resized: both panels were rendered at PANEL_W_MM x PANEL_H_MM
    ims = [Image.open(p).convert("RGB") for p in paths]
    gap = int(round(2.0 / 25.4 * DPI))
    W = sum(im.width for im in ims) + gap * (len(ims) - 1)
    H = max(im.height for im in ims)
    sheet = Image.new("RGB", (W, H), "white")
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width + gap

    out = Path(a.out_dir) / (a.out_basename + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, dpi=(DPI, DPI))
    print(f"wrote {out}  {W/DPI*25.4:.1f} x {H/DPI*25.4:.1f} mm")


if __name__ == "__main__":
    main()
