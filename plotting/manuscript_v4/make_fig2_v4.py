# -*- coding: utf-8 -*-
"""
Compose manuscript v4 **Figure 2** -- "is it apo or holo, and what sets that".

Six panels, ST_100 only, from three generators:

    a  class composition across shared-evidence bins      old Fig 2b   figure1_main_REV4 col 2
    b  class prevalence vs shared-evidence cutoff k       old Fig 2d   figure1_main_REV4 col 4
    c  experimental evidence per pocket, by class         old Fig 2c   figure1_main_REV4 col 3
    d  classification prevalence vs evidence ratio        old Fig 3a   figure2_main_REV1 col 1
    e  pocket counts vs evidence ratio                    old Fig 3b   figure2_main_REV1 col 2
    f  probability ridgeline across ratio bins            old Fig 4b   Figure3A_ridgeline_REV2

Panel c is RESTORED. MN asked for it to be cut (C21/A5) and the first v4 pass complied; the author
put it back on 2026-08-24. Note it sits on the shared-evidence axis while d and e sit on the
evidence-ratio axis, so row 2 mixes two x variables -- the caption has to say which is which.

NOTHING IS RESCALED
-------------------
Every part is rendered with `--exact_canvas` at the exact millimetre size it occupies, so the file
on disk IS that size and this composer only *places* it. The previous version rendered parts at
whatever size came out of a tight crop and then resized whole rows to make them line up, which
scaled each row by a different factor and left nominally equal type at different sizes in the same
figure. A tight crop trims by however much outboard label a panel happens to carry, so its output
size is not knowable in advance -- that is the thing that forced the rescaling.

Sizes are millimetres throughout, from `common/figspec.py`, which also fixes the colours: panels
a/b/c come from a generator that drew apo/holo in the muted pair while d/e/f came from generators
using the brighter tab10 pair, so this figure previously carried two different blues for apo.

Usage (from the repository root; output folder: cfg.MANUSCRIPT_FIG_DIR, see mv4_paths.py):
    python plotting/manuscript_v4/make_fig2_v4.py --render            # render the parts, then compose
    python plotting/manuscript_v4/make_fig2_v4.py                     # compose the parts already rendered
    python plotting/manuscript_v4/make_fig2_v4.py --render --out_dir DIR --parts_dir DIR/parts
"""
import argparse
import os
import subprocess
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# roots, the generator interpreter and the figure spec all come from this repository
from mv4_paths import FIG_DIR, PARTS_DIR, PLOT_PYEXE, PUBLIC_REPO, figspec  # noqa: E402

V4 = str(FIG_DIR)
PARTS = str(PARTS_DIR)
DPI = 600

W = figspec.W2                 # 180 mm, double column
GAP = 1.5                      # mm between tiles

# PROPORTIONS (rev 3). Every panel here was wider than it was tall, c/d/e by about 3:2 and the
# ridgeline by 2.4:1, because each row's height was set by what fitted rather than by the shape
# the panel wanted. The row heights below are solved so that the AXES come out square, using each
# generator's own millimetre margins:
#
#   panel c   figure1_main_REV4 col 3, margins left 12.0 / right 1.5 / top 4.5 / bottom 9.0
#             -> axes = (C_W - 13.5) x (ROW2_H - 13.5), so C_W = ROW2_H gives a square
#   d and e   figure2_main_REV1 cols 1-2, margins left 9.0 / right 1.5 / top 4.5 / bottom 11.0,
#             wspace 0.30 -> axes width = (DE_W - 10.5) / 2.3, axes height = ROW2_H - 15.5
#
# Row 1 is not square and is not meant to be: a and b are half-page-wide time-series-like panels.
# It gets height instead -- 50 mm puts their axes at 74 x 36.5 mm (2.0:1, from 2.3:1).
ROW1_H = 50.0
ROW2_H = 59.0
ROW3_H = 70.0
C_W = ROW2_H                              # square axes: 45.5 x 45.5 mm
DE_W = 2.3 * (ROW2_H - 15.5) + 10.5       # square axes: 43.5 x 43.5 mm each
# The ridgeline is SQUARE (author's call, rev 4: empty space beats a stretched panel). Narrowing
# the canvas is the only lever on its proportions -- 23 ridges need the vertical room, so at any
# row height a 180 mm canvas gives a 151 mm-wide block. The ridgeline's own margins are
# millimetres now, so the width that squares the ridge block follows from the row height:
#   block width  = F_W - 11.0 (left) - 17.5 (strip + its gap)
#   block height = ROW3_H - 7.0 (top, the key) - 9.0 (bottom)
# It is centred on the row; the ~98 mm this leaves is deliberate white, not a layout accident.
F_W = (ROW3_H - 16.0) + 28.5


def mm2px(v):
    return int(round(v / 25.4 * DPI))


# (basename, script, args). Every entry passes --exact_canvas, so its size on disk is its size here.
def render_spec():
    def sz(w_mm, h_mm):
        return f"{w_mm / 25.4:.4f},{h_mm / 25.4:.4f}"
    return [
        # --hide_row_labels: with one threshold the rotated "ST 100%" strip down the left edge
        # names something the caption already states, and at 180 mm it fell outside the canvas
        # anyway -- it was costing 8.5 mm of left margin for ink that never reached the page.
        # --col2_overall (2026-10-07, author): panel a gains the AF-Apo / AF-Holo totals as an
        # 'overall' pair behind a dotted divider, like Figures 3a and 4a/b, on its own right-hand
        # scale; panel a widens and b narrows inside the same 180 mm strip.
        # --letters_at_ylabel (same day): letters up and left of each plot, on the left edge of its y
        # label, as in the other figures (f's generator already places it there).
        # --count_ylab_gap: "Pocket count" 2 mm further from panel a's "17.5k" ticks, which it touched.
        ("Fig2ab", "plotting/figure1_main_REV4.py",
         ["--columns", "2,4", "--thresholds", "ST_100", "--letter_start", "a",
          "--hide_row_labels", "--col2_overall", "--letters_at_ylabel", "--count_ylab_gap", "9.5",
          "--exact_canvas", "--fig_size", sz(W, ROW1_H)]),
        ("Fig2c", "plotting/figure1_main_REV4.py",
         ["--columns", "3", "--thresholds", "ST_100", "--letter_start", "c",
          "--hide_row_labels", "--letters_at_ylabel", "--exact_canvas", "--fig_size", sz(C_W, ROW2_H)]),
        ("Fig2de", "plotting/figure2_main_REV1.py",
         ["--columns", "1,2", "--thresholds", "ST_100", "--letter_start", "d",
          "--hide_row_labels", "--letters_at_ylabel", "--exact_canvas", "--fig_size", sz(DE_W, ROW2_H)]),
        ("Fig2f_ridgeline", "plotting/Figure3A_ridgeline_REV2.py",
         ["--thresholds", "ST_100", "--panel_letter", "f",
          # the "UBS - ST 100%" tag is stated before the figure in the manuscript now, and it was
          # the only thing between the top ridge and the plot border
          "--hide_threshold_annotation",
          "--exact_canvas", "--fig_size", sz(F_W, ROW3_H)]),
    ]


# where each rendered part goes, in mm from the top-left of the canvas
# Row 2 leaves ~10 mm between c and d/e rather than the 1.5 mm tile gap: c sits on the SHARED
# EVIDENCE axis and d/e on the EVIDENCE RATIO axis, so the two halves of the row are not one
# continuous scale, and the gap is where that reads. The ridgeline is centred on the row.
LAYOUT = [
    ("Fig2ab.png", 0.0, 0.0),
    ("Fig2c.png", 0.0, ROW1_H + GAP),
    ("Fig2de.png", W - DE_W, ROW1_H + GAP),
    ("Fig2f_ridgeline_ST_100.png", 0.5 * (W - F_W), ROW1_H + GAP + ROW2_H + GAP),
]
TOTAL_H = ROW1_H + GAP + ROW2_H + GAP + ROW3_H
OUT_BASE = "Figure2_apo_holo_and_what_sets_it"


def render_parts(repo, parts=PARTS):
    os.makedirs(parts, exist_ok=True)
    for base, script, extra in render_spec():
        cmd = [PLOT_PYEXE, os.path.join(repo, script),
               "--output_dir", parts, "--out_basename", base] + extra
        print("  ", os.path.basename(script), " ".join(extra[-4:]))
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAILED {script}\n{r.stdout}\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--render", action="store_true", help="regenerate the parts first")
    ap.add_argument("--out_dir", default=V4)
    ap.add_argument("--out_basename", default=OUT_BASE)
    ap.add_argument("--parts_dir", default=PARTS,
                    help="where parts are rendered and read (default: the live parts folder); "
                         "point it at a scratch folder to preview without touching the live parts")
    a = ap.parse_args()

    if a.render:
        print("rendering parts ->", a.parts_dir)
        render_parts(str(PUBLIC_REPO), a.parts_dir)

    sheet = Image.new("RGB", (mm2px(W), mm2px(TOTAL_H)), "white")
    for name, x_mm, y_mm in LAYOUT:
        path = os.path.join(a.parts_dir, name)
        if not os.path.isfile(path):
            sys.exit(f"missing part (run with --render): {path}")
        im = Image.open(path).convert("RGB")
        # placed, never resized -- if a part is not the size it was asked for, that is a bug in the
        # generator's --exact_canvas handling and must be fixed there, not papered over here
        print(f"  place {name:32s} {im.width/DPI*25.4:6.1f} x {im.height/DPI*25.4:5.1f} mm "
              f"at ({x_mm:5.1f}, {y_mm:5.1f})")
        sheet.paste(im, (mm2px(x_mm), mm2px(y_mm)))

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, a.out_basename + ".png")
    sheet.save(out, dpi=(DPI, DPI))
    print(f"\nwrote {out}\n  {W:.0f} x {TOTAL_H:.0f} mm"
          f"{'' if TOTAL_H <= figspec.H_MAX else '   ! OVER the %.0f mm page limit' % figspec.H_MAX}")


if __name__ == "__main__":
    main()
