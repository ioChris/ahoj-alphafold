# -*- coding: utf-8 -*-
"""
Compose manuscript v4 **Figure 4** -- "when does it matter".

Five panels, three tasks, one pattern:

    a  AlphaFill average clash count,  AF-Apo vs AF-Holo across separation   (+ Overall)
    b  AlphaFill average clash score,   same layout
    c  ALL THREE TASKS on one unit-free axis: Cliff's delta by separation tier
    d  the per-predictor detection gap, in percentage points
    e  winning-probability / density / separation tier composition, ST_100

Panel c is the point of the section. AlphaFill fit, detection and docking were previously drawn in
three different visual grammars -- half-violins, a line of gaps, and another line of gaps on a
different denominator -- which hid the very thing the section claims: that three methods sharing
almost nothing trace the same gradient. On one delta axis it is visible at a glance.

Panels c and d are NOT on the same scale and are not labelled as if they were: c is Cliff's delta
(unit-free, needed because clash score and success rate have no common unit), d is the gap in
percentage points (the quantity the manuscript quotes, consensus 3.4 -> 19.1). For a binary
outcome the two coincide, but consensus is the mean of three binaries and its delta is not its gap.

NOTHING IS RESCALED. Every part is rendered at the millimetre size it occupies and placed.

Usage (from the repository root; the manuscript figure is the AFDB v4 re-run):
    AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_fig4_v4.py --render
    AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_fig4_v4.py   # compose only
"""
import argparse
import os
import subprocess
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import FIG_DIR, PARTS_DIR, PLOT_PYEXE, PUBLIC_REPO, figspec  # noqa: E402

# This composer calls a sibling tool. Resolve it from THIS file's directory, so a worktree or a
# second clone renders its parts from its own code.
TOOLS = os.path.dirname(os.path.abspath(__file__))

V4 = str(FIG_DIR)
PARTS = str(PARTS_DIR)
DPI = 600

W = figspec.W2          # 180 mm
GAP = 2.0               # mm
ROW1_H = 62.0           # the AlphaFill pair
ROW2_H = 58.0           # the two separation panels
ROW3_H = 46.0           # the tier strip
CD_W = (W - GAP) / 2    # panels c and d share row 2

TOTAL_H = ROW1_H + GAP + ROW2_H + GAP + ROW3_H
OUT_BASE = "Figure4_when_does_it_matter"


def mm2px(v):
    return int(round(v / 25.4 * DPI))


def render_parts(repo, boot, parts=PARTS):
    os.makedirs(parts, exist_ok=True)
    jobs = [
        # the AlphaFill pair composes itself (both panels, one legend, shared baseline)
        [PLOT_PYEXE, os.path.join(TOOLS, "make_alphafill_panels_v4.py"), "--out_dir", parts,
         "--parts_dir", os.path.join(parts, "fig4_alphafill"),
         "--out_basename", "Figure4ab_alphafill_clash"],
        [PLOT_PYEXE, os.path.join(repo, "06_lbs_generality", "analysis",
                             "plot_separation_gap_panels.py"),
         "--out_dir", parts, "--panel", "both", "--letter", "c,d",
         "--fig_size", f"{CD_W:.2f},{ROW2_H:.2f}", "--boot", str(boot)],
        [PLOT_PYEXE, os.path.join(repo, "plotting", "figure3C_bars_REV18.py"),
         "--thresholds", "ST_100", "--show_all_group", "--hide_class_legend",
         "--panel_letter", "e",
         "--fig_size", f"{W / 25.4:.4f},{ROW3_H / 25.4:.4f}",
         "--output_dir", parts, "--out_basename", "Fig4e_confidence_tiers"],
    ]
    for cmd in jobs:
        print("  ", os.path.basename(cmd[1]))
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"FAILED {cmd[1]}\n{r.stdout}\n{r.stderr}")


LAYOUT = [
    ("Figure4ab_alphafill_clash.png", 0.0, 0.0),
    ("Figure4_crosstask.png", 0.0, ROW1_H + GAP),
    ("Figure4_predictors.png", CD_W + GAP, ROW1_H + GAP),
    ("Fig4e_confidence_tiers.png", 0.0, ROW1_H + GAP + ROW2_H + GAP),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--render", action="store_true", help="regenerate the parts first")
    ap.add_argument("--parts_dir", default=PARTS,
                    help="where parts are rendered and read (default: <figure dir>/parts)")
    ap.add_argument("--out_dir", default=V4)
    ap.add_argument("--out_basename", default=OUT_BASE)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()

    if a.render:
        print("rendering parts ->", a.parts_dir)
        render_parts(str(PUBLIC_REPO), a.boot, a.parts_dir)

    sheet = Image.new("RGB", (mm2px(W), mm2px(TOTAL_H)), "white")
    for name, x_mm, y_mm in LAYOUT:
        path = os.path.join(a.parts_dir, name)
        if not os.path.isfile(path):
            sys.exit(f"missing part (run with --render): {path}")
        im = Image.open(path).convert("RGB")
        print(f"  place {name:34s} {im.width/DPI*25.4:6.1f} x {im.height/DPI*25.4:5.1f} mm "
              f"at ({x_mm:5.1f}, {y_mm:5.1f})")
        sheet.paste(im, (mm2px(x_mm), mm2px(y_mm)))

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, a.out_basename + ".png")
    sheet.save(out, dpi=(DPI, DPI))
    over = "" if TOTAL_H <= figspec.H_MAX else f"   ! OVER the {figspec.H_MAX:.0f} mm page limit"
    print(f"\nwrote {out}\n  {W:.0f} x {TOTAL_H:.0f} mm{over}")


if __name__ == "__main__":
    main()
