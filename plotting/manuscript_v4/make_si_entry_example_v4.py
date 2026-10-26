# -*- coding: utf-8 -*-
"""
Render and compose SI Figure S9 (the authors' build called it FigureS10_entry_example): the worked AHoJ-DB entry
2yiw A I46 1354 (MAPK14 = p38alpha, UniProt Q16539).

    a  KDE violins of the four pocket metrics       figure_pocket_example.py (draw_violin_panel), classification code
    b  per-metric confidence bars                   "                                   (draw_bar_group)
    c  overall pocket confidence bars               "
    d  W197 / M198 in 5XYY (apo), 2YIW (holo query) and the AlphaFold DB v4 model, with the query's ligand I46
                                                    render_si_entry_pymol_v4.py (the v3 panel's PyMOL session; py311 env)

This replaces the static `Figure1_example.png` of draft v3 (the former main Figure 1), whose panels a-c were this
same parse-ahoj generator's output with uppercase 11 pt letters, and whose panel d was three PyMOL snapshots of
3000 x 2250 px shrunk onto the page with a PowerPoint "D".

a-c are drawn by the classification code's own functions on its own grid -- the per-entry figure the pipeline generates and the
web resource serves -- so they are identical to `figure_pocket_example.py`'s output except for the letters, which
are lowercase, bold, 8 pt here (the house rule; common/figspec.py in the public repo). Run in the `plotting` env:
it reproduces the July render of a-c pixel for pixel (matplotlib 3.10; the base env's 3.9 is not checked).

d keeps the v3 panel: the same PyMOL session, camera and render settings, and the same layout (apo and holo side by
side, the model below between them; titles and residue labels where v3 had them, measured on the v3 file). Each view
is now ray-traced at the size of its frame on the page (600 dpi) and placed with `figimage`, never resampled, and the
text is drawn here in the figure's font. New against v3: a key line saying what the grey ligand is.

The canvas is the classification code's: 180 x 234 mm, drawn at final size. Outputs:
<cfg.MANUSCRIPT_FIG_DIR>/SI/FigureS9_entry_example.{png,pdf} and the PyMOL parts in
<cfg.MANUSCRIPT_FIG_DIR>/parts/si_entry_example/ (with render.json).

Inputs: the classification code (figure_pocket_example.py, kde_plot.py, kde_api.py, run_types.py, configs/, fonts/) in
00_classification/kde_classifier/ (override: AHOJ_AF_KDE_CLASSIFIER_DIR); the entry's statistics folder under
cfg.EXAMPLE_ENTRIES_DIR (<example_entries_dir>/*/2yiw-A-I46-1354/statistics); the PyMOL session cfg.ENTRY_EXAMPLE_PSE;
a python with open-source PyMOL in AHOJ_AF_PYMOL_PYEXE.

Usage (from the repository root, plotting env):
    python plotting/manuscript_v4/make_si_entry_example_v4.py              # render d with PyMOL, then compose
    python plotting/manuscript_v4/make_si_entry_example_v4.py --compose_only
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import KDE_CLASSIFIER_DIR, PARTS_DIR, PYMOL_PYEXE, SI_DIR, cfg  # noqa: E402

# the classification code (figure_pocket_example.py and its imports); AHOJ_AF_KDE_CLASSIFIER_DIR overrides
CLASSIFIER_DIR = str(KDE_CLASSIFIER_DIR)
# PYMOL_PYEXE: a python with open-source PyMOL (pymol-open-source 3.1.0 was used); AHOJ_AF_PYMOL_PYEXE
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(str(SI_DIR), "FigureS9_entry_example")
PARTS = os.path.join(str(PARTS_DIR), "si_entry_example")
DPI = 600
MM = 1.0 / 25.4

FS_LETTER = 8.0                   # Nature's rule: bold, lowercase, 8 pt
PX_PER_MM = DPI * MM
# ---- panel d: the v3 panel's layout, measured on figures/natcomm_v3/Figure1_example.png (same canvas, same a-c).
# v3 shrank each 3000 x 2250 px snapshot by 0.344 (the ligand's surface: 826 px tall in the snapshot, 284 on the page),
# so a frame is 43.65 x 32.77 mm; positions are mm from the canvas's top-left corner, text centres mm from the frame's.
FRAME_MM = (43.65, 32.77)
FRAMES_MM = {"apo": (93.26, 178.94), "holo": (133.22, 178.94), "model": (112.69, 201.25)}
TITLES = {"apo": "APO", "holo": "HOLO", "model": "AlphaFold"}
TITLE_AT_MM = (19.9, 5.1)         # the same in all three frames in v3, to 0.4 mm
LABELS_MM = {
    "apo": {"M198": (30.16, 10.37), "W197": (25.10, 25.15)},
    "holo": {"W197": (32.05, 9.02), "M198": (37.57, 20.91)},
    "model": {"M198": (31.60, 11.94), "W197": (28.22, 26.59)},
}
LABEL_COLOUR = "#4C4C4C"          # v3's label grey
KEY_TEXT = "Grey: the ligand of the\nholo query (I46), shown\nin all three"
KEY_BOTTOM_MM = 229.0             # the key block's bottom, mm from the top; left-aligned with the letter d


def load_fpe():
    if not os.path.isfile(os.path.join(CLASSIFIER_DIR, "figure_pocket_example.py")):
        sys.exit(f"figure_pocket_example.py not found in {CLASSIFIER_DIR}; set AHOJ_AF_KDE_CLASSIFIER_DIR")
    sys.path.insert(0, CLASSIFIER_DIR)
    import figure_pocket_example as fpe
    return fpe


def classifier_commit():
    try:
        return subprocess.run(["git", "-C", CLASSIFIER_DIR, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "?"
    except OSError:
        return "?"


def stats_dir_of(fpe):
    """The entry's statistics folder: under cfg.EXAMPLE_ENTRIES_DIR when that anchor is set, else wherever the
    classification code's own default (fpe.EXAMPLE_ROOT) points."""
    if not str(cfg.EXAMPLE_ENTRIES_DIR).startswith("<UNSET:"):
        return fpe.resolve_stats_dir(example_root=str(cfg.EXAMPLE_ENTRIES_DIR))
    return fpe.resolve_stats_dir()


def render_d(w_px, h_px, parts):
    os.makedirs(parts, exist_ok=True)
    cmd = [PYMOL_PYEXE, os.path.join(HERE, "render_si_entry_pymol_v4.py"), "--out_dir", parts,
           "--w_px", str(w_px), "--h_px", str(h_px), "--dpi", str(DPI)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode:
        sys.exit(f"PyMOL render failed\n{r.stdout}\n{r.stderr}")
    for line in r.stdout.splitlines():
        if line.startswith("[pymol]"):
            print("  ", line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--compose_only", action="store_true", help="reuse the PyMOL parts already rendered")
    ap.add_argument("--out", default=OUT, help="output path without extension")
    ap.add_argument("--parts_dir", default=PARTS, help="where the PyMOL views and render.json are written and read")
    a = ap.parse_args()

    fpe = load_fpe()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from PIL import Image

    fpe._register_bundled_fonts()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": fpe.FONT_FAMILY,
        "font.size": fpe.FS_BASE, "axes.labelsize": fpe.FS_BASE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    stats_dir = stats_dir_of(fpe)
    print(f"[stats] {stats_dir}  (classification code {CLASSIFIER_DIR} @ {classifier_commit()})")
    results, df = fpe.load_entry(stats_dir)
    kde = fpe.compute_pocket_kde(df, fpe.kde_kwargs_from_config(fpe.CONFIG_PATH))
    print(f"[entry] n_apo {np.size(kde['apo']['volume'])}, n_holo {np.size(kde['holo']['volume'])}, "
          f"overall P(apo) {results['overall']['avg_P(apo)']:.3f} -> {results['overall']['overall_classification']}")

    fig = plt.figure(figsize=(fpe.CANVAS_W_MM * MM, fpe.CANVAS_H_MM * MM), dpi=DPI)
    gs = GridSpec(5, 2, height_ratios=fpe.GRID_HEIGHT_RATIOS, width_ratios=fpe.GRID_WIDTH_RATIOS,
                  hspace=fpe.GRID_HSPACE, wspace=fpe.GRID_WSPACE, **fpe.MARGIN)

    # ---- a-c: figure_pocket_example's build_figure, verbatim apart from the letters and the d placeholder
    groups = []
    for i, metric in enumerate(fpe.METRICS):
        axA = fig.add_subplot(gs[i, 0])
        fpe.draw_violin_panel(axA, metric, kde, is_top=(i == 0))
        groups.append((fpe.draw_bar_group(fig, gs[i, 1], results[metric], is_overall=False), None))
    c_cell = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[4, 0], height_ratios=[fpe.C_TOP_PAD, 1.0], hspace=0.0)
    axesC = fpe.draw_bar_group(fig, c_cell[1], results["overall"], is_overall=True)
    groups.append((axesC, "Overall pocket classification"))
    fig.canvas.draw()
    for axes, title in groups:
        fpe._titles_and_brackets(fig, axes, title)
    letter_x = {}
    for cell, letter in [(gs[0, 0], "a"), (gs[0, 1], "b"), (gs[4, 0], "c"), (gs[4, 1], "d")]:
        pos = cell.get_position(fig)
        letter_x[letter] = max(0.006, pos.x0 - fpe.LETTER_DX)
        fig.text(letter_x[letter], pos.y1 + fpe.LETTER_DY, letter,
                 fontsize=FS_LETTER, fontweight="bold", va="bottom", ha="left")

    # ---- d: the v3 panel, re-rendered at true size and placed
    W_mm, H_mm = fpe.CANVAS_W_MM, fpe.CANVAS_H_MM
    w_px, h_px = int(round(FRAME_MM[0] * PX_PER_MM)), int(round(FRAME_MM[1] * PX_PER_MM))
    print(f"[d] three frames of {w_px / PX_PER_MM:.2f} x {h_px / PX_PER_MM:.2f} mm ({w_px} x {h_px} px)")
    if not a.compose_only:
        render_d(w_px, h_px, a.parts_dir)
    with open(os.path.join(a.parts_dir, "render.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    if (rec["w_px"], rec["h_px"]) != (w_px, h_px):
        sys.exit(f"PyMOL parts are {rec['w_px']} x {rec['h_px']} px, the layout needs {w_px} x {h_px}; "
                 "re-run without --compose_only")
    print(f"[d] session {os.path.basename(rec['pse'])} (md5 {rec['pse_md5']}), scene {rec['scene']}, PyMOL {rec['pymol']}")

    def frac(x_mm, y_mm):                                    # mm from the top-left corner -> figure fraction
        return x_mm / W_mm, 1.0 - y_mm / H_mm

    for key, (fx, fy) in FRAMES_MM.items():
        im = np.asarray(Image.open(os.path.join(a.parts_dir, rec["files"][key])).convert("RGBA"))
        assert im.shape[:2] == (h_px, w_px), (im.shape, w_px, h_px)
        xo = int(round(fx * PX_PER_MM))
        yo = int(round((H_mm - fy) * PX_PER_MM)) - h_px     # figimage counts from the bottom edge
        fig.figimage(im, xo=xo, yo=yo, origin="upper", zorder=1)   # transparent background: frames may overlap
        fig.text(*frac(fx + TITLE_AT_MM[0], fy + TITLE_AT_MM[1]), TITLES[key], ha="center", va="center",
                 fontsize=fpe.FS_TITLE, fontweight="bold", zorder=3)
        for lab, (lx, ly) in LABELS_MM[key].items():
            fig.text(*frac(fx + lx, fy + ly), lab, ha="center", va="center", fontsize=fpe.FS_BASE,
                     color=LABEL_COLOUR, zorder=3)
    fig.text(letter_x["d"], frac(0, KEY_BOTTOM_MM)[1], KEY_TEXT, ha="left", va="bottom", multialignment="left",
             fontsize=fpe.FS_BASE, color=fpe.NTEXT_COLOR, zorder=3)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=DPI, facecolor="white")    # no bbox_inches: the canvas is the page size
    plt.close(fig)
    with Image.open(f"{a.out}.png") as im:
        print(f"wrote {a.out}.png  {im.width} x {im.height} px = {im.width / DPI / MM:.1f} x "
              f"{im.height / DPI / MM:.1f} mm, and .pdf")


if __name__ == "__main__":
    main()
