# -*- coding: utf-8 -*-
"""
Compose SI Figure S7 -- AlphaFill fit under the permissive `any` match scope, 2 x 2.

    a  average clash count     b  average global RMSD
    c  average clash score     d  average local RMSD

Each panel is a vertical split half-violin (AF-Apo | AF-Holo) across the five verbal separation
bins, ST_100 UBSs with at least one AlphaFill transplant from ANY UniProt at alignment identity
>= 0.30, drawn by 03_alphafill/ubs/plot_clash_ridgelines.py (`--make_violins`, default box summary).
The four panels are scaled to one common width and pasted onto a white sheet with an 18 px pad;
the lowercase letters are drawn here. Extracted from the authors' figure stager (`compose_2x2` /
`build_s8`, where the figure was keyed FigureS8_alphafill_anyscope).

Inputs: <cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR>/ridgeline_plots/st_100/any/ge_0p30/
        st_100_any_ge_0p30_<metric>_halfviolin_vertical_verbal_sep.png
        (or, with --render, the same files re-drawn under <parts_dir>/ridgeline_plots/).
Output: <cfg.MANUSCRIPT_FIG_DIR>/SI/FigureS7_alphafill_anyscope.png

Usage (from the repository root):
    python plotting/manuscript_v4/make_si_alphafill_anyscope_v4.py            # compose the existing panels
    python plotting/manuscript_v4/make_si_alphafill_anyscope_v4.py --render   # re-draw the four panels first
"""
import argparse
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import PARTS_DIR, PLOT_PYEXE, PUBLIC_REPO, SI_DIR, cfg  # noqa: E402

THRESHOLD, SCOPE, IDENTITY = "ST_100", "any", "ge_0p30"
METRICS = ("avg_clash_count", "avg_global_rmsd", "avg_clash", "avg_local_rmsd")   # panel order a-d
RIDGELINE_DIR = os.path.join(str(cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR), "ridgeline_plots")
PARTS = os.path.join(str(PARTS_DIR), "si_alphafill_anyscope")
OUT = os.path.join(str(SI_DIR), "FigureS7_alphafill_anyscope.png")


def font(sz):
    for f in ("arialbd.ttf", "Arial_Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def compose_2x2(paths, labels, out, pad=18, lab_sz=64):
    ims = [Image.open(p).convert("RGB") for p in paths]
    tw = min(im.width for im in ims)
    ims = [im.resize((tw, int(im.height * tw / im.width))) for im in ims]
    rh, rh2 = max(ims[0].height, ims[1].height), max(ims[2].height, ims[3].height)
    sheet = Image.new("RGB", (tw * 2 + pad * 3, rh + rh2 + pad * 3), "white")
    d = ImageDraw.Draw(sheet)
    F = font(lab_sz)
    for im, (x, y), lab in zip(ims, [(pad, pad), (pad * 2 + tw, pad),
                                     (pad, pad * 2 + rh), (pad * 2 + tw, pad * 2 + rh)], labels):
        sheet.paste(im, (x, y))
        d.text((x + 6, y + 2), lab, fill="black", font=F)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    return out


def panel_paths(ridgeline_dir):
    an = os.path.join(ridgeline_dir, THRESHOLD.lower(), SCOPE, IDENTITY)
    return [os.path.join(an, f"{THRESHOLD.lower()}_{SCOPE}_{IDENTITY}_{m}_halfviolin_vertical_verbal_sep.png")
            for m in METRICS]


def render(ridgeline_dir):
    """Re-draw the four any-scope half-violin panels (the generator also writes its ridgelines)."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    script = os.path.join(str(PUBLIC_REPO), "03_alphafill", "ubs", "plot_clash_ridgelines.py")
    for m in METRICS:
        cmd = [PLOT_PYEXE, script, "--analysis_dir", str(cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR),
               "--output_dir", ridgeline_dir, "--thresholds", THRESHOLD, "--scopes", SCOPE,
               "--identity_tags", IDENTITY, "--metric", m, "--make_violins"]
        print("  ", os.path.basename(script), m)
        r = subprocess.run(cmd, cwd=str(PUBLIC_REPO), capture_output=True, text=True, env=env)
        if r.returncode:
            sys.exit(f"FAILED {script} --metric {m}\n{r.stdout}\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--render", action="store_true",
                    help="re-draw the four panels into <parts_dir>/ridgeline_plots first and compose those")
    ap.add_argument("--parts_dir", default=PARTS)
    ap.add_argument("--panels_dir", default=None,
                    help="ridgeline_plots folder to read the panels from (default: the analysis folder's, "
                         "or <parts_dir>/ridgeline_plots with --render)")
    ap.add_argument("--out_dir", default=os.path.dirname(OUT))
    a = ap.parse_args()

    ridgeline_dir = a.panels_dir or (os.path.join(a.parts_dir, "ridgeline_plots") if a.render else RIDGELINE_DIR)
    if a.render:
        print("rendering panels ->", ridgeline_dir)
        render(ridgeline_dir)
    src = panel_paths(ridgeline_dir)
    missing = [p for p in src if not os.path.isfile(p)]
    if missing:
        sys.exit("missing panels (run with --render):\n  " + "\n  ".join(missing))
    out = compose_2x2(src, ["a", "b", "c", "d"], os.path.join(a.out_dir, os.path.basename(OUT)))
    print("wrote", out)


if __name__ == "__main__":
    main()
