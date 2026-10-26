# -*- coding: utf-8 -*-
"""
Render and compose the v4 SI figure "ligand placement across apo-holo separation" -- the merge of
the round-2 S10 (docking affinity half-violins) into S11 (redock RMSD, redock success, AlphaFill
local RMSD), decided in the 2026-09-15 SI curation. It is SI Figure S8 of the submitted manuscript.

    a  docking top-pose affinity, split half-violins      05_docking/runner/plot_docking_ridgelines.py
    b  redock RMSD (docked vs crystal ligand, AF frame)   05_docking/redock/plot_remodeled_fig5.py
    c  redock success (<= 2 A)                            "
    d  AlphaFill local RMSD, pocket level (the parallel)  "

Every panel is drawn at 88 x 62 mm by its generator, in the Figure 4 grammar (lowercase tiers,
"overall" LAST behind a dotted divider, AF-Apo / AF-Holo fills, stacked n row), and is PLACED here,
never resized -- the same rule as make_alphafill_panels_v4.py. 2 x 2 with 2 mm gaps = 178 x 126 mm.

Panel a fixes two things the July S10 got wrong (both flagged by the author in round 2): the y axis
floor was a fixed -13 kcal/mol with 0.7% of pockets below -12, leaving two kcal/mol of empty axis
under every violin, and the class key sat on the tallest violin. It is now --floor -12 with 15%
headroom. The July fig5real_* files regenerate unchanged (all options default to the July values).

Usage (from the repository root; the manuscript figure is the AFDB v4 re-run):
    AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_si_placement_v4.py   # render the four parts, then compose
    AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_si_placement_v4.py --compose_only
"""
import argparse
import os
import subprocess
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import PARTS_DIR, PLOT_PYEXE, PUBLIC_REPO, SI_DIR, cfg  # noqa: E402

# named by the submitted manuscript's number (the authors' build called it FigureS9_ligand_placement)
PARTS = os.path.join(str(PARTS_DIR), "si_placement")
OUT = os.path.join(str(SI_DIR), "FigureS8_ligand_placement.png")
# <work_dir>/docking/run3[_afdbv4]/analysis/docking_run3_merged.csv (follows AHOJ_AF_MODEL_SET)
MERGED_CSV = os.path.join(str(cfg.DOCKING_RUN3_DIR), "analysis", "docking_run3_merged.csv")
DPI = 600
GAP_MM = 2.0

PANELS = [  # (file, letter) in reading order; letters are drawn by the generators
    ("si_docking_affinity_separation.png", "a"),
    ("panel_b_redock_rmsd.png", "b"),
    ("panel_c_redock_success.png", "c"),
    ("panel_d_afill_local_rmsd.png", "d"),
]


def render(repo, parts):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    jobs = [
        [PLOT_PYEXE, os.path.join(repo, "05_docking", "runner", "plot_docking_ridgelines.py"),
         "--merged_csv", MERGED_CSV, "--out_dir", parts, "--which", "separation",
         "--out_basename", "si_docking_affinity", "--all_position", "last",
         "--floor", "-12", "--headroom", "1.15", "--canvas_mm", "88,62",
         "--letter", "a", "--no_title", "--stacked_counts"],
        [PLOT_PYEXE, os.path.join(repo, "05_docking", "redock", "plot_remodeled_fig5.py"),
         # --out_dir: its legacy 1 x 3 fig5_remodeled_placement.* goes beside the parts, not into
         # the docking run folder (the authors' build left it at <DOCKING_RUN3_DIR>/reference_poses)
         "--v4_panels_dir", parts, "--v4_letters", "b,c,d", "--out_dir", parts],
    ]
    for cmd in jobs:
        print("  ", os.path.basename(cmd[1]))
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
        if r.returncode:
            sys.exit(f"FAILED {cmd[1]}\n{r.stdout}\n{r.stderr}")
        for line in r.stdout.splitlines():
            if line.startswith(("[data]", "[v4]", "[saved]")):
                print("     ", line)


def compose(parts, out):
    ims = [Image.open(os.path.join(parts, f)).convert("RGB") for f, _ in PANELS]
    w, h = ims[0].size
    bad = [f for (f, _), im in zip(PANELS, ims) if im.size != (w, h)]
    if bad:
        sys.exit(f"parts differ in size (all must be 88 x 62 mm at {DPI} dpi): {bad}")
    gap = int(round(GAP_MM / 25.4 * DPI))
    sheet = Image.new("RGB", (2 * w + gap, 2 * h + gap), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % 2) * (w + gap), (k // 2) * (h + gap)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out, dpi=(DPI, DPI))
    print(f"wrote {out}  {sheet.width / DPI * 25.4:.1f} x {sheet.height / DPI * 25.4:.1f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--compose_only", action="store_true")
    ap.add_argument("--parts_dir", default=PARTS, help="where the four parts are rendered and read")
    ap.add_argument("--out_dir", default=os.path.dirname(OUT), help="folder of the composed figure")
    a = ap.parse_args()
    if not a.compose_only:
        os.makedirs(a.parts_dir, exist_ok=True)
        print("rendering parts:")
        render(str(PUBLIC_REPO), a.parts_dir)
    compose(a.parts_dir, os.path.join(a.out_dir, os.path.basename(OUT)))


if __name__ == "__main__":
    main()
