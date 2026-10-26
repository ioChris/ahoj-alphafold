# -*- coding: utf-8 -*-
"""
PyMOL render of SI Figure S9d (the authors' build called it FigureS10_entry_example): W197 and M198 of the AHoJ-DB entry
2yiw A I46 1354 (MAPK14 = p38alpha, UniProt Q16539) in three structures, with the query's ligand I46 in every view.

    apo    5XYY chain A, the representative apo structure      session object cbse_obj
    holo   2YIW chain A, the holo query                        session object qbse_obj
    model  AlphaFold DB version 4 model AF-Q16539-F1           session object mbse_obj
    (all)  I46 of 2YIW A 1354, sticks and a gray70 surface     session object qlig_obj

Rendered from the PyMOL session the v3 panel was made from (extended_ahoj_pocket.pse, 2026-03; its snapshots
cbse6 / qbse6 / mbse6.png are the images that went into the v3 panel), with the
session's own camera (stored scene "003") and its own render settings, so the look and the orientation are the v3
panel's. The session's coordinates are AHoJ's aligned chains of the 2024-05-27 run (the webserver copy of the job), to
0.0005 A: the query's frame, by AHoJ's own US-align superposition.

What changes against v3: each view is ray-traced at the pixel size of its frame on the page (true size at 600 dpi)
instead of at 3000 x 2250 px and shrunk, the background is transparent so the frames may overlap as they did, and the
labels and titles are drawn by the composer in the figure's font.

Run with a python that has open-source PyMOL (3.1.0 was used). An unlicensed Schrodinger build stamps
"No License File - For Evaluation Only" into some ray-traced images. The session path comes from common/config.py
(anchor `entry_example_pse`, env AHOJ_AF_ENTRY_EXAMPLE_PSE) or --pse.

    <pymol python> plotting/manuscript_v4/render_si_entry_pymol_v4.py --out_dir DIR --w_px 1031 --h_px 774

make_si_entry_example_v4.py calls it with the frame size of its layout.
"""
import argparse
import hashlib
import json
import os
import sys

from pathlib import Path

from pymol import cmd

# common/config.py (stdlib only, so it imports in the PyMOL env too)
for _root in Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
from common import config as cfg  # noqa: E402

PSE = str(cfg.ENTRY_EXAMPLE_PSE)  # .../2yiw-A-I46-1354/extended_ahoj_pocket.pse
PSE_MD5 = "c76d1a6dc6819642bc7c830257ddd067"  # the session as of 2026-03-06; a different file stops the render (see --any_session)
SCENE = "003"                     # the camera of the v3 snapshots (scene 004 is the same view, 1 A nearer: 1.7% larger)
LIGAND = "qlig_obj"
VIEWS = {"apo": "cbse_obj", "holo": "qbse_obj", "model": "mbse_obj"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--w_px", type=int, required=True)
    ap.add_argument("--h_px", type=int, required=True)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--pse", default=PSE)
    ap.add_argument("--any_session", action="store_true", help="render even if the session is not the 2026-03-06 one")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not os.path.isfile(a.pse):
        sys.exit(f"PyMOL session not found: {a.pse}. Set entry_example_pse in common/paths.ini "
                 "(or AHOJ_AF_ENTRY_EXAMPLE_PSE), or pass --pse")
    digest = md5(a.pse)
    if digest != PSE_MD5 and not a.any_session:
        sys.exit(f"{a.pse}: md5 {digest}, expected {PSE_MD5}. The session changed since the figure was made; "
                 "check it and re-run with --any_session")
    cmd.load(a.pse)
    cmd.scene(SCENE, "recall", animate=0)
    view = cmd.get_view()
    cmd.reinitialize()                         # the scene's camera only; representations and colours as saved
    cmd.load(a.pse)
    cmd.set("ray_opaque_background", 0)        # transparent: the frames overlap on the page, as in v3
    record = {"pse": a.pse, "pse_md5": digest, "scene": SCENE, "view": list(view), "w_px": a.w_px, "h_px": a.h_px,
              "dpi": a.dpi, "pymol": cmd.get_version()[0], "files": {}}
    for name, obj in VIEWS.items():
        cmd.disable("all")
        cmd.enable(LIGAND)
        cmd.enable(obj)
        cmd.set_view(view)
        out = os.path.join(a.out_dir, f"pymol_{name}.png")
        cmd.png(out, width=a.w_px, height=a.h_px, dpi=a.dpi, ray=1)
        record["files"][name] = os.path.basename(out)
        print(f"[pymol] {name:5s} {obj} -> {out} ({a.w_px} x {a.h_px} px)")
    with open(os.path.join(a.out_dir, "render.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main()
