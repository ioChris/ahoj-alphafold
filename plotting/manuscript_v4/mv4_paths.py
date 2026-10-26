# -*- coding: utf-8 -*-
"""
Roots shared by the manuscript v4 figure composers in this folder.

Every data path comes from the repository's path registry, `common/config.py` (anchors in the
git-ignored `common/paths.ini`, or the matching environment variables); nothing here is
machine-specific. The few extra knobs are environment variables:

    AHOJ_AF_PLOT_PYEXE          python that runs the panel generators (default: the interpreter
                                running the composer). The generators need matplotlib + scipy.
    AHOJ_AF_PYMOL_PYEXE         python with open-source PyMOL, for render_si_entry_pymol_v4.py
                                (default: the interpreter running the composer).
    AHOJ_AF_KDE_CLASSIFIER_DIR  folder holding figure_pocket_example.py and the rest of the
                                classification code (default: 00_classification/kde_classifier/).
    AHOJ_AF_MODEL_SET=afdbv4    read the AFDB v4 re-run of the detection and docking arms and write
                                the figures to the `_afdbv4` figure folder (see common/config.py).

Output folder: `cfg.MANUSCRIPT_FIG_DIR` (anchor `manuscript_figures_dir`, else
`<work_dir>/unified_output/run5/plots/manuscript_v4`), with `parts/` and `SI/` underneath. Every
composer also takes `--out_dir`.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_REPO = HERE.parents[1]                 # plotting/manuscript_v4 -> repository root
if str(PUBLIC_REPO) not in sys.path:
    sys.path.insert(0, str(PUBLIC_REPO))

from common import config as cfg  # noqa: E402
from common import figspec  # noqa: E402,F401  (re-exported for the composers)

PLOT_PYEXE = os.environ.get("AHOJ_AF_PLOT_PYEXE") or sys.executable
PYMOL_PYEXE = os.environ.get("AHOJ_AF_PYMOL_PYEXE") or sys.executable
KDE_CLASSIFIER_DIR = Path(os.environ.get("AHOJ_AF_KDE_CLASSIFIER_DIR")
                          or PUBLIC_REPO / "00_classification" / "kde_classifier")

MODEL_SET = cfg.MODEL_SET
MODEL_SFX = cfg.AFDBV4_SUFFIX if MODEL_SET == "afdbv4" else ""

FIG_DIR = Path(cfg.MANUSCRIPT_FIG_DIR)         # composed figures (model-set aware)
PARTS_DIR = FIG_DIR / "parts"                  # rendered panels, placed by the composers
SI_DIR = FIG_DIR / "SI"                        # composed SI figures
