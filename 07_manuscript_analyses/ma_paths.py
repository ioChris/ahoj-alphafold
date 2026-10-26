# -*- coding: utf-8 -*-
"""Roots for the manuscript's supplementary analyses (07_manuscript_analyses/).

Every script in this folder starts with ``import ma_paths as P`` (the folder is on ``sys.path`` when a
script runs from here) and takes its roots from it:

    P.REPO            the repository root
    P.WD              the project work_dir (cfg.WORK_DIR)
    P.OUT             where these analyses write their tables and logs, and read each other's
                      intermediates (``manuscript_analyses_dir`` / AHOJ_AF_MANUSCRIPT_ANALYSES_DIR;
                      default <work_dir>/manuscript_analyses)
    P.MODEL_SFX       "_afdbv4" when AHOJ_AF_MODEL_SET=afdbv4, else ""
    P.cfg             common/config.py itself, for its named paths (cfg.ANALYSIS_UNIFIED_LEVELS_CSV, ...)
    P.anchor(key, env)  any further machine-specific root: env var, else ``key`` in common/paths.ini,
                      else a ``<UNSET:ENV>`` placeholder that fails loudly where it is first used

Importing it also puts the classification code (00_classification/kde_classifier/) on ``sys.path``, so the
analyses that reuse the classifier import its modules by name, as they did in the authors' working copy.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from common import config as cfg  # noqa: E402

CLASSIFIER_DIR = REPO / "00_classification" / "kde_classifier"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.append(str(CLASSIFIER_DIR))

HERE = Path(__file__).resolve().parent


def anchor(key: str, env_var: str) -> Path:
    """A machine-specific root: ``env_var``, else ``key`` in common/paths.ini [paths], else a placeholder."""
    return cfg._anchor(key, env_var)


WD = cfg.WORK_DIR
MODEL_SFX = cfg.AFDBV4_SUFFIX if cfg.MODEL_SET == "afdbv4" else ""
_out = anchor("manuscript_analyses_dir", "AHOJ_AF_MANUSCRIPT_ANALYSES_DIR")
OUT = (WD / "manuscript_analyses") if str(_out).startswith("<UNSET:") else _out


def out(name: str) -> str:
    """Path of an output / intermediate file in P.OUT (created on first use)."""
    os.makedirs(OUT, exist_ok=True)
    return str(OUT / name)
