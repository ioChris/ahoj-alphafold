# -*- coding: utf-8 -*-
"""
Supplementary Figure S1 — Apo-vs-Holo variation of standard structural descriptors across AHoJ-DB.

Framing: we characterise each AlphaFold-model pocket against the apo/holo experimental spectrum using
a set of STANDARD, well-known structural descriptors — grouped into LOCAL (pocket) and GLOBAL
(chain / whole-structure) measures. Most show a clear apo/holo difference across the database; radius
of gyration is a valid geometric descriptor that happens to shift little (pockets keep a similar
spatial spread even as volume/SASA change). The apo/holo classifier uses four of the local descriptors
(marked *).

Quantity per metric (the genuine, non-cancelling, symmetric apo-vs-holo difference — NOT the old
query-anchored |candidate-query| delta):
    pct = 100 * mean_entries(|apo_mean - holo_mean|) / mean_entries((apo_mean + holo_mean)/2)
computed from the per-entry query summary of
`batch_process_database.py --processing_function do_collect_apoholo_metric_stats`.

Colour encodes the Local/Global category (deliberately NOT the manuscript's apo=blue / holo=orange
hues). Within each category, bars are sorted by magnitude. Outputs PNG (600 dpi) + SVG + PDF.

@author: Christos P. Feidakis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# --------------------------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------------------------
# Full-DB run (all 561 batches of v2c/output_fix, classifier/full_run_3 filters, 242,526 paired
# entries; produced 2026-07-14), under the AHoJ-DB root (common/config.py). Override via argv[1].
INPUT_CSV = str(cfg.APOHOLO_METRIC_STATS_CSV)
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_BASENAME = "FigureS1_apoholo_pctdiff"

SASA_VARIANT = "sasa"        # "sasa" (total, the classifier metric) or "sasa_avg" (per-residue)
SORT_WITHIN_GROUP = True     # sort bars by magnitude within each Local/Global block

# (metric key, display label, group in {"local","global"}, is_classifier_metric)
#   group: local = pocket-level descriptor; global = chain / whole-structure descriptor
#   classifier: True = one of the four descriptors the apo/holo POCKET classifier uses (gets a *)
METRICS = [
    ("pocket_rms",  "Pocket RMSD",         "local",  True),
    ("volume",      "Volume",              "local",  True),
    (SASA_VARIANT,  "SASA",                "local",  True),
    ("RoG",         "Radius of gyration",  "local",  True),
    ("pocket_dist", "Pocket center motion","global", False),
    ("rmsd",        "Chain RMSD",          "global", False),
    ("tm_score",    "TM-score",            "global", False),
]
# NOTE: B-factor and Resolution intentionally excluded — they are experimental (flexibility /
# data-quality) rather than geometric, and B-factor is confounded with the apo/holo resolution gap.
CLASSIFIER_MARK = " *"       # suffix appended to classifier-metric labels

# --------------------------------------------------------------------------
# Style (manuscript / SI figures)
# --------------------------------------------------------------------------
FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT_PREFERENCE,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})
# Local/Global categorical colours — colourblind-aware, deliberately away from apo/holo blue/orange.
COL_LOCAL   = "#2F9E8F"      # teal-green  (pocket)
COL_GLOBAL  = "#9B6FB0"      # muted purple (chain / structure)
EDGE_COLOR  = "#2B2B2B"
GROUP_GAP   = 1.7            # blank bar-slots between the Local and Global blocks
LABEL_FS, TICK_FS, VAL_FS, TITLE_FS, NOTE_FS, GRP_FS = 9.5, 8.5, 8.5, 10.5, 7.6, 8.8


def load_values(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    n_entries = len(df)
    rows = []
    for key, label, group, is_clf in METRICS:
        dcol, acol, hcol = f"{key}_diff_means", f"{key}_apo_mean", f"{key}_holo_mean"
        for c in (dcol, acol, hcol):
            if c not in df.columns:
                raise SystemExit(f"Missing column '{c}' for metric '{key}' in {csv_path}")
        diff = pd.to_numeric(df[dcol], errors="coerce")
        sym = (pd.to_numeric(df[acol], errors="coerce") + pd.to_numeric(df[hcol], errors="coerce")) / 2.0
        pct = 100.0 * np.nanmean(diff) / np.nanmean(sym) if np.nanmean(sym) else np.nan
        rows.append({"key": key, "label": label, "group": group,
                     "is_clf": is_clf, "value": pct})
    return pd.DataFrame(rows), n_entries


def _ordered_with_positions(data):
    """Local block first (top), then a gap, then Global block; sorted within block.
    Returns the rows top->bottom plus their y-positions (higher y = higher on the chart)."""
    blocks = []
    for grp in ("local", "global"):
        sub = data[data["group"] == grp].copy()
        if SORT_WITHIN_GROUP:
            sub = sub.sort_values("value", ascending=False)
        blocks.append(sub)
    top_to_bottom = pd.concat(blocks, ignore_index=True)
    n_local = len(blocks[0])
    # y from top: local rows 0..n_local-1, gap, then global rows
    slot = 0.0
    ys = []
    for i in range(len(top_to_bottom)):
        if i == n_local:
            slot += GROUP_GAP
        ys.append(slot)
        slot += 1.0
    max_slot = slot - 1.0
    ys = [max_slot - v for v in ys]          # flip so first row sits at the top
    top_to_bottom["y"] = ys
    return top_to_bottom, n_local


def make_figure(data, n_entries, out_dir):
    rows, n_local = _ordered_with_positions(data)

    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    colors = [COL_LOCAL if g == "local" else COL_GLOBAL for g in rows["group"]]
    ax.barh(rows["y"], rows["value"], height=0.66, color=colors,
            edgecolor=EDGE_COLOR, linewidth=0.7, zorder=3)

    xmax = float(rows["value"].max())
    for yi, v in zip(rows["y"], rows["value"]):
        ax.text(v + xmax * 0.012, yi, f"{v:.1f}%", va="center", ha="left",
                fontsize=VAL_FS, color="#1A1A1A", zorder=4)

    ylabels = [lab + (CLASSIFIER_MARK if clf else "") for lab, clf in zip(rows["label"], rows["is_clf"])]
    ax.set_yticks(rows["y"])
    ax.set_yticklabels(ylabels, fontsize=LABEL_FS)
    ax.set_xlim(0, xmax * 1.14)
    ax.set_ylim(rows["y"].min() - 0.7, rows["y"].max() + 1.15)
    ax.set_xlabel("Apo–Holo difference  (% of the apo/holo mean)", fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    for sname in ("top", "right", "left"):
        ax.spines[sname].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#DADADA", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # faint divider in the gap between the two blocks
    y_div = (rows["y"].iloc[n_local - 1] + rows["y"].iloc[n_local]) / 2.0
    ax.axhline(y_div, color="#CFCFCF", linewidth=0.8, zorder=1)

    # group labels at the left, hugging each block's first bar (kept clear of the divider line)
    HDR_OFF = 0.55
    ax.text(0, rows["y"].iloc[0] + HDR_OFF, "Local  (pocket)", transform=ax.transData,
            ha="left", va="bottom", fontsize=GRP_FS, fontweight="bold", color=COL_LOCAL)
    ax.text(0, rows["y"].iloc[n_local] + HDR_OFF, "Global  (chain / structure)", transform=ax.transData,
            ha="left", va="bottom", fontsize=GRP_FS, fontweight="bold", color=COL_GLOBAL)

    ax.set_title("Apo–Holo variation of standard structural descriptors (AHoJ-DB)",
                 fontsize=TITLE_FS, pad=10, loc="left")

    handles = [Patch(facecolor=COL_LOCAL,  edgecolor=EDGE_COLOR, linewidth=0.7, label="Local metric (pocket)"),
               Patch(facecolor=COL_GLOBAL, edgecolor=EDGE_COLOR, linewidth=0.7, label="Global metric (chain / structure)")]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=TICK_FS, handlelength=1.2, borderpad=0.3)

    fig.text(0.012, 0.008, f"* metric used by the apo/holo pocket classifier      N = {n_entries:,} paired entries",
             fontsize=NOTE_FS, color="#555555", ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, OUT_BASENAME)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base + "." + ext, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved:", base + ".{png,pdf,svg}")


def main(csv_path=None, out_dir=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    data, n_entries = load_values(csv_path or INPUT_CSV)
    print(f"\nN paired entries: {n_entries:,}")
    print(data[["label", "group", "is_clf", "value"]].to_string(index=False))
    make_figure(data, n_entries, out_dir or OUTPUT_DIR)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    main(csv, out)
