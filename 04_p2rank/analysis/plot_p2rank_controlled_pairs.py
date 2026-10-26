from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence_labeling import pretty_level

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


WORK_DIR = cfg.WORK_DIR
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "controlled_pairs_v1"

CLASS_ORDER = ["apo", "holo"]
CLASS_LABELS = {"apo": "AF-Apo", "holo": "AF-Holo"}
CLASS_COLORS = {"apo": "#2C7FB8", "holo": "#F28E2B"}
FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
DATASET_LABELS = {"all": "All ST_100 pockets", "alphafill_matched": "AlphaFill-matched only"}
TITLE_FS = 9.0
LAB_FS = 8.0
TICK_FS = 7.0
SMALL_FS = 6.2
FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_PREFERENCE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot controlled-pair P2Rank success analyses.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def load_tables(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tables_dir = analysis_dir / "tables"
    summary = pd.read_csv(tables_dir / "controlled_pair_success_summary.csv")
    means = pd.read_csv(tables_dir / "controlled_pair_mean_summary.csv")
    manifest = json.loads((tables_dir / "manifest.json").read_text(encoding="utf-8"))
    return summary, means, manifest


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pretty_bin(label: str) -> str:
    if label in FULL_LEVEL_ORDER:
        return pretty_level(label)
    return label


def display_bin(label: str, manifest: dict, *, axis: str) -> str:
    mapping = manifest.get(f"{axis}_display_labels", {}) or {}
    if label in mapping:
        shown = mapping[label]
        return pretty_level(shown) if shown in FULL_LEVEL_ORDER else shown
    return pretty_bin(label)


def plot_controlled_lines(summary: pd.DataFrame, manifest: dict, outpath: Path) -> None:
    focus_order = manifest["focus_order"]
    control_order = manifest["control_order"]
    success_label = "DCA4 success (%)"
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.1), constrained_layout=True, sharey=True)
    for row, dataset in enumerate(["all", "alphafill_matched"]):
        for col, pocket_class in enumerate(CLASS_ORDER):
            ax = axes[row, col]
            sub = summary[(summary["dataset"] == dataset) & (summary["pocket_class"] == pocket_class)].copy()
            x = np.arange(len(focus_order))
            palette = plt.cm.Greys(np.linspace(0.3, 0.9, max(len(control_order), 2)))
            for idx, control_bin in enumerate(control_order):
                grp = sub[sub["control_bin"] == str(control_bin)].set_index("focus_bin").reindex(focus_order)
                if grp["n"].fillna(0).sum() == 0:
                    continue
                ax.plot(
                    x,
                    grp["success_percent"],
                    marker="o",
                    lw=1.7,
                    ms=4.0,
                    color=palette[idx],
                    label=display_bin(str(control_bin), manifest, axis="control"),
                )
            ax.set_xticks(x)
            ax.set_xticklabels([display_bin(str(v), manifest, axis="focus") for v in focus_order], rotation=35, ha="right", fontsize=TICK_FS)
            ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.set_ylim(0, 103)
            ax.set_title(f"{DATASET_LABELS[dataset]} | {CLASS_LABELS[pocket_class]}", fontsize=TITLE_FS)
            if row == 1:
                ax.set_xlabel(manifest["focus_metric"], fontsize=LAB_FS)
            if col == 0:
                ax.set_ylabel(success_label, fontsize=LAB_FS)
    axes[0, 1].legend(frameon=False, fontsize=SMALL_FS, title=manifest["control_metric"], title_fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath)


def plot_controlled_heatmaps(summary: pd.DataFrame, manifest: dict, outpath: Path) -> None:
    focus_order = manifest["focus_order"]
    control_order = manifest["control_order"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), constrained_layout=True)
    ims = []
    for row, dataset in enumerate(["all", "alphafill_matched"]):
        for col, pocket_class in enumerate(CLASS_ORDER):
            ax = axes[row, col]
            sub = summary[(summary["dataset"] == dataset) & (summary["pocket_class"] == pocket_class)].copy()
            mat = (
                sub.pivot_table(index="control_bin", columns="focus_bin", values="success_rate", aggfunc="mean")
                .reindex(index=control_order, columns=focus_order)
            )
            cnt = (
                sub.pivot_table(index="control_bin", columns="focus_bin", values="n", aggfunc="sum")
                .reindex(index=control_order, columns=focus_order)
            )
            im = ax.imshow(mat.to_numpy(dtype=float), vmin=0.35, vmax=1.0, cmap="Greys", aspect="auto")
            ims.append(im)
            for i, cbin in enumerate(control_order):
                for j, fbin in enumerate(focus_order):
                    v = mat.loc[cbin, fbin]
                    n = cnt.loc[cbin, fbin]
                    if pd.notna(v) and pd.notna(n) and n > 0:
                        ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=SMALL_FS, color=("white" if v >= 0.68 else "0.15"))
            ax.set_xticks(np.arange(len(focus_order)))
            ax.set_xticklabels([display_bin(str(v), manifest, axis="focus") for v in focus_order], rotation=35, ha="right", fontsize=TICK_FS)
            ax.set_yticks(np.arange(len(control_order)))
            ax.set_yticklabels([display_bin(str(v), manifest, axis="control") for v in control_order], fontsize=TICK_FS)
            ax.set_xlabel(manifest["focus_metric"], fontsize=LAB_FS)
            ax.set_ylabel(manifest["control_metric"], fontsize=LAB_FS)
            ax.set_title(f"{DATASET_LABELS[dataset]} | {CLASS_LABELS[pocket_class]}", fontsize=TITLE_FS)
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("Any-rank DCA4 success", fontsize=LAB_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.analysis_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, means, manifest = load_tables(args.analysis_dir)
    plot_controlled_lines(summary, manifest, output_dir / "controlled_pair_success_lines.png")
    plot_controlled_heatmaps(summary, manifest, output_dir / "controlled_pair_success_heatmaps.png")
    print(f"[SAVE] {output_dir}")


if __name__ == "__main__":
    main()
