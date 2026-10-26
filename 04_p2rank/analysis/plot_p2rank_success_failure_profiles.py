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
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "success_failure_profiles_v1"

CLASS_ORDER = ["apo", "holo"]
CLASS_LABELS = {"apo": "AF-Apo", "holo": "AF-Holo"}
OUTCOME_ORDER = ["success", "failure"]
OUTCOME_LABELS = {"success": "Predicted", "failure": "Failed"}
OUTCOME_COLORS = {"success": "#4d4d4d", "failure": "#d95f02"}
FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
VERBAL_LABELS = {
    "separation_level_db": "Pocket separation",
    "density_level_db": "Pocket density",
    "prob_level_db": "Winning probability",
}
QUANTILE_LABELS = {
    "pocket_overlap_pair": "Pocket overlap pair",
    "log2_ratio_holo_to_apo_avg_plus1": "log2 holo/apo avg evidence (+1)",
    "pocket_separation_pair": "Pocket separation",
}
SIZE_METRICS = {
    "separation_level_db": "Pocket separation",
    "density_level_db": "Pocket density",
    "prob_level_db": "Winning probability",
    "pocket_overlap_pair": "Pocket overlap pair",
    "log2_ratio_holo_to_apo_avg_plus1": "log2 holo/apo avg evidence (+1)",
}

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
    p = argparse.ArgumentParser(description="Plot P2Rank success-vs-failure pocket profiles.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def load_tables(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tables_dir = analysis_dir / "tables"
    verbal = pd.read_csv(tables_dir / "p2rank_success_failure_verbal_profiles.csv")
    quantile = pd.read_csv(tables_dir / "p2rank_success_failure_quantile_profiles.csv")
    size_strat = pd.read_csv(tables_dir / "p2rank_success_failure_size_stratified.csv")
    manifest = json.loads((tables_dir / "manifest.json").read_text(encoding="utf-8"))
    return verbal, quantile, size_strat, manifest


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _pretty_bin(metric: str, label: str) -> str:
    if label in FULL_LEVEL_ORDER:
        return pretty_level(label)
    return label


def plot_verbal_profiles(verbal: pd.DataFrame, outpath: Path) -> None:
    metrics = list(VERBAL_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), constrained_layout=True, sharey=True)
    for row, pocket_class in enumerate(CLASS_ORDER):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sub = verbal[(verbal["pocket_class"] == pocket_class) & (verbal["metric"] == metric)].copy()
            x = np.arange(len(FULL_LEVEL_ORDER))
            for outcome in OUTCOME_ORDER:
                grp = sub[sub["outcome"] == outcome].set_index("bin").reindex(FULL_LEVEL_ORDER)
                ax.plot(
                    x,
                    grp["share_within_group"] * 100.0,
                    marker="o",
                    lw=1.8,
                    ms=4.0,
                    color=OUTCOME_COLORS[outcome],
                    label=OUTCOME_LABELS[outcome],
                )
            ax.set_xticks(x)
            ax.set_xticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], rotation=35, ha="right", fontsize=TICK_FS)
            ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.set_title(f"{CLASS_LABELS[pocket_class]} | {VERBAL_LABELS[metric]}", fontsize=TITLE_FS)
            if row == 1:
                ax.set_xlabel("Metric bin", fontsize=LAB_FS)
            if col == 0:
                ax.set_ylabel("Share within outcome (%)", fontsize=LAB_FS)
    axes[0, 2].legend(frameon=False, fontsize=SMALL_FS, loc="upper right")
    savefig(fig, outpath)


def plot_quantile_profiles(quantile: pd.DataFrame, outpath: Path) -> None:
    metrics = list(QUANTILE_LABELS.keys())
    fig, axes = plt.subplots(2, len(metrics), figsize=(8.8, 5.8), constrained_layout=True, sharey=True)
    for row, pocket_class in enumerate(CLASS_ORDER):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sub = quantile[(quantile["pocket_class"] == pocket_class) & (quantile["metric"] == metric)].copy()
            order = [f"Q{i}" for i in range(1, 6) if f"Q{i}" in set(sub["bin"].dropna())]
            x = np.arange(len(order))
            for outcome in OUTCOME_ORDER:
                grp = sub[sub["outcome"] == outcome].set_index("bin").reindex(order)
                ax.plot(
                    x,
                    grp["share_within_group"] * 100.0,
                    marker="o",
                    lw=1.8,
                    ms=4.0,
                    color=OUTCOME_COLORS[outcome],
                    label=OUTCOME_LABELS[outcome],
                )
            ax.set_xticks(x)
            ax.set_xticklabels(order, fontsize=TICK_FS)
            ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.set_title(f"{CLASS_LABELS[pocket_class]} | {QUANTILE_LABELS[metric]}", fontsize=TITLE_FS)
            if row == 1:
                ax.set_xlabel("Metric quantile", fontsize=LAB_FS)
            if col == 0:
                ax.set_ylabel("Share within outcome (%)", fontsize=LAB_FS)
    axes[0, -1].legend(frameon=False, fontsize=SMALL_FS, loc="upper right")
    savefig(fig, outpath)


def plot_size_stratified_heatmaps(size_strat: pd.DataFrame, manifest: dict, outpath: Path) -> None:
    metrics = list(SIZE_METRICS.keys())
    size_order = manifest["size_order"]
    fig, axes = plt.subplots(2, len(metrics), figsize=(12.8, 5.8), constrained_layout=True)
    ims = []
    for row, pocket_class in enumerate(CLASS_ORDER):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sub = size_strat[(size_strat["pocket_class"] == pocket_class) & (size_strat["metric"] == metric)].copy()
            if metric in VERBAL_LABELS:
                metric_order = FULL_LEVEL_ORDER
                xticklabels = [pretty_level(v) for v in metric_order]
            else:
                metric_order = [f"Q{i}" for i in range(1, 6) if f"Q{i}" in set(sub["metric_bin"].dropna())]
                xticklabels = metric_order
            mat = (
                sub.pivot_table(index="size_bin", columns="metric_bin", values="success_rate", aggfunc="mean")
                .reindex(index=size_order, columns=metric_order)
            )
            cnt = (
                sub.pivot_table(index="size_bin", columns="metric_bin", values="n", aggfunc="sum")
                .reindex(index=size_order, columns=metric_order)
            )
            im = ax.imshow(mat.to_numpy(dtype=float), vmin=0.2, vmax=1.0, cmap="Greys", aspect="auto")
            ims.append(im)
            for i, sbin in enumerate(size_order):
                for j, mbin in enumerate(metric_order):
                    v = mat.loc[sbin, mbin]
                    n = cnt.loc[sbin, mbin]
                    if pd.notna(v) and pd.notna(n) and n > 0:
                        ax.text(j, i, f"{(1-v)*100:.0f}%", ha="center", va="center", fontsize=SMALL_FS, color=("white" if v <= 0.45 else "0.15"))
            ax.set_xticks(np.arange(len(metric_order)))
            ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=TICK_FS)
            ax.set_yticks(np.arange(len(size_order)))
            ax.set_yticklabels(size_order, fontsize=TICK_FS)
            ax.set_xlabel(SIZE_METRICS[metric], fontsize=LAB_FS)
            ax.set_ylabel("Pocket size quantile", fontsize=LAB_FS)
            ax.set_title(CLASS_LABELS[pocket_class], fontsize=TITLE_FS)
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("Success rate", fontsize=LAB_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.analysis_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    verbal, quantile, size_strat, manifest = load_tables(args.analysis_dir)
    plot_verbal_profiles(verbal, output_dir / "p2rank_topn_success_failure_verbal_profiles.png")
    plot_quantile_profiles(quantile, output_dir / "p2rank_topn_success_failure_quantile_profiles.png")
    plot_size_stratified_heatmaps(size_strat, manifest, output_dir / "p2rank_topn_failure_size_stratified_heatmaps.png")
    print(f"[SAVE] {output_dir}")


if __name__ == "__main__":
    main()
