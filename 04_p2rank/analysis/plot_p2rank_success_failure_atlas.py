from __future__ import annotations

import argparse
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
DEFAULT_P2RANK_TABLE = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "tables" / "p2rank_st100_merged_per_pocket.csv"
DEFAULT_AF_TABLE = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "alphafill_matched_v1" / "tables" / "p2rank_alphafill_matched_merged.csv"
DEFAULT_OUTPUT_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "success_failure_atlas_v1"

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

GROUPS = [
    ("apo", True, "Apo\nSuccess", "#2C7FB8"),
    ("apo", False, "Apo\nFailure", "#A9CFE8"),
    ("holo", True, "Holo\nSuccess", "#F28E2B"),
    ("holo", False, "Holo\nFailure", "#F8D1A6"),
]

VERBAL_METRICS = [
    ("prob_level_db", ["very_low", "low", "moderate", "high", "very_high"], "Winning probability"),
    ("separation_level_db", ["very_low", "low", "moderate", "high", "very_high"], "Pocket separation"),
    ("density_level_db", ["very_low", "low", "moderate", "high", "very_high"], "Pocket density"),
]

CONT_METRICS = [
    ("pocket_p_holo", "p_holo"),
    ("pocket_p_apo", "p_apo"),
    ("winning_p", "winning p"),
    ("pocket_separation_pair", "pair separation"),
    ("pocket_overlap_pair", "pair overlap"),
    ("pocket_density_pair", "pair density"),
    ("log2_ratio_holo_to_apo_avg_plus1", "log2 holo/apo avg"),
    ("n_holo_avg", "n_holo_avg"),
    ("n_apo_avg", "n_apo_avg"),
    ("ligand_atoms", "ligand atoms"),
    ("predicted_pockets_total", "predicted pockets"),
    ("model_pocket_plddt", "pocket pLDDT"),
]

AF_METRICS = [
    ("avg_clash", "avg clash"),
    ("avg_clash_count", "avg clash count"),
    ("avg_local_rmsd", "avg local RMSD (A)"),
    ("avg_global_rmsd", "avg global RMSD (A)"),
    ("avg_pae_mean", "avg PAE mean"),
    ("avg_pae_median", "avg PAE median"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create success-vs-failure metric atlas for P2Rank pockets.")
    p.add_argument("--p2rank_table", type=Path, default=DEFAULT_P2RANK_TABLE)
    p.add_argument("--alphafill_table", type=Path, default=DEFAULT_AF_TABLE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--success_col", default="success_tns")
    return p.parse_args()


def load_tables(p2rank_table: Path, alphafill_table: Path, success_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(p2rank_table, low_memory=False)
    full = full[full["threshold"] == "ST_100"].copy()
    full["outcome_bool"] = pd.to_numeric(full["dca4rank"], errors="coerce") >= 1 if success_col == "success_topn" else full[success_col].astype(bool)
    af = pd.read_csv(alphafill_table, low_memory=False)
    af["outcome_bool"] = af[success_col].astype(bool)
    return full, af


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _subset(df: pd.DataFrame, pocket_class: str, outcome_bool: bool) -> pd.DataFrame:
    return df[(df["pocket_class"] == pocket_class) & (df["outcome_bool"] == outcome_bool)].copy()


def plot_categorical_atlas(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, len(VERBAL_METRICS), figsize=(10.5, 3.6), constrained_layout=True, sharey=True)
    for ax, (metric, order, title) in zip(axes, VERBAL_METRICS):
        bottoms = np.zeros(len(GROUPS))
        x = np.arange(len(GROUPS))
        palette = plt.cm.Greys(np.linspace(0.25, 0.85, len(order)))
        for idx, bin_label in enumerate(order):
            heights = []
            for pocket_class, outcome_bool, _, _ in GROUPS:
                sub = _subset(df, pocket_class, outcome_bool)
                n = len(sub)
                if n == 0:
                    heights.append(0.0)
                else:
                    heights.append((sub[metric].astype(str) == bin_label).mean() * 100.0)
            ax.bar(x, heights, bottom=bottoms, color=palette[idx], edgecolor="white", linewidth=0.4, label=pretty_level(bin_label))
            bottoms += np.array(heights)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, _, label, _ in GROUPS], fontsize=TICK_FS)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=TITLE_FS)
        ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Share within group (%)", fontsize=LAB_FS)
    axes[-1].legend(frameon=False, fontsize=SMALL_FS, loc="upper right")
    savefig(fig, outpath)


def _box_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str) -> None:
    data = []
    positions = []
    colors = []
    labels = []
    for i, (pocket_class, outcome_bool, label, color) in enumerate(GROUPS):
        sub = _subset(df, pocket_class, outcome_bool)
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        data.append(vals.to_numpy())
        positions.append(i + 1)
        colors.append(color)
        labels.append(label)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.0},
        whiskerprops={"color": "0.35", "linewidth": 0.8},
        capprops={"color": "0.35", "linewidth": 0.8},
        boxprops={"edgecolor": "0.35", "linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_FS)


def plot_continuous_atlas(df: pd.DataFrame, metrics: list[tuple[str, str]], outprefix: Path, cols: int = 3) -> None:
    pages = [metrics[i : i + cols * 3] for i in range(0, len(metrics), cols * 3)]
    for page_idx, page_metrics in enumerate(pages, start=1):
        rows = int(np.ceil(len(page_metrics) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10.5, 3.2 * rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (metric, title) in zip(axes, page_metrics):
            _box_panel(ax, df, metric, title)
        for ax in axes[len(page_metrics):]:
            ax.axis("off")
        axes[0].set_ylabel("Metric value", fontsize=LAB_FS)
        savefig(fig, outprefix.with_name(f"{outprefix.stem}_page{page_idx}.png"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full, af = load_tables(args.p2rank_table, args.alphafill_table, args.success_col)
    plot_categorical_atlas(full, output_dir / "p2rank_topn_success_failure_categorical_atlas.png")
    plot_continuous_atlas(full, CONT_METRICS, output_dir / "p2rank_topn_success_failure_continuous_atlas.png")
    plot_continuous_atlas(af, AF_METRICS, output_dir / "p2rank_topn_success_failure_alphafill_atlas.png")
    print(f"[SAVE] {output_dir}")


if __name__ == "__main__":
    main()
