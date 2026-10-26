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
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1" / "alphafill_matched_v1"

CLASS_ORDER = ["apo", "holo"]
CLASS_LABELS = {"apo": "AF-Apo", "holo": "AF-Holo"}
CLASS_COLORS = {"apo": "#2C7FB8", "holo": "#F28E2B"}
FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
METRIC_ORDER = ["avg_clash", "avg_clash_count", "avg_local_rmsd", "avg_global_rmsd"]
METRIC_LABELS = {
    "avg_clash": "Average clash score",
    "avg_clash_count": "Average clash count",
    "avg_local_rmsd": "Average local RMSD (A)",
    "avg_global_rmsd": "Average global RMSD (A)",
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
    p = argparse.ArgumentParser(description="Plot P2Rank success against matched AlphaFill metrics.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--success_col", default="success_tns")
    return p.parse_args()


def load_tables(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = analysis_dir / "tables"
    sep = pd.read_csv(tables_dir / "p2rank_alphafill_matched_success_by_separation.csv")
    metric_q = pd.read_csv(tables_dir / "p2rank_alphafill_matched_success_by_metric_quantiles.csv")
    combo = pd.read_csv(tables_dir / "p2rank_alphafill_matched_success_by_metric_and_separation.csv")
    return sep, metric_q, combo


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _success_label(success_col: str) -> str:
    mapping = {
        "success_topn": "Any-rank DCA4 success (%)",
        "success_tns": "TNS DCA4 success (%)",
        "success_tnm": "TNM DCA4 success (%)",
        "success_dca4_sites": "DCA4-#sites success (%)",
    }
    return mapping.get(success_col, f"{success_col} (%)")


def _metric_tag(success_col: str) -> str:
    mapping = {
        "success_topn": "topn",
        "success_tns": "tns",
        "success_tnm": "tnm",
        "success_dca4_sites": "dca4_sites",
    }
    return mapping.get(success_col, success_col.replace("success_", ""))


def plot_success_by_separation(sep: pd.DataFrame, outpath: Path, success_col: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    x = np.arange(len(FULL_LEVEL_ORDER))
    for pocket_class in CLASS_ORDER:
        grp = sep[sep["pocket_class"] == pocket_class].set_index("separation_level_db").reindex(FULL_LEVEL_ORDER)
        ax.plot(x, grp["success_percent"], marker="o", lw=1.8, ms=4.2, color=CLASS_COLORS[pocket_class], label=CLASS_LABELS[pocket_class])
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], rotation=35, ha="right", fontsize=TICK_FS)
    ax.set_xlabel("Separation bin", fontsize=LAB_FS)
    ax.set_ylabel(_success_label(success_col), fontsize=LAB_FS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    ax.set_title("P2Rank success across separation bins in AlphaFill-matched pockets", fontsize=TITLE_FS)
    savefig(fig, outpath)


def plot_success_by_metric_quantiles(metric_q: pd.DataFrame, outpath: Path, success_col: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True, sharey=True)
    axes = axes.flatten()
    for ax, metric in zip(axes, METRIC_ORDER):
        sub = metric_q[metric_q["metric"] == metric].copy()
        order = [f"Q{i}" for i in range(1, 6)]
        x = np.arange(len(order))
        for pocket_class in CLASS_ORDER:
            grp = sub[sub["pocket_class"] == pocket_class].set_index("bin").reindex(order)
            ax.plot(x, grp["success_percent"], marker="o", lw=1.8, ms=4.0, color=CLASS_COLORS[pocket_class], label=CLASS_LABELS[pocket_class])
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=TICK_FS)
        ax.set_xlabel("Metric quantile", fontsize=LAB_FS)
        ax.set_title(METRIC_LABELS[metric], fontsize=TITLE_FS)
        ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(_success_label(success_col), fontsize=LAB_FS)
    axes[2].set_ylabel(_success_label(success_col), fontsize=LAB_FS)
    axes[1].legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath)


def plot_metric_separation_heatmaps(combo: pd.DataFrame, outpath: Path, success_col: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    axes = axes.flatten()
    ims = []
    for ax, metric in zip(axes, METRIC_ORDER):
        sub = combo[(combo["metric"] == metric) & (combo["pocket_class"] == "holo")].copy()
        mat = (
            sub.pivot_table(index="separation_level_db", columns="metric_bin", values="success_rate", aggfunc="mean")
            .reindex(index=FULL_LEVEL_ORDER, columns=[f"Q{i}" for i in range(1, 6)])
        )
        cnt = (
            sub.pivot_table(index="separation_level_db", columns="metric_bin", values="n", aggfunc="sum")
            .reindex(index=FULL_LEVEL_ORDER, columns=[f"Q{i}" for i in range(1, 6)])
        )
        im = ax.imshow(mat.to_numpy(dtype=float), vmin=0.35, vmax=1.0, cmap="Greys", aspect="auto")
        ims.append(im)
        for i, row_label in enumerate(FULL_LEVEL_ORDER):
            for j, col_label in enumerate([f"Q{i}" for i in range(1, 6)]):
                v = mat.loc[row_label, col_label]
                n = cnt.loc[row_label, col_label]
                if pd.notna(v) and pd.notna(n) and n > 0:
                    ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=SMALL_FS, color=("white" if v >= 0.68 else "0.15"))
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([f"Q{i}" for i in range(1, 6)], fontsize=TICK_FS)
        ax.set_yticks(np.arange(len(FULL_LEVEL_ORDER)))
        ax.set_yticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], fontsize=TICK_FS)
        ax.set_xlabel("Metric quantile", fontsize=LAB_FS)
        ax.set_ylabel("Separation bin", fontsize=LAB_FS)
        ax.set_title(f"{METRIC_LABELS[metric]} | AF-Holo", fontsize=TITLE_FS)
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.tolist(), fraction=0.03, pad=0.02)
        cbar.set_label(_success_label(success_col).replace(" (%)", ""), fontsize=LAB_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.analysis_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    sep, metric_q, combo = load_tables(args.analysis_dir)
    tag = _metric_tag(args.success_col)
    plot_success_by_separation(sep, output_dir / f"p2rank_{tag}_afill_matched_success_by_separation.png", args.success_col)
    plot_success_by_metric_quantiles(metric_q, output_dir / f"p2rank_{tag}_afill_matched_success_by_metric_quantiles.png", args.success_col)
    plot_metric_separation_heatmaps(combo, output_dir / f"p2rank_{tag}_afill_matched_holo_metric_separation_heatmaps.png", args.success_col)
    print(f"[SAVE] {output_dir}")


if __name__ == "__main__":
    main()
