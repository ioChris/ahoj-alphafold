from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

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
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"
PRIMARY_SUCCESS_COL = "success_tns"

CLASS_ORDER = ["apo", "holo"]
CLASS_LABELS = {"apo": "AF-Apo", "holo": "AF-Holo"}
CLASS_COLORS = {"apo": "#2C7FB8", "holo": "#F28E2B"}
FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high"]
X_MIN, X_MAX = -3.0, 3.0
BIN_W = 0.10
X_EDGES = np.arange(X_MIN, X_MAX + BIN_W, BIN_W)
X_CENTERS = (X_EDGES[:-1] + X_EDGES[1:]) / 2.0

TITLE_FS = 9.0
LAB_FS = 8.0
TICK_FS = 7.0
SMALL_FS = 6.2
PANEL_LETTER_FS = 10.0
FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]
EXPORT_DPI = 300
EXPORT_TRANSPARENT_BACKGROUND = False

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_PREFERENCE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "axes.titlepad": 4,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot P2Rank ST_100 UBS success analyses.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--success_col", default=PRIMARY_SUCCESS_COL)
    p.add_argument("--quantile_bins", type=int, default=5)
    p.add_argument("--probability_bins_include_all_group", action="store_true", default=True)
    p.add_argument("--no_probability_bins_all_group", dest="probability_bins_include_all_group", action="store_false")
    p.add_argument("--probability_bins_all_position", choices=["first", "last"], default="first")
    p.add_argument("--hide_contour_outlines", action="store_true")
    p.add_argument("--x_region_mode", choices=["none", "labels", "labels_line"], default="labels")
    p.add_argument("--hide_panel_letters", action="store_true")
    p.add_argument(
        "--save_svg",
        dest="save_svg",
        action="store_true",
        default=True,
        help="Save SVG copies of generated plots (enabled by default).",
    )
    p.add_argument("--no_save_svg", dest="save_svg", action="store_false", help="Skip SVG copies.")
    p.add_argument(
        "--transparent_background",
        action="store_true",
        help="Save with a transparent figure background while keeping plotting areas white.",
    )
    return p.parse_args()


def pretty_metric_label(success_col: str) -> str:
    mapping = {
        "success_top1": "Top-1 success",
        "success_topn": "Any-rank DCA4 success",
        "success_tns": "TNS DCA4 success",
        "success_tnm": "TNM DCA4 success",
        "success_dca4_sites": "DCA4-#sites success",
        "success_dca_4A": "Success within 4A",
        "success_dca_5A": "Success within 5A",
        "success_dca_6A": "Success within 6A",
        "success_dca_8A": "Success within 8A",
        "success_dca_10A": "Success within 10A",
    }
    return mapping.get(success_col, success_col)


def _metric_tag(success_col: str) -> str:
    mapping = {
        "success_topn": "topn",
        "success_top1": "top1",
        "success_tns": "tns",
        "success_tnm": "tnm",
        "success_dca4_sites": "dca4_sites",
        "success_dca_4A": "dca4",
    }
    return mapping.get(success_col, success_col.replace("success_", "").replace(".", "p"))


def _surface_limits(values: np.ndarray) -> tuple[float, float]:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.floor(np.quantile(arr, 0.01) * 10.0) / 10.0)
    vmin = min(max(vmin, 0.0), 0.9)
    return vmin, 1.0


def _surface_cmap(vmin: float) -> mpl.colors.Colormap:
    base = mpl.colormaps["viridis"]
    start = float(min(max(vmin, 0.0), 0.85))
    colors = base(np.linspace(start, 1.0, 256))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("viridis_trunc", colors)
    cmap.set_bad(alpha=0.0)
    return cmap


def _success_surface_label(success_col: str) -> str:
    if success_col in {"success_topn", "success_tns", "success_tnm", "success_dca4_sites"}:
        return "DCA success rate"
    return pretty_metric_label(success_col)


def _repair_surface_holes(surface: np.ndarray) -> np.ndarray:
    out = np.array(surface, copy=True)
    valid = np.isfinite(out)
    filled_region = binary_fill_holes(valid)
    hole_mask = filled_region & ~valid
    if not np.any(hole_mask):
        return out
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    out[hole_mask] = out[tuple(nearest[:, hole_mask])]
    return out


def load_tables(
    analysis_dir: Path,
    success_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = analysis_dir / "tables"
    merged = pd.read_csv(tables_dir / "p2rank_st100_merged_per_pocket.csv")
    class_summary = pd.read_csv(tables_dir / "p2rank_class_success_by_cutoff.csv")
    tag = _metric_tag(success_col)
    bin_summary = pd.read_csv(tables_dir / f"p2rank_{tag}_success_by_verbal_bins.csv")
    quantile_summary = pd.read_csv(tables_dir / f"p2rank_{tag}_success_by_quantiles.csv")
    correlations = pd.read_csv(tables_dir / f"p2rank_{tag}_continuous_correlations.csv")
    pair_rankings = pd.read_csv(tables_dir / f"p2rank_{tag}_pairwise_combo_rankings.csv") if (tables_dir / f"p2rank_{tag}_pairwise_combo_rankings.csv").exists() else pd.DataFrame()
    pair_grid = pd.read_csv(tables_dir / f"p2rank_{tag}_pairwise_combo_heatmap_data.csv") if (tables_dir / f"p2rank_{tag}_pairwise_combo_heatmap_data.csv").exists() else pd.DataFrame()
    pair_comp = pd.read_csv(tables_dir / f"p2rank_{tag}_pairwise_composite_correlations.csv") if (tables_dir / f"p2rank_{tag}_pairwise_composite_correlations.csv").exists() else pd.DataFrame()
    combined = pd.read_csv(tables_dir / "p2rank_dca4_combined_prob_sep_groups.csv")
    return merged, class_summary, bin_summary, quantile_summary, correlations, combined, pair_rankings, pair_grid, pair_comp


def savefig(fig: plt.Figure, outpath: Path, *, save_svg: bool = False) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if EXPORT_TRANSPARENT_BACKGROUND:
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        for ax in fig.axes:
            ax.set_facecolor("white")
    else:
        fig.patch.set_facecolor("white")
    save_kwargs = {
        "dpi": EXPORT_DPI,
        "bbox_inches": "tight",
        "facecolor": fig.get_facecolor(),
    }
    fig.savefig(outpath, **save_kwargs)
    if save_svg:
        svg_dir = outpath.parent / "SVG"
        svg_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_dir / outpath.with_suffix(".svg").name, **save_kwargs)
    plt.close(fig)


def _text_bbox(pad: float) -> dict[str, object]:
    facecolor = "none" if EXPORT_TRANSPARENT_BACKGROUND else "white"
    return {"facecolor": facecolor, "edgecolor": "none", "pad": pad}


def assemble_2x2_figure(
    panel_paths: list[Path],
    outpath: Path,
    *,
    target_width_px: int = 2100,
    outer_pad: int = 36,
    col_gap: int = 42,
    row_gap: int = 34,
) -> None:
    if len(panel_paths) != 4:
        raise ValueError("assemble_2x2_figure expects exactly four panel paths")
    imgs = [Image.open(path).convert("RGB") for path in panel_paths]
    cell_width = (target_width_px - 2 * outer_pad - col_gap) // 2

    resized = []
    row_heights = [0, 0]
    for idx, img in enumerate(imgs):
        w, h = img.size
        new_h = max(1, round(h * (cell_width / w)))
        img_r = img.resize((cell_width, new_h), Image.Resampling.LANCZOS)
        resized.append(img_r)
        row_heights[idx // 2] = max(row_heights[idx // 2], new_h)

    total_height = 2 * outer_pad + row_heights[0] + row_heights[1] + row_gap
    canvas = Image.new("RGB", (target_width_px, total_height), "white")

    positions = [
        (outer_pad, outer_pad),
        (outer_pad + cell_width + col_gap, outer_pad),
        (outer_pad, outer_pad + row_heights[0] + row_gap),
        (outer_pad + cell_width + col_gap, outer_pad + row_heights[0] + row_gap),
    ]
    for idx, (img, (x0, y0)) in enumerate(zip(resized, positions)):
        row = idx // 2
        x = x0
        y = y0 + (row_heights[row] - img.size[1]) // 2
        canvas.paste(img, (x, y))

    outpath.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(outpath, dpi=(EXPORT_DPI, EXPORT_DPI))


def plot_success_by_cutoff(class_summary: pd.DataFrame, outpath: Path, *, save_svg: bool = False) -> None:
    order = ["success_top1", "success_topn", "success_dca_4A", "success_dca_5A", "success_dca_6A", "success_dca_8A", "success_dca_10A"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    for pocket_class in ["all", "apo", "holo"]:
        sub = class_summary[class_summary["pocket_class"] == pocket_class].copy()
        sub["x"] = sub["success_metric"].map({name: i for i, name in enumerate(order)})
        sub = sub.sort_values("x")
        label = "All" if pocket_class == "all" else CLASS_LABELS[pocket_class]
        color = "0.35" if pocket_class == "all" else CLASS_COLORS[pocket_class]
        ax.plot(sub["x"], sub["success_percent"], marker="o", lw=1.8, ms=4.5, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Top1", "TopN", "4A", "5A", "6A", "8A", "10A"], fontsize=TICK_FS)
    ax.set_ylabel("Success rate (%)", fontsize=LAB_FS)
    ax.set_xlabel("Success definition", fontsize=LAB_FS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(frameon=False, fontsize=SMALL_FS, ncol=3, loc="lower right")
    ax.set_title("P2Rank success by class and distance cutoff", fontsize=TITLE_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_probability_bins(
    bin_summary: pd.DataFrame,
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    include_all_group: bool = False,
    all_position: str = "first",
    panel_letter: str | None = "A",
    save_svg: bool = False,
) -> None:
    sub = bin_summary[bin_summary["bin_type"] == "prob_level_db"].copy()
    sub = sub[sub["bin_label"].isin(FULL_LEVEL_ORDER)].copy()
    x_order = FULL_LEVEL_ORDER.copy()
    if include_all_group:
        if all_position == "first":
            x_order = ["all"] + x_order
        else:
            x_order = x_order + ["all"]
    fig, ax = plt.subplots(figsize=(4.70, 3.22), constrained_layout=True)
    if include_all_group and all_position == "first":
        x = np.array([0.0, 1.7, 2.7, 3.7, 4.7, 5.7], dtype=float)
        divider_x = 0.85
    elif include_all_group and all_position == "last":
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.7], dtype=float)
        divider_x = 4.85
    else:
        x = np.arange(len(x_order), dtype=float)
        divider_x = None
    width = 0.40
    ann_color = "white"
    ann_effect = [pe.withStroke(linewidth=1.15, foreground="0.22")]
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    count_label_min_x = np.inf
    count_label_max_x = -np.inf
    if panel_letter:
        ax.text(-0.082, 1.01, panel_letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=PANEL_LETTER_FS, fontweight="bold")
    ax.set_axisbelow(True)
    for idx, pocket_class in enumerate(CLASS_ORDER):
        grp = sub[sub["pocket_class"] == pocket_class].set_index("bin_label").reindex(FULL_LEVEL_ORDER)
        rates = grp["success_percent"].to_dict()
        counts = pd.to_numeric(grp["n"], errors="coerce").to_dict()
        if include_all_group:
            overall = pd.to_numeric(
                merged.loc[merged["pocket_class"] == pocket_class, success_col],
                errors="coerce",
            )
            rates["all"] = float(overall.mean() * 100.0) if overall.notna().any() else np.nan
            counts["all"] = int(overall.notna().sum()) if overall.notna().any() else np.nan
        xpos = x + (-width / 2 if pocket_class == "apo" else width / 2)
        heights = [rates.get(label, np.nan) for label in x_order]
        bars = ax.bar(
            xpos,
            heights,
            width=width,
            color=CLASS_COLORS[pocket_class],
            edgecolor="0.30",
            linewidth=0.8,
            label=CLASS_LABELS[pocket_class],
            zorder=3,
        )
        for bar, label, height in zip(bars, x_order, heights):
            count_label_min_x = min(count_label_min_x, bar.get_x())
            count_label_max_x = max(count_label_max_x, bar.get_x() + bar.get_width())
            n_val = counts.get(label, np.nan)
            if np.isfinite(n_val):
                txt = ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.02,
                    f"{int(n_val):,}",
                    transform=trans,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=SMALL_FS,
                    color=ann_color,
                    zorder=4,
                )
                txt.set_path_effects(ann_effect)
            if np.isfinite(height):
                txt = ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    max(height - 1.6, 3.2),
                    f"{height:.1f}",
                    ha="center",
                    va="top",
                    fontsize=SMALL_FS,
                    color=ann_color,
                    zorder=4,
                )
                txt.set_path_effects(ann_effect)
    if np.isfinite(count_label_min_x) and np.isfinite(count_label_max_x):
        x_pad = max(0.35, 0.08 * (count_label_max_x - count_label_min_x))
        ax.set_xlim(count_label_min_x - x_pad, count_label_max_x + x_pad)
        ax.text(
            count_label_max_x + 0.58 * x_pad,
            0.02,
            "n pockets",
            transform=trans,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=SMALL_FS,
            color="0.25",
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["All" if v == "all" else pretty_level(v) for v in x_order], fontsize=TICK_FS)
    ax.set_ylabel("Success rate (%)", fontsize=LAB_FS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS, pad=3)
    if divider_x is not None:
        ax.axvline(divider_x, color="0.75", lw=0.9, ymin=0.0, ymax=0.92)
        if all_position == "first":
            ax.text(0.0, -0.14, "Overall", transform=trans, ha="center", va="top", fontsize=TICK_FS, color="0.25")
            ax.text(np.mean(x[1:]), -0.14, "Winning-probability bins", transform=trans, ha="center", va="top", fontsize=TICK_FS, color="0.25")
        else:
            ax.text(np.mean(x[:-1]), -0.14, "Winning-probability bins", transform=trans, ha="center", va="top", fontsize=TICK_FS, color="0.25")
            ax.text(x[-1], -0.14, "Overall", transform=trans, ha="center", va="top", fontsize=TICK_FS, color="0.25")
    ax.legend(frameon=False, fontsize=SMALL_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_log10_ratio(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    panel_letter: str | None = None,
    show_title: bool = True,
    annotate_stats: bool = False,
    save_svg: bool = False,
) -> None:
    sub = merged.copy()
    sub["evidence_n_apo"] = pd.to_numeric(sub["evidence_n_apo"], errors="coerce")
    sub["evidence_n_holo"] = pd.to_numeric(sub["evidence_n_holo"], errors="coerce")
    sub = sub[sub["pocket_class"].isin(CLASS_ORDER)].copy()
    sub = sub[sub["evidence_n_apo"] >= 1].copy()
    sub["evidence_n_holo_adj"] = sub["evidence_n_holo"] + 1.0
    ratio = sub["evidence_n_holo_adj"] / sub["evidence_n_apo"]
    sub["log10_ratio"] = np.log10(ratio.where(ratio > 0))
    sub[success_col] = pd.to_numeric(sub[success_col], errors="coerce")
    sub = sub[np.isfinite(sub["log10_ratio"]) & sub[success_col].notna()].copy()
    sub = sub[(sub["log10_ratio"] >= X_MIN) & (sub["log10_ratio"] <= X_MAX)].copy()

    fig, ax = plt.subplots(figsize=(4.7, 3.25), constrained_layout=True)
    if panel_letter:
        ax.text(-0.08, 1.01, panel_letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=PANEL_LETTER_FS, fontweight="bold")
    stat_lines = []
    for pocket_class in CLASS_ORDER:
        cls = sub[sub["pocket_class"] == pocket_class].copy()
        if cls.empty:
            continue
        bin_id = np.digitize(cls["log10_ratio"].to_numpy(float), X_EDGES) - 1
        success = cls[success_col].to_numpy(float)
        rates = np.full(len(X_EDGES) - 1, np.nan, dtype=float)
        counts = np.zeros(len(X_EDGES) - 1, dtype=int)
        for b in range(len(rates)):
            m = bin_id == b
            if not np.any(m):
                continue
            counts[b] = int(np.sum(m))
            rates[b] = float(np.mean(success[m]) * 100.0)
        ax.plot(
            X_CENTERS,
            rates,
            color=CLASS_COLORS[pocket_class],
            lw=2.0,
            label=CLASS_LABELS[pocket_class],
        )
        if annotate_stats:
            rho = float(cls["log10_ratio"].corr(cls[success_col], method="spearman"))
            stat_lines.append((pocket_class, rho))

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.grid(axis="x", color="#f0f0f0", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xlabel(r"Evidence balance ($\log_{10}$ holo+1/apo ratio)", fontsize=LAB_FS)
    ax.set_ylabel("Success rate (%)", fontsize=LAB_FS)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    if show_title:
        ax.set_title("P2Rank success versus evidence balance", fontsize=TITLE_FS)
    if annotate_stats and stat_lines:
        y0 = 0.97
        for pocket_class, rho in stat_lines:
            txt = ax.text(
                0.03,
                y0,
                f"{CLASS_LABELS[pocket_class]} rho={rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=SMALL_FS,
                color=CLASS_COLORS[pocket_class],
            )
            txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])
            y0 -= 0.08
    savefig(fig, outpath, save_svg=save_svg)


def _heatmap_panel(ax: plt.Axes, panel: pd.DataFrame, title: str) -> None:
    rows = FULL_LEVEL_ORDER
    cols = FULL_LEVEL_ORDER
    rate = (
        panel.pivot_table(index="separation_level_db", columns="density_level_db", values="success_dca_4A", aggfunc="mean")
        .reindex(index=rows, columns=cols)
    )
    count = (
        panel.pivot_table(index="separation_level_db", columns="density_level_db", values="unit_uid", aggfunc="count")
        .reindex(index=rows, columns=cols)
    )
    im = ax.imshow(rate.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="Greys", aspect="auto")
    for i, row_label in enumerate(rows):
        for j, col_label in enumerate(cols):
            r = rate.iloc[i, j]
            n = count.iloc[i, j]
            if pd.notna(r) and pd.notna(n) and n > 0:
                txt = f"{r*100:.0f}%\n(n={int(n)})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=SMALL_FS, color=("white" if r > 0.55 else "0.15"))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([pretty_level(v) for v in cols], rotation=35, ha="right", fontsize=TICK_FS)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([pretty_level(v) for v in rows], fontsize=TICK_FS)
    ax.set_xlabel("Density bin", fontsize=LAB_FS)
    ax.set_ylabel("Separation bin", fontsize=LAB_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    return im


def plot_success_heatmaps(merged: pd.DataFrame, outpath: Path, *, save_svg: bool = False) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(8.2, 3.9), constrained_layout=True)
    ims = []
    for ax, pocket_class in zip(axes, CLASS_ORDER):
        sub = merged[merged["pocket_class"] == pocket_class].copy()
        ims.append(_heatmap_panel(ax, sub, CLASS_LABELS[pocket_class]))
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Success rate", fontsize=LAB_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_support_trends(
    merged: pd.DataFrame,
    outpath: Path,
    success_col: str = "success_dca_4A",
    n_bins: int = 5,
    *,
    save_svg: bool = False,
) -> None:
    features = [
        ("pocket_separation_pair", "Pocket separation"),
        ("pocket_density_pair", "Pocket density"),
        ("winning_p", "Winning probability"),
        ("log2_ratio_holo_to_apo_avg_plus1", "log2 holo/apo avg evidence (+1)"),
    ]
    fig, axes = plt.subplots(ncols=4, figsize=(10.6, 3.3), constrained_layout=True)
    for ax, (feature, xlabel) in zip(axes, features):
        vals = pd.to_numeric(merged[feature], errors="coerce")
        valid = vals.notna() & pd.to_numeric(merged[success_col], errors="coerce").notna()
        if valid.sum() < n_bins * 10:
            continue
        try:
            bins = pd.qcut(vals[valid], q=n_bins, labels=[f"Q{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
        tmp = merged.loc[valid, ["pocket_class", success_col]].copy()
        tmp["bin"] = bins.astype(str)
        tmp["feature_value"] = vals[valid]
        order = list(dict.fromkeys(tmp["bin"].tolist()))
        x = np.arange(len(order))
        for pocket_class in CLASS_ORDER:
            grp = (
                tmp[tmp["pocket_class"] == pocket_class]
                .groupby("bin", sort=False)
                .agg(success_rate=(success_col, "mean"), feature_mean=("feature_value", "mean"))
                .reindex(order)
            )
            ax.plot(x, grp["success_rate"] * 100.0, marker="o", lw=1.8, ms=4.0, color=CLASS_COLORS[pocket_class], label=CLASS_LABELS[pocket_class])
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=TICK_FS)
        ax.set_xlabel(xlabel, fontsize=LAB_FS)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FS)
    axes[0].set_ylabel(f"{pretty_metric_label(success_col)} (%)", fontsize=LAB_FS)
    axes[-1].legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_probability_continuous(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    n_bins: int = 20,
    panel_letter: str | None = None,
    show_title: bool = True,
    save_svg: bool = False,
) -> None:
    feature = "pocket_p_holo"
    xlabel = r"$p_{holo}$"
    vals = pd.to_numeric(merged[feature], errors="coerce")
    success = pd.to_numeric(merged[success_col], errors="coerce")
    valid = vals.notna() & success.notna()
    fig, ax = plt.subplots(figsize=(4.7, 3.25), constrained_layout=True)
    if panel_letter:
        ax.text(-0.08, 1.01, panel_letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=PANEL_LETTER_FS, fontweight="bold")
    if valid.sum() >= n_bins * 8:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        for pocket_class in CLASS_ORDER:
            cls = merged.loc[valid & (merged["pocket_class"] == pocket_class), [feature, success_col]].copy()
            if cls.empty:
                continue
            x = pd.to_numeric(cls[feature], errors="coerce").to_numpy(float)
            y = pd.to_numeric(cls[success_col], errors="coerce").to_numpy(float)
            bin_id = np.digitize(x, edges) - 1
            rates = np.full(len(centers), np.nan, dtype=float)
            for b in range(len(centers)):
                m = bin_id == b
                if not np.any(m):
                    continue
                rates[b] = float(np.mean(y[m]) * 100.0)
            ax.plot(centers, rates, marker="o", lw=2.0, ms=3.8, color=CLASS_COLORS[pocket_class], label=CLASS_LABELS[pocket_class])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 100)
    ax.set_xlabel(xlabel, fontsize=LAB_FS)
    ax.set_ylabel("Success rate (%)", fontsize=LAB_FS)
    if show_title:
        ax.set_title("P2Rank success versus continuous pocket probability", fontsize=TITLE_FS)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.grid(axis="x", color="#f0f0f0", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_probability_ratio_heatmaps(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    x_bins: int = 18,
    y_bins: int = 18,
    annotation_mode: str = "full",
    save_svg: bool = False,
) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(8.2, 4.0), constrained_layout=True, sharex=True, sharey=True)
    ims = []
    x_min, x_max = -3.0, 3.0
    y_min, y_max = 0.0, 1.0
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    for ax, pocket_class in zip(axes, CLASS_ORDER):
        sub = merged[merged["pocket_class"] == pocket_class].copy()
        x = pd.to_numeric(sub["log2_ratio_holo_to_apo_avg_plus1"], errors="coerce")
        y = pd.to_numeric(sub["pocket_p_holo"], errors="coerce")
        s = pd.to_numeric(sub[success_col], errors="coerce")
        valid = x.notna() & y.notna() & s.notna()
        x = x[valid].to_numpy(float)
        y = y[valid].to_numpy(float)
        s = s[valid].to_numpy(float)

        rate = np.full((y_bins, x_bins), np.nan, dtype=float)
        count = np.zeros((y_bins, x_bins), dtype=int)
        x_bin = np.digitize(x, x_edges) - 1
        y_bin = np.digitize(y, y_edges) - 1
        for i in range(y_bins):
            for j in range(x_bins):
                m = (x_bin == j) & (y_bin == i)
                if not np.any(m):
                    continue
                count[i, j] = int(np.sum(m))
                rate[i, j] = float(np.mean(s[m]))

        im = ax.imshow(
            rate,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            vmin=0.0,
            vmax=1.0,
            cmap="Greys",
            aspect="auto",
        )
        ims.append(im)

        # Annotate only well-populated bins to avoid clutter.
        for i, yc in enumerate(y_centers):
            for j, xc in enumerate(x_centers):
                n = count[i, j]
                v = rate[i, j]
                if annotation_mode == "none":
                    continue
                if annotation_mode == "low_n_only":
                    if n > 0 and n < 90 and pd.notna(v):
                        ax.text(
                            xc,
                            yc,
                            "Â·",
                            ha="center",
                            va="center",
                            fontsize=8.0,
                            color=("white" if v >= 0.68 else "0.20"),
                        )
                    continue
                if n >= 90 and pd.notna(v):
                    ax.text(
                        xc,
                        yc,
                        f"{v*100:.0f}%\n{n}",
                        ha="center",
                        va="center",
                        fontsize=5.6,
                        color=("white" if v >= 0.68 else "0.15"),
                    )

        ax.set_title(CLASS_LABELS[pocket_class], fontsize=TITLE_FS)
        ax.set_xlabel(r"Evidence balance ($\log_{2}$ holo+1/apo ratio)", fontsize=LAB_FS)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=TICK_FS)

    axes[0].set_ylabel(r"$p_{holo}$", fontsize=LAB_FS)
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(f"{pretty_metric_label(success_col)}", fontsize=LAB_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_probability_ratio_heatmap_all(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    x_bins: int = 18,
    y_bins: int = 18,
    annotation_mode: str = "full",
    save_svg: bool = False,
) -> None:
    sub = merged.copy()
    x = pd.to_numeric(sub["log2_ratio_holo_to_apo_avg_plus1"], errors="coerce")
    y = pd.to_numeric(sub["pocket_p_holo"], errors="coerce")
    s = pd.to_numeric(sub[success_col], errors="coerce")
    valid = x.notna() & y.notna() & s.notna()
    x = x[valid].to_numpy(float)
    y = y[valid].to_numpy(float)
    s = s[valid].to_numpy(float)

    x_min, x_max = -3.0, 3.0
    y_min, y_max = 0.0, 1.0
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    rate = np.full((y_bins, x_bins), np.nan, dtype=float)
    count = np.zeros((y_bins, x_bins), dtype=int)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    for i in range(y_bins):
        for j in range(x_bins):
            m = (x_bin == j) & (y_bin == i)
            if not np.any(m):
                continue
            count[i, j] = int(np.sum(m))
            rate[i, j] = float(np.mean(s[m]))

    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    im = ax.imshow(
        rate,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        vmin=0.0,
        vmax=1.0,
        cmap="Greys",
        aspect="auto",
    )

    for i, yc in enumerate(y_centers):
        for j, xc in enumerate(x_centers):
            n = count[i, j]
            v = rate[i, j]
            if annotation_mode == "none":
                continue
            if annotation_mode == "low_n_only":
                if n > 0 and n < 140 and pd.notna(v):
                    ax.text(
                        xc,
                        yc,
                        "Â·",
                        ha="center",
                        va="center",
                        fontsize=8.0,
                        color=("white" if v >= 0.68 else "0.20"),
                    )
                continue
            if n >= 140 and pd.notna(v):
                ax.text(
                    xc,
                    yc,
                    f"{v*100:.0f}%\n{n}",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color=("white" if v >= 0.68 else "0.15"),
                )

    ax.set_xlabel(r"Evidence balance ($\log_{2}$ holo+1/apo ratio)", fontsize=LAB_FS)
    ax.set_ylabel(r"$p_{holo}$", fontsize=LAB_FS)
    ax.set_title("All pockets", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(f"{pretty_metric_label(success_col)}", fontsize=LAB_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def _prepare_ratio_prob_success(
    merged: pd.DataFrame,
    success_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = merged.copy()
    x = pd.to_numeric(sub["log2_ratio_holo_to_apo_avg_plus1"], errors="coerce")
    y = pd.to_numeric(sub["pocket_p_holo"], errors="coerce")
    s = pd.to_numeric(sub[success_col], errors="coerce")
    valid = x.notna() & y.notna() & s.notna()
    return x[valid].to_numpy(float), y[valid].to_numpy(float), s[valid].to_numpy(float)


def _prepare_ratio_prob_success_generic(
    merged: pd.DataFrame,
    success_col: str,
    *,
    x_scale: str = "log2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = merged.copy()
    ratio = pd.to_numeric(sub["ratio_holo_to_apo_avg_plus1"], errors="coerce")
    y = pd.to_numeric(sub["pocket_p_holo"], errors="coerce")
    s = pd.to_numeric(sub[success_col], errors="coerce")
    valid = ratio.notna() & y.notna() & s.notna() & (ratio > 0)
    ratio = ratio[valid].to_numpy(float)
    if x_scale == "log10":
        x = np.log10(ratio)
    elif x_scale == "log4":
        x = np.log(ratio) / np.log(4.0)
    else:
        x = np.log2(ratio)
    return x, y[valid].to_numpy(float), s[valid].to_numpy(float)


def plot_success_vs_probability_ratio_hexbin(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    gridsize: int = 28,
    save_svg: bool = False,
) -> None:
    x, y, s = _prepare_ratio_prob_success(merged, success_col)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    hb = ax.hexbin(
        x,
        y,
        C=s,
        reduce_C_function=np.mean,
        gridsize=gridsize,
        extent=(-3.0, 3.0, 0.0, 1.0),
        mincnt=10,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
    )
    ax.set_xlabel(r"Evidence balance ($\log_{2}$ holo+1/apo ratio)", fontsize=LAB_FS)
    ax.set_ylabel(r"$p_{holo}$", fontsize=LAB_FS)
    ax.set_title("All pockets: hexbin mean DCA4 success", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(f"{pretty_metric_label(success_col)}", fontsize=LAB_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_probability_ratio_contours(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    x_bins: int = 55,
    y_bins: int = 45,
    sigma: float = 1.2,
    panel_letter: str | None = None,
    save_svg: bool = False,
) -> None:
    x, y, s = _prepare_ratio_prob_success(merged, success_col)
    x_edges = np.linspace(-3.0, 3.0, x_bins + 1)
    y_edges = np.linspace(0.0, 1.0, y_bins + 1)
    success_sum, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=s)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = success_sum / counts
    valid_mask = counts > 0
    filled = np.where(valid_mask, rate, 0.0)
    smooth_num = gaussian_filter(filled * valid_mask.astype(float), sigma=sigma)
    smooth_den = gaussian_filter(valid_mask.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth_rate = smooth_num / smooth_den
    smooth_rate[smooth_den <= 0.03] = np.nan
    smooth_rate = _repair_surface_holes(smooth_rate)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)
    vmin, vmax = _surface_limits(smooth_rate)
    smooth_rate = np.where(np.isfinite(smooth_rate), np.maximum(smooth_rate, vmin), smooth_rate)
    cmap = _surface_cmap(vmin)

    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    if panel_letter:
        ax.text(-0.08, 1.01, panel_letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=PANEL_LETTER_FS, fontweight="bold")
    levels = np.linspace(vmin, vmax, 13)
    cf = ax.contourf(
        X,
        Y,
        smooth_rate,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        antialiased=True,
    )
    cs = ax.contour(X, Y, smooth_rate, levels=[0.5, 0.6, 0.7, 0.8, 0.9], colors="white", linewidths=0.7, alpha=0.9)
    ax.clabel(cs, inline=True, fmt=lambda v: f"{int(round(v*100))}", fontsize=SMALL_FS)
    ax.set_xlabel(r"Evidence balance ($\log_{2}$ holo+1/apo ratio)", fontsize=LAB_FS)
    ax.set_ylabel(r"$p_{holo}$", fontsize=LAB_FS)
    ax.set_title("All pockets: smoothed DCA4 success contours", fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(_success_surface_label(success_col), fontsize=LAB_FS)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_probability_ratio_contours_marginals(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    x_bins: int = 55,
    y_bins: int = 45,
    sigma: float = 1.2,
    hist_bins: int = 70,
    show_contour_outlines: bool = True,
    n_color_steps: int | None = None,
    x_scale: str = "log2",
    x_limits: tuple[float, float] | None = None,
    x_region_mode: str = "none",
    quadrant_inset: bool = False,
    annotate_stats: bool = False,
    panel_letter: str | None = None,
    save_svg: bool = False,
) -> None:
    x, y, s = _prepare_ratio_prob_success_generic(merged, success_col, x_scale=x_scale)
    if x_limits is not None:
        x_min, x_max = x_limits
    elif x_scale in {"log10", "log4"}:
        q_lo, q_hi = np.quantile(x[np.isfinite(x)], [0.005, 0.995])
        x_limit = 3.0 if x_scale == "log10" else 5.0
        x_min = max(-x_limit, 0.5 * np.floor((q_lo - 0.10) / 0.5))
        x_max = min(x_limit, 0.5 * np.ceil((q_hi + 0.10) / 0.5))
    else:
        x_min, x_max = -3.0, 3.0
    in_range = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (x >= x_min) & (x <= x_max) & (y >= 0.0) & (y <= 1.0)
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(0.0, 1.0, y_bins + 1)
    success_sum, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=s)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = success_sum / counts
    valid_mask = counts > 0
    filled = np.where(valid_mask, rate, 0.0)
    smooth_num = gaussian_filter(filled * valid_mask.astype(float), sigma=sigma)
    smooth_den = gaussian_filter(valid_mask.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth_rate = smooth_num / smooth_den
    smooth_rate[smooth_den <= 0.01] = np.nan
    smooth_rate = _repair_surface_holes(smooth_rate)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)

    x_hist_edges = np.linspace(x_min, x_max, hist_bins + 1)
    y_hist_edges = np.linspace(0.0, 1.0, hist_bins + 1)
    x_hist_centers = (x_hist_edges[:-1] + x_hist_edges[1:]) / 2.0
    y_hist_centers = (y_hist_edges[:-1] + y_hist_edges[1:]) / 2.0
    vmin, vmax = _surface_limits(smooth_rate)
    smooth_rate = np.where(np.isfinite(smooth_rate), np.maximum(smooth_rate, vmin), smooth_rate)
    cmap = _surface_cmap(vmin)
    norm = None
    if n_color_steps is not None and n_color_steps >= 3:
        cmap = cmap.resampled(int(n_color_steps))
        boundaries = np.linspace(vmin, vmax, int(n_color_steps) + 1)
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    fig = plt.figure(figsize=(3.45, 3.12))
    ax = fig.add_axes([0.255, 0.205, 0.58, 0.58])
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="13%", pad=0.035, sharex=ax)
    ax_right = divider.append_axes("right", size="13%", pad=0.035, sharey=ax)
    cax = divider.append_axes("right", size="7.5%", pad=0.075)
    ax.set_box_aspect(1)
    if panel_letter:
        ax_top.text(-0.20, 0.90, panel_letter, transform=ax_top.transAxes, ha="left", va="bottom", fontsize=PANEL_LETTER_FS, fontweight="bold")

    def add_y_region_annotation(y0: float, y1: float, label: str) -> None:
        ymid = (y0 + y1) / 2.0
        trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        x_line = -0.168
        x_text = -0.072
        y_start = y0 + (0.04 if y0 <= 0.0 else 0.0)
        y_end = y1 - (0.04 if y1 >= 1.0 else 0.0)
        ax.plot([x_line, x_line], [y_start, y_end], transform=trans, color="0.48", lw=0.75, clip_on=False, zorder=4)
        ax.text(
            x_text,
            ymid,
            label,
            transform=trans,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.1,
            color="0.28",
            bbox=_text_bbox(0.14),
            clip_on=False,
            zorder=5,
        )

    def add_x_region_annotation(x0: float, x1: float, label: str) -> None:
        xmid = (x0 + x1) / 2.0
        trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            xmid,
            -0.070,
            label,
            transform=trans,
            ha="center",
            va="center",
            fontsize=7.1,
            color="0.28",
            bbox=_text_bbox(0.12),
            clip_on=False,
            zorder=5,
        )

    levels = np.linspace(vmin, vmax, 13)
    image_kwargs = {
        "origin": "lower",
        "extent": [x_min, x_max, 0.0, 1.0],
        "interpolation": "bicubic",
        "aspect": "auto",
        "cmap": cmap,
    }
    if norm is None:
        image_kwargs["vmin"] = vmin
        image_kwargs["vmax"] = vmax
    else:
        image_kwargs["norm"] = norm
    mesh = ax.imshow(smooth_rate, **image_kwargs)
    if show_contour_outlines:
        cs = ax.contour(
            X,
            Y,
            smooth_rate,
            levels=[0.5, 0.6, 0.7, 0.8, 0.9],
            colors="white",
            linewidths=0.55,
            alpha=0.68,
        )
        ax.clabel(cs, inline=True, fmt=lambda v: f"{int(round(v * 100))}", fontsize=SMALL_FS)
    guide_kw = dict(color="white", lw=0.75, linestyle=(0, (2.0, 2.8)), alpha=0.42, zorder=3)
    ax.axhline(0.5, **guide_kw)
    ax.axvline(0.0, **guide_kw)

    x_hist_max = 0.0
    y_hist_max = 0.0
    for pocket_class in CLASS_ORDER:
        sub = merged[merged["pocket_class"] == pocket_class].copy()
        ratio_cls = pd.to_numeric(sub["ratio_holo_to_apo_avg_plus1"], errors="coerce")
        y_cls = pd.to_numeric(sub["pocket_p_holo"], errors="coerce")
        s_cls = pd.to_numeric(sub[success_col], errors="coerce")
        valid_cls = ratio_cls.notna() & y_cls.notna() & s_cls.notna() & (ratio_cls > 0)
        ratio_vals = ratio_cls[valid_cls].to_numpy(float)
        if x_scale == "log10":
            x_vals = np.log10(ratio_vals)
        elif x_scale == "log4":
            x_vals = np.log(ratio_vals) / np.log(4.0)
        else:
            x_vals = np.log2(ratio_vals)
        y_vals = y_cls[valid_cls].to_numpy(float)
        if len(x_vals) == 0:
            continue
        if pocket_class == "apo":
            w_cls = pd.to_numeric(sub["n_apo_avg"], errors="coerce")
        else:
            w_cls = pd.to_numeric(sub["n_holo_avg_plus1"], errors="coerce")
        weights = w_cls[valid_cls].to_numpy(float)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if np.nansum(weights) <= 0:
            continue
        x_hist, _ = np.histogram(x_vals, bins=x_hist_edges, weights=weights, density=True)
        y_hist, _ = np.histogram(y_vals, bins=y_hist_edges, weights=weights, density=True)
        x_hist = gaussian_filter1d(x_hist.astype(float), sigma=1.0)
        y_hist = gaussian_filter1d(y_hist.astype(float), sigma=1.0)
        x_hist_max = max(x_hist_max, float(np.nanmax(x_hist)))
        y_hist_max = max(y_hist_max, float(np.nanmax(y_hist)))
        top_line = ax_top.plot(x_hist_centers, x_hist, color=CLASS_COLORS[pocket_class], lw=1.2, zorder=2, solid_capstyle="butt", clip_on=True)[0]
        right_line = ax_right.plot(y_hist, y_hist_centers, color=CLASS_COLORS[pocket_class], lw=1.2, zorder=2, solid_capstyle="butt", clip_on=True)[0]
        top_line.set_clip_path(ax_top.patch)
        right_line.set_clip_path(ax_right.patch)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim(x_min, x_max)
    ax_top.set_xlim(x_min, x_max)
    ax_top.set_ylim(0.0, x_hist_max * 1.34 if x_hist_max > 0 else 1.0)
    ax_top.margins(x=0.0)
    ax_top.set_yticks([])
    ax_top.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_linewidth(0.8)

    ax_right.set_xlim(0.0, y_hist_max * 1.34 if y_hist_max > 0 else 1.0)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.margins(y=0.0)
    ax_right.set_xticks([])
    ax_right.tick_params(axis="y", labelleft=False, left=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["left"].set_linewidth(0.8)
    ax_right.spines["left"].set_bounds(0.0, 1.0)

    log_label = {"log2": "2", "log10": "10", "log4": "4"}.get(x_scale, "2")
    ax.set_xlabel(rf"Evidence balance ($\log_{{{log_label}}}$ holo/apo ratio)", fontsize=LAB_FS)
    ax.set_ylabel(r"$p_{holo}$", fontsize=LAB_FS, labelpad=22)
    ax.yaxis.set_label_coords(-0.196, 0.5)
    ax.set_xticks([x_min, 0.0, x_max])
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS, pad=1.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for tick_label in ax.get_yticklabels():
        tick_label.set_bbox(_text_bbox(0.14))
    add_y_region_annotation(0.0, 0.5, "AF-Apo")
    add_y_region_annotation(0.5, 1.0, "AF-Holo")
    if x_region_mode in {"labels", "labels_line"}:
        if x_region_mode == "labels_line":
            x_trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            y_line = -0.112
            # Use separated guide segments so the line clears the x=0 tick and sits behind neither label.
            seg_pad = 0.10 * min(abs(0.0 - x_min), abs(x_max - 0.0))
            center_gap = 0.18 * (x_max - x_min)
            ax.plot(
                [x_min + seg_pad, -center_gap / 2.0],
                [y_line, y_line],
                transform=x_trans,
                color="0.48",
                lw=0.75,
                clip_on=False,
                zorder=4,
            )
            ax.plot(
                [center_gap / 2.0, x_max - seg_pad],
                [y_line, y_line],
                transform=x_trans,
                color="0.48",
                lw=0.75,
                clip_on=False,
                zorder=4,
            )
            ax.xaxis.set_label_coords(0.5, -0.165)
        add_x_region_annotation(x_min, 0.0, "Exp-Apo ↑")
        add_x_region_annotation(0.0, x_max, "Exp-Holo ↑")

    if annotate_stats:
        rho_x = float(pd.Series(x[in_range]).corr(pd.Series(s[in_range]), method="spearman"))
        rho_y = float(pd.Series(y[in_range]).corr(pd.Series(s[in_range]), method="spearman"))
        log_label = {"log2": "2", "log4": "4", "log10": "10"}.get(x_scale, "2")
        t1 = fig.text(0.24, 0.84, rf"$\rho$(log$_{{{log_label}}}$ ratio)={rho_x:.2f}", ha="left", va="top", fontsize=6.2, color="0.22")
        t2 = fig.text(0.24, 0.79, rf"$\rho$($p_{{holo}}$)={rho_y:.2f}", ha="left", va="top", fontsize=6.2, color="0.22")
        for txt in (t1, t2):
            txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])

    if quadrant_inset:
        q_ax = fig.add_axes([0.806, 0.835, 0.105, 0.105])
        q_ax.set_facecolor("white")
        xq = x[in_range]
        yq = y[in_range]
        sq = s[in_range]
        quad = np.full((2, 2), np.nan, dtype=float)
        quad_n = np.zeros((2, 2), dtype=int)
        masks = {
            (0, 0): (yq >= 0.5) & (xq < 0.0),
            (0, 1): (yq >= 0.5) & (xq >= 0.0),
            (1, 0): (yq < 0.5) & (xq < 0.0),
            (1, 1): (yq < 0.5) & (xq >= 0.0),
        }
        for (i, j), m in masks.items():
            if np.any(m):
                quad[i, j] = float(np.mean(sq[m]))
                quad_n[i, j] = int(np.sum(m))
        q_ax.imshow(quad, cmap=_surface_cmap(0.3), vmin=0.3, vmax=1.0, aspect="equal")
        q_ax.set_xticks([])
        q_ax.set_yticks([])
        q_ax.tick_params(length=0, pad=0)
        for spine in q_ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_edgecolor("0.45")
        q_ax.axhline(0.5, color="white", lw=0.8)
        q_ax.axvline(0.5, color="white", lw=0.8)
        q_ax.text(1.10, 0.5, "Quadrants", transform=q_ax.transAxes, rotation=270, ha="left", va="center", fontsize=4.6, color="0.25")
        for i in range(2):
            for j in range(2):
                if np.isfinite(quad[i, j]):
                    q_ax.text(
                        j,
                        i - 0.10,
                        f"{quad[i, j]*100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=4.0,
                        color="white",
                        fontweight="bold",
                    )
                    q_ax.text(
                        j,
                        i + 0.22,
                        f"{quad_n[i, j]:,}",
                        ha="center",
                        va="center",
                        fontsize=3.5,
                        color="white",
                    )

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Success rate", fontsize=LAB_FS, rotation=270, labelpad=12)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_vs_probability_ratio_surface3d(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    x_bins: int = 45,
    y_bins: int = 35,
    sigma: float = 1.2,
    save_svg: bool = False,
) -> None:
    x, y, s = _prepare_ratio_prob_success(merged, success_col)
    x_edges = np.linspace(-3.0, 3.0, x_bins + 1)
    y_edges = np.linspace(0.0, 1.0, y_bins + 1)
    success_sum, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=s)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = success_sum / counts
    valid_mask = counts > 0
    filled = np.where(valid_mask, rate, 0.0)
    smooth_num = gaussian_filter(filled * valid_mask.astype(float), sigma=sigma)
    smooth_den = gaussian_filter(valid_mask.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth_rate = smooth_num / smooth_den
    smooth_rate[smooth_den <= 0.03] = np.nan
    smooth_rate = _repair_surface_holes(smooth_rate)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)
    vmin, vmax = _surface_limits(smooth_rate)
    smooth_rate = np.where(np.isfinite(smooth_rate), np.maximum(smooth_rate, vmin), smooth_rate)
    cmap = _surface_cmap(vmin)

    fig = plt.figure(figsize=(6.4, 4.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        np.ma.masked_invalid(smooth_rate),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.25,
        edgecolor=(1.0, 1.0, 1.0, 0.35),
        antialiased=True,
    )
    ax.contour(
        X,
        Y,
        np.nan_to_num(smooth_rate, nan=vmin),
        levels=[0.5, 0.6, 0.7, 0.8, 0.9],
        zdir="z",
        offset=vmin,
        cmap=cmap,
        linewidths=1.0,
    )
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel(r"Evidence balance ($\log_{2}$ holo+1/apo ratio)", fontsize=LAB_FS, labelpad=8)
    ax.set_ylabel(r"$p_{holo}$", fontsize=LAB_FS, labelpad=8)
    ax.set_zlabel("DCA success rate", fontsize=LAB_FS, labelpad=6)
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=2)
    ax.zaxis.set_tick_params(labelsize=TICK_FS, pad=2)
    ax.view_init(elev=28, azim=-128)
    ax.set_title("All pockets: 3D DCA4 success surface", fontsize=TITLE_FS, pad=8)
    cbar = fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.08, shrink=0.82)
    cbar.set_label("DCA success rate", fontsize=LAB_FS)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_support_trends_verbal(
    merged: pd.DataFrame,
    outpath: Path,
    success_col: str = PRIMARY_SUCCESS_COL,
    *,
    save_svg: bool = False,
) -> None:
    features = [
        ("separation_level_db", "Pocket separation"),
        ("density_level_db", "Pocket density"),
        ("prob_level_db", "Winning probability"),
    ]
    fig, axes = plt.subplots(ncols=3, figsize=(8.3, 3.3), constrained_layout=True, sharey=True)
    x = np.arange(len(FULL_LEVEL_ORDER))
    for ax, (feature, xlabel) in zip(axes, features):
        tmp = merged[["pocket_class", feature, success_col]].copy()
        tmp[feature] = tmp[feature].astype(str)
        for pocket_class in CLASS_ORDER:
            grp = (
                tmp[tmp["pocket_class"] == pocket_class]
                .groupby(feature, sort=False)
                .agg(success_rate=(success_col, "mean"))
                .reindex(FULL_LEVEL_ORDER)
            )
            ax.plot(
                x,
                grp["success_rate"] * 100.0,
                marker="o",
                lw=1.8,
                ms=4.0,
                color=CLASS_COLORS[pocket_class],
                label=CLASS_LABELS[pocket_class],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], rotation=35, ha="right", fontsize=TICK_FS)
        ax.set_xlabel(xlabel, fontsize=LAB_FS)
        ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(f"{pretty_metric_label(success_col)} (%)", fontsize=LAB_FS)
    axes[-1].legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath, save_svg=save_svg)


def plot_success_by_ratio_bins(
    merged: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    x_limits: tuple[float, float] = (-3.0, 3.0),
    save_svg: bool = False,
) -> None:
    ratio_edges = np.arange(x_limits[0], x_limits[1] + 1.0, 1.0, dtype=float)
    if len(ratio_edges) < 2:
        return
    labels = []
    for lo, hi in zip(ratio_edges[:-1], ratio_edges[1:]):
        labels.append(f"{2**lo:.3g}-{2**hi:.3g}")

    sub = merged[["pocket_class", "ratio_holo_to_apo_avg_plus1", success_col]].copy()
    ratio = pd.to_numeric(sub["ratio_holo_to_apo_avg_plus1"], errors="coerce")
    sub["log2_ratio"] = np.log2(ratio.where(ratio > 0))
    sub[success_col] = pd.to_numeric(sub[success_col], errors="coerce")
    valid = sub["pocket_class"].isin(CLASS_ORDER) & sub["log2_ratio"].notna() & sub[success_col].notna()
    sub = sub[valid].copy()
    sub = sub[(sub["log2_ratio"] >= x_limits[0]) & (sub["log2_ratio"] <= x_limits[1])].copy()
    sub["ratio_bin"] = pd.cut(
        sub["log2_ratio"],
        bins=ratio_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    fig, ax = plt.subplots(figsize=(5.1, 3.3), constrained_layout=True)
    x = np.arange(len(labels))
    for pocket_class in CLASS_ORDER:
        grp = (
            sub[sub["pocket_class"] == pocket_class]
            .groupby("ratio_bin", observed=False)
            .agg(success_rate=(success_col, "mean"), n=(success_col, "size"))
            .reindex(labels)
        )
        ax.plot(
            x,
            grp["success_rate"] * 100.0,
            marker="o",
            lw=1.8,
            ms=4.2,
            color=CLASS_COLORS[pocket_class],
            label=CLASS_LABELS[pocket_class],
        )
    ax.axvline(2.5, color="0.75", lw=0.9, ymin=0.0, ymax=0.94)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=TICK_FS)
    ax.set_xlabel("Holo/apo evidence ratio bins", fontsize=LAB_FS)
    ax.set_ylabel(f"{_success_surface_label(success_col)} (%)", fontsize=LAB_FS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath, save_svg=save_svg)


def plot_top_correlations(correlations: pd.DataFrame, outpath: Path, *, save_svg: bool = False) -> None:
    top = correlations.head(12).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    ax.barh(top["feature"], top["spearman_r_success"], color="0.45", edgecolor="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", lw=0.9)
    ax.set_xlabel("Spearman r with DCA4 success", fontsize=LAB_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#ebebeb", lw=0.7, linestyle="--")
    ax.set_title("Top continuous correlates of P2Rank success", fontsize=TITLE_FS)
    savefig(fig, outpath, save_svg=save_svg)


def _combined_matrix(combined: pd.DataFrame, pocket_class: str, value_col: str) -> pd.DataFrame:
    sub = combined[combined["pocket_class"] == pocket_class].copy()
    return (
        sub.pivot_table(index="prob_level_db", columns="separation_level_db", values=value_col, aggfunc="mean")
        .reindex(index=FULL_LEVEL_ORDER, columns=FULL_LEVEL_ORDER)
    )


def plot_combined_success_heatmaps(
    combined: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str,
    save_svg: bool = False,
) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(8.2, 4.1), constrained_layout=True)
    ims = []
    value_col = f"{success_col}_rate"
    for ax, pocket_class in zip(axes, CLASS_ORDER):
        rate = _combined_matrix(combined, pocket_class, value_col)
        count = _combined_matrix(combined, pocket_class, "n")
        im = ax.imshow(rate.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="Greys", aspect="auto")
        ims.append(im)
        for i, row_label in enumerate(FULL_LEVEL_ORDER):
            for j, col_label in enumerate(FULL_LEVEL_ORDER):
                r = rate.loc[row_label, col_label]
                n = count.loc[row_label, col_label]
                if pd.notna(r) and pd.notna(n) and n > 0:
                    ax.text(
                        j,
                        i,
                        f"{r*100:.0f}%\n(n={int(n)})",
                        ha="center",
                        va="center",
                        fontsize=SMALL_FS,
                        color=("white" if r >= 0.58 else "0.15"),
                    )
        ax.set_xticks(np.arange(len(FULL_LEVEL_ORDER)))
        ax.set_xticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], rotation=35, ha="right", fontsize=TICK_FS)
        ax.set_yticks(np.arange(len(FULL_LEVEL_ORDER)))
        ax.set_yticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], fontsize=TICK_FS)
        ax.set_xlabel("Separation bin", fontsize=LAB_FS)
        ax.set_ylabel("Winning-probability bin", fontsize=LAB_FS)
        ax.set_title(CLASS_LABELS[pocket_class], fontsize=TITLE_FS)
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(pretty_metric_label(success_col), fontsize=LAB_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_combined_distance_heatmaps(combined: pd.DataFrame, outpath: Path, *, save_svg: bool = False) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(8.2, 4.1), constrained_layout=True)
    ims = []
    vmax = float(pd.to_numeric(combined["closest_pocket_dist_median"], errors="coerce").quantile(0.95))
    vmax = max(vmax, 4.0)
    for ax, pocket_class in zip(axes, CLASS_ORDER):
        dist = _combined_matrix(combined, pocket_class, "closest_pocket_dist_median")
        count = _combined_matrix(combined, pocket_class, "n")
        im = ax.imshow(dist.to_numpy(dtype=float), vmin=0.0, vmax=vmax, cmap="magma_r", aspect="auto")
        ims.append(im)
        for i, row_label in enumerate(FULL_LEVEL_ORDER):
            for j, col_label in enumerate(FULL_LEVEL_ORDER):
                d = dist.loc[row_label, col_label]
                n = count.loc[row_label, col_label]
                if pd.notna(d) and pd.notna(n) and n > 0:
                    ax.text(
                        j,
                        i,
                        f"{d:.1f}A\n(n={int(n)})",
                        ha="center",
                        va="center",
                        fontsize=SMALL_FS,
                        color=("white" if d >= vmax * 0.55 else "0.12"),
                    )
        ax.set_xticks(np.arange(len(FULL_LEVEL_ORDER)))
        ax.set_xticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], rotation=35, ha="right", fontsize=TICK_FS)
        ax.set_yticks(np.arange(len(FULL_LEVEL_ORDER)))
        ax.set_yticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], fontsize=TICK_FS)
        ax.set_xlabel("Separation bin", fontsize=LAB_FS)
        ax.set_ylabel("Winning-probability bin", fontsize=LAB_FS)
        ax.set_title(CLASS_LABELS[pocket_class], fontsize=TITLE_FS)
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label("Median center distance (A)", fontsize=LAB_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_combined_prob_sep_lines(
    combined: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    save_svg: bool = False,
) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(8.6, 3.6), constrained_layout=True, sharey=True)
    prob_palette = {
        "very_low": "#d9d9d9",
        "low": "#bdbdbd",
        "moderate": "#969696",
        "high": "#636363",
        "very_high": "#252525",
    }
    x = np.arange(len(FULL_LEVEL_ORDER))
    for ax, pocket_class in zip(axes, CLASS_ORDER):
        sub = combined[combined["pocket_class"] == pocket_class].copy()
        for prob_level in FULL_LEVEL_ORDER:
            grp = (
                sub[sub["prob_level_db"] == prob_level]
                .set_index("separation_level_db")
                .reindex(FULL_LEVEL_ORDER)
            )
            if grp["n"].fillna(0).sum() == 0:
                continue
            success_percent_col = f"{success_col}_percent"
            ax.plot(
                x,
                grp[success_percent_col],
                marker="o",
                lw=1.7,
                ms=4.0,
                color=prob_palette[prob_level],
                label=pretty_level(prob_level),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_level(v) for v in FULL_LEVEL_ORDER], rotation=35, ha="right", fontsize=TICK_FS)
        ax.set_xlabel("Separation bin", fontsize=LAB_FS)
        ax.set_title(CLASS_LABELS[pocket_class], fontsize=TITLE_FS)
        ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FS)
    axes[0].set_ylabel(f"{_success_surface_label(success_col)} (%)", fontsize=LAB_FS)
    axes[-1].legend(frameon=False, fontsize=SMALL_FS, title="Winning p", title_fontsize=SMALL_FS, loc="lower right")
    savefig(fig, outpath, save_svg=save_svg)


def _pretty_feature_label(feature: str) -> str:
    mapping = {
        "pocket_p_holo": "p_holo",
        "pocket_p_apo": "p_apo",
        "winning_p": "winning p",
        "pocket_overlap_pair": "pair overlap",
        "pocket_separation_pair": "pair separation",
        "pocket_density_pair": "pair density",
        "log2_ratio_holo_to_apo_avg_plus1": "log2 holo/apo avg",
        "ligand_atoms": "ligand atoms",
        "n_holo_avg": "n_holo_avg",
    }
    return mapping.get(feature, feature)


def plot_pairwise_combo_rankings(pair_rankings: pd.DataFrame, outpath: Path, *, save_svg: bool = False) -> None:
    if pair_rankings.empty:
        return
    top = pair_rankings.head(12).iloc[::-1].copy()
    top["label"] = top.apply(lambda r: f"{_pretty_feature_label(r['feature_x'])} + {_pretty_feature_label(r['feature_y'])}", axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    ax.barh(top["label"], top["delta_success_percent"], color="0.45", edgecolor="0.25", linewidth=0.8)
    ax.set_xlabel("Favorable-vs-unfavorable success gap (pp)", fontsize=LAB_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="x", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Strongest pairwise success enrichments", fontsize=TITLE_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_pairwise_combo_heatmaps(
    pair_grid: pd.DataFrame,
    pair_rankings: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    save_svg: bool = False,
) -> None:
    if pair_grid.empty or pair_rankings.empty:
        return
    preferred_pairs = [
        ("pocket_p_holo", "pocket_overlap_pair"),
        ("pocket_p_holo", "pocket_separation_pair"),
        ("pocket_p_holo", "log2_ratio_holo_to_apo_avg_plus1"),
        ("pocket_p_holo", "ligand_atoms"),
    ]
    selected = []
    for fx, fy in preferred_pairs:
        sub = pair_grid[(pair_grid["feature_x"] == fx) & (pair_grid["feature_y"] == fy)]
        if not sub.empty:
            selected.append((fx, fy))
    if len(selected) < 4:
        for _, row in pair_rankings.iterrows():
            pair = (row["feature_x"], row["feature_y"])
            if pair not in selected:
                selected.append(pair)
            if len(selected) == 4:
                break
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    axes = axes.flatten()
    ims = []
    for ax, pair in zip(axes, selected):
        fx, fy = pair
        sub = pair_grid[(pair_grid["feature_x"] == fx) & (pair_grid["feature_y"] == fy)].copy()
        mat = (
            sub.pivot_table(index="y_bin", columns="x_bin", values="success_rate", aggfunc="mean")
            .reindex(index=[f"Q{i}" for i in range(1, 6)], columns=[f"Q{i}" for i in range(1, 6)])
        )
        cnt = (
            sub.pivot_table(index="y_bin", columns="x_bin", values="n", aggfunc="sum")
            .reindex(index=[f"Q{i}" for i in range(1, 6)], columns=[f"Q{i}" for i in range(1, 6)])
        )
        im = ax.imshow(mat.to_numpy(dtype=float), vmin=0.4, vmax=0.95, cmap="Greys", aspect="auto")
        ims.append(im)
        for i, row_lab in enumerate(mat.index):
            for j, col_lab in enumerate(mat.columns):
                v = mat.loc[row_lab, col_lab]
                n = cnt.loc[row_lab, col_lab]
                if pd.notna(v) and pd.notna(n) and n > 0:
                    ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=SMALL_FS, color=("white" if v >= 0.68 else "0.15"))
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([f"Q{i}" for i in range(1, 6)], fontsize=TICK_FS)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels([f"Q{i}" for i in range(1, 6)], fontsize=TICK_FS)
        ax.set_xlabel(_pretty_feature_label(fx), fontsize=LAB_FS)
        ax.set_ylabel(_pretty_feature_label(fy), fontsize=LAB_FS)
        ax.set_title(f"{_pretty_feature_label(fx)} + {_pretty_feature_label(fy)}", fontsize=TITLE_FS)
    for ax in axes[len(selected):]:
        ax.axis("off")
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.tolist(), fraction=0.03, pad=0.02)
        cbar.set_label(_success_surface_label(success_col), fontsize=LAB_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)
    savefig(fig, outpath, save_svg=save_svg)


def plot_pairwise_composite_correlations(
    pair_comp: pd.DataFrame,
    outpath: Path,
    *,
    success_col: str = PRIMARY_SUCCESS_COL,
    save_svg: bool = False,
) -> None:
    if pair_comp.empty:
        return
    top = pair_comp.sort_values("spearman_r_composite", ascending=False, key=lambda s: s.abs()).head(14).iloc[::-1].copy()
    top["label"] = top.apply(lambda r: f"{_pretty_feature_label(r['feature_x'])} + {_pretty_feature_label(r['feature_y'])}", axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    colors = ["#4d4d4d" if v >= 0 else "#9c9c9c" for v in top["spearman_r_composite"]]
    ax.barh(top["label"], top["spearman_r_composite"], color=colors, edgecolor="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", lw=0.9)
    ax.set_xlabel(f"Spearman r with {_success_surface_label(success_col).lower()}", fontsize=LAB_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="x", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Pairwise composite correlations without ligand size", fontsize=TITLE_FS)
    savefig(fig, outpath, save_svg=save_svg)


def main() -> None:
    global EXPORT_TRANSPARENT_BACKGROUND
    args = parse_args()
    EXPORT_TRANSPARENT_BACKGROUND = args.transparent_background
    output_dir = args.output_dir or (args.analysis_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged, class_summary, bin_summary, quantile_summary, correlations, combined, pair_rankings, pair_grid, pair_comp = load_tables(args.analysis_dir, args.success_col)
    tag = _metric_tag(args.success_col)
    panel_letters_enabled = not args.hide_panel_letters

    plot_success_by_cutoff(class_summary, output_dir / "p2rank_success_by_cutoff.png", save_svg=args.save_svg)
    plot_probability_bins(
        bin_summary,
        merged,
        output_dir / f"p2rank_{tag}_success_by_probability_bins.png",
        success_col=args.success_col,
        include_all_group=args.probability_bins_include_all_group,
        all_position=args.probability_bins_all_position,
        panel_letter="A" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_log10_ratio(
        merged,
        output_dir / f"p2rank_{tag}_success_vs_log10_holo_apo_ratio.png",
        success_col=args.success_col,
        panel_letter="C" if panel_letters_enabled else None,
        show_title=False,

        save_svg=args.save_svg,
    )
    plot_success_heatmaps(merged, output_dir / f"p2rank_{tag}_success_heatmaps_by_class.png", save_svg=args.save_svg)
    plot_support_trends(
        merged,
        output_dir / f"p2rank_{tag}_success_vs_support_trends.png",
        success_col=args.success_col,
        n_bins=args.quantile_bins,
        save_svg=args.save_svg,
    )
    plot_success_by_ratio_bins(
        merged,
        output_dir / f"p2rank_{tag}_success_by_ratio_bins.png",
        success_col=args.success_col,
        x_limits=(-3.0, 3.0),
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_continuous(
        merged,
        output_dir / f"p2rank_{tag}_success_vs_continuous_probabilities.png",
        success_col=args.success_col,
        panel_letter="B" if panel_letters_enabled else None,
        show_title=False,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_heatmap_all(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_heatmap_all.png",
        success_col=args.success_col,
        annotation_mode="low_n_only",
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_hexbin(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_hexbin.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours.png",
        success_col=args.success_col,
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log2_x4.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        x_scale="log2",
        x_limits=(-4.0, 4.0),
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log2_x4_reduced_colors.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        n_color_steps=8,
        x_scale="log2",
        x_limits=(-4.0, 4.0),
        x_region_mode=args.x_region_mode,
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log2_x4_reduced_colors_xline.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        n_color_steps=8,
        x_scale="log2",
        x_limits=(-4.0, 4.0),
        x_region_mode="labels_line",
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log2_x4_reduced_colors_quadrants.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        n_color_steps=8,
        x_scale="log2",
        x_limits=(-4.0, 4.0),
        quadrant_inset=True,
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log2_x5.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        x_scale="log2",
        x_limits=(-5.0, 5.0),
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log10.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        x_scale="log10",
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log4.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        x_scale="log4",
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_contours_marginals(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_reduced_colors.png",
        success_col=args.success_col,
        show_contour_outlines=not args.hide_contour_outlines,
        n_color_steps=8,
        panel_letter="D" if panel_letters_enabled else None,
        save_svg=args.save_svg,
    )
    plot_success_vs_probability_ratio_surface3d(
        merged,
        output_dir / f"p2rank_{tag}_success_probability_ratio_surface3d.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    plot_support_trends_verbal(
        merged,
        output_dir / f"p2rank_{tag}_success_vs_support_verbal_bins.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    plot_top_correlations(correlations, output_dir / f"p2rank_{tag}_top_correlations.png", save_svg=args.save_svg)
    plot_combined_success_heatmaps(
        combined,
        output_dir / f"p2rank_{tag}_combined_prob_sep_heatmaps.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    plot_combined_distance_heatmaps(combined, output_dir / f"p2rank_{tag}_combined_prob_sep_distance_heatmaps.png", save_svg=args.save_svg)
    plot_combined_prob_sep_lines(
        combined,
        output_dir / f"p2rank_{tag}_combined_prob_sep_lines.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    plot_pairwise_combo_rankings(pair_rankings, output_dir / f"p2rank_{tag}_pairwise_combo_rankings.png", save_svg=args.save_svg)
    plot_pairwise_combo_heatmaps(
        pair_grid,
        pair_rankings,
        output_dir / f"p2rank_{tag}_pairwise_combo_heatmaps.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    plot_pairwise_composite_correlations(
        pair_comp,
        output_dir / f"p2rank_{tag}_pairwise_composite_correlations.png",
        success_col=args.success_col,
        save_svg=args.save_svg,
    )
    figure_panels = [
        output_dir / f"p2rank_{tag}_success_by_probability_bins.png",
        output_dir / f"p2rank_{tag}_success_vs_continuous_probabilities.png",
        output_dir / f"p2rank_{tag}_success_vs_log10_holo_apo_ratio.png",
        output_dir / f"p2rank_{tag}_success_probability_ratio_contours_marginals_log2_x4_reduced_colors.png",
    ]
    assemble_2x2_figure(
        figure_panels,
        output_dir / f"p2rank_{tag}_manuscript_figure_panels_ABCD.png",
    )
    print(f"[SAVE] {output_dir}")


if __name__ == "__main__":
    main()






