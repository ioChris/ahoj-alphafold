from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from confidence_labeling import FULL_LEVEL_ORDER

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg
from common import figspec


WORK_DIR = cfg.WORK_DIR
DEFAULT_ANALYSIS_DIR = WORK_DIR / "alphafill_unified" / "run1" / "analysis_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "ridgeline_plots"

CLASS_ORDER = ["apo", "holo"]
VERBAL_ORDER = FULL_LEVEL_ORDER[:-1]
FIG_DPI = 600           # manuscript raster density; was 220
FIG_SIZE_IN = (5.0, 5.15)   # overridden per call site when composing at true size
EXACT_CANVAS = False    # save the full canvas so a composed part is its nominal size
# Where the per-group "n_apo | n_holo" row sits, in axes fractions BELOW the violin axes. The
# gutter between the violin axes and the Worst-10% row is ~0.23 of the violin axes height (row
# ratios 4.9 : 0.52 with hspace 0.42), so -0.075 put the counts hard against the violins and a
# long way from the tail bars they annotate. -0.155 sits them low in that gutter.
N_ROW_Y = -0.155
CLASS_LABELS = {"apo": "AF-Apo", "holo": "AF-Holo"}
CLASS_FILL = {"apo": "#8DB9DC", "holo": "#F4BD84"}
CLASS_EDGE = {"apo": "#2F6F9E", "holo": "#D8893A"}
INNER_DARK = "#4A4A4A"
METRIC_CHOICES = ["avg_clash", "avg_local_rmsd", "avg_global_rmsd", "avg_clash_count"]

# type from common/figspec.py, so this panel matches the rest of the manuscript once it is
# drawn at its true composed size rather than downscaled by a composer
TITLE_FS = figspec.FS_TITLE
LAB_FS = figspec.FS_TITLE
TICK_FS = figspec.FS_TICK
SMALL_FS = figspec.FS_NOTE
AX_LW = 0.8
TICK_W = 0.8
TICK_LEN = 3.5
FONT_PREFERENCE = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]
MONO_FONT = "DejaVu Sans Mono"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_PREFERENCE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "axes.linewidth": AX_LW,
        "xtick.major.width": TICK_W,
        "ytick.major.width": TICK_W,
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "axes.titlepad": 4,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AlphaFill metric ridgelines and violins by pocket class and separation bins.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--thresholds", nargs="*", default=["ST_100"])
    p.add_argument("--scopes", nargs="*", default=["same_uniprot", "any"])
    p.add_argument("--identity_tags", nargs="*", default=["ge_0p30"])
    p.add_argument("--metric", default="avg_clash", choices=METRIC_CHOICES)
    p.add_argument("--x_upper_quantile", type=float, default=0.995)
    p.add_argument("--n_sep_quantiles", type=int, default=5)
    p.add_argument("--min_points", type=int, default=15)
    p.add_argument("--density_modes", nargs="*", default=["hist", "kde"], choices=["hist", "kde"])
    p.add_argument("--hist_smooth_sigma", type=float, default=1.2)
    p.add_argument("--kde_bw_scale", type=float, default=0.75)
    p.add_argument("--n_points", type=int, default=180)
    p.add_argument(
        "--metric_cap_mode",
        choices=["quantile", "fixed", "none"],
        default="quantile",
        help="How to cap plotted metric tails. Default uses a data-driven quantile cap.",
    )
    p.add_argument(
        "--metric_cap_quantile",
        type=float,
        default=0.975,
        help="Upper quantile used when --metric_cap_mode=quantile.",
    )
    p.add_argument("--metric_cap", type=float, default=None, help="Hard upper cap used when --metric_cap_mode=fixed.")
    p.add_argument("--make_violins", action="store_true")
    p.add_argument(
        "--vertical_summary_style",
        choices=["box", "dashed"],
        default="box",
        help="Summary style inside vertical half-violins.",
    )
    p.add_argument(
        "--vertical_whiskers",
        choices=["show", "hide"],
        default="show",
        help="Whether to draw whisker extensions for the vertical box-style summary.",
    )
    p.add_argument(
        "--vertical_include_all_group",
        action="store_true",
        help="Add a synthetic All category to the vertical half-violin plots.",
    )
    p.add_argument(
        "--vertical_all_position",
        choices=["first", "last"],
        default="last",
        help="Where to place the synthetic All category in vertical half-violin plots.",
    )
    return p.parse_args()


def pretty_level(label: str) -> str:
    return label.replace("_", " ").title()


def metric_slug(metric: str) -> str:
    return str(metric).strip().lower()


def metric_label(metric: str) -> str:
    # sentence case: every other axis title in the manuscript is sentence case, and these were
    # the only Title Case ones in the set. The angstrom is a real \u00c5 -- the font stack has it.
    labels = {
        "avg_clash": "Average clash score",
        "avg_local_rmsd": "Average local RMSD (\u00c5)",
        "avg_global_rmsd": "Average global RMSD (\u00c5)",
        "avg_clash_count": "Average clash count",
    }
    return labels.get(metric, metric)


def metric_title_fragment(metric: str) -> str:
    fragments = {
        "avg_clash": "clash-score",
        "avg_local_rmsd": "local-RMSD",
        "avg_global_rmsd": "global-RMSD",
        "avg_clash_count": "clash-count",
    }
    return fragments.get(metric, metric.replace("_", "-"))


def metric_tick_step(max_value: float) -> float:
    if not np.isfinite(max_value) or max_value <= 0:
        return 0.2
    candidates = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], dtype=float)
    approx = max_value / 6.0
    return float(candidates[np.argmin(np.abs(candidates - approx))])


def normalize_class(series: pd.Series) -> pd.Series:
    vals = series.fillna("").astype(str).str.strip().str.lower()
    vals = vals.where(vals.isin(CLASS_ORDER), "other")
    return vals


def load_per_ubs_table(analysis_dir: Path, scope: str, identity_tag: str) -> pd.DataFrame:
    path = analysis_dir / f"per_ubs_summary_{scope}_{identity_tag}.csv"
    return pd.read_csv(path)


def assign_sep_quantiles(series: pd.Series, n_bins: int = 5) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    labels = [f"q{i + 1}" for i in range(n_bins)]
    try:
        binned = pd.qcut(vals, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(
            pd.Categorical([pd.NA] * len(vals), categories=labels, ordered=True),
            index=series.index,
        )
    return pd.Series(pd.Categorical(binned, categories=labels, ordered=True), index=series.index)


def display_level_order(
    level_order: list[str],
    include_all_group: bool = False,
    all_position: str = "last",
) -> list[str]:
    levels = list(level_order)
    if not include_all_group:
        return levels
    if all_position == "first":
        return ["all"] + levels
    return levels + ["all"]


def _gauss_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gauss_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values
    kernel = _gauss_kernel_1d(sigma)
    return np.convolve(values, kernel, mode="same")


def density_curve(
    values: np.ndarray,
    x_min: float,
    x_max: float,
    n_points: int,
    *,
    mode: str = "hist",
    smooth_sigma: float = 0.8,
    bw_scale: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    x_grid = np.linspace(float(x_min), float(x_max), int(n_points))
    if v.size == 0:
        return x_grid, np.zeros_like(x_grid)

    mode = (mode or "hist").lower()
    if mode == "kde":
        if v.size <= 1:
            bw = 0.05
            dens = np.exp(-0.5 * ((x_grid - float(v[0])) / bw) ** 2)
            peak = float(np.nanmax(dens)) if dens.size else 0.0
            return x_grid, dens / (peak if peak > 0 else 1.0)

        std = float(np.nanstd(v, ddof=1))
        iqr = float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25))
        sigma = min(std, iqr / 1.349) if np.isfinite(iqr) and iqr > 0 else std
        if (not np.isfinite(sigma)) or sigma <= 0:
            sigma = max(1e-3, float(np.nanstd(v)))
        bw = max(0.9 * sigma * (v.size ** (-1 / 5)) * float(bw_scale), np.finfo(float).eps)

        kde = gaussian_kde(v)
        kde.set_bandwidth(kde.factor * float(bw_scale))
        dens = kde.evaluate(x_grid)
        peak = float(np.nanmax(dens)) if dens.size else 0.0
        return x_grid, dens / (peak if peak > 0 else 1.0)

    hist, edges = np.histogram(
        v,
        bins=max(int(n_points) - 1, 20),
        range=(float(x_min), float(x_max)),
        density=True,
    )
    hist = _gauss_smooth_1d(hist, float(smooth_sigma))
    peak = float(np.nanmax(hist)) if hist.size else 0.0
    dens = hist / (peak if peak > 0 else 1.0)
    xc = 0.5 * (edges[:-1] + edges[1:])
    return xc, dens


def make_row_specs(levels: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for pocket_class in CLASS_ORDER:
        for level in levels:
            specs.append((pocket_class, level))
    return specs


def make_row_positions(n_levels: int, intra_group_spacing: float, inter_group_gap: float) -> list[float]:
    positions: list[float] = []
    start = (n_levels - 1) * intra_group_spacing + inter_group_gap + (n_levels - 1) * intra_group_spacing
    current = start
    for cls_idx, _cls in enumerate(CLASS_ORDER):
        for _ in range(n_levels):
            positions.append(current)
            current -= intra_group_spacing
        if cls_idx != len(CLASS_ORDER) - 1:
            current -= inter_group_gap
    return positions


def half_violin_span_at_y(body, y_value: float, x_center: float, side: str) -> tuple[float, float]:
    verts = body.get_paths()[0].vertices
    xs: list[float] = []
    for i in range(len(verts) - 1):
        x1, y1 = verts[i]
        x2, y2 = verts[i + 1]
        if y1 == y2:
            if np.isclose(y_value, y1):
                xs.extend([x1, x2])
            continue
        if (y1 <= y_value <= y2) or (y2 <= y_value <= y1):
            t = (y_value - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return x_center, x_center
    xs_arr = np.asarray(xs, dtype=float)
    if side == "apo":
        inner = float(np.max(xs_arr))
        outer = float(np.min(xs_arr))
    else:
        inner = float(np.min(xs_arr))
        outer = float(np.max(xs_arr))
    return outer, inner


def collect_series_by_row(
    df: pd.DataFrame,
    level_col: str,
    level_order: list[str],
    metric: str,
    include_all_group: bool = False,
    all_group_position: str = "last",
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    display_levels = display_level_order(
        level_order,
        include_all_group=include_all_group,
        all_position=all_group_position,
    )
    row_specs = make_row_specs(display_levels)
    series_list: list[np.ndarray] = []
    for pocket_class, level in row_specs:
        if level == "all":
            sub = df.loc[df["_class"] == pocket_class, metric].dropna().to_numpy()
        else:
            sub = df.loc[(df["_class"] == pocket_class) & (df[level_col] == level), metric].dropna().to_numpy()
        series_list.append(sub)
    return row_specs, series_list


def resolve_metric_cap(
    df: pd.DataFrame,
    metric: str,
    metric_cap_mode: str = "quantile",
    metric_cap: float | None = None,
    metric_cap_quantile: float = 0.975,
) -> float | None:
    if metric_cap_mode == "none":
        return None
    if metric_cap_mode == "fixed":
        if metric_cap is None or not np.isfinite(metric_cap):
            return None
        return float(metric_cap)
    vals = pd.to_numeric(df[metric], errors="coerce").dropna()
    if vals.empty:
        return None
    q = float(metric_cap_quantile)
    if not np.isfinite(q):
        return None
    q = min(max(q, 0.0), 1.0)
    return float(vals.quantile(q))


def apply_metric_cap(
    df: pd.DataFrame,
    metric: str,
    metric_cap_mode: str = "quantile",
    metric_cap: float | None = None,
    metric_cap_quantile: float = 0.975,
) -> pd.DataFrame:
    resolved_cap = resolve_metric_cap(
        df,
        metric,
        metric_cap_mode=metric_cap_mode,
        metric_cap=metric_cap,
        metric_cap_quantile=metric_cap_quantile,
    )
    if resolved_cap is None or not np.isfinite(resolved_cap):
        return df
    out = df.copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out.loc[out[metric].notna() & (out[metric] <= float(resolved_cap))].copy()


def plot_ridgelines(
    df: pd.DataFrame,
    level_col: str,
    level_order: list[str],
    outpath: Path,
    title: str,
    metric: str = "avg_clash",
    x_upper_quantile: float = 0.995,
    min_points: int = 15,
    density_mode: str = "hist",
    hist_smooth_sigma: float = 0.8,
    kde_bw_scale: float = 0.75,
    n_points: int = 180,
    metric_cap_mode: str = "quantile",
    metric_cap_quantile: float = 0.975,
    metric_cap: float | None = None,
    include_all_group: bool = False,
    all_group_position: str = "last",
) -> None:
    d = df.copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d = d.loc[d["matched"].fillna(False)].copy()
    d = d.loc[d["_class"].isin(CLASS_ORDER)].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.loc[d[metric].notna()].copy()
    d = apply_metric_cap(
        d,
        metric,
        metric_cap_mode=metric_cap_mode,
        metric_cap=metric_cap,
        metric_cap_quantile=metric_cap_quantile,
    )
    d[level_col] = d[level_col].astype(str).str.strip().str.lower()
    d = d.loc[d[level_col].isin(level_order)].copy()
    if d.empty:
        return

    display_levels = display_level_order(
        level_order,
        include_all_group=include_all_group,
        all_position=all_group_position,
    )
    upper = float(d[metric].quantile(x_upper_quantile))
    upper = max(upper, float(d[metric].max())) * 1.04

    row_specs = make_row_specs(display_levels)
    row_positions = make_row_positions(
        n_levels=len(display_levels),
        intra_group_spacing=0.55,
        inter_group_gap=1.05,
    )

    fig, ax = plt.subplots(figsize=(4.8, 5.9), constrained_layout=True)
    ytick_positions: list[float] = []
    ytick_labels: list[str] = []
    max_height = 0.88

    for idx, (pos, (pocket_class, level)) in enumerate(zip(row_positions, row_specs)):
        if level == "all":
            sub = d.loc[d["_class"] == pocket_class, metric].dropna().to_numpy()
        else:
            sub = d.loc[(d["_class"] == pocket_class) & (d[level_col] == level), metric].dropna().to_numpy()
        ax.hlines(pos, 0.0, upper, color="#e2e2e2", lw=0.8, zorder=1)
        if sub.size < min_points:
            ytick_positions.append(pos)
            ytick_labels.append(f"{'All' if level == 'all' else pretty_level(level)}\n$n$={sub.size}")
            continue

        x_vals, curve = density_curve(
            sub,
            0.0,
            upper,
            n_points,
            mode=density_mode,
            smooth_sigma=hist_smooth_sigma,
            bw_scale=kde_bw_scale,
        )
        ridge = pos + curve * max_height
        z = 10.0 + idx * 0.5

        ax.fill_between(x_vals, pos, ridge, color=CLASS_FILL[pocket_class], alpha=1.0, lw=0.0, zorder=z)
        ax.plot(x_vals, ridge, color=CLASS_EDGE[pocket_class], lw=1.1, zorder=z + 0.1)

        median_val = float(np.median(sub))
        if np.isfinite(median_val) and 0.0 <= median_val <= upper:
            median_height = np.interp(median_val, x_vals, ridge)
            ax.vlines(
                median_val,
                pos,
                median_height,
                color=CLASS_EDGE[pocket_class],
                lw=0.95,
                ls="--",
                alpha=0.9,
                zorder=z + 0.2,
            )

        ytick_positions.append(pos)
        ytick_labels.append(f"{'All' if level == 'all' else pretty_level(level)}\n$n$={sub.size}")

    ax.set_xlim(0.0, upper)
    ax.set_ylim(-0.45, row_positions[0] + max_height + 0.10)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=TICK_FS, linespacing=0.88)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", pad=1.8)
    ax.set_xlabel(metric_label(metric), fontsize=LAB_FS)
    ax.set_ylabel("Separation bins", fontsize=LAB_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#ebebeb", lw=0.7, linestyle="--")
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([], [], lw=6, color=CLASS_FILL["apo"], alpha=1.0, label=CLASS_LABELS["apo"]),
        Line2D([], [], lw=6, color=CLASS_FILL["holo"], alpha=1.0, label=CLASS_LABELS["holo"]),
    ]
    leg = ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.86, 0.50),
        frameon=True,
        fontsize=SMALL_FS,
        handlelength=2.0,
        ncol=1,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_full_violins(
    df: pd.DataFrame,
    level_col: str,
    level_order: list[str],
    outpath: Path,
    title: str,
    metric: str = "avg_clash",
    x_upper_quantile: float = 0.995,
    min_points: int = 15,
    overlay_style: str = "trend",
    metric_cap_mode: str = "quantile",
    metric_cap_quantile: float = 0.975,
    metric_cap: float | None = None,
    include_all_group: bool = False,
    all_group_position: str = "last",
) -> None:
    d = df.copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d = d.loc[d["matched"].fillna(False)].copy()
    d = d.loc[d["_class"].isin(CLASS_ORDER)].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.loc[d[metric].notna()].copy()
    d = apply_metric_cap(
        d,
        metric,
        metric_cap_mode=metric_cap_mode,
        metric_cap=metric_cap,
        metric_cap_quantile=metric_cap_quantile,
    )
    d[level_col] = d[level_col].astype(str).str.strip().str.lower()
    d = d.loc[d[level_col].isin(level_order)].copy()
    if d.empty:
        return

    display_levels = display_level_order(
        level_order,
        include_all_group=include_all_group,
        all_position=all_group_position,
    )
    upper = float(d[metric].quantile(x_upper_quantile))
    upper = max(upper, float(d[metric].median()))

    row_positions = make_row_positions(
        n_levels=len(display_levels),
        intra_group_spacing=1.00,
        inter_group_gap=0.85,
    )
    row_specs, series_list = collect_series_by_row(
        d,
        level_col,
        level_order,
        metric,
        include_all_group=include_all_group,
        all_group_position=all_group_position,
    )

    fig, ax = plt.subplots(figsize=(4.8, 6.4), constrained_layout=True)
    ytick_positions: list[float] = []
    ytick_labels: list[str] = []
    medians_by_class: dict[str, list[tuple[float, float]]] = {cls: [] for cls in CLASS_ORDER}
    means_by_class: dict[str, list[tuple[float, float]]] = {cls: [] for cls in CLASS_ORDER}
    violin_width = 0.72
    half_violin_height = violin_width / 2.0
    y_clearance = 0.42

    for pos, (pocket_class, level), vals in zip(row_positions, row_specs, series_list):
        ax.hlines(pos, 0.0, upper, color="#e5e5e5", lw=0.8, zorder=1)
        n_vals = int(vals.size)
        ytick_positions.append(pos)
        ytick_labels.append(f"{'All' if level == 'all' else pretty_level(level)}\n$n$={n_vals}")
        if n_vals < min_points:
            continue

        parts = ax.violinplot(
            [vals],
            positions=[pos],
            vert=False,
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(CLASS_FILL[pocket_class])
            body.set_edgecolor(CLASS_EDGE[pocket_class])
            body.set_alpha(1.0)
            body.set_linewidth(1.0)

        median_val = float(np.median(vals))
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        mean_val = float(np.mean(vals))

        if overlay_style in {"iqr", "box"}:
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            low_whisk = float(np.min(vals[vals >= lower_fence])) if np.any(vals >= lower_fence) else float(np.min(vals))
            high_whisk = float(np.max(vals[vals <= upper_fence])) if np.any(vals <= upper_fence) else float(np.max(vals))
            ax.plot([low_whisk, high_whisk], [pos, pos], color=INNER_DARK, lw=1.0, zorder=5)
            ax.plot([q1, q3], [pos, pos], color=INNER_DARK, lw=3.0, solid_capstyle="round", zorder=6)
            ax.plot([median_val, median_val], [pos - 0.11, pos + 0.11], color="white", lw=1.4, zorder=7)
        else:
            medians_by_class[pocket_class].append((median_val, pos))
            means_by_class[pocket_class].append((mean_val, pos))
            ax.plot(median_val, pos, marker="o", ms=3.8, color=CLASS_EDGE[pocket_class], mec=CLASS_EDGE[pocket_class], zorder=5)
            ax.plot(mean_val, pos, marker="s", ms=3.7, mfc="white", mec=CLASS_EDGE[pocket_class], mew=0.9, zorder=5)

    if overlay_style != "iqr":
        for pocket_class in CLASS_ORDER:
            if medians_by_class[pocket_class]:
                xs, ys = zip(*medians_by_class[pocket_class])
                ax.plot(xs, ys, color=CLASS_EDGE[pocket_class], lw=1.1, ls="-", zorder=4)
            if means_by_class[pocket_class]:
                xs, ys = zip(*means_by_class[pocket_class])
                ax.plot(xs, ys, color=CLASS_EDGE[pocket_class], lw=0.9, ls=":", zorder=4)

    ax.set_xlim(0.0, upper)
    y_min_lim = min(row_positions) - half_violin_height - y_clearance
    y_max_lim = max(row_positions) + half_violin_height + y_clearance
    ax.set_ylim(y_min_lim, y_max_lim)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=TICK_FS, linespacing=1.18)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", pad=1.8)
    ax.set_xlabel(metric_label(metric), fontsize=LAB_FS)
    ax.set_ylabel("Separation bins", fontsize=LAB_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#ebebeb", lw=0.7, linestyle="--")
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(ax.spines["left"].get_edgecolor())
    ax.spines["bottom"].set_linewidth(0.8)

    handles = [
        Line2D([], [], lw=6, color=CLASS_FILL["apo"], label=CLASS_LABELS["apo"]),
        Line2D([], [], lw=6, color=CLASS_FILL["holo"], label=CLASS_LABELS["holo"]),
    ]
    if overlay_style in {"iqr", "box"}:
        handles.extend(
            [
                Line2D([], [], color=INNER_DARK, lw=1.0, label="Whiskers"),
                Line2D([], [], color=INNER_DARK, lw=3.0, label="IQR"),
                Line2D([], [], color="white", lw=1.4, label="Median notch"),
            ]
        )
    else:
        handles.extend(
            [
                Line2D([], [], marker="o", ms=4, color="0.25", lw=1.0, label="Median"),
                Line2D([], [], marker="s", ms=4, mfc="white", mec="0.25", mew=0.9, color="0.25", lw=0.9, ls=":", label="Mean"),
            ]
        )
    leg = ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.86, 0.50),
        frameon=True,
        fontsize=SMALL_FS,
        handlelength=2.0,
        ncol=2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_half_violins_vertical(
    df: pd.DataFrame,
    level_col: str,
    level_order: list[str],
    outpath: Path,
    title: str,
    metric: str = "avg_clash",
    x_upper_quantile: float = 0.995,
    min_points: int = 15,
    summary_style: str = "box",
    show_whiskers: bool = True,
    show_counts: bool = True,
    show_legend: bool = True,
    letter: str | None = None,
    metric_cap_mode: str = "quantile",
    metric_cap_quantile: float = 0.975,
    metric_cap: float | None = None,
    include_all_group: bool = False,
    all_group_position: str = "last",
    counts_stacked: bool = False,
) -> None:
    d = df.copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d = d.loc[d["matched"].fillna(False)].copy()
    d = d.loc[d["_class"].isin(CLASS_ORDER)].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.loc[d[metric].notna()].copy()
    d[level_col] = d[level_col].astype(str).str.strip().str.lower()
    d = d.loc[d[level_col].isin(level_order)].copy()
    if d.empty:
        return

    d_plot = apply_metric_cap(
        d.copy(),
        metric,
        metric_cap_mode=metric_cap_mode,
        metric_cap=metric_cap,
        metric_cap_quantile=metric_cap_quantile,
    )
    if d_plot.empty:
        return

    upper_q = float(d_plot[metric].quantile(x_upper_quantile))
    upper_data = float(d_plot[metric].max())
    upper = max(upper_q, upper_data) * 1.08
    display_levels = display_level_order(
        level_order,
        include_all_group=include_all_group,
        all_position=all_group_position,
    )

    d["_level"] = pd.Categorical(d[level_col], categories=level_order, ordered=True)
    d["_level_label"] = d["_level"].astype(str).map(lambda s: pretty_level(s))
    d["_class_label"] = d["_class"].map(CLASS_LABELS)
    d_plot["_level"] = pd.Categorical(d_plot[level_col], categories=level_order, ordered=True)
    d_plot["_level_label"] = d_plot["_level"].astype(str).map(lambda s: pretty_level(s))
    d_plot["_class_label"] = d_plot["_class"].map(CLASS_LABELS)

    counts = d.groupby(["_level", "_class"], observed=True)[metric].size().rename("n").reset_index()
    count_map = {(str(r["_level"]), str(r["_class"])): int(r["n"]) for _, r in counts.iterrows()}
    if include_all_group:
        for cls in CLASS_ORDER:
            count_map[("all", cls)] = int(d.loc[d["_class"] == cls, metric].notna().sum())

    # constrained_layout reflows to whatever is actually drawn, so a panel with show_counts=False
    # would grow its axes into the space the counts row leaves -- which is precisely why the pair
    # of AlphaFill panels came out with different y=0 baselines. At a fixed canvas the margins are
    # set explicitly instead, so every panel of a pair has an identical axes box.
    fig, (ax, ax_tail) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FIG_SIZE_IN,
        sharex=True,
        gridspec_kw={"height_ratios": [4.9, 0.52],
                     "hspace": (0.42 if EXACT_CANVAS else 0.06)},
        constrained_layout=not EXACT_CANVAS,
    )
    if EXACT_CANVAS:
        w_in, h_in = FIG_SIZE_IN
        fig.subplots_adjust(**figspec.margins(w_in, h_in, left=13.0, right=2.0,
                                              top=3.0, bottom=15.0))
    half_width = 0.37 if summary_style == "dashed" else 0.31
    half_gap = 0.040 if summary_style == "dashed" else 0.000
    max_digits = max(len(str(v)) for v in count_map.values()) if count_map else 1
    worst10_threshold = float(d[metric].quantile(0.90))
    for i, level in enumerate(display_levels):
        for cls in CLASS_ORDER:
            if level == "all":
                vals = d_plot.loc[d_plot["_class"] == cls, metric].dropna().to_numpy()
            else:
                vals = d_plot.loc[(d_plot["_level"] == level) & (d_plot["_class"] == cls), metric].dropna().to_numpy()
            if vals.size < min_points:
                continue
            x_center = i
            if summary_style == "dashed":
                y0_cls = max(0.0, float(np.min(vals)) - 0.02 * upper)
                y1_cls = min(upper, float(np.max(vals)) + 0.02 * upper)
                if y1_cls <= y0_cls:
                    continue
                if np.allclose(vals, vals[0]):
                    continue

                y_probe = np.linspace(y0_cls, y1_cls, 320)
                kde = gaussian_kde(vals)
                dens = kde.evaluate(y_probe)
                peak = float(np.nanmax(dens)) if dens.size else 0.0
                if peak <= 0:
                    continue
                dens_full = dens / peak * half_width

                center_x = float(i)
                split_edge_x = center_x - (half_gap / 2.0) if cls == "apo" else center_x + (half_gap / 2.0)
                visible_mask = np.asarray(dens_full > (half_gap / 2.0 + 1e-4), dtype=bool)
                if not np.any(visible_mask):
                    continue

                visible_idx = np.flatnonzero(visible_mask)
                split_idx = np.where(np.diff(visible_idx) > 1)[0] + 1
                blocks = [block for block in np.split(visible_idx, split_idx) if block.size >= 2]
                if not blocks:
                    continue
                main_block = max(blocks, key=lambda b: (b.size, y_probe[b[-1]] - y_probe[b[0]]))
                y_seg = y_probe[main_block]
                dens_seg = dens_full[main_block]
                if cls == "apo":
                    outer_x = center_x - dens_seg
                else:
                    outer_x = center_x + dens_seg
                inner_x = np.full_like(y_seg, split_edge_x)
                ax.fill_betweenx(
                    y_seg,
                    outer_x,
                    inner_x,
                    color=CLASS_FILL[cls],
                    alpha=1.0,
                    linewidth=0.0,
                    zorder=6,
                )
                ax.plot(outer_x, y_seg, color=INNER_DARK, lw=1.0, zorder=10)
                ax.plot(inner_x, y_seg, color=INNER_DARK, lw=1.0, zorder=10)
                ax.plot(
                    [float(outer_x[0]), float(inner_x[0])],
                    [float(max(y_seg[0], 0.0) + (0.002 if y_seg[0] <= 0.002 else 0.0)), float(max(y_seg[0], 0.0) + (0.002 if y_seg[0] <= 0.002 else 0.0))],
                    color=INNER_DARK,
                    lw=1.0,
                    zorder=12,
                    solid_capstyle="butt",
                    clip_on=False,
                )
                ax.plot(
                    [float(outer_x[-1]), float(inner_x[-1])],
                    [float(y_seg[-1]), float(y_seg[-1])],
                    color=INNER_DARK,
                    lw=1.0,
                    zorder=12,
                    solid_capstyle="butt",
                    clip_on=False,
                )

                q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
                pad = 0.008
                q_dash = (0, (1.1, 1.6))
                med_dash = (0, (4.0, 1.8))
                for y_stat, linestyle in ((q1, q_dash), (q3, q_dash), (med, med_dash)):
                    span_full = float(np.interp(y_stat, y_probe, dens_full, left=0.0, right=0.0))
                    visible_span = span_full - (half_gap / 2.0)
                    if visible_span <= pad * 2.0:
                        continue
                    if cls == "apo":
                        x_left = split_edge_x - visible_span + pad
                        x_right = split_edge_x - pad
                    else:
                        x_left = split_edge_x + pad
                        x_right = split_edge_x + visible_span - pad
                    if x_right <= x_left:
                        continue
                    ax.plot(
                        [x_left, x_right],
                        [y_stat, y_stat],
                        color=INNER_DARK,
                        lw=1.0,
                        ls=linestyle,
                        zorder=7,
                    )
                continue

            parts = ax.violinplot(
                [vals],
                positions=[x_center],
                vert=True,
                widths=half_width * 2.0,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(CLASS_FILL[cls])
                body.set_edgecolor(INNER_DARK)
                body.set_alpha(1.0)
                body.set_linewidth(1.0)
                if cls == "apo":
                    clip_rect = Rectangle(
                        (i - 1.0, -0.2),
                        1.0 - (half_gap / 2.0),
                        upper + 0.4,
                        transform=ax.transData,
                    )
                else:
                    clip_rect = Rectangle(
                        (i + (half_gap / 2.0), -0.2),
                        1.0 - (half_gap / 2.0),
                        upper + 0.4,
                        transform=ax.transData,
                    )
                body.set_clip_path(clip_rect)

            q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
            iqr = q3 - q1
            low_fence = q1 - 1.5 * iqr
            high_fence = q3 + 1.5 * iqr
            low_whisk = float(np.min(vals[vals >= low_fence])) if np.any(vals >= low_fence) else float(np.min(vals))
            high_whisk = float(np.max(vals[vals <= high_fence])) if np.any(vals <= high_fence) else float(np.max(vals))
            body = parts["bodies"][0]
            outer_q1, inner_q1 = half_violin_span_at_y(body, float(q1), x_center, cls)
            outer_q3, inner_q3 = half_violin_span_at_y(body, float(q3), x_center, cls)
            outer_med, inner_med = half_violin_span_at_y(body, float(med), x_center, cls)
            outer_low, inner_low = half_violin_span_at_y(body, float(low_whisk), x_center, cls)
            outer_high, inner_high = half_violin_span_at_y(body, float(high_whisk), x_center, cls)

            inner_pad = 0.006
            half_span_q = abs(outer_q1 - inner_q1)
            half_span_med = abs(outer_med - inner_med)
            box_width = max(0.026, min(half_span_med * 0.42, 0.070))
            whisk_width = max(0.008, min(box_width * 0.34, half_span_q * 0.20, 0.016))
            if cls == "apo":
                box_right = i
                box_left = box_right - box_width
                whisk_left = i - whisk_width
                whisk_right = i
            else:
                box_left = i
                box_right = box_left + box_width
                whisk_left = i
                whisk_right = i + whisk_width
            med_width = max(0.020, min(box_width * 0.82, half_span_med - 2.0 * inner_pad))
            if cls == "apo":
                med_right = i
                med_left = med_right - med_width
            else:
                med_left = i
                med_right = med_left + med_width

            ax.add_patch(
                Rectangle(
                    (min(box_left, box_right), q1),
                    abs(box_right - box_left),
                    q3 - q1,
                    facecolor=INNER_DARK,
                    edgecolor=INNER_DARK,
                    linewidth=0.0,
                    zorder=8,
                )
            )
            if show_whiskers:
                ax.add_patch(
                    Rectangle(
                        (whisk_left, low_whisk),
                        whisk_right - whisk_left,
                        max(q1 - low_whisk, 0.0),
                        facecolor=INNER_DARK,
                        edgecolor=INNER_DARK,
                        linewidth=0.0,
                        zorder=8,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (whisk_left, q3),
                        whisk_right - whisk_left,
                        max(high_whisk - q3, 0.0),
                        facecolor=INNER_DARK,
                        edgecolor=INNER_DARK,
                        linewidth=0.0,
                        zorder=8,
                    )
                )
            ax.plot([med_left, med_right], [med, med], color="white", lw=1.3, solid_capstyle="butt", zorder=9)


    # lowercase, and "overall" rather than "Overall": the same five separation tiers are named
    # in lowercase in Figure 4's panels c and d and in Figure 3's binned rows, and the summary
    # group is bold-lowercase "overall" there too. pretty_level() is left alone -- the horizontal
    # ridgeline variants above use it for their y axis and are not part of this figure.
    xticklabels = ["overall" if level == "all" else pretty_level(level).lower()
                   for level in display_levels]
    x_positions = np.arange(len(display_levels))
    ax.set_xticks(x_positions)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    step = metric_tick_step(upper_data)
    top_tick = max(step, float(np.floor(upper_data / step) * step))
    yticks = np.arange(0.0, top_tick + 0.001, step)
    if yticks.size == 0:
        yticks = np.array([0.0])
    ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel(metric_label(metric), fontsize=LAB_FS)
    ax.set_ylim(0.0, upper)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_bounds(0.0, top_tick)

    bar_width = 0.18
    bar_offset = 0.11
    count_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    tail_fracs: list[float] = []
    if show_counts:
        ax.text(
            -0.010,
            N_ROW_Y,
            r"$n$",
            ha="right",
            va="bottom",
            fontsize=TICK_FS,
            color="0.30",
            transform=ax.transAxes,
            clip_on=False,
        )
    for i, level in enumerate(display_levels):
        if level == "all":
            n_apo = int(d.loc[d["_class"] == "apo", metric].notna().sum())
            n_holo = int(d.loc[d["_class"] == "holo", metric].notna().sum())
        else:
            n_apo = count_map.get((level, "apo"), 0)
            n_holo = count_map.get((level, "holo"), 0)
        if show_counts and counts_stacked:
            # apo above holo, each in its class edge colour, centred on the tick. The side-by-side
            # row needs ~11 mm per group once counts reach five digits and the docking panels
            # (n up to 38,206 per class) collided at the 88 mm canvas; two lines fit.
            for dy, n_val, cls in ((0.055, n_apo, "apo"), (-0.055, n_holo, "holo")):
                ax.text(
                    i,
                    N_ROW_Y + dy,
                    f"{n_val}",
                    ha="center",
                    va="bottom",
                    fontsize=SMALL_FS,
                    color=CLASS_EDGE[cls],
                    fontfamily=MONO_FONT,
                    clip_on=False,
                    transform=count_transform,
                )
        elif show_counts:
            count_offset = 0.055
            ax.text(
                i - count_offset,
                N_ROW_Y,
                f"{n_apo:>{max_digits}}",
                ha="right",
                va="bottom",
                fontsize=SMALL_FS,
                color="0.30",
                fontfamily=MONO_FONT,
                clip_on=False,
                transform=count_transform,
            )
            ax.text(
                i,
                N_ROW_Y,
                "|",
                ha="center",
                va="bottom",
                fontsize=SMALL_FS,
                color="0.30",
                fontfamily=MONO_FONT,
                clip_on=False,
                transform=count_transform,
            )
            ax.text(
                i + count_offset,
                N_ROW_Y,
                f"{n_holo:<{max_digits}}",
                ha="left",
                va="bottom",
                fontsize=SMALL_FS,
                color="0.30",
                fontfamily=MONO_FONT,
                clip_on=False,
                transform=count_transform,
            )
        if level == "all":
            vals_apo = d.loc[d["_class"] == "apo", metric].dropna().to_numpy(dtype=float)
            vals_holo = d.loc[d["_class"] == "holo", metric].dropna().to_numpy(dtype=float)
        else:
            vals_apo = d.loc[(d["_level"] == level) & (d["_class"] == "apo"), metric].dropna().to_numpy(dtype=float)
            vals_holo = d.loc[(d["_level"] == level) & (d["_class"] == "holo"), metric].dropna().to_numpy(dtype=float)
        frac_apo = float(np.mean(vals_apo >= worst10_threshold)) if vals_apo.size else np.nan
        frac_holo = float(np.mean(vals_holo >= worst10_threshold)) if vals_holo.size else np.nan
        if np.isfinite(frac_apo):
            tail_fracs.append(frac_apo)
        if np.isfinite(frac_holo):
            tail_fracs.append(frac_holo)
        if np.isfinite(frac_apo):
            ax_tail.bar(
                i - bar_offset,
                frac_apo,
                width=bar_width,
                color=CLASS_FILL["apo"],
                edgecolor=INNER_DARK,
                linewidth=0.9,
                zorder=3,
            )
        if np.isfinite(frac_holo):
            ax_tail.bar(
                i + bar_offset,
                frac_holo,
                width=bar_width,
                color=CLASS_FILL["holo"],
                edgecolor=INNER_DARK,
                linewidth=0.9,
                zorder=3,
            )

    tail_max = max(tail_fracs) if tail_fracs else 0.10
    tail_ylim = max(0.12, tail_max * 1.12)
    tail_ylim = float(np.ceil(tail_ylim / 0.1) * 0.1)
    tail_ymin = -0.02 * tail_ylim
    ax_tail.set_ylim(tail_ymin, tail_ylim)
    midpoint = tail_ylim / 2.0
    ax_tail.set_yticks([0.0, tail_ylim])
    ax_tail.set_yticklabels([f"{0.0:.1f}", f"{tail_ylim:.1f}"])
    ax_tail.axhline(0.0, color="#ebebeb", lw=0.6, linestyle="--", zorder=0)
    ax_tail.axhline(midpoint, color="#ebebeb", lw=0.6, linestyle="--", zorder=0)
    ax_tail.axhline(tail_ylim, color="#ebebeb", lw=0.6, linestyle="--", zorder=0)
    ax_tail.tick_params(axis="y", labelsize=TICK_FS)
    ax_tail.set_ylabel("Worst 10%", fontsize=LAB_FS, labelpad=8)
    ax_tail.set_xticks(x_positions)
    ax_tail.set_xticklabels(xticklabels, fontsize=LAB_FS, linespacing=1.12)
    ax_tail.tick_params(axis="x", labelsize=LAB_FS, color="none", labelcolor="0.15", pad=11, length=0)
    # the summary group is set bold and fenced off by the divider below, as in Figure 3
    if "all" in display_levels:
        ax_tail.get_xticklabels()[display_levels.index("all")].set_fontweight("bold")
    # "Apo-holo separation", not "Separation bins": panels c and d of Figure 4 already use the
    # former, so the figure was naming one variable two ways across its four panels. The x axis
    # being binned is evident from the categorical ticks; what it is a binning OF is not.
    ax_tail.set_xlabel("Apo-holo separation", fontsize=LAB_FS, labelpad=10)
    ax_tail.grid(False, axis="both")
    ax_tail.spines["top"].set_visible(False)
    ax_tail.spines["right"].set_visible(False)
    ax_tail.spines["bottom"].set_visible(False)
    ax_tail.spines["left"].set_bounds(0.0, tail_ylim)

    if show_legend:
        handles = [
            Patch(facecolor=CLASS_FILL["apo"], edgecolor=INNER_DARK, label=CLASS_LABELS["apo"]),
            Patch(facecolor=CLASS_FILL["holo"], edgecolor=INNER_DARK, label=CLASS_LABELS["holo"]),
        ]
        leg = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fontsize=SMALL_FS,
            ncol=2,
            ncols=2,
            handlelength=1.2,
            handletextpad=0.7,
            columnspacing=0.9,
            borderpad=0.65,
            borderaxespad=0.25,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.85")

    if "all" in display_levels and len(display_levels) > 1:
        # divider between the binned groups and the summary group, matching the detection figure
        i_all = display_levels.index("all")
        ax.axvline(i_all - 0.5 if i_all else 0.5, color="0.78", lw=0.8, ls=":", zorder=0)
        ax_tail.axvline(i_all - 0.5 if i_all else 0.5, color="0.78", lw=0.8, ls=":", zorder=0)

    if letter:
        fig.text(0.5 / (FIG_SIZE_IN[0] / figspec.MM), 1 - 0.5 / (FIG_SIZE_IN[1] / figspec.MM),
                 letter, ha="left", va="top", fontsize=figspec.FS_LETTER, fontweight="bold")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches=(None if EXACT_CANVAS else "tight"))
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def build_title(threshold: str, scope: str, identity_tag: str, mode_label: str, density_mode: str, metric: str) -> str:
    scope_label = "same-UniProt" if scope == "same_uniprot" else "any matched UniProt"
    identity_label = identity_tag.replace("ge_", ">=").replace("p", ".")
    density_label = "histogram-smoothed" if density_mode == "hist" else "Gaussian KDE"
    return (
        f"{threshold} AlphaFill {metric_title_fragment(metric)} ridgelines\n"
        f"Scope: {scope_label} | identity {identity_label} | separation: {mode_label} | density: {density_label}"
    )


def build_violin_title(threshold: str, scope: str, identity_tag: str, mode_label: str, metric: str) -> str:
    scope_label = "same-UniProt" if scope == "same_uniprot" else "any matched UniProt"
    identity_label = identity_tag.replace("ge_", ">=").replace("p", ".")
    return (
        f"{threshold} AlphaFill {metric_title_fragment(metric)} violins\n"
        f"Scope: {scope_label} | identity {identity_label} | separation: {mode_label}"
    )


def main() -> None:
    args = parse_args()
    metric_key = metric_slug(args.metric)

    for scope in args.scopes:
        for identity_tag in args.identity_tags:
            per_ubs = load_per_ubs_table(args.analysis_dir, scope, identity_tag)
            per_ubs["threshold"] = per_ubs["threshold"].astype(str).str.strip()
            per_ubs["_class"] = normalize_class(per_ubs["pocket_class"])
            per_ubs = per_ubs.loc[per_ubs["_class"].isin(CLASS_ORDER)].copy()
            per_ubs["sep_quantile_5"] = assign_sep_quantiles(
                per_ubs["pocket_separation_pair"],
                n_bins=args.n_sep_quantiles,
            )

            for threshold in args.thresholds:
                sub = per_ubs.loc[per_ubs["threshold"] == threshold].copy()
                if sub.empty:
                    continue

                for density_mode in args.density_modes:
                    verbal_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_ridgeline_verbal_sep_{density_mode}.png"
                    )
                    plot_ridgelines(
                        sub,
                        level_col="separation_level_db",
                        level_order=VERBAL_ORDER,
                        outpath=verbal_out,
                        title=build_title(threshold, scope, identity_tag, "5 verbal bins", density_mode, args.metric),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        density_mode=density_mode,
                        hist_smooth_sigma=args.hist_smooth_sigma,
                        kde_bw_scale=args.kde_bw_scale,
                        n_points=args.n_points,
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                    quant_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_ridgeline_sep_quantiles5_{density_mode}.png"
                    )
                    plot_ridgelines(
                        sub,
                        level_col="sep_quantile_5",
                        level_order=[f"q{i}" for i in range(1, args.n_sep_quantiles + 1)],
                        outpath=quant_out,
                        title=build_title(
                            threshold,
                            scope,
                            identity_tag,
                            f"{args.n_sep_quantiles} separation quantiles",
                            density_mode,
                            args.metric,
                        ),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        density_mode=density_mode,
                        hist_smooth_sigma=args.hist_smooth_sigma,
                        kde_bw_scale=args.kde_bw_scale,
                        n_points=args.n_points,
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                if args.make_violins:
                    verbal_violin_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_violin_verbal_sep.png"
                    )
                    plot_full_violins(
                        sub,
                        level_col="separation_level_db",
                        level_order=VERBAL_ORDER,
                        outpath=verbal_violin_out,
                        title=build_violin_title(threshold, scope, identity_tag, "5 verbal bins", args.metric),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        overlay_style="trend",
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                    quant_violin_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_violin_sep_quantiles5.png"
                    )
                    plot_full_violins(
                        sub,
                        level_col="sep_quantile_5",
                        level_order=[f"q{i}" for i in range(1, args.n_sep_quantiles + 1)],
                        outpath=quant_violin_out,
                        title=build_violin_title(
                            threshold,
                            scope,
                            identity_tag,
                            f"{args.n_sep_quantiles} separation quantiles",
                            args.metric,
                        ),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        overlay_style="trend",
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                    verbal_violin_iqr_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_violin_verbal_sep_iqr.png"
                    )
                    plot_full_violins(
                        sub,
                        level_col="separation_level_db",
                        level_order=VERBAL_ORDER,
                        outpath=verbal_violin_iqr_out,
                        title=build_violin_title(threshold, scope, identity_tag, "5 verbal bins", args.metric),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        overlay_style="box",
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                    quant_violin_iqr_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_violin_sep_quantiles5_iqr.png"
                    )
                    plot_full_violins(
                        sub,
                        level_col="sep_quantile_5",
                        level_order=[f"q{i}" for i in range(1, args.n_sep_quantiles + 1)],
                        outpath=quant_violin_iqr_out,
                        title=build_violin_title(
                            threshold,
                            scope,
                            identity_tag,
                            f"{args.n_sep_quantiles} separation quantiles",
                            args.metric,
                        ),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        overlay_style="box",
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                    vertical_suffix_parts: list[str] = []
                    if args.vertical_summary_style != "box":
                        vertical_suffix_parts.append(args.vertical_summary_style)
                    if args.vertical_whiskers == "hide":
                        vertical_suffix_parts.append("no_whiskers")
                    vertical_suffix = "" if not vertical_suffix_parts else "_" + "_".join(vertical_suffix_parts)

                    verbal_half_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_halfviolin_vertical_verbal_sep{vertical_suffix}.png"
                    )
                    plot_split_half_violins_vertical(
                        sub,
                        level_col="separation_level_db",
                        level_order=VERBAL_ORDER,
                        outpath=verbal_half_out,
                        title=build_violin_title(threshold, scope, identity_tag, "5 verbal bins", args.metric),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        summary_style=args.vertical_summary_style,
                        show_whiskers=args.vertical_whiskers == "show",
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )

                    quant_half_out = (
                        args.output_dir
                        / threshold.lower()
                        / scope
                        / identity_tag
                        / f"{threshold.lower()}_{scope}_{identity_tag}_{metric_key}_halfviolin_vertical_sep_quantiles5{vertical_suffix}.png"
                    )
                    plot_split_half_violins_vertical(
                        sub,
                        level_col="sep_quantile_5",
                        level_order=[f"q{i}" for i in range(1, args.n_sep_quantiles + 1)],
                        outpath=quant_half_out,
                        title=build_violin_title(
                            threshold,
                            scope,
                            identity_tag,
                            f"{args.n_sep_quantiles} separation quantiles",
                            args.metric,
                        ),
                        metric=args.metric,
                        x_upper_quantile=args.x_upper_quantile,
                        min_points=args.min_points,
                        summary_style=args.vertical_summary_style,
                        show_whiskers=args.vertical_whiskers == "show",
                        metric_cap_mode=args.metric_cap_mode,
                        metric_cap_quantile=args.metric_cap_quantile,
                        metric_cap=args.metric_cap,
                        include_all_group=args.vertical_include_all_group,
                        all_group_position=args.vertical_all_position,
                    )


    print(f"[SAVE] {args.output_dir}")


if __name__ == "__main__":
    main()
