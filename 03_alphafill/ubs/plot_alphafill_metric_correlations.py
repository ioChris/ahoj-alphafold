"""
Plot correlation-focused AlphaFill analysis panels from rebuilt per-UBS tables.

This script is intentionally separate from the ridgeline workflow. It focuses on
continuous relationships between AlphaFill fit metrics and AHoJ/AlphaFold support
metrics such as separation and confidence.

Outputs per threshold/scope/identity tag:
- correlation_matrix_<threshold>_<scope>_<tag>.csv
- correlation_heatmap_<threshold>_<scope>_<tag>.png
- metric_vs_separation_<threshold>_<scope>_<tag>.png
- pair_matrix_<threshold>_<scope>_<tag>.png
- manifest_<threshold>_<scope>_<tag>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg


DEFAULT_ANALYSIS_DIR = cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "correlation_panels_v1"

AF_METRICS = [
    "avg_clash",
    "avg_local_rmsd",
    "avg_global_rmsd",
    "avg_clash_count",
]
SUPPORT_METRICS = [
    "pocket_separation_pair",
    "winning_p",
    "pocket_score",
    "pocket_density_pair",
    "pocket_overlap_pair",
    "model_pocket_plddt",
]
PAIR_METRICS = [
    "avg_clash",
    "avg_local_rmsd",
    "avg_global_rmsd",
    "avg_clash_count",
    "pocket_separation_pair",
]
GRID_X_METRICS = [
    "pocket_separation_pair",
    "pocket_density_pair",
    "winning_p",
]

CLASS_ORDER = ["apo", "holo"]
CLASS_LABELS = {"apo": "AF-Apo", "holo": "AF-Holo"}
CLASS_COLORS = {"apo": "#8DB9DC", "holo": "#F4BD84"}
EDGE_DARK = "0.35"
SIGNED_PROB_CMAP = LinearSegmentedColormap.from_list(
    "apo_holo_prob",
    ["#4F92C9", "#D9D9D9", "#E9A24A"],
    N=256,
)
EVIDENCE_RATIO_CMAP = LinearSegmentedColormap.from_list(
    "apo_holo_evidence",
    ["#4F92C9", "#E5E5E5", "#E9A24A"],
    N=256,
)

FONT_STACK = ["Helvetica Neue", "Arial Nova", "Arial", "Liberation Sans", "DejaVu Sans"]
TITLE_FS = 10
LAB_FS = 8.4
TICK_FS = 7.2
SMALL_FS = 6.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AlphaFill metric correlation panels.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--thresholds", nargs="*", default=["ST_100"])
    p.add_argument("--scopes", nargs="*", default=["same_uniprot", "any"])
    p.add_argument("--identity_tags", nargs="*", default=["ge_0p30"])
    p.add_argument("--min_points", type=int, default=40)
    p.add_argument("--corr_method", choices=["spearman", "pearson"], default="spearman")
    p.add_argument(
        "--holo_plus_one",
        type=float,
        default=0.0,
        help="Add this value to n_holo_avg/evidence_n_holo before deriving evidence-based backgrounds.",
    )
    return p.parse_args()


def setup_mpl() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = FONT_STACK
    mpl.rcParams["axes.titlesize"] = TITLE_FS
    mpl.rcParams["axes.labelsize"] = LAB_FS
    mpl.rcParams["xtick.labelsize"] = TICK_FS
    mpl.rcParams["ytick.labelsize"] = TICK_FS
    mpl.rcParams["legend.fontsize"] = SMALL_FS
    mpl.rcParams["svg.fonttype"] = "path"


def save_figure(fig: plt.Figure, outpath: Path, dpi: int = 220) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")


def pretty_metric(metric: str) -> str:
    mapping = {
        "avg_clash": "Average clash score",
        "avg_local_rmsd": "Average local RMSD (Å)",
        "avg_global_rmsd": "Average global RMSD (Å)",
        "avg_clash_count": "Average clash count",
        "pocket_separation_pair": "Pocket separation",
        "winning_p": "Winning probability",
        "pocket_score": "Pocket score",
        "pocket_density_pair": "Pocket density",
        "pocket_overlap_pair": "Pocket overlap pair",
        "model_pocket_plddt": "Pocket pLDDT",
    }
    return mapping.get(metric, metric)


def safe_slug(text: str) -> str:
    return str(text).replace(".", "p").replace(">=", "ge_").replace(" ", "_")


def pretty_scope(scope: str) -> str:
    return scope.replace("_", "-")


def pretty_identity(identity_tag: str) -> str:
    return identity_tag.replace("ge_", ">=").replace("p", ".")


def normalize_class(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.lower()
    out = out.where(out.isin(CLASS_ORDER))
    return out


def load_summary_csv(analysis_dir: Path, scope: str, identity_tag: str) -> pd.DataFrame:
    path = analysis_dir / f"per_ubs_summary_{scope}_{identity_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def prepare_df(df: pd.DataFrame, threshold: str, min_points: int, holo_plus_one: float = 0.0) -> pd.DataFrame:
    d = df.copy()
    d = d.loc[d["threshold"].astype(str) == str(threshold)].copy()
    d = d.loc[d["matched"].fillna(False)].copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d = d.loc[d["_class"].isin(CLASS_ORDER)].copy()
    for col in AF_METRICS + SUPPORT_METRICS:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if "winning_p" not in d.columns:
        p_apo = pd.to_numeric(d.get("pocket_p_apo"), errors="coerce")
        p_holo = pd.to_numeric(d.get("pocket_p_holo"), errors="coerce")
        d["winning_p"] = pd.concat([p_apo, p_holo], axis=1).max(axis=1)
    p_apo = pd.to_numeric(d.get("pocket_p_apo"), errors="coerce")
    p_holo = pd.to_numeric(d.get("pocket_p_holo"), errors="coerce")
    if p_apo.notna().any() and p_holo.notna().any():
        d["signed_probability"] = p_holo - p_apo
    else:
        sign = np.where(d["_class"].eq("holo"), 1.0, -1.0)
        d["signed_probability"] = sign * ((pd.to_numeric(d["winning_p"], errors="coerce") - 0.5) / 0.5)
    n_apo_avg = pd.to_numeric(d.get("n_apo_avg"), errors="coerce")
    n_holo_avg = pd.to_numeric(d.get("n_holo_avg"), errors="coerce")
    # Evidence ratio: holo-only +1 (re-adds the query holo excluded upstream), apo raw — decision 2,
    # applied EXACTLY ONCE and computed on the RAW counts (independent of --holo_plus_one, which only
    # adjusts the stored evidence columns below). apo==0 -> NaN guard. Matches Figure3A / 04_p2rank.
    d["signed_evidence_ratio"] = np.log2((n_holo_avg + 1.0) / n_apo_avg.replace(0, np.nan))
    if holo_plus_one:
        n_holo_avg = n_holo_avg + float(holo_plus_one)
        if "evidence_n_holo" in d.columns:
            d["evidence_n_holo"] = pd.to_numeric(d.get("evidence_n_holo"), errors="coerce") + float(holo_plus_one)
        d["n_holo_avg"] = n_holo_avg
    d["total_avg_evidence"] = n_apo_avg + n_holo_avg
    d = d.dropna(subset=["avg_clash", "avg_local_rmsd", "avg_global_rmsd", "avg_clash_count", "pocket_separation_pair"])
    counts = d["_class"].value_counts()
    keep_classes = [cls for cls in CLASS_ORDER if counts.get(cls, 0) >= min_points]
    d = d.loc[d["_class"].isin(keep_classes)].copy()
    return d


def nice_cap(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 1.0
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    q95 = float(s.quantile(0.95))
    q99 = float(s.quantile(0.99))
    iqr = q3 - q1
    hybrid = min(q99, q3 + 3.0 * iqr) if np.isfinite(iqr) else q99
    cap = max(q95, hybrid)
    if cap <= 1.0:
        return float(np.ceil(cap * 10.0) / 10.0)
    if cap <= 10:
        return float(np.ceil(cap))
    if cap <= 100:
        return float(np.ceil(cap / 5.0) * 5.0)
    return float(np.ceil(cap / 10.0) * 10.0)


def plot_caps(df: pd.DataFrame) -> dict[str, float]:
    caps = {}
    for metric in AF_METRICS:
        caps[metric] = nice_cap(df[metric])
    caps["pocket_separation_pair"] = 1.0
    caps["winning_p"] = 1.0
    caps["pocket_score"] = 1.0
    caps["pocket_density_pair"] = 1.0
    caps["pocket_overlap_pair"] = 1.0
    caps["model_pocket_plddt"] = float(np.ceil(df["model_pocket_plddt"].quantile(0.995) / 5.0) * 5.0)
    return caps


def build_correlation_table(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for i, x in enumerate(cols):
        for y in cols[i + 1 :]:
            tmp = df[[x, y]].dropna()
            if tmp.empty:
                continue
            r = float(tmp[x].corr(tmp[y], method=method))
            rows.append({"x": x, "y": y, "r": r, "n": int(len(tmp)), "method": method})
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, cols: list[str], outpath: Path, title: str, method: str) -> None:
    corr = df[cols].corr(method=method)
    labels = [pretty_metric(c) for c in cols]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=SMALL_FS, color=("white" if abs(val) > 0.55 else "0.15"))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(f"{method.title()} r")
    save_figure(fig, outpath)
    plt.close(fig)


def add_binned_class_lines(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, cap_y: float) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2.0
    for cls in CLASS_ORDER:
        sub = df.loc[df["_class"] == cls, [x_col, y_col]].dropna().copy()
        if sub.empty:
            continue
        sub = sub.loc[sub[y_col] <= cap_y].copy()
        sub["_bin"] = pd.cut(sub[x_col], bins=bins, include_lowest=True, labels=False)
        med = sub.groupby("_bin", observed=True)[y_col].median()
        xs = [centers[int(i)] for i in med.index]
        ys = med.to_numpy()
        ax.plot(xs, ys, color=EDGE_DARK, lw=1.2, alpha=0.95, zorder=5)
        ax.scatter(xs, ys, s=15, color=CLASS_COLORS[cls], edgecolor=EDGE_DARK, linewidth=0.5, zorder=6, label=CLASS_LABELS[cls])


def plot_metric_vs_separation(df: pd.DataFrame, caps: dict[str, float], outpath: Path, title: str, method: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, AF_METRICS):
        sub = df.loc[df[metric] <= caps[metric], ["pocket_separation_pair", metric, "_class"]].dropna().copy()
        hb = ax.hexbin(
            sub["pocket_separation_pair"],
            sub[metric],
            gridsize=34,
            mincnt=1,
            cmap="Greys",
            linewidths=0.0,
            bins="log",
        )
        add_binned_class_lines(ax, df, "pocket_separation_pair", metric, caps[metric])
        tmp = sub[["pocket_separation_pair", metric]].dropna()
        r = float(tmp["pocket_separation_pair"].corr(tmp[metric], method=method)) if len(tmp) >= 3 else np.nan
        ax.text(0.98, 0.96, f"{method[0]}={r:.2f}", ha="right", va="top", transform=ax.transAxes, fontsize=SMALL_FS, bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.2"})
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, caps[metric])
        ax.set_xlabel("Pocket separation")
        ax.set_ylabel(pretty_metric(metric))
        ax.set_title(pretty_metric(metric), fontsize=LAB_FS)
        ax.grid(axis="y", color="#e8e8e8", lw=0.6, linestyle="--")
        ax.grid(False, axis="x")
    handles = [
        plt.Line2D([], [], marker="o", markersize=5, linestyle="-", lw=1.2, color=EDGE_DARK, markerfacecolor=CLASS_COLORS["apo"], markeredgecolor=EDGE_DARK, label=CLASS_LABELS["apo"]),
        plt.Line2D([], [], marker="o", markersize=5, linestyle="-", lw=1.2, color=EDGE_DARK, markerfacecolor=CLASS_COLORS["holo"], markeredgecolor=EDGE_DARK, label=CLASS_LABELS["holo"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.985), handlelength=1.4, columnspacing=1.0)
    fig.suptitle(title, y=1.035, fontsize=TITLE_FS)
    save_figure(fig, outpath)
    plt.close(fig)


def compute_global_hexbin_max(df: pd.DataFrame, x_metrics: list[str], y_metrics: list[str], caps: dict[str, float], gridsize: int = 34) -> int:
    max_count = 1
    for x_col in x_metrics:
        x_cap = caps.get(x_col, 1.0)
        for y_col in y_metrics:
            y_cap = caps[y_col]
            tmp = df[[x_col, y_col]].dropna().copy()
            tmp = tmp.loc[(tmp[x_col] >= 0.0) & (tmp[x_col] <= x_cap) & (tmp[y_col] >= 0.0) & (tmp[y_col] <= y_cap)]
            if len(tmp) < 2:
                continue
            x = tmp[x_col].to_numpy()
            y = tmp[y_col].to_numpy()
            counts, _, _ = np.histogram2d(x, y, bins=gridsize, range=[[0.0, x_cap], [0.0, y_cap]])
            panel_max = int(np.nanmax(counts))
            if panel_max > max_count:
                max_count = panel_max
    return max_count


def support_axis_range(df: pd.DataFrame, x_col: str, x_cap: float) -> tuple[float, float]:
    if x_col == "winning_p":
        xmin = float(pd.to_numeric(df[x_col], errors="coerce").dropna().quantile(0.01))
        xmin = max(0.5, np.floor(xmin * 20.0) / 20.0)
        return xmin, x_cap
    if x_col == "pocket_score":
        return -1.0, 1.0
    return 0.0, x_cap


def plot_metric_vs_support_grid(df: pd.DataFrame, caps: dict[str, float], outpath: Path, title: str, method: str, x_metrics: list[str] | None = None) -> None:
    plot_metric_vs_support_grid_mode(df, caps, outpath, title, method, x_metrics=x_metrics, background_mode="density")


def plot_metric_vs_support_grid_mode(
    df: pd.DataFrame,
    caps: dict[str, float],
    outpath: Path,
    title: str,
    method: str,
    x_metrics: list[str] | None = None,
    background_mode: str = "density",
) -> None:
    x_metrics = x_metrics or GRID_X_METRICS
    y_metrics = AF_METRICS
    nrows = len(y_metrics)
    ncols = len(x_metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6, 9.6), constrained_layout=True, sharex="col", sharey="row")
    axes = np.asarray(axes)
    gridsize = 34
    vmax = compute_global_hexbin_max(df, x_metrics, y_metrics, caps, gridsize=gridsize)
    count_norm = LogNorm(vmin=1, vmax=max(2, vmax))
    signed_norm = Normalize(vmin=-1.0, vmax=1.0)
    evidence_norm = Normalize(vmin=-3.0, vmax=3.0)
    total_evidence_norm = LogNorm(vmin=1.0, vmax=max(2.0, float(np.nanpercentile(df["total_avg_evidence"].dropna(), 99.5)) if df["total_avg_evidence"].notna().any() else 2.0))
    last_hb = None
    class_handles = [
        plt.Line2D([], [], marker="o", markersize=5, linestyle="-", lw=1.2, color=EDGE_DARK, markerfacecolor=CLASS_COLORS["apo"], markeredgecolor=EDGE_DARK, label=CLASS_LABELS["apo"]),
        plt.Line2D([], [], marker="o", markersize=5, linestyle="-", lw=1.2, color=EDGE_DARK, markerfacecolor=CLASS_COLORS["holo"], markeredgecolor=EDGE_DARK, label=CLASS_LABELS["holo"]),
    ]
    for r, y_col in enumerate(y_metrics):
        for c, x_col in enumerate(x_metrics):
            ax = axes[r, c]
            x_cap = caps.get(x_col, 1.0)
            x_min, x_max = support_axis_range(df, x_col, x_cap)
            sub = df[[x_col, y_col, "_class"]].dropna().copy()
            sub = sub.loc[(sub[x_col] >= x_min) & (sub[x_col] <= x_max) & (sub[y_col] >= 0.0) & (sub[y_col] <= caps[y_col])]
            if background_mode == "signed_probability":
                sub_bg = df[[x_col, y_col, "signed_probability"]].dropna().copy()
                sub_bg = sub_bg.loc[(sub_bg[x_col] >= x_min) & (sub_bg[x_col] <= x_max) & (sub_bg[y_col] >= 0.0) & (sub_bg[y_col] <= caps[y_col])]
                last_hb = ax.hexbin(
                    sub_bg[x_col],
                    sub_bg[y_col],
                    C=sub_bg["signed_probability"],
                    reduce_C_function=np.mean,
                    gridsize=gridsize,
                    mincnt=1,
                    cmap=SIGNED_PROB_CMAP,
                    linewidths=0.0,
                    norm=signed_norm,
                )
            elif background_mode == "evidence_ratio":
                sub_bg = df[[x_col, y_col, "signed_evidence_ratio"]].dropna().copy()
                sub_bg = sub_bg.loc[(sub_bg[x_col] >= x_min) & (sub_bg[x_col] <= x_max) & (sub_bg[y_col] >= 0.0) & (sub_bg[y_col] <= caps[y_col])]
                last_hb = ax.hexbin(
                    sub_bg[x_col],
                    sub_bg[y_col],
                    C=sub_bg["signed_evidence_ratio"].clip(-3.0, 3.0),
                    reduce_C_function=np.mean,
                    gridsize=gridsize,
                    mincnt=1,
                    cmap=EVIDENCE_RATIO_CMAP,
                    linewidths=0.0,
                    norm=evidence_norm,
                )
            elif background_mode == "total_evidence":
                sub_bg = df[[x_col, y_col, "total_avg_evidence"]].dropna().copy()
                sub_bg = sub_bg.loc[(sub_bg[x_col] >= x_min) & (sub_bg[x_col] <= x_max) & (sub_bg[y_col] >= 0.0) & (sub_bg[y_col] <= caps[y_col])]
                last_hb = ax.hexbin(
                    sub_bg[x_col],
                    sub_bg[y_col],
                    C=sub_bg["total_avg_evidence"].clip(lower=1.0),
                    reduce_C_function=np.mean,
                    gridsize=gridsize,
                    mincnt=1,
                    cmap="Greys",
                    linewidths=0.0,
                    norm=total_evidence_norm,
                )
            else:
                last_hb = ax.hexbin(
                    sub[x_col],
                    sub[y_col],
                    gridsize=gridsize,
                    mincnt=1,
                    cmap="Greys",
                    linewidths=0.0,
                    norm=count_norm,
                )
            add_binned_class_lines(ax, df, x_col, y_col, caps[y_col])
            tmp = sub[[x_col, y_col]].dropna()
            r_val = float(tmp[x_col].corr(tmp[y_col], method=method)) if len(tmp) >= 3 else np.nan
            ax.text(
                0.98,
                0.96,
                f"{method[0]}={r_val:.2f}",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=SMALL_FS,
                bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.18"},
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.0, caps[y_col])
            ax.grid(axis="y", color="#e8e8e8", lw=0.6, linestyle="--")
            ax.grid(False, axis="x")
            if c == 0:
                ax.set_ylabel(pretty_metric(y_col), fontsize=LAB_FS)
            else:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
            if y_col == "avg_local_rmsd":
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            if r < nrows - 1:
                ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
    axes[0, 1].legend(
        class_handles,
        [h.get_label() for h in class_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.1,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes.ravel().tolist(), fraction=0.022, pad=0.012)
        if background_mode == "signed_probability":
            cbar.set_label("Mean signed pocket probability")
        elif background_mode == "evidence_ratio":
            cbar.set_label("Mean log2(holo/apo evidence)")
        elif background_mode == "total_evidence":
            cbar.set_label("Mean total avg evidence per hexbin")
        else:
            cbar.set_label("Matched UBS count per hexbin (log)")
        cbar.ax.tick_params(labelsize=SMALL_FS)
    fig.canvas.draw()
    for c, x_col in enumerate(x_metrics):
        col_axes = axes[:, c]
        x0 = min(ax.get_position().x0 for ax in col_axes)
        x1 = max(ax.get_position().x1 for ax in col_axes)
        y0 = min(ax.get_position().y0 for ax in col_axes)
        fig.text(
            (x0 + x1) / 2.0,
            y0 - 0.03,
            pretty_metric(x_col),
            va="top",
            ha="center",
            fontsize=LAB_FS,
            color="0.15",
        )
    fig.suptitle(title, y=1.01, fontsize=TITLE_FS)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_pair_matrix(df: pd.DataFrame, caps: dict[str, float], outpath: Path, title: str, method: str) -> None:
    plot_df = df.copy()
    for metric in AF_METRICS:
        plot_df = plot_df.loc[plot_df[metric] <= caps[metric]].copy()
    n = len(PAIR_METRICS)
    fig, axes = plt.subplots(n, n, figsize=(9.0, 9.0), constrained_layout=True)
    for i, ycol in enumerate(PAIR_METRICS):
        for j, xcol in enumerate(PAIR_METRICS):
            ax = axes[i, j]
            if i == j:
                vals = plot_df[xcol].dropna()
                ax.hist(vals, bins=28, color="0.70", edgecolor="white")
                ax.set_title(pretty_metric(xcol), fontsize=SMALL_FS)
            elif i > j:
                tmp = plot_df[[xcol, ycol]].dropna()
                ax.hexbin(tmp[xcol], tmp[ycol], gridsize=26, mincnt=1, cmap="Greys", linewidths=0.0)
            else:
                tmp = plot_df[[xcol, ycol]].dropna()
                if len(tmp) >= 3:
                    r = float(tmp[xcol].corr(tmp[ycol], method=method))
                    ax.text(0.5, 0.58, f"{method.title()} r", ha="center", va="center", transform=ax.transAxes, fontsize=SMALL_FS, color="0.45")
                    ax.text(0.5, 0.40, f"{r:.2f}", ha="center", va="center", transform=ax.transAxes, fontsize=LAB_FS, color=("0.10" if abs(r) < 0.5 else "0.0"))
                ax.set_facecolor("#fafafa")
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(pretty_metric(xcol), fontsize=SMALL_FS)
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(pretty_metric(ycol), fontsize=SMALL_FS)
            ax.tick_params(labelsize=SMALL_FS - 0.2)
    fig.suptitle(title, y=1.01, fontsize=TITLE_FS)
    save_figure(fig, outpath)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_mpl()
    for scope in args.scopes:
        for identity_tag in args.identity_tags:
            df = load_summary_csv(args.analysis_dir, scope, identity_tag)
            for threshold in args.thresholds:
                d = prepare_df(df, threshold, args.min_points, holo_plus_one=args.holo_plus_one)
                if d.empty:
                    continue
                outdir = args.output_dir / threshold.lower() / scope / identity_tag
                outdir.mkdir(parents=True, exist_ok=True)
                corr_cols = AF_METRICS + SUPPORT_METRICS
                corr_df = build_correlation_table(d, corr_cols, args.corr_method)
                corr_csv = outdir / f"correlation_matrix_{threshold.lower()}_{scope}_{identity_tag}.csv"
                corr_df.to_csv(corr_csv, index=False)

                caps = plot_caps(d)
                heatmap_title = f"{threshold} AlphaFill/support correlation heatmap\nScope: {pretty_scope(scope)} | identity {pretty_identity(identity_tag)}"
                plot_heatmap(
                    d,
                    corr_cols,
                    outdir / f"correlation_heatmap_{threshold.lower()}_{scope}_{identity_tag}.png",
                    heatmap_title,
                    args.corr_method,
                )

                sep_title = f"{threshold} AlphaFill metrics versus pocket separation\nScope: {pretty_scope(scope)} | identity {pretty_identity(identity_tag)}"
                plot_metric_vs_separation(
                    d,
                    caps,
                    outdir / f"metric_vs_separation_{threshold.lower()}_{scope}_{identity_tag}.png",
                    sep_title,
                    args.corr_method,
                )

                support_grid_title = f"{threshold} AlphaFill fit metrics versus support axes | {pretty_scope(scope)} | {pretty_identity(identity_tag)}"
                plot_metric_vs_support_grid(
                    d,
                    caps,
                    outdir / f"metric_vs_support_grid_{threshold.lower()}_{scope}_{identity_tag}.png",
                    support_grid_title,
                    args.corr_method,
                )
                plot_metric_vs_support_grid_mode(
                    d,
                    caps,
                    outdir / f"metric_vs_support_grid_signedprob_{threshold.lower()}_{scope}_{identity_tag}.png",
                    support_grid_title + " | signed-probability background",
                    args.corr_method,
                    background_mode="signed_probability",
                )
                plot_metric_vs_support_grid_mode(
                    d,
                    caps,
                    outdir / f"metric_vs_support_grid_evidence_{threshold.lower()}_{scope}_{identity_tag}.png",
                    support_grid_title + " | holo/apo evidence background",
                    args.corr_method,
                    background_mode="evidence_ratio",
                )
                plot_metric_vs_support_grid_mode(
                    d,
                    caps,
                    outdir / f"metric_vs_support_grid_totalevidence_{threshold.lower()}_{scope}_{identity_tag}.png",
                    support_grid_title + " | total-evidence background",
                    args.corr_method,
                    background_mode="total_evidence",
                )

                pair_title = f"{threshold} AlphaFill fit metric relationships\nScope: {pretty_scope(scope)} | identity {pretty_identity(identity_tag)}"
                plot_pair_matrix(
                    d,
                    caps,
                    outdir / f"pair_matrix_{threshold.lower()}_{scope}_{identity_tag}.png",
                    pair_title,
                    args.corr_method,
                )

                manifest = {
                    "threshold": threshold,
                    "scope": scope,
                    "identity_tag": identity_tag,
                    "n_rows_matched": int(len(d)),
                    "corr_method": args.corr_method,
                    "holo_plus_one": args.holo_plus_one,
                    "plot_caps": caps,
                    "outputs": {
                        "correlation_csv": str(corr_csv),
                        "heatmap_png": str(outdir / f"correlation_heatmap_{threshold.lower()}_{scope}_{identity_tag}.png"),
                        "metric_vs_separation_png": str(outdir / f"metric_vs_separation_{threshold.lower()}_{scope}_{identity_tag}.png"),
                        "metric_vs_support_grid_png": str(outdir / f"metric_vs_support_grid_{threshold.lower()}_{scope}_{identity_tag}.png"),
                        "metric_vs_support_grid_signedprob_png": str(outdir / f"metric_vs_support_grid_signedprob_{threshold.lower()}_{scope}_{identity_tag}.png"),
                        "metric_vs_support_grid_evidence_png": str(outdir / f"metric_vs_support_grid_evidence_{threshold.lower()}_{scope}_{identity_tag}.png"),
                        "metric_vs_support_grid_totalevidence_png": str(outdir / f"metric_vs_support_grid_totalevidence_{threshold.lower()}_{scope}_{identity_tag}.png"),
                        "pair_matrix_png": str(outdir / f"pair_matrix_{threshold.lower()}_{scope}_{identity_tag}.png"),
                    },
                }
                with open(outdir / f"manifest_{threshold.lower()}_{scope}_{identity_tag}.json", "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
