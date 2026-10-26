# -*- coding: utf-8 -*-
"""
Recreate and clean up the most useful legacy AlphaFill distribution plots from
the rebuilt unified per-UBS tables.

This script focuses on:
- clash-score histograms by class and separation
- split half-violin plots by class and separation
- support-vs-support scatter panels split by clash quantiles plus unmatched
- upper-tail ("worst clash") summaries versus separation categories/quantiles

Plots are produced separately for both `same_uniprot` and `any` match scopes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from confidence_labeling import FULL_LEVEL_ORDER, MERGED_LEVEL_ORDER, PROB_FULL_SUBSETS, PROB_MERGED_SUBSETS, pretty_level

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

DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "legacy_style_v1"

CLASS_ORDER = ["apo", "holo"]
CLASS_COLORS = {
    "apo": "#4C9ED9",
    "holo": "#E76F51",
    "unmatched": "#8C8C8C",
}
BIN_COLORS = ["#4CAF50", "#F4A261", "#E76F51", "#B22222"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recreate cleaned legacy-style AlphaFill distribution plots.")
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--match_scopes", nargs="*", default=["same_uniprot", "any"])
    p.add_argument("--identity_tags", nargs="*", default=["ge_0p30", "ge_0p90"])
    p.add_argument("--thresholds", nargs="*", default=["ST_100"])
    p.add_argument("--metric", default="avg_clash", choices=["avg_clash", "avg_local_rmsd"])
    p.add_argument(
        "--level_mode",
        choices=["merged", "full"],
        default="merged",
        help="Use merged verbal bins for plotting, or the full database-facing verbal labels.",
    )
    return p.parse_args()


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, usecols=usecols)


def normalize_class(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series("other", index=series.index, dtype="string")
    out.loc[s.str.contains("apo", na=False)] = "apo"
    out.loc[s.str.contains("holo", na=False)] = "holo"
    return out


def probability_subset(df: pd.DataFrame, allowed_levels: tuple[str, ...], *, prob_col: str = "prob_level") -> pd.DataFrame:
    if prob_col not in df.columns:
        return df.copy()
    levels = (
        df[prob_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )
    allowed = {x.replace("_", " ").lower() for x in allowed_levels}
    return df.loc[levels.isin(allowed)].copy()


def metric_label(metric: str) -> str:
    return {
        "avg_clash": "Average Clash Score",
        "avg_local_rmsd": "Average Local RMSD (A)",
        "avg_global_rmsd": "Average Global RMSD (A)",
        "avg_pae_mean": "Average PAE Mean",
        "avg_pae_median": "Average PAE Median",
    }.get(metric, metric)


def scope_label(scope: str) -> str:
    return "Same UniProt" if scope == "same_uniprot" else "Any UniProt"


def infer_threshold_label(threshold: str) -> str:
    return threshold.replace("_", " ")


def level_columns(level_mode: str) -> tuple[str, str, list[str], dict[str, tuple[str, ...]]]:
    if level_mode == "full":
        return "prob_level_db", "separation_level_db", FULL_LEVEL_ORDER[:-1], PROB_FULL_SUBSETS
    return "prob_level", "separation_level", MERGED_LEVEL_ORDER[:-1], PROB_MERGED_SUBSETS


def overlap_area(values_a: np.ndarray, values_b: np.ndarray, bins: np.ndarray) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan
    ha, _ = np.histogram(values_a, bins=bins, density=True)
    hb, _ = np.histogram(values_b, bins=bins, density=True)
    widths = np.diff(bins)
    return float(np.sum(np.minimum(ha, hb) * widths))


def plot_histograms_by_sep(
    df: pd.DataFrame,
    outpath: Path,
    metric: str,
    title: str,
    sep_col: str,
    sep_order: list[str] | None = None,
    x_upper_quantile: float = 0.995,
) -> None:
    sep_order = sep_order or MERGED_LEVEL_ORDER[:-1]
    d = df.copy()
    d = d.loc[d["matched"].fillna(False)].copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.loc[d["_class"].isin(CLASS_ORDER) & d[metric].notna() & d[sep_col].notna()].copy()
    if d.empty:
        return

    upper = float(d[metric].quantile(x_upper_quantile))
    if np.isfinite(upper):
        d = d.loc[d[metric] <= upper].copy()
    if d.empty:
        return

    x_min = float(d[metric].min())
    x_max = float(d[metric].max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return
    bins = np.linspace(x_min, x_max, 36)

    fig, axes = plt.subplots(len(sep_order), 1, figsize=(7.0, 2.5 * len(sep_order)), sharex=True, constrained_layout=True)
    if len(sep_order) == 1:
        axes = [axes]

    for ax, sep in zip(axes, sep_order):
        sub = d.loc[d[sep_col].astype(str).str.lower() == sep].copy()
        vals_apo = sub.loc[sub["_class"] == "apo", metric].to_numpy(dtype=float)
        vals_holo = sub.loc[sub["_class"] == "holo", metric].to_numpy(dtype=float)
        n_apo = len(vals_apo)
        n_holo = len(vals_holo)
        total = n_apo + n_holo

        if total == 0:
            ax.set_visible(False)
            continue

        if n_apo:
            ax.hist(vals_apo, bins=bins, density=True, color=CLASS_COLORS["apo"], alpha=0.45, label="apo")
        if n_holo:
            ax.hist(vals_holo, bins=bins, density=True, color=CLASS_COLORS["holo"], alpha=0.45, label="holo")

        ov = overlap_area(vals_apo, vals_holo, bins=bins)
        ann = "overlap = NA" if not np.isfinite(ov) else f"overlap = {ov:.2f}"
        ax.text(0.98, 0.92, ann, transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.text(
            -0.05,
            0.50,
            f"{pretty_level(sep)}\n(n={total})",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.20)

    axes[0].legend(frameon=False, loc="upper center", ncol=2)
    axes[-1].set_xlabel(metric_label(metric))
    fig.suptitle(title, fontsize=13)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def split_half_violin(
    ax,
    values: np.ndarray,
    center: float,
    side: str,
    color: str,
    width: float = 0.42,
    min_points: int = 10,
) -> None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < min_points or np.allclose(vals, vals[0]):
        return
    kde = gaussian_kde(vals)
    bw = np.std(vals) * kde.factor
    if not np.isfinite(bw) or bw <= 0:
        return
    y = np.linspace(np.min(vals) - 2.0 * bw, np.max(vals) + 2.0 * bw, 400)
    dens = kde(y)
    dens = np.clip(dens, 0, None)
    if dens.max() <= 0:
        return
    dens = dens / dens.max() * width
    if side == "left":
        x1 = center - dens
        x2 = np.full_like(y, center)
    else:
        x1 = np.full_like(y, center)
        x2 = center + dens
    ax.fill_betweenx(y, x1, x2, color=color, alpha=0.90, linewidth=0.0)
    ax.plot(x1 if side == "left" else x2, y, color="#555555", linewidth=1.0)


def plot_half_violins_by_sep(
    df: pd.DataFrame,
    outpath: Path,
    metric: str,
    title: str,
    sep_col: str,
    sep_order: list[str] | None = None,
    val_upper_quantile: float = 0.995,
) -> None:
    sep_order = sep_order or MERGED_LEVEL_ORDER[:-1]
    d = df.copy()
    d = d.loc[d["matched"].fillna(False)].copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.loc[d["_class"].isin(CLASS_ORDER) & d[metric].notna() & d[sep_col].notna()].copy()
    if d.empty:
        return

    upper = float(d[metric].quantile(val_upper_quantile))
    if np.isfinite(upper):
        d = d.loc[d[metric] <= upper].copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    centers = np.arange(1, len(sep_order) + 1)

    for center, sep in zip(centers, sep_order):
        sub = d.loc[d[sep_col].astype(str).str.lower() == sep].copy()
        vals_apo = sub.loc[sub["_class"] == "apo", metric].to_numpy(dtype=float)
        vals_holo = sub.loc[sub["_class"] == "holo", metric].to_numpy(dtype=float)
        split_half_violin(ax, vals_apo, center=center, side="left", color=CLASS_COLORS["apo"])
        split_half_violin(ax, vals_holo, center=center, side="right", color=CLASS_COLORS["holo"])

        for vals, shift, style in [
            (vals_apo, -0.12, CLASS_COLORS["apo"]),
            (vals_holo, 0.12, CLASS_COLORS["holo"]),
        ]:
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            ax.plot([center, center + shift], [med, med], color="#444444", linewidth=2.0)
            ax.plot([center, center + shift], [q1, q1], color="#666666", linewidth=1.2, linestyle="--")
            ax.plot([center, center + shift], [q3, q3], color="#666666", linewidth=1.2, linestyle="--")

        ax.text(center, ax.get_ylim()[0] if np.isfinite(ax.get_ylim()[0]) else 0, "", alpha=0)

    y_min, y_max = ax.get_ylim()
    y_text = y_min + 0.04 * (y_max - y_min)
    for center, sep in zip(centers, sep_order):
        sub = d.loc[d[sep_col].astype(str).str.lower() == sep].copy()
        n_apo = int((sub["_class"] == "apo").sum())
        n_holo = int((sub["_class"] == "holo").sum())
        ax.text(center, y_text, f"{n_apo} | {n_holo}", ha="center", va="bottom", fontsize=10, color="#444444")

    ax.set_xticks(centers)
    ax.set_xticklabels([pretty_level(x) for x in sep_order])
    ax.set_xlabel("Separation Category")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.20)
    handles = [
        plt.Line2D([0], [0], color=CLASS_COLORS["apo"], lw=8, label="apo"),
        plt.Line2D([0], [0], color=CLASS_COLORS["holo"], lw=8, label="holo"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_prob_scaled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = pd.to_numeric(out["winning_p"], errors="coerce").clip(0.5, 1.0)
    out["prob_scaled"] = 2.0 * (p - 0.5)
    return out


def fit_line(x: pd.Series, y: pd.Series) -> tuple[float, float, float, int]:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n
    x_use = xv[mask].to_numpy(dtype=float)
    y_use = yv[mask].to_numpy(dtype=float)
    if np.std(x_use) == 0 or np.std(y_use) == 0:
        return np.nan, np.nan, np.nan, n
    r = float(np.corrcoef(x_use, y_use)[0, 1])
    slope, intercept = np.polyfit(x_use, y_use, 1)
    return r, float(slope), float(intercept), n


def plot_binned_support_with_unmatched(
    df: pd.DataFrame,
    pocket_class: str,
    outpath: Path,
    title: str,
    n_bins: int = 4,
) -> None:
    d = df.copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d = d.loc[d["_class"] == pocket_class].copy()
    d = add_prob_scaled(d)
    d["pocket_density_pair"] = pd.to_numeric(d["pocket_density_pair"], errors="coerce")
    d["pocket_separation_pair"] = pd.to_numeric(d["pocket_separation_pair"], errors="coerce")
    d["avg_clash"] = pd.to_numeric(d["avg_clash"], errors="coerce")

    matched = d.loc[d["matched"].fillna(False) & d["avg_clash"].notna()].copy()
    unmatched = d.loc[~d["matched"].fillna(False)].copy()
    matched = matched.dropna(subset=["pocket_density_pair", "pocket_separation_pair"])
    unmatched = unmatched.dropna(subset=["pocket_density_pair", "pocket_separation_pair"])
    if matched.empty and unmatched.empty:
        return

    if len(matched) >= n_bins:
        matched["clash_bin"] = pd.qcut(matched["avg_clash"], q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates="drop")
        labels = [str(x) for x in matched["clash_bin"].cat.categories] + ["No Match"]
    else:
        matched["clash_bin"] = "Matched"
        labels = ["Matched", "No Match"]

    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.2), sharex=True, sharey=True, constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for idx, label in enumerate(labels):
        ax = axes[idx]
        if label == "No Match":
            sub = unmatched.copy()
            color = CLASS_COLORS["unmatched"]
        else:
            sub = matched.loc[matched["clash_bin"].astype(str) == label].copy()
            color = BIN_COLORS[min(idx, len(BIN_COLORS) - 1)]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(
            sub["pocket_density_pair"],
            sub["pocket_separation_pair"],
            s=10,
            alpha=0.55,
            color=color,
            edgecolors="none",
        )
        r, slope, intercept, n = fit_line(sub["pocket_density_pair"], sub["pocket_separation_pair"])
        if np.isfinite(slope):
            xs = np.linspace(0, 1, 100)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color="#444444", linewidth=1.3)
        ax.text(
            0.02,
            0.98,
            f"{label}\nr={r:.2f}\nn={n}" if np.isfinite(r) else f"{label}\nr=NA\nn={n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Pocket Density (pair)")
        if idx == 0:
            ax.set_ylabel("Pocket Separation (pair)")

    fig.suptitle(title, fontsize=13)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def assign_sep_quantiles(df: pd.DataFrame, n_bins: int = 4) -> pd.Series:
    vals = pd.to_numeric(df["pocket_separation_pair"], errors="coerce")
    if vals.notna().sum() < n_bins:
        return pd.Series(np.where(vals.notna(), "all", pd.NA), index=df.index, dtype="string")
    bins = pd.qcut(vals, q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates="drop")
    return bins.astype("string")


def plot_worst_tail_summary(
    df: pd.DataFrame,
    outpath: Path,
    title: str,
    metric: str = "avg_clash",
    sep_col: str = "separation_level",
    sep_order: list[str] | None = None,
    include_sep_quantiles: bool = True,
) -> None:
    d = df.copy()
    d = d.loc[d["matched"].fillna(False)].copy()
    d["_class"] = normalize_class(d["pocket_class"])
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d["pocket_separation_pair"] = pd.to_numeric(d["pocket_separation_pair"], errors="coerce")
    d = d.loc[
        d["_class"].isin(CLASS_ORDER)
        & d[metric].notna()
        & d[sep_col].notna()
        & d["pocket_separation_pair"].notna()
    ].copy()
    if d.empty:
        return

    q90 = float(d[metric].quantile(0.90))
    q95 = float(d[metric].quantile(0.95))
    d["worst_q10"] = d[metric] >= q90
    d["worst_q05"] = d[metric] >= q95
    if include_sep_quantiles:
        d["sep_quantile"] = assign_sep_quantiles(d)

    specs = [
            (sep_col, "worst_q10", f"Worst 10% {metric_label(metric).lower()} rate by separation level"),
            (sep_col, "worst_q05", f"Worst 5% {metric_label(metric).lower()} rate by separation level"),
        ]
    if include_sep_quantiles:
        specs = [
            (sep_col, "worst_q10", f"Worst 10% {metric_label(metric).lower()} rate by separation level"),
            ("sep_quantile", "worst_q10", f"Worst 10% {metric_label(metric).lower()} rate by separation quantile"),
            (sep_col, "worst_q05", f"Worst 5% {metric_label(metric).lower()} rate by separation level"),
            ("sep_quantile", "worst_q05", f"Worst 5% {metric_label(metric).lower()} rate by separation quantile"),
        ]

    ncols = 2 if include_sep_quantiles else 1
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5 if include_sep_quantiles else 6.8, 8.5), constrained_layout=True)
    axes_iter = axes.ravel() if include_sep_quantiles else np.array(axes).ravel()

    for ax, (group_col, tail_col, panel_title) in zip(axes_iter, specs):
        rows = []
        for cls in CLASS_ORDER:
            sub_cls = d.loc[d["_class"] == cls].copy()
            for grp, grp_df in sub_cls.groupby(group_col, dropna=False):
                if pd.isna(grp):
                    continue
                rows.append(
                    {
                        "class": cls,
                        "group": str(grp),
                        "rate": float(pd.to_numeric(grp_df[tail_col], errors="coerce").mean()),
                        "n": int(len(grp_df)),
                    }
                )
        plot_df = pd.DataFrame(rows)
        if plot_df.empty:
            ax.set_visible(False)
            continue
        order = (sep_order or MERGED_LEVEL_ORDER[:-1]) if group_col == sep_col else sorted(plot_df["group"].unique())
        x = np.arange(len(order))
        width = 0.36
        for idx, cls in enumerate(CLASS_ORDER):
            part = plot_df.loc[plot_df["class"] == cls].set_index("group").reindex(order).reset_index()
            ax.bar(
                x + (idx - 0.5) * width,
                pd.to_numeric(part["rate"], errors="coerce"),
                width=width,
                color=CLASS_COLORS[cls],
                alpha=0.85,
                label=cls,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_level(x) if group_col == sep_col else x for x in order])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Fraction in worst {metric_label(metric).lower()} tail")
        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.20)

    first_ax = axes[0, 0] if include_sep_quantiles else axes_iter[0]
    first_ax.legend(frameon=False, loc="upper left")
    fig.suptitle(
        title + f"\nUpper-tail thresholds: q90={q90:.3f}, q95={q95:.3f}",
        fontsize=13,
    )
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_output_dir(root: Path, threshold: str, scope: str, identity_tag: str) -> Path:
    outdir = root / threshold.lower() / scope / identity_tag
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_title(prefix: str, threshold: str, scope: str, identity_tag: str, suffix: str = "") -> str:
    base = f"{prefix} | {infer_threshold_label(threshold)} | {scope_label(scope)} | {identity_tag}"
    return base if not suffix else f"{base} | {suffix}"


def main() -> None:
    args = parse_args()
    prob_col, sep_col, sep_order, prob_subsets = level_columns(args.level_mode)
    for scope in args.match_scopes:
        for identity_tag in args.identity_tags:
            path = args.analysis_dir / f"per_ubs_summary_{scope}_{identity_tag}.csv"
            df = read_csv(path)
            for threshold in args.thresholds:
                sub_thr = df.loc[df["threshold"].astype(str) == threshold].copy()
                if sub_thr.empty:
                    continue
                outdir = make_output_dir(args.output_dir, threshold, scope, identity_tag)

                for subset_name, allowed_levels in prob_subsets.items():
                    sub = probability_subset(sub_thr, allowed_levels=allowed_levels, prob_col=prob_col)
                    if sub.empty:
                        continue
                    hist_title = build_title(
                        prefix=f"{metric_label(args.metric)} histogram by class and separation",
                        threshold=threshold,
                        scope=scope,
                        identity_tag=identity_tag,
                        suffix=subset_name,
                    )
                    hist_out = outdir / f"{threshold.lower()}_{scope}_{identity_tag}_{subset_name}_{args.metric}_hist_by_class_sep.png"
                    plot_histograms_by_sep(sub, outpath=hist_out, metric=args.metric, title=hist_title, sep_col=sep_col, sep_order=sep_order)

                    violin_title = build_title(
                        prefix=f"{metric_label(args.metric)} split violins by class and separation",
                        threshold=threshold,
                        scope=scope,
                        identity_tag=identity_tag,
                        suffix=subset_name,
                    )
                    violin_out = outdir / f"{threshold.lower()}_{scope}_{identity_tag}_{subset_name}_{args.metric}_halfviolin_by_class_sep.png"
                    plot_half_violins_by_sep(sub, outpath=violin_out, metric=args.metric, title=violin_title, sep_col=sep_col, sep_order=sep_order)

                for cls in CLASS_ORDER:
                    binned_title = build_title(
                        prefix=f"{cls.upper()} density vs separation by clash quantile plus unmatched",
                        threshold=threshold,
                        scope=scope,
                        identity_tag=identity_tag,
                    )
                    binned_out = outdir / f"{threshold.lower()}_{scope}_{identity_tag}_{cls}_density_vs_separation_clash_quantiles_plus_unmatched.png"
                    plot_binned_support_with_unmatched(sub_thr, pocket_class=cls, outpath=binned_out, title=binned_title)

                tail_title = build_title(
                    prefix="Worst-clash enrichment versus separation",
                    threshold=threshold,
                    scope=scope,
                    identity_tag=identity_tag,
                )
                tail_out = outdir / f"{threshold.lower()}_{scope}_{identity_tag}_worst_clash_tail_vs_separation.png"
                plot_worst_tail_summary(
                    sub_thr,
                    outpath=tail_out,
                    title=tail_title,
                    metric="avg_clash",
                    sep_col=sep_col,
                    sep_order=sep_order,
                    include_sep_quantiles=True,
                )

                for tail_metric in ["avg_local_rmsd", "avg_global_rmsd", "avg_pae_mean", "avg_pae_median"]:
                    tail_title = build_title(
                        prefix=f"Worst-{metric_label(tail_metric).lower()} enrichment versus separation",
                        threshold=threshold,
                        scope=scope,
                        identity_tag=identity_tag,
                    )
                    tail_out = outdir / f"{threshold.lower()}_{scope}_{identity_tag}_worst_{tail_metric}_tail_vs_separation_levels.png"
                    plot_worst_tail_summary(
                        sub_thr,
                        outpath=tail_out,
                        title=tail_title,
                        metric=tail_metric,
                        sep_col=sep_col,
                        sep_order=sep_order,
                        include_sep_quantiles=False,
                    )

    print(f"[SAVE] {args.output_dir}")


if __name__ == "__main__":
    main()
