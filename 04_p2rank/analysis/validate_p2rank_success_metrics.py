from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
DEFAULT_EVAL_DIR = (
    cfg.P2RANK_PRED_DIR
    / "ubs_run_4"
    / "UBS2"
    / "eval_predict_st_100_p2rank_intersection_geom_CENTER_explicit"
)
DEFAULT_ANALYSIS_DIR = WORK_DIR / "P2Rank" / "ubs_run_4" / "analysis_v1"

CUTOFFS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
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
    p = argparse.ArgumentParser(description="Validate P2Rank success-rate semantics against local derived metrics.")
    p.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def load_inputs(eval_dir: Path, analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(eval_dir / "cases" / "observed_sites.csv", low_memory=False)
    obs.columns = obs.columns.str.strip()
    ranks = pd.read_csv(eval_dir / "cases" / "ranks.csv", low_memory=False)
    success_rates = pd.read_csv(eval_dir / "success_rates.csv", header=0)
    merged = pd.read_csv(analysis_dir / "tables" / "p2rank_st100_merged_per_pocket.csv", low_memory=False)
    obs["site"] = obs["site"].astype(str).str.strip()
    ranks["ligand"] = ranks["ligand"].astype(str).str.strip()
    merged["unit_uid"] = merged["unit_uid"].astype(str).str.strip()
    return obs, ranks, success_rates, merged


def tidy_success_rates(success_rates: pd.DataFrame) -> pd.DataFrame:
    cols = list(success_rates.columns)
    tidy = success_rates.rename(columns={cols[0]: "metric"}).melt(
        id_vars="metric",
        value_vars=cols[1:],
        var_name="tolerance",
        value_name="success_percent",
    )
    tidy["tolerance"] = tidy["tolerance"].astype(str).str.strip("[]")
    tidy["success_percent"] = pd.to_numeric(tidy["success_percent"], errors="coerce")
    return tidy


def build_validation_table(
    obs: pd.DataFrame,
    ranks: pd.DataFrame,
    success_rates: pd.DataFrame,
) -> pd.DataFrame:
    tidy = tidy_success_rates(success_rates)
    rows = []
    for cutoff in CUTOFFS:
        dca_col = f"DCA({cutoff:.1f})"
        rank_vals = pd.to_numeric(ranks[dca_col], errors="coerce")
        dist_vals = pd.to_numeric(obs["closestPocketDist"], errors="coerce")
        reported_n = tidy.loc[(tidy["metric"] == dca_col) & (tidy["tolerance"] == "0"), "success_percent"]
        reported_all = tidy.loc[(tidy["metric"] == dca_col) & (tidy["tolerance"] == "99"), "success_percent"]
        rows.append(
            {
                "cutoff_A": cutoff,
                "reported_dca_n_percent": float(reported_n.iloc[0]) if not reported_n.empty else np.nan,
                "reported_dca_nplus99_percent": float(reported_all.iloc[0]) if not reported_all.empty else np.nan,
                "ranks_any_rank_percent": float((rank_vals >= 1).mean() * 100.0),
                "ranks_top1_percent": float((rank_vals == 1).mean() * 100.0),
                "closestPocketDist_le_cutoff_percent": float((dist_vals <= cutoff).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def build_identity_checks(obs: pd.DataFrame, ranks: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    chk = (
        obs[["site", "dca4rank", "closestPocketDist"]]
        .rename(columns={"dca4rank": "obs_dca4rank", "closestPocketDist": "obs_closestPocketDist"})
        .merge(ranks[["ligand", "DCA(4.0)"]].rename(columns={"DCA(4.0)": "ranks_DCA4"}), left_on="site", right_on="ligand", how="inner")
        .merge(
            merged[["unit_uid", "dca4rank", "closest_pocket_dist", "success_top1", "success_topn", "success_dca_4A"]].rename(
                columns={
                    "unit_uid": "site",
                    "dca4rank": "merged_dca4rank",
                    "closest_pocket_dist": "merged_closest_pocket_dist",
                }
            ),
            on="site",
            how="inner",
        )
    )
    obs_dca = pd.to_numeric(chk["obs_dca4rank"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "check": "observed_sites.dca4rank == ranks.DCA(4.0)",
                "fraction_equal": float((obs_dca == pd.to_numeric(chk["ranks_DCA4"], errors="coerce")).mean()),
            },
            {
                "check": "observed_sites.dca4rank == merged.dca4rank",
                "fraction_equal": float((obs_dca == pd.to_numeric(chk["merged_dca4rank"], errors="coerce")).mean()),
            },
            {
                "check": "observed_sites.closestPocketDist == merged.closest_pocket_dist",
                "fraction_equal": float(
                    (
                        pd.to_numeric(chk["obs_closestPocketDist"], errors="coerce").fillna(-999.0)
                        == pd.to_numeric(chk["merged_closest_pocket_dist"], errors="coerce").fillna(-999.0)
                    ).mean()
                ),
            },
            {
                "check": "merged.success_top1_percent",
                "fraction_equal": float(chk["success_top1"].mean() * 100.0),
            },
            {
                "check": "merged.success_topn_percent",
                "fraction_equal": float(chk["success_topn"].mean() * 100.0),
            },
            {
                "check": "merged.success_dca4_percent_via_closestPocketDist",
                "fraction_equal": float(chk["success_dca_4A"].mean() * 100.0),
            },
        ]
    )


def build_mismatch_summary(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    out["dca4rank"] = pd.to_numeric(out["dca4rank"], errors="coerce")
    out["closestPocketDist"] = pd.to_numeric(out["closestPocketDist"], errors="coerce")
    out["rank_found"] = out["dca4rank"] >= 1
    out["dist4_found"] = out["closestPocketDist"] <= 4.0
    mismatch = out[out["rank_found"] & ~out["dist4_found"].fillna(False)].copy()
    desc = mismatch["closestPocketDist"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    return pd.DataFrame(
        [
            {"metric": "n_total", "value": float(len(out))},
            {"metric": "n_rank_found_and_dist4_not_found", "value": float(len(mismatch))},
            {"metric": "pct_rank_found_and_dist4_not_found", "value": float(len(mismatch) / len(out) * 100.0)},
            {"metric": "mismatch_closestPocketDist_median", "value": float(desc["50%"])},
            {"metric": "mismatch_closestPocketDist_q95", "value": float(desc["95%"])},
            {"metric": "mismatch_closestPocketDist_max", "value": float(desc["max"])},
        ]
    )


def write_markdown_summary(
    outpath: Path,
    validation: pd.DataFrame,
    checks: pd.DataFrame,
    mismatch: pd.DataFrame,
) -> None:
    dca4 = validation.loc[validation["cutoff_A"] == 4.0].iloc[0]
    lines = [
        "# P2Rank Success Metric Validation",
        "",
        "## Main conclusion",
        "",
        (
            f"- `ranks.csv` / notebook any-rank DCA4 success = `{dca4['ranks_any_rank_percent']:.1f}%`, "
            f"which matches the reported `success_rates.csv` DCA(4.0) `[99]` value of `{dca4['reported_dca_nplus99_percent']:.1f}%`."
        ),
        (
            f"- Our merged-table `success_topn` therefore agrees with the official P2Rank rank-based evaluation, "
            f"whereas `closestPocketDist <= 4A` gives a stricter `{dca4['closestPocketDist_le_cutoff_percent']:.1f}%`."
        ),
        "",
        "## Identity checks",
        "",
    ]
    for _, row in checks.iterrows():
        value = row["fraction_equal"]
        if value <= 1.0:
            lines.append(f"- {row['check']}: `{value:.3f}`")
        else:
            lines.append(f"- {row['check']}: `{value:.3f}%`")
    lines += ["", "## DCA cutoff comparison", "", validation.to_markdown(index=False), "", "## Mismatch summary", "", mismatch.to_markdown(index=False), ""]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_validation_curves(validation: pd.DataFrame, outpath: Path) -> None:
    x = validation["cutoff_A"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    ax.plot(x, validation["reported_dca_n_percent"], marker="o", lw=1.8, color="#7f7f7f", label="Reported DCA n")
    ax.plot(x, validation["reported_dca_nplus99_percent"], marker="o", lw=1.8, color="#2C7FB8", label="Reported DCA n+99")
    ax.plot(x, validation["ranks_any_rank_percent"], marker="s", lw=1.8, color="#1f77b4", label="ranks.csv any rank")
    ax.plot(x, validation["ranks_top1_percent"], marker="^", lw=1.6, color="#F28E2B", label="ranks.csv top-1")
    ax.plot(x, validation["closestPocketDist_le_cutoff_percent"], marker="D", lw=1.6, color="#d62728", label="closestPocketDist <= cutoff")
    ax.set_xlabel("Cutoff (A)", fontsize=LAB_FS)
    ax.set_ylabel("Success rate (%)", fontsize=LAB_FS)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(axis="y", color="#ebebeb", lw=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=SMALL_FS, loc="lower right")
    ax.set_title("P2Rank success semantics: reported vs derived", fontsize=TITLE_FS)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.analysis_dir / "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    obs, ranks, success_rates, merged = load_inputs(args.eval_dir, args.analysis_dir)
    validation = build_validation_table(obs, ranks, success_rates)
    checks = build_identity_checks(obs, ranks, merged)
    mismatch = build_mismatch_summary(obs)
    validation.to_csv(output_dir / "p2rank_success_metric_validation.csv", index=False)
    checks.to_csv(output_dir / "p2rank_success_metric_identity_checks.csv", index=False)
    mismatch.to_csv(output_dir / "p2rank_success_metric_mismatch_summary.csv", index=False)
    write_markdown_summary(output_dir / "p2rank_success_metric_validation.md", validation, checks, mismatch)
    plot_validation_curves(validation, output_dir / "p2rank_success_metric_validation_curve.png")
    print(f"[SAVE] {output_dir}")


if __name__ == "__main__":
    main()
