"""
Summarize pocket/site size across unified thresholds and export manuscript-ready tables.

This script reads `analysis_unified_levels.csv` or `.zip`, keeps only confidently
classified pockets/sites (`pocket_class` in {apo, holo}), and summarizes size
across:
    UNCLUSTERED, ST_100, ST_070, ST_050, ST_030, ST_LJ

Key design choices
------------------
- Size column used across thresholds: `resis3_unit`
- UNCLUSTERED rows are taken from `level == "entry"`
- Clustered thresholds are taken from `level == "ubs"`
- Member-weighted statistics use `members` as weights
- Ties are excluded independently at each threshold after threshold-specific
  aggregation/classification, matching the manuscript interpretation

Outputs
-------
1. Raw summary table with internal column names
2. Main-text reduced table with publication-friendly headers
3. Supplementary (SI) table with a richer but still clean set of columns

Spyder-friendly: edit the CONFIG block and run.
"""

from __future__ import annotations

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

INPUT_PATH = cfg.ANALYSIS_UNIFIED_LEVELS_ZIP
OUTPUT_DIR = cfg.UNIFIED_RUN_DIR / "site_size_analysis" / "v2d"
OUTPUT_BASENAME = "Site_size_vs_threshold__resis3_unit"

# Thresholds to include and their expected level in the unified table.
THRESHOLD_LEVELS = [
    ("UNCLUSTERED", "entry"),
    ("ST_100", "ubs"),
    ("ST_070", "ubs"),
    ("ST_050", "ubs"),
    ("ST_030", "ubs"),
    ("ST_LJ", "ubs"),
]

CLASS_KEEP = {"apo", "holo"}
SIZE_COL = "resis3_unit"

USECOLS = [
    "threshold",
    "level",
    "pocket_class",
    "members",
    SIZE_COL,
    "uniprots_str",
    "job_names",
]


# =========================
# HELPERS
# =========================
def weighted_quantile(values, quantiles, sample_weight=None):
    """Return weighted quantiles for 1D numeric arrays."""
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(quantiles).astype(float)

    if sample_weight is None:
        sample_weight = np.ones(len(values), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[mask]
    sample_weight = sample_weight[mask]

    if len(values) == 0:
        out = np.full(len(quantiles), np.nan)
        return out[0] if out.size == 1 else out

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cumulative_weight = np.cumsum(sample_weight)
    total_weight = cumulative_weight[-1]
    weighted_cdf = (cumulative_weight - 0.5 * sample_weight) / total_weight

    out = np.interp(quantiles, weighted_cdf, values)
    return out[0] if out.size == 1 else out


def safe_round(value, ndigits=2):
    """Round numeric values while preserving NaN."""
    return np.nan if pd.isna(value) else round(float(value), ndigits)


def parse_uniprot_tokens(value) -> set[str]:
    """Parse whitespace-delimited UniProt accessions from a cell."""
    tokens = set()
    if pd.isna(value):
        return tokens
    for token in str(value).split():
        token = token.strip()
        if token:
            tokens.add(token)
    return tokens


def parse_job_tokens(value) -> list[str]:
    """Parse semicolon-delimited job names from a cell."""
    jobs = []
    if pd.isna(value):
        return jobs
    for token in str(value).split(";"):
        token = token.strip()
        if token:
            jobs.append(token)
    return jobs


def count_unique_uniprots(series: pd.Series) -> int:
    """Count unique UniProt accessions from whitespace-delimited `uniprots_str` values."""
    unique_uniprots = set()

    for value in series.dropna().astype(str):
        unique_uniprots.update(parse_uniprot_tokens(value))

    return len(unique_uniprots)


def count_unique_uniprots_from_superset_filter(series: pd.Series, allowed_uniprots: set[str]) -> int:
    """Count unique UniProts after intersecting row-native UniProt sets with an allowed superset."""
    unique_uniprots = set()
    for value in series.dropna().astype(str):
        unique_uniprots.update(parse_uniprot_tokens(value) & allowed_uniprots)
    return len(unique_uniprots)


def build_unclustered_job_to_uniprots_map(df_filtered: pd.DataFrame) -> dict[str, set[str]]:
    """Map filtered UNCLUSTERED job names to their UniProt sets."""
    ref = df_filtered[
        (df_filtered["threshold"] == "UNCLUSTERED") &
        (df_filtered["level"] == "entry")
    ].copy()

    job_to_uniprots: dict[str, set[str]] = {}
    for _, row in ref.iterrows():
        jobs = parse_job_tokens(row.get("job_names"))
        uniprots = parse_uniprot_tokens(row.get("uniprots_str"))
        for job in jobs:
            if job not in job_to_uniprots:
                job_to_uniprots[job] = set()
            job_to_uniprots[job].update(uniprots)
    return job_to_uniprots


def count_unique_uniprots_from_job_mapping(
    df_threshold: pd.DataFrame,
    job_to_uniprots: dict[str, set[str]],
) -> int:
    """Count unique UniProts supported by threshold members that map to filtered UNCLUSTERED jobs."""
    unique_uniprots = set()
    allowed_jobs = set(job_to_uniprots)

    for value in df_threshold["job_names"].dropna().astype(str):
        for job in parse_job_tokens(value):
            if job in allowed_jobs:
                unique_uniprots.update(job_to_uniprots[job])

    return len(unique_uniprots)


def summarize_group(
    df: pd.DataFrame,
    threshold_label: str,
    level_label: str,
    n_unique_uniprots_all_classes: int | None = None,
    n_unique_uniprots_unclustered_superset: int | None = None,
    n_unique_uniprots_unclustered_jobmapped: int | None = None,
) -> dict:
    """Compute raw summary statistics for one threshold/level subset."""
    n_unique_uniprots = count_unique_uniprots(df["uniprots_str"])
    if n_unique_uniprots_all_classes is None:
        n_unique_uniprots_all_classes = n_unique_uniprots
    sizes = pd.to_numeric(df[SIZE_COL], errors="coerce")
    members = pd.to_numeric(df["members"], errors="coerce")

    valid = sizes.notna() & members.notna()
    sizes = sizes.loc[valid].astype(float)
    members = members.loc[valid].astype(float)

    if len(sizes) == 0:
        return {
            "threshold": threshold_label,
            "level": level_label,
            "n_unique_uniprots": n_unique_uniprots,
            "n_unique_uniprots_all_classes": n_unique_uniprots_all_classes,
            "n_unique_uniprots_unclustered_superset": n_unique_uniprots_unclustered_superset,
            "n_unique_uniprots_unclustered_jobmapped": n_unique_uniprots_unclustered_jobmapped,
            "n_sites": 0,
            "total_members": np.nan,
            "members_mean": np.nan,
            "members_median": np.nan,
            "members_p90": np.nan,
            "members_max": np.nan,
            "mean_site_size": np.nan,
            "median_site_size": np.nan,
            "p90_site_size": np.nan,
            "p99_site_size": np.nan,
            "max_site_size": np.nan,
            "wmean_site_size_by_members": np.nan,
            "wmedian_site_size_by_members": np.nan,
        }

    return {
        "threshold": threshold_label,
        "level": level_label,
        "n_unique_uniprots": n_unique_uniprots,
        "n_unique_uniprots_all_classes": n_unique_uniprots_all_classes,
        "n_unique_uniprots_unclustered_superset": n_unique_uniprots_unclustered_superset,
        "n_unique_uniprots_unclustered_jobmapped": n_unique_uniprots_unclustered_jobmapped,
        "n_sites": int(len(sizes)),
        "total_members": int(round(members.sum())),
        "members_mean": safe_round(members.mean(), 2),
        "members_median": safe_round(members.median(), 2),
        "members_p90": safe_round(np.quantile(members, 0.90), 2),
        "members_max": safe_round(members.max(), 2),
        "mean_site_size": safe_round(sizes.mean(), 2),
        "median_site_size": safe_round(sizes.median(), 2),
        "p90_site_size": safe_round(np.quantile(sizes, 0.90), 2),
        "p99_site_size": safe_round(np.quantile(sizes, 0.99), 2),
        "max_site_size": safe_round(sizes.max(), 2),
        "wmean_site_size_by_members": safe_round(np.average(sizes, weights=members), 2),
        "wmedian_site_size_by_members": safe_round(weighted_quantile(sizes, 0.50, sample_weight=members), 2),
    }


def load_input_table(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Load CSV directly or first CSV found inside a ZIP archive."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV found inside zip: {path}")
            if len(csv_names) > 1:
                print(f"[INFO] Multiple CSV files found in zip; using first: {csv_names[0]}")
            with zf.open(csv_names[0]) as handle:
                return pd.read_csv(handle, usecols=usecols)

    return pd.read_csv(path, usecols=usecols)


def build_raw_summary(df_filtered: pd.DataFrame, df_all_classes: pd.DataFrame) -> pd.DataFrame:
    """Build the full internal/raw threshold summary table."""
    unclustered_allowed_uniprots = countable_uniprots = set()
    unclustered_ref = df_filtered[
        (df_filtered["threshold"] == "UNCLUSTERED") &
        (df_filtered["level"] == "entry")
    ].copy()
    for value in unclustered_ref["uniprots_str"].dropna().astype(str):
        countable_uniprots.update(parse_uniprot_tokens(value))
    unclustered_allowed_uniprots = countable_uniprots
    unclustered_job_to_uniprots = build_unclustered_job_to_uniprots_map(df_filtered)

    rows = []
    for threshold, level in THRESHOLD_LEVELS:
        sub_filtered = df_filtered[
            (df_filtered["threshold"] == threshold) & (df_filtered["level"] == level)
        ].copy()
        sub_all = df_all_classes[
            (df_all_classes["threshold"] == threshold) & (df_all_classes["level"] == level)
        ].copy()

        n_unique_uniprots_all_classes = count_unique_uniprots(sub_all["uniprots_str"])
        n_unique_uniprots_unclustered_superset = count_unique_uniprots_from_superset_filter(
            sub_filtered["uniprots_str"],
            unclustered_allowed_uniprots,
        )
        n_unique_uniprots_unclustered_jobmapped = count_unique_uniprots_from_job_mapping(
            sub_filtered,
            unclustered_job_to_uniprots,
        )

        rows.append(
            summarize_group(
                sub_filtered,
                threshold,
                level,
                n_unique_uniprots_all_classes=n_unique_uniprots_all_classes,
                n_unique_uniprots_unclustered_superset=n_unique_uniprots_unclustered_superset,
                n_unique_uniprots_unclustered_jobmapped=n_unique_uniprots_unclustered_jobmapped,
            )
        )
    return pd.DataFrame(rows)


def build_main_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the reduced main-text manuscript table."""
    main_df = raw_df[[
        "threshold",
        "n_unique_uniprots",
        "n_sites",
        "mean_site_size",
        "wmean_site_size_by_members",
        "median_site_size",
        "wmedian_site_size_by_members",
    ]].copy()

    main_df = main_df.rename(columns={
        "threshold": "Threshold",
        "n_unique_uniprots": "N unique UniProts",
        "n_sites": "N sites/UBSs",
        "mean_site_size": "Mean size",
        "median_site_size": "Median size",
        "wmean_site_size_by_members": "Weighted mean size",
        "wmedian_site_size_by_members": "Weighted median size",
    })

    # Main-text table: use 1 decimal for manuscript readability
    cols_to_round = [
        "Mean size",
        "Median size",
        "Weighted mean size",
        "Weighted median size",
    ]
    main_df[cols_to_round] = main_df[cols_to_round].round(1)

    return main_df


def build_si_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the richer supplementary table."""
    si_df = raw_df[[
        "threshold",
        "n_unique_uniprots",
        "n_unique_uniprots_all_classes",
        "level",
        "n_sites",
        "total_members",
        "mean_site_size",
        "wmean_site_size_by_members",
        "median_site_size",
        "wmedian_site_size_by_members",
        "p90_site_size",
        "p99_site_size",
        "max_site_size",
        "members_mean",
        "members_median",
        "members_p90",
        "members_max",
    ]].copy()

    si_df = si_df.rename(columns={
        "threshold": "Threshold",
        "n_unique_uniprots": "N unique UniProts",
        "n_unique_uniprots_all_classes": "N unique UniProts (all classes)",
        "level": "Representation level",
        "n_sites": "N sites/UBSs",
        "total_members": "Total member pockets",
        "mean_site_size": "Mean size",
        "median_site_size": "Median size",
        "p90_site_size": "P90 size",
        "p99_site_size": "P99 size",
        "max_site_size": "Max size",
        "wmean_site_size_by_members": "Weighted mean size",
        "wmedian_site_size_by_members": "Weighted median size",
        "members_mean": "Mean members per UBS",
        "members_median": "Median members per UBS",
        "members_p90": "P90 members per UBS",
        "members_max": "Max members per UBS",
    })
    return si_df


# =========================
# MAIN
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = load_input_table(INPUT_PATH, USECOLS)

    # Keep only confidently classified rows. Ties/undefined are excluded here.
    df_filtered = df_all[df_all["pocket_class"].isin(CLASS_KEEP)].copy()

    raw_summary_df = build_raw_summary(df_filtered, df_all)
    main_table_df = build_main_table(raw_summary_df)
    si_table_df = build_si_table(raw_summary_df)

    raw_out = OUTPUT_DIR / f"{OUTPUT_BASENAME}__raw.csv"
    main_out = OUTPUT_DIR / f"{OUTPUT_BASENAME}__main_table.csv"
    si_out = OUTPUT_DIR / f"{OUTPUT_BASENAME}__SI_table.csv"

    raw_summary_df.to_csv(raw_out, index=False)
    main_table_df.to_csv(main_out, index=False)
    si_table_df.to_csv(si_out, index=False)

    print(f"[OK] Wrote raw summary: {raw_out}")
    print(f"[OK] Wrote main-text table: {main_out}")
    print(f"[OK] Wrote SI table: {si_out}")

    print("\n=== Raw summary ===")
    print(raw_summary_df.to_string(index=False))
    print("\n=== Main-text table ===")
    print(main_table_df.to_string(index=False))
    print("\n=== SI table ===")
    print(si_table_df.to_string(index=False))


if __name__ == "__main__":
    main()
