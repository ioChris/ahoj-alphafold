# -*- coding: utf-8 -*-
"""
UBS processing pipeline with AFDB presence pre-filter.

Pre-filter:
  - Recursively look in AFDB_DOWNLOAD_DIR for AF-<UNIPROT>-F1-model_v*.cif.gz
  - Require ALL UniProts in each UBS (configurable) to be present in the set
  - Save excluded UBSs to ubs_excluded_missing_afdb.csv with 'missing_uniprots'

Downloading (commented out):
  - Download AlphaFold-DB models (v6) for the filtered UBSs (.cif.gz)

Changes:
  - Removed redundant 'final_class'; use 'winning_class' everywhere
  - 'mean_separation' rounded to 3 decimals
"""

import os
import re
import pandas as pd
import numpy as np
import gzip, requests
from pathlib import Path


def get_2level_cwd():
    npath = os.path.normpath(os.getcwd())   # Normalize the path string for the OS
    path0 = os.path.join(npath.split(os.sep)[0], os.sep, npath.split(os.sep)[1], os.path.sep.join(npath.split(os.sep)[2:3]))
    # the above keeps the behavior of your original function (first three path segments)
    if os.path.exists(path0):
        print("Root path found >> " + path0)
    else:
        print("Error finding root path in working dir:" + npath)
    return path0

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# ---- Config (EDIT ME) ----
INPUT_CSV_PATH      = str(cfg.WORK_DIR / "prevalence_sweep_output" / "text" / "sub_df_ST_LJ.csv")  # unzipped CSV path
OUTPUT_DIR          = str(cfg.WORK_DIR / "afdb_ubs_sets")
AFDB_DOWNLOAD_DIR   = str(cfg.AFDB_DIR)
LATEST_AFDB_VERSION = 6  # per your requirement
REQUIRE_ALL_UNIPROTS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Helpers ----

def _safe_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _parse_monomeric_only(val) -> bool:
    if pd.isna(val):
        return False
    try:
        if float(val) == 1 and (isinstance(val, (int, float)) or str(val).strip() == "1"):
            return True
    except Exception:
        pass
    s = str(val).strip()
    if s == "1":
        return True
    if any(sep in s for sep in (";", ",", " ")):
        return False
    return False

def _winning_class_and_p(row, col_apo, col_holo):
    a = row[col_apo]
    h = row[col_holo]
    if pd.isna(a) and pd.isna(h):
        return pd.Series({"winning_class": np.nan, "winning_p": np.nan})
    if pd.isna(h) or (a >= h):
        return pd.Series({"winning_class": "apo", "winning_p": a})
    return pd.Series({"winning_class": "holo", "winning_p": h})

def _label_from_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.65:
        return "low"
    if p < 0.70:
        return "moderate"
    if p < 0.925:
        return "high"
    return "very high"

def _verbal_band_merged(u: float) -> str:
    if pd.isna(u):
        return "n/a"
    if u <= 0.30:
        return "low"
    if u <= 0.60:
        return "moderate"
    return "high"

def _summarize_by_class_and_label(df, class_col, label_col):
    total = len(df)
    grp = df.groupby([class_col, label_col], dropna=False).size().reset_index(name="count")
    grp["pct_within_class"] = grp.groupby(class_col)["count"].transform(lambda x: x / x.sum() * 100.0)
    grp["pct_of_total"] = grp["count"] / total * 100.0 if total > 0 else 0.0
    return grp[[class_col, label_col, "count", "pct_within_class", "pct_of_total"]]

_UNIPROT_RE = re.compile(r"^AF-([A-Z0-9]+)-F1-model_v\d+\.cif\.gz$", re.I)

def list_afdb_uniprots(path_root: str) -> set[str]:
    found = set()
    for root, _, files in os.walk(path_root):
        for name in files:
            m = _UNIPROT_RE.match(name)
            if m:
                found.add(m.group(1))
    return found

def download_alphafold_model(pathAF, uniprot_id, latest_version: int):
    base_url = "https://alphafold.ebi.ac.uk/files/"
    for version in range(latest_version, 0, -1):
        file_name = f"AF-{uniprot_id}-F1-model_v{version}.cif.gz"
        full_file_path = os.path.join(pathAF, file_name)
        if os.path.exists(full_file_path):
            print(f"Model {file_name} already downloaded.")
            return full_file_path
        download_url = f"{base_url}AF-{uniprot_id}-F1-model_v{version}.cif"
        response = requests.get(download_url)
        if response.status_code == 200:
            os.makedirs(pathAF, exist_ok=True)
            with gzip.open(full_file_path, 'wb') as f:
                f.write(response.content)
            print(f"Model {file_name} downloaded and compressed successfully.")
            return full_file_path
    print(f"Failed to download any model for {uniprot_id}.")
    return None

def process_ubs_table_with_afdb_filter(df: pd.DataFrame,
                                       afdb_dir: str,
                                       require_all_uniprots: bool = True) -> dict:
    COL_APO = "avg_P(apo)_mean_pocket"
    COL_HOLO = "avg_P(holo)_mean_pocket"
    COL_MULTI = "multimeric_status_all_unique"
    COL_UNIPROTS = "site_uniprots"
    COL_DENSITY = "pair_density_mean_pocket"
    COL_OVERLAP = "pair_overlap_mean_pocket"

    df0 = df.copy()

    # Pre-filter by AFDB presence on disk
    if COL_UNIPROTS not in df0.columns:
        raise KeyError(f"Expected column '{COL_UNIPROTS}' not found.")
    df0["site_uniprot_list"] = df0[COL_UNIPROTS].astype(str).str.split("_").apply(
        lambda xs: sorted(set([x.strip() for x in xs if x.strip()]))
    )
    available_ids = list_afdb_uniprots(afdb_dir)

    def _row_has_models(ids: list[str]) -> bool:
        s = set(ids)
        if not s:
            return False
        if require_all_uniprots:
            return s.issubset(available_ids)
        else:
            return len(s.intersection(available_ids)) > 0

    has_models_mask = df0["site_uniprot_list"].apply(_row_has_models)
    df_missing = df0.loc[~has_models_mask].copy()

    def _missing(ids):
        return sorted(list(set(ids) - set(available_ids)))
    if not df_missing.empty:
        df_missing["missing_uniprots"] = df_missing["site_uniprot_list"].apply(_missing)

    df1 = df0.loc[has_models_mask].copy()

    # Task 1: require at least one of apo/holo p
    for c in (COL_APO, COL_HOLO):
        if c not in df1.columns:
            raise KeyError(f"Expected column '{c}' not found.")
        df1[c] = _safe_to_float(df1[c])
    m_has_any_prob = df1[COL_APO].apply(np.isfinite) | df1[COL_HOLO].apply(np.isfinite)
    df2 = df1.loc[m_has_any_prob].copy()

    # Task 2: strictly monomeric
    if COL_MULTI not in df2.columns:
        raise KeyError(f"Expected column '{COL_MULTI}' not found.")
    df2["_monomeric_only"] = df2[COL_MULTI].apply(_parse_monomeric_only)
    df3 = df2.loc[df2["_monomeric_only"]].drop(columns=["_monomeric_only"]).copy()

    # Task 3+4:
    unique_uniprots = sorted({u for lst in df3["site_uniprot_list"] for u in lst})
    add = df3.apply(lambda r: _winning_class_and_p(r, COL_APO, COL_HOLO), axis=1)
    df3 = pd.concat([df3, add], axis=1)
    df3["prob_level"] = df3["winning_p"].apply(_label_from_p)

    if COL_DENSITY in df3.columns:
        df3["mean_density_level"] = _safe_to_float(df3[COL_DENSITY]).apply(_verbal_band_merged)
    else:
        df3["mean_density_level"] = "n/a"

    if COL_OVERLAP in df3.columns:
        sep = 1.0 - _safe_to_float(df3[COL_OVERLAP])
        df3["mean_separation"] = np.round(sep, 3)  # rounded to 3 decimals
        df3["mean_separation_level"] = df3["mean_separation"].apply(_verbal_band_merged)
    else:
        df3["mean_separation"] = np.nan
        df3["mean_separation_level"] = "n/a"

    class_counts = df3["winning_class"].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    sep_summary = _summarize_by_class_and_label(df3, "winning_class", "mean_separation_level")
    dens_summary = _summarize_by_class_and_label(df3, "winning_class", "mean_density_level")
    prob_summary = _summarize_by_class_and_label(df3, "winning_class", "prob_level")

    return {
        "filtered_df": df3,
        "unique_uniprots": unique_uniprots,
        "class_counts": class_counts,
        "sep_summary": sep_summary,
        "dens_summary": dens_summary,
        "prob_summary": prob_summary,
        "excluded_missing_afdb": df_missing,
        "afdb_available_ids": sorted(list(available_ids)),
        "require_all_uniprots": require_all_uniprots,
    }

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV_PATH)
    art = process_ubs_table_with_afdb_filter(df, AFDB_DOWNLOAD_DIR, REQUIRE_ALL_UNIPROTS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = Path(INPUT_CSV_PATH).stem
    art["filtered_df"].to_csv(os.path.join(OUTPUT_DIR, f"{stem}_ubs_filtered_annotated.csv"), index=False)
    pd.DataFrame({"uniprot": art["unique_uniprots"]}).to_csv(os.path.join(OUTPUT_DIR, "unique_uniprots.csv"), index=False)
    art["class_counts"].to_csv(os.path.join(OUTPUT_DIR, "summary_class_counts.csv"), index=False)
    art["sep_summary"].to_csv(os.path.join(OUTPUT_DIR, "summary_separation_by_class.csv"), index=False)
    art["dens_summary"].to_csv(os.path.join(OUTPUT_DIR, "summary_density_by_class.csv"), index=False)
    art["prob_summary"].to_csv(os.path.join(OUTPUT_DIR, "summary_probability_level_by_class.csv"), index=False)

    excl_path = os.path.join(OUTPUT_DIR, "ubs_excluded_missing_afdb.csv")
    if not art["excluded_missing_afdb"].empty:
        art["excluded_missing_afdb"].to_csv(excl_path, index=False)
    else:
        import pandas as pd
        pd.DataFrame(columns=list(df.columns) + ["missing_uniprots"]).to_csv(excl_path, index=False)

    pd.DataFrame({"uniprot": art["afdb_available_ids"]}).to_csv(os.path.join(OUTPUT_DIR, "afdb_available_uniprots.csv"), index=False)

    # Optional:
    # for u in art["unique_uniprots"]:
    #     download_alphafold_model(AFDB_DOWNLOAD_DIR, u, LATEST_AFDB_VERSION)
