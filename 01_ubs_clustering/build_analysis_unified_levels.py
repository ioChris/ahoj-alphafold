# -*- coding: utf-8 -*-
"""
Build a unified, analysis-ready table across:
  1) Unclustered entries (filtered_survivors.csv)
  2) Clustered UBS aggregates (ubs_allthresholds_aggregated_classifications_filtered.csv)

Spyder-friendly: edit INPUT_* paths and OUTPUT_DIR, then run.

Outputs:
  - analysis_unified_levels.csv
  - analysis_unified_levels_qc.csv
  - analysis_unified_levels_manifest.json  (includes input hashes + mapping choices)

Notes on evidence columns:
  - evidence_n_apo / evidence_n_holo are the DEFAULT evidence fields intended for plotting.
    * entry rows: evidence_n_* == n_* (counts for that entry)
    * ubs rows:   evidence_n_* == avg_n_* (mean counts per member entry)
  - n_*_sum always exists too (total across members; for entries == n_*)

Version: 2026-01-28
"""

from __future__ import annotations

import os
import json
import zipfile
import hashlib
import datetime
import platform
import socket
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------- USER SETTINGS (EDIT) --------------------------

# You can point either to a .csv or to a .zip containing the csv.
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

INPUT_SURVIVORS = str(cfg.FILTERED_SURVIVORS_ZIP)
INPUT_UBS_FILTERED = str(cfg.UBS_AGG_FILTERED_ZIP)
OUTPUT_DIR = str(cfg.UNIFIED_RUN_DIR)

OUT_UNIFIED_CSV = "analysis_unified_levels.csv"
OUT_QC_CSV = "analysis_unified_levels_qc.csv"
OUT_MANIFEST_JSON = "analysis_unified_levels_manifest.json"

# -------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes_iter(byte_iter) -> str:
    h = hashlib.sha256()
    for chunk in byte_iter:
        h.update(chunk)
    return h.hexdigest()


def read_csv_auto(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Reads a CSV either directly or from a ZIP (first member or the only .csv member).

    Returns:
      (df, meta)
    where meta includes:
      - input_path
      - member_name (if zipped)
      - sha256 (of the csv bytes)
    """
    path = str(path)
    meta = {"input_path": path, "member_name": None, "sha256": None}

    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            members = z.namelist()
            # Prefer a single CSV member if present
            csv_members = [m for m in members if m.lower().endswith(".csv")]
            member = csv_members[0] if csv_members else members[0]
            meta["member_name"] = member

            def _iter_bytes():
                with z.open(member) as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        yield chunk

            meta["sha256"] = sha256_bytes_iter(_iter_bytes())

            with z.open(member) as f:
                df = pd.read_csv(f)
        return df, meta

    # plain csv
    meta["sha256"] = sha256_file(path)
    df = pd.read_csv(path)
    return df, meta


def normalize_uniprots(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    toks = [t.strip() for t in str(s).split() if t.strip()]
    toks = sorted(set(toks))
    return " ".join(toks)


def uniprots_key(uniprots_str: str) -> str:
    toks = [t for t in str(uniprots_str).split() if t]
    return "_".join(toks)


def map_class(x) -> str:
    """
    Normalize labels to: apo / holo / undefined / tie
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "undefined"
    s = str(x).strip().lower()
    if s in ("apo", "a"):
        return "apo"
    if s in ("holo", "h"):
        return "holo"
    if s in ("undefined", "nan", "none", ""):
        return "undefined"
    if s in ("tie", "ambiguous"):
    #if s == "tie":
        return "tie"
    if "apo" in s:
        return "apo"
    if "holo" in s:
        return "holo"
    return s


def build_entry_table(df_surv: pd.DataFrame) -> pd.DataFrame:
    d = df_surv.copy()

    d["level"] = "entry"
    d["threshold"] = "UNCLUSTERED"
    d["threshold_detail"] = "UNCLUSTERED"

    d["unit_uid"] = d["job_name"].astype(str)
    d["unit_recipe"] = np.nan

    d["members"] = 1
    d["members_total"] = 1

    d["uniprots_str"] = d["UNPs"].apply(normalize_uniprots)
    d["uniprots_key"] = d["uniprots_str"].apply(uniprots_key)
    d["n_uniprots"] = d["uniprots_str"].apply(lambda x: len(x.split()) if x else 0)

    d["n_chains"] = d["chains3"].fillna("").astype(str).apply(
        lambda x: len([c for c in x.split("-") if c]) if x else 0
    )
    d["is_interface"] = d["n_uniprots"].gt(1) | d["n_chains"].gt(1)

    # numeric coercions
    for col in ["n_apo", "n_holo", "n_all", "#resis3", "model_pocket_pLDDT_avg"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # evidence (entry-level counts)
    d["n_apo_avg"] = d["n_apo"]
    d["n_holo_avg"] = d["n_holo"]
    d["n_apo_sum"] = d["n_apo"]
    d["n_holo_sum"] = d["n_holo"]

    def _ratio(h, a):
        if pd.isna(h) or pd.isna(a):
            return np.nan
        if a == 0 and h > 0:
            return np.inf
        if h == 0 and a > 0:
            return 0.0
        if a == 0 and h == 0:
            return np.nan
        return h / a

    d["ratio_n_holo_to_apo_avg"] = d.apply(lambda r: _ratio(r["n_holo_avg"], r["n_apo_avg"]), axis=1)
    d["ratio_n_holo_to_apo_sum"] = d["ratio_n_holo_to_apo_avg"]

    d["evidence_basis"] = "entry_counts"
    d["evidence_n_apo"] = d["n_apo_avg"]
    d["evidence_n_holo"] = d["n_holo_avg"]

    # probabilities
    d["pocket_p_apo"] = pd.to_numeric(d["pocket_avg_P(apo)"], errors="coerce")
    d["pocket_p_holo"] = pd.to_numeric(d["pocket_avg_P(holo)"], errors="coerce")
    d["chain_p_apo"] = pd.to_numeric(d["chain_avg_P(apo)"], errors="coerce")
    d["chain_p_holo"] = pd.to_numeric(d["chain_avg_P(holo)"], errors="coerce")

    # scores (entry-only)
    d["pocket_score"] = pd.to_numeric(d.get("pocket_overall_score"), errors="coerce")
    d["chain_score"] = pd.to_numeric(d.get("chain_overall_score"), errors="coerce")

    # densities
    d["pocket_density_combined"] = pd.to_numeric(d.get("pocket_avg_normalized_density_combined"), errors="coerce")
    d["pocket_density_strongest"] = pd.to_numeric(d.get("pocket_avg_normalized_density_strongest"), errors="coerce")
    d["pocket_density_pair"] = (d["pocket_density_combined"] + d["pocket_density_strongest"]) / 2

    d["chain_density_combined"] = pd.to_numeric(d.get("chain_avg_normalized_density_combined"), errors="coerce")
    d["chain_density_strongest"] = pd.to_numeric(d.get("chain_avg_normalized_density_strongest"), errors="coerce")
    d["chain_density_pair"] = (d["chain_density_combined"] + d["chain_density_strongest"]) / 2

    # overlaps
    d["pocket_overlap_overall"] = pd.to_numeric(d.get("pocket_overall_overlap"), errors="coerce")
    d["pocket_overlap_mode"] = pd.to_numeric(d.get("pocket_overall_mode_overlap"), errors="coerce")
    d["pocket_overlap_pair"] = (d["pocket_overlap_overall"] + d["pocket_overlap_mode"]) / 2

    d["chain_overlap_overall"] = pd.to_numeric(d.get("chain_overall_overlap"), errors="coerce")
    d["chain_overlap_mode"] = pd.to_numeric(d.get("chain_overall_mode_overlap"), errors="coerce")
    d["chain_overlap_pair"] = (d["chain_overlap_overall"] + d["chain_overlap_mode"]) / 2

    # classifications
    d["pocket_class"] = d["pocket_overall_classification"].apply(map_class)
    d["chain_class"] = d["chain_overall_classification"].apply(map_class)

    # votes (derived 1-of-K so plotting code can treat entry+UBS uniformly)
    d["votes_apo_pocket"] = (d["pocket_class"] == "apo").astype(int)
    d["votes_holo_pocket"] = (d["pocket_class"] == "holo").astype(int)
    d["votes_undef_pocket"] = (d["pocket_class"].isin(["undefined", "tie"])).astype(int)

    d["votes_apo_chain"] = (d["chain_class"] == "apo").astype(int)
    d["votes_holo_chain"] = (d["chain_class"] == "holo").astype(int)
    d["votes_undef_chain"] = (d["chain_class"].isin(["undefined", "tie"])).astype(int)

    # representative job is the entry itself
    d["rep_job"] = d["job_name"].astype(str)
    d["rep_job_overlap"] = np.nan
    d["rep_job_resis3"] = pd.to_numeric(d["#resis3"], errors="coerce")
    d["job_names"] = d["job_name"].astype(str)

    d["resis3_unit"] = pd.to_numeric(d["#resis3"], errors="coerce")
    d["model_pocket_plddt"] = pd.to_numeric(d["model_pocket_pLDDT_avg"], errors="coerce")

    # unified site-size fields (entry: pocket size)
    d["site_size_unit"] = pd.to_numeric(d["#resis3"], errors="coerce")
    d["site_size_intersection"] = pd.to_numeric(d["#resis3"], errors="coerce")

    # Keep a stable set of canonical columns (others can be added later)
    keep_cols = [
        "level", "threshold", "threshold_detail",
        "unit_uid", "unit_recipe",
        "uniprots_str", "uniprots_key", "n_uniprots",
        "is_interface",
        "members", "members_total",
        "n_chains", "resis3_unit",
        "site_size_unit", "site_size_intersection", # New
        "model_pocket_plddt",
        "evidence_basis", "evidence_n_apo", "evidence_n_holo",
        "n_apo_avg", "n_holo_avg", "n_apo_sum", "n_holo_sum",
        "ratio_n_holo_to_apo_avg", "ratio_n_holo_to_apo_sum",
        "pocket_p_apo", "pocket_p_holo", "pocket_score",
        "pocket_density_combined", "pocket_density_strongest", "pocket_density_pair",
        "pocket_overlap_overall", "pocket_overlap_mode", "pocket_overlap_pair",
        "pocket_class", "votes_apo_pocket", "votes_holo_pocket", "votes_undef_pocket",
        "chain_p_apo", "chain_p_holo", "chain_score",
        "chain_density_combined", "chain_density_strongest", "chain_density_pair",
        "chain_overlap_overall", "chain_overlap_mode", "chain_overlap_pair",
        "chain_class", "votes_apo_chain", "votes_holo_chain", "votes_undef_chain",
        "rep_job", "rep_job_overlap", "rep_job_resis3",
        "job_names",
        "query_path",
    ]
    return d[keep_cols]


def build_ubs_table(df_ubs: pd.DataFrame) -> pd.DataFrame:
    d = df_ubs.copy()

    d["level"] = "ubs"
    d["unit_uid"] = d["site_uid"].astype(str)
    d["unit_recipe"] = d.get("site_recipe")

    d["uniprots_str"] = d["site_uniprots"].apply(normalize_uniprots)
    d["uniprots_key"] = d["uniprots_str"].apply(uniprots_key)
    d["n_uniprots"] = pd.to_numeric(d.get("n_uniprots"), errors="coerce")

    d["is_interface"] = d["is_interface"].astype(bool)

    d["members"] = pd.to_numeric(d.get("n_entries"), errors="coerce")
    d["members_total"] = pd.to_numeric(d.get("n_entries_total", d.get("n_entries")), errors="coerce")

    d["n_chains"] = pd.to_numeric(d.get("max_n_chains3"), errors="coerce")
    d["resis3_unit"] = pd.to_numeric(d.get("max_n_resis3"), errors="coerce")

    # unified site-size fields (UBS: union/intersection sizes)
    d["site_size_unit"] = pd.to_numeric(d.get("n_union"), errors="coerce")
    d["site_size_intersection"] = pd.to_numeric(d.get("n_intersection"), errors="coerce")

    d["model_pocket_plddt"] = pd.to_numeric(d.get("model_pocket_pLDDT_avg_mean"), errors="coerce")

    # evidence (UBS member averages by default; sums retained too)
    d["n_apo_avg"] = pd.to_numeric(d.get("avg_n_apo"), errors="coerce")
    d["n_holo_avg"] = pd.to_numeric(d.get("avg_n_holo"), errors="coerce")
    d["n_apo_sum"] = pd.to_numeric(d.get("sum_n_apo"), errors="coerce")
    d["n_holo_sum"] = pd.to_numeric(d.get("sum_n_holo"), errors="coerce")

    d["ratio_n_holo_to_apo_avg"] = pd.to_numeric(d.get("ratio_avg_n_holo_to_apo"), errors="coerce")
    d["ratio_n_holo_to_apo_sum"] = pd.to_numeric(d.get("ratio_n_holo_to_apo"), errors="coerce")

    d["evidence_basis"] = "ubs_member_avg"
    d["evidence_n_apo"] = d["n_apo_avg"]
    d["evidence_n_holo"] = d["n_holo_avg"]

    # probabilities
    d["pocket_p_apo"] = pd.to_numeric(d.get("avg_P(apo)_mean_pocket"), errors="coerce")
    d["pocket_p_holo"] = pd.to_numeric(d.get("avg_P(holo)_mean_pocket"), errors="coerce")
    d["chain_p_apo"] = pd.to_numeric(d.get("avg_P(apo)_mean_chain"), errors="coerce")
    d["chain_p_holo"] = pd.to_numeric(d.get("avg_P(holo)_mean_chain"), errors="coerce")

    # scores not present at UBS level
    d["pocket_score"] = np.nan
    d["chain_score"] = np.nan

    # densities
    d["pocket_density_combined"] = pd.to_numeric(d.get("avg_normalized_density_combined_mean_pocket"), errors="coerce")
    d["pocket_density_strongest"] = pd.to_numeric(d.get("avg_normalized_density_strongest_mean_pocket"), errors="coerce")
    d["pocket_density_pair"] = pd.to_numeric(d.get("pair_density_mean_pocket"), errors="coerce")

    d["chain_density_combined"] = pd.to_numeric(d.get("avg_normalized_density_combined_mean_chain"), errors="coerce")
    d["chain_density_strongest"] = pd.to_numeric(d.get("avg_normalized_density_strongest_mean_chain"), errors="coerce")
    d["chain_density_pair"] = pd.to_numeric(d.get("pair_density_mean_chain"), errors="coerce")

    # overlaps
    d["pocket_overlap_overall"] = pd.to_numeric(d.get("overall_overlap_mean_pocket"), errors="coerce")
    d["pocket_overlap_mode"] = pd.to_numeric(d.get("mode_overlap_mean_pocket"), errors="coerce")
    d["pocket_overlap_pair"] = pd.to_numeric(d.get("pair_overlap_mean_pocket"), errors="coerce")

    d["chain_overlap_overall"] = pd.to_numeric(d.get("overall_overlap_mean_chain"), errors="coerce")
    d["chain_overlap_mode"] = pd.to_numeric(d.get("mode_overlap_mean_chain"), errors="coerce")
    d["chain_overlap_pair"] = pd.to_numeric(d.get("pair_overlap_mean_chain"), errors="coerce")

    # classifications (UBS default: majority_vote_*; normalized) # change to probability avg
    d["pocket_class"] = d.get("highest_p_mean_pocket").apply(map_class) #  majority_vote_pocket
    d["chain_class"] = d.get("highest_p_mean_chain").apply(map_class) # majority_vote_chain

    # votes (UBS provided)
    d["votes_apo_pocket"] = pd.to_numeric(d.get("votes_n_apo_pocket"), errors="coerce")
    d["votes_holo_pocket"] = pd.to_numeric(d.get("votes_n_holo_pocket"), errors="coerce")
    d["votes_undef_pocket"] = pd.to_numeric(d.get("votes_n_undefined_pocket"), errors="coerce")

    d["votes_apo_chain"] = pd.to_numeric(d.get("votes_n_apo_chain"), errors="coerce")
    d["votes_holo_chain"] = pd.to_numeric(d.get("votes_n_holo_chain"), errors="coerce")
    d["votes_undef_chain"] = pd.to_numeric(d.get("votes_n_undefined_chain"), errors="coerce")

    # representative job info
    d["rep_job"] = d.get("representative_job")
    d["rep_job_overlap"] = pd.to_numeric(d.get("representative_job_overlap"), errors="coerce")
    d["rep_job_resis3"] = pd.to_numeric(d.get("representative_job_resis3"), errors="coerce")
    d["job_names"] = d.get("job_names")

    # not applicable
    d["query_path"] = np.nan

    keep_cols = [
        "level", "threshold", "threshold_detail",
        "unit_uid", "unit_recipe",
        "uniprots_str", "uniprots_key", "n_uniprots",
        "is_interface",
        "members", "members_total",
        "n_chains", "resis3_unit", 
        "site_size_unit", "site_size_intersection", # New
        "model_pocket_plddt",
        "evidence_basis", "evidence_n_apo", "evidence_n_holo",
        "n_apo_avg", "n_holo_avg", "n_apo_sum", "n_holo_sum",
        "ratio_n_holo_to_apo_avg", "ratio_n_holo_to_apo_sum",
        "pocket_p_apo", "pocket_p_holo", "pocket_score",
        "pocket_density_combined", "pocket_density_strongest", "pocket_density_pair",
        "pocket_overlap_overall", "pocket_overlap_mode", "pocket_overlap_pair",
        "pocket_class", "votes_apo_pocket", "votes_holo_pocket", "votes_undef_pocket",
        "chain_p_apo", "chain_p_holo", "chain_score",
        "chain_density_combined", "chain_density_strongest", "chain_density_pair",
        "chain_overlap_overall", "chain_overlap_mode", "chain_overlap_pair",
        "chain_class", "votes_apo_chain", "votes_holo_chain", "votes_undef_chain",
        "rep_job", "rep_job_overlap", "rep_job_resis3",
        "job_names",
        "query_path",
    ]
    return d[keep_cols]


def unique_accessions(series: pd.Series) -> int:
    acc = set()
    for s in series.dropna().astype(str):
        for tok in s.split():
            tok = tok.strip()
            if tok:
                acc.add(tok)
    return len(acc)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read inputs ----
    df_surv, meta_surv = read_csv_auto(INPUT_SURVIVORS)
    df_ubs, meta_ubs = read_csv_auto(INPUT_UBS_FILTERED)

    # ---- Build unified ----
    entry_u = build_entry_table(df_surv)
    ubs_u = build_ubs_table(df_ubs)

    # Align columns + concatenate
    all_cols = sorted(set(entry_u.columns) | set(ubs_u.columns))
    df_unified = pd.concat(
        [entry_u.reindex(columns=all_cols), ubs_u.reindex(columns=all_cols)],
        ignore_index=True
    )

    # ---- Validation: site_size_unit summary ----
    if "site_size_unit" in df_unified.columns:
        chk = (
            df_unified
            .groupby(["threshold", "level"], dropna=False)["site_size_unit"]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
            .sort_values(["level", "threshold"])
        )
        print("\n[QC] site_size_unit by threshold/level:")
        print(chk.to_string(index=False))
    else:
        print("\n[QC] site_size_unit missing from df_unified (unexpected).")

    out_csv = out_dir / OUT_UNIFIED_CSV
    df_unified.to_csv(out_csv, index=False)

    # ---- QC ----
    qc_rows = []
    for (level, threshold), sub in df_unified.groupby(["level", "threshold"]):
        qc_rows.append({
            "level": level,
            "threshold": threshold,
            "rows": int(len(sub)),
            "unique_uniprot_accessions": int(unique_accessions(sub["uniprots_str"])),
            "unique_uniprot_groups": int(sub["uniprots_str"].nunique(dropna=True)),
        })
    qc_df = pd.DataFrame(qc_rows).sort_values(["level", "threshold"])
    (out_dir / OUT_QC_CSV).write_text(qc_df.to_csv(index=False), encoding="utf-8")

    # ---- Manifest ----
    manifest = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": {
            "filtered_survivors": meta_surv,
            "ubs_aggregated_filtered": meta_ubs,
        },
        "outputs": {
            "analysis_unified_levels.csv": {
                "path": str(out_csv),
                "sha256": sha256_file(str(out_csv)),
                "rows": int(len(df_unified)),
                "columns": df_unified.columns.tolist(),
            },
            "analysis_unified_levels_qc.csv": {
                "path": str(out_dir / OUT_QC_CSV),
                "rows": int(len(qc_df)),
            }
        },
        "evidence_choice": {
            "default_evidence_basis": "Per-analysis-unit/member average evidence so entry and UBS are comparable.",
            "entry_basis": "n_apo/n_holo",
            "ubs_basis": "avg_n_apo/avg_n_holo",
            "notes": "Sum evidence retained in n_apo_sum/n_holo_sum for plots that want total evidence per UBS.",
        },
        "env": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
        },
    }
    with open(out_dir / OUT_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_dir / OUT_QC_CSV}")
    print(f"[OK] wrote: {out_dir / OUT_MANIFEST_JSON}")


if __name__ == "__main__":
    main()
