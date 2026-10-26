# -*- coding: utf-8 -*-
"""
Pool per-tool LBS success tables into the generality panel.
===========================================================

Reads one or more per-tool success tables (`lbs_<tool>_st100_merged_per_pocket.csv`
from score/compute_lbs_success.py) — optionally plus the P2Rank merged table as
the reference — and computes, per tool:
  * TNS by class (AF-Apo vs AF-Holo) + the holo-apo gradient (the C4 claim);
  * TNS across classification-probability and separation bins;
  * TNS across evidence-ratio quintiles.
The pooled long table drives one figure showing the SAME gradient in every tool.

For the size-controlled / controlled-pairs version, reuse the P2Rank script
directly on each per-tool table (they share the schema):

    python 04_p2rank/analysis/analyze_p2rank_controlled_pairs.py \
        --p2rank_table  <lbs_<tool>_...csv>  --success_col success_tns

Base env (pandas + numpy). See analysis/README.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

SUCCESS_COL = "success_tns"
LEVELS = ["very_low", "low", "moderate", "high", "very_high"]


def tns_by_class(df: pd.DataFrame, tool: str) -> pd.DataFrame:
    rows = []
    for cls in ["apo", "holo"]:
        v = pd.to_numeric(df.loc[df["pocket_class"] == cls, SUCCESS_COL], errors="coerce")
        rows.append({"tool": tool, "pocket_class": cls, "n": int(v.notna().sum()),
                     "tns_percent": None if v.notna().sum() == 0 else round(float(v.mean()) * 100, 2)})
    apo = next((r["tns_percent"] for r in rows if r["pocket_class"] == "apo"), None)
    holo = next((r["tns_percent"] for r in rows if r["pocket_class"] == "holo"), None)
    gap = None if (apo is None or holo is None) else round(holo - apo, 2)
    for r in rows:
        r["holo_minus_apo_pts"] = gap
    return pd.DataFrame(rows)


def tns_by_bin(df: pd.DataFrame, tool: str, bin_col: str) -> pd.DataFrame:
    if bin_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for cls in ["apo", "holo"]:
        sub = df[df["pocket_class"] == cls]
        for lab, grp in sub.groupby(bin_col, dropna=True):
            v = pd.to_numeric(grp[SUCCESS_COL], errors="coerce")
            rows.append({"tool": tool, "bin_type": bin_col, "bin": str(lab), "pocket_class": cls,
                         "n": int(v.notna().sum()),
                         "tns_percent": None if v.notna().sum() == 0 else round(float(v.mean()) * 100, 2)})
    return pd.DataFrame(rows)


def tns_by_prob_quintile(df: pd.DataFrame, tool: str) -> pd.DataFrame:
    col = "log2_ratio_holo_to_apo_avg_plus1"
    if col not in df.columns:
        return pd.DataFrame()
    vals = pd.to_numeric(df[col], errors="coerce")
    ok = vals.notna() & pd.to_numeric(df[SUCCESS_COL], errors="coerce").notna()
    if ok.sum() < 50:
        return pd.DataFrame()
    q = pd.qcut(vals[ok], 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
    tmp = df.loc[ok, [SUCCESS_COL]].copy()
    tmp["q"] = q.astype(str)
    rows = []
    for lab, grp in tmp.groupby("q"):
        v = pd.to_numeric(grp[SUCCESS_COL], errors="coerce")
        rows.append({"tool": tool, "bin_type": "evidence_ratio_quintile", "bin": lab,
                     "n": int(v.notna().sum()), "tns_percent": round(float(v.mean()) * 100, 2)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pool per-tool LBS success tables into the generality panel.")
    p.add_argument("--tool_tables", type=Path, nargs="+", required=True,
                   help="One or more lbs_<tool>_...csv success tables.")
    p.add_argument("--p2rank_table", type=Path, default=cfg.P2RANK_MERGED_PER_POCKET_CSV,
                   help="P2Rank merged per-pocket table (reference); pass 'none' to skip.")
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--out_dir", type=Path, default=cfg.LBS_ANALYSIS_DIR / "tables")
    return p.parse_args()


def _load(path: Path, threshold: str, tool_name: str | None) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    if "threshold" in df.columns:
        df = df[df["threshold"] == threshold].copy()
    df["pocket_class"] = df["pocket_class"].astype(str).str.lower()
    tool = tool_name or (df["tool"].iloc[0] if "tool" in df.columns and len(df) else path.stem)
    return str(tool), df


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    for t in args.tool_tables:
        tables.append(_load(t, args.threshold, None))
    if str(args.p2rank_table).lower() != "none" and Path(args.p2rank_table).is_file():
        tables.append(_load(Path(args.p2rank_table), args.threshold, "p2rank"))

    byclass, bybins = [], []
    for tool, df in tables:
        byclass.append(tns_by_class(df, tool))
        for bc in ["prob_level_db", "separation_level_db"]:
            bybins.append(tns_by_bin(df, tool, bc))
        bybins.append(tns_by_prob_quintile(df, tool))

    byclass_df = pd.concat(byclass, ignore_index=True) if byclass else pd.DataFrame()
    bybins_df = pd.concat([b for b in bybins if not b.empty], ignore_index=True) if any(not b.empty for b in bybins) else pd.DataFrame()
    byclass_df.to_csv(args.out_dir / "lbs_generality_tns_by_class.csv", index=False)
    bybins_df.to_csv(args.out_dir / "lbs_generality_tns_by_bins.csv", index=False)

    manifest = {
        "tools": [t for t, _ in tables],
        "threshold": args.threshold,
        "note": "For size-controlled gradients reuse "
                "04_p2rank/analysis/analyze_p2rank_controlled_pairs.py --p2rank_table <tool table>.",
    }
    with (args.out_dir / "lbs_generality_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[SAVE] {args.out_dir / 'lbs_generality_tns_by_class.csv'}")
    if not byclass_df.empty:
        print(byclass_df.to_string(index=False))


if __name__ == "__main__":
    main()
