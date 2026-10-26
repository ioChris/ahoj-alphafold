# -*- coding: utf-8 -*-
"""
Tool-agnostic DCA / TNS scorer for the LBS-generality stage (Analysis 7, T3-2nd).
=================================================================================

The REUSABLE HEART of 06_lbs_generality. It takes *normalized* ranked pocket
predictions from ANY binding-site predictor and computes a single, tool-agnostic
success metric identical to the P2Rank stage's, so the holo>apo detection gradient
(claim C4) can be recomputed per tool and pooled. Its output CSV carries the same
success columns + class / probability / evidence / size metadata as
`p2rank_st100_merged_per_pocket.csv`, so the existing analysis is reused verbatim:

    04_p2rank/analysis/analyze_p2rank_controlled_pairs.py --p2rank_table <this.csv>

THE METRIC (why it is defined this way)  -- verified against P2Rank on disk
---------------------------------------------------------------------------
For each pocket the ground truth is the set of **union binding-site residue heavy
atoms** on the AF model (the SAME `union_chain_resi` definition the P2Rank stage
uses). For a predicted pocket with centre c:

    DCA(pocket) = min over ground-truth atoms a of  || c - a ||           (Angstrom)
    dca_rank    = rank (tool's own order) of the first pocket with DCA <= cutoff
    closest_pocket_dist = min over predicted pockets of DCA
    success_topn = dca_rank > 0
    success_top1 = dca_rank == 1
    success_tns  = dca_rank > 0  AND  dca_rank <= n_unp_pockets      (MAIN metric)
    success_dca_XA = closest_pocket_dist <= X

This centre-to-nearest-site-atom distance reproduces P2Rank's own
`closest_pocket_dist` to < 1e-4 Angstrom on all shared ST_100 pockets, so it is a
faithful, tool-independent yardstick.

GROUND TRUTH CONFIRMED FROM P2Rank's OWN EVAL CONFIG (`params.txt` / `summary.csv`):
the P2Rank ground-truth "ligand" was the pocket RESIDUE set, NOT a single centre
point -- `ligSize = 97.2 atoms/site` (our union-residue heavy-atom store averages
98.1, a match), `min_ligand_atoms = 5` (a 1-atom point would be rejected), and
`site_eval_center_method = explicit` (an explicit centre WAS supplied -- that is the
"CENTER_explicit" in the eval dir name -- but it is only the reference centre, not
the DCA target). A single-centre-point ground truth would have measured ~31% TNS
(apo 22 / holo 36) vs the actual ~75% DCA4 / native apo 63 / holo 84.

NOTE: P2Rank's *native* `dca4rank` is slightly LOOSER than this scorer because
P2Rank additionally treats the site's SAS surface points as ground-truth atoms
(`site_eval_sas_pts_as_atoms = true`), extending the target cloud ~1-4 A outward.
Re-scoring P2Rank's own centres with THIS (heavy-atom) metric therefore gives a
stricter TNS than the manuscript's native P2Rank TNS (2,500-pocket check: harmonized
apo 44% / holo 61% vs native apo 63% / holo 83%; the holo>apo gradient persists).
For the generality panel, re-score P2Rank with this same scorer (one yardstick for
every tool) and report the manuscript's native P2Rank TNS alongside as the reference.
To reproduce native P2Rank *exactly*, build the ground-truth store from SAS points
rather than heavy atoms (a documented option; needs a surface sampler).

GROUND-TRUTH ATOMS
------------------
Pass `--ground_truth_npz` produced by `dataset/build_ground_truth_atoms.py`
(gemmi; run once in the analysis env). The scorer itself needs only pandas+numpy
(base env). Without a store, `--center_fallback` uses the single
`union_center_geom` point as a 1-atom ground truth -- a coarse approximation that
does NOT match P2Rank; for quick smoke tests only.

INPUT predictions CSV (schema: ../schemas/predictions.schema.md): one row per
(pocket x predicted-pocket): site_uid, pocket_rank (1-based), center_x/y/z, [score].
Pockets with no predicted rows are still scored (as failures).

Self-test:  python compute_lbs_success.py --selftest
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = None
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        _REPO_ROOT = _root
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Verbal-confidence binning helpers, vendored in the P2Rank analysis dir, so
# prob_level_db / separation_level_db match the P2Rank table exactly.
if _REPO_ROOT is not None:
    _cl_dir = _REPO_ROOT / "04_p2rank" / "analysis"
    if _cl_dir.is_dir() and str(_cl_dir) not in _sys.path:
        _sys.path.insert(0, str(_cl_dir))
try:
    from confidence_labeling import probability_level, support_level  # type: ignore
except Exception:  # pragma: no cover
    probability_level = None
    support_level = None


DCA_CUTOFFS = [4.0, 5.0, 6.0, 8.0, 10.0]     # 1:1 with analyze_p2rank_ubs_run4.CUTOFFS
PRIMARY_CUTOFF = 4.0
GT_CENTER_COLS = ("union_center_geom_x", "union_center_geom_y", "union_center_geom_z")

GEOM_META_COLS = [
    "site_uid", "site_uniprot", "threshold",
    "pocket_class", "pocket_p_apo", "pocket_p_holo", "pocket_score",
    "pocket_density_pair", "pocket_overlap_pair", "pocket_separation_pair",
    "model_pocket_plddt", "n_apo_avg", "n_holo_avg",
    "union_rg", "union_n_residues_found",
    "n_unp_pockets", "n_unp_pockets_multichain",
    "rep_job", "afdb_filename",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce a tool-agnostic DCA/TNS success table for any predictor.")
    p.add_argument("--predictions_csv", type=Path, required=False,
                   help="Normalized ranked-pocket predictions (schema: ../schemas/predictions.schema.md).")
    p.add_argument("--geometry_csv", type=Path, default=cfg.AFDB_ST100_DATASET_GEOM_CSV)
    p.add_argument("--ground_truth_npz", type=Path, default=None,
                   help="Per-site ground-truth atom store from dataset/build_ground_truth_atoms.py.")
    p.add_argument("--center_fallback", action="store_true",
                   help="Use union_center_geom as a 1-atom ground truth (coarse; does NOT match P2Rank).")
    p.add_argument("--tool", default="tool")
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--dca_cutoff", type=float, default=PRIMARY_CUTOFF,
                   help="Fixed DCA cutoff (Angstrom) used when --tolerance_col is not set.")
    p.add_argument("--tolerance_col", default=None,
                   help="Geometry column (e.g. union_rg) giving a per-site size; when set, the per-site "
                        "DCA cutoff = --tolerance_scale * <this column>, a size-adaptive tolerance. "
                        "Sites with a missing/non-finite value become non-scorable. Absolute DCA-sweep "
                        "columns (success_dca_XA) stay fixed. Falls back to --dca_cutoff when unset.")
    p.add_argument("--tolerance_scale", type=float, default=None,
                   help="Scale factor k for the per-site cutoff (k * <tolerance_col>); e.g. 0.5/1.0/2.0.")
    p.add_argument("--tolerance_floor", type=float, default=None,
                   help="Lower bound (Angstrom) on the per-site cutoff: cutoff = max(floor, k*<col>). "
                        "Keeps small sites from collapsing to a sub-Angstrom tolerance because of a tiny Rg. "
                        "Only applies in tolerance mode.")
    p.add_argument("--tolerance_cap", type=float, default=None,
                   help="Upper bound (Angstrom) on the per-site cutoff: cutoff = min(cap, ...). Only in tolerance mode.")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--holo_plus_one", type=float, default=1.0)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def load_ground_truth_npz(path: Path) -> dict:
    """Load the ragged per-site ground-truth atom store.

    Expected arrays: site_uids (str), starts (int), counts (int), coords (N,3).
    Returns {site_uid: (M,3) float array}.
    """
    z = np.load(path, allow_pickle=True)
    uids = [str(u) for u in z["site_uids"]]
    starts, counts, coords = z["starts"], z["counts"], z["coords"]
    out = {}
    for uid, s, c in zip(uids, starts, counts):
        out[uid] = coords[int(s):int(s) + int(c)]
    return out


def compute_site_success(site_rows: pd.DataFrame, gt_atoms: np.ndarray, *, primary_cutoff: float) -> dict:
    """Per-pocket success from that pocket's ranked predicted pockets vs gt atoms.

    ``primary_cutoff`` is the DCA cutoff for THIS site (Angstrom). Under a fixed
    cutoff it is the same scalar for every site; under a size-adaptive tolerance
    (``score(..., tolerance_col=..., tolerance_scale=k)``) it is ``k * site_size``.
    A non-finite ``primary_cutoff`` (e.g. site with missing size) makes the
    rank-based metrics (dca_rank / success_top1 / success_topn / success_tns)
    non-scorable (NaN); the absolute DCA-sweep columns are unaffected.
    """
    cutoff_finite = bool(np.isfinite(primary_cutoff))
    base = {"predicted_pockets_total": int(len(site_rows)),
            "dca_cutoff_used": float(primary_cutoff) if cutoff_finite else np.nan}
    if gt_atoms is None or len(gt_atoms) == 0:
        # No ground-truth atoms -> cannot score this pocket.
        base.update({"dca_rank": np.nan, "closest_pocket_dist": np.nan,
                     "success_top1": np.nan, "success_topn": np.nan})
        for c in DCA_CUTOFFS:
            base[f"success_dca_{int(c)}A"] = np.nan
        return base
    if site_rows.empty:
        # No predicted pocket = genuine miss when we can score; non-scorable if the
        # per-site cutoff is undefined (missing size). Absolute DCA sweep = miss either way.
        base.update({"dca_rank": (-1 if cutoff_finite else np.nan), "closest_pocket_dist": np.nan,
                     "success_top1": (False if cutoff_finite else np.nan),
                     "success_topn": (False if cutoff_finite else np.nan)})
        for c in DCA_CUTOFFS:
            base[f"success_dca_{int(c)}A"] = False
        return base

    ranks = pd.to_numeric(site_rows["pocket_rank"], errors="coerce").to_numpy(dtype=float)
    centers = site_rows[["center_x", "center_y", "center_z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # per-pocket DCA = min distance from the pocket centre to any ground-truth atom
    gt = np.asarray(gt_atoms, dtype=float)
    dca = np.full(len(centers), np.nan)
    for i, c in enumerate(centers):
        if np.all(np.isfinite(c)):
            dca[i] = np.sqrt(((gt - c) ** 2).sum(axis=1)).min()

    finite = np.isfinite(dca)
    closest = float(np.nanmin(dca)) if finite.any() else np.nan
    if cutoff_finite:
        within = finite & (dca <= primary_cutoff)
        dca_rank = int(np.nanmin(ranks[within])) if within.any() else -1
        base.update({
            "dca_rank": dca_rank,
            "closest_pocket_dist": closest,
            "success_top1": bool(dca_rank == 1),
            "success_topn": bool(dca_rank > 0),
        })
    else:
        # Undefined per-site cutoff -> rank-based metrics non-scorable (NaN).
        base.update({"dca_rank": np.nan, "closest_pocket_dist": closest,
                     "success_top1": np.nan, "success_topn": np.nan})
    for c in DCA_CUTOFFS:
        base[f"success_dca_{int(c)}A"] = bool(finite.any() and closest <= c)
    return base


def score(predictions: pd.DataFrame, geometry: pd.DataFrame, gt_store: dict, *,
          tool: str, threshold: str, primary_cutoff: float, holo_plus_one: float,
          tolerance_col: str | None = None, tolerance_scale: float | None = None,
          tolerance_floor: float | None = None, tolerance_cap: float | None = None) -> pd.DataFrame:
    geom = geometry[geometry["threshold"] == threshold].copy()
    geom["site_uid"] = geom["site_uid"].astype(str).str.strip()
    geom = geom.drop_duplicates(subset=["site_uid"])

    # Size-adaptive tolerance: per-site cutoff = tolerance_scale * geom[tolerance_col],
    # optionally floored (max) / capped (min) so small/large sites don't collapse or blow up.
    # Sites with a missing/non-finite value get a NaN cutoff -> non-scorable (excluded,
    # identically for every tool; np.maximum/np.minimum propagate NaN so a floor never rescues
    # a missing-size site). When unset, every site uses the fixed primary_cutoff.
    per_site_cutoff = None
    if tolerance_col is not None and tolerance_scale is not None:
        if tolerance_col not in geom.columns:
            raise SystemExit(f"[ERR] --tolerance_col '{tolerance_col}' not found in geometry columns.")
        tv = pd.to_numeric(geom[tolerance_col], errors="coerce").to_numpy(dtype=float) * float(tolerance_scale)
        if tolerance_floor is not None:
            tv = np.maximum(tv, float(tolerance_floor))
        if tolerance_cap is not None:
            tv = np.minimum(tv, float(tolerance_cap))
        per_site_cutoff = dict(zip(geom["site_uid"].tolist(), tv.tolist()))

    preds = predictions.copy()
    if not preds.empty:
        preds["site_uid"] = preds["site_uid"].astype(str).str.strip()
        preds = preds[preds["site_uid"].isin(set(geom["site_uid"]))].copy()
    pred_by_site = {uid: grp for uid, grp in preds.groupby("site_uid", sort=False)} if not preds.empty else {}
    empty = preds.iloc[0:0]

    rows = []
    for uid in geom["site_uid"]:
        gt = gt_store.get(uid)
        cutoff = per_site_cutoff.get(uid, np.nan) if per_site_cutoff is not None else primary_cutoff
        succ = compute_site_success(pred_by_site.get(uid, empty), gt, primary_cutoff=cutoff)
        succ["site_uid"] = uid
        rows.append(succ)
    succ_df = pd.DataFrame(rows)

    meta = geom[[c for c in GEOM_META_COLS if c in geom.columns]].copy()
    merged = succ_df.merge(meta, on="site_uid", how="left", validate="one_to_one")

    merged["dca4rank"] = merged["dca_rank"]  # P2Rank column-name alias
    n_unp = pd.to_numeric(merged.get("n_unp_pockets"), errors="coerce")
    n_unp_mc = pd.to_numeric(merged.get("n_unp_pockets_multichain"), errors="coerce")
    dr = pd.to_numeric(merged["dca_rank"], errors="coerce")
    scorable = dr.notna()
    merged["success_tns"] = np.where(scorable & n_unp.notna() & (n_unp >= 1),
                                     (dr > 0) & (dr <= n_unp), np.nan)
    merged["success_tnm"] = np.where(scorable & n_unp_mc.notna() & (n_unp_mc >= 1),
                                     (dr > 0) & (dr <= n_unp_mc), np.nan)
    merged["success_dca4_sites"] = pd.to_numeric(merged["success_top1"], errors="coerce")

    merged["winning_p"] = merged[["pocket_p_apo", "pocket_p_holo"]].max(axis=1)
    merged["ratio_holo_to_apo_avg_plus1"] = (
        (pd.to_numeric(merged["n_holo_avg"], errors="coerce") + float(holo_plus_one))
        / pd.to_numeric(merged["n_apo_avg"], errors="coerce").replace(0, np.nan)
    )
    merged["log2_ratio_holo_to_apo_avg_plus1"] = np.log2(merged["ratio_holo_to_apo_avg_plus1"])
    merged["rg"] = pd.to_numeric(merged.get("union_rg"), errors="coerce")

    if probability_level is not None and support_level is not None:
        merged["prob_level_db"] = merged["winning_p"].map(lambda v: probability_level(v, merge=False))
        merged["prob_level"] = merged["winning_p"].map(lambda v: probability_level(v, merge=True))
        merged["separation_level_db"] = merged["pocket_separation_pair"].map(lambda v: support_level(v, merge=False))
        merged["density_level_db"] = merged["pocket_density_pair"].map(lambda v: support_level(v, merge=False))

    merged["pocket_class"] = merged["pocket_class"].fillna("unknown").astype(str).str.lower()
    merged["tool"] = tool
    merged["unit_uid"] = merged["site_uid"]
    return merged


def _summarize(merged: pd.DataFrame) -> dict:
    def rate(col, cls=None):
        d = merged if cls is None else merged[merged["pocket_class"] == cls]
        v = pd.to_numeric(d[col], errors="coerce")
        return None if v.notna().sum() == 0 else round(float(v.mean()) * 100.0, 2)
    return {
        "n_pockets": int(len(merged)),
        "n_scorable": int(pd.to_numeric(merged["dca_rank"], errors="coerce").notna().sum()),
        "n_with_prediction": int((merged["predicted_pockets_total"] > 0).sum()),
        "tns_percent_all": rate("success_tns"),
        "tns_percent_apo": rate("success_tns", "apo"),
        "tns_percent_holo": rate("success_tns", "holo"),
        "topn_percent_all": rate("success_topn"),
    }


def main() -> None:
    args = parse_args()
    if args.selftest:
        _selftest()
        return
    if args.predictions_csv is None:
        raise SystemExit("[ERR] --predictions_csv is required (or use --selftest).")

    out_dir = args.out_dir or (cfg.WORK_DIR / "lbs_generality" / args.tool / "analysis_v1" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions_csv, low_memory=False)
    geometry = pd.read_csv(args.geometry_csv, low_memory=False)

    if args.ground_truth_npz is not None:
        gt_store = load_ground_truth_npz(args.ground_truth_npz)
    elif args.center_fallback:
        g = geometry[geometry["threshold"] == args.threshold]
        gt_store = {str(r.site_uid).strip(): np.array([[r[GT_CENTER_COLS[0]], r[GT_CENTER_COLS[1]], r[GT_CENTER_COLS[2]]]], dtype=float)
                    for r in g.itertuples(index=False)}
        print("[WARN] --center_fallback: single-point ground truth; does NOT match P2Rank. Smoke tests only.")
    else:
        raise SystemExit("[ERR] provide --ground_truth_npz (recommended) or --center_fallback.")

    tol_mode = args.tolerance_col is not None and args.tolerance_scale is not None
    merged = score(predictions, geometry, gt_store, tool=args.tool, threshold=args.threshold,
                   primary_cutoff=args.dca_cutoff, holo_plus_one=args.holo_plus_one,
                   tolerance_col=args.tolerance_col, tolerance_scale=args.tolerance_scale,
                   tolerance_floor=args.tolerance_floor, tolerance_cap=args.tolerance_cap)

    # A size-adaptive run gets a distinct suffix so it never clobbers the fixed-cutoff
    # reference table (lbs_<tool>_st100_merged_per_pocket.csv).
    suffix = ""
    if tol_mode:
        suffix = f"_tol_{args.tolerance_col}_k{args.tolerance_scale:g}"
        if args.tolerance_floor is not None:
            suffix += f"_floor{args.tolerance_floor:g}"
        if args.tolerance_cap is not None:
            suffix += f"_cap{args.tolerance_cap:g}"
    out_csv = out_dir / f"lbs_{args.tool}_st100_merged_per_pocket{suffix}.csv"
    merged.to_csv(out_csv, index=False)
    manifest = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "tool": args.tool, "predictions_csv": str(args.predictions_csv),
        "geometry_csv": str(args.geometry_csv),
        "ground_truth_npz": str(args.ground_truth_npz) if args.ground_truth_npz else None,
        "center_fallback": bool(args.center_fallback),
        "threshold": args.threshold, "dca_cutoff": args.dca_cutoff,
        "tolerance_col": args.tolerance_col, "tolerance_scale": args.tolerance_scale,
        "tolerance_floor": args.tolerance_floor, "tolerance_cap": args.tolerance_cap,
        "tolerance_mode": bool(tol_mode),
        "summary": _summarize(merged),
    }
    with (out_dir / f"lbs_{args.tool}_manifest{suffix}.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[SAVE] {out_csv}")
    print(f"[SUMMARY] {json.dumps(manifest['summary'])}")


def _selftest() -> None:
    """Synthetic correctness check of the DCA / rank / TNS logic (no I/O)."""
    geom = pd.DataFrame({
        "threshold": ["ST_100"] * 3,
        "site_uid": ["S_hit_r1", "S_hit_r3", "S_miss"],
        "site_uniprot": ["U1", "U2", "U3"],
        "pocket_class": ["holo", "apo", "holo"],
        "pocket_p_apo": [0.2, 0.7, 0.1], "pocket_p_holo": [0.8, 0.3, 0.9],
        "pocket_separation_pair": [1.0, 1.0, 1.0], "pocket_density_pair": [1.0, 1.0, 1.0],
        "pocket_overlap_pair": [0.0, 0.0, 0.0], "pocket_score": [1.0, 1.0, 1.0],
        "model_pocket_plddt": [90, 90, 90], "n_apo_avg": [1.0, 1.0, 1.0], "n_holo_avg": [2.0, 0.0, 3.0],
        "union_rg": [10.0, 10.0, 10.0], "union_n_residues_found": [8, 8, 8],
        "n_unp_pockets": [2, 2, 2], "n_unp_pockets_multichain": [2, 2, 2],
        "rep_job": ["a", "b", "c"], "afdb_filename": ["x", "y", "z"],
    })
    # ground truth: a single site atom at the origin for each site
    gt_store = {"S_hit_r1": np.array([[0.0, 0.0, 0.0]]),
                "S_hit_r3": np.array([[0.0, 0.0, 0.0]]),
                "S_miss": np.array([[0.0, 0.0, 0.0]])}
    preds = pd.DataFrame({
        "site_uid": ["S_hit_r1", "S_hit_r1", "S_hit_r3", "S_hit_r3", "S_hit_r3", "S_miss"],
        "pocket_rank": [1, 2, 1, 2, 3, 1],
        "center_x": [1.0, 20.0, 20.0, 15.0, 2.0, 9.0],
        "center_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "center_z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "score": [9, 1, 9, 5, 1, 9],
    })
    m = score(preds, geom, gt_store, tool="selftest", threshold="ST_100", primary_cutoff=4.0, holo_plus_one=1.0).set_index("site_uid")
    assert int(m.loc["S_hit_r1", "dca_rank"]) == 1
    assert bool(m.loc["S_hit_r1", "success_tns"]) is True
    assert int(m.loc["S_hit_r3", "dca_rank"]) == 3
    assert bool(m.loc["S_hit_r3", "success_topn"]) is True
    assert bool(m.loc["S_hit_r3", "success_tns"]) is False   # rank 3 > n_unp 2
    assert int(m.loc["S_miss", "dca_rank"]) == -1
    assert bool(m.loc["S_miss", "success_topn"]) is False
    assert abs(float(m.loc["S_hit_r1", "closest_pocket_dist"]) - 1.0) < 1e-9
    assert abs(float(m.loc["S_hit_r1", "dca_cutoff_used"]) - 4.0) < 1e-9  # fixed cutoff recorded
    print("[selftest] OK - DCA / dca_rank / success_topn / success_tns / closest_pocket_dist correct")
    _selftest_tolerance()


def _selftest_tolerance() -> None:
    """Size-adaptive (per-site) tolerance path: cutoff = k * union_rg; missing rg -> non-scorable."""
    geom = pd.DataFrame({
        "threshold": ["ST_100"] * 3,
        "site_uid": ["S_big", "S_small", "S_norg"],
        "site_uniprot": ["U1", "U2", "U3"],
        "pocket_class": ["holo", "apo", "holo"],
        "pocket_p_apo": [0.2, 0.7, 0.1], "pocket_p_holo": [0.8, 0.3, 0.9],
        "pocket_separation_pair": [1.0, 1.0, 1.0], "pocket_density_pair": [1.0, 1.0, 1.0],
        "pocket_overlap_pair": [0.0, 0.0, 0.0], "pocket_score": [1.0, 1.0, 1.0],
        "model_pocket_plddt": [90, 90, 90], "n_apo_avg": [1.0, 1.0, 1.0], "n_holo_avg": [2.0, 0.0, 3.0],
        "union_rg": [12.0, 3.0, np.nan], "union_n_residues_found": [8, 8, 8],
        "n_unp_pockets": [2, 2, 2], "n_unp_pockets_multichain": [2, 2, 2],
        "rep_job": ["a", "b", "c"], "afdb_filename": ["x", "y", "z"],
    })
    gt_store = {"S_big": np.array([[0.0, 0.0, 0.0]]),
                "S_small": np.array([[0.0, 0.0, 0.0]]),
                "S_norg": np.array([[0.0, 0.0, 0.0]])}
    # each site has one rank-1 pocket at distance 5.0 from the site atom
    preds = pd.DataFrame({
        "site_uid": ["S_big", "S_small", "S_norg"],
        "pocket_rank": [1, 1, 1],
        "center_x": [5.0, 5.0, 5.0], "center_y": [0.0, 0.0, 0.0], "center_z": [0.0, 0.0, 0.0],
        "score": [9, 9, 9],
    })
    # k=0.5: S_big cutoff = 6.0 (5.0 <= 6 -> hit), S_small cutoff = 1.5 (5.0 > 1.5 -> miss),
    #        S_norg cutoff = NaN -> non-scorable.
    m = score(preds, geom, gt_store, tool="selftest", threshold="ST_100", primary_cutoff=4.0,
              holo_plus_one=1.0, tolerance_col="union_rg", tolerance_scale=0.5).set_index("site_uid")
    assert abs(float(m.loc["S_big", "dca_cutoff_used"]) - 6.0) < 1e-9
    assert int(m.loc["S_big", "dca_rank"]) == 1 and bool(m.loc["S_big", "success_tns"]) is True
    assert abs(float(m.loc["S_small", "dca_cutoff_used"]) - 1.5) < 1e-9
    assert int(m.loc["S_small", "dca_rank"]) == -1 and bool(m.loc["S_small", "success_topn"]) is False
    assert pd.isna(m.loc["S_norg", "dca_rank"]) and pd.isna(m.loc["S_norg", "success_tns"])
    assert pd.isna(m.loc["S_norg", "dca_cutoff_used"])
    # closest_pocket_dist is cutoff-independent -> still recorded for the non-scorable site
    assert abs(float(m.loc["S_norg", "closest_pocket_dist"]) - 5.0) < 1e-9
    print("[selftest] OK - size-adaptive per-site tolerance (k*rg), missing-rg non-scorable")

    # Floor: cutoff = max(4, 0.5*rg). S_small (rg=3) -> max(4, 1.5) = 4.0 (rescued from 1.5),
    # so its pocket at 5.0 is still a miss but the cutoff is the floor; S_big (rg=12) -> max(4, 6) = 6.0
    # (floor does not bind) -> hit. Missing rg stays non-scorable (max propagates NaN).
    mf = score(preds, geom, gt_store, tool="selftest", threshold="ST_100", primary_cutoff=4.0,
               holo_plus_one=1.0, tolerance_col="union_rg", tolerance_scale=0.5, tolerance_floor=4.0).set_index("site_uid")
    assert abs(float(mf.loc["S_small", "dca_cutoff_used"]) - 4.0) < 1e-9   # floored up from 1.5
    assert abs(float(mf.loc["S_big", "dca_cutoff_used"]) - 6.0) < 1e-9     # floor does not bind
    assert int(mf.loc["S_big", "dca_rank"]) == 1
    assert pd.isna(mf.loc["S_norg", "dca_cutoff_used"])                    # floor never rescues missing rg
    print("[selftest] OK - size-adaptive tolerance floor max(floor, k*rg)")


if __name__ == "__main__":
    main()
