"""
Validate the harmonized LBS metric against P2Rank's NATIVE classic-atom DCA.
=============================================================================

The LBS-generality panel scores every tool (P2Rank re-scored, fpocket, IF-SitePred)
with ONE tool-agnostic metric: predicted pocket centre -> nearest ``union_chain_resi``
heavy atom <= 4 A, ranked <= ``n_unp_pockets`` (**TNS**). P2Rank's own run_3 eval,
however, used ``-site_eval_sas_pts_as_atoms 1``, which measures DCA to the site's SAS
shell (~3.4 A outward) and inflates success (native TNS apo 66.1 / holo 84.8). That SAS
projection is PDB-incomparable (confirmed by R. Krivak) and the other tools cannot see
P2Rank's SAS points, so it is only a labelled reference.

This script cross-checks that our harmonized re-score reproduces P2Rank's OWN classic
residue-atom DCA, by re-running P2Rank's native eval with the SAS flag OFF and joining
its per-site ``dca4rank`` to the pocket class. Re-run the eval identically except for the
one flag (deterministic; predicted pockets + ``closestPocketDist`` are byte-identical,
only ``dca4rank`` changes), e.g.::

    prank eval-predict st_100_p2rank_intersection_geom.ds -c alphafold -l CENTER_explicit \
      -site_eval_center_method explicit -site_eval_sas_pts_as_atoms 0 \
      -visualizations 0 -threads 28 -out_subdir UBS2_run3_sasoff

then point this script at the resulting eval dir::

    python validate_native_dca_sasoff.py <eval_dir> [--out_csv tns_by_class.csv]

Result (run_3, n = 34,900 apo / 51,751 holo):
  SAS-native  (flag 1)          apo 66.11 / holo 84.76
  SAS-off native (flag 0)       apo 45.52 / holo 61.43   <- classic residue-atom DCA
  ad-hoc harmonized (this panel) apo 46.78 / holo 63.08
=> classic-atom native ~= harmonized (within ~1.3-1.7 pp); the ~20 pp gap vs SAS-native
   was ENTIRELY the SAS shell. The small residual is one-directional and nested (native
   success is a strict subset of harmonized): the shared heavy-atom store carries the FULL
   residue heavy-atom set (~101 atoms/site) while P2Rank's internal ``site.atoms`` is a
   subset (~99/site), so the store is marginally more permissive on borderline ~4 A sites.
   Residue sets are identical (union == intersection at ST_100), so the eval basis does not
   matter -- verified by re-running on the union .ds (identical numbers to 6 dp).

The join here reproduces the SAS-on native TNS (66.11/84.76) exactly, so applying it to the
SAS-off eval yields a directly comparable number. Mirrors ``analyze_p2rank_ubs_run4.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
for _root in Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
from common import config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate harmonized TNS vs P2Rank native classic-atom DCA (SAS off).")
    p.add_argument("eval_dir", type=Path, help="P2Rank eval-predict output dir (contains cases/observed_sites.csv).")
    p.add_argument("--union_geom_table", type=Path, default=None,
                   help="ST_100 union-geom CSV with site_uid + n_unp_pockets (default: cfg run_3 path).")
    p.add_argument("--unified_table", type=Path, default=None,
                   help="analysis_unified_levels.csv with unit_uid + pocket_class (default: cfg run_3 path).")
    p.add_argument("--threshold", default="ST_100")
    p.add_argument("--level", default="ubs")
    p.add_argument("--out_csv", type=Path, default=None, help="Optional: write the per-class table here.")
    p.add_argument("--dump_merged", type=Path, default=None, help="Optional: write the per-site merged join here.")
    return p.parse_args()


def compute(eval_dir: Path, union_geom_table: Path, unified_table: Path,
            threshold: str, level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(eval_dir / "cases" / "observed_sites.csv")
    obs.columns = [str(c).strip().replace(" ", "_") for c in obs.columns]
    obs["unit_uid"] = obs["site"].astype(str).str.strip()
    obs["dca4rank"] = pd.to_numeric(obs["dca4rank"], errors="coerce")
    obs["closest_pocket_dist"] = pd.to_numeric(obs["closestPocketDist"], errors="coerce")
    obs["ligands_per_protein"] = pd.to_numeric(obs["#sites"], errors="coerce")
    obs = obs.drop_duplicates(subset=["unit_uid"])

    g = pd.read_csv(union_geom_table, usecols=["threshold", "site_uid", "n_unp_pockets"], low_memory=False)
    g = g[g["threshold"] == threshold].rename(columns={"site_uid": "unit_uid"}).drop_duplicates("unit_uid")

    u = pd.read_csv(unified_table, usecols=["threshold", "level", "unit_uid", "pocket_class"], low_memory=False)
    u = u[(u["threshold"] == threshold) & (u["level"] == level)].drop_duplicates("unit_uid")

    m = obs.merge(g[["unit_uid", "n_unp_pockets"]], on="unit_uid", how="left")
    m = m.merge(u[["unit_uid", "pocket_class"]], on="unit_uid", how="left", validate="many_to_one")
    m["n_unp_pockets"] = pd.to_numeric(m["n_unp_pockets"], errors="coerce")
    m["pocket_class"] = m["pocket_class"].fillna("unknown").astype(str).str.lower()

    valid_tns = m["n_unp_pockets"].notna() & (m["n_unp_pockets"] >= 1)
    m["success_tns"] = np.where(valid_tns, (m["dca4rank"] > 0) & (m["dca4rank"] <= m["n_unp_pockets"]), np.nan)
    m["success_topn"] = m["dca4rank"] > 0
    valid_sites = m["ligands_per_protein"].notna() & (m["ligands_per_protein"] >= 1)
    m["success_dca4_sites"] = np.where(valid_sites, (m["dca4rank"] > 0) & (m["dca4rank"] <= m["ligands_per_protein"]), np.nan)

    rows = []
    for col in ["success_tns", "success_topn", "success_dca4_sites"]:
        for cls in ["apo", "holo", "tie", "all"]:
            s = pd.to_numeric(m[col] if cls == "all" else m.loc[m["pocket_class"] == cls, col], errors="coerce")
            rows.append({"metric": col, "class": cls, "n": int(s.notna().sum()),
                         "success_percent": round(float(s.mean() * 100.0), 4) if s.notna().any() else np.nan})
    return pd.DataFrame(rows), m


def main() -> None:
    a = parse_args()
    union_geom_table = a.union_geom_table or (
        cfg.AFDB_DATASET_ST100_DIR / "p2rank_datasets" / "st_100_union_geom" / "st_100_p2rank_union_geom.csv")
    unified_table = a.unified_table or (cfg.UNIFIED_RUN_DIR / "analysis_unified_levels.csv")

    tab, merged = compute(a.eval_dir, union_geom_table, unified_table, a.threshold, a.level)
    print(f"[eval_dir] {a.eval_dir}")
    print(tab.to_string(index=False))
    if a.out_csv:
        tab.to_csv(a.out_csv, index=False)
        print(f"[SAVE] {a.out_csv}")
    if a.dump_merged:
        merged.to_csv(a.dump_merged, index=False)
        print(f"[SAVE] {a.dump_merged}")


if __name__ == "__main__":
    main()
