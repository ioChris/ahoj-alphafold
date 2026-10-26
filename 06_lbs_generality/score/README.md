# 06_lbs_generality/score — the tool-agnostic DCA / TNS scorer

`compute_lbs_success.py` is the reusable heart of the stage. It maps any predictor's normalized
`predictions.csv` → a per-pocket success table with the **same schema and success columns** as
`p2rank_st100_merged_per_pocket.csv`, so the P2Rank analysis is reused verbatim.

## The metric (and why it is trustworthy)
For each pocket, ground truth = the union binding-site residue heavy atoms on the AF model
(`ground_truth_atoms.npz` from `dataset/build_ground_truth_atoms.py`). Per predicted pocket
centre *c*: `DCA = min over ground-truth atoms ||c − a||`; `dca_rank` = rank of the first pocket
with `DCA ≤ cutoff` (default 4 Å); `closest_pocket_dist` = min DCA over pockets;
`success_tns = dca_rank > 0 AND dca_rank ≤ n_unp_pockets`.

**Validation (done):**
- **Self-test** (`--selftest`) covers the rank / DCA / TNS / closest-distance logic on a
  synthetic case.
- **End-to-end vs P2Rank:** feeding P2Rank's own predicted pocket centres through this scorer
  reproduces P2Rank's `closest_pocket_dist` to a **3e-5 Å median** (300-pocket check), confirming
  the ground-truth model matches P2Rank's. The holo>apo gradient is preserved
  (harmonized TNS: apo ≈ 44–49% vs holo ≈ 60–68%).

**Step 0 for a new run — re-score P2Rank as the in-study baseline.** Build a `predictions.csv`
from P2Rank's raw `*_predictions.csv` centres (columns `name, rank, center_x/y/z`) → run this
scorer with `--tool p2rank_harmonized`. That gives P2Rank on the exact same yardstick as the new
tools. Report the manuscript's native P2Rank TNS (84.0 / 63.6) alongside as the reference.

## Usage
```bash
python compute_lbs_success.py --tool fpocket \
    --predictions_csv  <work>/lbs_generality/fpocket/predictions.csv \
    --ground_truth_npz <work>/lbs_generality/ground_truth_atoms.npz
# -> <work>/lbs_generality/fpocket/analysis_v1/tables/lbs_fpocket_st100_merged_per_pocket.csv
```
`--center_fallback` uses the single `union_center_geom` point instead of the atom store (coarse;
does NOT match P2Rank) — smoke tests only. Base env (pandas + numpy). Output contract:
[`../schemas/lbs_success.schema.md`](../schemas/lbs_success.schema.md).

## Optional: size-adaptive DCA tolerance (per-site, ∝ pocket Rg)
By default the DCA cutoff is a fixed `--dca_cutoff` (4 Å) for every site. Passing
`--tolerance_col union_rg --tolerance_scale k` switches to a **per-site** cutoff
`= k · union_rg` (k typically 0.5 / 1.0 / 2.0), a size-normalized criterion applied
identically to every tool. `success_tns` / `dca_rank` / `success_top1/topn` then use the
per-site cutoff; the absolute `success_dca_XA` sweep and `closest_pocket_dist` are unchanged,
and the effective cutoff is recorded per row in `dca_cutoff_used`. Sites with a missing/non-finite
`union_rg` become non-scorable (NaN), identically for every tool. A tolerance run is written to a
suffixed file (`lbs_<tool>_st100_merged_per_pocket_tol_union_rg_k<k>.csv`) so it never clobbers the
fixed-cutoff reference. This is a built-in **pocket-size-confound control**: if the AF-Holo>AF-Apo
gradient survives a size-adaptive tolerance, the effect is not a pocket-size artifact. The
cross-tool re-score + side-by-side table is driven by
[`../analysis/rescore_size_adaptive_tolerance.py`](../analysis/rescore_size_adaptive_tolerance.py).

## Optional: residue-only predictors
Sequence/embedding tools (PeSTo, GPSite, bindEmbed21DL) emit residues, not centres. Their
adapter clusters predicted binding residues and sets `center_*` = cluster centroid on the AF
model (`residue_centroids.py`, gemmi — a documented TODO), and/or the pooling analysis reports a
**residue-overlap** success (recall of `union_chain_resi`) alongside DCA.
