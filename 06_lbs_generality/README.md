# 06 — LBS generality (2nd/3rd binding-site predictor)

**Analysis 7 (new).** Run 2–3 *additional* ligand-binding-site / pocket predictors on the
**same** AFDB ST_100 pocket set and recompute the **same** DCA / **TNS** success as the P2Rank
stage, to show the **AF-Holo > AF-Apo detection gradient is not P2Rank-specific** (claim **C4**
generality; experiment **T3-2nd**). If any competent detector finds holo-like pockets more
easily than apo-like ones, the effect is a property of the *structures*, not of one tool.

> **Read first:** [`docs/LBS_GENERALITY_DESIGN.md`](docs/LBS_GENERALITY_DESIGN.md) — methods
> summary, method/pipeline survey highlights, and the recommendation. The **full** method
> survey, pipeline survey, and recommendation rationale live in the private project repo
> (`ahoj-af-project/lbs_generality/`, experiment **T3-2nd**); the code repo stays code-only.

Paths come from [`../common/config.py`](../common/config.py) (`LBS_*`). Nothing is hard-coded to
a classification run: after a new round regenerates `AFDB_ST100_DATASET_GEOM_CSV`, re-run the
ground-truth builder + adapters + scorer.

## The one idea that makes this cheap
Every predictor is reduced to a **normalized `predictions.csv`** (ranked pocket centres) by a
small adapter, and then **one tool-agnostic scorer** reproduces the P2Rank success columns.
The scorer's metric (centre → nearest union-residue atom, ≤ 4 Å, ranked ≤ `n_unp_pockets`)
reproduces P2Rank's own `closest_pocket_dist` to < 1e-4 Å, so all tools — **P2Rank re-scored
included** — share one yardstick and land in the **same schema** as
`p2rank_st100_merged_per_pocket.csv`. The existing size-controlled analysis is then reused
verbatim.

## Recommended predictors (see design doc)
1. **fpocket** — unsupervised Voronoi alpha-sphere geometry (MIT, mmCIF-native, CPU,
   seconds/structure). A different algorithm class from P2Rank; emits ranked centres +
   residues + a druggability score. **Non-negotiable first add.**
2. **IF-SitePred** — ESM-IF1 inverse-folding embeddings → ranked 3D site centres on the AF
   coordinates directly (BSD-3). A *learned-representation* axis (not geometry), and the
   cleanest deep-learning drop-in because it outputs centres natively.
3. *(optional 3rd)* **GrASP** (graph-attention, ranked centres + residues) or a residue-level
   PLM predictor (**PeSTo** / GPSite) scored by residue-overlap — the strongest "even a tool
   that never detects a cavity shows the gradient" statement.

## Directory layout
```
06_lbs_generality/
├── docs/
│   └── LBS_GENERALITY_DESIGN.md    # methods summary + survey highlights + recommendation (public)
├── schemas/
│   ├── predictions.schema.md       # normalized predicted-pockets input contract (tool-agnostic)
│   └── lbs_success.schema.md        # per-pocket success output contract (== P2Rank success cols)
├── dataset/                         # inputs + per-tool ADAPTERS
│   ├── build_ground_truth_atoms.py  # gemmi: per-pocket union-residue heavy atoms -> ground_truth_atoms.npz
│   ├── build_lbs_inputs.py          # geometry -> receptors.csv + receptor_site_map.csv (+ pilot subset)
│   └── runners/
│       ├── run_fpocket.py           # fpocket output -> raw pockets table (SCAFFOLD: parser done, --run TODO)
│       └── normalize_predictions.py # generic: raw per-receptor pockets -> normalized predictions.csv
├── score/
│   └── compute_lbs_success.py       # THE scorer: predictions + ground truth -> per-pocket TNS table
└── analysis/
    ├── analyze_lbs_generality.py    # pool per-tool tables -> generality panel (+ reuse P2Rank controlled-pairs)
    └── ...                          # downstream analyses: cutoff robustness, ratio-vs-class (continuous
                                     #   and 2x2 quadrant), Fig 6A quality gates -- see analysis/README.md
```

## Quickstart — a pilot bundle
Mirror the docking pilot: validate the effect on a stratified ~1.5k-pocket subset first.
```bash
cd 06_lbs_generality

# 0) (once, gemmi env) build the shared ground-truth atom store for the metric
python dataset/build_ground_truth_atoms.py                # ~82k pockets; --limit 200 to smoke-test

# 1) draw the SAME stratified pilot as docking, then build receptor lists for it
python 05_docking/dataset/sample_pilot_pockets.py --target_n 1500   # -> pilot_pockets.csv
python dataset/build_lbs_inputs.py --pilot_site_uids <...>/pilot_pockets.csv

# 2) run each predictor (fpocket under WSL; IF-SitePred on the AF models) then normalize
python dataset/runners/run_fpocket.py --fpocket_out_root <...>
python dataset/runners/normalize_predictions.py --raw_pockets <...>/fpocket_raw.csv \
    --tool fpocket --out_csv <work>/lbs_generality/fpocket/predictions.csv

# 3) score every tool with the ONE shared metric (base env)
python score/compute_lbs_success.py --tool fpocket \
    --predictions_csv <work>/lbs_generality/fpocket/predictions.csv \
    --ground_truth_npz <work>/lbs_generality/ground_truth_atoms.npz

# 4) pool + show the gradient; controlled-pairs via the P2Rank script
python analysis/analyze_lbs_generality.py --tool_tables <...>/lbs_fpocket_*.csv
python 04_p2rank/analysis/analyze_p2rank_controlled_pairs.py \
    --p2rank_table <...>/lbs_fpocket_*.csv --success_col success_tns
```

> Status: `score/compute_lbs_success.py` and `dataset/build_ground_truth_atoms.py` are
> implemented and **validated** (the scorer reproduces P2Rank's `closest_pocket_dist` to a
> 3e-5 Å median on a 300-pocket end-to-end check, and a self-test covers the rank/TNS logic).
> The full runs used the cluster scripts in `hpc/` (fpocket: `fpocket_run.sbatch` →
> `fpocket_node.sh`, which runs fpocket and then this parser; IF-SitePred: `ifs_array.sbatch`)
> and, locally, `dataset/runners/run_*_wsl.sh`. `run_fpocket.py --run` (fpocket called from
> Python) was not used and remains unimplemented.
