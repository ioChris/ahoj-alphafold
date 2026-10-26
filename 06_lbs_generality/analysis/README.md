# 06_lbs_generality/analysis — the generality panel

`analyze_lbs_generality.py` pools the per-tool success tables
(`lbs_<tool>_st100_merged_per_pocket.csv`) — optionally with the P2Rank merged table as the
reference — into the generality panel: **TNS by class (AF-Apo vs AF-Holo) + the holo−apo gap per
tool**, plus TNS across probability / separation bins and evidence-ratio quintiles. One figure,
the same gradient in every tool.

```bash
python analyze_lbs_generality.py \
    --tool_tables  <...>/lbs_fpocket_st100_merged_per_pocket.csv \
                   <...>/lbs_ifsitepred_st100_merged_per_pocket.csv \
    --p2rank_table <...>/p2rank_st100_merged_per_pocket.csv
```

## Size control — reuse the P2Rank script directly
Because every per-tool success table shares the P2Rank schema (same `unit_uid`, `success_tns`,
`pocket_class`, `pocket_separation_pair`, `winning_p`, `log2_ratio_holo_to_apo_avg_plus1`, `rg`,
verbal bins), the size-controlled / controlled-pairs analysis needs **no new code** — point the
existing script at each tool's table:

```bash
python ../../04_p2rank/analysis/analyze_p2rank_controlled_pairs.py \
    --p2rank_table <...>/lbs_fpocket_st100_merged_per_pocket.csv \
    --success_col  success_tns \
    --focus_metric pocket_separation_pair \
    --control_metric log2_ratio_holo_to_apo_avg_plus1
```

This confirms per tool that the holo>apo advantage survives pocket-size control (partial-ρ /
controlled-pairs / CMH), mirroring the P2Rank size-confound analysis (T2-size). Do this for the
harmonized P2Rank table too, so the whole panel is on one metric.

## Size control (2) — a size-adaptive DCA tolerance (∝ pocket Rg)
`rescore_size_adaptive_tolerance.py` re-scores the *existing* per-tool predictions with a
**per-site** DCA cutoff `= k · union_rg` (k = 0.5 / 1.0 / 2.0) instead of the fixed 4 Å, applied
identically to every tool (fpocket, IF-SitePred, harmonized P2Rank; native P2Rank left as a
labelled SAS-eval reference). No tool re-runs. It emits a side-by-side apo/holo/gap table on the
full finite-Rg population (~86,946) and the balanced 1,494 pilot:

```bash
python rescore_size_adaptive_tolerance.py
# -> analysis_v1/tables/lbs_size_adaptive_tolerance_{long,table}.csv (+ manifest)
```

A tolerance that scales with pocket size is a built-in size-confound control: the holo−apo gap
surviving it (it does) is extra evidence the effect is not a pocket-size artifact, complementing
the size-matched pilot. Rg is often larger than 4 Å, so absolute TNS rises with k, but the
cross-tool comparison stays fair because the same criterion is applied to every tool.

## Ratio vs class, quality-gated — does it generalize across predictors?
`analyze_lbs_ratio_vs_class.py` extends the quality-gated ratio-vs-class analysis (parent
`05_docking/runner/analyze_quality_gated_ratio_vs_class.py`, T2-qualgate: docking + P2Rank-native +
AlphaFill) to the **three LBS predictors** (fpocket, IF-SitePred, harmonized P2Rank) on the default
harmonized **DCA-TNS `success_tns`**, plus docking (native dockability). It reuses the parent
module's machinery unchanged (size-controlled partial Spearman, paired-bootstrap CIs on the ρ
difference, nested 5-fold CV logistic AUC, threshold sweep, per-tool gate figure) and adds only an
**rg-only** LBS loader (binding-site detection has no ligand → no ligand-size covariate), a
**zero-order** pass, and the **cross-tool** generality figure + synthesis report.

```bash
python analyze_lbs_ratio_vs_class.py            # fpocket, IF-SitePred, P2Rank, docking; 3 gates
```

Gates (undirected, on `winning_p`): `all` / `prob_mod+` / `prob_high+`. Predictors compared vs
success: `log2_ratio`, `p_holo`, `is_holo`. **Result:** ungated, `ratio ≥ p_holo ≥ is_holo` holds in
**all four** predictors in both regimes (the generality result → C4); the size-controlled,
high-confidence **is_holo-surpasses-ratio crossover** is predictor-dependent (robust in IF-SitePred
and docking, absent in fpocket/P2Rank) and vanishes at zero order — the same bounded claim as
T2-qualgate. Outputs → `work_dir/lbs_generality/quality_gated/{fpocket,ifsitepred,p2rank,docking}/`
+ cross-method `REPORT.md` and `fig_cross_tool_ratio_vs_class.png` (candidate **SI** panel).

The cross-tool figure also carries a pooled **`all methods (mean of 4)`** group per panel: the overall ρ
per predictor = the **equal-weight Fisher-z mean** of the four per-method partial ρ, with a **joint
pocket-level cluster bootstrap** CI (resample shared pockets, recompute all four methods together — the
three LBS tables share the same pockets, so an independent bootstrap would be anticonservative). Because
that equal-weight mean is LBS-family-heavy (3 of 4 votes are the same pockets), a **2-family sensitivity**
(LBS trio → one vote vs docking) and a per-method `k/4` crossover count are reported alongside
(`pooled_overall_ratio_vs_class.csv`, REPORT §2b). **Overall answer:** the evidence ratio has the highest
correlation in every gate/regime **except** the size-controlled `prob_high+` corner, where pooled
`is_holo` edges past (a 2/4 majority that *grows* under the 2-family weighting) — the pooled winner is
identical under Fisher-z mean / simple mean / observation-stacking.

## Ratio vs class, quadrant read — *which* of the two facilitators is stronger?
`analyze_ratio_vs_class_quadrants.py` + `plot_ratio_vs_class_quadrants.py` answer the eyeball
question the Figure 6 contour surfaces raise: both axes clearly help, but which one is doing more
work, and does the answer differ between predictors? The surface is collapsed onto the four
quadrants of the two substantive cut points — `log2((n_holo+1)/n_apo) = 0` and `p_holo = 0.5`
(the class boundary) — and the pooled success rate is read in each:

```
    p_holo >= 0.5  |  Q2 apo-evidence + AF-Holo  |  Q1 holo-evidence + AF-Holo  |
    p_holo <  0.5  |  Q3 apo-evidence + AF-Apo   |  Q4 holo-evidence + AF-Apo   |
```

`E_ratio` and `E_class` are the two 2×2 main effects and **`PREF = E_ratio − E_class = Q4 − Q2`**
— an identity, so the headline is literally the difference between the two *discordant* corners
and a reader can verify it off the tile. Every estimand is reported raw **and** under direct
standardization to the method's own `union_rg` decile distribution (holo-leaning sites are bigger,
and size dominates detection), with cluster-bootstrap CIs that resample **UniProts**. Two extra cut
schemes (`median`, `quartile` — both equal-marginal) test how much the answer depends on where the
axes are cut, because a 2×2 compares one step along x with one step along y.

```bash
python analyze_ratio_vs_class_quadrants.py --B 1000     # -> analysis_v1/tables/quadrants_*.csv
python plot_ratio_vs_class_quadrants.py \
    --manuscript_dir "<...>/manuscript/figures/SI"      # -> SI_ratio_vs_class_quadrants.{png,pdf,svg}
```

### The two size adjustments (`adjust` column)
Every estimand is reported three ways, and the two adjusted ones are deliberately redundant —
agreement between two different estimators is what shows the shrinkage is a property of the data:

| `adjust` | what it is | reads as |
|---|---|---|
| `raw` | the observable rate | what a user meets when they point a predictor at an AF model |
| `size_std` | **direct** standardization: reweight each quadrant to the method's own size distribution | "the rate this quadrant would show if its pockets were sized like the population" |
| `size_resid` | **indirect** standardization: `grand + mean(success − E[success given size])` | "success net of what size alone predicts" |

They agree to 1–2 pp throughout. `--size_col` selects the size variable (`union_rg`, the default,
or `union_n_residues_found` — a residue COUNT that reads no coordinates, so it is immune to "you
computed that from the AF model", and the stronger proxy, ρ +0.64 vs +0.59, so it removes more).
**Neither adjustment discards a pocket** — both reweight. The cost is effective sample size (Kish
ESS 92 / 88 / 67 / 95 % of n for Q1..Q4; Q3 pays most, being the quadrant least like the population
in size). The only real n cost anywhere here is the `quartile` cut, which keeps 24,136 of 86,946
because the discordant extreme corners are rare.

## The surfaces themselves — raw and size-adjusted, side by side
`plot_ratio_class_surfaces.py` draws the continuous version of the same thing: success contoured
over evidence balance × p_holo, per method, **raw over size-adjusted**. It shares
`analyze_ratio_vs_class_quadrants.load_long()`, so the tiles and the surfaces can never drift apart.

```bash
python plot_ratio_class_surfaces.py \
    --manuscript_dir "<...>/manuscript/figures/SI"      # -> SI_ratio_class_surfaces.{png,pdf,svg}
                                                        #  + surface_<method>[_adjusted].* singles
```

The surfaces use the **residual** adjustment, not the direct one, and that is forced rather than
chosen: at the plotted 46×38 resolution the median (grid cell × size decile) holds **4 pockets** and
52 % hold fewer than 5, so direct standardization has nothing to reweight. Indirect standardization
needs only the global size→success curve and is defined in every cell. The two agree at quadrant
level, which is what licenses using different estimators in the two figures.

Two things to keep straight when quoting these panels:
* **Every panel has its own colour bar, directly beneath it**, and the rows use them differently:
  row a's four detectors share one range (colours comparable across methods, the main-figure
  convention); row b stretches each panel to its own range. The adjustment removes 45–61 % of the
  dynamic range (5–95 percentile spread over cells with ≥10 pockets) AND leaves the methods with
  different residual spreads (23–35 pp), so any shared range flattens some panels into a wash.
  `--adjusted_scale shared` gives the conservative variant. Compare the contour *labels*, which are
  absolute %, never the colours — across rows, and across methods within row b.
* **The surface changes SHAPE, not just contrast**: raw contours run near-vertical (evidence
  dominates), size-adjusted they run diagonally (both axes grade success). That is the same story as
  the quadrant matched-extreme result, and it is the reason the figure exists.

**Result:** at the sign cut the evidence balance is the stronger facilitator in all three predictors
and their consensus — raw PREF +17 to +22 pp, and still +7 to +10 pp size-standardized (all CIs
exclude 0). Roughly half the raw advantage is pocket-size composition. The three predictors do
**not** differ from one another in preference (all pairwise CIs span 0) — IF-SitePred's more
diagonal-looking surface is a larger *dynamic range* on both axes, not a different balance.
**Docking is the exception**: it leans on the classification significantly more than any detector,
and once size is held fixed its two facilitators are of comparable strength (PREF −0.5 pp
[−3.7, +2.6]).

⚠ **The ratio's win is cut-dependent, and the sign cut gives it the longer lever** (a wide,
heavy-tailed axis split at 0 vs p_holo split near its own median). At the **matched-extreme
(quartile) cut with size held fixed**, PREF stays positive in point estimate but its CI spans zero
for P2Rank (+3.1 [−5.6, +11.2]), IF-SitePred (+3.6 [−5.3, +11.6]) and the consensus
(+5.7 [−0.6, +12.2]); only fpocket stays clearly positive (+10.4 [+1.9, +19.7]). Equal lever arms
**and** size removed ⇒ the two facilitators are statistically indistinguishable for two of the three
predictors. Quote the sign-cut result as the headline and this as its scope, never the reverse.

This is the descriptive companion to the continuous, inferential ranking above — quote that one for
"which predictor is better", this one for "which pocket property matters more".

## What goes in the paper
A single multi-tool panel: AF-Holo vs AF-Apo TNS for P2Rank (native + harmonized), fpocket,
IF-SitePred (+ optional 3rd), with the size-controlled deltas underneath. The takeaway is the
**shared positive holo−apo gap**, not any single tool's absolute rate.
