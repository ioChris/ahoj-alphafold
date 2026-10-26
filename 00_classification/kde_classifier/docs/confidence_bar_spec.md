# Confidence-bar verbal bands — authoritative specification

*Source of truth for the per-entry KDE "confidence bar" quality labels. This document is
written to be cited in the manuscript. All numbers here are **derived from the code
constants** by `kde_plot.band_spec_table()` and pinned by
`tests/test_confidence_bar_bands.py` (issue `AHOJ-016`), so the spec and the pipeline cannot
drift apart.*

## 1. What the bars encode

Each per-entry confidence bar carries a **verbal quality label** — one of

> **Very low · Low · Moderate · High · Very high**

for one of two kinds of quantity, each labeled on **its own native scale**:

1. **Densities and overlaps** — dimensionless quantities already on `[0, 1]`
   (`normalized_density_strongest`, `normalized_density_combined`, `1 − overlap`,
   `1 − mode_overlap`). Labeled directly on `[0, 1]`.
2. **Apo/holo classification probability** — the *winning* posterior
   `p = max(P(apo), P(holo))`, which by construction lies on `[0.5, 1.0]` (the winner is
   always ≥ 0.5, so `p = 0.5` means a tie / no discrimination). Labeled directly on
   `[0.5, 1.0]`.

Both kinds use the **same five verbal bands**, but each has its own inclusive-upper-edge
table on its own scale. There is **no remap between the two** — in particular, the
probability is *not* squashed onto `[0, 1]`; its band edges are stated directly on the
probability axis as round, interpretable numbers, the same way the standardized
verbal-probability scales (IPCC, ICD 203) define theirs.

## 2. The band definitions (single source of truth)

In code (`kde_plot.py`) these constants are the only place the numbers live:

```python
VERBAL_BANDS    = ("Very low", "Low", "Moderate", "High", "Very high")
UNIT_BAND_EDGES = (0.20, 0.30, 0.60, 0.85)   # density/overlap, inclusive upper edges on [0,1]
PROB_BAND_EDGES = (0.55, 0.60, 0.70, 0.85)   # apo/holo probability, inclusive upper edges on [0.5,1]
```

Each edge belongs to the **lower** band (`value ≤ edge`). Formally, for a value `x` and its
edge tuple `(e1, e2, e3, e4)`: `x ≤ e1 →` Very low; `e1 < x ≤ e2 →` Low; `e2 < x ≤ e3 →`
Moderate; `e3 < x ≤ e4 →` High; `x > e4 →` Very high. The same helper
(`_verbal_from_value`) applies this to either scale.

## 3. Authoritative table

Probability edges are given to 2 decimal places and, for manuscript prose, as **percentage
points (pp)**.

| Verbal label | Density / overlap `u` ∈ [0, 1] | Apo/holo probability `p` ∈ [0.5, 1] | Probability (pp) |
| ------------ | ------------------------------ | ----------------------------------- | ---------------- |
| Very low     | `0.00 – 0.20`                  | `0.50 – 0.55`                       | `50 – 55 pp`     |
| Low          | `0.20 – 0.30`                  | `0.55 – 0.60`                       | `55 – 60 pp`     |
| Moderate     | `0.30 – 0.60`                  | `0.60 – 0.70`                       | `60 – 70 pp`     |
| High         | `0.60 – 0.85`                  | `0.70 – 0.85`                       | `70 – 85 pp`     |
| Very high    | `0.85 – 1.00`                  | `0.85 – 1.00`                       | `85 – 100 pp`    |

Every number is round to 2 dp by construction — there are no derived multi-decimal edges.

## 4. Worked examples

| Input                          | Scale        | Verbal band | Rendered label                                     |
| ------------------------------ | ------------ | ----------- | -------------------------------------------------- |
| overlap-based value `0.42`     | density [0,1]| Moderate    | `Moderate (0.42)`                                  |
| density `0.71`                 | density [0,1]| High        | `High (0.71)`                                      |
| `p = max(P) = 0.52`            | prob [0.5,1] | Very low    | `Classification: … - Probability: Very low (0.52)` |
| `p = 0.58`                     | prob [0.5,1] | Low         | `… Probability: Low (0.58)`                        |
| `p = 0.65`                     | prob [0.5,1] | Moderate    | `… Probability: Moderate (0.65)`                  |
| `p = 0.78`                     | prob [0.5,1] | High        | `… Probability: High (0.78)`                       |
| `p = 0.90`                     | prob [0.5,1] | Very high   | `… Probability: Very high (0.90)`                  |

Edge behavior: `p = 0.85 →` **High** (the edge belongs to the lower band); `p = 0.851 →`
Very high.

## 5. Choice of thresholds (rationale)

- **Five levels, empirically-set [0,1] edges.** The verbal vocabulary and the density/overlap
  edges (`0.20 / 0.30 / 0.60 / 0.85`) were chosen from empirical observation and **validated by
  their correlation with independent tools (P2Rank, AlphaFill)**. That validation is itself the
  justification for the cutpoints.
- **Probability edges defined directly on the native scale.** The apo/holo edges
  (`0.55 / 0.60 / 0.70 / 0.85`) are round, interpretable numbers stated directly on the
  `[0.5, 1.0]` winning-posterior axis. This is the **same methodological approach as the
  recognized verbal-probability standards** — IPCC likelihood and the US ODNI/ICD 203 scale are
  both fixed, expert-chosen cutpoints, *not* quantiles of any dataset. Direct thresholds are
  (i) **portable / reproducible** — a published "Very high means `p > 0.85`" does not move when
  the sample changes, and (ii) **correlation-preserving** — quantile (equal-mass) binning would
  impose artificial uniformity and can attenuate the very P2Rank/AlphaFill relationships the
  bins are validated against.
- **Semantics.** `p = 0.5` is a tie / no discrimination (the winner is ≥ 0.5 by construction),
  so this is a *relative confidence descriptor*, not a calibrated probability of correctness.
  The `[0.5, 1]` restriction is why the standard `[0, 1]` scales are not transplanted directly,
  and why the top band begins at `0.85` rather than the AlphaFold-pLDDT-style `0.90` (a `0.90`
  cut would leave "Very high" almost never populated for this quantity). Equivalently, this
  confidence is the posterior *margin* `|P(holo) − P(apo)| = 2p − 1 ∈ [0, 1]` (the score-bar
  marker's distance from centre); since the two posteriors sum to 1, the `[0.5, 1]`
  winning-posterior and the `[0, 1]` margin are a linear bijection carrying identical information,
  so labeling on `[0.5, 1]` is a presentation choice, not a loss of range.
- **Top edge is provisional.** `0.85` is a deliberate, round choice; if "Very high" proves too
  sparse on the full-database run it can be lowered to `0.80` (still round, more populated)
  without any other change.

### Relationship to the earlier sigmoid revision

An earlier revision remapped `p` onto `[0, 1]` with a logistic function of gain `k = 1.64`,
which produced derived edges `0.5615 / 0.5933 / 0.6996 / 0.8223`. The direct edges above are
essentially those rounded to interpretable values, so **reclassification is minimal**. The two
schemes disagree only on four narrow probability slivers (≈ 9.3 % of the `[0.5, 1]` range):

| `p` interval        | sigmoid (old) → direct (new) |
| ------------------- | ---------------------------- |
| `(0.550, 0.5615]`   | Very low → Low               |
| `(0.5933, 0.600]`   | Moderate → Low               |
| `(0.6996, 0.700]`   | High → Moderate              |
| `(0.8223, 0.850]`   | **Very high → High**         |

On a small real sample of **29 winning posteriors** (local smoke-test entries plus the
prototype entry `2yiw-A-I46-1354`) **0 labels changed** between the two schemes. The only edge
worth monitoring at scale is the top one (`0.82 → 0.85`), which makes "Very high" marginally
stricter.

## 6. Presentation and rounding policy

- **Displayed numbers are 2 decimal places.** Labels show the raw value in parentheses:
  densities/overlaps as the `[0, 1]` value (`Moderate (0.42)`); probability as
  `max(P(apo), P(holo)) ∈ [0.5, 1.0]` (`High (0.78)`).
- **Rounding never changes a label.** The band is decided on the **full-precision** value, on a
  code path independent of the 2 dp display, so rounding the shown number cannot flip the band —
  asserted by the regression test.
- **Percentage points for prose.** A probability may be quoted as pp in text (`0.85 = 85 pp`);
  the emitted figure labels keep the decimal form.

## 7. Where this lives / how to reproduce

- **Python source of truth:** `kde_plot.py` — `VERBAL_BANDS`, `UNIT_BAND_EDGES`,
  `PROB_BAND_EDGES`; consumed by `_verbal_from_value`, `_verbal_from_unit`, `_unitize`,
  `verbose_scoring_label`, `verbose_probability_label`. The full table is produced by
  `band_spec_table()`.
- **JavaScript mirror** (in parse-ahoj only, not part of this copy; its parity tests skip here):
  `analysis/interactive_confbars_prototype/confbars.js` keeps the same
  three constants; it is held numerically identical to the Python by the regression test.
- **Regression test:** `tests/test_confidence_bar_bands.py` (`AHOJ-016`) — checks labels on a
  fine grid of `u ∈ [0,1]` and `p ∈ [0.5,1.0]`, the boundary edges, rounding-independence,
  Python↔JS constant parity, the old-sigmoid-vs-new-direct flip regions, and this document's
  numbers against `band_spec_table()`.

Run the tests with:

```
python -m pytest tests/test_confidence_bar_bands.py
```

## 8. Selected references (verbal ↔ numeric confidence labels)

- IPCC AR5 calibrated uncertainty language (likelihood scale numeric; confidence qualitative):
  Mastrandrea et al. 2010 guidance note.
- US ODNI **ICD 203** analytic standards — estimative probability lexicon with percentage ranges
  (almost no chance 01–05 … almost certain 95–99).
- Kent, *Words of Estimative Probability* (origin of standardized numeric ranges for verbal
  terms).
- Budescu & Wallsten — verbal probability expressions as fuzzy membership functions; large
  inter-/intra-individual variability; publish numeric ranges alongside terms.
- AlphaFold pLDDT confidence bands (Very high > 90, Confident 70–90, Low 50–70, Very low < 50) —
  a different quantity (accuracy score), noted for contrast.
- Classifier-calibration literature on binning: equal-width bins are sparse at high confidence;
  quantile bins are dataset-dependent — motivating fixed, interpretable cutpoints here.
