A compact, publication-ready set of plot descriptions that can be dropped into a README, paper supplement, or methods section.

# Classification prevalence plots (AHoJ / UBS)

## Notation & units (applies to all figures)

* **UBS** = Unified Binding Site (one row in the dataset).
* **Class** = final label for a UBS: **apo** or **holo**.
* **avg_n_apo / avg_n_holo** = average number of independent **PDB entries** (structures) supporting the apo/holo label for that UBS.
* **k** = a threshold applied to avg_n counts (unit: **# PDB entries**).
* **n** = number of UBSs included in a calculation or bin (unit: **# UBS**).
* **Prevalence** = share of UBSs labeled as apo/holo within the current denominator:
  [
  \text{prevalence}*\text{holo} = \frac{#\text{holo UBS}}{#\text{UBS in set}} \quad ; \quad \text{prevalence}*\text{apo}=1-\text{prevalence}_\text{holo}
  ]
* **Ambiguity filter (optional)**: UBS rows with weak pocket probabilities (max of `avg_P(apo)_mean_pocket`, `avg_P(holo)_mean_pocket`) < τ (default τ=0.60) are dropped before plotting.
* **NaN handling**: When enabled, `nan_as_zero=True` treats missing avg_n as 0.

---

# 1) Prevalence vs k (line)

**File**: `prevalence_vs_k`
**Axes**: x = **k** (avg_n threshold, # entries); y = **prevalence (%)** of apo and holo.
**What it shows**: For each k, the fraction of UBSs labeled **apo** vs **holo** among UBSs that **survive** the k-filter.

* **Modes (denominator definition)**

  * **global**: include UBS if **avg_n_apo ≥ k** **and** **avg_n_holo ≥ k** (mutuality; “both sides populated”).
  * **per_class**: include UBS if its **own** supporting side meets k (apo UBS needs avg_n_apo ≥ k; holo UBS needs avg_n_holo ≥ k).

**Interpretation tips**

* Steep drops in **n** between k and k+1 indicate many UBSs with zero or low support on one side.
* Convergence toward ~50/50 under **global** mode suggests that among sufficiently supported sites, classes are balanced.

**Caption template**

> Classification prevalence across avg_n threshold k. At each k, prevalence is computed among UBSs that pass the k-filter (mode = global/per_class). Shaded markers show apo and holo prevalence (%). n indicates the number of UBSs at each k.

---

# 2) Counts vs k (line)

**File**: `counts_vs_k`
**Axes**: x = **k**; y = **n_apo**, **n_holo** (unit: # UBS).
**What it shows**: Raw counts of apo/holo UBSs that remain after applying the k-filter (same denominator as the paired prevalence plot).
**Use**: Diagnoses where class imbalance arises and how quickly data thins with k.

**Caption template**

> Number of apo and holo UBSs surviving the avg_n threshold k under the selected mode (global/per_class).

---

# 3) Prevalence vs k — bubble size for counts

**File**: `prevalence_bubble_vs_k`
**Axes**: x = **k**; y = **prevalence (%)**; marker size encodes **n_apo** (for apo series) and **n_holo** (for holo series).
**What it shows**: Same prevalence as (1) but coupled with count magnitude in the same panel.
**Use**: Read prevalence while keeping an eye on the reliability (bigger bubbles → more UBSs).

**Caption template**

> Prevalence of apo/holo vs k with marker area proportional to class-specific counts at each k.

---

# 4) Scatter vs avg_n (single-UBS cloud with trend)

**File**: `prevalence_scatter_avg_n`
**Purpose**: Visualize individual UBS points against an x-axis derived from avg_n, with an optional **trend overlay** summarizing prevalence along x.

**Axes**

* **x** depends on `scatter_x_mode`:

  * **min**: ( \min(\text{avg_n_apo}, \text{avg_n_holo}) )
  * **class**: apo UBS uses **avg_n_apo**, holo UBS uses **avg_n_holo**
  * **max**: ( \max(\text{avg_n_apo}, \text{avg_n_holo}) )
  * **common**: points at **min**; **trend** computed with the **cumulative-tail** rule (see below)
  * **delta**: ( \text{avg_n_holo} - \text{avg_n_apo} ) (or the reverse), unit: # entries
* **y** depends on `scatter_y_mode`:

  * **class**: points at **0** (apo) or **1** (holo)
  * **trend**: point y equals local **prevalence** in its x neighborhood (unit: fraction or %)

**Trend overlay & binning**

* **Binned** trend (for x-modes: min, class, max, delta): prevalence is computed **within x-bins**.
* **Cumulative-tail** trend (for x-mode: common): at each k, include only UBS with **avg_n_apo ≥ k** **and** **avg_n_holo ≥ k** (mirrors “global” line plot).
* **Grid position**: you can evaluate trend at **bin edges** (0,1,2,…) or **centers** (0.5,1.5,…). Using **edges** groups **x=0** cleanly into the first bin.

**Styling options**

* **points**: ultra-light dots for dense clouds (large N).
* **jitter**: adds small vertical jitter to separate overlapping class bands.
* **hexbin_facets**: per-class count hexbins (top = apo, bottom = holo) with optional trend overlay.

**On-figure annotations**

* Legend shows **apo (n=…)**, **holo (n=…)** for plotted points.
* Top-right tag: **Total n = …** (UBSs plotted after filters).
* Top-left note indicates the mapping (e.g., “Y = prevalence • binned along x” or “Y = prevalence • cumulative-tail (apo & holo ≥ k)” or “Y = class bands (0/1)”).

**Interpretation tips**

* If the **scatter trend (common)** is evaluated at **edges**, the leftmost point (k=0) should match the **prevalence_vs_k** curve at k=0. Using **centers** starts at 0.5 and will reduce zero-side UBS in the first point, shifting the apparent mix.
* Dense early-x clouds often reflect UBSs with low support on one side; the **global**/tail view reveals how the balance looks once both sides are adequately sampled.

**Caption template**

> Single-UBS scatter vs x (x-mode = min/class/max/common/delta). Points show UBS class; the thin line shows class prevalence along x (binned or cumulative-tail, as indicated). n denotes the number of UBSs plotted after filters.

---

# 5) 2D prevalence heatmap (apo vs holo support)

**File**: *internal helper* `_plot_prevalence_heatmap_2d`
**Axes**: x = **avg_n_apo**, y = **avg_n_holo**; color = **holo prevalence** in each 2D bin (fraction).
**What it shows**: A field view of class balance across the joint space of apo/holo support.
**Use**: Diagnoses regimes where class flips or where one side dominates at specific support combinations.

**Caption template**

> Holo prevalence across the joint space of avg_n_apo and avg_n_holo. Color indicates the fraction of holo UBSs per 2D bin.

---

# 6) Prevalence vs min(avg_n) — bubble/points (optional helper)

**File**: `prevalence_by_min_avg` + `_plot_bubble_prevalence_by_min_avg`
**Axes**: x = ( \min(\text{avg_n_apo}, \text{avg_n_holo}) ) (binned); y = **prevalence (%)**; bubble sizes ∝ **n_apo**/**n_holo** in each bin.
**Grid**: evaluated at **edges** or **centers** (your switch). **x=0** is grouped in the 0-bin when using edges.
**Use**: A compact alternative to the scatter+trend, summarizing how prevalence evolves as the common support increases.

**Caption template**

> Apo/holo prevalence vs the minimum of apo/holo support per UBS, with marker area proportional to class counts per bin. Bins evaluated at [edges/centers] as indicated.

---

## General reporting notes (for all figures)

* **Threshold group label**: include the UBS threshold (e.g., `ST_LJ`, `ST_100`) in the title or caption.
* **Filters used**: report whether the **ambiguity filter** (τ=0.60) and **nan_as_zero** were enabled.
* **Grid mode**: if relevant, specify **edges** vs **centers** for trend evaluation.
* **Bin width**: report the bin width used for binned trends (default = 1.0).
* **Denominator clarity**: when comparing figures, ensure the same denominator concept (binned vs cumulative-tail) to avoid mismatched interpretations.

---

## One-liner captions you can paste next to plots

* **Prevalence vs k (global)**: “Prevalence of apo/holo among UBSs with avg_n_apo ≥ k and avg_n_holo ≥ k (k in # PDB entries).”
* **Counts vs k**: “Number of apo/holo UBSs that pass the threshold k.”
* **Bubble vs k**: “Prevalence vs k with bubble sizes proportional to class counts.”
* **Scatter (x=common, y=trend)**: “UBS scatter vs common min support; trend shows cumulative-tail prevalence (apo & holo ≥ k).”
* **Scatter (x=class, y=class)**: “Class bands with x = class-specific support (avg_n_apo for apo, avg_n_holo for holo).”
* **2D heatmap**: “Holo prevalence across the joint space of apo/holo support.”

If you want, I can also generate a small “Methods” paragraph tying these figures into your pipeline (KDE classification → UBS aggregation → prevalence summaries) so it’s manuscript-ready.
