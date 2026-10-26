# AHoJ-AlphaFold — apo/holo analysis of AlphaFold-DB binding pockets

Analysis and plotting code for the **AHoJ-AlphaFold** project, which places
AlphaFold-DB (AFDB) model binding pockets on an experimentally-anchored
**apo ↔ holo** continuum using apo/holo reference distributions from
[AHoJ-DB](https://doi.org/10.1016/j.jmb.2024.168545). Each AFDB pocket is scored against
experimental apo/holo distributions over four pocket metrics (all-atom RMSD, SASA,
concave volume, radius of gyration), aggregated into Unified Binding Sites (UBS), and
then compared against AlphaFill ligand transplants and P2Rank binding-site predictions.

> This repository contains the apo/holo **classification** code ([`00_classification/`](00_classification/)),
> the **binding-site unification** ([`01_ubs_clustering/unification/`](01_ubs_clustering/unification/)), the
> downstream analyses and the figure generators of the manuscript.

> **Status:** research code accompanying the manuscript *A tale of two states: AlphaFold binding
> pockets on the apo–holo continuum* (submitted; preprint DOI to follow). The data are at Zenodo,
> [10.5281/zenodo.23243349](https://doi.org/10.5281/zenodo.23243349). Scripts are run
> interactively (e.g. in Spyder) or from a shell; settings live in a top-of-file config block or
> argparse defaults, both sourced from a single central path registry ([`common/`](common/)).

---

## The analyses

The repository is organized around the project's analysis arc. Each numbered directory is
one stage; data flows top to bottom.

| Stage | Directory | What it does |
|------|-----------|--------------|
| 1 | [`00_classification/`](00_classification/) | **Probabilistic apo/holo classification** of each AFDB pocket vs AHoJ-DB apo/holo reference distributions (KDE density-ratio over 4 pocket metrics, equal priors): the batch classifier in `kde_classifier/`, then the analysis-set filter `filter_survivors.py`. |
| 2 | [`01_ubs_clustering/`](01_ubs_clustering/) | Unify binding sites per UniProt (`unification/`) and cluster AHoJ entries into **Unified Binding Sites (UBS)** by Jaccard similarity of UniProt residue sets (thresholds `ST_100/070/050/030/ST_LJ`; `UNCLUSTERED` = per-entry). Builds the master `analysis_unified_levels` table. |
| 3 | [`02_class_summary/`](02_class_summary/) | Summarize AFDB pocket apo/holo **class composition per clustering threshold** (prevalence shifts as redundancy is reduced), plus site-size and Sankey-flow summaries. |
| 4 | [`03_alphafill/`](03_alphafill/) | Compare AFDB classes vs **AlphaFill** ligand-transplant fit (clash score/count, local/global RMSD), stratified by apo/holo experimental-state separation. |
| 5 | [`04_p2rank/`](04_p2rank/) | Build the P2Rank input dataset and evaluate **P2Rank binding-site prediction** success vs apo/holo class, probability, and evidence balance (with size-controlled checks). |
| 6 | [`05_docking/`](05_docking/) | Dock biologically-relevant ligands into AFDB pockets (**AutoDock Vina / Uni-Dock**) and relate docking success (affinity + pose recovery) to apo/holo class — the direct functional test complementing P2Rank/AlphaFill. Strategy: [`05_docking/docs/DOCKING_DESIGN.md`](05_docking/docs/DOCKING_DESIGN.md). |
| 7 | [`06_lbs_generality/`](06_lbs_generality/) | Run *additional* binding-site predictors (**fpocket**, **IF-SitePred**) on the same AFDB pockets and recompute the same DCA/**TNS** success, showing the AF-Holo>AF-Apo detection gradient is **not P2Rank-specific**. One tool-agnostic heavy-atom scorer reproduces P2Rank's metric; the existing size-controlled analysis is reused. Results: manuscript Figure 3 and Supplementary Figures S4–S6. Strategy: [`06_lbs_generality/docs/LBS_GENERALITY_DESIGN.md`](06_lbs_generality/docs/LBS_GENERALITY_DESIGN.md). |
| — | [`plotting/`](plotting/) | Renders the main manuscript figures and SI figures for the classification / prevalence analyses (stages 1–3). |
| — | [`common/`](common/) | Central path configuration (see below). |

Key conventions:
- Main P2Rank success metric **`TNS`** = `(dca4rank > 0) and (dca4rank <= n_unp_pockets)`.
- Evidence balance = `log2((n_holo_avg + 1) / n_apo_avg)` (the `+1` is the query holo structure).

---

## Repository structure

```
.
├── common/                 # central path registry (config.py + paths.ini)
├── 00_classification/      # methodology provision (README pointer; see below)
├── 01_ubs_clustering/      # UBS clustering + master analysis table
├── 02_class_summary/       # prevalence / size / Sankey summaries
├── 03_alphafill/           # AlphaFill ligand-transplant comparison
│   ├── (download → parse → merge prep scripts)
│   ├── ubs/                # UBS-level branch
│   └── entries/            # entry-level branch
├── 04_p2rank/              # P2Rank analysis
│   ├── dataset/            # build the ST_100 AFDB P2Rank input dataset
│   ├── analysis/           # analyze + plot P2Rank success
│   └── prep/               # legacy P2Rank input-prep utilities
├── 06_lbs_generality/      # 2nd/3rd binding-site predictor (Analysis 7)
│   ├── docs/               # methods summary + survey + recommendation (LBS_GENERALITY_DESIGN)
│   ├── schemas/            # normalized predictions + per-pocket success contracts
│   ├── dataset/            # ground-truth atoms builder + input lists + per-tool adapters
│   ├── score/              # tool-agnostic DCA/TNS scorer (reproduces P2Rank's metric)
│   └── analysis/           # pool per-tool tables into the generality panel
├── 05_docking/             # docking on AFDB pockets (Analysis 6)
│   ├── docs/               # method summary (DOCKING_DESIGN)
│   ├── schemas/            # manifest (input) + results (output) contracts
│   ├── dataset/            # the adapter: build the docking manifest / bundle
│   ├── runner/             # CPU pilot runner + docking analyses and plots
│   ├── hpc/                # GPU full-run container + SLURM array scripts
│   ├── redock/             # redocking RMSD against experimental ligand poses
│   └── analysis/           # ingest results + correlate with apo/holo class
├── plotting/               # main + SI manuscript figure scripts
├── requirements.txt / environment.yml
└── README.md
```

---

## Script → figure / table map

Figure numbers are those of the submitted manuscript. Script file names keep their historical
figure labels (`figure1_main_REV4.py` draws panels of manuscript Figure 2, and so on), so go by
this table, not by the file name. Where a figure is composed from several panels, the
composer is in [`plotting/manuscript_v4/`](plotting/manuscript_v4/) (marked ‡); it calls the
panel generators listed, and its README gives the exact commands and options.

| Manuscript figure / table | Generator(s) in this repository |
|---|---|
| Table 1 | `plotting/manuscript_v4/make_table1_v4.py` |
| Figure 1 | `plotting/figure1/make_figure1.py --out_dir <dir>` |
| Figure 2 | ‡ `make_fig2_v4.py --render`: a–b, c: `plotting/figure1_main_REV4.py --thresholds ST_100`; d–e: `plotting/figure2_main_REV1.py --thresholds ST_100`; f: `plotting/Figure3A_ridgeline_REV2.py --thresholds ST_100` |
| Figure 3 | ‡ `make_combined_figure.py --layout v4 --overall bars`, from the per-pocket tables of `06_lbs_generality/` and `05_docking/` (`AHOJ_AF_MODEL_SET=afdbv4`) |
| Figure 4 | ‡ `make_fig4_v4.py --render`: a–b: ‡ `make_alphafill_panels_v4.py` (plotting functions of `03_alphafill/ubs/plot_clash_ridgelines.py`); c–d: `06_lbs_generality/analysis/plot_separation_gap_panels.py`; e: `plotting/figure3C_bars_REV18.py --show_all_group` (`AHOJ_AF_MODEL_SET=afdbv4`) |
| Figure 5 | `plotting/figure5_plddt_reliability.py` |
| Suppl. Figure S1 | `plotting/figureS1_apoholo_pctdiff.py` |
| Suppl. Figure S2 | `plotting/figureS1_supplementary.py` |
| Suppl. Figure S3 | `plotting/figureS2_supplementary_REV4.py` |
| Suppl. Figure S4 | `04_p2rank/analysis/figure_p2rank_size_controlled_class.py` (`AHOJ_AF_MODEL_SET=afdbv4`) |
| Suppl. Figure S5 | `06_lbs_generality/analysis/plot_cutoff_robustness.py` (`AHOJ_AF_MODEL_SET=afdbv4`) |
| Suppl. Figure S6 | `06_lbs_generality/analysis/analyze_lbs_ratio_vs_class.py` (`AHOJ_AF_MODEL_SET=afdbv4`) |
| Suppl. Figure S7 | ‡ `make_si_alphafill_anyscope_v4.py` (2×2): `03_alphafill/ubs/plot_clash_ridgelines.py`, match scope `any`, four half-violin panels |
| Suppl. Figure S8 | ‡ `make_si_placement_v4.py`: a: `05_docking/runner/plot_docking_ridgelines.py`; b–d: `05_docking/redock/plot_remodeled_fig5.py` (`AHOJ_AF_MODEL_SET=afdbv4`) |
| Suppl. Figure S9 | ‡ `make_si_entry_example_v4.py`: a–c: the classification code's per-entry figure (`figure_pocket_example.py`, see [`00_classification/`](00_classification/)); d: ‡ `render_si_entry_pymol_v4.py` (PyMOL) |
| Suppl. Figure S10 | `06_lbs_generality/analysis/plot_fig6a_quality_gates.py --wp_series very_high,high+,moderate+` (`AHOJ_AF_MODEL_SET=afdbv4`) |

Scripts in `plotting/` that appear in no row (`Figure3B_sankey_REV1.py`, the `figureS3_*` family,
`plot_ratio_vs_separation.py`) drew figures of earlier drafts or are diagnostics.

Per-stage details (inputs, outputs, which table each script reads/writes) are in each
directory's own `README.md`.

---

## Getting started

### 1. Install dependencies
```bash
pip install -r requirements.txt
# or:  conda env create -f environment.yml && conda activate ahoj-alphafold
```
Docking and the extra binding-site predictors need their own environments:
[`05_docking/environment.docking.yml`](05_docking/environment.docking.yml) (RDKit, Meeko,
AutoDock Vina, spyrmsd) and [`06_lbs_generality/environment.lbs.yml`](06_lbs_generality/environment.lbs.yml);
the cluster runs used the Apptainer definitions in `05_docking/hpc/` and `06_lbs_generality/hpc/`.

### 2. Configure data paths (once)
All machine-specific paths are centralized in [`common/`](common/). Copy the template and
fill in your local roots (this file is git-ignored and never published):
```bash
cp common/paths.example.ini common/paths.ini   # then edit the roots
```
…or set the equivalent environment variables (`AHOJ_AF_WORK_DIR`, `AHOJ_DB_DIR`,
`AFDB_STRUCTURES_DIR`, `P2RANK_PRED_DIR`, `UBS_ROOT_DIR`, and the optional ones listed in the
template). Scripts import these via
`common/config.py`; see [`common/README.md`](common/README.md).

### 3. Run a script
Scripts are standalone — open one in Spyder and run it, or from a shell:
```bash
python plotting/figure2_main_REV1.py
```
Settings are top-of-file globals (or argparse defaults). Input/output locations resolve
from your `common/paths.ini`, so no code edits are needed to point at your data.

---

## Data availability

The heavy inputs and outputs (classified survivor tables, the unified analysis table,
AlphaFill JSONs, AlphaFold-DB models, P2Rank predictions, derived figures/tables) are **not
tracked in this repository**. They live in an external project `work_dir` and the AHoJ-DB
database directory, referenced through `common/config.py`. The repository tracks **code and
documentation only**. The tables, predictions, docking results and AlphaFold models behind the
manuscript are deposited at Zenodo: [10.5281/zenodo.23243349](https://doi.org/10.5281/zenodo.23243349).

---

## Classification code

The probabilistic apo/holo classifier (the KDE density-ratio scorer over the four pocket metrics,
developed with AHoJ-DB) is in [`00_classification/kde_classifier/`](00_classification/kde_classifier/),
copied from the authors' development repository at a pinned commit; its README gives the commit,
the run configuration of the published run and the checks that the copy reproduces it.
[`00_classification/filter_survivors.py`](00_classification/filter_survivors.py) then selects the
analysis set from its output.

---

## License & citation

- **License:** [MIT](LICENSE).
- **Citation:** see [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository"). The preprint
  DOI will be added when it is posted. Please also cite **AHoJ-DB** and **AlphaFill** where appropriate.

---

## Acknowledgements

Built on data and tools from AlphaFold-DB, AHoJ / AHoJ-DB, AlphaFill, and P2Rank. _Funding and
author acknowledgements to be added on release._
