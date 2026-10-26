# plotting/manuscript_v4 — figure composers of the submitted manuscript

The tools here compose the manuscript figures that are built from several panels, and write
Table 1. Each composer either calls the panel generators elsewhere in this repository
(`--render`) or places panels they have already drawn. Panels are drawn at the millimetre size
they occupy on the page and are placed without resizing; SI Figure S7 is the one exception (see
below). With the data roots configured, these tools and the single-panel scripts listed in the
[top-level figure map](../../README.md#script--figure--table-map) regenerate every figure of
the manuscript.

| Manuscript | Composer | Panels drawn by |
|---|---|---|
| Figure 2 | `make_fig2_v4.py --render` | a–b, c: `plotting/figure1_main_REV4.py`; d–e: `plotting/figure2_main_REV1.py`; f: `plotting/Figure3A_ridgeline_REV2.py` |
| Figure 3 | `make_combined_figure.py --layout v4 --overall bars` | itself (from the per-pocket tables of `06_lbs_generality/` and `05_docking/`) |
| Figure 4 | `make_fig4_v4.py --render` | a–b: `make_alphafill_panels_v4.py` (calls `03_alphafill/ubs/plot_clash_ridgelines.py`); c–d: `06_lbs_generality/analysis/plot_separation_gap_panels.py`; e: `plotting/figure3C_bars_REV18.py` |
| Suppl. Figure S7 | `make_si_alphafill_anyscope_v4.py` | `03_alphafill/ubs/plot_clash_ridgelines.py`, match scope `any` (four metrics) |
| Suppl. Figure S8 | `make_si_placement_v4.py` | a: `05_docking/runner/plot_docking_ridgelines.py`; b–d: `05_docking/redock/plot_remodeled_fig5.py` |
| Suppl. Figure S9 | `make_si_entry_example_v4.py` | a–c: `figure_pocket_example.py` of the classification code (`00_classification/kde_classifier/`); d: `render_si_entry_pymol_v4.py` (PyMOL) |
| Table 1 | `make_table1_v4.py` | — (counts from the master table) |

`mv4_paths.py` holds the roots the tools share. It is not a script.

## Setup

* **Data roots** come from [`common/config.py`](../../common/config.py), configured in
  `common/paths.ini` or by environment variables (see the top-level README). The composers read:
  `work_dir` (all of them), `example_entries_dir` and `entry_example_pse` (SI Figure S9 only).
* **Output folder**: the `manuscript_figures_dir` anchor (`AHOJ_AF_MANUSCRIPT_FIGURES_DIR`); if it is
  not set, `<work_dir>/unified_output/run5/plots/manuscript_v4`. Main figures go into that folder,
  SI figures into `SI/`, and rendered panels into `parts/`. With `AHOJ_AF_MODEL_SET=afdbv4` the folder
  name gets an `_afdbv4` suffix, so the two model sets never overwrite each other. Every composer
  also takes `--out_dir` (and `--parts_dir` where it renders parts).
* **Model set.** Figures 3 and 4 and SI Figure S8 were drawn from the re-run of the detection and
  docking arms on AlphaFold DB v4 models: run them with `AHOJ_AF_MODEL_SET=afdbv4`. The other outputs
  do not depend on the model files.
* **Interpreters.** The panel generators run under `AHOJ_AF_PLOT_PYEXE`, or under the interpreter that
  runs the composer if that is unset. The figures were made in a conda env with Python 3.11,
  matplotlib 3.10.0, Pillow 11.3, numpy 2.3, pandas 2.3, scipy. SI Figure S9d needs open-source PyMOL
  (`pymol-open-source` 3.1.0 was used) in the interpreter named by `AHOJ_AF_PYMOL_PYEXE`.
* **Fonts.** The panels use the font stack of `common/figspec.py` (Helvetica Neue first); a machine
  without those fonts renders with a fallback, so text metrics and pixels differ slightly.
* **Classification code.** SI Figure S9 a–c imports `figure_pocket_example.py` and its modules from
  `00_classification/kde_classifier/`; `AHOJ_AF_KDE_CLASSIFIER_DIR` points elsewhere.

## Commands

Run from the repository root. On Windows PowerShell set the variable with
`$env:AHOJ_AF_MODEL_SET = "afdbv4"` first instead of the `VAR=value` prefix.

```bash
# Figure 2: render the four parts (parts/Fig2ab, Fig2c, Fig2de, Fig2f_ridgeline_ST_100), then compose
python plotting/manuscript_v4/make_fig2_v4.py --render

# Figure 3 (png, pdf, svg)
AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_combined_figure.py --layout v4 --overall bars

# Figure 4: render a-b (make_alphafill_panels_v4.py), c-d and e into parts/, then compose
AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_fig4_v4.py --render

# Suppl. Figure S7: compose the four any-scope half-violins the AlphaFill analysis already drew ...
python plotting/manuscript_v4/make_si_alphafill_anyscope_v4.py
# ... or re-draw them first (into parts/si_alphafill_anyscope/ridgeline_plots; current styling, see below)
python plotting/manuscript_v4/make_si_alphafill_anyscope_v4.py --render

# Suppl. Figure S8: render the four parts into parts/si_placement, then compose
AHOJ_AF_MODEL_SET=afdbv4 python plotting/manuscript_v4/make_si_placement_v4.py

# Suppl. Figure S9 (png, pdf): PyMOL render of d into parts/si_entry_example, then compose
AHOJ_AF_PYMOL_PYEXE=/path/to/pymol-env/python python plotting/manuscript_v4/make_si_entry_example_v4.py

# Table 1 (markdown; printed, and written with --out)
python plotting/manuscript_v4/make_table1_v4.py --out Table1_analysis_set.md
```

Without `--render`, `make_fig2_v4.py` and `make_fig4_v4.py` compose the parts already in the parts
folder; `make_si_placement_v4.py --compose_only` and `make_si_entry_example_v4.py --compose_only`
do the same.

The panel generators each composer runs, with the options it passes (sizes in mm or inches as the
generator expects; `<parts>` is the parts folder):

* **Figure 2** — `plotting/figure1_main_REV4.py --columns 2,4 --thresholds ST_100 --letter_start a
  --hide_row_labels --col2_overall --letters_at_ylabel --count_ylab_gap 9.5 --exact_canvas`
  (a–b); the same script with `--columns 3 --letter_start c` (c); `plotting/figure2_main_REV1.py
  --columns 1,2 --letter_start d` (d–e); `plotting/Figure3A_ridgeline_REV2.py --panel_letter f
  --hide_threshold_annotation --exact_canvas` (f). Each gets `--fig_size` at its slot's size and
  `--output_dir <parts> --out_basename <name>`.
* **Figure 4** — `make_alphafill_panels_v4.py` (half-violins of `per_ubs_summary_same_uniprot_ge_0p30.csv`,
  ST_100, clash count and clash score, 88 × 62 mm each); `06_lbs_generality/analysis/plot_separation_gap_panels.py
  --panel both --letter c,d --fig_size 89.00,58.00 --boot 300`; `plotting/figure3C_bars_REV18.py
  --thresholds ST_100 --show_all_group --hide_class_legend --panel_letter e`.
* **Suppl. Figure S7** — `03_alphafill/ubs/plot_clash_ridgelines.py --thresholds ST_100 --scopes any
  --identity_tags ge_0p30 --make_violins --metric <m>` for `avg_clash_count`, `avg_global_rmsd`,
  `avg_clash`, `avg_local_rmsd`; the composer scales the four to one width and adds the letters.
  The submitted S7 was composed from panels this generator drew on 2026-07-05, which still sit in
  `<work_dir>/alphafill_unified/run2/analysis_v1/ridgeline_plots/`; the default command composes
  those. The generator has since moved to the Figure 4 styling (600 dpi, lowercase bins, the
  "Apo-holo separation" axis title, sentence-case axis labels), so `--render` gives the same data and
  counts but a restyled, larger image on which the 64 px letters come out small.
* **Suppl. Figure S8** — `05_docking/runner/plot_docking_ridgelines.py --which separation --all_position last
  --floor -12 --headroom 1.15 --canvas_mm 88,62 --letter a --no_title --stacked_counts` on
  `<DOCKING_RUN3_DIR>/analysis/docking_run3_merged.csv`; `05_docking/redock/plot_remodeled_fig5.py
  --v4_panels_dir <parts> --v4_letters b,c,d`.
* **Suppl. Figure S9** — `render_si_entry_pymol_v4.py --out_dir <parts> --w_px 1031 --h_px 774 --dpi 600`
  (three ray-traced views from the session's scene `003`; the session's md5 is checked).

## Output names

Outputs carry the submitted manuscript's numbers: `Figure2_apo_holo_and_what_sets_it.png`,
`Figure3_detection_and_docking.{png,pdf,svg}`, `Figure4_when_does_it_matter.png`,
`SI/FigureS7_alphafill_anyscope.png`, `SI/FigureS8_ligand_placement.png`,
`SI/FigureS9_entry_example.{png,pdf}`. The authors' private build named the three SI figures
`FigureS8_alphafill_anyscope`, `FigureS9_ligand_placement` and `FigureS10_entry_example` (numbers from
before the SI was renumbered); the images are the same.
