# Binding-site unification (Unified Binding Sites, UBS)

Clusters the ligand-binding sites of each UniProt accession by the Jaccard similarity of their
UniProt residue sets and writes one table per similarity threshold (`ST_100`, `ST_070`,
`ST_050`, `ST_030`) plus the large-jump cut (`ST_LJxxx`). These per-UniProt tables are the
`UBS_ROOT_DIR` input of [`../classification_aggregation_and_interface_ubs_clustering_r.py`](../classification_aggregation_and_interface_ubs_clustering_r.py),
which aggregates the apo/holo classification per UBS (Analysis 2).

**Provenance.** Copied from the authors' `unify-binding-sites` repository at commit `8c11d16`
(2026-09-19), the version whose defaults reproduce the code that wrote the published tables
(`UBS_root_2`, 29,463 accessions, written 2025-09-02). Checked: re-running this copy on the
published per-UniProt inputs (`<ACC>/input_data.csv.gz`) gives byte-identical `ST_*.tsv`
tables for all 320 accessions tried (300 at random plus the 20 with the largest inputs; see
"Reproducing the published tables" below). Later development in that repository
(canonical ordering, nested/pocket/region levels) changes the tables and is not part of this
study.

**The published run** used the defaults (thresholds `0.3 0.5 0.7 1.0`) on the per-entry
UniProt residue table `cfg.QUERY_SUMMARIES_CLASS_CSV`
(`<AHoJ-DB>/Summaries/extract_uniprot_residues/run_27062025_130510_do_collect_uniprot_residues/query_summaries_class.csv`):

```bash
python 01_ubs_clustering/unification/run_ubs_unification.py --input <query_summaries_class.csv> --out <UBS_root>
```

### Reproducing the published tables

Each published accession folder keeps its own input as `input_data.csv.gz`; unzip one and run
`run_ubs_unification.py --input input_data.csv --out <dir> --no-figures`, then compare
`<dir>/<ACC>/ST_*.tsv` with the published ones. The tables are in the Zenodo record
(10.5281/zenodo.23243349).

---

## Layout

```
.
├── run_ubs_unification.py   # main pipeline (CLI + run_pipeline())
├── collect_ubs.py           # collector: per-UniProt tables -> one sites.tsv (CLI + collect_ubs())
├── ubs_utils.py             # clustering, plotting, compression, manifest helpers
├── tests/                   # pytest suite on a synthetic CSV
└── README.md
```

## Requirements

- Python 3.9+
- pandas, numpy, scipy, matplotlib (matplotlib is imported even in figures-off mode)
- pytest (tests only)

Install:
```bash
pip install pandas numpy scipy matplotlib pytest
```

The optional font helper `set_plot_font` is taken from `../../00_classification/kde_classifier/stats_plot.py` when it imports; otherwise a no-op is used (figures render with matplotlib's default font).

## Input data

A CSV with:
- `job_name`: binding-site entry ID (unique per row)
- `UNPs`: UniProt ID(s); if multiple, space-separated (interfaces)
- `uniprot_residues`: Python list literal of tokens like `["P12345_50","P12345_51"]` (`ACC_idx`)
- `prot_chains_num`, `UNPs_num`: used for the multimeric status roll-up
- `pdb`: used to count unique PDBs for the figure annotation
- `ligands` (optional): used by the collector's `--input` option

Interfaces are automatically **split per UniProt** (`ubs_utils.split_interface_entries`): each accession of an interface row gets its own row holding only its `ACC_` tokens, so each run handles one UniProt at a time. The parsing of these columns is unchanged from the previous version.

## Running

```bash
python run_ubs_unification.py --input ubs_input.csv --out UBS_root
```

is the previous default run: thresholds `0.3 0.5 0.7 1.0`, one worker, `gz` archives, figures and `tree.pkl`, log file in `UBS_root`.

```
python run_ubs_unification.py --input CSV --out DIR
                              [--thresholds T [T ...]]      fixed similarity thresholds in (0, 1]; default 0.3 0.5 0.7 1.0
                              [--workers N]                 worker processes; default 1
                              [--subset ACC [ACC ...]]      only these accessions (interface rows involving them are kept)
                              [--no-figures]                tables, input_data and manifest only
                              [--compression {zip,gz}]      archive format; default gz
                              [--log-dir DIR]               run log location; default: the output root
```

Large table-only run for another pipeline:
```bash
python run_ubs_unification.py --input ubs_input.csv --out UBS_root --thresholds 1.0 0.7 --workers 8 --no-figures
```

Notes:
- The large-jump cut (`ST_LJXXX.tsv`) is always computed for accessions with ≥ 3 entries, whatever `--thresholds` says.
- `ST_100.tsv` is written when `1.0` is among `--thresholds` (it is by default).
- The linkage tree, the thresholds and the `ubs_id` numbering do not depend on `--no-figures`, `--workers` or `--subset`.

The pipeline:
1) Loads and splits entries by UniProt (interfaces become separate per-UniProt rows).
2) Clusters per UniProt using average-linkage hierarchical clustering on Jaccard distance. Computes **LJ** threshold when ≥ 2 linkage steps exist.
3) Writes TSVs for each fixed **ST_XXX** and for **ST_LJXXX**.
4) Unless `--no-figures`: saves dendrogram + `tree.pkl` (SVG, compressed; ≥ 2 entries) and binding-position histogram (SVG, compressed).
5) Writes a **`ubs_manifest.json`** that lists the generated files (`figures` is `{}` in figures-off mode).

### From Python

```python
from run_ubs_unification import run_pipeline
run_pipeline("ubs_input.csv", "UBS_root", thresholds=[1.0, 0.7], workers=8, figures=False)
```

`run_pipeline(input_file, out_dir, thresholds=(0.3, 0.5, 0.7, 1.0), workers=1, subset=None, figures=True, compression="gz", log_dir=None)`.

## Collecting one table (`collect_ubs.py`)

```bash
python collect_ubs.py UBS_root --threshold ST_100 --out sites.tsv --input ubs_input.csv
```

walks `UBS_root/<ACC>/` and writes one row per UBS of the requested table:

| column | content |
|---|---|
| `site_uid` | `<ACC>:<ST_XXX>:<ubs_id>` (e.g. `P00698:ST_100:3`) |
| `uniprot` | accession (directory name) |
| `residues` | UniProt indices of `total_residues` (the union), ascending, `+`-joined (`10+11+12+13`) |
| `members` | `entry_ids`, `;`-joined in table order |
| `n_members` | `num_entries` |
| `multimeric_status_all` | status codes present among the members, e.g. `1 3` |
| `ligands` | only with `--input`: the members' `ligands` values, `;`-joined in member order |

`--threshold` accepts `ST_100`, `100`, `1.0`, `0.7`, ... or `LJ` for each accession's large-jump table (then `site_uid` carries the actual label, e.g. `P00698:ST_LJ072:1`). Rows are sorted by accession, then `ubs_id`. Without `--out` the TSV goes to stdout.

From Python:
```python
from collect_ubs import collect_ubs
sites = collect_ubs("UBS_root", threshold="ST_100", input_csv="ubs_input.csv")  # pandas DataFrame
```

## Tests

```bash
python -m pytest tests
```

The suite builds a synthetic CSV (three accessions, one interface row) and checks that `--no-figures` yields byte-identical ST tables to the default run (also with `--workers 2`), that figures-off writes only tables, `input_data.csv.gz` and a manifest without figures, that `ST_100` groups identical residue sets only, that the collector reproduces the `ubs_id` groups, residues, ligands and status codes, that `--subset` keeps the interface halves of the requested accession, and that the CLI and `run_pipeline()` defaults match the previous `main()`.

## Outputs (per UniProt directory)

- `ST_XXX.tsv` — UBS table at fixed similarity threshold (e.g., `ST_050.tsv`).
- `ST_LJXXX.tsv` — UBS table at large-jump threshold (e.g., `ST_LJ072.tsv`).
- `dendrogram.svg.[zip|gz]` — annotated dendrogram (if ≥2 entries; not in figures-off mode).
- `binding_positions.svg.[zip|gz]` — prevalence histogram (not in figures-off mode).
- `input_data.csv.[zip|gz]` — original per-UniProt rows for provenance.
- `tree.pkl` — SciPy linkage (if ≥2 entries; not in figures-off mode).
- `ubs_manifest.json` — manifest for web consumption.

### TSV columns (summary)
- `ubs_id` — unified binding site ID (SciPy `fcluster` numbering, consecutive from 1)
- `num_shared_residues`, `num_total_residues`
- `multimeric_status_all` — set of status codes present among entries in the UBS (1 monomer, 2 homomer, 3 heteromer, 4 homo/hetero, 0 unknown)
- `num_entries`, `entry_ids`
- `shared_residues`, `total_residues` — intersection and union of the members' residue tokens

> Note: You intentionally **excluded** the `multimeric_status` mode and name columns in this revision; only the `_all` set is present.

### Manifest schema (example)

```json
{
  "uniprot": "P00698",
  "generatedAt": "2025-09-02T13:17:05",
  "archiveFormat": "gz",
  "thresholds": [
    {
      "JaccardSimilarityThreshold": 1.0,
      "label": "ST_LJ100",
      "type": "optimal",
      "file": "ST_LJ100.tsv",
      "shortDescription": "Optimal similarity threshold (large jump).",
      "longDescription": "Automatically selected threshold based on the largest linkage jump in the dendrogram."
    },
    {
      "JaccardSimilarityThreshold": 0.5,
      "label": "ST_050",
      "type": "fixed",
      "file": "ST_050.tsv",
      "shortDescription": "Fixed similarity threshold (predefined).",
      "longDescription": "Unified Binding Sites computed at a predefined Jaccard similarity threshold."
    }
  ],
  "figures": {
    "dendrogram": "dendrogram.svg.gz",
    "histogram": "binding_positions.svg.gz"
  },
  "inputData": "input_data.csv.gz",
  "numEntries": 42,
  "defaultThresholds": [0.3, 0.5, 0.7, 1.0]
}
```

## Dendrograms

- Axis shows **Jaccard Similarity (%)**; tick labeling converts distance to percent.
- Markers are derived from `--thresholds` (3-digit codes below 100) plus **LJ**; with the default thresholds that is `["030","050","070","LJ"]`.
- **ST lines** use progressively darker greys; **LJ** uses green.
- Legend is fixed (e.g., **upper right**) with a semi-transparent frame and outline.

## Binding-position histograms

- One SVG per UniProt; x starts at **1** (optionally span full sequence if a length column exists).
- “Stepfill” silhouette with **monomer** vs **interface (≥2 protein chains)** stacking.
- Left Y: percentage (of entries); Right Y: counts (auto-synced).
- Major ticks every 10 with labeled midpoints when step ≥20; small gaps at both ends.

## Logging

- All logs (main + workers) go through a **QueueListener** to a single log file (`ubs_unification_<timestamp>.log` in `--log-dir`, default the output root) and the console.
- Works in both **single-process** and **multiprocessing** modes; worker logs include the process name.

## Compression

- Figures and provenance are compressed using `compress_file(path, method="zip"|"gz")`.
- The chosen method is recorded in the manifest as `archiveFormat`.
