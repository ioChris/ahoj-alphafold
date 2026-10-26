# AFDB Dataset Workflow

This folder contains the current workflow for building AFDB-focused pocket datasets
from the unified AHoJ/AFDB analysis tables.

The intended pipeline is:

1. Build an ST_100 dataset from the current authoritative tables:
   - `analysis_unified_levels.csv/.zip`
   - `ubs_allthresholds_aggregated_classifications_filtered.csv/.zip`
   - `UBS_root_2/<UniProt>/ST_*.tsv`
2. Add AFDB residue selections, geometric centers, center-of-mass coordinates,
   and radius of gyration from AlphaFold DB mmCIF files.
3. Export P2Rank-ready subsets with normalized center and residue-set column
   names.

Scripts:

- `build_st100_afdb_dataset.py`
  - Creates a merged, analysis-ready ST_100 dataset for single-UniProt,
    single-chain, monomeric UBS rows.
  - Keeps selected pocket-level classification/confidence metrics from the
    unified table.
  - Adds per-UniProt UBS counts from the UBS root threshold files, including
    `ST_030`, `ST_050`, `ST_070`, `ST_100`, and the UniProt-specific `ST_LJ*`
    file normalized into `*_ST_LJ` columns.

- `compute_afdb_pocket_geometry.py`
  - Reads the dataset produced above.
  - Locates AFDB mmCIF files for each UniProt.
  - Maps union/intersection UniProt residues onto the AFDB model.
  - Appends residue selections (`A_50` style), centers, atom counts, and
    radius of gyration.

- `export_p2rank_subsets.py`
  - Reads the geometry dataset.
  - Exports P2Rank-ready subset CSVs with a single residue set and a single
    center triplet per file.
  - Currently writes geometric-center variants for both the union and
    intersection residue sets.

Notes:

- `union_residues` and `intersection_residues` remain in UniProt-coordinate
  form for provenance.
- `union_chain_resi` and `intersection_chain_resi` are the AFDB-facing residue
  selections used downstream for structure-based validation.
