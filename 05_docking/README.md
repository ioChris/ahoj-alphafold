# 05 — Docking on AFDB pockets

**Analysis 6 (new).** Dock biologically-relevant ligands into AlphaFold-DB model pockets and
relate docking success (binding affinity + pose recovery) to the apo/holo classification —
the most direct functional test of the project's central claim, complementing P2Rank
(detectability) and AlphaFill (transplant fit).

> **Read first:** [`docs/DOCKING_DESIGN.md`](docs/DOCKING_DESIGN.md) — methods-level summary
> (rationale, ligand groups, engine, box, metrics, controls). The **full** design doc and the
> colleague **handover/runbook** live in the private project repo (`ahoj-af-project/docking/`,
> experiment T3-redock); the code repo stays code-only.

Paths come from [`../common/config.py`](../common/config.py) (`DOCKING_*`). Nothing is
hard-coded to a classification run: after a new round regenerates
`AFDB_ST100_DATASET_GEOM_CSV` (+ the merged AlphaFill table), just re-run the adapter.

## What this stage does (our side)
Build a **handover bundle** — a per-(pocket × ligand) docking manifest + receptor/ligand
lists + resolved ligand files — that a colleague runs with AutoDock Vina (baseline) or
Uni-Dock (GPU, full run). We then ingest the returned results and correlate.

## Ligand groups (see design doc §2)
1. **Query holo ligands from AHoJ-DB** (from `rep_job`/`job_names` job keys) — MAIN.
2. **AlphaFill cognate CCDs** (from the merged AHoJ↔AlphaFill table) — cross-check.
3. **Probe library** (fixed drug-like set) — ligand-controlled dockability.

## Directory layout
```
05_docking/
├── docs/
│   └── DOCKING_DESIGN.md        # methods-level summary (full design + handover are private: ahoj-af-project/docking/)
├── schemas/
│   ├── docking_manifest.schema.md   # input contract (handover manifest columns)
│   └── docking_results.schema.md    # output contract (what we need back)
├── dataset/                     # the ADAPTER (build the handover bundle) — see dataset/README.md
│   ├── parse_ahoj_job_ligands.py    # job keys -> query holo ligands (group 1)
│   ├── build_docking_manifest.py    # geometry table -> per-job manifest + receptors/ligands
│   ├── sample_pilot_pockets.py      # stratified pilot subset (size × class × separation)
│   ├── resolve_ligands.py           # CCD -> SDF + probe embedding (+ default probe library)
│   └── build_reference_poses.py     # SCAFFOLD: experimental ligand -> AF frame (redock RMSD)
└── analysis/
    └── ingest_docking_results.py    # results -> correlations vs apo/holo (size-controlled)
```

## Quickstart — build a pilot bundle
Prerequisites: `common/paths.ini` configured (as for the rest of the repo) and an env with
`pandas`/`numpy` (adapter), `gemmi` (reference poses / bbox), `rdkit` (probe embedding).
See [`environment.docking.yml`](environment.docking.yml) — a one-stop env
(`rdkit` + `meeko` + `vina` + `gemmi`) for prep on our side and the colleague's Vina run.

```bash
# 0) (once) configure common/paths.ini as for the rest of the repo.
cd 05_docking/dataset

# 1) draw a stratified pilot sample (~1500 pockets across size × class × separation)
python sample_pilot_pockets.py --target_n 1500

# 2) build the manifest for those pockets (group 1 = main; add 2,3 as available)
python build_docking_manifest.py --groups 1,2,3 \
    --pilot_site_uids "$AHOJ_AF_WORK_DIR/docking/pilot/pilot_pockets.csv" \
    --output_dir "$AHOJ_AF_WORK_DIR/docking/pilot"

# 3) resolve ligands to 3D (fetch CCD ideal SDFs + embed probes)
python resolve_ligands.py --ligands_csv "$AHOJ_AF_WORK_DIR/docking/pilot/ligands.csv" \
    --out_dir "$AHOJ_AF_WORK_DIR/docking/pilot/ligands"

# -> hand the pilot/ folder to the colleague (runbook: private ahoj-af-project/docking/HANDOVER_GUIDE.md)

# 4) after results return:
cd ../analysis
python ingest_docking_results.py --results_csv "$AHOJ_AF_WORK_DIR/docking/results/docking_results.csv"
```

> Status: dataset adapter + analysis ingest are first-pass and follow the repo's existing
> idioms (cfg bootstrap, argparse, gemmi/pandas), but have not been executed in this
> environment — smoke-test on a small `--pilot_site_uids` subset first.
> `build_reference_poses.py` is a documented **scaffold** (see its header).
