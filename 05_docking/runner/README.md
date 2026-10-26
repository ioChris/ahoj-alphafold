# `05_docking/runner/` — CPU docking runner (AutoDock Vina baseline)

Self-contained runner that consumes the **handover bundle** (see
`../docs`/`../schemas`) and produces `docking_results.csv` per the output contract
(`05_docking/schemas/docking_results.schema.md`). Used for the **pilot** on this
machine (CPU-only); the same code scales to the full run, and the GPU colleague run
can swap the engine behind `run_vina()`.

## Toolchain
Conda env (`environment.docking.yml`): `rdkit`, `gemmi`, `openbabel`, `numpy`,
`pandas`, `scipy`, plus `meeko` + `spyrmsd` (pip). Engine = **AutoDock Vina 1.2.5**
standalone binary (`--vina path/to/vina[.exe]`). No GPU needed.

## Pipeline
```
prep_common.py                 shared prep/geometry/Vina helpers (no repo deps)
rmsd_utils.py                  symmetry-corrected, in-place redock RMSD (graph isomorphism)
prep_receptors.py              1013 AFDB cif.gz -> chain-A PDB (gemmi) -> rigid PDBQT (Meeko)
prep_ligands.py                1185 CCD SDF -> PDBQT (Meeko) + heavy-atom counts
build_reference_poses_local.py group-1 experimental ligand -> AF frame -> template SDF (local escorts)
run_docking.py                 dock every manifest job (Vina), parse affinities, pose_in_pocket,
                               export pose SDFs; per-job JSON -> resumable; aggregate -> docking_results.csv
compute_redock_rmsd.py         docked pose vs reference pose -> redock_rmsd_top/best
analyze_docking_pilot.py       join classification + P2Rank + AlphaFill + RMSD -> tables/figures/REPORT
```

## Run (pilot)
```bash
ENV=/path/to/ahoj-docking/python ; BUNDLE=.../pilot_handover ; WORK=.../work ; VINA=.../vina.exe
$ENV prep_receptors.py --bundle $BUNDLE --workdir $WORK --workers 14
$ENV prep_ligands.py   --bundle $BUNDLE --workdir $WORK --workers 14
$ENV build_reference_poses_local.py --manifest $BUNDLE/docking_manifest.csv --bundle $BUNDLE \
     --out_dir $WORK/ref_poses --workers 14
$ENV run_docking.py --manifest $BUNDLE/docking_manifest.csv --bundle $BUNDLE --workdir $WORK \
     --vina $VINA --out .../results --exhaustiveness 8 --workers 26 \
     --prep_ligands_csv $WORK/prep_ligands.csv
$ENV compute_redock_rmsd.py --poses .../results/poses --refs $WORK/ref_poses \
     --out .../results/redock_rmsd.csv --workers 14
# analysis (base env with matplotlib/seaborn):
python analyze_docking_pilot.py --results_csv .../results/docking_results.csv \
     --pilot_manifest .../pilot/docking_manifest.csv --geometry_csv ... --p2rank_csv ... \
     --alphafill_csv ... --redock_csv .../results/redock_rmsd.csv --out_dir .../analysis_v1
```

## Pilot parameter choices (recorded deviations)
* **exhaustiveness = 8** (Vina default), overriding the manifest's suggested 16.
  The 16 setting is impractical for large flexible cofactors on CPU (a single NADPH
  in a 28 Å box did not finish in 10 min at exh 16/1 cpu; exh 8 finishes in <1 min).
  Boxes, seed (42) and num_modes (9) are used **verbatim** from the manifest.
* cpu=1 per Vina job, jobs run big-ligand-first across 26 workers to avoid a tail.
* `pose_in_pocket` = ≥3 top-pose heavy atoms within 4 Å of the manifest pocket residues.
* `dockable` (binary success) = `pose_in_pocket AND affinity ≤ −6 kcal/mol`.
* redock success = symmetry-corrected in-place RMSD ≤ 2 Å to the experimental pose.

Everything is resumable (per-job JSON under `work/job_results/`; re-run to fill gaps).
