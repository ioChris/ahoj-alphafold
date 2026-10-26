# Docking on AFDB pockets — method summary

**Analysis 6.** Dock biologically-relevant ligands into AlphaFold-DB (AFDB)
model pockets and relate docking success to the apo/holo classification. This is the
public, methods-level summary; the full design rationale, hypotheses, cost/scale analysis,
open decisions, and the colleague handover guide are kept in the private project repo (see
[Provenance](#provenance)).

## Purpose
A single AF model sits at a biased point on each protein's experimental apo↔holo continuum.
Docking is the direct functional test of the consequence: **can a real ligand be placed and
energetically scored in that model's pocket, and does the model's apo/holo state predict how
well?** It complements P2Rank (pocket *detectability*) and AlphaFill (ligand *transplant fit*)
by adding a scored pose and, for cognate ligands, a pose-recovery RMSD.

## What is docked (three ligand groups)
1. **Query holo ligands (main)** — the AHoJ-DB query ligands that define each UBS (BioLiP-based,
   bound at exactly the ST_100 residues). Parsed from the `rep_job`/`job_names` job keys
   (`{pdb}-{chain}-{CCD}-{resnum}`; the CCD is docked, and chain/resnum — the ligand's PDB
   `auth_asym_id`/`auth_seq_id` — locate its experimental pose).
2. **AlphaFill cognate CCDs** — template-validated cognate ligands from the merged
   AHoJ↔AlphaFill table (a defensible cross-check with a ready reference pose).
3. **Probe library** — a fixed drug-like set docked into every pocket, controlling for ligand
   identity (dockability).

Monatomic ions and common buffers/cryoprotectants (biologically real but not meaningfully
dockable) are filtered from groups 1–2 by default.

## Receptor & box
Receptors are the project's AFDB monomer models (`AF-{UniProt}-F1-model_v*.cif.gz`, chain A);
the ST_100 single-UniProt/single-chain/monomeric/non-interface filter is inherited. Docking is
**targeted** (box-constrained) at the known pocket: box **center** = pocket-residue
center-of-mass (`union_center_com`); box **size** from pocket radius of gyration
(`2·rg + ligand_pad`, clamped 16–30 Å; median ≈ 24 Å on ST_100), or a tighter per-axis
bounding box when the AF model is re-opened.

## Engine
Engine-agnostic manifest. **AutoDock Vina 1.2.x** is the reproducible baseline (pilot);
**Uni-Dock** (GPU-batched, same scoring family) is the natural scale-up for the full run.
gnina CNN rescoring is an optional secondary metric. Toolchain (rdkit/meeko/vina/gemmi):
[`../environment.docking.yml`](../environment.docking.yml); Uni-Dock is installed separately (GPU).

## Outcomes & success
Per job: top binding affinity (kcal/mol) + all modes; top-pose in-pocket localization; and,
for cognate ligands with a reference pose, **redocking RMSD** (success ≤ 2 Å). Continuous
outcomes (affinity, RMSD) are primary for correlation; binary success rates for headline
sentences (mirroring the P2Rank `TNS` presentation).

## Analysis & controls
Outcomes are correlated (per ligand group) against `pocket_class`, `pocket_p_holo`,
`pocket_separation_pair`, and `model_pocket_plddt`, and cross-checked against AlphaFill clash
and P2Rank `TNS`. The central confound — pocket size (and ligand size) drive both class and
docking — is controlled from the start (partial correlation / controlled-pairs / CMH by size,
mirroring `04_p2rank/analysis/analyze_p2rank_controlled_pairs.py`), leaning on the fixed-probe
group where ligand identity is constant.

## I/O contracts
- Input manifest: [`../schemas/docking_manifest.schema.md`](../schemas/docking_manifest.schema.md)
- Returned results: [`../schemas/docking_results.schema.md`](../schemas/docking_results.schema.md)

## Provenance
The **full design doc** (rationale, hypotheses, scale/compute, controls in depth, open
decisions) and the **colleague handover/runbook** live in the private project repo
(`ahoj-af-project/docking/`), tracked as experiment **T3-redock** in its `EXPERIMENTS.md`.
This public stage ships the reproducibility artifacts only (adapter, schemas, ingest).
