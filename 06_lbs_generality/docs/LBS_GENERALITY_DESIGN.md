# LBS generality — method summary, survey, and design

**Analysis 7.** Run additional ligand-binding-site (LBS) predictors on the
same AFDB ST_100 pocket set and recompute the same DCA / **TNS** success as P2Rank, to
demonstrate that the **AF-Holo > AF-Apo detection gradient (claim C4) is not a P2Rank
artifact.** This is the public, methods-level summary; the exhaustive per-tool survey, the
pipeline/benchmark survey, all source URLs, and the full recommendation rationale live in the
private project repo (see [Provenance](#provenance)).

## Purpose (the reviewer question this answers)
C4 reports P2Rank finds binding sites more easily on AF-Holo-like pockets than AF-Apo-like
pockets (run_3: 84.76% vs 66.11% TNS DCA4; success rises with holo probability). A reviewer will ask
whether this is specific to P2Rank's model. T3-2nd answers it directly: if 2–3 predictors with
**different inductive biases** show the same holo>apo gradient on the identical pockets and
identical binding-site definitions, the effect is a property of the AF *structures*, not of one
tool. Note the framing from the 2024 LIGYSIS benchmark (Utgés & Barton): no method
convincingly beats P2Rank on recall, so the claim is **"the gradient generalizes across tools
with different recall/precision profiles,"** not "a better tool exists."

## Metric — one tool-agnostic yardstick (verified)
For each pocket, ground truth = the **union binding-site residue heavy atoms** on the AF model
(the same `union_chain_resi` P2Rank uses). For a predicted pocket with centre *c*:
`DCA = min over ground-truth atoms ||c − a||`; `dca_rank` = rank of the first pocket with
`DCA ≤ 4 Å`; **`TNS = dca_rank > 0 AND dca_rank ≤ n_unp_pockets`**. This centre-to-nearest-
site-atom distance **reproduces P2Rank's own `closest_pocket_dist` to < 1e-4 Å** on shared
ST_100 pockets, so every predictor — **P2Rank re-scored included** — is measured on one scale.
**Confirmed from P2Rank's own eval config** (`params.txt` / `summary.csv`): the ground-truth
"ligand" was the pocket RESIDUE set, not a single point — `ligSize = 97.2 atoms/site` (our
heavy-atom store averages 98.1), `min_ligand_atoms = 5` (a 1-atom point would be rejected), and
`site_eval_center_method = explicit` (an explicit centre *was* supplied — the "CENTER_explicit"
tag — but only as the reference centre, not the DCA target). A single-centre-point ground truth
would have measured **~31% TNS** (apo 22 / holo 36), not the actual ~75% DCA4. P2Rank's *native*
`dca4rank` is looser still because it also treats the site's SAS surface points as ground-truth
atoms (`site_eval_sas_pts_as_atoms = true`; the P2Rank author, R. Krivák, confirmed this projects
DCA onto the site's SAS shell and **inflates AFDB DCA — making it incomparable to PDB DCA** and
unreproducible for tools that don't expose SAS points). So native P2Rank TNS is reported ONLY as a
labelled tool-native reference; the harmonized heavy-atom TNS is the fair cross-tool comparator.
**Rebuilt on run_3 (2026-07-08, full set n=86,946):** re-scoring P2Rank's own centres harmonized
gives **apo 46.78% / holo 63.08%** (gap +16.3 pp) vs native **apo 66.11% / holo 84.76%** (+18.65) —
the ~20-pp absolute drop is the SAS inflation; the holo>apo gradient survives. To match native
P2Rank *exactly*, build the ground-truth store from SAS points (a documented `--sas_expand` option).

## Method survey (candidates that can batch on AF monomers)
Axes: license · CPU/GPU · input · output (ranked pockets? centre? residues?) · batch on ~82k on
one RTX 3090/4090 · maintenance. Fuller detail + URLs are in the private survey.

### Classic geometry / energy-probe (orthogonal to ML)
| Tool | Method | License | Runtime | Input | Output | 82k batch | Verdict |
|---|---|---|---|---|---|---|---|
| **fpocket** | Voronoi alpha-spheres | MIT | CPU, ~s/struct | **mmCIF native (v4)** | ranked pockets, centres + residues + druggability | trivial (`-F` / parallel) | **YES — pick #1** |
| AutoSite (ADFR) | energy fill-point clusters | LGPL | CPU | needs PDBQT | ranked clusters + centres | yes (needs conversion) | yes, caveats |
| Ghecom | math. morphology | LGPL | CPU (WSL) | PDB | pocketness + clusters | yes (custom parse) | yes, caveats |
| CASTp/CASTpFold · DoGSiteScorer · LIGSITE · SURFNET · POCASA · Q-SiteFinder · PASS | various | mixed | — | web/PDB | pockets | **server-only / abandoned** | **no** for batch |

### ML / deep-learning structure-based
| Tool | Bias vs P2Rank | License | GPU/stack | Output | 82k batch | Verdict |
|---|---|---|---|---|---|---|
| **IF-SitePred** | ESM-IF1 **inverse-folding embeddings** → DBSCAN | BSD-3 | GPU opt.; consumes **AF PDB directly**; no MSA | **ranked 3D site centres + residues** | per-structure loop | **YES — pick #2 (cleanest DL drop-in)** |
| **GrASP** | graph attention on surface atoms | MIT | PyG (Py3.7/OpenBabel pins); WSL | ranked centres + residues | dir-based batch | **YES, caveats — pick #3 option** |
| VN-EGNN | equivariant virtual-node centre regression (most orthogonal) | MIT | PyG (modern) | ranked centres + residues | **no turnkey inference — write loop** | with-caveats |
| PUResNetV2.0 | 3D segmentation; **has a P2Rank rescore adapter** | open | MinkowskiEngine (WSL) | pocket points | yes (least glue via P2Rank) | with-caveats |
| DeepPocket | CNN rescoring **on fpocket candidates** | open | libmolgrid (linux-64) | ranked (rescored fpocket) | WSL | less orthogonal (shares fpocket) |
| DeepSurf · PointSite · RecurPocket · Kalasanty | surface/point/segmentation CNN | open | **old TF1 / archived deps** | pockets | install-rot | mostly no |
| PocketMiner | GVP-GNN for **cryptic** residues (different task) | none | TF | per-residue (no centres) | one-at-a-time | orthogonal *cryptic flag*, not a ranker |

### Sequence / embedding (residue-level; a genuinely orthogonal axis)
These emit per-residue binding probabilities, not cavities — score by residue-overlap vs
`union_chain_resi`, or by centroid-of-predicted-residues DCA.
| Tool | Signal | License | MSA? | Output | 82k batch | Verdict |
|---|---|---|---|---|---|---|
| **PeSTo** | parameter-free geometric transformer; **no MSA/PLM** | open | no | per-residue small-molecule + ion interface | fast (structure-native, AF cif) | **yes — best residue-level add** |
| GPSite | ProtT5 + ESMFold + GNN | MIT | no | per-residue, 10 ligand classes (ATP/HEM/metals) | ~1–2 wk (re-folds internally) | yes, caveats |
| bindEmbed21DL | ProtT5 → CNN, structure-free | open | no | per-residue, 3 lumped classes | fastest (hrs) | yes, caveats (coarse) |
| LMetalSite / M-Ionic | PLM, **ions only** | mixed | no | per-residue metal/ion | fast | control for the ion subset only |
| ScanNet (PPI-centric) · GraphSite (DNA / MSA-heavy) · DeepCSeqSite/BiRDS/DELIA (stale/MSA) | — | — | — | — | — | no / weak |

## Pipeline / benchmark survey
- **No ready-made "PoseBusters/CASF for pocket detection"** batch-runs multiple tools *and*
  scores against custom site definitions. Run each tool individually, normalize once, score once.
- **Utgés & Barton "LBS-comparison"** (MIT; J Cheminform 2024) benchmarked exactly this tool set
  (fpocket, P2Rank, PUResNet, DeepPocket, GrASP, VN-EGNN, IF-SitePred, …). It is **not** an
  orchestrated runner (one parser notebook per tool) but ships a **reusable scoring layer**
  (`benchmark_lib.py`: DCC/DCA-by-centroid, residue-overlap, top-N+2 recall) against
  **UniProt-residue** ground truth — which matches our `union_residues` setup. Reusable as an
  alternative scorer to ours.
- **P2Rank's own evaluator**: `eval-predict` scores only P2Rank's predictions (ground truth =
  ligands in file, DCA/DCC default 4 Å). `rescore`/`eval-rescore` has **built-in adapters for
  fpocket and PUResNetV2.0** — so those two can be DCA/DCC-scored by P2Rank directly (a
  cross-check shortcut), but there is no generic "ingest arbitrary centres" mode.
- MetaPocket 2.0 = server-only; CryptoBench = cryptic-specific dataset. Neither is a general
  multi-tool harness.

**Conclusion:** we run the chosen tools individually and reuse **one DCA scoring layer** — our
`score/compute_lbs_success.py`, which is validated to reproduce P2Rank and joins straight to
`pocket_class` / holo probability. Optionally cross-check fpocket / PUResNetV2.0 via P2Rank's
`rescore`.

## Recommendation
- **fpocket** (unsupervised geometry) **+ IF-SitePred** (learned inverse-folding embeddings) as
  the core two — two distinct inductive biases, both output ranked centres → identical DCA/TNS,
  both feasible on the hardware. **Optional third:** **GrASP** (graph-attention centres) or
  **PeSTo** (residue-level, scored by overlap) for a third axis / revision-grade generality.
- **Least-glue variant:** fpocket + **PUResNetV2.0**, both scorable via P2Rank's `rescore`.

## Receptor, ground truth, and controls
Receptors = the project's AFDB monomer models (`AF-{UniProt}-F1-model_v*.cif.gz`, chain A;
same ST_100 monomeric/non-interface filter). Ground truth = `union_chain_resi` heavy atoms.
The central confound — pocket size drives both class and detectability — is controlled from the
start by **reusing** `04_p2rank/analysis/analyze_p2rank_controlled_pairs.py` on each tool's
success table (partial-ρ / controlled-pairs / CMH by size), exactly as for P2Rank.

## Pilot first
Mirror the docking pilot: a stratified ~1.5k-pocket subset (size × class × separation, drawn
with `05_docking/dataset/sample_pilot_pockets.py`), validate the per-tool holo>apo gradient with
size controls, then decide on the full ~82k run (fpocket is cheap enough to run fully; the DL
tools warrant a timing pilot since none publish per-structure runtimes).

## I/O contracts
- Normalized predictions: [`../schemas/predictions.schema.md`](../schemas/predictions.schema.md)
- Per-pocket success table: [`../schemas/lbs_success.schema.md`](../schemas/lbs_success.schema.md)

## Provenance
The **full method survey** (every tool, all 7 axes, all URLs), the **pipeline survey**, and the
**recommendation rationale + cost/scale notes** live in the private project repo
(`ahoj-af-project/lbs_generality/`), tracked as experiment **T3-2nd** in its `EXPERIMENTS.md`.
This public stage ships the reproducibility artifacts only (adapters, scorer, schemas).
