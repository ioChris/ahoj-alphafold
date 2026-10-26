# 04 — P2Rank binding-site prediction

**Analysis 5.** Run and evaluate **P2Rank** binding-site prediction on AFDB models (with
simulated ligand centers derived from AHoJ-DB UBS definitions), and relate prediction success
to apo/holo class, classification probability, and evidence balance. AF-Holo pockets are
easier to predict than AF-Apo (e.g. ~84% vs ~64% TNS DCA4 success), success rises with holo
probability and holo-favoured evidence balance, and the class effect **survives size control**.

Main success metric **`TNS`** = `(dca4rank > 0) and (dca4rank <= n_unp_pockets)`.

Paths come from [`../common/config.py`](../common/config.py).

### Subdirectories
- [`dataset/`](dataset/) — build the ST_100 AFDB **P2Rank input dataset**: select monomeric
  single-UniProt UBS rows, map pocket residues onto AFDB models, compute centers / radius of
  gyration, and export P2Rank-ready subsets. See [`dataset/README.md`](dataset/README.md).
- [`analysis/`](analysis/) — **analyze + plot** P2Rank predictions: build the merged
  per-pocket table, success-by-class/probability/separation, descriptor & query-metric
  correlations, size-controlled checks, and the figures. See [`analysis/README.md`](analysis/README.md).
  - Main panels: `plot_p2rank_ubs_run4.py`; size-control SI figure (manuscript SI Figure S4;
    harmonized DCA4-TNS by default, `--criterion native` for P2Rank's own evaluation):
    `figure_p2rank_size_controlled_class.py`.
- [`prep/`](prep/) — **legacy** input-prep utilities (UBS center computation, union/intersection
  splitting, UniProt counting). Predate the `dataset/` pipeline and are anchored under
  `WORK_DIR`; ⚑ likely superseded by `dataset/` — kept for provenance.

### Motivating hypothesis (design note)
UBSs in AFDB models are classified apo/holo with varying probability, and within a UBS the
apo/holo **separation** can be high/mid/low. The hypothesis tested here:

- AF-Apo pockets in **high-separation** UBS → the **hardest** targets for ligand-binding-site
  prediction software.
- AF-Apo in mid/low-separation UBS → medium difficulty.
- AF-Holo pockets (any separation) → comparatively **easy** targets.

P2Rank success is therefore stratified by class × classification probability × separation.
