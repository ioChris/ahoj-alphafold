# `predictions.csv` — normalized predicted-pockets contract (per predictor)

The **single tool-agnostic input** to `score/compute_lbs_success.py`. Every
predictor's native output is converted to this one schema by a small adapter
(`dataset/runners/*`), so all tools are scored identically and land in the same
per-pocket success table as P2Rank.

**One row per (pocket × predicted-pocket).** A pocket with no predicted rows is a
genuine miss (still scored, as a failure) — do **not** drop empty predictions.

## Required
| Column | Type | Notes |
|---|---|---|
| `site_uid` | str | Join key into `AFDB_ST100_DATASET_GEOM_CSV` (== P2Rank `unit_uid`). Every predicted pocket of a receptor is broadcast to each `site_uid` hosted on that receptor (a UniProt model can host several UBS pockets). |
| `pocket_rank` | int | 1-based rank in the **tool's own** ordering (1 = best). This is what `dca_rank`/TNS are computed against, so it must reflect the tool's confidence ordering (druggability score, probability, volume, …). |
| `center_x` / `center_y` / `center_z` | float | Predicted pocket centre in the **AF-model frame** (Å). fpocket: alpha-sphere centroid; IF-SitePred: DBSCAN cluster centre; GrASP/VN-EGNN: reported site centre. |

## Optional (carried through / useful for QC)
| Column | Type | Notes |
|---|---|---|
| `score` | float | The tool's native pocket score (druggability, probaTP, cluster size, …). |
| `pocket_id` | str | Tool-native pocket name (e.g. `pocket1`), for traceability. |
| `pocket_residues` | str | AF-model residue selection of the predicted pocket, e.g. `"A_50 A_51 …"`. Enables an alternative residue-overlap success (for residue-level predictors). |
| `tool` | str | Predictor name; also passed on the scorer CLI (`--tool`). |

## Notes on deriving `center_*` for residue-only predictors
Sequence/embedding predictors (GPSite, PeSTo, bindEmbed21DL, …) emit per-residue
binding probabilities, not cavity centres. Their adapter must first cluster the
high-probability residues (e.g. DBSCAN on Cα, or one cluster = one site) and set
`center_*` = the centroid of the cluster's residues on the AF model (needs the AF
coordinates; `score/residue_centroids.py`), with `pocket_rank` = cluster rank by
size/mean-probability. IF-SitePred already does this clustering internally, so its
adapter is a thin parser. For these tools also populate `pocket_residues` so the
residue-overlap metric can be reported alongside DCA.
