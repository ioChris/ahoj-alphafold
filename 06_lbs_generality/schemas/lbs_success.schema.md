# `lbs_<tool>_st100_merged_per_pocket.csv` — per-pocket success table (output)

Produced by `score/compute_lbs_success.py`. **One row per ST_100 pocket**
(`site_uid`), for one tool. Deliberately a **column-superset of the fields the
P2Rank analysis needs**, so the existing analysis is reused unchanged — in
particular:

```
04_p2rank/analysis/analyze_p2rank_controlled_pairs.py \
    --p2rank_table  <this file>  --success_col success_tns
```

## Identity / join
| Column | Notes |
|---|---|
| `site_uid` / `unit_uid` | Pocket id (both names carried; P2Rank scripts join on `unit_uid`). |
| `site_uniprot` | UniProt = receptor id. |
| `threshold` | `ST_100`. |
| `tool` | Predictor name. |

## Success metrics (the point of the stage) — see the scorer header for definitions
| Column | Notes |
|---|---|
| `dca_rank` / `dca4rank` | Rank of the first predicted pocket with DCA ≤ cutoff; `-1` = miss, `NaN` = unscorable (no ground-truth atoms). |
| `closest_pocket_dist` | Min centre-to-nearest-site-atom distance over predicted pockets (Å). Reproduces P2Rank's `closest_pocket_dist`. |
| `success_top1` | `dca_rank == 1`. |
| `success_topn` | `dca_rank > 0` (any pocket within cutoff). |
| **`success_tns`** | **MAIN metric:** `dca_rank > 0 & dca_rank ≤ n_unp_pockets`. |
| `success_tnm` | as TNS but with `n_unp_pockets_multichain`. |
| `success_dca_{4,5,6,8,10}A` | `closest_pocket_dist ≤ X` (distance sensitivity). |
| `predicted_pockets_total` | Number of predicted pockets for the receptor. |

## Classification / evidence / size metadata (joined from the geometry table)
`pocket_class` (apo/holo), `pocket_p_apo`, `pocket_p_holo`, `winning_p`,
`pocket_separation_pair`, `pocket_density_pair`, `pocket_overlap_pair`,
`model_pocket_plddt`, `n_apo_avg`, `n_holo_avg`, `ratio_holo_to_apo_avg_plus1`,
`log2_ratio_holo_to_apo_avg_plus1`, `n_unp_pockets`, `n_unp_pockets_multichain`,
`rg` (= `union_rg`, size covariate), `union_n_residues_found`, `rep_job`,
`afdb_filename`, and the verbal bins `prob_level_db`, `separation_level_db`,
`density_level_db` (identical helpers to the P2Rank stage).

### Sidecar `lbs_<tool>_manifest.json`
Records inputs (+ paths), threshold, DCA cutoff, and the headline summary
(`n_pockets`, `n_scorable`, `n_with_prediction`, `tns_percent_{all,apo,holo}`).
