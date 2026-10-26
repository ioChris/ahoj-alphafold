# 06_lbs_generality/dataset — inputs + per-tool adapters

Turns the ST_100 **geometry** table (the same one the P2Rank stage builds) into the shared
ground-truth atom store, the predictor input lists, and — per tool — a normalized
`predictions.csv`. Paths from [`../../common/config.py`](../../common/config.py).

Run order:

1. **`build_ground_truth_atoms.py`** *(gemmi env, once per classification round)* — extracts the
   `union_chain_resi` heavy atoms of every ST_100 pocket from its AF model into
   `ground_truth_atoms.npz` (ragged: `site_uids`, `starts`, `counts`, `coords`). This is the
   shared, tool-independent ground truth for the DCA metric (centre → nearest site atom, which
   reproduces P2Rank's `closest_pocket_dist`). `--limit N` for a smoke test.

2. **`build_lbs_inputs.py`** — geometry → `receptors.csv` (unique AFDB models to run each
   predictor on; a receptor hosts several pockets) + `receptor_site_map.csv` (long
   `afdb_filename → site_uid`, used to broadcast each predicted pocket onto the pockets it
   covers). `--pilot_site_uids <csv>` restricts to a stratified pilot (draw it with
   `05_docking/dataset/sample_pilot_pockets.py` — the SAME size × class × separation
   stratification as the docking pilot).

3. **per-tool adapter → normalized `predictions.csv`** (schema:
   [`../schemas/predictions.schema.md`](../schemas/predictions.schema.md)):
   - **`runners/run_fpocket.py`** — parse fpocket `<name>_out/` folders into a raw pockets table
     (centres from `pocket*_vert.pqr`, residues from `pocket*_atm.pdb`, druggability from
     `*_info.txt`). *SCAFFOLD: parser implemented; the fpocket invocation (+ cif→pdb, run under
     WSL/Linux) is a documented TODO.*
   - **`runners/normalize_predictions.py`** — generic: any per-receptor pocket table
     (receptor key + rank + centre x/y/z [+ score, residues]) → normalized `predictions.csv`,
     broadcasting pockets onto sites via `receptor_site_map.csv`. IF-SitePred / GrASP / VN-EGNN
     adapters are thin wrappers that first dump their native output to that raw format.

Inputs consumed (via config): `AFDB_ST100_DATASET_GEOM_CSV` (pockets + `union_chain_resi` +
class/size metadata), `AFDB_DIR` (receptor models — for the ground-truth builder and the
predictors themselves).
