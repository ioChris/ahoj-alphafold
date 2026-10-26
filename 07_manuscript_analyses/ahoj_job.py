# -*- coding: utf-8 -*-
"""Read an AHoJ job's global_results.csv and ligands.json under the names before AHoJ 2.6.0.

AHoJ 2.6.0 renamed every output column and JSON key (AHoJ2 PR #146; the map is AHoJ2's
documentation/output_names.md and output_names.py). Only the names changed, never a value. The
analysis scripts here, and the parse-ahoj functions they call (relabel_apoholo_by_ions,
sort_dataframe_v2c, ...), read the previous names, so every job is brought to those names on
reading: a v2c archive (AHoJ 2.1.0) passes through untouched, a 2.6.0 job is renamed.

Which names a job has: script_parameters.json `output_schema` (2 = the 2.6.0 names, 1 = the
previous names, written by `--legacy_names 1`; absent = AHoJ before 2.6.0). Without
script_parameters.json, or without the key, the CSV header decides. ligands.json carries its
own `schema_version` (3 = the 2.6.0 keys, 2 = the previous ones).

The maps below are copied from AHoJ2 output_names.py (RESULT_PREVIOUS, LIGANDS_ENTITY_PREVIOUS;
branch claude/output-column-names at 7e52e98). `python ahoj_job.py --check <AHoJ2>/output_names.py`
verifies that they still equal AHoJ's.
"""
import io
import json
import tarfile

import pandas as pd

# global_results.csv, query_pocket_info.csv, the filtered tables, alphafold_pocket_classification.csv:
# name since 2.6.0 -> name before it (AHoJ2 output_names.RESULT_PREVIOUS)
RESULT_PREVIOUS = {
    'poi': 'query_POI', 'poi_type': 'POI_type', 'label': 'apoholo_assignment',
    'n_contact_residues': 'holo_contact_resis', 'n_overlap_residues': 'holo_overlap_resis',
    'entity_cov_pct': 'entity_contact_coverage', 'query_cov_pct': 'query_pocket_coverage',
    'best_entity': 'holo_overlap_entity', 'label_detail': 'holo_assignment_detail',
    'n_seqres': '#resis5', 'n_observed': '#resis3', 'n_heavy_atoms': '#atoms3',
    'n_unobserved': 'unobserved_rsd', 'mapped_seqres_pct': '%mapped_sqr',
    'mapped_observed_pct': '%mapped_obs', 'unobserved_pct': '%unobserved_rsd',
    'n_pocket_identity_matches': 'pocket_identity_matches',
    'candidate_unp_seq_identity': 'candidate_unp_seq_id', 'chains_unp': 'chains3',
    'accessions': 'UNPs', 'rog': 'RoG', 'bfac_total': 'bfac_t', 'bfac_mean': 'bfac_avg',
    'sasa_mean': 'sasa_avg', 'rsa_mean': 'rsa_avg', 'sesa': 'asa', 'sesa_mean': 'asa_avg',
    'volume_convex': 'vol_convex', 'query_resns': 'query_ligs', 'nonquery_resns': 'nonquery_ligs',
    'n_solvent': '#solvent', 'unp_len_seqres': 'UNPlen_sqr', 'unp_len_observed': 'UNPlen_obs',
    'unp_overlap_seqres': 'UNPovrlp_sqr', 'unp_overlap_observed': 'UNPovrlp_obs',
    'unp_cov_seqres_pct': 'UNPcov_sqr', 'unp_cov_observed_pct': 'UNPcov_obs',
    'query_chains': 'aln_q_chains', 'tm_score_candidate': 'tm_score_i',
    'n_aligned_residues': 'aln_len', 'pocket_rmsd': 'pocket_rms',
    'n_pocket_residue_pairs': 'pocket_len', 'n_pocket_atom_pairs': 'pocket_aln_atoms',
    'pocket_lddt': 'pocket_lddt_map', 'rog_distance': 'RoG_distance',
}

# ligands.json entity records (AHoJ2 output_names.LIGANDS_ENTITY_PREVIOUS)
LIGANDS_ENTITY_PREVIOUS = {
    'resn': 'label', 'n_atoms': 'atom_count', 'n_heavy_atoms': 'heavy_atom_count',
    'entity_role': 'role', 'n_contact_residues': 'contact_resis',
    'n_overlap_residues': 'overlap_resis', 'entity_cov_pct': 'entity_contact_coverage',
    'query_cov_pct': 'query_pocket_coverage', 'label_detail': 'status',
}
# ligands.json pocket records: query and candidates
LIGANDS_POCKET_PREVIOUS = {'pocket': 'pocket_index'}

OUTPUT_SCHEMA_NEW = 2                 # script_parameters.json output_schema of the 2.6.0 names
LIGANDS_SCHEMA_VERSIONS = (3, 2)      # ligands.json schema_version: (2.6.0 keys, previous keys)

# names that occur on one side of the map only: they tell the header's naming apart
_NEW_ONLY = set(RESULT_PREVIOUS) - set(RESULT_PREVIOUS.values())
_OLD_ONLY = set(RESULT_PREVIOUS.values()) - set(RESULT_PREVIOUS)


def output_schema(params):
    """script_parameters.json (dict, bytes, str or None) -> its `output_schema`, None when absent."""
    if params is None:
        return None
    if isinstance(params, (bytes, str)):
        try:
            params = json.loads(params)
        except ValueError:
            return None
    value = params.get("output_schema") if isinstance(params, dict) else None
    return None if value is None else int(value)


def result_schema(columns, params=None):
    """The naming of a result table: 2 (AHoJ 2.6.0 names) or 1 (the names before them).
    script_parameters.json `output_schema` decides; without it, the header."""
    schema = output_schema(params)
    if schema is not None:
        return schema
    columns = set(columns)
    if columns & _OLD_ONLY:
        return 1
    return OUTPUT_SCHEMA_NEW if columns & _NEW_ONLY else 1


def legacy_results(df, params=None):
    """A result table (global_results.csv and the tables that share its columns) under the
    names before 2.6.0. A table already under them is returned as it is (the same object)."""
    if df is None or result_schema(df.columns, params) != OUTPUT_SCHEMA_NEW:
        return df
    return df.rename(columns=RESULT_PREVIOUS)


def _renamed(record, mapping):
    return {mapping.get(k, k): v for k, v in record.items()} if isinstance(record, dict) else record


def legacy_ligands(document):
    """ligands.json under the keys before 2.6.0 (schema_version 3 -> 2); any other document is
    returned as it is. The inverse of AHoJ2's output_names.legacy_ligands, which writes
    `--legacy_names 1`."""
    if not isinstance(document, dict) or document.get("schema_version") != LIGANDS_SCHEMA_VERSIONS[0]:
        return document
    document = dict(document)
    document["schema_version"] = LIGANDS_SCHEMA_VERSIONS[1]

    def entities(records):
        return [_renamed(r, LIGANDS_ENTITY_PREVIOUS) for r in records or []]

    pockets = []
    for pocket in document.get("pockets", []):
        pocket = dict(pocket)
        if "query" in pocket:
            query = _renamed(pocket["query"], LIGANDS_POCKET_PREVIOUS)
            for key in ("entities", "detected_entities"):
                if key in query:
                    query[key] = entities(query[key])
            if "entity_group" in query:
                query["entity_group"] = _renamed(query["entity_group"], LIGANDS_ENTITY_PREVIOUS)
            pocket["query"] = query
        if "candidates" in pocket:
            candidates = []
            for cand in pocket["candidates"]:
                cand = _renamed(cand, LIGANDS_POCKET_PREVIOUS)
                if "detected_entities" in cand:
                    cand["detected_entities"] = entities(cand["detected_entities"])
                candidates.append(cand)
            pocket["candidates"] = candidates
        pockets.append(pocket)
    if "pockets" in document:
        document["pockets"] = pockets
    return document


# ----------------------------------------------------------------------------------- job archives
def read_members(tgz, names):
    """Top-level members of a job archive (<job>.tgz, stored as ./name or name) -> {name: bytes},
    in one pass over the archive; absent members are left out."""
    want = {f"./{n}": n for n in names}
    want.update({n: n for n in names})
    out = {}
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar:
            n = want.get(m.name)
            if n is not None and n not in out and m.isfile():
                out[n] = tar.extractfile(m).read()
                if len(out) == len(names):
                    break
    return out


def read_global_results(tgz, **read_csv):
    """A job archive's global_results.csv under the names before 2.6.0 (None if the archive has
    none). `read_csv` goes to pandas.read_csv unchanged."""
    mem = read_members(tgz, ("global_results.csv", "script_parameters.json"))
    if "global_results.csv" not in mem:
        return None
    df = pd.read_csv(io.BytesIO(mem["global_results.csv"]), **read_csv)
    return legacy_results(df, mem.get("script_parameters.json"))


def read_ligands(tgz):
    """A job archive's ligands.json under the keys before 2.6.0 (None if the archive has none)."""
    data = read_members(tgz, ("ligands.json",)).get("ligands.json")
    return None if not data else legacy_ligands(json.loads(data))


def _check(path):
    """Compare the maps above with AHoJ2's output_names.py at `path`."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ahoj_output_names", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok = True
    for name in ("RESULT_PREVIOUS", "LIGANDS_ENTITY_PREVIOUS", "LIGANDS_SCHEMA_VERSIONS"):
        same = getattr(mod, name) == globals()[name]
        ok &= same
        print(f"{name}: {'same' if same else 'DIFFERENT'}")
    same = mod.OUTPUT_SCHEMA == OUTPUT_SCHEMA_NEW
    ok &= same
    print(f"OUTPUT_SCHEMA: {'same' if same else 'DIFFERENT'}")
    return ok


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="OUTPUT_NAMES_PY", required=True,
                    help="AHoJ2's output_names.py: verify that the maps here equal it")
    sys.exit(0 if _check(ap.parse_args().check) else 1)
