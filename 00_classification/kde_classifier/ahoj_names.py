# -*- coding: utf-8 -*-
"""
AHoJ output names: one rename step from the names AHoJ writes since 2.6.0 to the names parse-ahoj uses.

AHoJ 2.6.0 renamed every output column and JSON key (AHoJ2 PR #146; documentation/output_names.md,
breaking-changes.md "Output Names of 2.6.0"). Only the names changed: every value is the one the job
computed before. parse-ahoj's code, configs and outputs keep the names before 2.6.0 (#resis3, RoG,
apoholo_assignment, ...), so a job written with the new names is renamed back right after reading and
the rest of the code sees one set of names whichever AHoJ wrote the job.

Which names a job has:
    script_parameters.json `output_schema`  2 = the names since 2.6.0; 1 = the names before them (also
                                            what AHoJ 2.6.0 writes under --legacy_names 1); absent = an
                                            AHoJ before 2.6.0, so the names before them
    global_results.csv header               the fallback when script_parameters.json is missing, and a
                                            check otherwise: a header that contradicts output_schema, or
                                            mixes both sets, raises OutputNamesError
    ligands.json `schema_version`           3 = the names since 2.6.0, 2 = before (no schema_version:
                                            the pocket key decides; AHoJ 2.1 wrote a bare list)

The maps below are copies of AHoJ2's output_names.py (RESULT_PREVIOUS, LIGANDS_ENTITY_PREVIOUS,
LIGANDS_SCHEMA_VERSIONS) at 0e280af, branch claude/output-column-names (PR #146): name since 2.6.0 ->
name before. Only the files parse-ahoj reads are mapped: global_results.csv (and the other result
tables, which share its names) and ligands.json. script_parameters.json renamed no key.

Pure module: pandas only for the DataFrame helper.
"""
from __future__ import annotations

from typing import Iterable, Optional

# script_parameters.json `output_schema`
OUTPUT_SCHEMA_NEW = 2
OUTPUT_SCHEMA_OLD = 1
OUTPUT_SCHEMAS = (OUTPUT_SCHEMA_OLD, OUTPUT_SCHEMA_NEW)

# ligands.json `schema_version`: (since 2.6.0, before)
LIGANDS_SCHEMA_VERSIONS = (3, 2)

# global_results.csv, query_pocket_info.csv, the filtered tables, alphafold_pocket_classification.csv
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

# ligands.json entity records (query entities, entity_group, detected_entities)
LIGANDS_ENTITY_PREVIOUS = {
    'resn': 'label', 'n_atoms': 'atom_count', 'n_heavy_atoms': 'heavy_atom_count',
    'entity_role': 'role', 'n_contact_residues': 'contact_resis',
    'n_overlap_residues': 'overlap_resis', 'entity_cov_pct': 'entity_contact_coverage',
    'query_cov_pct': 'query_pocket_coverage', 'label_detail': 'status',
}

# ligands.json pockets[].query and pockets[].candidates[]
LIGANDS_POCKET_PREVIOUS = {'pocket': 'pocket_index'}

# Names only one of the two sets has: the header sniff counts these. No name is in both sets, so a
# header with names from both is a mixed (broken) header, never an ambiguous one.
RESULT_NEW_ONLY = frozenset(RESULT_PREVIOUS) - frozenset(RESULT_PREVIOUS.values())
RESULT_OLD_ONLY = frozenset(RESULT_PREVIOUS.values()) - frozenset(RESULT_PREVIOUS)


class OutputNamesError(ValueError):
    """The names of a job cannot be told apart safely (contradicting or mixed)."""


def schema_from_header(columns: Iterable[str]) -> Optional[int]:
    """The output_schema a result-table header shows: 2 (only names since 2.6.0), 1 (only names before),
    None (neither set is present). Raises OutputNamesError on a header with names from both."""
    cols = set(map(str, columns))
    new, old = sorted(cols & RESULT_NEW_ONLY), sorted(cols & RESULT_OLD_ONLY)
    if new and old:
        raise OutputNamesError(f"result header mixes the AHoJ 2.6.0 names {new[:5]} with the names before "
                               f"them {old[:5]}")
    if new:
        return OUTPUT_SCHEMA_NEW
    if old:
        return OUTPUT_SCHEMA_OLD
    return None


def output_schema_of(script_params: Optional[dict], columns: Iterable[str] = ()) -> int:
    """The output_schema of a job: script_parameters.json `output_schema`, checked against the result
    header; the header alone when script_parameters.json is missing; 1 when neither decides (an AHoJ
    before 2.6.0 writes no output_schema). Raises OutputNamesError on a contradiction or an unknown value."""
    sniffed = schema_from_header(columns)
    declared = None if not script_params else script_params.get("output_schema")
    if declared is None:
        # absent: AHoJ before 2.6.0, no script_parameters.json, or a writer that does not record it
        # (kaleidoscope's AHoJ-schema export); the two name sets are disjoint, so the header decides
        return sniffed or OUTPUT_SCHEMA_OLD
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        raise OutputNamesError(f"unreadable output_schema {declared!r}") from None
    if declared not in OUTPUT_SCHEMAS:
        raise OutputNamesError(f"output_schema {declared} is not known to parse-ahoj (known: {OUTPUT_SCHEMAS})")
    if sniffed is not None and sniffed != declared:
        raise OutputNamesError(f"script_parameters.json output_schema {declared} but the global_results.csv "
                               f"header has the names of output_schema {sniffed}")
    return declared


def results_to_internal(df, script_params: Optional[dict] = None):
    """A result table (global_results.csv) under parse-ahoj's names: renamed from the AHoJ 2.6.0 names
    when the job has them, returned as it was otherwise. The schema read is kept in
    df.attrs['ahoj_output_schema']. Raises OutputNamesError (see output_schema_of)."""
    schema = output_schema_of(script_params, df.columns)
    if schema == OUTPUT_SCHEMA_NEW:
        df = df.rename(columns=RESULT_PREVIOUS)
    df.attrs["ahoj_output_schema"] = schema
    return df


def _renamed_keys(record, mapping):
    return {mapping.get(key, key): value for key, value in record.items()} if isinstance(record, dict) else record


def ligands_schema_of(document) -> Optional[int]:
    """ligands.json schema_version; for a document without one, 3 when its pockets use the key `pocket`
    and 2 when they use `pocket_index`. None for AHoJ 2.1's bare list or a document that does not decide."""
    if not isinstance(document, dict):
        return None
    if document.get("schema_version") is not None:
        return int(document["schema_version"])
    for pocket in document.get("pockets", []) or []:
        for record in [pocket.get("query") or {}] + list(pocket.get("candidates") or []):
            if "pocket" in record:
                return LIGANDS_SCHEMA_VERSIONS[0]
            if "pocket_index" in record:
                return LIGANDS_SCHEMA_VERSIONS[1]
    return None


def ligands_to_internal(document):
    """ligands.json under parse-ahoj's keys (schema 2): a schema-3 document (AHoJ 2.6.0) is renamed,
    with its schema_version set to 2; any other document is returned as it was. The same rename as
    AHoJ2's output_names.legacy_ligands."""
    if ligands_schema_of(document) != LIGANDS_SCHEMA_VERSIONS[0]:
        return document
    document = dict(document)
    document["schema_version"] = LIGANDS_SCHEMA_VERSIONS[1]

    def entities(records):
        return [_renamed_keys(record, LIGANDS_ENTITY_PREVIOUS) for record in records or []]

    pockets = []
    for pocket in document.get("pockets", []):
        pocket = dict(pocket)
        if "query" in pocket:
            query = _renamed_keys(pocket["query"], LIGANDS_POCKET_PREVIOUS)
            for key in ("entities", "detected_entities"):
                if key in query:
                    query[key] = entities(query[key])
            if "entity_group" in query:
                query["entity_group"] = _renamed_keys(query["entity_group"], LIGANDS_ENTITY_PREVIOUS)
            pocket["query"] = query
        if "candidates" in pocket:
            candidates = []
            for candidate in pocket["candidates"]:
                candidate = _renamed_keys(candidate, LIGANDS_POCKET_PREVIOUS)
                if "detected_entities" in candidate:
                    candidate["detected_entities"] = entities(candidate["detected_entities"])
                candidates.append(candidate)
            pocket["candidates"] = candidates
        pockets.append(pocket)
    if "pockets" in document:
        document["pockets"] = pockets
    return document
