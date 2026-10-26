# -*- coding: utf-8 -*-
"""
Run-type awareness for parse-ahoj (arm 4 of the AHoJ-DB build program).

Design: docs/RUN_TYPES.md. Three paths share one classifier:

    holo-2.1   the legacy AHoJ-DB v2c archives (AHoJ 2.1.x). Results must not change.
    holo-2.4   one job per ligand instance on AHoJ 2.4.x/2.5.x: gate labels, the convex volume, new columns.
    afc-2.4    AHoJ 2.4.x/2.5.x UniProt-anchored mode: the AFDB model is the query and the pocket is a
               locked residue set. The query row IS the model, so it never joins the apo or holo set.
Which AHoJ minor takes which profile is the table AHOJ_MINORS; any other minor is a loud failure.

Everything a path decides is a field of a RunProfile, so a config can override any single rule and
the summary row records which rules were in force. Loud failures (unknown archive layout, a key that
does not parse, a profile that contradicts the archive, a missing volume column, add_query_row=1 on
an AF-centric archive) raise RunTypeError instead of changing a number silently.

SIFTS: this module reads only the archive's own mapping artifacts (residue_mappings.zip). Any lookup
beyond them goes through AHoJ2's ahoj_sifts package at the run's pinned commit (see sifts_access.py).
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass, field, replace, asdict
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

import ahoj_names


class RunTypeError(ValueError):
    """A loud failure: the archive, key, bundle or config cannot be interpreted safely."""


RUN_TYPES = ("holo-2.1", "holo-2.4", "afc-2.4")

# AHoJ (major, minor) -> the profile minor that reads it; detect_run_type (archives) and read_run_yaml
# (bundles) both go through run_type_for, so they cannot disagree. 2.5.x takes the 2.4 profiles: 2.5.1
# and develop 7fd6825 write the 2.4.0 global_results.csv columns and archive members (script_parameters
# gains 'config'), and their one label change, A-, is apo (AHOJ-020). 2.6.x takes them too: 2.6.0 renames
# every output column and JSON key and changes no value (AHoJ2 PR #146), and load_global_results /
# load_ligands_json rename them back (ahoj_names). A minor missing here raises until its archives are
# checked and it is added (docs/RUN_TYPES.md §8 item 9).
AHOJ_MINORS: Dict[Tuple[int, int], str] = {(2, 1): "2.1", (2, 4): "2.4", (2, 5): "2.4", (2, 6): "2.4"}

# The apo/holo label sets the classifier pools (parse-ahoj convention, AHOJ-001, AHOJ-020). Every
# active apo/holo filter reads these two tuples; do not spell the labels out again.
#   A    apo: nothing in the pocket
#   A-   apo: AHoJ's holo-overlap gate demoted a query-ligand hit (the query ligand is present but does
#        not occupy the pocket). AHoJ develop from 431b220 on; 2.4.0 and 2.5.1 wrote plain A for it.
#   A*   apo: a non-query entity is present but does not count (AHoJ: it fails the gate, or
#        --lig_free_sites 0 tolerates it; the afc entity relabel: it qualifies but fails the gate)
#   A**  apo: a holo call demoted by parse-ahoj's ion relabel (the v2c mark)
#   H    holo: the query ligand occupies the pocket
#   H*   holo: another entity occupies the pocket
APO_LABELS = ("A", "A-", "A*", "A**")
HOLO_LABELS = ("H", "H*")
# Candidate labels that are known and pooled in neither set: U (unobserved pocket, #resis3 = 0).
UNPOOLED_LABELS = ("U",)
KNOWN_CANDIDATE_LABELS = frozenset(APO_LABELS + HOLO_LABELS + UNPOOLED_LABELS)

# Labels that a relabel must never touch: unobserved (U), the documented-but-unwritten '*', and '-'.
PROTECTED_LABELS_24 = frozenset({"U", "*", "-"})
PROTECTED_LABELS_21 = frozenset({"U"})           # frozen: the v2c release was made with this set

# AHoJ 2.4.0 holo-overlap gate defaults (script_parameters.json holo_overlap_min_*).
GATE_DEFAULTS = {"min_resis": 2, "min_qcov": 20.0, "min_ecov": 30.0}

# Common crystallisation additives, cryoprotectants and buffer components. A compact stand-in for the
# BioLiP artifact list, copied from the pilot (ahoj-af-project af_centric_rerun.py ADDITIVES). Only used
# when additive_rule == "site"; v2c counted additives as holo and that is the default on every path.
ADDITIVES_DEFAULT = frozenset({
    "SO4", "PO4", "PI", "2HP", "NO3", "SCN", "AZI", "CO3", "BCT",
    "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "P6G", "PE4", "PE8", "2PE", "12P", "15P", "P33", "PEU",
    "PG0", "PG5", "PG6", "7PE", "XPE", "MPD", "MRD", "PGO", "PGR", "BU1", "BU3", "IPA", "EOH", "MOH",
    "DMS", "DMF", "ACT", "ACY", "FMT", "ACE", "CIT", "FLC", "TAR", "TLA", "MLI", "MLA", "SIN", "MAL",
    "TRS", "EPE", "MES", "BTB", "B3P", "CXS", "HEZ", "IMD", "BME", "DTT", "DTU", "TCE", "GLY",
    "NH4", "NH2", "OXY", "PEO", "UNL", "UNX", "LMT", "BOG", "LDA", "C8E", "SDS", "OLC", "CPS",
})


# =============================================================================== profiles
@dataclass(frozen=True)
class RunProfile:
    name: str
    # what the archive must look like
    ahoj_minor: Tuple[int, int]                 # (major, minor) the profile is written for
    query_modes: Tuple[Optional[str], ...]      # accepted global_results.csv query_mode values
    key_format: str                             # "holo" <pdb>-<chain>-<lig>-<resi> (or '_') | "afc" <UNP>_ST_100_<n>
    # classification rules
    add_query_row: int                          # 1: the query row joins the holo set (pocket RMSD excluded)
    relabel_mode: str                           # v2c_ions | v2c_ions_demote_only | entity_gate_site | none
    ion_rule: str                               # query_disables (v2c per job) | site | nonion | all
    additive_rule: str                          # v2c (additives count) | site (only at additive-defined sites)
    gate: str                                   # none | archive | apply  (entity relabel only)
    volume_column: str                          # column copied into 'volume' before curation
    volume_nan_to_zero: bool                    # v2c rule: NaN volume with #resis3 > 0 -> 0.0 (PyVOL)
    model_row: str                              # AF_model: the query's own model (select_model_rows) | first_M
    protect_labels: frozenset
    # what the numbers mean afterwards
    delta_anchor: str                           # query | model  (query-summary deltas are |candidate - anchor|)
    rog_anchor: str                             # query | model  (the +/-60% RoG window)
    mapped_denominator: str                     # member_pocket | locked_union (the 75% mapped cut)
    chain_axes_meaningful: bool                 # False when the model is the anchor (0/0/1/0 by construction)

    def as_record(self) -> dict:
        d = asdict(self)
        d["protect_labels"] = "".join(sorted(self.protect_labels))
        d["query_modes"] = "|".join(str(q) for q in self.query_modes)
        d["ahoj_minor"] = ".".join(map(str, self.ahoj_minor))
        return d


PROFILES: Dict[str, RunProfile] = {
    "holo-2.1": RunProfile(
        name="holo-2.1", ahoj_minor=(2, 1), query_modes=(None,), key_format="holo",
        add_query_row=1, relabel_mode="v2c_ions", ion_rule="query_disables", additive_rule="v2c",
        gate="none", volume_column="volume", volume_nan_to_zero=True, model_row="AF_model",
        protect_labels=PROTECTED_LABELS_21,
        delta_anchor="query", rog_anchor="query", mapped_denominator="member_pocket",
        chain_axes_meaningful=True),
    "holo-2.4": RunProfile(
        name="holo-2.4", ahoj_minor=(2, 4), query_modes=("pdb", "holo", None), key_format="holo",
        add_query_row=1, relabel_mode="v2c_ions_demote_only", ion_rule="query_disables", additive_rule="v2c",
        gate="archive", volume_column="volume", volume_nan_to_zero=True, model_row="AF_model",
        protect_labels=PROTECTED_LABELS_24,
        delta_anchor="query", rog_anchor="query", mapped_denominator="member_pocket",
        chain_axes_meaningful=True),
    "afc-2.4": RunProfile(
        name="afc-2.4", ahoj_minor=(2, 4), query_modes=("uniprot",), key_format="afc",
        add_query_row=0, relabel_mode="entity_gate_site", ion_rule="site", additive_rule="v2c",
        gate="apply", volume_column="volume", volume_nan_to_zero=True, model_row="AF_model",
        protect_labels=PROTECTED_LABELS_24,
        delta_anchor="model", rog_anchor="model", mapped_denominator="locked_union",
        chain_axes_meaningful=False),
}

RELABEL_MODES = ("v2c_ions", "v2c_ions_demote_only", "entity_gate_site", "none")
ION_RULES = ("query_disables", "site", "nonion", "all")
ADDITIVE_RULES = ("v2c", "site")
GATE_MODES = ("none", "archive", "apply")
MODEL_ROW_MODES = ("AF_model", "first_M")

# Config keys that override a profile field. None (or the string "profile") keeps the profile value.
OVERRIDE_KEYS = ("add_query_row", "relabel_mode", "ion_rule", "additive_rule", "gate",
                 "volume_column", "volume_nan_to_zero", "model_row", "key_format")
KEY_FORMATS = ("holo", "afc", "any")     # 'any': accept either key shape (the pilot named holo runs by site)


def _is_set(v) -> bool:
    return v is not None and str(v).lower() not in ("profile", "none_", "")


# =============================================================================== archive access
def _members(tar: tarfile.TarFile) -> Dict[str, str]:
    """bare member name -> stored name (archives store './x' or 'x')."""
    out = {}
    for n in tar.getnames():
        out[n[2:] if n.startswith("./") else n] = n
    return out


def read_member(query_path: str, member: str) -> Optional[bytes]:
    """One member of a job (a .tgz or an extracted directory) as bytes, or None when absent.
    Both member spellings ('./x' and 'x') are accepted: item 1 of the edit list."""
    if os.path.isdir(query_path):
        p = os.path.join(query_path, member)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                return fh.read()
        return None
    with tarfile.open(query_path, "r:gz") as tar:
        m = _members(tar)
        if member not in m:
            return None
        f = tar.extractfile(m[member])
        return None if f is None else f.read()


def read_members(query_path: str, members: Iterable[str]) -> Dict[str, Optional[bytes]]:
    """Several members of a job read in one pass over the archive: member -> bytes, None when absent."""
    members = list(members)
    if os.path.isdir(query_path):
        return {m: read_member(query_path, m) for m in members}
    out: Dict[str, Optional[bytes]] = {}
    with tarfile.open(query_path, "r:gz") as tar:
        names = _members(tar)
        for m in members:
            f = tar.extractfile(names[m]) if m in names else None
            out[m] = None if f is None else f.read()
    return out


def load_global_results(query_path: str) -> Optional[pd.DataFrame]:
    """global_results.csv under parse-ahoj's names (the names before AHoJ 2.6.0). A job written with the
    2.6.0 names is renamed right after reading (ahoj_names: script_parameters.json output_schema, the
    header as the fallback), so every reader downstream sees one set of names. Raises RunTypeError when
    the header contradicts output_schema or mixes both sets."""
    b = read_members(query_path, ("global_results.csv", "script_parameters.json"))
    if b["global_results.csv"] is None:
        return None
    df = pd.read_csv(io.BytesIO(b["global_results.csv"]), low_memory=False)
    sp = b["script_parameters.json"]
    sp = None if sp is None else json.loads(sp.decode("utf-8"))
    try:
        return ahoj_names.results_to_internal(df, sp)
    except ahoj_names.OutputNamesError as e:
        raise RunTypeError(f"{os.path.basename(str(query_path))}: {e}") from None


def load_script_parameters(query_path: str) -> Optional[dict]:
    b = read_member(query_path, "script_parameters.json")
    return None if b is None else json.loads(b.decode("utf-8"))


def load_ligands_json(query_path: str) -> Optional[dict]:
    """ligands.json under parse-ahoj's keys (schema 2); AHoJ 2.6.0's schema 3 is renamed on reading."""
    b = read_member(query_path, "ligands.json")
    return None if b is None else ahoj_names.ligands_to_internal(json.loads(b.decode("utf-8")))


def load_poi_dict_unp(query_path: str, zip_name: str = "residue_mappings.zip",
                      json_name: str = "poi_dict_unp.json") -> dict:
    """AHoJ's own residue mapping of the query pocket (PDB -> UniProt), unchanged between 2.1 and 2.4."""
    b = read_member(query_path, zip_name)
    if b is None:
        raise FileNotFoundError(f"{zip_name} not found in {query_path}")
    with zipfile.ZipFile(io.BytesIO(b)) as zf:
        if json_name not in zf.namelist():
            raise FileNotFoundError(f"{json_name} not found in {zip_name} of {query_path}")
        with zf.open(json_name) as jf:
            return json.load(jf)


# =============================================================================== provenance
def ahoj_version(script_params: Optional[dict]) -> Optional[str]:
    return None if not script_params else str(script_params.get("script_version") or "") or None


def ahoj_version_tuple(script_params: Optional[dict]) -> Optional[Tuple[int, int]]:
    v = ahoj_version(script_params)
    if not v:
        return None
    m = re.match(r"^(\d+)\.(\d+)", v)
    if not m:
        raise RunTypeError(f"unreadable script_version {v!r}")
    return int(m.group(1)), int(m.group(2))


def schema_of(df: pd.DataFrame) -> str:
    """Which global_results.csv layout wrote the archive. 'v2.4' has the gate columns and the query_mode
    triple; 'v2.1' has neither."""
    cols = set(df.columns)
    if {"holo_assignment_detail", "query_mode"} <= cols:
        return "v2.4"
    if "holo_assignment_detail" in cols or "query_mode" in cols:
        return "v2.4-partial"
    return "v2.1"


def query_mode_of(df: pd.DataFrame) -> Optional[str]:
    if "query_mode" not in df.columns:
        return None
    q = df.loc[df["role"] == "Q", "query_mode"] if "role" in df.columns else df["query_mode"]
    q = q.dropna().astype(str)
    return q.iloc[0] if len(q) else None


def query_triple(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """query_mode / query_anchor / query_model as AHoJ 2.4 records them (None on 2.1 archives)."""
    out = {"query_mode": None, "query_anchor": None, "query_model": None}
    if "role" not in df.columns:
        return out
    q = df[df["role"] == "Q"]
    for c in out:
        if c in df.columns and len(q):
            v = q[c].dropna().astype(str)
            out[c] = v.iloc[0] if len(v) else None
    return out


_AFC_QUERY_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s+([0-9]+(?:\+[0-9]+)*)\s*$")


def known_ahoj_minors() -> str:
    """AHOJ_MINORS for an error message: '2.1.x -> holo-2.1; 2.4.x, 2.5.x -> holo-2.4 / afc-2.4'."""
    by: Dict[str, list] = {}
    for (major, minor), pm in sorted(AHOJ_MINORS.items()):
        by.setdefault(pm, []).append(f"{major}.{minor}.x")
    return "; ".join(f"{', '.join(v)} -> {' / '.join(n for n in RUN_TYPES if n.endswith('-' + pm))}"
                     for pm, v in by.items())


def run_type_for(kind: str, ver: Tuple[int, int]) -> str:
    """'holo' | 'afc' on AHoJ (major, minor) -> its run type through AHOJ_MINORS, or RunTypeError."""
    pm = AHOJ_MINORS.get(tuple(ver))
    if pm is None or f"{kind}-{pm}" not in PROFILES:
        raise RunTypeError(f"{kind} on AHoJ {ver[0]}.{ver[1]} has no parse-ahoj profile "
                           f"(known: {known_ahoj_minors()})")
    return f"{kind}-{pm}"


def detect_run_type(script_params: Optional[dict], df: pd.DataFrame) -> str:
    """The run type of one archive from what AHoJ wrote into it (the minor through AHOJ_MINORS).

    2.1.x               -> holo-2.1
    2.4.x/2.5.x, query_mode 'uniprot' (or a '<UNP> <i>+<j>+...' query string) -> afc-2.4
    2.4.x/2.5.x, otherwise -> holo-2.4
    anything else, or no script_parameters.json with a layout that does not decide -> RunTypeError
    """
    ver = ahoj_version_tuple(script_params)
    schema = schema_of(df)
    if ver is None:
        if schema == "v2.1":
            return "holo-2.1"
        if schema == "v2.4":
            return "afc-2.4" if query_mode_of(df) == "uniprot" else "holo-2.4"
        raise RunTypeError("no script_parameters.json and the global_results.csv layout does not decide "
                           f"the run type (schema {schema})")
    pm = AHOJ_MINORS.get(ver)
    if pm is None:
        raise RunTypeError(f"AHoJ {ahoj_version(script_params)} has no parse-ahoj profile "
                           f"(known: {known_ahoj_minors()})")
    if pm == "2.1":
        if schema != "v2.1":
            raise RunTypeError(f"AHoJ {ahoj_version(script_params)} archive with a {schema} layout")
        return run_type_for("holo", ver)
    qm = query_mode_of(df)
    if qm is None:
        q = str((script_params or {}).get("query") or "")
        qm = "uniprot" if _AFC_QUERY_RE.match(q) else "pdb"
    return run_type_for("afc" if qm == "uniprot" else "holo", ver)


def gate_from_script_params(script_params: Optional[dict]) -> Tuple[Optional[str], dict]:
    """(holo_overlap_mode, thresholds) as the archive records them; defaults where 2.4 wrote none."""
    sp = script_params or {}
    mode = sp.get("holo_overlap_mode")
    thr = {"min_resis": sp.get("holo_overlap_min_resis", GATE_DEFAULTS["min_resis"]),
           "min_qcov": sp.get("holo_overlap_min_query_cov", GATE_DEFAULTS["min_qcov"]),
           "min_ecov": sp.get("holo_overlap_min_entity_cov", GATE_DEFAULTS["min_ecov"])}
    return mode, {k: float(v) for k, v in thr.items()}


# =============================================================================== profile resolution
def resolve_profile(args, script_params: Optional[dict], df: pd.DataFrame, job_name: str = "",
                    have_sites: Optional[bool] = None) -> RunProfile:
    """The profile in force for one archive: detected (or forced by args.run_type), then overridden by
    the config keys in OVERRIDE_KEYS, then checked against the archive. Raises RunTypeError."""
    detected = detect_run_type(script_params, df)
    wanted = getattr(args, "run_type", None) or "auto"
    if wanted != "auto":
        if wanted not in PROFILES:
            raise RunTypeError(f"unknown run_type {wanted!r}; known: {', '.join(RUN_TYPES)}")
        if wanted != detected:
            raise RunTypeError(f"[{job_name}] run_type {wanted} was requested but the archive is {detected} "
                               f"(AHoJ {ahoj_version(script_params)}, schema {schema_of(df)}, "
                               f"query_mode {query_mode_of(df)})")
    prof = PROFILES[detected]

    # explicit overrides from the config / CLI
    over = {}
    for k in OVERRIDE_KEYS:
        v = getattr(args, k, None)
        if _is_set(v):
            over[k] = v
    if "add_query_row" in over:
        over["add_query_row"] = int(over["add_query_row"])
    if "volume_nan_to_zero" in over:
        over["volume_nan_to_zero"] = bool(int(over["volume_nan_to_zero"]))
    # relabel_apoholo=0 is the legacy switch for 'no relabel at all'
    if int(getattr(args, "relabel_apoholo", 1) or 0) == 0:
        over["relabel_mode"] = "none"
    if over:
        prof = replace(prof, **over)

    # -- checks (loud)
    if prof.relabel_mode not in RELABEL_MODES:
        raise RunTypeError(f"relabel_mode {prof.relabel_mode!r} not in {RELABEL_MODES}")
    if prof.ion_rule not in ION_RULES:
        raise RunTypeError(f"ion_rule {prof.ion_rule!r} not in {ION_RULES}")
    if prof.additive_rule not in ADDITIVE_RULES:
        raise RunTypeError(f"additive_rule {prof.additive_rule!r} not in {ADDITIVE_RULES}")
    if prof.gate not in GATE_MODES:
        raise RunTypeError(f"gate {prof.gate!r} not in {GATE_MODES}")
    if prof.model_row not in MODEL_ROW_MODES:
        raise RunTypeError(f"model_row {prof.model_row!r} not in {MODEL_ROW_MODES}")
    if prof.key_format not in KEY_FORMATS:
        raise RunTypeError(f"key_format {prof.key_format!r} not in {KEY_FORMATS}")
    if detected == "afc-2.4" and prof.add_query_row != 0:
        raise RunTypeError(f"[{job_name}] add_query_row={prof.add_query_row} on an AF-centric archive: the "
                           "query row IS the AlphaFold model and must never join the apo or holo set")
    if prof.relabel_mode == "v2c_ions" and detected == "holo-2.4":
        mode, _ = gate_from_script_params(script_params)
        if mode == "gate" and not int(getattr(args, "allow_undo_gate", 0) or 0):
            raise RunTypeError(f"[{job_name}] relabel_mode v2c_ions would undo AHoJ's holo-overlap gate "
                               "(holo_overlap_mode=gate); use v2c_ions_demote_only, a report_only run, or "
                               "allow_undo_gate=1 for a deliberate v2c-rule comparison")
    if prof.relabel_mode == "entity_gate_site":
        if detected == "holo-2.1":
            raise RunTypeError(f"[{job_name}] entity_gate_site needs ligands.json (AHoJ >= 2.4)")
        if prof.ion_rule in ("site",) or prof.additive_rule == "site":
            if have_sites is None:
                have_sites = bool(getattr(args, "sites_tsv", None)) or bool(_SITES)
            if not have_sites:
                raise RunTypeError(f"[{job_name}] ion_rule/additive_rule 'site' needs a sites table "
                                   "(--sites_tsv or a bundle's sites.tsv) to know which sites are ion- or "
                                   "additive-defined; choose ion_rule nonion|all to run without one")
    if prof.volume_column not in df.columns:
        raise RunTypeError(f"[{job_name}] volume_column {prof.volume_column!r} is not in global_results.csv "
                           f"(available: {[c for c in df.columns if c.startswith('vol')]})")
    if detected != "holo-2.1" and prof.volume_column == "volume":
        vm = (script_params or {}).get("volume_mode")
        if vm is None:
            raise RunTypeError(f"[{job_name}] a 2.4 archive without volume_mode in script_parameters.json")
    return prof


# =============================================================================== job keys
@dataclass(frozen=True)
class JobKey:
    key: str
    kind: str                        # holo | afc
    pdb: Optional[str] = None
    chain: Optional[str] = None
    lig: Optional[str] = None
    resi: Optional[str] = None
    case_suffix: Optional[str] = None  # holo: arm 1's case disambiguator ('u', 'l', 'ul', ...), else None
    uniprot: Optional[str] = None
    site_uid: Optional[str] = None   # afc: <UNP>:ST_100:<n>
    threshold: Optional[str] = None  # afc: ST_100


# holo keys may end in arm 1's case disambiguator (ahoj-db-pipeline queries.disambiguate_keys): keys that
# differ only by case (chains 2Q and 2q) get '.' plus one u/l per case-differing letter position.
# Two separators: '-' (v2c archives, 1abc-A-ATP-501) and '_' (kaleidoscope's AHoJ-schema export, its
# design §3.9: 3cqv_A_HEM_601, 1abc_B_ZN_-5; '-' is ambiguous there, residue numbers can be negative).
# The PDB id is the classic 4 characters or the extended pdb_ + 8. A PDB id starts with a digit or
# 'pdb_', a UniProt accession with a letter, so a holo key is never an afc key.
_PDB_ID = r"(?:[0-9][A-Za-z0-9]{3}|pdb_[A-Za-z0-9]{8})"
_HOLO_KEY_RES = tuple(
    re.compile(rf"^({_PDB_ID}){s}([^{s}]+){s}([^{s}]+){s}(-?[0-9]+[A-Za-z]?)(?:\.([ul]+))?$") for s in "-_")
_AFC_KEY_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9.-]*?)_((?:[A-Za-z]+_)?[0-9]+)_([0-9]+)$")


def _match_holo_key(base: str) -> Optional[re.Match]:
    for r in _HOLO_KEY_RES:
        m = r.match(base)
        if m:
            return m
    return None


def parse_job_key(job_name: str, key_format: Optional[str] = None) -> JobKey:
    """Parse a job key. holo: <pdb>-<chain>-<lig>-<resi>[.<u|l...>] (v2c, e.g. 1abc-A-ATP-501) or
    <pdb>_<chain>_<lig>_<resi>[.<u|l...>] (kaleidoscope's export, e.g. 3cqv_A_HEM_601, 1abc_B_ZN_-5);
    <pdb> is 4 characters or pdb_ + 8. Arm 1's case disambiguator, as in 6xhv-2Q-MG-203.u, stays in
    JobKey.key and is left out of pdb, chain, lig and resi. afc: <UniProt>_<threshold>_<n> (e.g.
    P46883_ST_100_14, the site_uid P46883:ST_100:14 with ':' -> '_'). Raises RunTypeError otherwise."""
    base = os.path.basename(str(job_name))
    for suf in (".tgz", ".tar.gz"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
    m = _match_holo_key(base)
    if key_format in (None, "holo", "any"):
        if m:
            pdb, chain, lig, resi, case_suffix = m.groups()
            return JobKey(base, "holo", pdb=pdb, chain=chain, lig=lig, resi=resi, case_suffix=case_suffix)
        if key_format == "holo":
            raise RunTypeError(f"job key {base!r} is not <pdb>-<chain>-<lig>-<resi>[.<u|l...>] "
                               "(or with '_')")
    # a holo key is not read as afc under key_format afc either: 1abc_A_100_5 (CCD code 100) would pass
    # for <UniProt>_<threshold>_<n>
    a = None if m else _AFC_KEY_RE.match(base)
    if a:
        unp, thr, n = a.groups()
        return JobKey(base, "afc", uniprot=unp, threshold=thr, site_uid=f"{unp}:{thr}:{n}")
    raise RunTypeError(f"job key {base!r} is neither <pdb>-<chain>-<lig>-<resi> (or with '_') "
                       "nor <UniProt>_ST_100_<n>")


def site_uid_to_key(site_uid: str) -> str:
    return str(site_uid).replace(":", "_")


# =============================================================================== sites table
@dataclass
class SiteInfo:
    site_uid: str
    uniprot: Optional[str] = None
    residues: Tuple[int, ...] = ()
    members: Tuple[str, ...] = ()
    ligands: frozenset = frozenset()          # ligand codes of the defining instances
    ion_defined: bool = False                 # any defining ligand is a single-atom ion
    additive_defined: bool = False            # any defining ligand is a crystallisation additive
    source: Optional[str] = None


_SITES: Dict[str, SiteInfo] = {}


def ion_set() -> Set[str]:
    from parsing_functions_DBv2 import EXCLUDED_LIGANDS_TOTAL_LE1
    return {str(x).upper() for x in EXCLUDED_LIGANDS_TOTAL_LE1}


def additive_set(path: Optional[str] = None) -> Set[str]:
    if not path:
        return set(ADDITIVES_DEFAULT)
    with open(path, encoding="utf-8") as fh:
        return {ln.strip().upper() for ln in fh if ln.strip() and not ln.startswith("#")}


def _split_list(v) -> list:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in re.split(r"[;,\s]+", s) if x.strip()]


def load_sites_table(path: str, ions: Optional[Set[str]] = None,
                     additives: Optional[Set[str]] = None) -> Dict[str, SiteInfo]:
    """sites.tsv (run-bundle.md): site_uid, uniprot, residues ('+'-joined), source, members (';'-joined
    ligand instances or v2c jobs), n_members, anchorable, anchor_note. An optional 'ligands' column
    names the defining ligand codes directly; otherwise they are parsed from the members' job keys.
    The pilot's CSV (site_uid, job_names, ...) is accepted too. Keyed by BOTH site_uid and its key form."""
    ions = ion_set() if ions is None else ions
    additives = set(ADDITIVES_DEFAULT) if additives is None else additives
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    t = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if "site_uid" not in t.columns:
        raise RunTypeError(f"{path}: no site_uid column (columns: {list(t.columns)})")
    mem_col = "members" if "members" in t.columns else ("job_names" if "job_names" in t.columns else None)
    out: Dict[str, SiteInfo] = {}
    bad = []
    for r in t.to_dict("records"):
        uid = r["site_uid"].strip()
        members = tuple(_split_list(r.get(mem_col))) if mem_col else ()
        if "ligands" in t.columns and r.get("ligands"):
            ligs = {x.upper() for x in _split_list(r["ligands"])}
        else:
            ligs = set()
            for m in members:
                try:
                    k = parse_job_key(m, "holo")
                except RunTypeError:
                    bad.append((uid, m))
                    continue
                ligs.add(k.lig.upper())
        resis = tuple(int(x) for x in _split_list(str(r.get("residues", "")).replace("+", " ")) if x.isdigit())
        unp = r.get("uniprot") or r.get("site_uniprot") or (uid.split(":")[0] if ":" in uid else None)
        info = SiteInfo(site_uid=uid, uniprot=unp, residues=resis, members=members, ligands=frozenset(ligs),
                        ion_defined=bool(ligs & ions), additive_defined=bool(ligs & additives),
                        source=r.get("source"))
        out[uid] = info
        out[site_uid_to_key(uid)] = info
    if bad:
        ex = "; ".join(f"{u}: {m}" for u, m in bad[:5])
        raise RunTypeError(f"{path}: {len(bad)} member(s) are not <pdb>-<chain>-<lig>-<resi> (or '_') job keys and "
                           f"no 'ligands' column names the defining ligands (first: {ex})")
    return out


def set_sites(sites: Dict[str, SiteInfo]) -> None:
    """Install the sites table for this process (the pool initializer calls this in every worker)."""
    _SITES.clear()
    _SITES.update(sites or {})


def site_for_job(job_name: str, key_format: str) -> Optional[SiteInfo]:
    """The SiteInfo of an AF-centric job (None for a holo key, or when the table lacks the site)."""
    if key_format == "holo":
        return None
    k = parse_job_key(job_name, None if key_format == "any" else "afc")
    if k.kind != "afc":
        return None
    return _SITES.get(k.site_uid) or _SITES.get(k.key)


# =============================================================================== label vocabulary
def unknown_candidate_labels(df: pd.DataFrame, role: str = "C") -> Dict[str, int]:
    """Candidate-row apoholo_assignment values outside KNOWN_CANDIDATE_LABELS, with their counts.

    Such rows join neither the apo nor the holo set. AHoJ's A- was dropped that way without a word
    (AHOJ-020), so the caller reports whatever this returns; a missing label counts as '<NA>'."""
    if "apoholo_assignment" not in df.columns or "role" not in df.columns:
        return {}
    lab = df.loc[df["role"] == role, "apoholo_assignment"].astype(object).where(lambda s: s.notna(), "<NA>")
    lab = lab.astype(str)
    counts = lab[~lab.isin(KNOWN_CANDIDATE_LABELS)].value_counts()
    return {str(k): int(v) for k, v in sorted(counts.items())}


def format_label_counts(counts: Dict[str, int]) -> Optional[str]:
    """{'Hweak': 3, 'X': 1} -> 'Hweak:3;X:1' (None when empty), the summary-row spelling."""
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items())) or None


# =============================================================================== model rows
def _accessions(v) -> Set[str]:
    return set(_split_list(v))


def select_model_rows(df: pd.DataFrame, mode: str = "AF_model") -> pd.DataFrame:
    """The AlphaFold model row(s) of the query's own UniProt. 'first_M': every role-M row, as the legacy
    code took it. 'AF_model': the role-M rows of the query's own model, without UniWeave's neighbour-
    UniProt model rows:
        AHoJ before PR #143 (v2c, 2.4.0, 2.5.1)  structure == 'AF_model'; a neighbour's is 'AF_<accession>'
        since #143 (develop, 2.6.0)              structure is the model id (AF-P0A884-F1-model_v6), and the
                                                 query's own model is the row whose UNPs are all query
                                                 UNPs (AHoJ's rule; a neighbour's UNPs are not)."""
    if "role" not in df.columns:
        return df.iloc[0:0]
    m = df[df["role"] == "M"]
    if mode == "AF_model" and "structure" in df.columns:
        st = m["structure"].astype(str)
        own = st == "AF_model"
        named = ~st.str.startswith("AF_")
        if named.any() and "UNPs" in df.columns:
            query_unps = set().union(*[_accessions(v) for v in df.loc[df["role"] == "Q", "UNPs"]])
            own |= named & m["UNPs"].map(lambda v: bool(_accessions(v)) and _accessions(v) <= query_unps)
        m = m[own]
    return m


# =============================================================================== entity relabel (afc)
def entities_by_pocket(lig_json: Optional[dict]) -> Dict[Tuple[str, str], list]:
    """(structure, pocket_index) -> detected entities, from ligands.json (schema 2)."""
    out: Dict[Tuple[str, str], list] = {}
    for pk in (lig_json or {}).get("pockets", []):
        for c in pk.get("candidates", []):
            out[(str(c.get("structure")), str(c.get("pocket_index")))] = list(c.get("detected_entities", []))
    return out


def entity_qualifies(label, ion_rule: str, additive_rule: str, ions: Set[str], additives: Set[str],
                     site: Optional[SiteInfo]) -> bool:
    """Does a detected entity count as holo evidence?
        ion_rule   all: every entity | nonion: single-atom ions never | site: ions only at ion-defined sites
                   query_disables: not meaningful without a query ligand -> treated as 'site'
        additive_rule  v2c: additives count | site: only at additive-defined sites"""
    lab = str(label).upper()
    if lab in ions and ion_rule != "all":
        if ion_rule == "nonion":
            return False
        if not (site and site.ion_defined):
            return False
    if additive_rule == "site" and lab in additives and not (site and site.additive_defined):
        return False
    return True


def entity_passes_gate(e: dict, gate: Optional[dict]) -> bool:
    if gate is None:
        return True
    try:
        return (float(e.get("overlap_resis") or 0) >= gate["min_resis"]
                and float(e.get("query_pocket_coverage") or 0) >= gate["min_qcov"]
                and float(e.get("entity_contact_coverage") or 0) >= gate["min_ecov"])
    except (TypeError, ValueError):
        return False


def relabel_by_entities(df: pd.DataFrame, lig_json: Optional[dict], gate: Optional[dict], ions: Set[str],
                        additives: Set[str], ion_rule: str, additive_rule: str, site: Optional[SiteInfo],
                        protect_labels: Iterable[str] = PROTECTED_LABELS_24) -> Tuple[pd.DataFrame, pd.Series]:
    """Candidate apo/holo grades from AHoJ's entity evidence (ligands.json), for a query with no ligand.

    Observed candidate pockets only (role C, label not protected, #resis3 > 0). Three grades, all
    pooled by the classifier as today (holo = H*, apo = A*/A**/A):
        H*    a qualifying entity passes the gate                         (holo)
        A*    an entity is present, qualifies, but fails the gate         (apo-contact, AHoJ's A*)
        A**   an entity is present but does not qualify (ion/additive)    (apo, the v2c demotion mark)
        A     nothing detected                                            (apo)
    Returns (relabelled df, changed mask)."""
    if lig_json is None:
        raise RunTypeError("relabel_by_entities needs ligands.json (AHoJ >= 2.4 writes it)")
    ents = entities_by_pocket(lig_json)
    protect = {str(p).upper() for p in protect_labels}
    out = df.copy()
    lab = out["apoholo_assignment"].astype(str).str.upper()
    resis3 = pd.to_numeric(out["#resis3"], errors="coerce")
    obs = (out["role"] == "C") & ~lab.isin(protect | {"NAN", "NONE", ""}) & (resis3 > 0)
    new = out["apoholo_assignment"].copy()
    for idx, st, pk in zip(out.index[obs], out.loc[obs, "structure"].astype(str), out.loc[obs, "pocket"].astype(str)):
        es = ents.get((st, pk), [])
        hit = any(entity_qualifies(e.get("label"), ion_rule, additive_rule, ions, additives, site)
                  and entity_passes_gate(e, gate) for e in es)
        if hit:
            new.at[idx] = "H*"
        elif any(entity_qualifies(e.get("label"), ion_rule, additive_rule, ions, additives, site) for e in es):
            new.at[idx] = "A*"
        elif es:
            new.at[idx] = "A**"
        else:
            new.at[idx] = "A"
    changed = obs & (out["apoholo_assignment"].astype(str) != new.astype(str))
    out["apoholo_assignment"] = new
    return out, changed


# =============================================================================== bundles
def read_run_yaml(bundle_dir: str) -> dict:
    """run.yaml of a bundle, with '_parse_ahoj_run_type' set from run_type and ahoj.version through
    AHOJ_MINORS, the rule detect_run_type applies to each archive ('2.5.1+develop.7fd6825' -> 2.5 ->
    holo-2.4). Raises RunTypeError."""
    p = os.path.join(bundle_dir, "run.yaml")
    if not os.path.isfile(p):
        raise RunTypeError(f"no run.yaml in bundle {bundle_dir}")
    try:
        import yaml  # PyYAML
    except ImportError as e:  # pragma: no cover
        raise RunTypeError("reading run.yaml needs PyYAML (pip install pyyaml)") from e
    with open(p, encoding="utf-8") as fh:
        y = yaml.safe_load(fh) or {}
    rt = str(y.get("run_type") or "").lower()
    ver = str(((y.get("ahoj") or {}).get("version")) or "")
    if rt not in ("afc", "holo"):
        raise RunTypeError(f"{p}: run_type must be 'afc' or 'holo', got {rt!r}")
    m = re.match(r"^(\d+)\.(\d+)", ver)
    if not m:
        raise RunTypeError(f"{p}: ahoj.version {ver!r} is unreadable")
    try:
        y["_parse_ahoj_run_type"] = run_type_for(rt, (int(m.group(1)), int(m.group(2))))
    except RunTypeError as e:
        raise RunTypeError(f"{p}: ahoj.version {ver}: {e}") from None
    return y


def bundle_paths(bundle_dir: str) -> dict:
    return {"run_yaml": os.path.join(bundle_dir, "run.yaml"),
            "sites_tsv": os.path.join(bundle_dir, "sites.tsv"),
            "jobs_tsv": os.path.join(bundle_dir, "jobs.tsv"),
            "archives": os.path.join(bundle_dir, "output", "archives"),
            "meta": os.path.join(bundle_dir, "output", "meta")}


# =============================================================================== summary columns
def provenance_columns(profile: RunProfile, script_params: Optional[dict], df: pd.DataFrame,
                       site: Optional[SiteInfo], job_key: Optional[JobKey]) -> dict:
    """The run-type columns every summary row carries: which AHoJ version and schema wrote the archive,
    which rules parse-ahoj applied, and the query triple."""
    sp = script_params or {}
    mode, thr = gate_from_script_params(script_params)
    row = {"run_type": profile.name,
           "ahoj_version": ahoj_version(script_params),
           "ahoj_schema": schema_of(df),
           "ahoj_date": sp.get("date"),
           "ahoj_volume_mode": sp.get("volume_mode", "pyvol" if profile.name == "holo-2.1" else None),
           "ahoj_pocket_alignment_method": sp.get("pocket_alignment_method", 1 if profile.name == "holo-2.1" else None),
           "ahoj_holo_overlap_mode": mode,
           "ahoj_uniweave": sp.get("uniweave"),
           "ahoj_model_version": sp.get("model_version", sp.get("afdb_version")),
           "rule_add_query_row": profile.add_query_row,
           "rule_relabel_mode": profile.relabel_mode,
           "rule_ion_rule": profile.ion_rule,
           "rule_additive_rule": profile.additive_rule,
           "rule_gate": profile.gate,
           "rule_gate_thresholds": f"{thr['min_resis']:g}/{thr['min_qcov']:g}/{thr['min_ecov']:g}"
                                   if profile.relabel_mode == "entity_gate_site" and profile.gate != "none" else None,
           "rule_volume_column": profile.volume_column,
           "rule_volume_nan_to_zero": int(profile.volume_nan_to_zero),
           "rule_model_row": profile.model_row,
           "delta_anchor": profile.delta_anchor,
           "rog_anchor": profile.rog_anchor,
           "mapped_denominator": profile.mapped_denominator}
    row.update(query_triple(df))
    row["site_uid"] = job_key.site_uid if job_key else None
    row["site_ion_defined"] = int(site.ion_defined) if site else None
    row["site_additive_defined"] = int(site.additive_defined) if site else None
    row["site_n_members"] = len(site.members) if site else None
    return row
