# -*- coding: utf-8 -*-
"""Run-type paths (docs/RUN_TYPES.md): detection, keys, sites, relabel rules, the loud failures, and
end-to-end classification on small fixture archives.

Fixtures (tests/fixtures, read-only copies, 8-31 KB each):
  v2c_000/1a7s-A-EOH-429.tgz   AHoJ 2.1.0, AHoJ-DB v2c. Released: n_apo 2, n_holo 1, P(holo) 0.255, Apo.
  v2c_000/1a0b-A-ZN-800.tgz    AHoJ 2.1.0, ion query (ZN). Released: n_apo 1, n_holo 1, P(holo) 0.873, Holo.
  af2/A0A094YZ24_ST_100_1.tgz  AHoJ 2.4.0 UniProt-anchored (pilot af2). gate_site: 2 / 4 / 0.67 / holo.
  af2/Q5H8A6_ST_100_4.tgz      same, an ion-defined site (CA). gate_site: 4 / 2 / 0.762 / holo; nonion: 6 / 0.
  hq1/<same keys>.tgz          AHoJ 2.4.0 on the rep job's holo query (pilot hq1). v2c relabel:
                               A0A094YZ24 2 / 2 / 0.524; Q5H8A6 4 / 1 / 0.637.
  dev7fd6825/8i5b-A-Y01-2005.tgz  AHoJ develop 7fd6825 (script_version 2.5.1, gate), holo query from arm 1's
                               calibration repeat (holo-cal-biolip-500-dev7fd6825). One candidate is A- (21tq A
                               p1, query ligand present, weak_holo). The 2.4.0 archive of the job labels it A:
                               holo-2.4 gives 12 / 17 / 0.716 there; without A- in the apo set develop gave 11 / 0.72.
Expected values: the full_run_3 release (query_summaries_do_kde_plots_and_classification.csv) and the
pilot's pilot_sites_collected.csv (ahoj-af-project, AF_CENTRIC_RERUN.md).
"""
import json
import os
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
FIX = os.path.join(HERE, "fixtures")

import run_types as rt  # noqa: E402
from run_types import RunTypeError  # noqa: E402

V2C_A = os.path.join(FIX, "v2c_000", "1a7s-A-EOH-429.tgz")
V2C_ZN = os.path.join(FIX, "v2c_000", "1a0b-A-ZN-800.tgz")
AF2_A = os.path.join(FIX, "af2", "A0A094YZ24_ST_100_1.tgz")
AF2_ION = os.path.join(FIX, "af2", "Q5H8A6_ST_100_4.tgz")
HQ1_A = os.path.join(FIX, "hq1", "A0A094YZ24_ST_100_1.tgz")
HQ1_ION = os.path.join(FIX, "hq1", "Q5H8A6_ST_100_4.tgz")
SITES = os.path.join(FIX, "sites.tsv")
DEV_AMINUS = os.path.join(FIX, "dev7fd6825", "8i5b-A-Y01-2005.tgz")


def make_args(tmp_path, **over):
    """The full_run_3 classifier config, numbers only (no plots, no per-entry files)."""
    from types import SimpleNamespace
    with open(os.path.join(REPO, "configs", "run_config_full_run_3.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.update(dict(general_violins=0, dp_curves=0, kde_curves=0, kde_violins=0, conf_bars=0,
                    save_statistics=0, save_filtered_df=0, save_per_method_json=0, save_relabeled_csv=0,
                    zip_plots_dir=0, zip_stats_dir=0, keep_no_model=0, root_mirror_dir=str(tmp_path / "mirror"),
                    run_type="auto", bundle=None, key_format=None, allow_undo_gate=0, sites_tsv=None,
                    relabel_mode=None, ion_rule=None, additive_rule=None, gate=None, volume_column=None,
                    volume_nan_to_zero=None, model_row=None, additives_file=None, add_query_row=None))
    cfg.update(over)
    return SimpleNamespace(**cfg)


def classify(args, path):
    from batch_job_functions import do_kde_plots_and_classification
    job = os.path.basename(path)[:-4]
    status, detail = do_kde_plots_and_classification(args, job, path)
    return status, (detail[4].iloc[0].to_dict() if status == "Success" else str(detail[1]))


@pytest.fixture(autouse=True)
def _clear_sites():
    rt.set_sites({})
    yield
    rt.set_sites({})


# ------------------------------------------------------------------ keys
def test_parse_job_key_holo_and_afc():
    k = rt.parse_job_key("4v85-AL-MG-201")
    assert (k.kind, k.pdb, k.chain, k.lig, k.resi) == ("holo", "4v85", "AL", "MG", "201")
    k = rt.parse_job_key("P46883_ST_100_14.tgz")
    assert (k.kind, k.uniprot, k.site_uid, k.threshold) == ("afc", "P46883", "P46883:ST_100:14", "ST_100")
    assert rt.site_uid_to_key("P46883:ST_100:14") == "P46883_ST_100_14"
    with pytest.raises(RunTypeError):
        rt.parse_job_key("P46883_ST_100_14", "holo")
    with pytest.raises(RunTypeError):
        rt.parse_job_key("4v85-AL-MG-201", "afc")
    with pytest.raises(RunTypeError):
        rt.parse_job_key("not a key")


def test_parse_job_key_accepts_arm1_case_suffix():
    # ahoj-db-pipeline queries.disambiguate_keys: keys equal case-insensitively get .<u|l per differing
    # letter>. 6xhv-2Q-MG-203.u is a job of holo-cal-biolip-500; its sibling 6xhv-2q-MG-203.l is not.
    u = rt.parse_job_key("6xhv-2Q-MG-203.u", "holo")
    assert (u.key, u.kind, u.pdb, u.chain, u.lig, u.resi, u.case_suffix) == \
        ("6xhv-2Q-MG-203.u", "holo", "6xhv", "2Q", "MG", "203", "u")
    lo = rt.parse_job_key("6xhv-2q-MG-203.l.tgz", "holo")
    assert (lo.key, lo.chain, lo.lig, lo.resi, lo.case_suffix) == ("6xhv-2q-MG-203.l", "2q", "MG", "203", "l")
    k = rt.parse_job_key("1abc-Ab-HEM-133A.ul", "any")        # two case-differing letters, insertion code
    assert (k.kind, k.chain, k.resi, k.case_suffix) == ("holo", "Ab", "133A", "ul")
    assert rt.parse_job_key("4v85-AL-MG-201").case_suffix is None
    for bad in ("6xhv-2Q-MG-203.x", "6xhv-2Q-MG-203.2", "6xhv-2Q-MG-203.", "6xhv-2Q-MG-203.U"):
        with pytest.raises(RunTypeError):
            rt.parse_job_key(bad, "holo")
        with pytest.raises(RunTypeError):
            rt.parse_job_key(bad)


def test_parse_job_key_accepts_underscore_holo_keys():
    # kaleidoscope's AHoJ-schema export (its design §3.9) writes <pdb>_<chain>_<lig>_<resi>: residue
    # numbers can be negative, so '-' would be ambiguous there. v2c's hyphen form stays.
    k = rt.parse_job_key("3cqv_A_HEM_601", "holo")
    assert (k.key, k.kind, k.pdb, k.chain, k.lig, k.resi, k.case_suffix) == \
        ("3cqv_A_HEM_601", "holo", "3cqv", "A", "HEM", "601", None)
    k = rt.parse_job_key("1abc_B_ZN_-5.tgz")
    assert (k.key, k.kind, k.pdb, k.chain, k.lig, k.resi) == ("1abc_B_ZN_-5", "holo", "1abc", "B", "ZN", "-5")
    k = rt.parse_job_key("6xhv_2Q_MG_203.u", "any")
    assert (k.key, k.kind, k.chain, k.lig, k.resi, k.case_suffix) == ("6xhv_2Q_MG_203.u", "holo", "2Q", "MG", "203", "u")
    assert rt.parse_job_key("1a3n_A_HEM_142").kind == "holo"
    assert rt.parse_job_key("1abc_A_HEM_133A").resi == "133A"
    k = rt.parse_job_key("1abc-B-ZN--5")                       # the hyphen form with a negative number
    assert (k.chain, k.lig, k.resi) == ("B", "ZN", "-5")
    # extended PDB ids, pdb_ + 8 characters, in both forms
    for key in ("pdb_00001abc_A_HEM_601", "pdb_00001abc-A-HEM-601"):
        k = rt.parse_job_key(key, "holo")
        assert (k.pdb, k.chain, k.lig, k.resi) == ("pdb_00001abc", "A", "HEM", "601")
    # an afc key is still afc: a UniProt accession starts with a letter, a PDB id with a digit
    k = rt.parse_job_key("P69905_ST_100_4")
    assert (k.kind, k.uniprot, k.threshold, k.site_uid) == ("afc", "P69905", "ST_100", "P69905:ST_100:4")
    with pytest.raises(RunTypeError):
        rt.parse_job_key("P69905_ST_100_4", "holo")
    # and a holo key is never afc, even when a numeric CCD code makes it look like one
    for key in ("3cqv_A_HEM_601", "1abc_A_100_5"):
        assert rt.parse_job_key(key, "any").kind == "holo"
        with pytest.raises(RunTypeError):
            rt.parse_job_key(key, "afc")
    # separators are not mixed; a short or long id is not a PDB id
    for bad in ("3cqv_A-HEM_601", "3cqv-A_HEM-601", "3cqv_A_HEM", "3cqv_A_HEM_601.U", "3cqv__HEM_601",
                "pdb_0001abc_A_HEM_601", "13cqv_A_HEM_601"):
        with pytest.raises(RunTypeError):
            rt.parse_job_key(bad)


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("path,expected,schema,qmode", [
    (V2C_A, "holo-2.1", "v2.1", None), (AF2_A, "afc-2.4", "v2.4", "uniprot"), (HQ1_A, "holo-2.4", "v2.4", "pdb")])
def test_detect_run_type_from_archive(path, expected, schema, qmode):
    df = rt.load_global_results(path)
    sp = rt.load_script_parameters(path)
    assert rt.detect_run_type(sp, df) == expected
    assert rt.schema_of(df) == schema
    assert rt.query_mode_of(df) == qmode
    assert rt.ahoj_version(sp) in ("2.1.0", "2.4.0")


def test_detect_without_script_parameters_uses_layout():
    df = rt.load_global_results(AF2_A)
    assert rt.detect_run_type(None, df) == "afc-2.4"
    assert rt.detect_run_type(None, rt.load_global_results(V2C_A)) == "holo-2.1"
    with pytest.raises(RunTypeError):
        rt.detect_run_type({"script_version": "2.2.0"}, rt.load_global_results(V2C_A))


def test_member_name_fallback_reads_both_spellings(tmp_path):
    # the pilot's packing stores './global_results.csv'; a bundle may store 'global_results.csv'
    src = rt.read_member(V2C_A, "global_results.csv")
    p = tmp_path / "bare.tgz"
    with tarfile.open(p, "w:gz") as tar:
        import io
        ti = tarfile.TarInfo("global_results.csv")
        ti.size = len(src)
        tar.addfile(ti, io.BytesIO(src))
    assert rt.load_global_results(str(p)).shape == rt.load_global_results(V2C_A).shape
    assert rt.read_member(V2C_A, "does_not_exist.txt") is None


# ------------------------------------------------------------------ profiles and loud failures
def test_profiles_carry_the_author_key_point():
    assert rt.PROFILES["holo-2.1"].add_query_row == 1
    assert rt.PROFILES["holo-2.4"].add_query_row == 1
    assert rt.PROFILES["afc-2.4"].add_query_row == 0
    assert rt.PROFILES["afc-2.4"].relabel_mode == "entity_gate_site"
    assert rt.PROFILES["holo-2.4"].relabel_mode == "v2c_ions_demote_only"
    assert rt.PROFILES["holo-2.1"].protect_labels == frozenset({"U"})
    assert rt.PROFILES["afc-2.4"].protect_labels == frozenset({"U", "*", "-"})


def test_resolve_profile_refuses_add_query_row_on_afc(tmp_path):
    df, sp = rt.load_global_results(AF2_A), rt.load_script_parameters(AF2_A)
    with pytest.raises(RunTypeError, match="IS the AlphaFold model"):
        rt.resolve_profile(make_args(tmp_path, add_query_row=1, sites_tsv=SITES), sp, df, "x")
    prof = rt.resolve_profile(make_args(tmp_path, sites_tsv=SITES), sp, df, "x")
    assert prof.name == "afc-2.4" and prof.add_query_row == 0


def test_resolve_profile_refuses_mismatch_and_gate_undo(tmp_path):
    df, sp = rt.load_global_results(V2C_A), rt.load_script_parameters(V2C_A)
    with pytest.raises(RunTypeError, match="requested but the archive is holo-2.1"):
        rt.resolve_profile(make_args(tmp_path, run_type="afc-2.4"), sp, df, "x")
    df, sp = rt.load_global_results(HQ1_A), rt.load_script_parameters(HQ1_A)
    with pytest.raises(RunTypeError, match="undo AHoJ's holo-overlap gate"):
        rt.resolve_profile(make_args(tmp_path, relabel_mode="v2c_ions", key_format="any"), sp, df, "x")
    prof = rt.resolve_profile(make_args(tmp_path, relabel_mode="v2c_ions", key_format="any", allow_undo_gate=1), sp, df, "x")
    assert prof.relabel_mode == "v2c_ions"
    df, sp = rt.load_global_results(AF2_A), rt.load_script_parameters(AF2_A)
    with pytest.raises(RunTypeError, match="needs a sites table"):
        rt.resolve_profile(make_args(tmp_path), sp, df, "x")
    with pytest.raises(RunTypeError, match="volume_column"):
        rt.resolve_profile(make_args(tmp_path, sites_tsv=SITES, volume_column="volume_pyvol"), sp, df, "x")
    prof = rt.resolve_profile(make_args(tmp_path, sites_tsv=SITES, volume_column="volume2"), sp, df, "x")
    assert prof.volume_column == "volume2"


def test_relabel_apoholo_0_means_none(tmp_path):
    df, sp = rt.load_global_results(V2C_A), rt.load_script_parameters(V2C_A)
    assert rt.resolve_profile(make_args(tmp_path, relabel_apoholo=0), sp, df, "x").relabel_mode == "none"


# ------------------------------------------------------------------ sites
def test_sites_table_ion_and_additive_flags():
    s = rt.load_sites_table(SITES)
    a, b = s["A0A094YZ24:ST_100:1"], s["Q5H8A6:ST_100:4"]
    assert s["Q5H8A6_ST_100_4"] is b                       # keyed by site_uid and by its key form
    assert a.ligands == frozenset({"ELU"}) and not a.ion_defined and not a.additive_defined
    assert b.ligands == frozenset({"CA"}) and b.ion_defined
    assert a.residues[:3] == (51, 52, 54) and len(b.residues) == 7 and b.uniprot == "Q5H8A6"
    rt.set_sites(s)
    assert rt.site_for_job("Q5H8A6_ST_100_4", "afc") is b
    assert rt.site_for_job("Q5H8A6_ST_100_4", "any") is b
    assert rt.site_for_job("4v85-AL-MG-201", "holo") is None
    assert rt.site_for_job("4v85-AL-MG-201", "any") is None


def test_sites_table_bad_member_is_loud(tmp_path):
    p = tmp_path / "sites.tsv"
    p.write_text("site_uid\tuniprot\tresidues\tmembers\nP1:ST_100:1\tP1\t1+2\tbiolip:1abc:A:ATP\n", encoding="utf-8")
    with pytest.raises(RunTypeError, match="not <pdb>-<chain>-<lig>-<resi>"):
        rt.load_sites_table(str(p))
    p.write_text("site_uid\tuniprot\tresidues\tmembers\tligands\nP1:ST_100:1\tP1\t1+2\tbiolip:1abc:A:ATP\tSO4;ATP\n", encoding="utf-8")
    s = rt.load_sites_table(str(p))["P1:ST_100:1"]
    assert s.ligands == frozenset({"SO4", "ATP"}) and s.additive_defined and not s.ion_defined


def test_sites_table_members_keep_the_case_suffix(tmp_path):
    # the BioLiP harvest's sites_st100.tsv row P60489:ST_100:42: members are the instance keys, suffix
    # included (355 such members in biolip-2026-04-30); without a ligands column the codes come from them
    members = "4y4p-2Q-MG-3003;5hcp-2Q-MG-3005;6oxa-YQ-MG-3003;6xhv-2Q-MG-203.u;9o3j-2x-MG-101.l"
    p = tmp_path / "sites.tsv"
    p.write_text(f"site_uid\tuniprot\tresidues\tmembers\nP60489:ST_100:42\tP60489\t76+78+89\t{members}\n",
                 encoding="utf-8")
    s = rt.load_sites_table(str(p))["P60489:ST_100:42"]
    assert s.members == tuple(members.split(";")) and s.ligands == frozenset({"MG"}) and s.ion_defined


def test_sites_table_members_in_the_underscore_form(tmp_path):
    members = "3cqv_A_HEM_601;1a3n_A_HEM_142;1abc_B_ZN_-5;6xhv_2Q_MG_203.u"
    p = tmp_path / "sites.tsv"
    p.write_text(f"site_uid\tuniprot\tresidues\tmembers\nP69905:ST_100:4\tP69905\t58+87\t{members}\n",
                 encoding="utf-8")
    s = rt.load_sites_table(str(p))["P69905:ST_100:4"]
    assert s.members == tuple(members.split(";")) and s.ligands == frozenset({"HEM", "ZN", "MG"})


# ------------------------------------------------------------------ relabel rules
def _entities(structure, pocket, ents):
    return {"pockets": [{"candidates": [{"structure": structure, "pocket_index": pocket, "detected_entities": ents}]}]}


def _ent(label, ov=5, qcov=50.0, ecov=80.0):
    return {"label": label, "overlap_resis": ov, "query_pocket_coverage": qcov, "entity_contact_coverage": ecov}


def _cand_df(labels):
    return pd.DataFrame({"structure": ["1abc"] * len(labels), "pocket": [f"p{i}" for i in range(len(labels))],
                         "role": ["C"] * len(labels), "apoholo_assignment": labels, "#resis3": [10] * len(labels)})


def test_relabel_by_entities_grades_and_protections():
    ions, adds = {"ZN", "CA", "MG"}, {"SO4"}
    df = _cand_df(["A", "A", "A", "A", "U", "*", "-", "H*"])
    lj = {"pockets": [{"candidates": [
        {"structure": "1abc", "pocket_index": "p0", "detected_entities": [_ent("ATP")]},             # hit -> H*
        {"structure": "1abc", "pocket_index": "p1", "detected_entities": [_ent("ATP", qcov=5.0)]},   # fails gate -> A*
        {"structure": "1abc", "pocket_index": "p2", "detected_entities": [_ent("ZN")]},              # ion, site not ion-defined -> A**
        {"structure": "1abc", "pocket_index": "p3", "detected_entities": []},                        # nothing -> A
        {"structure": "1abc", "pocket_index": "p4", "detected_entities": [_ent("ATP")]},             # U protected
        {"structure": "1abc", "pocket_index": "p5", "detected_entities": [_ent("ATP")]},             # * protected
        {"structure": "1abc", "pocket_index": "p6", "detected_entities": [_ent("ATP")]},             # - protected
        {"structure": "1abc", "pocket_index": "p7", "detected_entities": [_ent("SO4")]},             # additive counts (v2c)
    ]}]}
    site = rt.SiteInfo("X:ST_100:1", ion_defined=False, additive_defined=False)
    out, changed = rt.relabel_by_entities(df, lj, rt.GATE_DEFAULTS, ions, adds, "site", "v2c", site)
    assert out.apoholo_assignment.tolist() == ["H*", "A*", "A**", "A", "U", "*", "-", "H*"]
    # ion-defined site: the ion counts
    out, _ = rt.relabel_by_entities(df, lj, rt.GATE_DEFAULTS, ions, adds, "site",
                                    "v2c", rt.SiteInfo("X", ion_defined=True))
    assert out.apoholo_assignment.iloc[2] == "H*"
    # nonion: never; all: always; additive_rule site: SO4 out unless additive-defined
    assert rt.relabel_by_entities(df, lj, rt.GATE_DEFAULTS, ions, adds, "nonion", "v2c",
                                  rt.SiteInfo("X", ion_defined=True))[0].apoholo_assignment.iloc[2] == "A**"
    assert rt.relabel_by_entities(df, lj, rt.GATE_DEFAULTS, ions, adds, "all", "v2c", None)[0].apoholo_assignment.iloc[2] == "H*"
    assert rt.relabel_by_entities(df, lj, rt.GATE_DEFAULTS, ions, adds, "site", "site", site)[0].apoholo_assignment.iloc[7] == "A**"
    # no gate: the v2c contact rule
    assert rt.relabel_by_entities(df, lj, None, ions, adds, "site", "v2c", site)[0].apoholo_assignment.iloc[1] == "H*"
    with pytest.raises(RunTypeError):
        rt.relabel_by_entities(df, None, None, ions, adds, "site", "v2c", site)


def test_relabel_by_entities_on_fixture_matches_ahoj_gate_labels():
    # at a non-ion site with the gate applied, the entity relabel must agree with AHoJ 2.4.0's own labels
    df = rt.load_global_results(AF2_A)
    out, changed = rt.relabel_by_entities(df, rt.load_ligands_json(AF2_A), rt.GATE_DEFAULTS, rt.ion_set(),
                                          set(rt.ADDITIVES_DEFAULT), "site", "v2c",
                                          rt.SiteInfo("A0A094YZ24:ST_100:1"))
    c = out[out.role == "C"]
    assert c.apoholo_assignment.tolist() == df[df.role == "C"].apoholo_assignment.tolist()
    assert not changed.any()


def test_relabel_apoholo_by_ions_demote_only_never_promotes():
    from parsing_functions_DBv2 import relabel_apoholo_by_ions
    df = pd.DataFrame({"role": ["Q", "C", "C", "C", "C"],
                       "apoholo_assignment": ["H", "H*", "A", "A", "U"],
                       "query_ligs": ["ATP", None, None, None, None],
                       "nonquery_ligs": [None, "ZN", "ATP", None, "ATP"]})
    full, _ = relabel_apoholo_by_ions(df.copy(), builtin="total_le1", verbose=False)
    assert full.apoholo_assignment.tolist() == ["H", "A**", "H*", "A", "U"]     # the v2c rule promotes row 2
    dem, _ = relabel_apoholo_by_ions(df.copy(), builtin="total_le1", verbose=False, demote_only=True)
    assert dem.apoholo_assignment.tolist() == ["H", "A**", "A", "A", "U"]       # demote only
    prot, _ = relabel_apoholo_by_ions(df.assign(apoholo_assignment=["H", "*", "-", "A", "U"]), builtin="total_le1",
                                      verbose=False, protect_labels={"U", "*", "-"})
    assert prot.apoholo_assignment.tolist() == ["H", "*", "-", "A", "U"]


def test_select_model_rows_ignores_uniweave_neighbours():
    df = pd.DataFrame({"role": ["Q", "M", "M", "C"], "structure": ["1abc", "AF-P1-F1-model_v6_neighbour", "AF_model", "2xyz"]})
    assert rt.select_model_rows(df).structure.tolist() == ["AF_model"]
    assert rt.select_model_rows(df, "first_M").structure.tolist()[0].startswith("AF-P1")
    assert rt.select_model_rows(pd.DataFrame({"role": ["Q"]})).empty


# ------------------------------------------------------------------ end to end
@pytest.mark.parametrize("path,n_apo,n_holo,p,cls", [
    (V2C_A, 2, 1, 0.255, "Apo"), (V2C_ZN, 1, 1, 0.873, "Holo")])
def test_legacy_reproduces_release(tmp_path, path, n_apo, n_holo, p, cls):
    status, row = classify(make_args(tmp_path), path)
    assert status == "Success", row
    assert (row["n_apo"], row["n_holo"]) == (n_apo, n_holo)
    assert abs(row["pocket_avg_P(holo)"] - p) < 1e-9 and row["pocket_overall_classification"] == cls
    assert row["run_type"] == "holo-2.1" and row["ahoj_version"] == "2.1.0" and row["ahoj_schema"] == "v2.1"
    assert row["rule_add_query_row"] == 1 and row["rule_relabel_mode"] == "v2c_ions"
    assert row["query_mode"] is None and row["chain_axes_meaningful"] == 1


@pytest.mark.parametrize("path,n_apo,n_holo,p,ion_defined", [
    (AF2_A, 2, 4, 0.67, 0), (AF2_ION, 4, 2, 0.762, 1)])
def test_afc_reproduces_pilot_gate_site(tmp_path, path, n_apo, n_holo, p, ion_defined):
    rt.set_sites(rt.load_sites_table(SITES))
    status, row = classify(make_args(tmp_path, sites_tsv=SITES), path)
    assert status == "Success", row
    assert (row["n_apo"], row["n_holo"]) == (n_apo, n_holo)
    assert abs(row["pocket_avg_P(holo)"] - p) < 1e-9 and row["pocket_overall_classification"] == "Holo"
    assert row["run_type"] == "afc-2.4" and row["rule_add_query_row"] == 0
    assert row["rule_relabel_mode"] == "entity_gate_site" and row["rule_ion_rule"] == "site"
    assert row["rule_gate_thresholds"] == "2/20/30" and row["query_mode"] == "uniprot"
    assert row["query_model"].startswith("AF-") and row["site_ion_defined"] == ion_defined
    assert row["query_ligands"] is None and np.isnan(row["query_resolution"]) and np.isnan(row["model_pocket_rms"])
    assert row["chain_overall_classification"] == "N/A (model is anchor)" and row["chain_axes_meaningful"] == 0


def test_afc_ion_rule_nonion_removes_ion_evidence(tmp_path):
    rt.set_sites(rt.load_sites_table(SITES))
    status, row = classify(make_args(tmp_path, sites_tsv=SITES, ion_rule="nonion"), AF2_ION)
    assert status == "Success", row
    assert (row["n_apo"], row["n_holo"]) == (6, 0)          # the pilot's af2_gate_nonion_* for this site


def test_afc_without_sites_is_an_error(tmp_path):
    status, reason = classify(make_args(tmp_path), AF2_A)
    assert status == "Error" and "needs a sites table" in reason
    status, reason = classify(make_args(tmp_path, sites_tsv=SITES, add_query_row=1), AF2_A)
    assert status == "Error" and "IS the AlphaFold model" in reason


@pytest.mark.parametrize("path,n_apo,n_holo,p", [(HQ1_A, 2, 2, 0.524), (HQ1_ION, 4, 1, 0.637)])
def test_holo24_v2c_relabel_reproduces_pilot_hq1(tmp_path, path, n_apo, n_holo, p):
    # the pilot's hq1_ahoj_* used the v2c relabel on 2.4 labels; the profile refuses that unless asked
    status, reason = classify(make_args(tmp_path, key_format="any", relabel_mode="v2c_ions"), path)
    assert status == "Error" and "undo AHoJ's holo-overlap gate" in reason
    status, row = classify(make_args(tmp_path, key_format="any", relabel_mode="v2c_ions", allow_undo_gate=1), path)
    assert status == "Success", row
    assert (row["n_apo"], row["n_holo"]) == (n_apo, n_holo) and abs(row["pocket_avg_P(holo)"] - p) < 1e-9
    assert row["run_type"] == "holo-2.4" and row["query_mode"] == "pdb" and row["rule_add_query_row"] == 1


def test_holo24_profile_default_runs_and_keeps_gate(tmp_path):
    status, reason = classify(make_args(tmp_path), HQ1_A)      # site-named holo run: the holo key check fires
    assert status == "Error" and "not <pdb>-<chain>-<lig>-<resi>" in reason
    status, row = classify(make_args(tmp_path, key_format="any"), HQ1_A)
    assert status == "Success", row
    assert row["rule_relabel_mode"] == "v2c_ions_demote_only" and row["ahoj_holo_overlap_mode"] == "gate"
    assert row["rule_volume_column"] == "volume" and row["ahoj_volume_mode"] == "convex"
    assert row["chain_axes_meaningful"] == 1 and row["rule_add_query_row"] == 1
    status, row2 = classify(make_args(tmp_path, key_format="any", volume_column="volume2"), HQ1_A)
    assert status == "Success" and row2["rule_volume_column"] == "volume2"


# ------------------------------------------------------------------ bundles
def _bundle(tmp_path, run_type="afc", version="2.4.0", archive=AF2_A, name=None):
    b = tmp_path / f"{run_type}-test"
    (b / "output" / "archives" / "ab").mkdir(parents=True)
    (b / "run.yaml").write_text(
        f"bundle_version: 0\nrun_id: {run_type}-test\nrun_type: {run_type}\nahoj:\n  repo: ioChris/AHoJ2\n  ref: v2.4.x\n"
        f"  commit: 5c3a191\n  version: {version}\n  args: [--models, afdb]\n", encoding="utf-8")
    if run_type == "afc":
        shutil.copy(SITES, b / "sites.tsv")
    shutil.copy(archive, b / "output" / "archives" / "ab" / (f"{name}.tgz" if name else os.path.basename(archive)))
    return b


def test_read_run_yaml_and_bundle_paths(tmp_path):
    b = _bundle(tmp_path)
    y = rt.read_run_yaml(str(b))
    assert y["_parse_ahoj_run_type"] == "afc-2.4" and y["ahoj"]["commit"] == "5c3a191"
    bp = rt.bundle_paths(str(b))
    assert os.path.isfile(bp["sites_tsv"]) and os.path.isdir(bp["archives"])
    with pytest.raises(RunTypeError):
        rt.read_run_yaml(str(_bundle(tmp_path / "x", run_type="afc", version="2.2.0")))
    with pytest.raises(RunTypeError):
        rt.read_run_yaml(str(_bundle(tmp_path / "y", run_type="mixed")))
    with pytest.raises(RunTypeError):
        rt.read_run_yaml(str(tmp_path))


def test_ahoj_25_takes_the_24_profiles_in_archives_and_bundles(tmp_path):
    """§8 item 9: one table (AHOJ_MINORS) for detect_run_type and read_run_yaml. 2.5.x and 2.6.x read as
    2.4 (2.6.0 only renames, tests/test_ahoj_names.py); a minor nobody has checked (2.7, 3.0, 2.3) is a
    loud failure in both."""
    hq1, af2 = rt.load_global_results(HQ1_A), rt.load_global_results(AF2_A)
    for v in ("2.4.0", "2.5.1", "2.5.1+develop.7fd6825", "2.6.0"):
        assert rt.detect_run_type({"script_version": v}, hq1) == "holo-2.4"
        assert rt.detect_run_type({"script_version": v}, af2) == "afc-2.4"
    assert rt.detect_run_type(rt.load_script_parameters(DEV_AMINUS), rt.load_global_results(DEV_AMINUS)) == "holo-2.4"
    for v in ("2.7.0", "3.0.0", "2.3.1"):
        with pytest.raises(RunTypeError, match="has no parse-ahoj profile"):
            rt.detect_run_type({"script_version": v}, hq1)
    assert rt.known_ahoj_minors() == "2.1.x -> holo-2.1; 2.4.x, 2.5.x, 2.6.x -> holo-2.4 / afc-2.4"
    cases = [("holo", "2.5.1+develop.7fd6825", "holo-2.4"), ("afc", "2.5.1", "afc-2.4"),
             ("holo", "2.4.0", "holo-2.4"), ("holo", "2.1.0", "holo-2.1"), ("afc", "2.6.0", "afc-2.4")]
    for i, (kind, v, want) in enumerate(cases):
        assert rt.read_run_yaml(str(_bundle(tmp_path / f"ok{i}", run_type=kind, version=v)))["_parse_ahoj_run_type"] == want
    for i, (kind, v) in enumerate([("holo", "2.7.0"), ("afc", "3.0.0"), ("afc", "2.1.0")]):
        with pytest.raises(RunTypeError, match="has no parse-ahoj profile"):
            rt.read_run_yaml(str(_bundle(tmp_path / f"bad{i}", run_type=kind, version=v)))


def test_batch_driver_reads_bundle_end_to_end(tmp_path, monkeypatch):
    """batch_process_database.main on a one-job bundle: run type from run.yaml, sites from sites.tsv,
    archives from output/archives/<ab>/, the summary written where --summary_dir says."""
    import batch_process_database as bpd
    b = _bundle(tmp_path)
    summary = tmp_path / "summary"
    argv = ["batch_process_database.py", "--config", os.path.join(REPO, "configs", "run_config_afc_2.4.json"),
            "--bundle", str(b), "--summary_dir", str(summary), "--root_mirror_dir", str(tmp_path / "mirror"),
            "--num_processes", "1", "--dp_curves", "0", "--kde_curves", "0", "--kde_violins", "0",
            "--save_statistics", "0", "--save_filtered_df", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(REPO)
    bpd.main()
    runs = [d for d in summary.iterdir() if d.is_dir()]
    assert len(runs) == 1
    q = pd.read_csv(runs[0] / "query_summaries_do_kde_plots_and_classification.csv")
    assert q.job_name.tolist() == ["A0A094YZ24_ST_100_1"]
    assert (q.n_apo.iloc[0], q.n_holo.iloc[0], q.run_type.iloc[0]) == (2, 4, "afc-2.4")
    man = json.load(open(runs[0] / "run_manifest.json", encoding="utf-8"))
    assert man["run_type"] == "afc-2.4" and man["run_yaml"]["run_id"] == "afc-test" and man["run_types_in_summary"] == ["afc-2.4"]


@pytest.mark.parametrize("name", ["8i5b-A-Y01-2005.u", "8i5b_A_Y01_2005.u"])
def test_batch_driver_reads_a_develop_bundle_with_a_case_suffixed_key(tmp_path, monkeypatch, name):
    """Arm 1's develop repeat in small: run.yaml ahoj.version 2.5.1+develop.7fd6825 and a job key with
    the case suffix (the shape of 6xhv-2Q-MG-203.u). Both used to stop the bundle. The '_' spelling is
    kaleidoscope's export."""
    import batch_process_database as bpd
    b = _bundle(tmp_path, run_type="holo", version="2.5.1+develop.7fd6825", archive=DEV_AMINUS, name=name)
    summary = tmp_path / "summary"
    argv = ["batch_process_database.py", "--config", os.path.join(REPO, "configs", "run_config_holo_2.4.json"),
            "--bundle", str(b), "--summary_dir", str(summary), "--root_mirror_dir", str(tmp_path / "mirror"),
            "--num_processes", "1", "--dp_curves", "0", "--kde_curves", "0", "--kde_violins", "0",
            "--save_statistics", "0", "--save_filtered_df", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(REPO)
    bpd.main()
    runs = [d for d in summary.iterdir() if d.is_dir()]
    q = pd.read_csv(runs[0] / "query_summaries_do_kde_plots_and_classification.csv")
    assert q.job_name.tolist() == [name]
    assert (q.n_apo.iloc[0], q.n_holo.iloc[0], q.run_type.iloc[0], q.ahoj_version.iloc[0]) == (12, 17, "holo-2.4", "2.5.1")
    man = json.load(open(runs[0] / "run_manifest.json", encoding="utf-8"))
    assert man["run_type"] == "holo-2.4" and man["run_yaml"]["ahoj"]["version"] == "2.5.1+develop.7fd6825"


def test_batch_driver_refuses_run_type_conflict_and_missing_paths(tmp_path, monkeypatch):
    import batch_process_database as bpd
    b = _bundle(tmp_path)
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["x", "--bundle", str(b), "--run_type", "holo-2.4", "--summary_dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="run.yaml says afc-2.4"):
        bpd.main()
    monkeypatch.setattr(sys, "argv", ["x", "--summary_dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="root_data_dir"):
        bpd.main()
    monkeypatch.setattr(sys, "argv", ["x", "--bundle", str(b)])
    with pytest.raises(SystemExit, match="summary_dir"):
        bpd.main()


def test_uniprot_residue_collector_handles_both_keys(tmp_path):
    from batch_job_functions import do_collect_uniprot_residues
    status, detail = do_collect_uniprot_residues(make_args(tmp_path), "A0A094YZ24_ST_100_1", AF2_A)
    assert status == "Success"
    r = detail[4][0]
    assert r["key_kind"] == "afc" and r["site_uid"] == "A0A094YZ24:ST_100:1" and r["num_binding_residues"] == 26
    assert r["prot_chains"] == "A" and r["UNPs"] == "A0A094YZ24"
    status, detail = do_collect_uniprot_residues(make_args(tmp_path), "1a7s-A-EOH-429", V2C_A)
    assert status == "Success" and detail[4][0]["pdb"] == "1a7s" and detail[4][0]["lig_id"] == "EOH"
    status, _ = do_collect_uniprot_residues(make_args(tmp_path), "bad key", V2C_A)
    assert status == "Error"


def test_case_suffixed_jobs_classify_like_the_plain_key(tmp_path):
    """The develop fixture under arm 1's two disambiguated spellings: the same numbers as the plain key,
    the full key kept as the job name, the suffix out of the key fields."""
    from batch_job_functions import do_collect_uniprot_residues
    status, ref = classify(make_args(tmp_path), DEV_AMINUS)
    assert status == "Success", ref
    (tmp_path / "arch").mkdir()
    for name, chain in (("8i5b-A-Y01-2005.u", "A"), ("8i5b-a-Y01-2005.l", "a")):
        p = tmp_path / "arch" / f"{name}.tgz"
        shutil.copy(DEV_AMINUS, p)
        status, row = classify(make_args(tmp_path), str(p))
        assert status == "Success", row
        assert row["job_name"] == name
        assert (row["n_apo"], row["n_holo"], row["pocket_avg_P(holo)"]) == (ref["n_apo"], ref["n_holo"], ref["pocket_avg_P(holo)"])
        status, detail = do_collect_uniprot_residues(make_args(tmp_path), name, str(p))
        assert status == "Success"
        r = detail[4][0]
        assert (r["job_name"], r["pdb"], r["lig_chain"], r["lig_id"], r["lig_index"]) == (name, "8i5b", chain, "Y01", "2005")


def test_underscore_keyed_jobs_classify_like_the_hyphen_key(tmp_path):
    """kaleidoscope's export names jobs <pdb>_<chain>_<lig>_<resi>. The develop fixture under that
    spelling (plain and case-suffixed) gives the hyphen key's numbers, keeps the key as the job name in
    the summary row, and fills the key fields of the residue collector."""
    from batch_job_functions import do_collect_uniprot_residues
    status, ref = classify(make_args(tmp_path), DEV_AMINUS)
    assert status == "Success", ref
    (tmp_path / "arch").mkdir()
    for name, chain in (("8i5b_A_Y01_2005", "A"), ("8i5b_a_Y01_2005.l", "a")):
        p = tmp_path / "arch" / f"{name}.tgz"
        shutil.copy(DEV_AMINUS, p)
        status, row = classify(make_args(tmp_path, save_statistics=1), str(p))
        assert status == "Success", row
        assert row["job_name"] == name
        closest = tmp_path / "mirror" / "arch" / name / "statistics" / "closest_pocket.json"
        assert json.load(open(closest, encoding="utf-8"))
        assert (row["n_apo"], row["n_holo"], row["pocket_avg_P(holo)"], row["pocket_overall_classification"]) == \
            (ref["n_apo"], ref["n_holo"], ref["pocket_avg_P(holo)"], ref["pocket_overall_classification"])
        status, detail = do_collect_uniprot_residues(make_args(tmp_path), name, str(p))
        assert status == "Success"
        r = detail[4][0]
        assert (r["job_name"], r["key_kind"], r["pdb"], r["lig_chain"], r["lig_id"], r["lig_index"]) == \
            (name, "holo", "8i5b", chain, "Y01", "2005")


def test_provenance_record_lists_every_rule():
    df, sp = rt.load_global_results(AF2_A), rt.load_script_parameters(AF2_A)
    row = rt.provenance_columns(rt.PROFILES["afc-2.4"], sp, df, None, rt.parse_job_key("A0A094YZ24_ST_100_1"))
    for k in ("run_type", "ahoj_version", "ahoj_schema", "ahoj_volume_mode", "ahoj_holo_overlap_mode",
              "rule_add_query_row", "rule_relabel_mode", "rule_ion_rule", "rule_gate", "rule_volume_column",
              "delta_anchor", "rog_anchor", "mapped_denominator", "query_mode", "query_anchor", "query_model", "site_uid"):
        assert k in row
    assert row["ahoj_pocket_alignment_method"] == 2 and row["ahoj_model_version"] == 6
    assert rt.PROFILES["holo-2.1"].as_record()["protect_labels"] == "U"


# ------------------------------------------------------------------ label vocabulary (AHOJ-020)
def _labels_df(labels):
    """A Q row, an M row and one candidate per label."""
    n = len(labels)
    return pd.DataFrame({"role": ["Q", "M"] + ["C"] * n, "apoholo_assignment": ["H", np.nan] + list(labels),
                         "query_ligs": ["ATP", None] + [None] * n, "nonquery_ligs": [None] * (n + 2)})


def test_a_minus_is_apo_everywhere_the_sets_are_pooled():
    from batch_job_functions import has_minimum_apo_holo
    from parsing_functions_DBv2 import split_dataframe
    import stats_core
    assert "A-" in rt.APO_LABELS and "A-" not in rt.HOLO_LABELS
    assert stats_core.APO_LABELS is rt.APO_LABELS and stats_core.HOLO_LABELS is rt.HOLO_LABELS
    df = _labels_df(["A", "A-", "A*", "A**", "H", "H*", "U"])
    c = df[df.role == "C"]
    assert has_minimum_apo_holo(c, min_apo=4, min_holo=2)[:4] == (True, 4, 2, 6)
    apo, holo, dis = split_dataframe(df)
    assert sorted(apo.apoholo_assignment) == ["A", "A*", "A**", "A-"] and len(holo) == 2 and len(dis) == 1
    assert rt.unknown_candidate_labels(df) == {}             # Q and M rows are not audited


def test_relabels_keep_a_minus_distinguishable():
    from parsing_functions_DBv2 import relabel_apoholo_by_ions
    df = _labels_df(["A-", "H", "A"]).assign(query_ligs=["ATP", None, "ATP", "ATP", None])
    dem, changed = relabel_apoholo_by_ions(df.copy(), builtin="total_le1", verbose=False, demote_only=True,
                                           protect_labels=set(rt.PROTECTED_LABELS_24))
    assert dem.apoholo_assignment.iloc[2] == "A-" and not changed.iloc[2]     # holo-2.4: the gate label stands
    # v2c_ions (allow_undo_gate) rebuilds from query_ligs and undoes the gate, as for any other gate label
    full, _ = relabel_apoholo_by_ions(df.copy(), builtin="total_le1", verbose=False)
    assert full.apoholo_assignment.iloc[2] == "H"


def test_unknown_candidate_labels_are_counted_and_loud(capsys):
    from batch_job_functions import audit_candidate_labels, has_minimum_apo_holo
    df = _labels_df(["A", "Hweak", "Hweak", "X", np.nan, "U"])
    assert rt.unknown_candidate_labels(df) == {"<NA>": 1, "Hweak": 2, "X": 1}
    assert rt.format_label_counts({}) is None
    out = audit_candidate_labels(df, "job-1")
    assert out == {"n_unknown_label": 4, "unknown_labels": "<NA>:1;Hweak:2;X:1"}
    err = capsys.readouterr().out
    assert "[WARN] [job-1] 4 candidate row(s)" in err and "Hweak:2" in err and "(apo A A- A* A**, holo H H*, unpooled U)" in err
    assert has_minimum_apo_holo(df[df.role == "C"], 0, 0)[1:] == (1, 0, 1)   # still in neither set
    assert audit_candidate_labels(_labels_df(["A-", "H"]), "job-2") == {"n_unknown_label": 0, "unknown_labels": None}
    assert capsys.readouterr().out == ""


def test_develop_a_minus_archive_matches_the_240_pooling(tmp_path):
    status, row = classify(make_args(tmp_path, save_filtered_df=1), DEV_AMINUS)
    assert status == "Success", row
    assert row["run_type"] == "holo-2.4" and row["ahoj_version"] == "2.5.1" and row["ahoj_holo_overlap_mode"] == "gate"
    assert (row["n_apo"], row["n_holo"]) == (12, 17) and abs(row["pocket_avg_P(holo)"] - 0.716) < 1e-9
    assert row["n_apo_query_demoted"] == 1 and row["n_unknown_label"] == 0 and row["unknown_labels"] is None
    # the label survives into global_results_plots.csv
    plots = [os.path.join(d, f) for d, _, fs in os.walk(tmp_path / "mirror") for f in fs if f == "global_results_plots.csv"]
    assert len(plots) == 1
    assert (pd.read_csv(plots[0]).apoholo_assignment == "A-").sum() == 1

    from batch_job_functions import do_collect_apoholo_metric_stats
    status, detail = do_collect_apoholo_metric_stats(make_args(tmp_path), "8i5b-A-Y01-2005", DEV_AMINUS)
    assert status == "Success" and (detail[4]["n_apo"], detail[4]["n_apo_query_demoted"]) == (12, 1)
    assert detail[4]["n_unknown_label"] == 0


def test_unknown_label_in_an_archive_is_reported_not_pooled(tmp_path, capsys):
    # the develop archive with its A- row respelled as a label parse-ahoj has never seen
    df = rt.load_global_results(DEV_AMINUS)
    df.loc[df.apoholo_assignment == "A-", "apoholo_assignment"] = "A~"
    p = tmp_path / "arch" / "8i5b-A-Y01-2005.tgz"
    p.parent.mkdir()
    import io
    with tarfile.open(DEV_AMINUS, "r:gz") as src, tarfile.open(p, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else None
            if m.name.lstrip("./") == "global_results.csv":
                data = df.to_csv(index=False).encode("utf-8")
                m.size = len(data)
            dst.addfile(m, io.BytesIO(data) if data is not None else None)
    status, row = classify(make_args(tmp_path), str(p))
    assert status == "Success", row
    assert (row["n_apo"], row["n_holo"]) == (11, 17)
    assert row["n_unknown_label"] == 1 and row["unknown_labels"] == "A~:1" and row["n_apo_query_demoted"] == 0
    assert "[WARN] [8i5b-A-Y01-2005] 1 candidate row(s)" in capsys.readouterr().out


def test_label_totals_for_the_run_manifest():
    import batch_process_database as bpd
    q = pd.DataFrame({"job_name": ["a", "b", "c"], "n_apo_query_demoted": [1, 0, 4],
                      "n_unknown_label": [0, 3, 1], "unknown_labels": [None, "Hweak:2;X:1", "Hweak:1"]})
    assert bpd.label_totals(q) == {"n_apo_query_demoted": 5, "n_jobs_with_apo_query_demoted": 2,
                                   "unknown_labels": {"Hweak": 3, "X": 1}, "n_unknown_label": 4,
                                   "n_jobs_with_unknown_label": 2}
    assert bpd.label_totals(pd.DataFrame({"job_name": ["a"]})) is None and bpd.label_totals(None) is None
