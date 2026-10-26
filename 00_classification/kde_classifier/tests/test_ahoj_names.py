# -*- coding: utf-8 -*-
"""AHoJ 2.6.0 output names (ahoj_names, run_types.load_global_results / load_ligands_json): a job written
with the 2.6.0 names reads, classifies and writes exactly as the same job written with the names before.

Fixtures (tests/fixtures):
  dev0e280af/8i5b-A-Y01-2005.tgz          AHoJ2 0e280af (branch claude/output-column-names, PR #146; the
  dev0e280af/A0A094YZ24_ST_100_1.tgz      2.6.0 names, output_schema 2, ligands.json schema 3), offline
                                          on D:\\ahoj_mirror. script_version is 2.5.1: develop keeps the
                                          last release's until 2.6.0 is cut. Model rows name their model
                                          (AF-Q15858-F1-model_v6), as since AHoJ2 PR #143.
  dev0e280af_legacy/<same keys>.tgz       the same queries on the same commit with --legacy_names 1
                                          (output_schema 1, ligands.json schema 2): the names before 2.6.0.
The other fixtures (v2c, af2, hq1, dev7fd6825) are rewritten here under the 2.6.0 names
(_renamed_archive) and must classify as they do under their own.
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import ahoj_names as an  # noqa: E402
import run_types as rt  # noqa: E402
from run_types import RunTypeError  # noqa: E402
from test_run_types import (FIX, SITES, V2C_A, V2C_ZN, AF2_A, AF2_ION, HQ1_A, HQ1_ION,  # noqa: E402
                            DEV_AMINUS, make_args)

NEW_8I5B = os.path.join(FIX, "dev0e280af", "8i5b-A-Y01-2005.tgz")
OLD_8I5B = os.path.join(FIX, "dev0e280af_legacy", "8i5b-A-Y01-2005.tgz")
NEW_AFC = os.path.join(FIX, "dev0e280af", "A0A094YZ24_ST_100_1.tgz")
OLD_AFC = os.path.join(FIX, "dev0e280af_legacy", "A0A094YZ24_ST_100_1.tgz")
PAIRS = [(NEW_8I5B, OLD_8I5B), (NEW_AFC, OLD_AFC)]

# Every output a classification run writes per entry, numbers only (the plots are a separate test)
SAVE_ALL = dict(save_filtered_df=1, save_statistics=1, save_per_method_json=1, apo_holo_sums=1)
PLOTS = dict(general_violins=1, kde_curves=1, kde_violins=1, conf_bars=1, dp_curves=1)
PAIR_IDS = ["8i5b-holo", "A0A094YZ24-afc"]


@pytest.fixture(autouse=True)
def _clear_sites():
    rt.set_sites({})
    yield
    rt.set_sites({})


def _overrides(path):
    """The arguments each fixture classifies with in test_run_types.py."""
    if os.path.basename(path).startswith(("A0A094YZ24", "Q5H8A6")) and os.sep + "hq1" + os.sep not in path:
        rt.set_sites(rt.load_sites_table(SITES))
        return {"sites_tsv": SITES}
    if os.sep + "hq1" + os.sep in path:
        return {"key_format": "any"}
    return {}


def _run(tmp_path, path, **over):
    """do_kde_plots_and_classification on one archive: (status, every returned table, every file written
    under the entry's mirrored directory as {relative path: bytes})."""
    from batch_job_functions import do_kde_plots_and_classification
    mirror = tmp_path / "mirror"
    args = make_args(tmp_path, **{**_overrides(path), **over})
    job = os.path.basename(path)[:-4]
    status, detail = do_kde_plots_and_classification(args, job, path)
    files = {}
    for d, _, fs in os.walk(mirror):
        for f in fs:
            full = os.path.join(d, f)
            rel = os.path.relpath(full, mirror).split(os.sep, 1)[1]       # drop the archive's parent dir
            with open(full, "rb") as fh:
                files[rel] = fh.read()
    return status, detail, files


def _without_paths(df):
    return df.drop(columns=[c for c in ("query_path",) if c in df.columns])


def _assert_same_run(a, b):
    (sa, da, fa), (sb, db, fb) = a, b
    assert sa == sb == "Success", (da[1], db[1])
    for i in (2, 3, 4, 5):            # apo summary, holo summary, query summary row, metric summary row
        pd.testing.assert_frame_equal(_without_paths(da[i]), _without_paths(db[i]))
    assert sorted(fa) == sorted(fb)
    for rel in fa:
        assert fa[rel] == fb[rel], rel


# ------------------------------------------------------------------ the maps
def test_the_two_name_sets_are_disjoint_and_one_to_one():
    new, old = set(an.RESULT_PREVIOUS), set(an.RESULT_PREVIOUS.values())
    assert not new & old and len(old) == len(new)
    ent_new, ent_old = set(an.LIGANDS_ENTITY_PREVIOUS), set(an.LIGANDS_ENTITY_PREVIOUS.values())
    assert not ent_new & ent_old and len(ent_old) == len(ent_new)
    # the names parse-ahoj reads by name (configs, filters, summaries) are all old names or kept names
    for name in ("#resis3", "#resis5", "RoG", "bfac_avg", "bfac_t", "pocket_rms", "pocket_len", "%mapped_sqr",
                 "%mapped_obs", "apoholo_assignment", "holo_assignment_detail", "UNPs", "chains3", "query_ligs",
                 "nonquery_ligs", "tm_score_i", "POI_type", "asa", "sasa_avg"):
        assert name in old, name
    for kept in ("structure", "pocket", "role", "sasa", "volume", "resolution", "exp_method", "tm_score", "rmsd",
                 "pocket_dist", "query_mode", "query_anchor", "query_model"):
        assert kept not in new and kept not in old


def test_vendored_maps_equal_ahoj2_output_names():
    """Set AHOJ2_DIR to an AHoJ2 checkout with output_names.py (2.6.0) to check the copies in ahoj_names."""
    d = os.environ.get("AHOJ2_DIR")
    if not d or not os.path.isfile(os.path.join(d, "output_names.py")):
        pytest.skip("AHOJ2_DIR does not point to an AHoJ2 checkout with output_names.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ahoj2_output_names", os.path.join(d, "output_names.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RESULT_PREVIOUS == an.RESULT_PREVIOUS
    assert mod.LIGANDS_ENTITY_PREVIOUS == an.LIGANDS_ENTITY_PREVIOUS
    assert tuple(mod.LIGANDS_SCHEMA_VERSIONS) == an.LIGANDS_SCHEMA_VERSIONS
    assert (mod.OUTPUT_SCHEMA, mod.LEGACY_OUTPUT_SCHEMA) == (an.OUTPUT_SCHEMA_NEW, an.OUTPUT_SCHEMA_OLD)


# ------------------------------------------------------------------ which names a job has
def test_output_schema_is_declared_and_checked_against_the_header():
    new_cols, old_cols = ["structure", "role", "label", "n_observed", "rog"], ["structure", "role", "apoholo_assignment", "#resis3", "RoG"]
    assert an.output_schema_of({"output_schema": 2}, new_cols) == 2
    assert an.output_schema_of({"output_schema": 1}, old_cols) == 1
    assert an.output_schema_of({"output_schema": "2"}, new_cols) == 2
    # absent: AHoJ before 2.6.0, no script_parameters.json, or a writer that does not record it
    assert an.output_schema_of({"script_version": "2.4.0"}, old_cols) == 1
    assert an.output_schema_of(None, new_cols) == 2
    assert an.output_schema_of({}, new_cols) == 2
    assert an.output_schema_of(None, ["structure", "role"]) == 1          # nothing renamed: either reads the same
    assert an.output_schema_of({"output_schema": 2}, ["structure", "role"]) == 2
    with pytest.raises(an.OutputNamesError, match="output_schema 1 but"):
        an.output_schema_of({"output_schema": 1}, new_cols)
    with pytest.raises(an.OutputNamesError, match="output_schema 2 but"):
        an.output_schema_of({"output_schema": 2}, old_cols)
    with pytest.raises(an.OutputNamesError, match="mixes"):
        an.output_schema_of(None, ["role", "label", "#resis3"])
    with pytest.raises(an.OutputNamesError, match="not known"):
        an.output_schema_of({"output_schema": 3}, new_cols)
    with pytest.raises(an.OutputNamesError, match="unreadable"):
        an.output_schema_of({"output_schema": "two"}, new_cols)


@pytest.mark.parametrize("new,old", PAIRS, ids=PAIR_IDS)
def test_a_260_job_reads_as_its_legacy_twin(new, old):
    sp_new, sp_old = rt.load_script_parameters(new), rt.load_script_parameters(old)
    assert (sp_new["output_schema"], sp_old["output_schema"]) == (2, 1)
    raw = pd.read_csv(io.BytesIO(rt.read_member(new, "global_results.csv")))
    assert {"label", "n_observed", "rog", "accessions"} <= set(raw.columns) and "apoholo_assignment" not in raw.columns
    a, b = rt.load_global_results(new), rt.load_global_results(old)
    assert (a.attrs["ahoj_output_schema"], b.attrs["ahoj_output_schema"]) == (2, 1)
    pd.testing.assert_frame_equal(a, b)                                   # every name and every value
    lj_new = json.loads(rt.read_member(new, "ligands.json"))
    assert lj_new["schema_version"] == 3
    assert rt.load_ligands_json(new) == rt.load_ligands_json(old) == json.loads(rt.read_member(old, "ligands.json"))
    assert rt.detect_run_type(sp_new, a) == rt.detect_run_type(sp_old, b)
    assert rt.schema_of(a) == rt.schema_of(b) == "v2.4"


def test_ligands_json_without_schema_version_is_told_by_its_pocket_key():
    doc = json.loads(rt.read_member(NEW_AFC, "ligands.json"))
    old = json.loads(rt.read_member(OLD_AFC, "ligands.json"))
    del doc["schema_version"]
    assert an.ligands_schema_of(doc) == 3
    renamed = an.ligands_to_internal(doc)
    assert renamed == old
    assert an.ligands_to_internal(old) is old                            # schema 2 is returned as it is
    v21 = json.loads(rt.read_member(V2C_A, "ligands.json"))
    assert isinstance(v21, list) and an.ligands_to_internal(v21) is v21    # AHoJ 2.1's bare list
    assert an.ligands_schema_of({"pockets": []}) is None


def test_entity_relabel_reads_the_260_entity_keys():
    df_new, df_old = rt.load_global_results(NEW_AFC), rt.load_global_results(OLD_AFC)
    args = (rt.GATE_DEFAULTS, rt.ion_set(), rt.additive_set(), "site", "v2c", None)
    a, ca = rt.relabel_by_entities(df_new, rt.load_ligands_json(NEW_AFC), *args)
    b, cb = rt.relabel_by_entities(df_old, rt.load_ligands_json(OLD_AFC), *args)
    pd.testing.assert_frame_equal(a, b)
    assert ca.equals(cb) and (a.loc[a.role == "C", "apoholo_assignment"] == "H*").sum() == 4
    # without the rename, the schema-3 entity keys would read as 0 coverage and fail every gate
    raw = json.loads(rt.read_member(NEW_AFC, "ligands.json"))
    assert rt.entities_by_pocket(raw) and all(k[1] == "None" for k in rt.entities_by_pocket(raw))


# ------------------------------------------------------------------ end to end, real 2.6.0 jobs
@pytest.mark.parametrize("new,old", PAIRS, ids=PAIR_IDS)
def test_a_260_job_classifies_and_writes_as_its_legacy_twin(tmp_path, new, old):
    a = _run(tmp_path / "new", new, **SAVE_ALL)
    b = _run(tmp_path / "old", old, **SAVE_ALL)
    _assert_same_run(a, b)
    row = a[1][4].iloc[0]
    assert row["n_model_rows"] == 1                                       # the model is found by its UNPs
    assert any(rel.endswith("global_results_plots.csv") for rel in a[2])
    plots = pd.read_csv(io.BytesIO(next(v for k, v in a[2].items() if k.endswith("global_results_plots.csv"))))
    assert {"apoholo_assignment", "#resis3", "RoG", "UNPs"} <= set(plots.columns)  # parse-ahoj's names, unchanged
    assert not a[1][2].empty and not a[1][3].empty                        # create_summary_df ran on both sets


def test_8i5b_on_260_matches_the_develop_archive(tmp_path):
    from test_run_types import classify
    status, row = classify(make_args(tmp_path), NEW_8I5B)
    assert status == "Success", row
    assert row["run_type"] == "holo-2.4" and (row["n_apo"], row["n_holo"]) == (12, 17)
    assert row["n_apo_query_demoted"] == 1 and row["n_unknown_label"] == 0


def test_afc_260_job_reproduces_the_pilot_site(tmp_path):
    from test_run_types import classify
    rt.set_sites(rt.load_sites_table(SITES))
    status, row = classify(make_args(tmp_path, sites_tsv=SITES), NEW_AFC)
    assert status == "Success", row
    assert row["run_type"] == "afc-2.4" and (row["n_apo"], row["n_holo"]) == (2, 4)
    assert row["query_model"] == "AF-A0A094YZ24-F1-model_v6"


@pytest.mark.parametrize("new,old", PAIRS, ids=PAIR_IDS)
def test_plots_are_the_same_files(tmp_path, new, old):
    a = _run(tmp_path / "new", new, **PLOTS)
    b = _run(tmp_path / "old", old, **PLOTS)
    assert a[0] == b[0] == "Success"
    names = sorted(a[2])
    assert names == sorted(b[2]) and any(n.endswith((".svg", ".png")) for n in names)


@pytest.mark.parametrize("new,old", PAIRS, ids=PAIR_IDS)
def test_lightweight_collectors_read_the_260_names(tmp_path, new, old):
    from batch_job_functions import do_collect_model_pLDDT, do_collect_apoholo_metric_stats
    for fn in (do_collect_model_pLDDT, do_collect_apoholo_metric_stats):
        ra = fn(make_args(tmp_path, **_overrides(new)), os.path.basename(new)[:-4], new)
        rb = fn(make_args(tmp_path, **_overrides(old)), os.path.basename(old)[:-4], old)
        assert ra[0] == rb[0] == "Success", (fn.__name__, ra[1], rb[1])
        da = {k: v for k, v in ra[1][4].items() if k != "query_path"}
        db = {k: v for k, v in rb[1][4].items() if k != "query_path"}
        assert da.keys() == db.keys()
        for k in da:
            assert (da[k] == db[k]) or (pd.isna(da[k]) and pd.isna(db[k])), (fn.__name__, k, da[k], db[k])
    m = do_collect_model_pLDDT(make_args(tmp_path), "x", new)[1][4]
    assert m["status"] == "ok" and m["n_model_rows"] == 1 and not np.isnan(m["model_pocket_pLDDT_avg"])
    assert not np.isnan(m["model_pocket_RoG"]) and not np.isnan(m["model_pocket_volume"])


# ------------------------------------------------------------------ the fallback and the loud failures
def _extract(path, dst):
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dst, filter="data")
    return str(dst)


def _edit_json(path, fn):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc = fn(doc)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


def test_header_decides_without_output_schema(tmp_path):
    ref = rt.load_global_results(OLD_8I5B)
    d = _extract(NEW_8I5B, tmp_path / "noschema" / "8i5b-A-Y01-2005")
    _edit_json(os.path.join(d, "script_parameters.json"), lambda sp: {k: v for k, v in sp.items() if k != "output_schema"})
    df = rt.load_global_results(d)
    assert df.attrs["ahoj_output_schema"] == 2
    pd.testing.assert_frame_equal(df, ref)
    os.remove(os.path.join(d, "script_parameters.json"))                   # nothing but the header
    pd.testing.assert_frame_equal(rt.load_global_results(d), ref)
    # and a run on the directory without script_parameters.json: the layout decides the run type
    assert rt.detect_run_type(None, rt.load_global_results(d)) == "holo-2.4"


def test_contradicting_or_mixed_names_are_an_error(tmp_path):
    from batch_job_functions import do_collect_model_pLDDT
    d = _extract(NEW_8I5B, tmp_path / "liar" / "8i5b-A-Y01-2005")
    _edit_json(os.path.join(d, "script_parameters.json"), lambda sp: {**sp, "output_schema": 1})
    with pytest.raises(RunTypeError, match="output_schema 1 but"):
        rt.load_global_results(d)
    from batch_job_functions import do_kde_plots_and_classification
    status, detail = do_kde_plots_and_classification(make_args(tmp_path), "8i5b-A-Y01-2005", d)
    assert status == "Error" and "output_schema 1 but" in detail[1]
    assert do_collect_model_pLDDT(make_args(tmp_path), "8i5b-A-Y01-2005", d)[0] == "Error"

    m = _extract(NEW_8I5B, tmp_path / "mixed" / "8i5b-A-Y01-2005")
    gr = os.path.join(m, "global_results.csv")
    df = pd.read_csv(gr).rename(columns={"rog": "RoG"})
    df.to_csv(gr, index=False)
    _edit_json(os.path.join(m, "script_parameters.json"), lambda sp: {k: v for k, v in sp.items() if k != "output_schema"})
    with pytest.raises(RunTypeError, match="mixes"):
        rt.load_global_results(m)


def test_ahoj_26_takes_the_24_profiles(tmp_path):
    df = rt.load_global_results(NEW_8I5B)
    for v in ("2.6.0", "2.6.1+develop.abc1234"):
        assert rt.detect_run_type({"script_version": v}, df) == "holo-2.4"
        assert rt.detect_run_type({"script_version": v}, rt.load_global_results(NEW_AFC)) == "afc-2.4"
    # the released 2.6.0 job, with its script_version as the release will write it
    d = _extract(NEW_8I5B, tmp_path / "rel" / "8i5b-A-Y01-2005")
    _edit_json(os.path.join(d, "script_parameters.json"), lambda sp: {**sp, "script_version": "2.6.0"})
    from batch_job_functions import do_kde_plots_and_classification
    status, detail = do_kde_plots_and_classification(make_args(tmp_path), "8i5b-A-Y01-2005", d)
    assert status == "Success", detail[1]
    row = detail[4].iloc[0]
    assert row["ahoj_version"] == "2.6.0" and row["run_type"] == "holo-2.4" and (row["n_apo"], row["n_holo"]) == (12, 17)


# ------------------------------------------------------------------ model rows since AHoJ2 PR #143
def test_model_rows_named_by_model_id():
    df = pd.DataFrame({
        "role": ["Q", "M", "M", "M", "C"],
        "structure": ["8i5b", "AF-Q15858-F1-model_v6", "AF-P99999-F1-model_v6", "AF_P88888", "2xyz"],
        "UNPs": ["Q15858", "Q15858", "P99999", "P88888", "Q15858"]})
    assert rt.select_model_rows(df).structure.tolist() == ["AF-Q15858-F1-model_v6"]
    assert len(rt.select_model_rows(df, "first_M")) == 3
    # a heteromeric pocket: the model row names each model and each accession, space-joined
    het = pd.DataFrame({"role": ["Q", "M"], "structure": ["1abc", "AF-P1-F1-model_v6 AF-P2-F1-model_v6"],
                        "UNPs": ["P1 P2", "P1 P2"]})
    assert len(rt.select_model_rows(het)) == 1
    # a model row whose UNPs is blank, or a query without UNPs, is not the query's model
    blank = pd.DataFrame({"role": ["Q", "M"], "structure": ["1abc", "AF-P1-F1-model_v6"], "UNPs": [np.nan, "P1"]})
    assert rt.select_model_rows(blank).empty
    # model as query (afc): the query row and the model row are the same model
    afc = rt.load_global_results(NEW_AFC)
    assert rt.select_model_rows(afc).structure.tolist() == ["AF-A0A094YZ24-F1-model_v6"]
    # archives before #143 select exactly as before
    for p in (V2C_A, AF2_A, HQ1_A, DEV_AMINUS):
        g = rt.load_global_results(p)
        assert rt.select_model_rows(g).structure.tolist() == ["AF_model"], p


# ------------------------------------------------------------------ every older fixture under the 2.6.0 names
_RESULT_NEW = {old: new for new, old in an.RESULT_PREVIOUS.items()}
_ENTITY_NEW = {old: new for new, old in an.LIGANDS_ENTITY_PREVIOUS.items()}


def _ligands_to_260(doc):
    """ligands.json schema 2 -> 3, as AHoJ 2.6.0 writes it (the inverse of ahoj_names.ligands_to_internal)."""
    if not isinstance(doc, dict) or doc.get("schema_version") != 2:
        return doc
    ren = lambda r, m: {m.get(k, k): v for k, v in r.items()}  # noqa: E731
    doc = dict(doc, schema_version=3)
    pockets = []
    for pk in doc.get("pockets", []):
        pk = dict(pk)
        if "query" in pk:
            q = ren(pk["query"], {"pocket_index": "pocket"})
            for key in ("entities", "detected_entities"):
                if key in q:
                    q[key] = [ren(e, _ENTITY_NEW) for e in q[key]]
            if "entity_group" in q:
                q["entity_group"] = ren(q["entity_group"], _ENTITY_NEW)
            pk["query"] = q
        if "candidates" in pk:
            cs = []
            for c in pk["candidates"]:
                c = ren(c, {"pocket_index": "pocket"})
                if "detected_entities" in c:
                    c["detected_entities"] = [ren(e, _ENTITY_NEW) for e in c["detected_entities"]]
                cs.append(c)
            pk["candidates"] = cs
        pockets.append(pk)
    doc["pockets"] = pockets
    return doc


def _renamed_archive(src, dst, output_schema=True):
    """A copy of an archive under the 2.6.0 names: global_results.csv header, ligands.json schema 3,
    script_parameters.json output_schema 2 (left out with output_schema=False: the header decides)."""
    def edit(name, data):
        if name == "global_results.csv":
            head, _, rest = data.partition(b"\n")
            cr = head.endswith(b"\r")
            cols = head.rstrip(b"\r").decode("utf-8").split(",")
            head = ",".join(_RESULT_NEW.get(c, c) for c in cols).encode("utf-8") + (b"\r" if cr else b"")
            return head + b"\n" + rest
        if name == "ligands.json":
            return json.dumps(_ligands_to_260(json.loads(data.decode("utf-8"))), indent=1).encode("utf-8")
        if name == "script_parameters.json" and output_schema:
            return json.dumps({**json.loads(data.decode("utf-8")), "output_schema": 2}, indent=4).encode("utf-8")
        return data
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tarfile.open(src, "r:gz") as s, tarfile.open(dst, "w:gz") as d:
        for m in s.getmembers():
            if not m.isfile():
                d.addfile(m)
                continue
            data = edit(m.name[2:] if m.name.startswith("./") else m.name, s.extractfile(m).read())
            m.size = len(data)
            d.addfile(m, io.BytesIO(data))
    return dst


OLDER = [V2C_A, V2C_ZN, AF2_A, AF2_ION, HQ1_A, HQ1_ION, DEV_AMINUS]


@pytest.mark.parametrize("src", OLDER, ids=lambda p: os.path.relpath(p, FIX).replace(os.sep, "/"))
@pytest.mark.parametrize("output_schema", [True, False], ids=["output_schema", "header_only"])
def test_older_fixtures_under_the_260_names_classify_the_same(tmp_path, src, output_schema):
    sub = os.path.basename(os.path.dirname(src))
    twin = _renamed_archive(src, str(tmp_path / "twin" / sub / os.path.basename(src)), output_schema)
    raw = pd.read_csv(io.BytesIO(rt.read_member(twin, "global_results.csv")))
    assert "label" in raw.columns and "apoholo_assignment" not in raw.columns
    assert rt.load_global_results(twin).attrs["ahoj_output_schema"] == 2
    pd.testing.assert_frame_equal(rt.load_global_results(twin), rt.load_global_results(src))
    assert rt.load_ligands_json(twin) == rt.load_ligands_json(src)
    _assert_same_run(_run(tmp_path / "twin_run", twin, **SAVE_ALL), _run(tmp_path / "src_run", src, **SAVE_ALL))
