"""
End-to-end tests on a small synthetic input (3 accessions, one interface row).

Expected clustering (average linkage on Jaccard distance):

P00001  j1 {10,11,12,13}  j2 {10,11,12,13}  j3 {10,11,12,14}  j4 {50,51,52}  j6 {50,51,52} (interface part)
        ST_100 : {j1,j2} {j3} {j4,j6}
        ST_070 : same as ST_100  (j1/j3 Jaccard = 3/5 = 0.6 < 0.7)
        ST_050 : {j1,j2,j3} {j4,j6}
        ST_030 : same as ST_050
        LJ     : linkage distances 0, 0, 0.4, 1.0 -> largest jump after 0.4 -> ST_LJ060, cut = ST_050
P00002  j5 {5,6}                              single entry -> ubs 1 at every threshold, no LJ, no tree
P00003  j6 {7,8} (interface part)  j7 {7,9}   Jaccard 1/3
        ST_100 / ST_070 / ST_050 : two sites;  ST_030 : one site;  no LJ (only one linkage step)
"""

import ast
import glob
import inspect
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import collect_ubs  # noqa: E402
import run_ubs_unification as rub  # noqa: E402
from ubs_utils import split_interface_entries  # noqa: E402

PY = sys.executable
SCRIPT = os.path.join(REPO_ROOT, "run_ubs_unification.py")
COLLECTOR = os.path.join(REPO_ROOT, "collect_ubs.py")

ROWS = [
    # job_name, UNPs, uniprot_residues, prot_chains_num, UNPs_num, pdb, ligands
    ("j1", "P00001", ["P00001_10", "P00001_11", "P00001_12", "P00001_13"], 1, 1, "1abc", "ATP"),
    ("j2", "P00001", ["P00001_10", "P00001_11", "P00001_12", "P00001_13"], 1, 1, "1abd", "ADP"),
    ("j3", "P00001", ["P00001_10", "P00001_11", "P00001_12", "P00001_14"], 1, 1, "1abe", "ATP"),
    ("j4", "P00001", ["P00001_50", "P00001_51", "P00001_52"], 1, 1, "1abf", "MG"),
    ("j5", "P00002", ["P00002_5", "P00002_6"], 1, 1, "2xyz", "ZN"),
    ("j6", "P00001 P00003", ["P00001_50", "P00001_51", "P00001_52", "P00003_7", "P00003_8"], 2, 2, "3int", "HEM"),
    ("j7", "P00003", ["P00003_7", "P00003_9"], 1, 1, "3mon", "HEM"),
]
COLUMNS = ["job_name", "UNPs", "uniprot_residues", "prot_chains_num", "UNPs_num", "pdb", "ligands"]
ACCESSIONS = ["P00001", "P00002", "P00003"]
FIXED_LABELS = ["ST_030", "ST_050", "ST_070", "ST_100"]

EXPECTED_ST100 = {
    "P00001": [{"j1", "j2"}, {"j3"}, {"j4", "j6"}],
    "P00002": [{"j5"}],
    "P00003": [{"j6"}, {"j7"}],
}
EXPECTED_ST050 = {
    "P00001": [{"j1", "j2", "j3"}, {"j4", "j6"}],
    "P00002": [{"j5"}],
    "P00003": [{"j6"}, {"j7"}],
}
EXPECTED_ST030 = {
    "P00001": [{"j1", "j2", "j3"}, {"j4", "j6"}],
    "P00002": [{"j5"}],
    "P00003": [{"j6", "j7"}],
}


def write_input_csv(path):
    df = pd.DataFrame(ROWS, columns=COLUMNS)
    df["uniprot_residues"] = df["uniprot_residues"].apply(repr)  # Python list literal, as in ubs_input.csv
    df.to_csv(path, index=False)
    return path


def run_cli(script, *args):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    proc = subprocess.run([PY, script, *map(str, args)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{os.path.basename(script)} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc


def read_st(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def as_groups(iterable_of_member_sets):
    """Order-independent view of a partition: {frozenset(members), ...}."""
    return {frozenset(g) for g in iterable_of_member_sets}


def groups_from_table(path):
    """Partition of the entries in an ST table (ubs_id numbering is scipy's, so order-free)."""
    df = read_st(path)
    assert sorted(df["ubs_id"].astype(int)) == list(range(1, len(df) + 1))   # consecutive ids from 1
    return as_groups(ast.literal_eval(x) for x in df["entry_ids"])


def st_tables(root):
    """{relative path: text} of every ST_*.tsv under root."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "ST_*.tsv"))):
        with open(path, encoding="utf-8") as fh:
            out[os.path.relpath(path, root)] = fh.read()
    return out


@pytest.fixture(scope="session")
def input_csv(tmp_path_factory):
    return write_input_csv(str(tmp_path_factory.mktemp("input") / "ubs_input.csv"))


@pytest.fixture(scope="session")
def default_run(tmp_path_factory, input_csv):
    out = str(tmp_path_factory.mktemp("default"))
    run_cli(SCRIPT, "--input", input_csv, "--out", out)
    return out


@pytest.fixture(scope="session")
def nofig_run(tmp_path_factory, input_csv):
    out = str(tmp_path_factory.mktemp("nofig"))
    run_cli(SCRIPT, "--input", input_csv, "--out", out, "--no-figures")
    return out


# --------------------------------------------------------------------------- CLI runs

def test_default_run_layout(default_run):
    assert sorted(d for d in os.listdir(default_run) if os.path.isdir(os.path.join(default_run, d))) == ACCESSIONS
    for acc in ACCESSIONS:
        d = os.path.join(default_run, acc)
        for label in FIXED_LABELS:
            assert os.path.isfile(os.path.join(d, f"{label}.tsv")), f"{acc}/{label}.tsv missing"
        assert os.path.isfile(os.path.join(d, "input_data.csv.gz"))          # default compression is gz
        assert os.path.isfile(os.path.join(d, "binding_positions.svg.gz"))   # histogram for every accession
        manifest = json.load(open(os.path.join(d, "ubs_manifest.json"), encoding="utf-8"))
        assert manifest["uniprot"] == acc
        assert manifest["archiveFormat"] == "gz"
        assert manifest["defaultThresholds"] == [0.3, 0.5, 0.7, 1.0]
        assert manifest["inputData"] == "input_data.csv.gz"
        assert manifest["figures"]["histogram"] == "binding_positions.svg.gz"
        assert {t["label"] for t in manifest["thresholds"] if t["type"] == "fixed"} == set(FIXED_LABELS)
    # Dendrogram + tree only where clustering happened (>1 entry)
    for acc in ("P00001", "P00003"):
        assert os.path.isfile(os.path.join(default_run, acc, "dendrogram.svg.gz"))
        assert os.path.isfile(os.path.join(default_run, acc, "tree.pkl"))
    assert not os.path.exists(os.path.join(default_run, "P00002", "tree.pkl"))
    # Large-jump table only for P00001 (>= 3 entries -> >= 2 linkage steps)
    assert os.path.isfile(os.path.join(default_run, "P00001", "ST_LJ060.tsv"))
    assert not glob.glob(os.path.join(default_run, "P00002", "ST_LJ*.tsv"))
    assert not glob.glob(os.path.join(default_run, "P00003", "ST_LJ*.tsv"))
    # Log written to the output root by default
    assert glob.glob(os.path.join(default_run, "ubs_unification_*.log"))


def test_no_figures_same_tables_as_default(default_run, nofig_run):
    default_tables = st_tables(default_run)
    nofig_tables = st_tables(nofig_run)
    assert default_tables.keys() == nofig_tables.keys()
    assert len(default_tables) == 3 * 4 + 1  # four fixed tables per accession + one LJ table
    for rel in default_tables:
        assert default_tables[rel] == nofig_tables[rel], f"{rel} differs between default and --no-figures"


def test_no_figures_writes_only_tables_input_and_manifest(nofig_run):
    for acc in ACCESSIONS:
        d = os.path.join(nofig_run, acc)
        names = set(os.listdir(d))
        assert not any(n.endswith((".svg", ".svg.gz", ".svg.zip", ".pkl")) for n in names), names
        assert "input_data.csv.gz" in names
        assert "ubs_manifest.json" in names
        others = {n for n in names if not n.startswith("ST_")}
        assert others == {"input_data.csv.gz", "ubs_manifest.json"}
        manifest = json.load(open(os.path.join(d, "ubs_manifest.json"), encoding="utf-8"))
        assert manifest["figures"] == {}
        assert manifest["inputData"] == "input_data.csv.gz"
        assert {t["file"] for t in manifest["thresholds"]} == {n for n in names if n.startswith("ST_")}


def test_expected_groupings(nofig_run):
    for acc, expected in EXPECTED_ST100.items():
        assert groups_from_table(os.path.join(nofig_run, acc, "ST_100.tsv")) == as_groups(expected)
        assert groups_from_table(os.path.join(nofig_run, acc, "ST_070.tsv")) == as_groups(expected)
    for acc, expected in EXPECTED_ST050.items():
        assert groups_from_table(os.path.join(nofig_run, acc, "ST_050.tsv")) == as_groups(expected)
    for acc, expected in EXPECTED_ST030.items():
        assert groups_from_table(os.path.join(nofig_run, acc, "ST_030.tsv")) == as_groups(expected)
    # Large jump (similarity 0.6) cuts P00001 like ST_050
    assert groups_from_table(os.path.join(nofig_run, "P00001", "ST_LJ060.tsv")) == as_groups(EXPECTED_ST050["P00001"])


def test_st100_groups_identical_sets_only(nofig_run, input_csv):
    df = pd.read_csv(input_csv)
    df["uniprot_residues"] = df["uniprot_residues"].apply(ast.literal_eval)
    split = split_interface_entries(df)
    for acc in ACCESSIONS:
        per_job = {r.job_name: frozenset(r.uniprot_residues) for r in split[split["UNPs"] == acc].itertuples()}
        table = read_st(os.path.join(nofig_run, acc, "ST_100.tsv"))
        seen_sets = []
        for r in table.itertuples():
            members = ast.literal_eval(r.entry_ids)
            sets = {per_job[m] for m in members}
            assert len(sets) == 1, f"{acc} ubs {r.ubs_id} groups different residue sets: {sets}"
            assert r.num_shared_residues == r.num_total_residues
            assert set(ast.literal_eval(r.total_residues)) == next(iter(sets))
            seen_sets.append(next(iter(sets)))
        # identical sets never end up in two different UBS, and every entry is placed
        assert len(seen_sets) == len(set(seen_sets))
        assert set(seen_sets) == set(per_job.values())


def test_multimeric_status_rollup(nofig_run):
    table = read_st(os.path.join(nofig_run, "P00001", "ST_100.tsv"))
    by_members = {frozenset(ast.literal_eval(r.entry_ids)): r.multimeric_status_all for r in table.itertuples()}
    assert by_members[frozenset({"j1", "j2"})] == "1"
    assert by_members[frozenset({"j4", "j6"})] == "1 3"   # monomer + heteromer interface member


def test_workers_two_same_tables(tmp_path, input_csv, nofig_run):
    out = str(tmp_path / "w2")
    run_cli(SCRIPT, "--input", input_csv, "--out", out, "--no-figures", "--workers", 2)
    assert st_tables(out) == st_tables(nofig_run)


def test_subset_flag(tmp_path, input_csv, nofig_run):
    out = str(tmp_path / "subset")
    run_cli(SCRIPT, "--input", input_csv, "--out", out, "--no-figures", "--subset", "P00003")
    dirs = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))]
    assert dirs == ["P00003"]
    # the interface row still contributes its P00003 half: same table as the full run
    assert st_tables(out)[os.path.join("P00003", "ST_100.tsv")] == \
        st_tables(nofig_run)[os.path.join("P00003", "ST_100.tsv")]


def test_thresholds_and_compression_flags(tmp_path, input_csv):
    out = str(tmp_path / "t100")
    run_cli(SCRIPT, "--input", input_csv, "--out", out, "--no-figures", "--thresholds", 1.0,
            "--compression", "zip", "--log-dir", str(tmp_path / "logs"))
    for acc in ACCESSIONS:
        names = set(os.listdir(os.path.join(out, acc)))
        fixed = {n for n in names if n.startswith("ST_") and "LJ" not in n}
        assert fixed == {"ST_100.tsv"}
        assert "input_data.csv.zip" in names
    assert os.path.isfile(os.path.join(out, "P00001", "ST_LJ060.tsv"))   # LJ is still computed
    assert glob.glob(str(tmp_path / "logs" / "ubs_unification_*.log"))
    assert not glob.glob(os.path.join(out, "*.log"))


# --------------------------------------------------------------------------- collector

def test_collector_reproduces_ubs_groups(nofig_run, input_csv):
    sites = collect_ubs.collect_ubs(nofig_run, threshold="ST_100", input_csv=input_csv)
    assert list(sites.columns) == ["site_uid", "uniprot", "residues", "members", "n_members",
                                   "multimeric_status_all", "ligands"]
    assert len(sites) == sum(len(v) for v in EXPECTED_ST100.values())
    for acc, expected in EXPECTED_ST100.items():
        table = read_st(os.path.join(nofig_run, acc, "ST_100.tsv"))
        sub = sites[sites["uniprot"] == acc]
        assert len(sub) == len(table)
        for r in table.itertuples():
            uid = f"{acc}:ST_100:{r.ubs_id}"
            row = sub[sub["site_uid"] == uid]
            assert len(row) == 1, uid
            row = row.iloc[0]
            members = ast.literal_eval(r.entry_ids)
            assert row["members"].split(";") == members            # member order kept
            assert row["n_members"] == len(members) == int(r.num_entries)
            assert row["multimeric_status_all"] == r.multimeric_status_all
            expected_res = sorted(int(t.rsplit("_", 1)[1]) for t in ast.literal_eval(r.total_residues))
            assert row["residues"] == "+".join(map(str, expected_res))
        assert as_groups(m.split(";") for m in sub["members"]) == as_groups(expected)
        # rows come out sorted by accession, then ubs_id
        assert sub["site_uid"].str.rsplit(":", n=1).str[1].astype(int).tolist() == list(range(1, len(sub) + 1))

    by_members = sites.set_index("members")
    assert by_members.loc["j1;j2", "residues"] == "10+11+12+13"
    assert by_members.loc["j1;j2", "ligands"] == "ATP;ADP"
    assert by_members.loc["j4;j6", "ligands"] == "MG;HEM"
    assert by_members.loc["j4;j6", "multimeric_status_all"] == "1 3"
    assert by_members.loc["j6", "residues"] == "7+8"        # P00003 half of the interface row


def test_collector_without_input_has_no_ligands(nofig_run):
    sites = collect_ubs.collect_ubs(nofig_run)
    assert "ligands" not in sites.columns
    assert sites["site_uid"].str.match(r"^P0000[123]:ST_100:\d+$").all()


def test_collector_other_thresholds(nofig_run):
    st050 = collect_ubs.collect_ubs(nofig_run, threshold=0.5)
    assert as_groups(m.split(";") for m in st050[st050.uniprot == "P00001"]["members"]) == as_groups(EXPECTED_ST050["P00001"])
    assert st050["site_uid"].str.contains(":ST_050:").all()
    lj = collect_ubs.collect_ubs(nofig_run, threshold="LJ")
    assert set(lj["uniprot"]) == {"P00001"}
    assert lj["site_uid"].str.startswith("P00001:ST_LJ060:").all()


def test_collector_label_normalization():
    n = collect_ubs.normalize_threshold_label
    assert n("ST_100") == n("100") == n("1.0") == n(1) == "ST_100"
    assert n("ST_070") == n("0.7") == n(70) == "ST_070"
    assert n("LJ") == n("ST_LJ") == "LJ"
    assert n("ST_LJ060") == "ST_LJ060"


def test_collector_cli(tmp_path, nofig_run, input_csv):
    out = tmp_path / "sites.tsv"
    run_cli(COLLECTOR, nofig_run, "--threshold", "ST_100", "--out", str(out), "--input", input_csv)
    df = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    expected = collect_ubs.collect_ubs(nofig_run, "ST_100", input_csv).astype(str)
    pd.testing.assert_frame_equal(df, expected)


# --------------------------------------------------------------------------- defaults / contract

def test_cli_defaults_unchanged():
    args = rub.build_parser().parse_args(["--input", "x.csv", "--out", "y"])
    assert args.thresholds == [0.3, 0.5, 0.7, 1.0]
    assert args.workers == 1
    assert args.subset is None
    assert args.figures is True
    assert args.compression == "gz"
    assert args.log_dir is None
    assert rub.build_parser().parse_args(["--input", "x", "--out", "y", "--no-figures"]).figures is False


def test_run_pipeline_defaults_unchanged():
    params = inspect.signature(rub.run_pipeline).parameters
    assert tuple(params["thresholds"].default) == (0.3, 0.5, 0.7, 1.0)
    assert params["workers"].default == 1
    assert params["subset"].default is None
    assert params["figures"].default is True
    assert params["compression"].default == "gz"
    assert params["save_trees"].default is True
    assert params["dendro_show_threshold"].default == "none"
    assert rub.default_dendro_marks([0.3, 0.5, 0.7, 1.0]) == ["030", "050", "070", "LJ"]
    assert inspect.signature(rub.process_uniprot).parameters["save_figures"].default is True


def test_run_pipeline_validates_arguments(tmp_path, input_csv):
    with pytest.raises(ValueError):
        rub.run_pipeline(input_csv, str(tmp_path / "a"), thresholds=[1.5])
    with pytest.raises(ValueError):
        rub.run_pipeline(input_csv, str(tmp_path / "b"), compression="tar")
    with pytest.raises(ValueError):
        rub.run_pipeline(input_csv, str(tmp_path / "c"), workers=0)


def test_input_contract_parsing_unchanged(input_csv):
    df = pd.read_csv(input_csv)
    assert list(df.columns) == COLUMNS
    df["uniprot_residues"] = df["uniprot_residues"].apply(ast.literal_eval)
    split = split_interface_entries(df)
    assert len(split) == len(ROWS) + 1                       # the interface row becomes two rows
    j6 = split[split["job_name"] == "j6"].set_index("UNPs")
    assert list(j6.index) == ["P00001", "P00003"]
    assert j6.loc["P00001", "uniprot_residues"] == ["P00001_50", "P00001_51", "P00001_52"]
    assert j6.loc["P00003", "uniprot_residues"] == ["P00003_7", "P00003_8"]
    assert int(j6.loc["P00003", "prot_chains_num"]) == 2 and int(j6.loc["P00003", "UNPs_num"]) == 2
    assert rub.multimeric_status_code(2, 2) == 3 and rub.multimeric_status_code(1, 1) == 1


def test_filter_subset_keeps_interface_rows(input_csv):
    df = pd.read_csv(input_csv)
    kept = rub.filter_subset(df, ["P00003"])
    assert sorted(kept["job_name"]) == ["j6", "j7"]
    assert rub.filter_subset(df, ["P00002"])["job_name"].tolist() == ["j5"]
