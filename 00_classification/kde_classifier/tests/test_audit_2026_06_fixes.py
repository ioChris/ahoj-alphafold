"""
Regression tests for the 2026-06-30 pre-rerun audit batch (AHOJ-004 … AHOJ-013)
plus the resolution/pLDDT summary fields and the --config loader.

Run:  .venv\\Scripts\\python.exe -m pytest tests/test_audit_2026_06_fixes.py
See KNOWN_ISSUES.md for the issue descriptions each test pins.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest


# --- F2 / F3: sort_dataframe_v2c resolution handling + numeric coercion ---------------------

from parsing_functions_DBv2 import (
    sort_dataframe_v2c, compute_metrics, relabel_apoholo_by_ions,
)
from stats_api import compute_absolute_statistics_v2
from stats_core import classify_point_composite, composite_distance


def _df_with_roles(resolutions, mapped_sqr=None):
    """Q + M + len(resolutions) candidate rows. resolutions may contain np.nan."""
    rows = [
        {"role": "Q", "resolution": 1.5, "%mapped_sqr": 100.0, "apoholo_assignment": "Q"},
        {"role": "M", "resolution": np.nan, "%mapped_sqr": 100.0, "apoholo_assignment": "M"},
    ]
    for i, r in enumerate(resolutions):
        rows.append({"role": "C", "resolution": r, "%mapped_sqr": 100.0,
                     "apoholo_assignment": "H"})
    return pd.DataFrame(rows)


class TestResolutionFilter:
    def test_drop_no_resolution_removes_nan_candidates_keeps_Q_M(self):
        # AHOJ-005: candidates with missing resolution (NMR '-' -> NaN) dropped; Q/M preserved.
        df = _df_with_roles([2.0, np.nan, 1.8])  # 3 candidates, middle is NMR-like
        out = sort_dataframe_v2c(df.copy(), drop_no_resolution=1)
        cand = out[out["role"] == "C"]
        assert len(cand) == 2
        assert cand["resolution"].notna().all()
        assert (out["role"] == "Q").any() and (out["role"] == "M").any()

    def test_drop_no_resolution_off_keeps_all(self):
        df = _df_with_roles([2.0, np.nan, 1.8])
        out = sort_dataframe_v2c(df.copy(), drop_no_resolution=0)
        assert (out["role"] == "C").sum() == 3  # nothing dropped on resolution

    def test_resolution_soft_role_guarded_and_keeps_nan(self):
        # AHOJ-005: soft cutoff drops only high-res candidates; NaN-res candidates kept here;
        #          Q/M never dropped even though M has NaN resolution.
        df = _df_with_roles([1.0, 3.0, np.nan])  # ok, too-coarse, NMR
        out = sort_dataframe_v2c(df.copy(), resolution_soft=2.0, drop_no_resolution=0)
        cand = out[out["role"] == "C"]
        assert set(np.round(cand["resolution"].fillna(-1), 1)) == {1.0, -1.0}  # 3.0 dropped, NaN kept
        assert (out["role"] == "M").any()  # M kept despite NaN resolution

    def test_string_resolution_and_object_mapped_cols_do_not_raise(self):
        # AHOJ-006: '%mapped_obs'/'pocket_dist' arrive as object dtype; resolution as '-'/strings.
        df = pd.DataFrame({
            "role":               ["Q",   "M",     "C",   "C",  "C"],
            "resolution":         ["1.5", "-",     "2.0", "-",  "model"],  # strings incl. '-'
            "%mapped_sqr":        [100.0, 100.0,   100.0, 80.0, 90.0],
            "%mapped_obs":        ["100", "100",   "100", "80", "90"],     # object dtype
            "pocket_dist":        ["0.2", "0.3",   "0.1", "5.0", "999"],   # object dtype
            "apoholo_assignment": ["Q",   "M",     "H",   "H",  "H"],
        })
        # The fix's guarantee: the >= / > comparisons on these columns must not raise on object
        # input. (sort_dataframe_v2c filters them, so reaching a result means no TypeError.)
        out = sort_dataframe_v2c(df.copy(), bndgrsds_prcnt_obs=75,
                                 pocket_distance=20.0, drop_no_resolution=1)
        # values were coerced to numeric (errors='raise' would throw if any string survived)
        assert pd.to_numeric(out["%mapped_obs"], errors="raise").notna().all()
        assert pd.to_numeric(out["pocket_dist"], errors="raise").notna().all()
        assert (out["role"] == "Q").any() and (out["role"] == "M").any()
        # '-'/'model' resolution candidates dropped; only the res=2.0 candidate survives.
        assert (out["role"] == "C").sum() == 1


# --- F4: compute_metrics query anchor -------------------------------------------------------

class TestQueryAnchor:
    def test_anchor_forces_min_zero_clean_keeps_real_min(self):
        df = pd.DataFrame({"pocket_rms": [0.27, 0.5, 1.0]})
        dq = pd.DataFrame({"pocket_rms": [0.0]})
        anchored = compute_metrics(df, dq, deltas=0, columns=["pocket_rms"], anchor_query=True)
        clean = compute_metrics(df, dq, deltas=0, columns=["pocket_rms"], anchor_query=False)
        assert anchored.iloc[0]["pocket_rms_min"] == 0.0           # synthetic 0 anchors the min
        assert clean.iloc[0]["pocket_rms_min"] == 0.27             # real candidate minimum
        # the anchor also shifts the mean downward
        assert clean.iloc[0]["pocket_rms_mean"] > anchored.iloc[0]["pocket_rms_mean"]


# --- F6: classify_point_composite weight order ----------------------------------------------

class TestCompositeWeights:
    def test_caller_passes_w_iqr_and_w_extremes_in_correct_slots(self):
        # AHOJ-009: classify_point_composite must call composite_distance(value, stats,
        #          w_center, w_iqr, w_extremes) — NOT swapped. Use w_iqr != w_extremes so a
        #          swap changes the number, then compare against composite_distance directly.
        stats = {
            "Apo":  {"mean": 1.0, "median": 1.0, "min": 0.0, "max": 2.0, "q1": 0.5, "q3": 1.5},
            "Holo": {"mean": 5.0, "median": 5.0, "min": 4.0, "max": 6.0, "q1": 4.5, "q3": 5.5},
            "All":  {"mean": 3.0, "median": 3.0, "min": 0.0, "max": 6.0, "q1": 1.0, "q3": 5.0},
        }
        comparisons = {"m": stats}
        df_point = pd.DataFrame({"m": [2.7]})
        w_center, w_iqr, w_extremes = 0.5, 0.3, 0.2
        res = classify_point_composite(df_point, comparisons, ["m"],
                                       w_center=w_center, w_iqr=w_iqr, w_extremes=w_extremes)
        overall_range = max(stats["Apo"]["max"], stats["Holo"]["max"]) - \
                        min(stats["Apo"]["min"], stats["Holo"]["min"])
        expected_apo = composite_distance(2.7, stats["Apo"], w_center, w_iqr, w_extremes,
                                          normalize=True, norm_factor=overall_range)
        # classify_point_composite rounds per-metric distances to 3 dp; compare to the rounded
        # correctly-ordered value. A weight swap would change this by >> 1e-3.
        assert res["per_metric"]["m"]["composite_distance_apo"] == pytest.approx(round(expected_apo, 3), abs=1e-9)
        # sanity: with these weights a swap would be detectable
        swapped = composite_distance(2.7, stats["Apo"], w_center, w_extremes, w_iqr,
                                     normalize=True, norm_factor=overall_range)
        assert abs(round(expected_apo, 3) - round(swapped, 3)) > 1e-3


# --- F7: diff_means with an exactly-zero group mean -----------------------------------------

class TestDiffMeansZeroMean:
    def test_zero_apo_mean_not_turned_into_nan(self):
        apo = pd.DataFrame({"pocket_rms": [0.0, 0.0, 0.0]})   # mean exactly 0.0
        holo = pd.DataFrame({"pocket_rms": [2.0, 3.0, 4.0]})  # mean 3.0
        out = compute_absolute_statistics_v2(apo, holo, ["pocket_rms"])
        assert out["pocket_rms"]["diff_means"] == pytest.approx(3.0, abs=1e-6)


# --- F9: relabel_apoholo_by_ions on a non-default index --------------------------------------

class TestRelabelIndexSafety:
    def test_relabel_correct_rows_with_shuffled_index(self):
        df = pd.DataFrame(
            {
                "role": ["C", "C", "Q"],
                "apoholo_assignment": ["H", "A", "Q"],
                "query_ligs": ["", "ATP", ""],
                "nonquery_ligs": ["ZN", "", ""],   # row0 binds only an ion
            },
            index=[5, 2, 8],   # deliberately non-contiguous / shuffled
        )
        out, _changed = relabel_apoholo_by_ions(df, builtin="total_le1", verbose=False)
        assert out.loc[5, "apoholo_assignment"] == "A**"  # ion-only holo -> apo marker
        assert out.loc[2, "apoholo_assignment"] == "H"    # real ligand -> holo
        assert out.loc[8, "apoholo_assignment"] == "Q"    # query untouched


# --- Combined methods summary + consensus ---------------------------------------------------

from batch_job_functions import (
    build_classification_summary, extract_model_pLDDT_and_query_meta,
)


class TestClassificationSummary:
    def _kde(self, cls, p_holo):
        return {"overall": {"overall_classification": cls, "avg_P(holo)": p_holo,
                            "avg_P(apo)": round(1 - p_holo, 3), "overall_score": 0.1,
                            "avg_normalized_density_strongest": 0.4, "overall_overlap": 0.5,
                            "overall_mode_overlap": 0.3}}

    def test_consensus_unanimous(self):
        s = build_classification_summary(
            self._kde("Holo", 0.7),
            {"overall": {"overall_classification": "Holo", "avg_score": 0.2, "avg_overlap": 0.4}},
            {"classification": "Holo", "score": 0.1, "avg_distance_apo": 0.3, "avg_distance_holo": 0.2},
            {"classification": "Holo", "score": 0.1, "avg_distance_apo": 0.3, "avg_distance_holo": 0.2},
            {"classification": "Holo", "score": 0.1, "avg_distance_apo": 0.3, "avg_distance_holo": 0.2},
        )
        assert s["consensus"] == {"class": "Holo", "frac": 1.0, "n_methods": 5, "n_holo": 5, "n_apo": 0}
        assert s["kde"]["p_holo"] == 0.7

    def test_consensus_split_and_ignores_undefined(self):
        s = build_classification_summary(
            self._kde("Apo", 0.3),
            {"overall": {"overall_classification": "Holo", "avg_score": 0.2, "avg_overlap": 0.4}},
            {"classification": "Apo", "score": 0.1, "avg_distance_apo": 0.2, "avg_distance_holo": 0.3},
            {"classification": "Undefined", "score": None, "avg_distance_apo": None, "avg_distance_holo": None},
            {"classification": "Holo", "score": 0.1, "avg_distance_apo": 0.3, "avg_distance_holo": 0.2},
        )
        # 2 Apo, 2 Holo among 4 valid votes (Undefined ignored) -> Tie
        assert s["consensus"]["n_methods"] == 4
        assert s["consensus"]["class"] == "Tie"


class TestModelMetaExtraction:
    def test_pLDDT_mean_and_query_meta(self):
        df = pd.DataFrame({
            "role": ["Q", "M", "C"],
            "bfac_avg": [30.0, 98.0, 40.0],          # only the M row matters for pLDDT
            "resolution": ["2.2", np.nan, "1.9"],
            "exp_method": ["X-RAY_DIFFRACTION", "model", "X-RAY_DIFFRACTION"],
        })
        out = extract_model_pLDDT_and_query_meta(df, meta_role="Q")
        assert out["model_pocket_pLDDT_avg"] == pytest.approx(98.0, abs=1e-6)
        assert out["n_model_rows"] == 1
        assert out["query_resolution"] == pytest.approx(2.2, abs=1e-6)
        assert out["query_exp_method"] == "X-RAY_DIFFRACTION"

    def test_no_model_row(self):
        df = pd.DataFrame({"role": ["Q", "C"], "bfac_avg": [30.0, 40.0],
                           "resolution": ["2.2", "1.9"], "exp_method": ["X-RAY_DIFFRACTION", "X"]})
        out = extract_model_pLDDT_and_query_meta(df)
        assert out["n_model_rows"] == 0
        assert np.isnan(out["model_pocket_pLDDT_avg"])


# --- AHOJ-013: DP classifier curve reuse (plot fidelity) ------------------------------------

class TestDPCurveReuse:
    def test_return_curves_matches_plain_and_exposes_arrays(self):
        from stats_core import dp_classify_new_point
        df = pd.DataFrame({
            "apoholo_assignment": ["A"] * 6 + ["H"] * 6,
            "m": [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 3.0, 3.1, 2.9, 3.2, 2.8, 3.05],
        })
        new = pd.DataFrame({"m": [1.0]})
        plain = dp_classify_new_point(df, ["m"], new)
        res, curves = dp_classify_new_point(df, ["m"], new, return_curves=True)
        # the classification result must be byte-identical whether or not curves are captured
        assert plain == res
        # and the captured curve carries the exact densities (equal-length x/apo/holo arrays)
        c = curves["m"]
        assert set(c.keys()) == {"x", "apo", "holo"}
        assert len(c["x"]) == len(c["apo"]) == len(c["holo"]) > 0


# --- --config loader (precedence) -----------------------------------------------------------

class TestConfigLoader:
    def test_config_applies_and_cli_overrides(self, monkeypatch):
        import batch_process_database as bpd
        cfg = {"pocket_dist": 20.0, "drop_no_resolution": 1, "num_processes": 32,
               "save_per_method_json": 0, "processing_function": "do_kde_plots_and_classification"}
        # Write the temp config next to this test (avoids system-temp permission issues).
        p = os.path.join(os.path.dirname(__file__), "_tmp_cfg_test.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        try:
            monkeypatch.setattr(sys, "argv", ["x", "--config", p])
            a = bpd.parse_arguments()
            assert a.pocket_dist == 20.0 and a.drop_no_resolution == 1 and a.num_processes == 32

            # CLI flag must win over the config value
            monkeypatch.setattr(sys, "argv", ["x", "--config", p, "--num_processes", "8"])
            a2 = bpd.parse_arguments()
            assert a2.num_processes == 8 and a2.pocket_dist == 20.0
        finally:
            if os.path.exists(p):
                os.remove(p)


# --- AHOJ-017: incomplete entry (no global_results.csv) -> 'File Not Found', not 'Error' -----

import io as _io
import shutil
import tarfile
import types

from batch_job_functions import do_kde_plots_and_classification, load_file_from_query


class TestMissingGlobalResultsCsv:
    """AHOJ-017: a query result that exists but lacks global_results.csv must be categorized
    'File Not Found', not a cryptic 'Error'. The old code unpacked rest[2] before checking
    job_status, and the loader returns a 2-element rest on not-found -> IndexError -> 'Error'.

    Temp artifacts are created NEXT TO this test (not pytest's tmp_path / system temp), matching
    TestConfigLoader above -- system-temp is unreliable on the maintainer's box.
    """

    def _here(self, name):
        return os.path.join(os.path.dirname(__file__), name)

    def test_loader_returns_short_rest_on_missing_csv(self):
        # Root cause: the loader signals not-found with a 2-element rest, so a rest[2] access
        # before the status check used to raise IndexError ("tuple index out of range").
        d = self._here("_tmp_ahoj017_a")
        os.makedirs(d, exist_ok=True)
        try:
            with open(os.path.join(d, "stdout_log"), "w", encoding="utf-8") as fh:
                fh.write("x")   # present, but no global_results.csv
            status, rest = load_file_from_query(d, "1abc-A-LIG-1", "./global_results.csv")
            assert status == "File Not Found"
            assert len(rest) < 3          # the shape that used to crash on rest[2]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_dir_without_csv_is_file_not_found(self):
        # The early File-Not-Found return happens before any args.* access, so a bare args stub
        # suffices (and documents that no config is needed to reach this branch).
        d = self._here("_tmp_ahoj017_b")
        os.makedirs(d, exist_ok=True)
        try:
            with open(os.path.join(d, "stdout_log"), "w", encoding="utf-8") as fh:
                fh.write("x")
            status, _detail = do_kde_plots_and_classification(types.SimpleNamespace(), "1abc-A-LIG-1", d)
            assert status == "File Not Found"   # not 'Error', and no exception raised
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_tgz_without_csv_is_file_not_found(self):
        # Faithful to the production vector 7y7a-ie-PEB-202: a .tgz that exists but has no CSV.
        p = self._here("_tmp_ahoj017.tgz")
        try:
            with tarfile.open(p, "w:gz") as tf:
                data = b"log"
                info = tarfile.TarInfo("./stdout_log")
                info.size = len(data)
                tf.addfile(info, _io.BytesIO(data))
            status, _detail = do_kde_plots_and_classification(types.SimpleNamespace(), "7y7a-ie-PEB-202", p)
            assert status == "File Not Found"
        finally:
            if os.path.exists(p):
                os.remove(p)


# --- AHOJ-018: keep no-model entries (conformational analysis without an AlphaFold model) --------

from batch_job_functions import create_query_summary_row
from batch_process_database import parse_job_list_file


class TestKeepNoModelSummaryRow:
    """AHOJ-018: an entry with no role 'M' row can still carry the (model-independent) conformational
    analysis. When kept (keep_no_model=1), its summary row must be flagged has_model=0 and the headline
    apo/holo verdicts relabeled 'N/A (no model)', while a normal entry stays has_model=1 with its
    real verdict. Pins the schema contract the webserver keys off."""

    def _args(self):
        # create_query_summary_row only reads query_summary_deltas/metrics off args; a stub suffices.
        return types.SimpleNamespace(query_summary_deltas=1, query_summary_metrics=[])

    def _kde_overall(self, verdict):
        return {"overall": {"avg_P(apo)": None, "avg_P(holo)": None, "overall_score": None,
                            "overall_classification": verdict}}

    def _row(self, has_model, verdict="Holo"):
        df = create_query_summary_row(
            args=self._args(), job_name="1abc-A-LIG-1", query_path="b/j/1abc-A-LIG-1.tgz",
            df_q=pd.DataFrame([{"chains3": "A", "UNPs": "P1", "#resis3": 5, "RoG": 1.0,
                                "volume": 10.0, "sasa": 20.0}]),
            n_apo=3, n_holo=4, n_all=7,
            kde_results_pocket=self._kde_overall(verdict if has_model else "Undefined"),
            kde_results_chain=self._kde_overall(verdict if has_model else "Insufficient Data"),
            classification_summary={"pocket": {"consensus": {"class": verdict if has_model else "Undefined"}},
                                    "chain":  {"consensus": {"class": verdict if has_model else "Undefined"}}},
            model_query_meta={"n_model_rows": 1 if has_model else 0},
            has_model=has_model)
        return df.iloc[0]

    def test_no_model_row_flagged_and_relabeled(self):
        r = self._row(has_model=False)
        assert r["has_model"] == 0
        assert r["pocket_overall_classification"] == "N/A (no model)"
        assert r["chain_overall_classification"] == "N/A (no model)"
        assert r["pocket_consensus_class"] == "N/A (no model)"
        assert r["chain_consensus_class"] == "N/A (no model)"
        # numeric probabilities stay empty (no model to score)
        assert pd.isna(r["pocket_avg_P(apo)"]) and pd.isna(r["pocket_overall_score"])

    def test_model_present_keeps_real_verdict(self):
        r = self._row(has_model=True, verdict="Holo")
        assert r["has_model"] == 1
        assert r["pocket_overall_classification"] == "Holo"
        assert r["pocket_consensus_class"] == "Holo"

    def test_has_model_defaults_true(self):
        # Back-compat: existing callers that omit has_model get the model-present behaviour.
        df = create_query_summary_row(
            args=self._args(), job_name="1abc-A-LIG-1", query_path="b/j/1abc-A-LIG-1.tgz",
            df_q=pd.DataFrame([{"chains3": "A"}]), n_apo=1, n_holo=1, n_all=2,
            kde_results_pocket=self._kde_overall("Apo"), kde_results_chain=self._kde_overall("Apo"))
        assert df.iloc[0]["has_model"] == 1


class TestJobListParsing:
    """AHOJ-018: --job_list accepts a bare job_name per line OR a full Discarded_jobs.txt line
    ('<job_name>: <reason>'), so a prior run's discard log feeds back verbatim."""

    def test_parses_bare_and_discard_lines(self):
        p = os.path.join(os.path.dirname(__file__), "_tmp_joblist.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("9b4h-A-A1AIO-2043: No model (role M) row\n"
                     "1abc-A-LIG-1\n"
                     "\n"                                   # blank line ignored
                     "  4xyz-B-ZN-300  : Low candidates after filtering: apo=0, holo=0, total=0\n")
        try:
            got = parse_job_list_file(p)
            assert got == {"9b4h-A-A1AIO-2043", "1abc-A-LIG-1", "4xyz-B-ZN-300"}
        finally:
            if os.path.exists(p):
                os.remove(p)
