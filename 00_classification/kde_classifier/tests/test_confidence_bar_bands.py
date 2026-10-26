"""
Regression tests for AHOJ-016: centralized + documented confidence-bar verbal-band spec, with
the apo/holo probability bands defined DIRECTLY on the [0.5, 1.0] winning-posterior scale.

Single source of truth in kde_plot.py:
    VERBAL_BANDS    = ("Very low", "Low", "Moderate", "High", "Very high")
    UNIT_BAND_EDGES = (0.20, 0.30, 0.60, 0.85)   # densities / overlaps, on [0,1]
    PROB_BAND_EDGES = (0.55, 0.60, 0.70, 0.85)   # apo/holo probability, on [0.5,1]  (no sigmoid)

read by _verbal_from_value / _verbal_from_unit / verbose_scoring_label / verbose_probability_label
and band_spec_table(); documented in docs/confidence_bar_spec.md; mirrored in confbars.js.

These tests:
  * DENSITY/OVERLAP path is unchanged -> compared against a frozen copy of the original logic;
  * PROBABILITY path uses the new direct edges -> boundaries, rounding-independence, exact labels;
  * MIGRATION: document exactly where the new direct edges differ from the previous sigmoid
    (gain k=1.64) revision -- four narrow slivers, dominated by (0.8223, 0.85] Very high -> High;
  * confbars.js mirrors the Python constants exactly;
  * docs/confidence_bar_spec.md matches band_spec_table() (no doc drift).

Run:  python -m pytest tests/test_confidence_bar_bands.py
"""
import math
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kde_plot as kp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFBARS_JS = os.path.join(REPO_ROOT, "analysis", "interactive_confbars_prototype", "confbars.js")
SPEC_DOC = os.path.join(REPO_ROOT, "docs", "confidence_bar_spec.md")

BANDS = ("Very low", "Low", "Moderate", "High", "Very high")


# --------------------------------------------------------------------------------------------
# Frozen reference for the DENSITY/OVERLAP path (unchanged by AHOJ-016). Do not edit to match.
# --------------------------------------------------------------------------------------------
def _ref_unitize(value, src_min=0.0, src_max=1.0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    rng = float(src_max - src_min) if src_max != src_min else 1.0
    v = (float(value) - src_min) / rng
    if np.isnan(v):
        return np.nan
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _ref_verbal_from_unit(u):
    if u is None or np.isnan(u):
        return "n/a"
    if u <= 0.20:
        return "Very low"
    elif u <= 0.30:
        return "Low"
    elif u <= 0.60:
        return "Moderate"
    elif u <= 0.85:
        return "High"
    else:
        return "Very high"


def _ref_verbose_scoring_label(score):
    abs_score = abs(score) if score is not None else None
    u = _ref_unitize(abs_score, 0.0, 1.0)
    descriptor = _ref_verbal_from_unit(u)
    if abs_score is None or (isinstance(abs_score, float) and np.isnan(abs_score)):
        return "n/a"
    return f"{descriptor} ({abs_score:.2f})"


# Frozen copy of the PREVIOUS (sigmoid, k=1.64) probability labeler -- the "old" side of the
# migration comparison. This is NOT the current behaviour; the current code uses direct edges.
def _old_sigmoid_prob_band(p, k=1.64):
    p = min(max(float(p), 0.5), 1.0 - 1e-12)
    u = ((1.0 / (1.0 + math.exp(-k * math.log(p / (1.0 - p))))) - 0.5) / 0.5
    for edge, label in zip((0.20, 0.30, 0.60, 0.85), BANDS):
        if u <= edge:
            return label
    return BANDS[-1]


def _new_prob_band(p):
    """The current behaviour: bin p directly against PROB_BAND_EDGES."""
    for edge, label in zip(kp.PROB_BAND_EDGES, BANDS):
        if p <= edge:
            return label
    return BANDS[-1]


def _grid(a, b, n):
    return list(np.linspace(a, b, n))


# density/overlap unit edges: (edge, at-edge band, just-above band)
_UNIT_EDGE_EXPECT = [
    (0.20, "Very low", "Low"), (0.30, "Low", "Moderate"),
    (0.60, "Moderate", "High"), (0.85, "High", "Very high"),
]
# probability edges on [0.5,1]
_PROB_EDGE_EXPECT = [
    (0.55, "Very low", "Low"), (0.60, "Low", "Moderate"),
    (0.70, "Moderate", "High"), (0.85, "High", "Very high"),
]


class TestConstants:
    def test_ssot_constants(self):
        assert kp.VERBAL_BANDS == BANDS
        assert kp.UNIT_BAND_EDGES == (0.20, 0.30, 0.60, 0.85)
        assert kp.PROB_BAND_EDGES == (0.55, 0.60, 0.70, 0.85)

    def test_sigmoid_helpers_removed(self):
        """The sigmoid machinery is gone (direct edges replace it)."""
        assert not hasattr(kp, "_unitize_probability")
        assert not hasattr(kp, "probability_band_edges_from_u")
        assert not hasattr(kp, "PROB_SIGMOID_GAIN_K")


class TestDensityOverlapUnchanged:
    """The density/overlap path must be byte-identical to the pre-AHOJ-016 implementation."""

    def test_unit_grid_and_edges(self):
        us = _grid(0.0, 1.0, 100001)
        for edge, _, _ in _UNIT_EDGE_EXPECT:
            us += [edge, edge - 1e-9, edge + 1e-9]
        bad = [u for u in us if kp._verbal_from_unit(u) != _ref_verbal_from_unit(u)]
        assert bad == [], f"{len(bad)} density-path mismatches, e.g. {bad[:5]}"

    def test_scoring_label_strings(self):
        for x in _grid(0.0, 1.0, 20001):
            assert kp.verbose_scoring_label(x) == _ref_verbose_scoring_label(x), x

    def test_nan_none(self):
        assert kp.verbose_scoring_label(None) == "n/a"
        assert kp.verbose_scoring_label(np.nan) == "n/a"
        assert kp.verbose_probability_label(np.nan, np.nan, "x") == "Unclassified"
        assert kp._verbal_from_value(np.nan, kp.PROB_BAND_EDGES) == "n/a"


class TestProbabilityDirectEdges:
    def _band(self, p):
        # descriptor emitted by verbose_probability_label for winning posterior p
        label = kp.verbose_probability_label(p, 1.0 - p, "Apo")
        return label.split("Probability: ")[1].rsplit(" (", 1)[0]

    def test_edges_resolve_to_lower_band(self):
        for edge, lower, upper in _PROB_EDGE_EXPECT:
            assert self._band(edge) == lower, (edge, "at edge -> lower")
            assert self._band(edge + 1e-9) == upper, (edge, "just above -> upper")
            assert self._band(edge - 1e-9) == lower, (edge, "just below -> lower")

    def test_matches_direct_binning_on_grid(self):
        for p in _grid(0.5, 1.0, 50001):
            assert self._band(p) == _new_prob_band(p), p

    def test_full_label_format(self):
        assert kp.verbose_probability_label(0.90, 0.10, "Apo") == \
            "Classification: Apo - Probability: Very high (0.90)"
        assert kp.verbose_probability_label(0.10, 0.78, "Holo") == \
            "Classification: Holo - Probability: High (0.78)"

    def test_worked_examples(self):
        assert self._band(0.52) == "Very low"
        assert self._band(0.58) == "Low"
        assert self._band(0.65) == "Moderate"
        assert self._band(0.78) == "High"
        assert self._band(0.90) == "Very high"
        assert self._band(0.85) == "High"       # edge belongs to lower band
        assert self._band(0.8501) == "Very high"


class TestRoundingNeverFlipsLabel:
    """The 2 dp display is independent of the band decision (which uses the unrounded value)."""

    def test_scoring_label_uses_unrounded_value(self):
        assert kp.verbose_scoring_label(0.204) == "Low (0.20)"      # >0.20 -> Low, shows 0.20
        assert kp.verbose_scoring_label(0.596) == "Moderate (0.60)"  # <=0.60 -> Moderate, shows 0.60

    def test_probability_label_uses_unrounded_value(self):
        # 0.701 rounds to "0.70" but exceeds the Moderate/High edge -> High.
        assert kp.verbose_probability_label(0.701, 0.299, "Apo") == \
            "Classification: Apo - Probability: High (0.70)"
        # 0.851 rounds to "0.85" but exceeds the High/Very-high edge -> Very high.
        assert kp.verbose_probability_label(0.851, 0.149, "Apo") == \
            "Classification: Apo - Probability: Very high (0.85)"

    def test_band_decided_on_unrounded_probability(self):
        differ_if_rounded_first = 0
        for p in _grid(0.5, 1.0, 20001):
            band_exact = _new_prob_band(p)
            label = kp.verbose_probability_label(p, 1.0 - p, "Apo")
            assert f"Probability: {band_exact} " in label, (p, band_exact, label)
            if _new_prob_band(round(p, 2)) != band_exact:
                differ_if_rounded_first += 1
        assert differ_if_rounded_first > 0, "expected boundary points where round-first would flip"


class TestSigmoidToDirectMigration:
    """Document exactly how the new direct edges differ from the previous sigmoid revision."""

    def test_flip_transitions_are_the_four_documented_slivers(self):
        transitions = set()
        for p in _grid(0.5, 1.0, 200001):
            old, new = _old_sigmoid_prob_band(p), _new_prob_band(p)
            if old != new:
                transitions.add((old, new))
        assert transitions == {
            ("Very low", "Low"),
            ("Moderate", "Low"),
            ("High", "Moderate"),
            ("Very high", "High"),
        }, transitions

    def test_flip_fraction_is_small(self):
        xs = _grid(0.5, 1.0, 200001)
        frac = sum(_old_sigmoid_prob_band(p) != _new_prob_band(p) for p in xs) / len(xs)
        assert 0.08 < frac < 0.11, frac  # ~9.3% of the [0.5,1] range

    def test_headline_top_edge_flip(self):
        # the meaningful change: p in (0.8223, 0.85] was "Very high", now "High"
        assert _old_sigmoid_prob_band(0.83) == "Very high"
        assert _new_prob_band(0.83) == "High"
        # well inside a shared band, the two agree
        assert _old_sigmoid_prob_band(0.95) == _new_prob_band(0.95) == "Very high"
        assert _old_sigmoid_prob_band(0.52) == _new_prob_band(0.52) == "Very low"


class TestBandSpecTable:
    def test_labels_and_edges(self):
        rows = kp.band_spec_table()
        assert tuple(r["label"] for r in rows) == BANDS
        assert [(r["u_lo"], r["u_hi"]) for r in rows] == [
            (0.0, 0.20), (0.20, 0.30), (0.30, 0.60), (0.60, 0.85), (0.85, 1.0)
        ]
        assert [(round(r["p_lo"], 2), round(r["p_hi"], 2)) for r in rows] == [
            (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.85), (0.85, 1.00)
        ]

    def test_edges_are_two_decimals(self):
        """No derived multi-decimal edges -- every probability edge is round to 2 dp."""
        for r in kp.band_spec_table():
            assert round(r["p_lo"], 2) == r["p_lo"]
            assert round(r["p_hi"], 2) == r["p_hi"]


# The JavaScript mirror (the webserver's confidence-bar renderer) stays in parse-ahoj; it is not
# part of this copy, so the parity checks run only where it is present.
@pytest.mark.skipif(not os.path.isfile(CONFBARS_JS), reason="confbars.js (parse-ahoj web prototype) not present")
class TestPythonJsParity:
    def _read_js(self):
        with open(CONFBARS_JS, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_constants_match(self):
        js = self._read_js()
        bands = re.search(r"VERBAL_BANDS\s*=\s*\[(.*?)\]", js, re.S).group(1)
        assert tuple(s.strip().strip('"\'') for s in bands.split(",")) == kp.VERBAL_BANDS

        u = re.search(r"UNIT_BAND_EDGES\s*=\s*\[(.*?)\]", js, re.S).group(1)
        assert tuple(float(x) for x in u.split(",")) == kp.UNIT_BAND_EDGES

        p = re.search(r"PROB_BAND_EDGES\s*=\s*\[(.*?)\]", js, re.S).group(1)
        assert tuple(float(x) for x in p.split(",")) == kp.PROB_BAND_EDGES

    def test_js_has_no_sigmoid_left(self):
        js = self._read_js()
        assert "PROB_SIGMOID_GAIN_K" not in js
        assert "unitizeProbabilitySigmoid" not in js

    def test_js_verbalFromValue_reads_edges(self):
        js = self._read_js()
        body = re.search(r"function verbalFromValue\(value, edges\)\s*\{(.*?)\}", js, re.S).group(1)
        assert "edges[i]" in body and "VERBAL_BANDS" in body
        assert not re.search(r"0\.\d\d", body)  # no bare numeric thresholds inside the function


class TestDocConsistency:
    def test_doc_matches_band_spec_table(self):
        with open(SPEC_DOC, "r", encoding="utf-8") as fh:
            doc = fh.read()
        # direct probability edges and their pp forms
        for two_dp in ("0.55", "0.60", "0.70", "0.85"):
            assert two_dp in doc, f"missing edge {two_dp}"
        for pp in ("55 pp", "60 pp", "70 pp", "85 pp"):
            assert pp in doc, f"missing pp edge {pp}"
        # explicitly a direct scheme, and the migration note is present
        assert "no sigmoid" in doc.lower() or "no remap" in doc.lower()
        for label in kp.VERBAL_BANDS:
            assert label in doc
