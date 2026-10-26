"""
Regression tests for GitHub issue #7:
    "[KDE plots] Overall chain classification showing disagreement between bar marker and
     overall Probability"  (example entry 5px2-A-ZN-903).

Root cause: a metric whose model value lands in a zero-density region for BOTH groups is
"Undefined" (denom == 0 -> P(apo)/P(holo) = NaN). Its NaN posteriors are correctly EXCLUDED from
avg_P(apo)/avg_P(holo) by weighted_mean_safe, but the per-metric `score` used a 0.0 fallback (a real
"tie" value), so it was INCLUDED in the overall score-average. The confidence-bar marker is drawn at
overall_score, while the Probability/Classification text comes from avg_P(*) -> the two disagreed.

Fix (kde_core.classify_metric): undefined metrics now carry score=NaN (mirrors the already-fixed
no-model-value branch), so they are excluded from overall_score exactly as they are from avg_P(*).
This is shared code, so it fixes BOTH the chain and pocket overall classifications.
"""
import sys, os
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kde_core import (classify_metric, aggregate_overall, MetricData, KDEResult,
                      DensityCurves, Priors, MetricClassResult)


def _make_mcr(p_apo, p_holo, score, nd=0.7, weight=1.0):
    """Minimal MetricClassResult as classify_metric emits it: an undefined metric (NaN P) carries
    NaN model-point fields (post-fix), a defined one carries real values."""
    undefined = np.isnan(p_apo)
    cls = "Undefined" if undefined else ("Apo" if p_apo > p_holo else "Holo")
    nd_val = np.nan if undefined else nd
    return MetricClassResult(
        classification=cls, P_apo=p_apo, P_holo=p_holo, score=score, density_sum=nd_val,
        norm_density_strongest=nd_val, norm_density_total=nd_val, norm_density_combined=nd_val,
        weight=(np.nan if undefined else weight), model_value=0.0, likelihood_apo=0.0,
        likelihood_holo=0.0, prior_apo=0.5, prior_holo=0.5,
    )


class TestUndefinedMetricScore:
    def test_zero_density_point_yields_nan_score(self):
        """A model value where both apo and holo densities are 0 (denom==0) -> score is NaN, not 0.0."""
        x = np.linspace(0, 10, 101)
        apo = np.exp(-((x - 2.0) ** 2) / 0.2)
        holo = np.exp(-((x - 3.0) ** 2) / 0.2)
        apo[x >= 6] = 0.0
        holo[x >= 6] = 0.0
        curves = DensityCurves(x=x, apo=apo, holo=holo, all=apo + holo,
                               bw_apo=0.1, bw_holo=0.1, bw_all=0.1, bw_mode="pooled")
        kde = KDEResult(curves=curves, max_apo=float(apo.max()), max_holo=float(holo.max()),
                        max_all=float((apo + holo).max()), overlap=0.5, mode_overlap=0.5)
        md = MetricData(name="pocket_dist", apo=np.array([1.9, 2.0, 2.1]),
                        holo=np.array([2.9, 3.0, 3.1]), combo=np.array([2.0, 3.0]),
                        model_value=9.0)  # far in the tail -> zero density for both groups
        res = classify_metric(md, kde, Priors(mode="equal"))
        assert res.classification == "Undefined"
        assert np.isnan(res.P_apo) and np.isnan(res.P_holo)
        assert np.isnan(res.score), "undefined metric must carry NaN score (was 0.0 -> issue #7)"
        # Issue #7 (extended): ALL model-point fields are NaN for an undefined point, so it is
        # excluded uniformly from every overall aggregate (matches the no-model-value branch; was 0.0).
        assert np.isnan(res.norm_density_strongest)
        assert np.isnan(res.norm_density_total)
        assert np.isnan(res.norm_density_combined)
        assert np.isnan(res.weight)

    def test_no_model_value_and_zero_density_branches_match(self):
        """Both 'Undefined' branches (no model value vs zero-density point) must return identical
        all-NaN model-point fields, so they aggregate identically."""
        x = np.linspace(0, 10, 101)
        apo = np.exp(-((x - 2.0) ** 2) / 0.2); holo = np.exp(-((x - 3.0) ** 2) / 0.2)
        apo[x >= 6] = 0.0; holo[x >= 6] = 0.0
        curves = DensityCurves(x=x, apo=apo, holo=holo, all=apo + holo,
                               bw_apo=0.1, bw_holo=0.1, bw_all=0.1, bw_mode="pooled")
        kde = KDEResult(curves=curves, max_apo=float(apo.max()), max_holo=float(holo.max()),
                        max_all=float((apo + holo).max()), overlap=0.5, mode_overlap=0.5)
        base = dict(apo=apo, holo=holo, combo=np.array([2.0, 3.0]))
        no_model = classify_metric(MetricData(name="m", model_value=np.nan, **base), kde, Priors())
        zero_dens = classify_metric(MetricData(name="m", model_value=9.0, **base), kde, Priors())
        for f in ("score", "norm_density_strongest", "norm_density_total", "norm_density_combined", "weight"):
            assert np.isnan(getattr(no_model, f)) and np.isnan(getattr(zero_dens, f)), f"{f} should be NaN in both"


class TestOverallMarkerAgreesWithProbability:
    """The bar marker (overall_score) must agree with the displayed Probability (avg_P) on the overall bar."""

    def _overall(self, undef_score):
        # 5px2-A-ZN-903 chain shape: 2 undefined metrics + 1 defined, strongly-apo metric.
        metric_results = {
            "pocket_dist": _make_mcr(np.nan, np.nan, undef_score),
            "tm_score":    _make_mcr(np.nan, np.nan, undef_score),
            "rmsd":        _make_mcr(1.0, 0.0, -1.0),
        }
        kr = SimpleNamespace(overlap=0.5, mode_overlap=0.5)
        return aggregate_overall(metric_results, {k: kr for k in metric_results})

    def test_marker_matches_probability_after_fix(self):
        ov = self._overall(undef_score=np.nan)  # post-fix: undefined -> NaN
        assert ov.avg_P_apo == 1.0 and ov.avg_P_holo == 0.0
        assert ov.overall_classification == "Apo"
        # marker position == position implied by the probabilities (score = P_holo - P_apo)
        implied = round(ov.avg_P_holo - ov.avg_P_apo, 3)
        assert ov.overall_score == implied == -1.0
        # marker on the correct side of centre for the class
        assert (ov.overall_score < 0) == (ov.overall_classification == "Apo")

    def test_documents_the_old_bug(self):
        """With the 0.0 fallback the marker (overall_score) disagreed with the probability."""
        ov = self._overall(undef_score=0.0)  # pre-fix behaviour
        assert ov.avg_P_apo == 1.0           # probability still says Apo 1.00
        implied = round(ov.avg_P_holo - ov.avg_P_apo, 3)  # == -1.0
        assert ov.overall_score == -0.333    # but marker landed at -1/3 ...
        assert ov.overall_score != implied   # ... disagreeing with the probability (the bug)

    def test_overall_density_excludes_undefined_metrics(self):
        """avg_normalized_density_strongest must reflect only the defined metric(s) (issue #7 extended):
        undefined metrics now carry NaN densities and are masked out, instead of dragging the average
        toward 0 with a spurious 0.0."""
        metric_results = {
            "pocket_dist": _make_mcr(np.nan, np.nan, np.nan),       # undefined -> NaN densities
            "tm_score":    _make_mcr(np.nan, np.nan, np.nan),       # undefined -> NaN densities
            "rmsd":        _make_mcr(1.0, 0.0, -1.0, nd=0.6),       # defined, strongest density 0.6
        }
        kr = SimpleNamespace(overlap=0.5, mode_overlap=0.5)
        ov = aggregate_overall(metric_results, {k: kr for k in metric_results})
        # only rmsd contributes -> 0.6, NOT (0+0+0.6)/3 = 0.2
        assert ov.avg_norm_density_strongest == 0.6
        assert ov.avg_norm_density_combined == 0.6
