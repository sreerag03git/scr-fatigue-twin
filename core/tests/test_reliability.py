"""FORM fatigue-reliability tests (DNV-RP-C210)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from scr_twin_core.montecarlo import UncertaintyModel, run_monte_carlo
from scr_twin_core.reliability import (
    SAFETY_CLASS_TARGET_PF,
    form_fatigue_reliability,
)

_PARAMS = UncertaintyModel().describe()


def _life():
    return run_monte_carlo(1.0 / 33.0, UncertaintyModel(), n_members=10000, seed=0).life_years


def test_beta_decreases_with_design_life():
    life = _life()
    b20 = form_fatigue_reliability(life, 20.0, _PARAMS).beta
    b40 = form_fatigue_reliability(life, 40.0, _PARAMS).beta
    assert b20 > b40


def test_pf_is_phi_of_minus_beta():
    r = form_fatigue_reliability(_life(), 25.0, _PARAMS)
    assert r.pf_cumulative == pytest.approx(float(norm.cdf(-r.beta)), rel=1e-9)


def test_importance_factors_sum_to_one_and_sn_dominates():
    r = form_fatigue_reliability(_life(), 25.0, _PARAMS)
    assert sum(r.importance.values()) == pytest.approx(1.0, abs=1e-9)
    assert r.importance["S-N scatter"] == max(r.importance.values())


def test_safety_class_targets():
    r = form_fatigue_reliability(_life(), 25.0, _PARAMS, safety_class="normal")
    assert r.target_pf == 1e-4
    assert r.target_beta == pytest.approx(float(norm.ppf(1 - 1e-4)), rel=1e-9)
    rh = form_fatigue_reliability(_life(), 25.0, _PARAMS, safety_class="high")
    assert rh.target_beta > r.target_beta  # higher class -> stricter


def test_mean_curve_life_exceeds_characteristic_median():
    life = _life()
    r = form_fatigue_reliability(life, 25.0, _PARAMS)
    # mean-basis life = characteristic median x 10^(2 sigma_logN)
    shift = 10.0 ** (2.0 * _PARAMS["sn_logN_std"])
    assert r.mean_curve_life_years == pytest.approx(math.exp(np.mean(np.log(life))) * shift, rel=1e-6)


def test_tighter_scatter_raises_beta():
    # Reduce the transfer/SCF scatter at fixed sn_logN_std (so the mean-curve shift
    # is unchanged): a tighter life distribution -> higher reliability index.
    tight = run_monte_carlo(1.0 / 33.0, UncertaintyModel(tf_gain_cov=0.02, scf_cov=0.02),
                            n_members=10000, seed=0)
    wide = run_monte_carlo(1.0 / 33.0, UncertaintyModel(tf_gain_cov=0.20, scf_cov=0.20),
                           n_members=10000, seed=0)
    b_tight = form_fatigue_reliability(tight.life_years, 25.0, tight.parameters).beta
    b_wide = form_fatigue_reliability(wide.life_years, 25.0, wide.parameters).beta
    assert b_tight > b_wide


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        form_fatigue_reliability(_life(), 25.0, _PARAMS, safety_class="bogus")
    with pytest.raises(ValueError):
        form_fatigue_reliability(_life(), -1.0, _PARAMS)
    with pytest.raises(ValueError):
        form_fatigue_reliability(np.array([1.0]), 25.0, _PARAMS)


def test_all_safety_classes_defined():
    assert set(SAFETY_CLASS_TARGET_PF) == {"low", "normal", "high"}


def test_beta_annual_is_on_the_annual_basis_and_consistent_with_pass():
    # The annual-basis index must equal Phi^-1(1 - pf_annual) and, being on the
    # same basis as the (annual) target, must exceed target_beta exactly when the
    # annual-Pf acceptance passes. This is what the UI headlines, so it must not
    # contradict the pass/fail badge the way the cumulative beta can.
    r = form_fatigue_reliability(_life(), 25.0, _PARAMS, safety_class="normal")
    assert r.beta_annual == pytest.approx(float(norm.ppf(1.0 - max(r.pf_annual, 1e-16))), rel=1e-9)
    assert (r.beta_annual >= r.target_beta) == r.passes
