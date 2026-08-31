"""Bayesian updating tests: 1/sqrt(T) contraction and halving by year 4."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.bayesian import BayesianRateEstimator

BLOCKS_PER_YEAR = 100
TRUE_RATE = 0.025  # 1/yr (design life 40 yr)
OBS_STD = 0.02


def _estimator_after_years(years: float, *, seed: int = 0) -> BayesianRateEstimator:
    est = BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD)
    rng = np.random.default_rng(seed)
    n = int(round(years * BLOCKS_PER_YEAR))
    obs = rng.normal(TRUE_RATE, OBS_STD, n)
    for y in obs:
        est.update_block(float(y))
    return est


def test_ci_halves_by_year_four():
    w1 = _estimator_after_years(1).posterior().ci90_width
    w4 = _estimator_after_years(4).posterior().ci90_width
    assert w4 / w1 == pytest.approx(0.5, abs=0.04)


def test_ci_width_scales_as_inverse_sqrt_T():
    years = np.array([1, 2, 3, 4, 6, 9], dtype=float)
    widths = np.array([_estimator_after_years(t).posterior().ci90_width for t in years])
    # width * sqrt(T) should be ~constant
    invariant = widths * np.sqrt(years)
    assert invariant.std() / invariant.mean() < 0.03


def test_posterior_mean_converges_to_truth():
    post = _estimator_after_years(10, seed=5).posterior()
    assert post.mean == pytest.approx(TRUE_RATE, abs=0.003)
    assert post.n_blocks == 1000


def test_remaining_life_maps_inversely():
    post = _estimator_after_years(5).posterior()
    low, med, high = post.remaining_life(accumulated_damage=0.0)
    # higher rate bound -> lower life bound
    assert low < med < high
    assert med == pytest.approx(1.0 / post.mean, rel=1e-9)


def test_accumulated_damage_reduces_remaining_life():
    post = _estimator_after_years(5).posterior()
    _, med_fresh, _ = post.remaining_life(0.0)
    _, med_spent, _ = post.remaining_life(0.5)
    assert med_spent == pytest.approx(0.5 * med_fresh, rel=1e-9)


def test_update_from_block_damage_equivalent():
    est = BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD)
    block_seconds = 1800.0
    sec_per_year = 365.25 * 24 * 3600
    damage = TRUE_RATE * block_seconds / sec_per_year
    est.update_from_block_damage(damage, block_seconds)
    assert est.posterior().n_blocks == 1


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        BayesianRateEstimator(prior_mean=0.02, block_obs_std=0.0)
    with pytest.raises(ValueError):
        BayesianRateEstimator(prior_mean=0.02, block_obs_std=0.02, obs_ar1=1.0)


# --- AR(1) effective-sample-size correction -------------------------------- #
def _feed(est: BayesianRateEstimator, obs: np.ndarray) -> BayesianRateEstimator:
    for y in obs:
        est.update_block(float(y))
    return est


def test_phi_zero_is_identical_to_iid():
    rng = np.random.default_rng(0)
    obs = rng.normal(TRUE_RATE, OBS_STD, 400)
    a = _feed(BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD), obs).posterior()
    b = _feed(BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD, obs_ar1=0.0),
              obs).posterior()
    assert a.mean == pytest.approx(b.mean, rel=1e-15)
    assert a.ci90_width == pytest.approx(b.ci90_width, rel=1e-15)


def test_ar1_inflates_ci_by_sqrt_vif():
    # In the data-dominated regime the AR(1) band is wider by sqrt((1+phi)/(1-phi)).
    rng = np.random.default_rng(1)
    obs = rng.normal(TRUE_RATE, OBS_STD, 6000)
    phi = 0.35
    w0 = _feed(BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD), obs).posterior().ci90_width
    wp = _feed(BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD, obs_ar1=phi),
               obs).posterior().ci90_width
    assert wp / w0 == pytest.approx(np.sqrt((1 + phi) / (1 - phi)), rel=0.02)  # ~1.441


def test_ar1_still_halves_by_year_four():
    # The AR(1) width factor is constant, so it cancels in the year-4/year-1
    # ratio: the honesty gate (CI halves by year 4) survives the correction.
    def width(years: float) -> float:
        est = BayesianRateEstimator(prior_mean=TRUE_RATE, block_obs_std=OBS_STD, obs_ar1=0.35)
        rng = np.random.default_rng(0)
        obs = rng.normal(TRUE_RATE, OBS_STD, int(years * BLOCKS_PER_YEAR))
        return _feed(est, obs).posterior().ci90_width
    assert width(4) / width(1) == pytest.approx(0.5, abs=0.04)
