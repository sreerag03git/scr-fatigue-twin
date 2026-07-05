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
