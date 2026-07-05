"""Full-chain pipeline tests: end-to-end run, determinism, provenance, env delta."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.config import AnalysisConfig, RiserConfig
from scr_twin_core.pipeline import run_full_analysis
from scr_twin_core.synthetic import synthetic_mru_motion


def _motion(seed: int = 5):
    return synthetic_mru_motion(duration=1800.0, fs=4.0, hs=3.5, tp=11.0, seed=seed)


def _config(seed: int = 0, env: bool = True) -> AnalysisConfig:
    cfg = AnalysisConfig(riser=RiserConfig.reference_scr(), seed=seed, n_monte_carlo=2000)
    cfg.environment.enabled = env
    return cfg


def test_full_chain_runs_and_reports():
    m = _motion()
    res = run_full_analysis(_config(), m.heave, m.fs, motion_is_synthetic=True)
    assert res.sea_state.hs > 0.0
    assert res.annual_damage_rate_time > 0.0
    assert 0.0 < res.deterministic_life_years < np.inf
    assert 0.0 < res.monte_carlo.p10 < res.monte_carlo.p50 < res.monte_carlo.p90
    s = res.summary()
    assert s["provenance"]["motion_is_synthetic"] is True
    # Default route is the illustrative reference H(f) (Route-2), not reduced-order.
    assert s["provenance"]["transfer_is_reduced_order"] is False


def test_analytic_route_is_flagged_reduced_order():
    m = _motion()
    cfg = _config()
    cfg.transfer.route = "analytic"
    res = run_full_analysis(cfg, m.heave, m.fs)
    assert res.provenance.transfer_is_reduced_order is True
    assert res.deterministic_life_years > 0.0


def test_reference_route_gives_realistic_life():
    # A moderate sea state should yield a credible (order 10^1-10^3 yr) life,
    # not the ~10^11 yr the reduced-order route under-predicts.
    m = synthetic_mru_motion(duration=1800.0, fs=4.0, hs=4.0, tp=11.0, seed=7)
    res = run_full_analysis(_config(env=False), m.heave, m.fs)
    assert 10.0 < res.deterministic_life_years < 5000.0


def test_determinism_same_seed_same_output():
    m = _motion()
    a = run_full_analysis(_config(seed=3), m.heave, m.fs)
    b = run_full_analysis(_config(seed=3), m.heave, m.fs)
    assert a.annual_damage_rate_time == b.annual_damage_rate_time
    np.testing.assert_array_equal(a.monte_carlo.life_years, b.monte_carlo.life_years)
    assert a.provenance.config_sha256 == b.provenance.config_sha256


def test_environment_correction_shortens_life():
    m = _motion()
    with_env = run_full_analysis(_config(env=True), m.heave, m.fs)
    without_env = run_full_analysis(_config(env=False), m.heave, m.fs)
    # Corrected life is shorter (higher damage rate) than standard DNV.
    assert with_env.deterministic_life_years < without_env.deterministic_life_years
    ratio = with_env.deterministic_life_years / without_env.deterministic_life_years
    assert with_env.environment_factor == pytest.approx(ratio, rel=1e-6)
    assert 0.61 <= with_env.environment_factor <= 0.70


def test_spectral_and_time_domain_agree():
    m = _motion()
    res = run_full_analysis(_config(env=False), m.heave, m.fs)
    # Both pathways now use the same two-slope DNV curve (Dirlik PDF integrated
    # against N(S) vs time-domain rainflow), so they agree within a factor.
    r = res.annual_damage_rate_spectral / res.annual_damage_rate_time
    assert 0.33 <= r <= 3.0, f"spectral/time ratio = {r:.3f}"


def test_too_short_motion_raises():
    with pytest.raises(ValueError):
        run_full_analysis(_config(), np.zeros(4), 4.0)
