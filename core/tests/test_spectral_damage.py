"""Spectral-damage tests and the cross-method agreement gate (project spec 5).

Dirlik, Tovo-Benasciutti, narrow-band and time-domain rainflow damage must agree
within a stated band on the same stress spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.rainflow import count_cycles
from scr_twin_core.sn import cycles_to_failure, get_curve
from scr_twin_core.spectral import spectral_moments
from scr_twin_core.spectral_damage import (
    dirlik_damage_rate,
    dirlik_damage_rate_curve,
    dirlik_range_pdf,
    narrowband_damage_rate,
    spectral_params,
    tovo_benasciutti_damage_rate,
)
from scr_twin_core.stress import random_phase_timeseries, rfft_frequencies

# Single-slope S-N used for the like-for-like comparison (class-D m1 branch).
SN_M = 3.0
SN_LOG_A = 12.164
A_BAR = 10.0**SN_LOG_A


def _gauss_psd(freqs, f0, sig_f, sigma_stress_mpa):
    """Gaussian-bump stress PSD [MPa^2/Hz] with target RMS stress."""
    shape = np.exp(-((freqs - f0) ** 2) / (2.0 * sig_f**2))
    m0_shape = np.trapezoid(shape, freqs)
    return shape * (sigma_stress_mpa**2) / m0_shape


def test_dirlik_pdf_integrates_to_one():
    fs, n = 2.0, 2**16
    freqs = rfft_frequencies(n, fs)
    psd = _gauss_psd(freqs, 0.1, 0.02, 20.0)
    moments = spectral_moments(freqs, psd, (0, 1, 2, 4))
    s = np.linspace(0.0, 300.0, 20000)
    pdf = dirlik_range_pdf(s, moments, stress_to_mpa=1.0)
    assert np.trapezoid(pdf, s) == pytest.approx(1.0, abs=1e-3)


def test_dirlik_closed_form_matches_numeric_expectation():
    fs, n = 2.0, 2**16
    freqs = rfft_frequencies(n, fs)
    psd = _gauss_psd(freqs, 0.1, 0.02, 20.0)
    moments = spectral_moments(freqs, psd, (0, 1, 2, 4))

    s = np.linspace(0.0, 400.0, 40000)
    pdf = dirlik_range_pdf(s, moments, stress_to_mpa=1.0)
    e_sm_numeric = np.trapezoid(s**SN_M * pdf, s)

    p = spectral_params(moments)
    rate = dirlik_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)
    e_sm_closed = rate * A_BAR / p.nup
    assert e_sm_closed == pytest.approx(e_sm_numeric, rel=1e-2)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cross_method_agreement(seed):
    # Same spectrum -> time-domain rainflow vs Dirlik/TB/NB damage.
    fs, n = 2.0, 2**20
    freqs = rfft_frequencies(n, fs)
    psd = _gauss_psd(freqs, 0.1, 0.02, 20.0)  # ~narrow-band, sigma = 20 MPa
    moments = spectral_moments(freqs, psd, (0, 1, 2, 4))
    duration = n / fs

    x = random_phase_timeseries(freqs, psd, fs=fs, n_samples=n, seed=seed)  # MPa
    cy = count_cycles(x)
    d_time = float(np.sum(cy.counts * cy.ranges**SN_M) / A_BAR)
    rate_time = d_time / duration

    rate_dk = dirlik_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)
    rate_tb = tovo_benasciutti_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)
    rate_nb = narrowband_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)

    # Dirlik is calibrated to rainflow: tight. TB close. NB conservative (upper).
    assert rate_dk == pytest.approx(rate_time, rel=0.12)
    assert rate_tb == pytest.approx(rate_time, rel=0.15)
    assert 0.9 <= rate_nb / rate_time <= 1.6


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_two_slope_integrator_matches_time_domain(seed):
    # Dirlik integrated against the real two-slope DNV-D curve vs time-domain
    # rainflow + Miner on the same curve. Stress in MPa (stress_to_mpa=1).
    fs, n = 2.0, 2**20
    freqs = rfft_frequencies(n, fs)
    psd = _gauss_psd(freqs, 0.1, 0.02, 25.0)  # sigma ~ 25 MPa -> exercises both branches
    moments = spectral_moments(freqs, psd, (0, 1, 2, 4))
    duration = n / fs
    curve = get_curve("D")

    x = random_phase_timeseries(freqs, psd, fs=fs, n_samples=n, seed=seed)  # MPa
    cy = count_cycles(x)
    n_fail = cycles_to_failure(cy.ranges * 1e6, curve)  # ranges MPa -> Pa
    d_time = float(np.sum(cy.counts / n_fail))
    rate_time = d_time / duration

    rate_spec = dirlik_damage_rate_curve(moments, curve, stress_to_mpa=1.0)
    assert rate_spec == pytest.approx(rate_time, rel=0.15)


def test_narrowband_limit_methods_converge():
    # Very narrow bump -> irregularity factor ~1 -> all methods coincide.
    fs, n = 2.0, 2**16
    freqs = rfft_frequencies(n, fs)
    psd = _gauss_psd(freqs, 0.1, 0.004, 20.0)
    moments = spectral_moments(freqs, psd, (0, 1, 2, 4))
    assert spectral_params(moments).alpha2 > 0.97

    nb = narrowband_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)
    dk = dirlik_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)
    tb = tovo_benasciutti_damage_rate(moments, sn_m=SN_M, sn_log_a=SN_LOG_A, stress_to_mpa=1.0)
    assert dk == pytest.approx(nb, rel=0.15)
    assert tb == pytest.approx(nb, rel=0.15)
