"""Spectral tests: JONSWAP moments, Hs recovery, Welch PSD, spectrum fit."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.spectral import (
    fit_jonswap,
    jonswap,
    pierson_moskowitz,
    significant_from_m0,
    spectral_moments,
    welch_psd,
)


@pytest.fixture
def freq_grid():
    return np.linspace(0.0, 0.6, 6000)


@pytest.mark.parametrize("hs,tp", [(2.0, 8.0), (3.5, 10.0), (5.0, 12.0)])
def test_normalized_jonswap_recovers_hs_exactly(freq_grid, hs, tp):
    s = jonswap(freq_grid, hs, tp, gamma=3.3, normalize=True)
    m0 = spectral_moments(freq_grid, s, (0,))[0]
    assert significant_from_m0(m0) == pytest.approx(hs, rel=1e-4)


@pytest.mark.parametrize("hs,tp", [(2.0, 8.0), (4.0, 11.0)])
def test_physical_alpha_recovers_hs_approximately(freq_grid, hs, tp):
    # Un-normalised (Phillips alpha) spectrum: Hs recovered within a few percent.
    s = jonswap(freq_grid, hs, tp, gamma=3.3, normalize=False)
    m0 = spectral_moments(freq_grid, s, (0,))[0]
    assert significant_from_m0(m0) == pytest.approx(hs, rel=0.06)


def test_peak_at_inverse_tp(freq_grid):
    tp = 10.0
    s = jonswap(freq_grid, 3.0, tp, gamma=3.3)
    fpk = freq_grid[np.argmax(s)]
    assert fpk == pytest.approx(1.0 / tp, abs=2 * (freq_grid[1] - freq_grid[0]))


def test_pm_is_jonswap_gamma_one(freq_grid):
    a = pierson_moskowitz(freq_grid, 3.0, 10.0)
    b = jonswap(freq_grid, 3.0, 10.0, gamma=1.0)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_cauchy_schwarz_moment_inequality(freq_grid):
    s = jonswap(freq_grid, 3.0, 10.0, gamma=3.3)
    m = spectral_moments(freq_grid, s, (0, 2, 4))
    assert m[2] ** 2 <= m[0] * m[4] * (1 + 1e-9)


def test_fit_recovers_parameters(freq_grid):
    hs, tp, gamma = 3.5, 9.0, 2.5
    s = jonswap(freq_grid, hs, tp, gamma=gamma, normalize=True)
    fit = fit_jonswap(freq_grid, s)
    assert fit.hs == pytest.approx(hs, rel=1e-3)
    assert fit.tp == pytest.approx(tp, rel=0.05)
    assert fit.gamma == pytest.approx(gamma, abs=0.5)


def test_welch_psd_parseval_on_sinusoid():
    fs = 10.0
    t = np.arange(0, 600, 1 / fs)
    amp, f0 = 2.0, 0.1
    x = amp * np.sin(2 * np.pi * f0 * t)
    f, pxx = welch_psd(x, fs, nperseg=1024)
    # Integral of PSD ~ signal variance = amp^2 / 2
    var_from_psd = np.trapezoid(pxx, f)
    assert var_from_psd == pytest.approx(amp**2 / 2, rel=0.1)
    # Peak at the sinusoid frequency
    assert f[np.argmax(pxx)] == pytest.approx(f0, abs=0.02)


def test_welch_rejects_bad_input():
    with pytest.raises(ValueError):
        welch_psd([1.0], fs=10.0)
    with pytest.raises(ValueError):
        welch_psd([1.0, 2.0, 3.0], fs=0.0)
