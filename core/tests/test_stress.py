"""Stress-reconstruction tests: FFT filtering, PSD propagation, synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.section import PipeSection
from scr_twin_core.spectral import spectral_moments
from scr_twin_core.stress import (
    moment_history,
    random_phase_timeseries,
    rfft_frequencies,
    stress_from_moment,
    stress_psd_from_motion_psd,
)
from scr_twin_core.transfer import InterpolatedTransferFunction


@pytest.fixture
def section():
    return PipeSection(outer_diameter=0.3239, wall_thickness=0.0206)


def _flat_tf(n, fs, gain):
    freqs = rfft_frequencies(n, fs)
    mag = np.full(freqs.shape, gain)
    phase = np.zeros_like(freqs)
    return InterpolatedTransferFunction(freqs, mag, phase).evaluate(freqs)


def test_flat_transfer_scales_history():
    n, fs, gain = 4096, 4.0, 3.0e6
    t = np.arange(n) / fs
    x = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.cos(2 * np.pi * 0.05 * t)
    tf = _flat_tf(n, fs, gain)
    m = moment_history(x, fs, tf)
    np.testing.assert_allclose(m, gain * x, atol=1e-6 * gain)


def test_stress_from_moment(section):
    m = np.array([1.0e6, -2.0e6])
    s = stress_from_moment(m, section, scf=1.2)
    np.testing.assert_allclose(s, 1.2 * m / section.section_modulus)


def test_moment_history_requires_matching_grid():
    n, fs = 1024, 4.0
    x = np.zeros(n)
    tf = _flat_tf(2048, fs, 1.0)  # wrong length
    with pytest.raises(ValueError):
        moment_history(x, fs, tf)


def test_stress_psd_matches_formula(section):
    n, fs = 2048, 4.0
    tf = _flat_tf(n, fs, 2.0e6)
    motion_psd = np.linspace(0.1, 1.0, tf.freqs.size)
    out = stress_psd_from_motion_psd(motion_psd, tf, section, scf=1.5)
    expected = (1.5 / section.section_modulus) ** 2 * tf.magnitude**2 * motion_psd
    np.testing.assert_allclose(out, expected)


def test_random_phase_timeseries_recovers_variance():
    n, fs = 2**16, 2.0
    freqs = rfft_frequencies(n, fs)
    # Band-limited flat PSD in [0.08, 0.12] Hz.
    psd = np.where((freqs >= 0.08) & (freqs <= 0.12), 100.0, 0.0)
    m0 = spectral_moments(freqs, psd, (0,))[0]
    x = random_phase_timeseries(freqs, psd, fs=fs, n_samples=n, seed=5)
    assert np.var(x) == pytest.approx(m0, rel=0.1)
    assert np.mean(x) == pytest.approx(0.0, abs=0.05 * np.sqrt(m0))


def test_random_phase_deterministic():
    n, fs = 2**14, 2.0
    freqs = rfft_frequencies(n, fs)
    psd = np.where((freqs >= 0.08) & (freqs <= 0.12), 100.0, 0.0)
    a = random_phase_timeseries(freqs, psd, fs=fs, n_samples=n, seed=9)
    b = random_phase_timeseries(freqs, psd, fs=fs, n_samples=n, seed=9)
    np.testing.assert_array_equal(a, b)
