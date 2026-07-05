"""Synthetic MRU generator tests: determinism, spectrum recovery, badging."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.spectral import significant_from_m0, spectral_moments, welch_psd
from scr_twin_core.synthetic import synthetic_mru_motion


def test_deterministic_from_seed():
    kw = dict(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=7)
    a = synthetic_mru_motion(**kw)
    b = synthetic_mru_motion(**kw)
    np.testing.assert_array_equal(a.heave, b.heave)
    assert a.is_synthetic is True


def test_different_seed_differs():
    a = synthetic_mru_motion(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=1)
    b = synthetic_mru_motion(duration=600.0, fs=4.0, hs=3.0, tp=10.0, seed=2)
    assert not np.array_equal(a.heave, b.heave)


def test_shape_and_rate():
    m = synthetic_mru_motion(duration=300.0, fs=4.0, hs=3.0, tp=10.0, seed=3)
    assert m.fs == 4.0
    assert m.time.size == m.heave.size
    assert m.time.size == pytest.approx(300.0 * 4.0, rel=0.01)


def test_unit_rao_recovers_wave_hs():
    # With a flat unit RAO the motion IS the wave elevation, so 4*sqrt(m0) ~ Hs.
    hs = 3.5
    m = synthetic_mru_motion(
        duration=6000.0, fs=4.0, hs=hs, tp=11.0, seed=11, n_components=400,
        rao=lambda f: np.ones_like(f),
    )
    f, pxx = welch_psd(m.heave, m.fs, nperseg=4096)
    m0 = spectral_moments(f, pxx, (0,))[0]
    assert significant_from_m0(m0) == pytest.approx(hs, rel=0.1)


@pytest.mark.parametrize("bad", [dict(n_components=50), dict(duration=-1.0), dict(f_low=0.6)])
def test_invalid_inputs_raise(bad):
    kw = dict(duration=100.0, fs=4.0, hs=3.0, tp=10.0, seed=1)
    kw.update(bad)
    with pytest.raises(ValueError):
        synthetic_mru_motion(**kw)
