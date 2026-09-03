"""Girth-weld misalignment SCF tests (DNV-RP-C203 App. 3)."""

from __future__ import annotations

import math

import pytest

from scr_twin_core.config import RiserConfig
from scr_twin_core.scf import effective_scf, misalignment_scf


def test_zero_misalignment_is_unity():
    assert misalignment_scf(0.0, 0.02, 0.32) == 1.0
    assert effective_scf(1.15, 0.0, 0.02, 0.32) == pytest.approx(1.15)


def test_reference_value():
    # delta_m=1 mm, t=20.6 mm, D=324 mm -> 1 + 3*(1/20.6)*exp(-sqrt(20.6/324))
    scf = misalignment_scf(1e-3, 20.6e-3, 324e-3)
    hand = 1.0 + 3.0 * (1e-3 / 20.6e-3) * math.exp(-math.sqrt(20.6e-3 / 324e-3))
    assert scf == pytest.approx(hand, rel=1e-12)
    assert 1.10 < scf < 1.13


def test_monotonic_in_misalignment():
    a = misalignment_scf(0.5e-3, 20e-3, 320e-3)
    b = misalignment_scf(2.0e-3, 20e-3, 320e-3)
    assert 1.0 < a < b


def test_effective_is_product():
    m = misalignment_scf(1.5e-3, 20e-3, 320e-3)
    assert effective_scf(1.2, 1.5e-3, 20e-3, 320e-3) == pytest.approx(1.2 * m)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        misalignment_scf(-1e-3, 20e-3, 320e-3)
    with pytest.raises(ValueError):
        misalignment_scf(1e-3, 0.0, 320e-3)
    with pytest.raises(ValueError):
        effective_scf(0.5, 1e-3, 20e-3, 320e-3)  # base < 1


def test_riser_config_effective_scf():
    r = RiserConfig.reference_scr().model_copy(update={"hi_lo_misalignment": 2e-3})
    assert r.misalignment_scf() > 1.0
    assert r.effective_scf() == pytest.approx(r.scf * r.misalignment_scf())
    # default (no misalignment) leaves the SCF unchanged
    r0 = RiserConfig.reference_scr()
    assert r0.effective_scf() == pytest.approx(r0.scf)
