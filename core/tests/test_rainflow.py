"""Rainflow tests: hand-verified sequences + cross-check vs. ``rainflow``/``fatpack``."""

from __future__ import annotations

import numpy as np
import pytest

from scr_twin_core.rainflow import CycleCount, count_cycles, find_reversals

rainflow_pkg = pytest.importorskip("rainflow")
fatpack = pytest.importorskip("fatpack")


def _damage_metric(cycles: CycleCount, m: float) -> float:
    return float(np.sum(cycles.counts * cycles.ranges**m))


def test_reversals_collapse_and_extrema():
    x = [0, 0, 1, 1, 1, 0, 2, 2, -1]
    rev = find_reversals(x)
    # duplicates removed, turning points kept, endpoints retained
    assert list(rev) == [0.0, 1.0, 0.0, 2.0, -1.0]


def test_single_triangle_is_two_half_cycles():
    # A lone peak has no closable loop: its residue is two range-1 half cycles
    # (0->1 and 1->0). Total count 1.0 == one equivalent full cycle of range 1,
    # which is what the reference ``rainflow`` package reports in aggregate.
    cy = count_cycles([0.0, 1.0, 0.0])
    assert float(np.sum(cy.counts)) == pytest.approx(1.0)
    np.testing.assert_allclose(cy.ranges, 1.0)


def test_full_cycle_extraction():
    # Reversals [0,2,1,3]: inner range |2-1|=1 <= both outer ranges (2 and 2),
    # so the (2,1) pair closes as a full cycle of range 1.
    cy = count_cycles([0.0, 2.0, 1.0, 3.0])
    full = cy.ranges[cy.counts == 1.0]
    assert 1.0 in np.round(full, 6)


def test_empty_and_flat_signals_do_not_crash():
    assert len(count_cycles([])) == 0
    assert len(count_cycles([5.0])) == 0
    assert len(count_cycles([3.0, 3.0, 3.0])) == 0


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_cross_check_rainflow_package_total_counts(seed):
    rng = np.random.default_rng(seed)
    signal = np.cumsum(rng.standard_normal(4000))
    mine = count_cycles(signal)

    # rainflow package aggregated (range, count) pairs
    pkg_pairs = rainflow_pkg.count_cycles(signal)
    pkg_total = sum(c for _, c in pkg_pairs)

    assert float(np.sum(mine.counts)) == pytest.approx(pkg_total, rel=1e-9)


@pytest.mark.parametrize("m", [3.0, 5.0])
@pytest.mark.parametrize("seed", [0, 3, 99])
def test_cross_check_damage_metric(seed, m):
    rng = np.random.default_rng(seed)
    signal = np.cumsum(rng.standard_normal(6000))

    mine = _damage_metric(count_cycles(signal), m)

    pkg = sum(c * rng_**m for rng_, c in rainflow_pkg.count_cycles(signal))
    assert mine == pytest.approx(pkg, rel=1e-9)


@pytest.mark.parametrize("seed", [2, 11, 55])
def test_cross_check_fatpack_interior_cycles(seed):
    # fatpack.find_reversals bins the signal into amplitude classes (default
    # k=64) and its residue is handled by a different (repetition) convention,
    # so raw damage totals are not directly comparable. To compare the *cycle-
    # extraction algorithm* itself, feed fatpack our own (unbinned) reversals
    # and compare the interior closed cycles: they must agree exactly.
    rng = np.random.default_rng(seed)
    signal = np.cumsum(rng.standard_normal(8000))

    mine = count_cycles(signal)
    my_full = np.sort(mine.ranges[mine.counts == 1.0])

    reversals = find_reversals(signal)
    fp_cycles, _residue = fatpack.find_rainflow_cycles(reversals)
    fp_full = np.sort(np.abs(fp_cycles[:, 1] - fp_cycles[:, 0]))

    assert my_full.size == fp_full.size
    np.testing.assert_allclose(my_full, fp_full, rtol=1e-12, atol=1e-12)
