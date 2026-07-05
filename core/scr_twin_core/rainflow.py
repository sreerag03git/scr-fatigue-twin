"""Rainflow cycle counting (ASTM E1049-85, four-point method).

This is an in-house implementation of the four-point rainflow algorithm used to
decompose an irregular stress history into closed hysteresis loops (cycles).
It is cross-checked against the third-party ``rainflow`` and ``fatpack``
packages and against hand-verifiable sequences in the test-suite
(``tests/test_rainflow.py``).

Reference
---------
ASTM E1049-85 (2017) "Standard Practices for Cycle Counting in Fatigue
Analysis", Sec. 5.4.4 (rainflow counting) and the equivalent four-point
formulation in I. Rychlik (1987), "A new definition of the rainflow cycle
counting method", Int. J. Fatigue 9(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CycleCount:
    """Result of a rainflow count as parallel arrays.

    Attributes
    ----------
    ranges:
        Peak-to-peak stress (or strain) range of each counted cycle.
    means:
        Mean level of each counted cycle.
    counts:
        Multiplicity of each cycle: ``1.0`` for a full closed loop, ``0.5`` for
        a residual half cycle.
    """

    ranges: NDArray[np.float64]
    means: NDArray[np.float64]
    counts: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.ranges.shape[0])


def find_reversals(series: ArrayLike) -> NDArray[np.float64]:
    """Return the turning points (reversals) of ``series``.

    Consecutive equal values are collapsed and only local extrema are kept, with
    the first and last samples always retained. This is the pre-processing step
    required by ASTM E1049 before cycles are extracted.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    if x.size == 0:
        return np.empty(0, dtype=np.float64)

    # Collapse runs of equal values.
    keep = np.ones(x.size, dtype=bool)
    keep[1:] = x[1:] != x[:-1]
    x = x[keep]
    if x.size <= 2:
        return x.copy()

    # A point is a reversal when the slope changes sign across it.
    dx = np.diff(x)
    slope_change = dx[1:] * dx[:-1] < 0.0
    is_reversal = np.concatenate(([True], slope_change, [True]))
    return x[is_reversal]


def count_cycles(series: ArrayLike) -> CycleCount:
    """Rainflow-count ``series`` with the ASTM E1049 four-point method.

    A sliding window over the reversal sequence extracts a full cycle whenever
    the inner range of four consecutive reversals is no larger than either
    adjacent (outer) range. Whatever remains after the pass (the *residue*) is
    reported as half cycles.

    Returns
    -------
    CycleCount
        Parallel ``ranges``/``means``/``counts`` arrays. Empty when fewer than
        two reversals exist.
    """
    reversals = find_reversals(series)
    n = reversals.size
    if n < 2:
        return CycleCount(
            np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64)
        )

    ranges: list[float] = []
    means: list[float] = []
    counts: list[float] = []

    stack: list[float] = []
    for value in reversals:
        stack.append(float(value))
        # Extract full cycles from the tail of the stack (four-point rule).
        while len(stack) >= 4:
            s1, s2, s3, s4 = stack[-4], stack[-3], stack[-2], stack[-1]
            r_outer_left = abs(s1 - s2)
            r_inner = abs(s2 - s3)
            r_outer_right = abs(s3 - s4)
            if r_inner <= r_outer_left and r_inner <= r_outer_right:
                ranges.append(r_inner)
                means.append(0.5 * (s2 + s3))
                counts.append(1.0)
                # Remove the inner pair (s2, s3), keeping s1 and s4 adjacent.
                del stack[-3:-1]
            else:
                break

    # Residue -> half cycles between successive remaining reversals.
    for a, b in zip(stack[:-1], stack[1:], strict=False):
        ranges.append(abs(a - b))
        means.append(0.5 * (a + b))
        counts.append(0.5)

    return CycleCount(
        np.asarray(ranges, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def range_histogram(
    cycles: CycleCount, bin_edges: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Bin counted cycles by range into ``bin_edges``.

    Returns an array of length ``len(bin_edges) - 1`` holding the summed cycle
    counts (halves included) whose range falls in each bin.
    """
    hist, _ = np.histogram(cycles.ranges, bins=bin_edges, weights=cycles.counts)
    return hist.astype(np.float64)
