"""Calibrated synthetic MRU motion generator (DEMO / FALLBACK ONLY).

Every value derived from this module is SYNTHETIC and must be badged as such in
the UI. Real MRU ingestion is the primary workflow; this exists to demonstrate
the pipeline offline and to seed reproducible tests.

Method (project spec 4.1): a wave elevation spectrum (JONSWAP) is discretised
into >=100 components; each is given a random phase and a small random frequency
jitter (the random-phase/random-frequency method, which avoids an artificially
periodic record), shaped by a vessel motion RAO, and summed into a motion time
history. Deterministic for a given ``seed``.

    x(t) = sum_i RAO(f_i) * a_i * cos(2 pi f_i t + phi_i),   a_i = sqrt(2 S(f_i) df_i)

References
----------
- Deterministic spectral amplitude method: DNV-RP-C205 Sec. 3.3;
  Tucker, Challenor & Carter (1984), Applied Ocean Research 6(2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .spectral import jonswap


@dataclass(frozen=True)
class SyntheticMotion:
    """A synthetic MRU motion record. ``is_synthetic`` is always ``True``."""

    time: NDArray[np.float64]
    heave: NDArray[np.float64]
    fs: float
    hs: float
    tp: float
    gamma: float
    seed: int
    is_synthetic: bool = True


def default_heave_rao(
    f: NDArray[np.float64], *, natural_period: float = 12.0, damping: float = 0.2
) -> NDArray[np.float64]:
    """Illustrative floating-unit heave RAO magnitude [m/m].

    A documented, clearly-illustrative 2nd-order shape with unit DC gain (the
    body contours long waves) rolling off past the heave natural frequency. NOT
    a real vessel RAO - supply a measured RAO table for project work.
    """
    fn = 1.0 / natural_period
    r = f / fn
    return 1.0 / np.sqrt((1.0 - r**2) ** 2 + (2.0 * damping * r) ** 2)


def synthetic_mru_motion(
    *,
    duration: float,
    fs: float,
    hs: float,
    tp: float,
    gamma: float = 3.3,
    seed: int,
    n_components: int = 200,
    f_low: float = 0.02,
    f_high: float = 0.5,
    rao: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> SyntheticMotion:
    """Generate a deterministic synthetic heave motion record.

    Parameters
    ----------
    duration, fs:
        Record length [s] and sample rate [Hz].
    hs, tp, gamma:
        JONSWAP sea-state parameters for the underlying wave elevation.
    seed:
        RNG seed - identical seed/config gives byte-identical output.
    n_components:
        Number of spectral components (>= 100 per spec).
    rao:
        Optional callable mapping frequency [Hz] -> RAO magnitude [m/m]. Defaults
        to :func:`default_heave_rao` (illustrative).
    """
    if duration <= 0.0 or fs <= 0.0:
        raise ValueError("duration and fs must be positive")
    if n_components < 100:
        raise ValueError("n_components must be >= 100 (spec 4.1)")
    if not (0.0 < f_low < f_high):
        raise ValueError("require 0 < f_low < f_high")

    rng = np.random.default_rng(seed)
    rao_fn = rao if rao is not None else default_heave_rao

    edges = np.linspace(f_low, f_high, n_components + 1)
    df = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    jitter = (rng.random(n_components) - 0.5) * df
    freqs = centers + jitter

    s = jonswap(freqs, hs, tp, gamma=gamma, normalize=True)
    amps = np.sqrt(2.0 * s * df)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    gains = np.asarray(rao_fn(freqs), dtype=np.float64)

    t = np.arange(0.0, duration, 1.0 / fs)
    # x(t) = sum_i g_i a_i cos(2 pi f_i t + phi_i); vectorised as a matrix product.
    phase_matrix = 2.0 * np.pi * np.outer(t, freqs) + phases
    heave = (np.cos(phase_matrix) * (gains * amps)).sum(axis=1)

    return SyntheticMotion(
        time=t.astype(np.float64),
        heave=heave.astype(np.float64),
        fs=fs,
        hs=hs,
        tp=tp,
        gamma=gamma,
        seed=seed,
    )
