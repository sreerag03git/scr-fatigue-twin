"""Palmgren-Miner linear cumulative damage.

Damage from a set of counted cycles is ``D = sum_i n_i / N_i`` where ``n_i`` is
the count at range ``dsigma_i`` and ``N_i = N(dsigma_i)`` from the S-N curve.
A block that spans ``duration`` seconds yields a damage *rate* ``D/duration``
that annualises to ``D/duration * seconds_per_year``; the deterministic fatigue
life is the reciprocal of the annual rate.

Reference
---------
M.A. Miner (1945), "Cumulative damage in fatigue", J. Appl. Mech. 12(3);
DNV-RP-C203 Sec. 2.3 (accumulated fatigue damage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rainflow import CycleCount
from .sn import MeanStressModel, SNCurve, apply_mean_stress, cycles_to_failure

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class DamageResult:
    """Outcome of a Miner summation over one analysis block.

    Attributes
    ----------
    damage:
        Dimensionless accumulated damage for the block (``sum n_i / N_i``).
    block_seconds:
        Duration represented by the block.
    damage_rate_per_year:
        ``damage / block_seconds * SECONDS_PER_YEAR``.
    life_years:
        Deterministic fatigue life ``1 / damage_rate_per_year`` (``inf`` if the
        block is non-damaging).
    """

    damage: float
    block_seconds: float
    damage_rate_per_year: float
    life_years: float


def _effective_ranges(
    cycles: CycleCount,
    model: MeanStressModel,
    ultimate_strength_pa: float | None,
    static_mean_pa: float,
) -> np.ndarray:
    """Ranges to enter the S-N curve with, after any mean-stress correction.

    The effective mean per cycle is the rainflow mean plus the ``static_mean_pa``
    offset (the standing tension/bending stress the dynamic response rides on -
    the wave-induced mean alone is ~0 through the bending transfer). With
    ``NONE`` the ranges are returned unchanged.
    """
    if model is MeanStressModel.NONE:
        return cycles.ranges
    if ultimate_strength_pa is None or ultimate_strength_pa <= 0.0:
        raise ValueError("ultimate_strength_pa is required for a mean-stress correction")
    eff_mean = cycles.means + static_mean_pa
    return apply_mean_stress(cycles.ranges, eff_mean, model, ultimate_strength_pa)


def miner_damage(
    cycles: CycleCount,
    curve: SNCurve,
    *,
    thickness_m: float | None = None,
    mean_stress_model: MeanStressModel = MeanStressModel.NONE,
    ultimate_strength_pa: float | None = None,
    static_mean_pa: float = 0.0,
) -> float:
    """Accumulated Miner damage from ``cycles`` on ``curve``.

    Half-cycles contribute their fractional count. Non-damaging cycles (range at
    or below the curve so that ``N -> inf``) contribute zero. When
    ``mean_stress_model`` is not ``NONE`` each cycle's range is first mapped to an
    equivalent fully-reversed range using its (rainflow + static) mean stress.
    """
    if len(cycles) == 0:
        return 0.0
    ranges = _effective_ranges(cycles, mean_stress_model, ultimate_strength_pa, static_mean_pa)
    n_fail = cycles_to_failure(ranges, curve, thickness_m=thickness_m)
    contributions = np.where(np.isfinite(n_fail), cycles.counts / n_fail, 0.0)
    return float(np.sum(contributions))


def block_damage(
    cycles: CycleCount,
    curve: SNCurve,
    block_seconds: float,
    *,
    thickness_m: float | None = None,
    mean_stress_model: MeanStressModel = MeanStressModel.NONE,
    ultimate_strength_pa: float | None = None,
    static_mean_pa: float = 0.0,
) -> DamageResult:
    """Damage, annualised rate and deterministic life for one time block."""
    if block_seconds <= 0.0:
        raise ValueError("block_seconds must be positive")
    damage = miner_damage(
        cycles, curve, thickness_m=thickness_m,
        mean_stress_model=mean_stress_model,
        ultimate_strength_pa=ultimate_strength_pa,
        static_mean_pa=static_mean_pa,
    )
    rate = damage / block_seconds * SECONDS_PER_YEAR
    life = np.inf if rate <= 0.0 else 1.0 / rate
    return DamageResult(
        damage=damage,
        block_seconds=block_seconds,
        damage_rate_per_year=rate,
        life_years=float(life),
    )
