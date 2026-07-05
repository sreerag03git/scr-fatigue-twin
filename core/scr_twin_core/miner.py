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
from .sn import SNCurve, cycles_to_failure

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


def miner_damage(
    cycles: CycleCount,
    curve: SNCurve,
    *,
    thickness_m: float | None = None,
) -> float:
    """Accumulated Miner damage from ``cycles`` on ``curve``.

    Half-cycles contribute their fractional count. Non-damaging cycles (range at
    or below the curve so that ``N -> inf``) contribute zero.
    """
    if len(cycles) == 0:
        return 0.0
    n_fail = cycles_to_failure(cycles.ranges, curve, thickness_m=thickness_m)
    contributions = np.where(np.isfinite(n_fail), cycles.counts / n_fail, 0.0)
    return float(np.sum(contributions))


def block_damage(
    cycles: CycleCount,
    curve: SNCurve,
    block_seconds: float,
    *,
    thickness_m: float | None = None,
) -> DamageResult:
    """Damage, annualised rate and deterministic life for one time block."""
    if block_seconds <= 0.0:
        raise ValueError("block_seconds must be positive")
    damage = miner_damage(cycles, curve, thickness_m=thickness_m)
    rate = damage / block_seconds * SECONDS_PER_YEAR
    life = np.inf if rate <= 0.0 else 1.0 / rate
    return DamageResult(
        damage=damage,
        block_seconds=block_seconds,
        damage_rate_per_year=rate,
        life_years=float(life),
    )
