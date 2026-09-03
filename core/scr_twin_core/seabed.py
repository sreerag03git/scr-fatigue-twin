"""Seabed-riser interaction at the touchdown zone (elastic-foundation correction).

The plain catenary assumes a rigid seabed, so the touchdown-point (TDP) curvature
is the cable value ``kappa = 1/a``. A real seabed is compliant: the riser embeds
and the curvature is distributed over a boundary layer, which reduces the peak TDP
bending stress. Two length scales set the behaviour:

    bending boundary layer   lambda_b = sqrt(EI / H)          (rigid seabed)
    elastic-foundation length l_s      = (4 EI / k_v)^(1/4)   (soil spring k_v)

A reduced-order correction factor ``Cs = lambda_b / (lambda_b + l_s)`` scales the
rigid-seabed TDP curvature: ``Cs -> 1`` as ``k_v -> inf`` (rigid) and ``Cs < 1``
for soft soil. The rigid-seabed base case is therefore conservative; this module
exposes the soft-to-stiff fatigue-life sensitivity so the assumption is badged
rather than hidden. Fatigue damage scales as stress^m, so the life scales as
``Cs^-m``.

Reference
---------
Pesce, Aranha & Martins (1998), "The touchdown-point boundary-layer solution";
Lenci & Callegari (2005), heavy cable on an elastic seabed; DNV-OS-F201.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Illustrative vertical seabed-stiffness bracket [Pa = N/m per m of riser].
K_V_SOFT_CLAY = 50.0e3
K_V_STIFF_SAND = 5000.0e3


def boundary_layer_length(bending_stiffness: float, tdp_tension: float) -> float:
    """Bending boundary-layer length ``lambda_b = sqrt(EI / H)`` [m]."""
    if bending_stiffness <= 0.0 or tdp_tension <= 0.0:
        raise ValueError("bending_stiffness and tdp_tension must be positive")
    return float(np.sqrt(bending_stiffness / tdp_tension))


def soil_length(bending_stiffness: float, k_v: float) -> float:
    """Elastic-foundation characteristic length ``l_s = (4 EI / k_v)^(1/4)`` [m]."""
    if bending_stiffness <= 0.0 or k_v <= 0.0:
        raise ValueError("bending_stiffness and k_v must be positive")
    return float((4.0 * bending_stiffness / k_v) ** 0.25)


def seabed_correction(bending_stiffness: float, tdp_tension: float, k_v: float) -> float:
    """TDP-curvature correction factor ``Cs`` for a compliant seabed (in (0, 1])."""
    lam_b = boundary_layer_length(bending_stiffness, tdp_tension)
    l_s = soil_length(bending_stiffness, k_v)
    return float(lam_b / (lam_b + l_s))


@dataclass(frozen=True)
class SeabedSensitivity:
    """Soft-to-stiff seabed fatigue-life sensitivity about the rigid base case."""

    k_v: NDArray[np.float64]           # seabed stiffness grid [Pa]
    correction: NDArray[np.float64]    # Cs per stiffness
    life_factor: NDArray[np.float64]   # life multiplier Cs^-m
    life_years: NDArray[np.float64]    # base_life * Cs^-m
    base_life_years: float             # rigid-seabed (Cs = 1) life
    life_soft: float                   # life at the soft-clay bound
    life_stiff: float                  # life at the stiff-sand bound


def seabed_sensitivity(
    bending_stiffness: float, tdp_tension: float, base_life_years: float,
    *, sn_slope_m: float = 3.0, k_v_min: float = K_V_SOFT_CLAY, k_v_max: float = K_V_STIFF_SAND,
    n: int = 40,
) -> SeabedSensitivity:
    """Fatigue-life sensitivity across a soft-to-stiff seabed stiffness range.

    The rigid-seabed life is the base; a seabed of stiffness ``k_v`` scales the TDP
    curvature by ``Cs`` and hence the life by ``Cs^-m`` (damage ~ stress^m).
    """
    k_v = np.geomspace(k_v_min, k_v_max, n)
    cs = np.array([seabed_correction(bending_stiffness, tdp_tension, k) for k in k_v])
    life_factor = cs ** (-sn_slope_m)
    life = base_life_years * life_factor if np.isfinite(base_life_years) else np.full_like(cs, np.inf)
    return SeabedSensitivity(
        k_v=k_v.astype(np.float64), correction=cs.astype(np.float64),
        life_factor=life_factor.astype(np.float64), life_years=life.astype(np.float64),
        base_life_years=base_life_years,
        life_soft=float(life[0]), life_stiff=float(life[-1]),
    )
