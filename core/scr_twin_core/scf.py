"""Stress-concentration factors from weld geometry (DNV-RP-C203 App. 3).

The girth-weld hot-spot SCF is not a free number: a fabrication hi-lo
misalignment (axial eccentricity ``delta_m`` between the two pipe ends) induces a
local secondary bending that amplifies the membrane stress. DNV-RP-C203 App. 3.1
gives, for equal wall thickness,

    SCF_misalignment = 1 + 3 * (delta_m / t) * exp(-sqrt(t / D))

which multiplies the detail (geometric) SCF to give the effective hot-spot SCF.
With ``delta_m = 0`` the factor is 1, so a run with no specified misalignment is
unchanged.

Reference
---------
DNV-RP-C203 (2016/2021) Sec. 3.3 and App. 3.1 (fabrication-tolerance /
eccentricity stress-concentration factors for girth welds).
"""

from __future__ import annotations

import math


def misalignment_scf(hi_lo_m: float, thickness_m: float, diameter_m: float) -> float:
    """Axial-misalignment (hi-lo eccentricity) SCF at a girth weld (DNV App. 3.1).

    ``hi_lo_m`` is the eccentricity between the two pipe ends [m]; ``thickness_m``
    the wall thickness; ``diameter_m`` the outer diameter. Returns >= 1.0.
    """
    if thickness_m <= 0.0 or diameter_m <= 0.0:
        raise ValueError("thickness and diameter must be positive")
    if hi_lo_m < 0.0:
        raise ValueError("hi_lo must be non-negative")
    return 1.0 + 3.0 * (hi_lo_m / thickness_m) * math.exp(-math.sqrt(thickness_m / diameter_m))


def effective_scf(base_scf: float, hi_lo_m: float, thickness_m: float, diameter_m: float) -> float:
    """Effective hot-spot SCF = detail (geometric) SCF x misalignment SCF."""
    if base_scf < 1.0:
        raise ValueError("base_scf must be >= 1")
    return base_scf * misalignment_scf(hi_lo_m, thickness_m, diameter_m)
