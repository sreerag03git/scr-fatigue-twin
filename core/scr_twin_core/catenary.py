"""Static catenary geometry for a steel catenary riser (SCR).

For a plain (non-lazy-wave) SCR the static shape is the classical inextensible
catenary. With the seabed touchdown point (TDP) at the origin and the line
tangent horizontal there, the shape is

    y(x) = a (cosh(x/a) - 1),   a = H / w

where ``H`` is the horizontal tension (constant along the line) and ``w`` the
submerged weight per unit length. Useful identities used below:

    curvature(x) = 1 / (a cosh^2(x/a));   kappa_TDP = 1/a = w/H
    tangent angle theta(x):  tan(theta) = sinh(x/a)
    arc length from TDP s(x) = a sinh(x/a)
    tension T(x) = H cosh(x/a) = H + w*y(x)

Given the water depth ``d`` (vertical TDP->hang-off distance) and the hang-off
tangent angle ``theta_top`` (from horizontal), the geometry closes in closed
form because ``cosh(x_top/a) = sec(theta_top)`` gives

    a = d cos(theta_top) / (1 - cos(theta_top)).

Reference
---------
Standard catenary mechanics; e.g. Bai & Bai, "Subsea Pipelines and Risers"
(2005), Ch. on SCR configuration; DNV-OS-F201 App. on catenary risers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlainCatenary:
    """Closed-form static solution of a plain SCR catenary.

    All quantities SI. ``top_angle`` is measured from the horizontal.
    """

    catenary_parameter: float  # a = H/w  [m]
    horizontal_tension: float  # H  [N]
    top_tension: float  # T at hang-off  [N]
    top_angle: float  # theta_top from horizontal  [rad]
    water_depth: float  # d  [m]
    submerged_weight: float  # w  [N/m]
    arc_length: float  # s from TDP to hang-off  [m]
    horizontal_span: float  # x from TDP to hang-off  [m]

    @property
    def tdp_curvature(self) -> float:
        """Curvature at the touchdown point ``kappa_TDP = 1/a`` [1/m]."""
        return 1.0 / self.catenary_parameter

    @property
    def top_angle_from_vertical(self) -> float:
        """Hang-off angle measured from vertical [rad]."""
        return math.pi / 2.0 - self.top_angle

    def shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vertical coordinate ``y(x)`` above the TDP [m]."""
        a = self.catenary_parameter
        return a * (np.cosh(x / a) - 1.0)

    def curvature(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Curvature ``kappa(x) = 1/(a cosh^2(x/a))`` [1/m]."""
        a = self.catenary_parameter
        return 1.0 / (a * np.cosh(x / a) ** 2)

    def tension(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Effective tension ``T(x) = H cosh(x/a)`` [N]."""
        a = self.catenary_parameter
        return self.horizontal_tension * np.cosh(x / a)


def solve_plain_catenary(
    water_depth: float,
    top_angle_deg: float,
    submerged_weight: float,
) -> PlainCatenary:
    """Solve the plain SCR catenary from water depth and hang-off angle.

    Parameters
    ----------
    water_depth:
        Vertical distance from TDP to the hang-off point ``d`` [m].
    top_angle_deg:
        Tangent angle at the hang-off, measured from the horizontal [deg].
        (For a steep SCR ~15-20 deg from vertical this is ~70-75 deg.)
    submerged_weight:
        Submerged weight per unit length ``w`` [N/m], must be positive.
    """
    if water_depth <= 0.0:
        raise ValueError("water_depth must be positive")
    if not (0.0 < top_angle_deg < 90.0):
        raise ValueError("top_angle_deg must be in (0, 90) measured from horizontal")
    if submerged_weight <= 0.0:
        raise ValueError("submerged_weight must be positive (net-buoyant SCR is invalid)")

    theta = math.radians(top_angle_deg)
    cos_t = math.cos(theta)
    a = water_depth * cos_t / (1.0 - cos_t)
    h = submerged_weight * a
    x_top = a * math.asinh(math.tan(theta))
    arc = a * math.tan(theta)
    t_top = h / cos_t
    return PlainCatenary(
        catenary_parameter=a,
        horizontal_tension=h,
        top_tension=t_top,
        top_angle=theta,
        water_depth=water_depth,
        submerged_weight=submerged_weight,
        arc_length=arc,
        horizontal_span=x_top,
    )


def solve_plain_catenary_fixed_span(
    horizontal_span: float,
    water_depth: float,
    submerged_weight: float,
) -> PlainCatenary:
    """Solve the plain catenary for a *fixed horizontal span* and water depth.

    Unlike :func:`solve_plain_catenary` (which fixes the hang-off angle), this
    holds the horizontal offset ``X`` between the TDP and the hang-off fixed and
    solves the transcendental relation ``d = a (cosh(X/a) - 1)`` for the
    catenary parameter ``a``. This is the boundary condition used to derive the
    quasi-static TDP-moment sensitivity to vessel heave (vertical motion at
    roughly fixed horizontal offset).

    Solved with a bracketed Brent root find; the root is unique because the sag
    ``a(cosh(X/a)-1)`` is strictly monotonic in ``a``.
    """
    if horizontal_span <= 0.0:
        raise ValueError("horizontal_span must be positive")
    if water_depth <= 0.0:
        raise ValueError("water_depth must be positive")
    if submerged_weight <= 0.0:
        raise ValueError("submerged_weight must be positive")

    from scipy.optimize import brentq

    def sag(a: float) -> float:
        return a * (math.cosh(horizontal_span / a) - 1.0) - water_depth

    # Bracket: small a -> huge sag (>0); large a -> sag ~ X^2/(2a) -> <0 eventually.
    a_lo = horizontal_span / 100.0
    a_hi = max(horizontal_span**2 / (2.0 * water_depth), horizontal_span) * 10.0
    # Ensure sign change; expand if needed.
    while sag(a_hi) > 0.0:
        a_hi *= 2.0
    a = float(brentq(sag, a_lo, a_hi, xtol=1e-9, rtol=1e-12))

    theta = math.atan(math.sinh(horizontal_span / a))
    h = submerged_weight * a
    arc = a * math.sinh(horizontal_span / a)
    t_top = h * math.cosh(horizontal_span / a)
    return PlainCatenary(
        catenary_parameter=a,
        horizontal_tension=h,
        top_tension=t_top,
        top_angle=theta,
        water_depth=water_depth,
        submerged_weight=submerged_weight,
        arc_length=arc,
        horizontal_span=horizontal_span,
    )


def solve_lazy_wave_catenary(*args: object, **kwargs: object) -> PlainCatenary:
    """Lazy-wave (buoyancy-section) configuration - NOT YET IMPLEMENTED.

    A lazy-wave SCR requires a three-segment analysis (hang-off sag catenary,
    buoyant arch, and touchdown catenary) with continuity of tension and slope
    at the segment boundaries. That solver is planned but deliberately not
    faked here: callers must use :func:`solve_plain_catenary` until it lands.
    """
    raise NotImplementedError(
        "Lazy-wave catenary solver is not implemented yet. Use solve_plain_catenary, "
        "or import a precomputed transfer function (transfer.Route 2) for a lazy-wave riser."
    )
