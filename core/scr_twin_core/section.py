"""Pipe cross-section geometry and submerged-weight bookkeeping.

A steel catenary riser is a thin-walled circular tube. This module derives the
section properties (area, second moment of area, section modulus, bending
stiffness) and the effective submerged weight per unit length used by the
catenary and stress modules.

Sign / unit convention: SI throughout. Bending stress from a moment ``M`` is
``sigma = M / Z = M (D/2) / I``; from a curvature ``kappa`` it is
``sigma = E (D/2) kappa`` with ``M = E I kappa``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import E_STEEL, RHO_SEAWATER, RHO_STEEL, G


@dataclass(frozen=True)
class PipeSection:
    """Circular steel pipe section.

    Parameters
    ----------
    outer_diameter:
        Steel outer diameter ``D`` [m].
    wall_thickness:
        Wall thickness ``t`` [m].
    youngs_modulus:
        Young's modulus ``E`` [Pa] (defaults to line-pipe steel).
    """

    outer_diameter: float
    wall_thickness: float
    youngs_modulus: float = E_STEEL

    def __post_init__(self) -> None:
        if self.outer_diameter <= 0.0:
            raise ValueError("outer_diameter must be positive")
        if not (0.0 < self.wall_thickness < self.outer_diameter / 2.0):
            raise ValueError("wall_thickness must satisfy 0 < t < D/2")

    @property
    def inner_diameter(self) -> float:
        return self.outer_diameter - 2.0 * self.wall_thickness

    @property
    def steel_area(self) -> float:
        """Cross-sectional steel area ``A_s`` [m^2]."""
        return math.pi / 4.0 * (self.outer_diameter**2 - self.inner_diameter**2)

    @property
    def bore_area(self) -> float:
        """Internal (bore) area ``A_i`` [m^2]."""
        return math.pi / 4.0 * self.inner_diameter**2

    @property
    def second_moment_area(self) -> float:
        """Second moment of area ``I`` [m^4] about a diameter."""
        return math.pi / 64.0 * (self.outer_diameter**4 - self.inner_diameter**4)

    @property
    def section_modulus(self) -> float:
        """Elastic section modulus ``Z = I / (D/2)`` [m^3]."""
        return self.second_moment_area / (self.outer_diameter / 2.0)

    @property
    def bending_stiffness(self) -> float:
        """Bending stiffness ``E I`` [N m^2]."""
        return self.youngs_modulus * self.second_moment_area

    def stress_from_moment(self, moment: float) -> float:
        """Outer-fibre bending stress ``sigma = M / Z`` [Pa]."""
        return moment / self.section_modulus

    def stress_from_curvature(self, curvature: float) -> float:
        """Outer-fibre bending stress ``sigma = E (D/2) kappa`` [Pa]."""
        return self.youngs_modulus * (self.outer_diameter / 2.0) * curvature

    def moment_from_curvature(self, curvature: float) -> float:
        """Bending moment ``M = E I kappa`` [N m]."""
        return self.bending_stiffness * curvature


def submerged_weight(
    section: PipeSection,
    *,
    contents_density: float = 0.0,
    coating_thickness: float = 0.0,
    coating_density: float = 0.0,
    seawater_density: float = RHO_SEAWATER,
    steel_density: float = RHO_STEEL,
    g: float = G,
) -> float:
    """Effective submerged weight per unit length ``w`` [N/m].

    ``w = (W_steel + W_coating + W_contents - B) `` where buoyancy ``B`` uses the
    total displaced volume including any coating layer. A negative result (net
    buoyant) is returned as-is so callers can detect an invalid configuration.
    """
    total_outer_d = section.outer_diameter + 2.0 * coating_thickness

    w_steel = section.steel_area * steel_density * g
    w_contents = section.bore_area * contents_density * g
    coating_area = math.pi / 4.0 * (total_outer_d**2 - section.outer_diameter**2)
    w_coating = coating_area * coating_density * g
    displaced_area = math.pi / 4.0 * total_outer_d**2
    buoyancy = displaced_area * seawater_density * g
    return w_steel + w_contents + w_coating - buoyancy
