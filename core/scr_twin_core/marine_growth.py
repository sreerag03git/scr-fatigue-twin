"""Marine growth (biofouling) on the riser (DNV-RP-C205 Sec. 4.8).

A layer of marine growth of thickness ``t_mg`` and density ``rho_mg`` builds up on
the riser. It does not change the steel section that carries bending, but it:

  - increases the hydrodynamic diameter ``D_eff = D + 2 t_mg`` (more drag, more
    added mass, and a lower vortex-shedding frequency ``f_s = St U / D_eff`` that
    shifts VIV lock-in);
  - adds structural mass ``rho_mg * A_annulus`` (lowers the natural frequencies);
  - adds net submerged weight ``(rho_mg - rho_sw) * A_annulus * g``.

This module gives those quantities; the VIV screening consumes ``D_eff`` and the
added mass. Thickness/density are project inputs (DNV-RP-C205 gives regional
defaults, e.g. 50-100 mm and ~1100-1400 kg/m^3).

Reference
---------
DNV-RP-C205 Sec. 4.8 (marine growth thickness, density, roughness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import G, RHO_SEAWATER


@dataclass(frozen=True)
class MarineGrowth:
    """Marine-growth layer properties (uniform representative thickness)."""

    thickness_m: float = 0.0
    density_kg_m3: float = 1300.0

    def __post_init__(self) -> None:
        if self.thickness_m < 0.0:
            raise ValueError("thickness_m must be non-negative")
        if self.density_kg_m3 <= 0.0:
            raise ValueError("density_kg_m3 must be positive")

    @property
    def enabled(self) -> bool:
        return self.thickness_m > 0.0

    def effective_diameter(self, base_diameter_m: float) -> float:
        """Hydrodynamic diameter ``D + 2 t_mg`` [m]."""
        return base_diameter_m + 2.0 * self.thickness_m

    def _annulus_area(self, base_diameter_m: float) -> float:
        d_eff = self.effective_diameter(base_diameter_m)
        return math.pi / 4.0 * (d_eff**2 - base_diameter_m**2)

    def mass_per_length(self, base_diameter_m: float) -> float:
        """Structural mass added by the growth layer [kg/m]."""
        return self.density_kg_m3 * self._annulus_area(base_diameter_m)

    def submerged_weight_per_length(
        self, base_diameter_m: float, *, seawater_density: float = RHO_SEAWATER, g: float = G,
    ) -> float:
        """Net submerged weight added by the growth ``(rho_mg - rho_sw) A g`` [N/m]."""
        return (self.density_kg_m3 - seawater_density) * self._annulus_area(base_diameter_m) * g
