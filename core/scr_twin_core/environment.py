"""Arabian Gulf corrosion-fatigue capacity correction (project spec 4.5).

The base DNV-RP-C203 in-air S-N capacity is knocked down by a combined factor
built from two separately-exposed contributions:

    temperature factor  F_T  in [0.72, 0.78]   (Vosikovsky, 1980)
    salinity factor     F_S  in [0.85, 0.90]
    combined            F    = F_T * F_S  in [0.61, 0.70]

Applying ``F`` as a multiplier on cycles-to-failure ``N`` (equivalently on the
S-N intercept ``a_bar``) scales the fatigue *life* by ``F``, i.e. a life
reduction of ``1 - F`` in [30%, 39%] relative to the standard curve - the
acceptance target in project spec 5.

IMPORTANT (traceability): this reduction range sits **outside the DNV database**.
The temperature dependence follows Vosikovsky (1980), "Effects of temperature on
corrosion fatigue crack growth"; the salinity factor is an engineering estimate
for high-salinity Gulf water and should be replaced by project-specific test
data before design use. Each factor is exposed and editable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .sn import SNCurve

# Documented admissible bounds (used for validation / UI clamping).
TEMP_FACTOR_BOUNDS = (0.72, 0.78)
SALINITY_FACTOR_BOUNDS = (0.85, 0.90)
COMBINED_FACTOR_BOUNDS = (0.61, 0.70)


@dataclass(frozen=True)
class EnvironmentCorrection:
    """A separable capacity-reduction correction.

    Defaults sit mid-range (F ~ 0.656, i.e. ~34% life reduction). Both factors
    are editable; :meth:`validate` checks them against the documented bounds.
    """

    temperature_factor: float = 0.75
    salinity_factor: float = 0.875
    label: str = "Arabian Gulf"

    @property
    def combined_factor(self) -> float:
        return self.temperature_factor * self.salinity_factor

    @property
    def life_reduction(self) -> float:
        """Fractional reduction in fatigue life ``1 - F`` (0..1)."""
        return 1.0 - self.combined_factor

    def validate(self) -> None:
        """Raise ``ValueError`` if factors fall outside the documented bounds."""
        lo, hi = TEMP_FACTOR_BOUNDS
        if not (lo <= self.temperature_factor <= hi):
            raise ValueError(f"temperature_factor {self.temperature_factor} outside {TEMP_FACTOR_BOUNDS}")
        lo, hi = SALINITY_FACTOR_BOUNDS
        if not (lo <= self.salinity_factor <= hi):
            raise ValueError(f"salinity_factor {self.salinity_factor} outside {SALINITY_FACTOR_BOUNDS}")

    def correct_life(self, standard_life_years: float) -> float:
        """Life under the correction: ``F * standard_life``."""
        return self.combined_factor * standard_life_years

    def correct_damage_rate(self, standard_rate: float) -> float:
        """Damage rate under the correction: ``standard_rate / F``."""
        return standard_rate / self.combined_factor

    def apply_to_curve(self, curve: SNCurve) -> SNCurve:
        """Return a knocked-down S-N curve (N scaled by ``F`` at every stress).

        Both branch intercepts shift by ``log10(F)`` and the transition cycle
        count scales by ``F`` so the branch-crossing *stress* (fatigue limit) is
        preserved - only the life is reduced.
        """
        log_f = math.log10(self.combined_factor)
        return SNCurve(
            name=f"{curve.name}[{self.label}]",
            m1=curve.m1,
            log_a1=curve.log_a1 + log_f,
            m2=curve.m2,
            log_a2=curve.log_a2 + log_f,
            thickness_exponent=curve.thickness_exponent,
            n_transition=curve.n_transition * self.combined_factor,
        )
