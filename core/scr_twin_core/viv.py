"""Vortex-Induced Vibration (VIV) fatigue SCREENING (DNV-RP-F204 / RP-F105).

On a real deep-water SCR the current-driven cross-flow VIV is frequently the
dominant or co-dominant fatigue mechanism, entirely separate from wave/vessel
motion. This module is a defensible *screening* chain - not a design tool:

    current profile U(s)               (power-law shear, DNV-RP-C205)
      -> tensioned-beam cross-flow modes fn, phi_n(s), phi_n''(s)
                                       (real FD eigensolve; self-checked vs the
                                        taut-string / EI limits)
      -> reduced velocity Vr = U/(fn D), Strouhal shedding, lock-in band
                                       (DNV-RP-F204 Sec. 4.4 screening)
      -> cross-flow response A/D from the stability parameter Ks
                                       (Griffin plot; DNV-RP-F105)
      -> mode-curvature stress range -> narrow-band damage at fn.

The response amplitude is a Griffin-plot upper bound, NOT a validated
fluid-structure response - design-grade multi-mode VIV needs Shear7 / VIVANA
(a separate import route). Every VIV result is badged ``is_screening=True``.

References
----------
DNV-RP-F204 Sec. 4 (riser VIV); DNV-RP-F105 (free-spanning pipelines, VIV
response & stability parameter); Griffin & Ramberg (1982) amplitude vs
stability parameter; Blevins, "Flow-Induced Vibration".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from .catenary import PlainCatenary
from .constants import RHO_SEAWATER, RHO_STEEL
from .section import PipeSection
from .sn import SNCurve, cycles_to_failure

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class CurrentProfile:
    """Sheared current speed over the water column (power-law, DNV-RP-C205 Sec.6).

    ``U(h) = surface_velocity * (h / water_depth) ** exponent`` with ``h`` the
    height above the seabed (0 at the TDP, ``water_depth`` at the surface), so the
    current is strongest near the surface and vanishes at the seabed. The default
    1/7 power law is the standard open-sea profile.
    """

    surface_velocity: float  # [m/s] at the surface
    water_depth: float  # [m]
    exponent: float = 1.0 / 7.0

    def speed_at_height(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        frac = np.clip(np.asarray(h, dtype=np.float64) / self.water_depth, 0.0, 1.0)
        return (self.surface_velocity * frac**self.exponent).astype(np.float64)


@dataclass(frozen=True)
class RiserModes:
    """Cross-flow tensioned-beam modes of the suspended span."""

    frequencies_hz: NDArray[np.float64]  # fn per mode
    arc: NDArray[np.float64]  # node arc-length coordinate s [m]
    shapes: NDArray[np.float64]  # (n_modes, n_nodes) mode shapes, max|phi|=1
    curvatures: NDArray[np.float64]  # (n_modes, n_nodes) phi_n''(s) [1/m per unit disp]
    span_length: float  # suspended arc length [m]


@dataclass(frozen=True)
class VivMode:
    """Screening result for one excited cross-flow mode."""

    mode: int
    frequency_hz: float
    reduced_velocity: float  # representative Vr at the power-in region
    excited: bool
    a_over_d: float
    stress_range_mpa: float
    annual_damage_rate: float


@dataclass(frozen=True)
class VivScreening:
    """VIV screening outcome (badged ``is_screening=True``)."""

    modes: list[VivMode]
    annual_damage_rate: float
    life_years: float
    dominant_mode: int
    stability_parameter: float  # Ks
    current_surface_velocity: float
    riser_modes: RiserModes  # the computed cross-flow modes (for the mode-shape diagram)
    is_screening: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "annual_damage_rate": self.annual_damage_rate,
            "life_years": self.life_years,
            "dominant_mode": self.dominant_mode,
            "stability_parameter": self.stability_parameter,
            "current_surface_velocity": self.current_surface_velocity,
            "is_screening": self.is_screening,
            "modes": [
                {
                    "mode": m.mode, "frequency_hz": m.frequency_hz,
                    "reduced_velocity": m.reduced_velocity, "excited": m.excited,
                    "a_over_d": m.a_over_d, "stress_range_mpa": m.stress_range_mpa,
                    "annual_damage_rate": m.annual_damage_rate,
                }
                for m in self.modes
            ],
        }


def effective_mass_per_length(
    section: PipeSection, *, contents_density: float, added_mass_coefficient: float = 1.0,
    hydro_diameter: float | None = None, extra_mass_per_length: float = 0.0,
) -> float:
    """Wet mass per unit length [kg/m]: steel + contents + added mass + marine growth.

    ``hydro_diameter`` (default the steel OD) sets the added-mass displaced area;
    ``extra_mass_per_length`` adds the marine-growth structural mass.
    """
    d_hydro = hydro_diameter if hydro_diameter is not None else section.outer_diameter
    a_disp = np.pi / 4.0 * d_hydro**2
    m_steel = section.steel_area * RHO_STEEL
    m_content = section.bore_area * contents_density
    m_added = added_mass_coefficient * RHO_SEAWATER * a_disp
    return float(m_steel + m_content + m_added + extra_mass_per_length)


def riser_modes(
    catenary: PlainCatenary,
    section: PipeSection,
    *,
    contents_density: float = 0.0,
    added_mass_coefficient: float = 1.0,
    n_modes: int = 8,
    n_nodes: int = 240,
    hydro_diameter: float | None = None,
    extra_mass_per_length: float = 0.0,
) -> RiserModes:
    """Cross-flow modes of the suspended span as a tensioned Euler-Bernoulli beam.

    Solves the pinned-pinned generalised eigenproblem ``K phi = omega^2 M phi``
    with a finite-difference energy formulation: bending energy ``EI (phi'')^2``,
    tension (geometric) energy ``T(s) (phi')^2`` using the spatially varying
    catenary tension, and consistent mass ``m (phi)^2``. Returns the first
    ``n_modes`` wet natural frequencies, mode shapes (normalised to unit peak
    displacement) and their curvatures. Reduces to the taut-string modes as
    ``EI -> 0`` and to the beam modes as ``T -> 0`` (self-checked in tests).
    """
    ei = section.bending_stiffness
    m = effective_mass_per_length(
        section, contents_density=contents_density, added_mass_coefficient=added_mass_coefficient,
        hydro_diameter=hydro_diameter, extra_mass_per_length=extra_mass_per_length,
    )
    length = catenary.arc_length
    # Interior nodes (phi = 0 at both ends). Map arc nodes to horizontal x for the
    # tension lookup (screening approximation: uniform arc<->x spacing).
    h = length / (n_nodes + 1)
    s = np.arange(1, n_nodes + 1) * h
    x = np.linspace(0.0, catenary.horizontal_span, n_nodes)
    tension = catenary.tension(x)  # effective tension along the span [N]

    # First- and second-derivative operators on interior nodes (Dirichlet ends).
    d1 = (np.diag(np.ones(n_nodes - 1), 1) - np.diag(np.ones(n_nodes - 1), -1)) / (2.0 * h)
    d2 = (np.diag(np.ones(n_nodes - 1), 1) - 2.0 * np.eye(n_nodes)
          + np.diag(np.ones(n_nodes - 1), -1)) / h**2

    k_bend = ei * (d2.T @ d2) * h
    k_tension = (d1.T @ (tension[:, None] * d1)) * h
    k = k_bend + k_tension
    mass = np.eye(n_nodes) * (m * h)

    # Symmetrise (guard tiny asymmetry from float ops) and solve the smallest modes.
    k = 0.5 * (k + k.T)
    w2, vecs = eigh(k, mass, subset_by_index=[0, n_modes - 1])
    w2 = np.clip(w2, 0.0, None)
    freqs = np.sqrt(w2) / (2.0 * np.pi)

    shapes = np.empty((n_modes, n_nodes))
    curvs = np.empty((n_modes, n_nodes))
    for i in range(n_modes):
        phi = vecs[:, i]
        peak = np.max(np.abs(phi))
        if peak > 0.0:
            phi = phi / peak
        shapes[i] = phi
        curvs[i] = d2 @ phi
    return RiserModes(
        frequencies_hz=freqs.astype(np.float64), arc=s.astype(np.float64),
        shapes=shapes, curvatures=curvs, span_length=float(length),
    )


def stability_parameter(
    mass_per_length: float, damping_ratio: float, diameter: float,
    *, water_density: float = RHO_SEAWATER,
) -> float:
    """Vandiver/Scruton stability parameter ``Ks = 2 m delta / (rho D^2)``.

    ``delta = 2 pi zeta`` is the log-decrement. High ``Ks`` (heavy, well-damped)
    suppresses VIV; low ``Ks`` allows large lock-in amplitudes.
    """
    delta = 2.0 * np.pi * damping_ratio
    return float(2.0 * mass_per_length * delta / (water_density * diameter**2))


def griffin_amplitude(ks: float) -> float:
    """Peak cross-flow ``A/D`` vs stability parameter (Griffin plot upper bound).

    ``A/D = 1.29 / (1 + 0.43 Ks)^3.35`` capped at 1.3 - a standard screening
    envelope (DNV-RP-F105 / Griffin & Ramberg 1982). This is an upper bound, not a
    validated response.
    """
    return float(min(1.3, 1.29 / (1.0 + 0.43 * max(ks, 0.0)) ** 3.35))


# Cross-flow lock-in reduced-velocity window (DNV-RP-F204 Sec. 4.4 screening).
VR_LOCK_IN_LOW, VR_LOCK_IN_HIGH, VR_PEAK = 3.0, 9.0, 6.0


def _lock_in_factor(vr: float) -> float:
    """Fraction of the peak A/D at reduced velocity ``vr`` (0 outside lock-in)."""
    if not (VR_LOCK_IN_LOW <= vr <= VR_LOCK_IN_HIGH):
        return 0.0
    # Triangular-ish window peaking at VR_PEAK.
    if vr <= VR_PEAK:
        return (vr - VR_LOCK_IN_LOW) / (VR_PEAK - VR_LOCK_IN_LOW)
    return (VR_LOCK_IN_HIGH - vr) / (VR_LOCK_IN_HIGH - VR_PEAK)


def viv_screening(
    catenary: PlainCatenary,
    section: PipeSection,
    curve: SNCurve,
    current: CurrentProfile,
    *,
    contents_density: float = 0.0,
    added_mass_coefficient: float = 1.0,
    strouhal: float = 0.18,
    damping_ratio: float = 0.02,
    n_modes: int = 60,
    thickness_m: float | None = None,
    marine_growth_thickness_m: float = 0.0,
    marine_growth_mass_per_length: float = 0.0,
) -> VivScreening:
    """Screen the SCR for cross-flow VIV fatigue over the excited modes.

    For each mode: the representative reduced velocity ``Vr = U_rep/(fn D)`` uses
    the current speed at the mode's power-in region; modes with ``Vr`` in the
    lock-in band are excited with amplitude ``A/D = griffin(Ks) * lock_in(Vr)``;
    the mode-curvature at that amplitude gives a stress range whose narrow-band
    damage at ``fn`` is summed. Because a long SCR locks in at a HIGH mode
    (``fn ~ St U/D``), enough modes are computed to reach the shedding frequency.
    Badged as a screening estimate.
    """
    # Marine growth increases the hydrodynamic diameter (drag / added mass / lock-in)
    # and the mass, but NOT the steel section that carries the bending stress.
    d_hydro = section.outer_diameter + 2.0 * marine_growth_thickness_m
    modes = riser_modes(
        catenary, section, contents_density=contents_density,
        added_mass_coefficient=added_mass_coefficient, n_modes=n_modes,
        n_nodes=max(300, 12 * n_modes),
        hydro_diameter=d_hydro, extra_mass_per_length=marine_growth_mass_per_length,
    )
    m_eff = effective_mass_per_length(
        section, contents_density=contents_density, added_mass_coefficient=added_mass_coefficient,
        hydro_diameter=d_hydro, extra_mass_per_length=marine_growth_mass_per_length,
    )
    ks = stability_parameter(m_eff, damping_ratio, d_hydro)
    a_over_d_peak = griffin_amplitude(ks)
    d_out = d_hydro                       # lock-in / reduced velocity / amplitude
    d_steel = section.outer_diameter      # bending stress uses the steel section
    e_mod = section.youngs_modulus

    # Current at each node's height above the seabed. Height ~ catenary shape y(x).
    x_nodes = np.linspace(0.0, catenary.horizontal_span, modes.shapes.shape[1])
    height = catenary.shape(x_nodes)  # y above TDP ~ height above seabed [m]
    u_node = current.speed_at_height(height)

    results: list[VivMode] = []
    total_rate = 0.0
    for i in range(n_modes):
        fn = float(modes.frequencies_hz[i])
        if fn <= 0.0:
            continue
        # Power-in region = where this mode's displacement is largest; use the
        # current there as the representative excitation speed.
        w = np.abs(modes.shapes[i])
        u_rep = float(np.sum(w * u_node) / np.sum(w)) if np.sum(w) > 0 else 0.0
        vr = u_rep / (fn * d_out) if fn * d_out > 0 else 0.0
        lock = _lock_in_factor(vr)
        a_over_d = a_over_d_peak * lock
        excited = a_over_d > 1e-3
        stress_range_pa = 0.0
        rate = 0.0
        if excited:
            amplitude = a_over_d * d_out  # cross-flow displacement amplitude ~ hydro D [m]
            max_curv = float(np.max(np.abs(modes.curvatures[i])))  # per unit peak disp
            # Stress amplitude = E (D_steel/2) * curvature_amplitude; range = 2*amplitude.
            stress_amp_pa = e_mod * (d_steel / 2.0) * amplitude * max_curv
            stress_range_pa = 2.0 * stress_amp_pa
            n_fail = float(cycles_to_failure(
                np.array([stress_range_pa]), curve, thickness_m=thickness_m)[0])
            if np.isfinite(n_fail) and n_fail > 0.0:
                rate = fn * SECONDS_PER_YEAR / n_fail
        total_rate += rate
        results.append(VivMode(
            mode=i + 1, frequency_hz=fn, reduced_velocity=vr, excited=excited,
            a_over_d=a_over_d, stress_range_mpa=stress_range_pa / 1e6, annual_damage_rate=rate,
        ))

    life = float("inf") if total_rate <= 0.0 else 1.0 / total_rate
    dominant = max(results, key=lambda r: r.annual_damage_rate, default=None)
    return VivScreening(
        modes=results, annual_damage_rate=total_rate, life_years=life,
        dominant_mode=dominant.mode if dominant and dominant.annual_damage_rate > 0 else 0,
        stability_parameter=ks, current_surface_velocity=current.surface_velocity,
        riser_modes=modes,
    )
