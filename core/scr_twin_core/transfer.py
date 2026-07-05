"""Layer 1: hang-off motion -> TDP bending-moment transfer function H(f).

Two consistent routes are provided (the user picks):

Route 1 - analytic (this module's :func:`analytic_transfer_function`)
    A **reduced-order** frequency-response model:

        H(f) = G_qs * DAF(f)                                     [N m per m heave]

    where ``G_qs`` is the quasi-static gain (change in TDP bending moment per
    unit vertical hang-off motion, obtained by re-solving the catenary at a
    fixed horizontal offset) and ``DAF(f)`` is a single-DOF dynamic-amplification
    factor whose damping includes the stochastically linearised Morison drag

        C_eq = sqrt(8/pi) * sigma_u * (1/2) rho C_d D            (per unit length)

    Documented assumptions (small-strain, planar motion, linearised drag,
    quasi-static TDP boundary, single dominant mode). This is an engineering
    approximation - results carry an ``is_reduced_order=True`` flag. For project
    rigor use Route 2.

Route 2 - imported (:func:`InterpolatedTransferFunction`)
    A magnitude/phase table H(f) exported from a validated riser analysis
    (OrcaFlex / RIFLEX / DeepLines) for the specific riser, interpolated onto the
    analysis grid. Preferred whenever available.

References
----------
- Morison et al. (1950), Petroleum Trans. AIME 189 (wave force on piles).
- Equivalent stochastic linearisation of drag: Borgman (1967);
  Roberts & Spanos, "Random Vibration and Statistical Linearization" (1990).
- Quéau et al. (2015), Ocean Engineering 96 (parametric SCR TDP stress TF) -
  used as the independent validation reference (see validation module).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .catenary import PlainCatenary, solve_plain_catenary_fixed_span
from .constants import RHO_SEAWATER
from .section import PipeSection


@dataclass(frozen=True)
class TransferFunction:
    """Complex TDP bending-moment transfer function on a frequency grid.

    ``value[k]`` is complex moment response [N m] per unit hang-off vertical
    displacement [m] at ``freqs[k]`` [Hz]. ``is_reduced_order`` marks Route-1
    (analytic) results so the UI can badge them as engineering approximations.
    """

    freqs: NDArray[np.float64]
    value: NDArray[np.complex128]
    is_reduced_order: bool

    @property
    def magnitude(self) -> NDArray[np.float64]:
        return np.abs(self.value)

    @property
    def phase(self) -> NDArray[np.float64]:
        return np.angle(self.value)


def quasi_static_moment_gain(
    catenary: PlainCatenary,
    section: PipeSection,
    *,
    rel_step: float = 1e-4,
) -> float:
    """Quasi-static TDP bending-moment gain ``dM_TDP/dz`` [N m / m].

    Central finite difference of ``M_TDP = E I / a`` with respect to a vertical
    perturbation of the hang-off (water depth), holding the horizontal offset
    fixed and re-solving the catenary each time.
    """
    x_span = catenary.horizontal_span
    d0 = catenary.water_depth
    w = catenary.submerged_weight
    ei = section.bending_stiffness
    dz = max(rel_step * d0, 1e-6)

    def moment_at_depth(depth: float) -> float:
        cat = solve_plain_catenary_fixed_span(x_span, depth, w)
        return ei * cat.tdp_curvature  # M = EI * kappa = EI / a

    m_plus = moment_at_depth(d0 + dz)
    m_minus = moment_at_depth(d0 - dz)
    return (m_plus - m_minus) / (2.0 * dz)


def linearized_drag_damping(
    section: PipeSection,
    *,
    sigma_velocity: float,
    drag_coefficient: float = 1.0,
    seawater_density: float = RHO_SEAWATER,
) -> float:
    """Equivalent linear (Morison) drag coefficient per unit length [N s / m^2].

    Stochastic linearisation of the quadratic drag ``(1/2) rho C_d D |u| u`` for a
    zero-mean Gaussian relative velocity with std ``sigma_velocity``:

        C_eq = sqrt(8/pi) * sigma_velocity * (1/2) rho C_d D.
    """
    if sigma_velocity < 0.0:
        raise ValueError("sigma_velocity must be non-negative")
    return math.sqrt(8.0 / math.pi) * sigma_velocity * 0.5 * seawater_density * drag_coefficient * section.outer_diameter


def hydro_damping_ratio(
    section: PipeSection,
    natural_frequency: float,
    *,
    sigma_velocity: float,
    drag_coefficient: float = 1.0,
    added_mass_coefficient: float = 1.0,
    contents_density: float = 0.0,
    steel_density: float = 7850.0,
    seawater_density: float = RHO_SEAWATER,
) -> float:
    """Modal damping ratio from linearised Morison drag.

    ``zeta = C_eq / (2 m_eff omega_n)`` with effective mass per length
    ``m_eff = m_steel + m_contents + rho C_a A_outer`` (structural + added mass).
    """
    c_eq = linearized_drag_damping(
        section,
        sigma_velocity=sigma_velocity,
        drag_coefficient=drag_coefficient,
        seawater_density=seawater_density,
    )
    a_outer = math.pi / 4.0 * section.outer_diameter**2
    m_steel = section.steel_area * steel_density
    m_contents = section.bore_area * contents_density
    m_added = seawater_density * added_mass_coefficient * a_outer
    m_eff = m_steel + m_contents + m_added
    omega_n = 2.0 * math.pi * natural_frequency
    if m_eff <= 0.0 or omega_n <= 0.0:
        raise ValueError("effective mass and natural frequency must be positive")
    return c_eq / (2.0 * m_eff * omega_n)


def analytic_transfer_function(
    freqs: ArrayLike,
    catenary: PlainCatenary,
    section: PipeSection,
    *,
    natural_frequency: float,
    sigma_velocity: float,
    drag_coefficient: float = 1.0,
    added_mass_coefficient: float = 1.0,
    structural_damping_ratio: float = 0.005,
    contents_density: float = 0.0,
) -> TransferFunction:
    """Route 1 reduced-order H(f): quasi-static gain x single-DOF DAF.

    Parameters
    ----------
    natural_frequency:
        Fundamental TDP-region natural frequency [Hz]. An engineering estimate;
        for rigor supply Route 2.
    sigma_velocity:
        RMS transverse relative water-particle velocity [m/s] used to linearise
        the Morison drag (sets the hydrodynamic damping).
    structural_damping_ratio:
        Structural modal damping added to the hydrodynamic damping.
    """
    f = np.asarray(freqs, dtype=np.float64)
    g_qs = quasi_static_moment_gain(catenary, section)
    zeta_h = hydro_damping_ratio(
        section,
        natural_frequency,
        sigma_velocity=sigma_velocity,
        drag_coefficient=drag_coefficient,
        added_mass_coefficient=added_mass_coefficient,
        contents_density=contents_density,
    )
    zeta = structural_damping_ratio + zeta_h

    r = f / natural_frequency
    # Single-DOF complex receptance normalised to unit static response.
    denom = (1.0 - r**2) + 2j * zeta * r
    daf = 1.0 / denom
    value = g_qs * daf
    return TransferFunction(freqs=f, value=value.astype(np.complex128), is_reduced_order=True)


class InterpolatedTransferFunction:
    """Route 2: H(f) imported from a validated riser analysis and interpolated.

    Parameters
    ----------
    table_freqs, table_magnitude, table_phase:
        Tabulated frequency [Hz], moment magnitude [N m / m] and phase [rad].
    """

    def __init__(
        self,
        table_freqs: ArrayLike,
        table_magnitude: ArrayLike,
        table_phase: ArrayLike,
    ) -> None:
        f = np.asarray(table_freqs, dtype=np.float64)
        mag = np.asarray(table_magnitude, dtype=np.float64)
        ph = np.asarray(table_phase, dtype=np.float64)
        if not (f.shape == mag.shape == ph.shape):
            raise ValueError("table_freqs, table_magnitude, table_phase must share shape")
        if f.size < 2:
            raise ValueError("need at least two table points to interpolate")
        order = np.argsort(f)
        self._f = f[order]
        self._mag = mag[order]
        self._ph = np.unwrap(ph[order])

    def evaluate(self, freqs: ArrayLike) -> TransferFunction:
        """Interpolate onto ``freqs`` (linear; clamped outside the table)."""
        f = np.asarray(freqs, dtype=np.float64)
        mag = np.interp(f, self._f, self._mag)
        ph = np.interp(f, self._f, self._ph)
        value = mag * np.exp(1j * ph)
        return TransferFunction(freqs=f, value=value.astype(np.complex128), is_reduced_order=False)


# Illustrative reference TDP moment-transfer magnitude scale [N m per m heave].
# Calibrated so the documented reference SCR under a moderate sea state yields a
# realistic TDP fatigue life (order 10^2 yr). This is an ILLUSTRATIVE Route-2
# table representative of a deep-water SCR - NOT a measured/OrcaFlex result.
# Real projects must import their own H(f); see InterpolatedTransferFunction.
REFERENCE_HF_SCALE: float = 4.0e3
REFERENCE_HF_PEAK_HZ: float = 0.14
REFERENCE_HF_WIDTH_HZ: float = 0.09
REFERENCE_HF_LAG_S: float = 1.5


def reference_transfer_function(
    freqs: ArrayLike, *, scale: float | None = None
) -> TransferFunction:
    """Illustrative reference (Route-2) TDP moment transfer function.

    A representative deep-water-SCR magnitude shape (a smooth wave-band bump
    peaking near ``REFERENCE_HF_PEAK_HZ``) with a linear transport-lag phase.
    Flagged ``is_reduced_order=False`` because it stands in for an imported,
    validated H(f); it is nonetheless ILLUSTRATIVE, not project data. Provided so
    the console shows realistic numbers offline while real H(f) import is wired.
    """
    f = np.asarray(freqs, dtype=np.float64)
    if scale is None:
        scale = REFERENCE_HF_SCALE
    shape = np.exp(-0.5 * ((f - REFERENCE_HF_PEAK_HZ) / REFERENCE_HF_WIDTH_HZ) ** 2)
    shape = np.where(f < 0.02, 0.0, shape)  # no response below the wave band
    mag = scale * shape
    phase = -2.0 * np.pi * f * REFERENCE_HF_LAG_S
    value = mag * np.exp(1j * phase)
    return TransferFunction(freqs=f, value=value.astype(np.complex128), is_reduced_order=False)


def apply_transfer_to_spectrum(
    motion_psd: ArrayLike, tf: TransferFunction
) -> NDArray[np.float64]:
    """Moment PSD ``S_M(f) = |H(f)|^2 S_motion(f)`` [ (N m)^2 / Hz ].

    Random-vibration input-output relation for a linear transfer function.
    """
    s = np.asarray(motion_psd, dtype=np.float64)
    if s.shape != tf.freqs.shape:
        raise ValueError("motion_psd must be sampled on the transfer-function grid")
    return (tf.magnitude**2 * s).astype(np.float64)
