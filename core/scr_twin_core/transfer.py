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

import io
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


@dataclass(frozen=True)
class TransferProvenance:
    """Where an imported H(f) came from - stored so a run is traceable.

    An imported vendor table is treated as *validated (project)* data; a
    reduced-order or illustrative table is not. ``as_dict`` feeds the run
    provenance and the report badge.
    """

    source_tool: str = "unknown"     # e.g. "OrcaFlex", "RIFLEX", "DeepLines"
    tool_version: str = ""
    load_case: str = ""              # e.g. "Hs=6.8m Tp=11s heading=180 draft=survival"
    notes: str = ""
    is_validated: bool = True
    n_points: int = 0
    freq_min_hz: float = 0.0
    freq_max_hz: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "source_tool": self.source_tool, "tool_version": self.tool_version,
            "load_case": self.load_case, "notes": self.notes,
            "is_validated": self.is_validated, "n_points": self.n_points,
            "freq_min_hz": self.freq_min_hz, "freq_max_hz": self.freq_max_hz,
        }


class InterpolatedTransferFunction:
    """Route 2: complex H(f) imported from a validated riser analysis.

    Consumes a magnitude/phase (or Re/Im) TDP moment-transfer table exported from
    OrcaFlex / RIFLEX / DeepLines / Flexcom for a specific (riser, vessel,
    heading, draft, top-tension) load case, and interpolates it onto the analysis
    grid. This is the mechanism the paper leans on; the table itself is the
    project's to supply (see :func:`load_transfer_csv`).

    Parameters
    ----------
    table_freqs, table_magnitude, table_phase:
        Tabulated frequency [Hz], TDP moment-transfer magnitude [N m / m] and
        phase [rad].
    provenance:
        Optional :class:`TransferProvenance`; ``n_points`` / frequency range are
        filled in from the data if not supplied.
    """

    def __init__(
        self,
        table_freqs: ArrayLike,
        table_magnitude: ArrayLike,
        table_phase: ArrayLike,
        *,
        provenance: TransferProvenance | None = None,
    ) -> None:
        f = np.asarray(table_freqs, dtype=np.float64)
        mag = np.asarray(table_magnitude, dtype=np.float64)
        ph = np.asarray(table_phase, dtype=np.float64)
        if not (f.shape == mag.shape == ph.shape):
            raise ValueError("table_freqs, table_magnitude, table_phase must share shape")
        if f.size < 2:
            raise ValueError("need at least two table points to interpolate")
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(mag)) or not np.all(np.isfinite(ph)):
            raise ValueError("transfer-function table contains non-finite values")
        if np.any(mag < 0.0):
            raise ValueError("transfer-function magnitude must be non-negative")
        order = np.argsort(f)
        self._f = f[order]
        self._mag = mag[order]
        self._ph = np.unwrap(ph[order])
        base = provenance or TransferProvenance()
        self.provenance = TransferProvenance(
            source_tool=base.source_tool, tool_version=base.tool_version,
            load_case=base.load_case, notes=base.notes, is_validated=base.is_validated,
            n_points=int(self._f.size),
            freq_min_hz=float(self._f.min()), freq_max_hz=float(self._f.max()),
        )

    def evaluate(self, freqs: ArrayLike) -> TransferFunction:
        """Interpolate onto ``freqs`` (linear; clamped outside the table)."""
        f = np.asarray(freqs, dtype=np.float64)
        mag = np.interp(f, self._f, self._mag)
        ph = np.interp(f, self._f, self._ph)
        value = mag * np.exp(1j * ph)
        return TransferFunction(freqs=f, value=value.astype(np.complex128), is_reduced_order=False)


# Column-name aliases for imported H(f) CSVs (lower-cased, stripped).
_TF_FREQ_COLS = {"freq", "freq_hz", "frequency", "frequency_hz", "f", "hz", "f_hz"}
_TF_MAG_COLS = {"magnitude", "mag", "abs", "amplitude", "h_mag", "|h|", "hmag"}
_TF_PHASE_COLS = {"phase", "phase_rad", "phase_deg", "arg", "angle", "angle_rad", "angle_deg"}
_TF_RE_COLS = {"re", "real", "h_re", "re_h"}
_TF_IM_COLS = {"im", "imag", "imaginary", "h_im", "im_h"}


def load_transfer_csv(source: object, **overrides: object) -> InterpolatedTransferFunction:
    """Load a complex H(f) table from a CSV exported by a riser-analysis tool.

    Accepts either magnitude/phase or real/imag columns plus a frequency column
    (case-insensitive header aliases). Provenance may be given as ``# key: value``
    comment lines in the file header (``source_tool``, ``tool_version``,
    ``load_case``, ``notes``) and/or overridden via keyword arguments. Phase is
    radians unless the phase column is named ``*_deg`` (then degrees).

    Raises ``ValueError`` with a clear message on any malformed input - an
    imported H(f) is safety-critical, so it is rejected loudly rather than
    degraded silently.
    """
    import pandas as pd

    # --- read provenance comment lines, then the data ---
    meta: dict[str, str] = {}
    text: str | None = None
    if isinstance(source, (str, bytes)) or hasattr(source, "read"):
        try:
            if hasattr(source, "read"):
                raw = source.read()
                text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            elif isinstance(source, bytes):
                text = source.decode("utf-8", "replace")
            elif isinstance(source, str) and ("\n" in source or "," in source):
                text = source
        except Exception:  # noqa: BLE001
            text = None
    if text is not None:
        for line in text.splitlines():
            s = line.strip()
            if not s.startswith("#") or ":" not in s:
                continue
            k, _, v = s.lstrip("#").strip().partition(":")
            if k.strip().lower() in {"source_tool", "tool_version", "load_case", "notes"}:
                meta[k.strip().lower()] = v.strip()

    try:
        buf = io.StringIO(text) if text is not None else source
        df = pd.read_csv(buf, comment="#", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse transfer-function CSV: {exc}") from exc
    if df.empty:
        raise ValueError("transfer-function CSV has no data rows")

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(aliases: set[str]) -> str | None:
        for a in aliases:
            if a in cols:
                return cols[a]
        return None

    fcol = pick(_TF_FREQ_COLS)
    if fcol is None:
        raise ValueError(f"no frequency column found (need one of {sorted(_TF_FREQ_COLS)})")
    freqs = pd.to_numeric(df[fcol], errors="coerce").to_numpy(dtype=np.float64)

    magcol, phcol = pick(_TF_MAG_COLS), pick(_TF_PHASE_COLS)
    recol, imcol = pick(_TF_RE_COLS), pick(_TF_IM_COLS)
    if magcol is not None and phcol is not None:
        mag = pd.to_numeric(df[magcol], errors="coerce").to_numpy(dtype=np.float64)
        phase = pd.to_numeric(df[phcol], errors="coerce").to_numpy(dtype=np.float64)
        if str(phcol).strip().lower().endswith("deg"):
            phase = np.deg2rad(phase)
    elif recol is not None and imcol is not None:
        re = pd.to_numeric(df[recol], errors="coerce").to_numpy(dtype=np.float64)
        im = pd.to_numeric(df[imcol], errors="coerce").to_numpy(dtype=np.float64)
        mag = np.hypot(re, im)
        phase = np.arctan2(im, re)
    else:
        raise ValueError("need magnitude+phase columns or real+imag columns")

    good = np.isfinite(freqs) & np.isfinite(mag) & np.isfinite(phase)
    freqs, mag, phase = freqs[good], mag[good], phase[good]
    if freqs.size < 2:
        raise ValueError("fewer than two valid (freq, H) rows after cleaning")

    prov = TransferProvenance(
        source_tool=str(overrides.get("source_tool", meta.get("source_tool", "imported"))),
        tool_version=str(overrides.get("tool_version", meta.get("tool_version", ""))),
        load_case=str(overrides.get("load_case", meta.get("load_case", ""))),
        notes=str(overrides.get("notes", meta.get("notes", ""))),
        is_validated=True,
    )
    return InterpolatedTransferFunction(freqs, mag, phase, provenance=prov)


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
