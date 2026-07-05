"""S-N (Woehler) curves per DNV-RP-C203 and mean-stress corrections.

Implements the standard two-slope high-cycle S-N formulation

    log10(N) = log10(a_bar) - m * log10(dsigma)

with a slope change at N = 1e7 cycles (m1 -> m2). Stress ranges are handled in
MPa because that is how the DNV curves are tabulated; the public API accepts SI
(Pa) and converts internally.

The tabulated parameters are the DNV-RP-C203 (Ed. 2016) *in-air* curves,
Table 2-1. Environmental knock-downs (seawater / Arabian Gulf) are applied
separately in :mod:`scr_twin_core.environment`, so this module holds only the
base capacity.

Reference
---------
DNV-RP-C203 "Fatigue design of offshore steel structures", Sec. 2 and
Table 2-1 (S-N curves in air) and Sec. 2.4.3 (thickness effect).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import T_REF_DNV


@dataclass(frozen=True)
class SNCurve:
    """Two-slope S-N curve parameters (stress in MPa, N in cycles).

    ``log_a1``/``m1`` govern the high-stress branch (N <= ``n_transition``);
    ``log_a2``/``m2`` govern the low-stress branch. ``thickness_exponent`` is the
    DNV thickness-correction exponent ``k``. ``fatigue_limit_mpa`` is the stress
    range at the slope change (equal on both branches by construction).
    """

    name: str
    m1: float
    log_a1: float
    m2: float
    log_a2: float
    thickness_exponent: float
    n_transition: float = 1.0e7

    @property
    def fatigue_limit_mpa(self) -> float:
        """Stress range (MPa) at the ``n_transition`` slope change."""
        return 10.0 ** ((self.log_a1 - np.log10(self.n_transition)) / self.m1)


# DNV-RP-C203 Table 2-1, S-N curves in air. Stress in MPa.
DNV_C203_IN_AIR: dict[str, SNCurve] = {
    "B1": SNCurve("B1", 4.0, 15.117, 5.0, 17.146, 0.00),
    "B2": SNCurve("B2", 4.0, 14.885, 5.0, 16.856, 0.00),
    "C": SNCurve("C", 3.0, 12.592, 5.0, 16.320, 0.05),
    "C1": SNCurve("C1", 3.0, 12.449, 5.0, 16.081, 0.10),
    "C2": SNCurve("C2", 3.0, 12.301, 5.0, 15.835, 0.15),
    "D": SNCurve("D", 3.0, 12.164, 5.0, 15.606, 0.20),
    "E": SNCurve("E", 3.0, 12.010, 5.0, 15.350, 0.20),
    "F": SNCurve("F", 3.0, 11.855, 5.0, 15.091, 0.25),
    "F1": SNCurve("F1", 3.0, 11.699, 5.0, 14.832, 0.25),
    "F3": SNCurve("F3", 3.0, 11.546, 5.0, 14.576, 0.25),
    "G": SNCurve("G", 3.0, 11.398, 5.0, 14.330, 0.25),
}


class MeanStressModel(StrEnum):
    """Mean-stress correction applied before entering the S-N curve."""

    NONE = "none"
    GOODMAN = "goodman"
    SWT = "swt"  # Smith-Watson-Topper


def get_curve(sn_class: str) -> SNCurve:
    """Look up a DNV-RP-C203 in-air curve by class name (case-insensitive)."""
    key = sn_class.strip().upper()
    if key not in DNV_C203_IN_AIR:
        raise KeyError(
            f"Unknown S-N class {sn_class!r}. Available: "
            f"{sorted(DNV_C203_IN_AIR)}"
        )
    return DNV_C203_IN_AIR[key]


def thickness_factor(thickness_m: float, exponent: float, t_ref: float = T_REF_DNV) -> float:
    """DNV thickness-correction factor ``(t/t_ref)^k`` for ``t > t_ref``.

    For members thinner than the reference thickness the factor is unity (no
    benefit is taken), per DNV-RP-C203 Sec. 2.4.3.
    """
    if thickness_m <= 0.0:
        raise ValueError("thickness_m must be positive")
    return float((max(thickness_m, t_ref) / t_ref) ** exponent)


def apply_mean_stress(
    stress_range_pa: NDArray[np.float64],
    mean_stress_pa: NDArray[np.float64],
    model: MeanStressModel,
    ultimate_strength_pa: float,
) -> NDArray[np.float64]:
    """Convert (range, mean) to an equivalent fully-reversed range.

    - ``GOODMAN``: dsigma_eq = dsigma / (1 - sigma_m / sigma_u), clipped so a
      mean at/above ultimate does not produce a non-physical negative range.
    - ``SWT``: equivalent range = 2 * sqrt(sigma_max * sigma_a) using
      sigma_a = dsigma/2, valid for tensile sigma_max (compressive maxima are
      left unchanged as non-damaging in the SWT sense).
    """
    dsig = np.asarray(stress_range_pa, dtype=np.float64)
    mean = np.asarray(mean_stress_pa, dtype=np.float64)
    if model is MeanStressModel.NONE:
        return dsig
    if model is MeanStressModel.GOODMAN:
        denom = 1.0 - np.clip(mean, 0.0, None) / ultimate_strength_pa
        denom = np.clip(denom, 1e-6, None)
        return dsig / denom
    if model is MeanStressModel.SWT:
        amp = dsig / 2.0
        smax = mean + amp
        smax = np.clip(smax, 0.0, None)
        return 2.0 * np.sqrt(smax * amp)
    raise ValueError(f"Unhandled mean-stress model: {model}")


def cycles_to_failure(
    stress_range_pa: ArrayLike,
    curve: SNCurve,
    *,
    thickness_m: float | None = None,
    t_ref: float = T_REF_DNV,
) -> NDArray[np.float64]:
    """Number of cycles to failure ``N(dsigma)`` on the two-slope curve.

    Parameters
    ----------
    stress_range_pa:
        Hot-spot stress range(s) in Pa (already including any SCF and
        mean-stress correction).
    curve:
        The :class:`SNCurve` to evaluate.
    thickness_m:
        If given, the DNV thickness correction ``(t/t_ref)^k`` scales the stress
        range before the curve is evaluated. ``None`` disables it.

    Returns
    -------
    NDArray[np.float64]
        Cycles to failure, same shape as the input. Zero/negative ranges map to
        ``+inf`` (non-damaging).
    """
    dsig_mpa = np.asarray(stress_range_pa, dtype=np.float64) / 1.0e6
    if thickness_m is not None:
        dsig_mpa = dsig_mpa * thickness_factor(thickness_m, curve.thickness_exponent, t_ref)

    out = np.full(dsig_mpa.shape, np.inf, dtype=np.float64)
    positive = dsig_mpa > 0.0
    dsp = dsig_mpa[positive]

    log_dsig = np.log10(dsp)
    high = dsp >= curve.fatigue_limit_mpa  # steep (m1) branch, N <= n_transition
    log_n = np.where(
        high,
        curve.log_a1 - curve.m1 * log_dsig,
        curve.log_a2 - curve.m2 * log_dsig,
    )
    out[positive] = np.power(10.0, log_n)
    return out
