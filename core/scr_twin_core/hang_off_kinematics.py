"""Layer 0: 6-DOF MRU motion -> resolved SCR hang-off (porch) motion.

The MRU reports rigid-body motion at the vessel reference point. The riser hangs
off a porch offset ``(x_p, y_p, z_p)`` from that reference, so the motion that
actually drives the touchdown point is the motion *of the porch*, resolved onto
the riser plane. This module implements that transform.

Small-angle (paper Eq. 6), vertical component::

    z_ho(t) = heave(t) - x_p * pitch(t) + y_p * roll(t)

and the in-plane horizontal component along the riser azimuth ``alpha``::

    dx = surge + z_p*pitch - y_p*yaw
    dy = sway  + x_p*yaw   - z_p*roll
    x_ho = dx*cos(alpha) + dy*sin(alpha)

The exact finite-rotation form (``exact=True``) applies the full yaw-pitch-roll
rotation ``R = Rz(yaw) Ry(pitch) Rx(roll)`` to the porch vector; the two agree to
second order in the angles. Angles are supplied in **degrees** (the MRU / ingest
convention); lengths and translations in metres.

Convention: right-handed vessel frame, x forward, y port, z up; roll about x,
pitch about y, yaw about z. ``riser_azimuth_deg`` is the riser departure heading
in the vessel frame (0 deg = +x / forward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# The six canonical DOF (translations [m], rotations [deg]).
DOF_NAMES = ("heave", "pitch", "roll", "surge", "sway", "yaw")
_ANGLE_DOF = {"pitch", "roll", "yaw"}


@dataclass(frozen=True)
class PorchGeometry:
    """Riser hang-off (porch) offset from the vessel reference and departure azimuth."""

    x_p: float = 0.0  # longitudinal offset [m], + forward
    y_p: float = 0.0  # transverse offset [m], + port
    z_p: float = 0.0  # vertical offset [m], + up
    riser_azimuth_deg: float = 0.0  # riser departure azimuth in vessel frame [deg]


@dataclass(frozen=True)
class ResolvedHangOff:
    """Resolved hang-off motion plus a per-DOF variance decomposition.

    ``vertical`` is the z_ho time series that drives the (vertical) TDP transfer
    function; ``inplane`` is the horizontal in-plane component. ``contributions``
    maps each DOF to its share of ``var(vertical)`` (Cov(term_i, z_ho), so the
    shares sum to the total variance) - i.e. "which DOF drives the fatigue".
    """

    vertical: NDArray[np.float64]
    inplane: NDArray[np.float64]
    contributions: dict[str, float]  # Cov(term_i, z_ho); sums to Var(z_ho) (can be negative)
    term_variances: dict[str, float]  # Var(term_i); non-negative
    used_channels: list[str]

    @property
    def vertical_variance(self) -> float:
        return float(np.var(self.vertical))

    def contribution_fractions(self) -> dict[str, float]:
        """Per-DOF share of the summed per-DOF motion variances (non-negative, sums to 1).

        A clean attribution for "which DOF drives the fatigue" (the share each DOF
        would carry acting alone). For the exact, signed decomposition that
        accounts for inter-DOF phase (and can be negative when a DOF partly
        cancels the resolved motion), use :attr:`contributions`.
        """
        total = sum(self.term_variances.values())
        if total < 1e-30:
            return {k: 0.0 for k in self.term_variances}
        return {k: v / total for k, v in self.term_variances.items()}


def _get(channels: dict[str, NDArray[np.float64]], name: str, n: int) -> NDArray[np.float64]:
    arr = channels.get(name)
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size != n:
        raise ValueError(f"channel '{name}' length {a.size} != {n}")
    return a


def resolve_hang_off(
    channels: dict[str, NDArray[np.float64]],
    geom: PorchGeometry,
    *,
    exact: bool = False,
) -> ResolvedHangOff:
    """Resolve 6-DOF MRU motion to the riser hang-off motion (paper Eq. 6).

    ``channels`` may contain any subset of ``heave, pitch, roll, surge, sway,
    yaw`` (missing DOF are treated as zero); angle channels are in degrees. With
    only ``heave`` present the vertical result is exactly the heave (backwards
    compatible with the heave-only pipeline).
    """
    present = [d for d in DOF_NAMES if d in channels and np.asarray(channels[d]).size]
    if not present:
        raise ValueError("no recognised motion channels to resolve")
    n = int(np.asarray(channels[present[0]]).ravel().size)

    heave = _get(channels, "heave", n)
    surge = _get(channels, "surge", n)
    sway = _get(channels, "sway", n)
    pitch = np.deg2rad(_get(channels, "pitch", n))
    roll = np.deg2rad(_get(channels, "roll", n))
    yaw = np.deg2rad(_get(channels, "yaw", n))
    xp, yp, zp = geom.x_p, geom.y_p, geom.z_p
    az = math.radians(geom.riser_azimuth_deg)

    if exact:
        cph, sph = np.cos(roll), np.sin(roll)
        cth, sth = np.cos(pitch), np.sin(pitch)
        cps, sps = np.cos(yaw), np.sin(yaw)
        # Displacement of the porch = R r_p - r_p + translation, with
        # R = Rz(yaw) Ry(pitch) Rx(roll).
        rz = -sth * xp + cth * sph * yp + cth * cph * zp
        vertical = heave + (rz - zp)
        rx = cps * cth * xp + (cps * sth * sph - sps * cph) * yp + (cps * sth * cph + sps * sph) * zp
        ry = sps * cth * xp + (sps * sth * sph + cps * cph) * yp + (sps * sth * cph - cps * sph) * zp
        dx = surge + (rx - xp)
        dy = sway + (ry - yp)
    else:
        vertical = heave - xp * pitch + yp * roll
        dx = surge + zp * pitch - yp * yaw
        dy = sway + xp * yaw - zp * roll
    inplane = dx * math.cos(az) + dy * math.sin(az)

    # Per-DOF contribution to var(vertical): additive terms of the small-angle
    # form; Cov(term_i, z_ho) sums to Var(z_ho).
    terms = {
        "heave": heave,
        "pitch": -xp * pitch,
        "roll": yp * roll,
    }
    active = {k: t for k, t in terms.items() if k in present or k == "heave"}
    zc = vertical - np.mean(vertical)
    contributions = {k: float(np.mean((t - np.mean(t)) * zc)) for k, t in active.items()}
    term_variances = {k: float(np.var(t)) for k, t in active.items()}
    return ResolvedHangOff(
        vertical=vertical.astype(np.float64),
        inplane=inplane.astype(np.float64),
        contributions=contributions,
        term_variances=term_variances,
        used_channels=present,
    )
