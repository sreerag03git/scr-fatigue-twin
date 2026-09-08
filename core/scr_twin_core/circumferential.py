"""Circumferential (clock-position) fatigue distribution at the TDP girth weld.

A girth-weld fatigue assessment does not stop at a single hot spot: the same weld
is damaged differently at each position around the circumference, and a planar
defect can lie anywhere, so knowing *where* damage concentrates targets
inspection. This module maps the annual fatigue damage around the pipe at the
touchdown point by combining the two frequency-separated cyclic mechanisms the
twin already resolves:

    wave-induced bending  - in the catenary (vertical) plane at wave frequency.
        For a wave heading ``beta`` the in-plane and out-of-plane wave components
        are taken IN PHASE (a single oscillation plane), so the fibre stress range
        at clock angle ``phi`` is ``S_wave * |cos(phi - beta)|`` - i.e. an oblique
        heading rotates the WAVE HOT SPOT to ``phi = beta`` (and ``beta + 180``).
        The geometric crown (phi = 0) and keel (phi = 180) are FIXED pipe
        positions; only the hot spot migrates.

    cross-flow VIV        - out of the catenary plane at the modal frequency
        ``fn``; its fibre stress at ``phi`` is ``S_mode * |cos(phi - 90)|``,
        peaking at the 90/270 saddles.

Because the two processes sit at (assumed) well-separated frequencies they do not
share cycles, so their damage adds by Miner at each position:
``D(phi) = D_wave(phi) + D_viv(phi)``.

What is reported. The **conservative governing life** is the *co-located* combined
life ``1 / (max_phi D_wave + max_phi D_viv)`` - it assumes the wave and VIV peaks
fall on the same weld point, matching the wave+VIV Miner life reported elsewhere.
The map additionally gives the **worst clock position** (largest ``D(phi)``) to
target inspection, and the per-position life distribution.

Idealizations (screening tool, NOT a life-extension basis). This model assumes a
single in-phase wave oscillation plane and PURE cross-flow VIV; it does NOT model
in-line (streamwise) VIV - which acts in-plane near the crown/keel at ~2*fn - nor
the oblique-heading phase whirl that smears the wave hot spot, nor weld-toe
multi-axiality. Any of these can co-locate the mechanisms, which is why the
governing life is taken as the conservative co-located combined life rather than
the (longer) position-resolved worst life. The Miner add also assumes ``fn`` is
well above the wave band; a soft low-order mode near the wave frequency would make
the true combined damage larger than the sum (the map does not correct for that).

It rescales the *actual* rainflow ranges and VIV mode ranges by the geometric
clock factor, applies the same effective SCF, thickness and mean-stress basis as
the primary assessment, and re-evaluates the DNV-RP-C203 two-slope S-N curve
exactly (no power-law shortcut).

Reference
---------
DNV-RP-C203 (hot-spot stress, girth-weld details); DNV-OS-F201 (SCR girth-weld
fatigue, in-plane vs out-of-plane bending); DNV-RP-F204 / RP-F105 (cross-flow and
in-line VIV). Standard riser-fatigue practice of reporting the worst
circumferential position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sn import MeanStressModel, SNCurve, apply_mean_stress, cycles_to_failure

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class CircumferentialResult:
    """Annual fatigue damage distributed around the TDP girth weld."""

    angles_deg: NDArray[np.float64]     # clock position phi, 0 = geometric crown (catenary plane)
    damage_rate: NDArray[np.float64]    # total annual damage at each phi
    wave_rate: NDArray[np.float64]      # wave (in-plane) contribution
    viv_rate: NDArray[np.float64]       # cross-flow VIV (out-of-plane) contribution
    life_years: NDArray[np.float64]     # 1 / damage_rate at each phi
    worst_angle_deg: float              # phi that maximises damage
    worst_life_years: float             # position-resolved minimum life (screening, NOT governing)
    crown_life_years: float             # life at the fixed geometric crown phi = 0
    best_life_years: float              # life at the least-damaged position
    combined_colocated_life_years: float  # conservative governing life (wave & VIV peaks co-located)
    heading_deg: float

    def as_dict(self) -> dict[str, object]:
        return {
            "angles_deg": [float(a) for a in self.angles_deg],
            "damage_rate": [float(d) for d in self.damage_rate],
            "wave_rate": [float(d) for d in self.wave_rate],
            "viv_rate": [float(d) for d in self.viv_rate],
            "life_years": [float(v) for v in self.life_years],
            "worst_angle_deg": self.worst_angle_deg,
            "worst_life_years": self.worst_life_years,
            "crown_life_years": self.crown_life_years,
            "best_life_years": self.best_life_years,
            "combined_colocated_life_years": self.combined_colocated_life_years,
            "heading_deg": self.heading_deg,
        }


def _mean_corrected(
    ranges_pa: NDArray[np.float64], model: MeanStressModel | None,
    static_mean_pa: float, ultimate_strength_pa: float | None,
) -> NDArray[np.float64]:
    """Equivalent fully-reversed ranges after the (optional) mean-stress model.

    Mirrors the primary pipeline: the cyclic ranges ride on the standing axial
    hot-spot mean (uniform around the circumference), so the same scalar mean and
    model apply at every clock position.
    """
    if (
        model is None or model is MeanStressModel.NONE
        or ultimate_strength_pa is None or static_mean_pa == 0.0
    ):
        return ranges_pa
    mean = np.full(ranges_pa.shape, float(static_mean_pa), dtype=np.float64)
    return apply_mean_stress(ranges_pa, mean, model, ultimate_strength_pa)


def _annual_damage_from_ranges(
    ranges_pa: NDArray[np.float64], counts: NDArray[np.float64],
    block_seconds: float, curve: SNCurve, thickness_m: float | None,
) -> float:
    """Miner annual damage from a (range, count) block (ranges already SN-ready)."""
    if block_seconds <= 0.0:
        return 0.0
    n_fail = cycles_to_failure(ranges_pa, curve, thickness_m=thickness_m)
    good = np.isfinite(n_fail) & (n_fail > 0.0)
    d_block = float(np.sum(np.where(good, counts / np.where(good, n_fail, 1.0), 0.0)))
    return d_block / block_seconds * SECONDS_PER_YEAR


def circumferential_damage_map(
    *,
    wave_ranges_pa: ArrayLike,
    wave_counts: ArrayLike,
    block_seconds: float,
    curve: SNCurve,
    thickness_m: float | None = None,
    viv_mode_ranges_pa: ArrayLike = (),
    viv_mode_freqs_hz: ArrayLike = (),
    heading_deg: float = 0.0,
    n_positions: int = 72,
    mean_stress_model: MeanStressModel | None = None,
    static_mean_pa: float = 0.0,
    ultimate_strength_pa: float | None = None,
) -> CircumferentialResult:
    """Map annual fatigue damage around the TDP girth weld (see module docstring).

    ``wave_ranges_pa`` / ``wave_counts`` are the hot-spot rainflow histogram (bin
    centres and counts, hot-spot stress already including the SCF) over a block of
    ``block_seconds``. ``viv_mode_ranges_pa`` / ``viv_mode_freqs_hz`` are the
    per-mode cross-flow VIV hot-spot stress ranges (SCF-consistent with the wave
    ranges) and frequencies - they must be the same length (empty disables VIV).
    ``heading_deg`` rotates the wave hot spot. ``mean_stress_model`` /
    ``static_mean_pa`` / ``ultimate_strength_pa`` apply the same mean-stress basis
    as the primary assessment (omit for the as-welded / no-correction default).
    """
    w_ranges = np.asarray(wave_ranges_pa, dtype=np.float64).ravel()
    w_counts = np.asarray(wave_counts, dtype=np.float64).ravel()
    vr = np.asarray(viv_mode_ranges_pa, dtype=np.float64).ravel()
    vf = np.asarray(viv_mode_freqs_hz, dtype=np.float64).ravel()
    if vr.size != vf.size:
        raise ValueError(
            f"viv_mode_ranges_pa and viv_mode_freqs_hz must be the same length "
            f"(got {vr.size} and {vf.size})"
        )

    phi = np.linspace(0.0, 360.0, int(n_positions), endpoint=False)
    beta = float(heading_deg)
    wave_rate = np.zeros_like(phi)
    viv_rate = np.zeros_like(phi)
    for i, p in enumerate(phi):
        f_ip = abs(np.cos(np.radians(p - beta)))          # wave: in-plane, peak at beta
        wr = _mean_corrected(w_ranges * f_ip, mean_stress_model, static_mean_pa, ultimate_strength_pa)
        wave_rate[i] = _annual_damage_from_ranges(wr, w_counts, block_seconds, curve, thickness_m)
        if vr.size:
            f_op = abs(np.cos(np.radians(p - 90.0)))       # VIV: out-of-plane, peak at 90
            ve = _mean_corrected(vr * f_op, mean_stress_model, static_mean_pa, ultimate_strength_pa)
            n_fail = cycles_to_failure(ve, curve, thickness_m=thickness_m)
            good = np.isfinite(n_fail) & (n_fail > 0.0)
            viv_rate[i] = float(np.sum(np.where(good, vf * SECONDS_PER_YEAR / np.where(good, n_fail, 1.0), 0.0)))

    total = wave_rate + viv_rate
    life = np.where(total > 0.0, 1.0 / np.where(total > 0.0, total, 1.0), np.inf)
    worst_idx = int(np.argmax(total))
    best_idx = int(np.argmin(total))
    # Conservative governing life: assume the wave and VIV peaks co-locate.
    colocated_rate = float(np.max(wave_rate)) + float(np.max(viv_rate))
    colocated_life = 1.0 / colocated_rate if colocated_rate > 0.0 else np.inf
    return CircumferentialResult(
        angles_deg=phi,
        damage_rate=total,
        wave_rate=wave_rate,
        viv_rate=viv_rate,
        life_years=life,
        worst_angle_deg=float(phi[worst_idx]),
        worst_life_years=float(life[worst_idx]),
        crown_life_years=float(life[0]),
        best_life_years=float(life[best_idx]),
        combined_colocated_life_years=float(colocated_life),
        heading_deg=beta,
    )
