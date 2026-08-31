"""Full-chain orchestration: MRU motion -> remaining-life posterior + provenance.

Ties the layers together into one deterministic call so a shell (app/API) or a
reviewer can reproduce any result from its exported provenance (inputs, seed,
library versions). Every heavy step delegates to the tested modules; this file
only sequences them and records what it did.

Chain: Welch PSD / JONSWAP fit (sea state) -> catenary + Route-1 H(f)
-> TDP hot-spot stress (time-domain and spectral) -> rainflow + S-N + Miner
-> Arabian Gulf correction -> 10k Monte Carlo remaining-life posterior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import BaseModel

from . import __version__
from .config import AnalysisConfig
from .environment import EnvironmentCorrection
from .hang_off_kinematics import resolve_hang_off
from .miner import SECONDS_PER_YEAR, DamageResult, FatigueAcceptance, block_damage, dff_acceptance
from .montecarlo import MonteCarloResult, UncertaintyModel, run_monte_carlo
from .rainflow import count_cycles, range_histogram
from .sn import MeanStressModel, SNCurve, get_curve
from .spectral import SeaState, fit_jonswap, spectral_moments, welch_psd
from .spectral_damage import dirlik_damage_rate_curve
from .stress import (
    rfft_frequencies,
    stress_history_from_motion,
    stress_psd_from_motion_psd,
)
from .transfer import (
    InterpolatedTransferFunction,
    TransferFunction,
    analytic_transfer_function,
    reference_transfer_function,
)


class Provenance(BaseModel):
    """Everything needed to reproduce a run (serialisable, deterministic)."""

    core_version: str
    numpy_version: str
    scipy_version: str
    seed: int
    config_sha256: str
    n_samples: int
    sample_rate_hz: float
    transfer_is_reduced_order: bool
    motion_is_synthetic: bool
    transfer_route: str = "reference"
    transfer_is_validated: bool = False
    transfer_source: str = ""
    mean_stress_model: str = "none"
    mean_stress_applied: bool = False
    static_mean_stress_pa: float = 0.0


@dataclass(frozen=True)
class FullResult:
    """Complete result of a full-chain analysis."""

    sea_state: SeaState
    motion_freqs: NDArray[np.float64]
    motion_psd: NDArray[np.float64]
    stress_psd: NDArray[np.float64]
    hf_freqs: NDArray[np.float64]
    hf_moment_mag: NDArray[np.float64]
    hf_stress_mag: NDArray[np.float64]
    hf_phase: NDArray[np.float64]
    transfer_route: str
    transfer_provenance: dict
    dof_contributions: dict
    catenary_profile: dict
    rainflow_hist: dict
    spectral_moments: dict
    time_domain_block: DamageResult
    annual_damage_rate_time: float
    annual_damage_rate_spectral: float
    deterministic_life_years: float
    fatigue_acceptance: FatigueAcceptance
    monte_carlo: MonteCarloResult
    environment_factor: float
    environment: EnvironmentCorrection | None
    provenance: Provenance
    parameters: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """JSON-able headline numbers for logging / UI / provenance export."""
        return {
            "hs": self.sea_state.hs,
            "tp": self.sea_state.tp,
            "gamma": self.sea_state.gamma,
            "annual_damage_rate_time": self.annual_damage_rate_time,
            "annual_damage_rate_spectral": self.annual_damage_rate_spectral,
            "deterministic_life_years": self.deterministic_life_years,
            "fatigue_utilisation": self.fatigue_acceptance.utilisation,
            "fatigue_passes": self.fatigue_acceptance.passes,
            "life_p10": self.monte_carlo.p10,
            "life_p50": self.monte_carlo.p50,
            "life_p90": self.monte_carlo.p90,
            "environment_factor": self.environment_factor,
            "provenance": self.provenance.model_dump(),
        }


def _config_hash(config: AnalysisConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def _corrected_curve(base: SNCurve, correction: EnvironmentCorrection | None) -> tuple[SNCurve, float]:
    if correction is None:
        return base, 1.0
    return correction.apply_to_curve(base), correction.combined_factor


def run_full_analysis(
    config: AnalysisConfig,
    motion_heave: NDArray[np.float64],
    fs: float,
    *,
    motion_is_synthetic: bool = False,
    imported_tf: InterpolatedTransferFunction | None = None,
    motion_channels: dict[str, NDArray[np.float64]] | None = None,
) -> FullResult:
    """Run the full chain deterministically for one motion block.

    Parameters
    ----------
    config:
        Validated :class:`AnalysisConfig` (riser, transfer, environment, seed).
    motion_heave:
        Hang-off vertical motion time series [m], uniformly sampled at ``fs``.
    fs:
        Sample rate [Hz].
    motion_is_synthetic:
        Whether ``motion_heave`` came from the synthetic generator (badged in
        provenance so downstream UI can label it).
    """
    # Layer 0: resolve the vertical hang-off (porch) motion that drives the TDP.
    # With 6-DOF channels this applies Eq. 6 (heave + pitch/roll lever arms);
    # with only heave the resolved motion is the heave itself.
    if motion_channels:
        resolved = resolve_hang_off(
            motion_channels, config.hang_off.geometry(), exact=config.hang_off.exact_rotation
        )
        x = np.asarray(resolved.vertical, dtype=np.float64).ravel()
        dof_contributions = resolved.contribution_fractions()
    else:
        x = np.asarray(motion_heave, dtype=np.float64).ravel()
        dof_contributions = {"heave": 1.0}
    if x.size < 16:
        raise ValueError("motion too short for analysis (need >= 16 samples)")
    if fs <= 0.0:
        raise ValueError("fs must be positive")

    riser = config.riser
    section = riser.pipe_section()
    catenary = riser.catenary()
    base_curve = get_curve(riser.sn_class, riser.sn_environment)
    correction = config.environment.correction()
    curve, env_factor = _corrected_curve(base_curve, correction)

    tcfg = config.transfer

    def build_tf(freqs: NDArray[np.float64]) -> TransferFunction:
        """Layer-1 H(f) on ``freqs`` per the configured route.

        ``imported`` -> validated project table (OrcaFlex/RIFLEX/...), preferred;
        ``reference`` -> illustrative Route-2 table (realistic magnitude, NOT data);
        ``analytic`` -> Route-1 reduced-order model (documented approximation).
        """
        if tcfg.route == "imported":
            if imported_tf is None:
                raise ValueError(
                    "transfer route 'imported' requires a validated H(f) table; "
                    "supply one via load_transfer_csv / the H(f) upload."
                )
            return imported_tf.evaluate(freqs)
        if tcfg.route == "analytic":
            return analytic_transfer_function(
                freqs, catenary, section,
                natural_frequency=tcfg.natural_frequency,
                sigma_velocity=tcfg.sigma_velocity,
                drag_coefficient=tcfg.drag_coefficient,
                added_mass_coefficient=tcfg.added_mass_coefficient,
                structural_damping_ratio=tcfg.structural_damping_ratio,
                contents_density=riser.contents_density,
            )
        return reference_transfer_function(freqs)

    # Transfer-function provenance: only an imported vendor table counts as
    # validated (project) data; reference/analytic are illustrative/approximate.
    if tcfg.route == "imported" and imported_tf is not None:
        transfer_is_validated = bool(imported_tf.provenance.is_validated)
        transfer_source = imported_tf.provenance.source_tool or "imported"
        transfer_provenance = imported_tf.provenance.as_dict()
    elif tcfg.route == "analytic":
        transfer_is_validated = False
        transfer_source = "reduced-order (Route 1)"
        transfer_provenance = {"is_validated": False,
                               "notes": "reduced-order Morison model - documented approximation"}
    else:
        transfer_is_validated = False
        transfer_source = "illustrative reference (Route 2)"
        transfer_provenance = {"is_validated": False,
                               "notes": "illustrative reference bump - NOT project data"}

    # --- Sea state (Welch + JONSWAP fit) ---
    f_w, pxx = welch_psd(x, fs)
    sea_state = fit_jonswap(f_w, pxx)

    # --- Layer 1: transfer function on the FFT grid (time domain) ---
    fft_freqs = rfft_frequencies(x.size, fs)
    tf_time: TransferFunction = build_tf(fft_freqs)

    # --- Layer 2: time-domain stress -> rainflow -> Miner ---
    stress = stress_history_from_motion(x, fs, tf_time, section, scf=riser.scf)
    cycles = count_cycles(stress)
    duration = x.size / fs

    # Mean-stress correction (DNV-RP-C203 Sec. 2.3 / mean-stress guidance).
    # As-welded girth welds carry near-yield tensile residual stress, so DNV
    # permits NO mean-stress benefit: the correction is suppressed and the full
    # range is used. For base-material / stress-relieved details the configured
    # model rides on the standing (static) hot-spot mean - by default the axial
    # membrane stress T_TDP/A_steel from the catenary (the wave-induced dynamic
    # mean alone is ~0 through the bending transfer, so an explicit static offset
    # is what makes the correction physical).
    static_mean = (
        riser.static_mean_stress
        if riser.static_mean_stress is not None
        else catenary.horizontal_tension / section.steel_area
    )
    resolved_mean_model = (
        MeanStressModel.NONE if riser.as_welded else riser.mean_stress_model
    )
    mean_stress_applied = resolved_mean_model is not MeanStressModel.NONE
    td_block = block_damage(
        cycles, curve, duration, thickness_m=riser.thickness_for_correction,
        mean_stress_model=resolved_mean_model,
        ultimate_strength_pa=riser.ultimate_strength,
        static_mean_pa=static_mean if mean_stress_applied else 0.0,
    )
    annual_rate_time = td_block.damage_rate_per_year
    life_years = float("inf") if annual_rate_time <= 0.0 else 1.0 / annual_rate_time
    acceptance = dff_acceptance(
        life_years, riser.design_service_life_years, riser.design_fatigue_factor
    )

    # --- Spectral pathway (Dirlik against the two-slope curve) as a cross-check ---
    tf_spec = build_tf(f_w)
    stress_psd = stress_psd_from_motion_psd(pxx, tf_spec, section, scf=riser.scf)
    moments = spectral_moments(f_w, stress_psd, (0, 1, 2, 4))
    if moments[0] > 0.0 and moments[2] > 0.0 and moments[4] > 0.0:
        dirlik_per_s = dirlik_damage_rate_curve(
            moments, curve, stress_to_mpa=1e-6, thickness_m=riser.thickness_for_correction
        )
        annual_rate_spectral = dirlik_per_s * SECONDS_PER_YEAR
    else:
        annual_rate_spectral = 0.0

    # --- Figure data: catenary profile, rainflow range histogram, moments ---
    prof_x = np.linspace(0.0, catenary.horizontal_span, 200)
    prof_y = catenary.shape(prof_x)
    catenary_profile = {
        "x": prof_x, "y": prof_y,
        "catenary_parameter": catenary.catenary_parameter,
        "horizontal_span": catenary.horizontal_span,
        "arc_length": catenary.arc_length,
        "water_depth": catenary.water_depth,
        "tdp_curvature": catenary.tdp_curvature,
        "top_angle_deg": float(np.degrees(catenary.top_angle)),
    }
    if cycles.ranges.size and float(cycles.ranges.max()) > 0.0:
        edges = np.linspace(0.0, float(cycles.ranges.max()), 41)
        hist_counts = range_histogram(cycles, edges)
    else:
        edges = np.linspace(0.0, 1.0, 41)
        hist_counts = np.zeros(40, dtype=np.float64)
    rainflow_hist = {"edges": edges, "counts": hist_counts, "stress_to_mpa": 1.0e-6}
    spectral_moments_out = {int(k): float(v) for k, v in moments.items()}

    # --- |H(f)| curve for the Fig-4 transfer-function view (wave band) ---
    # Reported both as TDP moment transfer [N m per m heave] and, matching the
    # paper's Fig 4, as stress transfer [MPa per m heave] = |H_moment| SCF / Z.
    hf_freqs = np.linspace(0.02, 0.40, 200)
    tf_hf = build_tf(hf_freqs)
    hf_moment_mag = tf_hf.magnitude
    hf_stress_mag = hf_moment_mag * riser.scf / section.section_modulus / 1.0e6
    hf_phase = tf_hf.phase

    # --- Layer 3: Monte Carlo remaining-life posterior ---
    model = UncertaintyModel(
        env_factor_mean=env_factor if correction is not None else 1.0,
        sn_slope_m=curve.m1,
    )
    nominal_rate = max(annual_rate_time, 1e-12)
    mc = run_monte_carlo(nominal_rate, model, n_members=config.n_monte_carlo, seed=config.seed)

    provenance = Provenance(
        core_version=__version__,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        seed=config.seed,
        config_sha256=_config_hash(config),
        n_samples=int(x.size),
        sample_rate_hz=float(fs),
        transfer_is_reduced_order=tf_time.is_reduced_order,
        motion_is_synthetic=motion_is_synthetic,
        transfer_route=tcfg.route,
        transfer_is_validated=transfer_is_validated,
        transfer_source=transfer_source,
        mean_stress_model=str(resolved_mean_model.value),
        mean_stress_applied=mean_stress_applied,
        static_mean_stress_pa=float(static_mean if mean_stress_applied else 0.0),
    )

    return FullResult(
        sea_state=sea_state,
        motion_freqs=f_w,
        motion_psd=pxx,
        stress_psd=stress_psd,
        hf_freqs=hf_freqs,
        hf_moment_mag=hf_moment_mag,
        hf_stress_mag=hf_stress_mag,
        hf_phase=hf_phase,
        transfer_route=tcfg.route,
        transfer_provenance=transfer_provenance,
        dof_contributions=dof_contributions,
        catenary_profile=catenary_profile,
        rainflow_hist=rainflow_hist,
        spectral_moments=spectral_moments_out,
        time_domain_block=td_block,
        annual_damage_rate_time=annual_rate_time,
        annual_damage_rate_spectral=annual_rate_spectral,
        deterministic_life_years=life_years,
        fatigue_acceptance=acceptance,
        monte_carlo=mc,
        environment_factor=env_factor,
        environment=correction,
        provenance=provenance,
        parameters=model.describe(),
    )
