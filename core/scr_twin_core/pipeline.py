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
from .miner import SECONDS_PER_YEAR, DamageResult, block_damage
from .montecarlo import MonteCarloResult, UncertaintyModel, run_monte_carlo
from .rainflow import count_cycles
from .sn import SNCurve, get_curve
from .spectral import SeaState, fit_jonswap, spectral_moments, welch_psd
from .spectral_damage import dirlik_damage_rate_curve
from .stress import (
    rfft_frequencies,
    stress_history_from_motion,
    stress_psd_from_motion_psd,
)
from .transfer import (
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


@dataclass(frozen=True)
class FullResult:
    """Complete result of a full-chain analysis."""

    sea_state: SeaState
    motion_freqs: NDArray[np.float64]
    motion_psd: NDArray[np.float64]
    stress_psd: NDArray[np.float64]
    time_domain_block: DamageResult
    annual_damage_rate_time: float
    annual_damage_rate_spectral: float
    deterministic_life_years: float
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
    x = np.asarray(motion_heave, dtype=np.float64).ravel()
    if x.size < 16:
        raise ValueError("motion_heave too short for analysis (need >= 16 samples)")
    if fs <= 0.0:
        raise ValueError("fs must be positive")

    riser = config.riser
    section = riser.pipe_section()
    catenary = riser.catenary()
    base_curve = get_curve(riser.sn_class)
    correction = config.environment.correction()
    curve, env_factor = _corrected_curve(base_curve, correction)

    tcfg = config.transfer

    def build_tf(freqs: NDArray[np.float64]) -> TransferFunction:
        """Layer-1 H(f) on ``freqs`` per the configured route.

        ``reference`` -> illustrative Route-2 table (realistic magnitude);
        ``analytic`` -> Route-1 reduced-order model (honest but under-predicting).
        (``imported`` requires a user table via the API and is not reachable here.)
        """
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
    td_block = block_damage(
        cycles, curve, duration, thickness_m=riser.thickness_for_correction
    )
    annual_rate_time = td_block.damage_rate_per_year
    life_years = float("inf") if annual_rate_time <= 0.0 else 1.0 / annual_rate_time

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
    )

    return FullResult(
        sea_state=sea_state,
        motion_freqs=f_w,
        motion_psd=pxx,
        stress_psd=stress_psd,
        time_domain_block=td_block,
        annual_damage_rate_time=annual_rate_time,
        annual_damage_rate_spectral=annual_rate_spectral,
        deterministic_life_years=life_years,
        monte_carlo=mc,
        environment_factor=env_factor,
        environment=correction,
        provenance=provenance,
        parameters=model.describe(),
    )
