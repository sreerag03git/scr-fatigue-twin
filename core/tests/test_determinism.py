"""Determinism gate (project spec 5): identical seed+config -> identical output.

Drives a full mini-pipeline (synthetic MRU -> H(f) -> stress -> rainflow ->
Miner) twice and asserts byte-identical numeric results, plus the standalone
Monte Carlo path.
"""

from __future__ import annotations

import numpy as np

from scr_twin_core.catenary import solve_plain_catenary
from scr_twin_core.miner import miner_damage
from scr_twin_core.montecarlo import UncertaintyModel, run_monte_carlo
from scr_twin_core.rainflow import count_cycles
from scr_twin_core.section import PipeSection
from scr_twin_core.sn import get_curve
from scr_twin_core.stress import rfft_frequencies, stress_history_from_motion
from scr_twin_core.synthetic import synthetic_mru_motion
from scr_twin_core.transfer import analytic_transfer_function


def _pipeline_damage(seed: int) -> float:
    section = PipeSection(outer_diameter=0.3239, wall_thickness=0.0206)
    cat = solve_plain_catenary(water_depth=1500.0, top_angle_deg=72.0, submerged_weight=1500.0)
    motion = synthetic_mru_motion(duration=1200.0, fs=4.0, hs=3.0, tp=10.0, seed=seed)
    freqs = rfft_frequencies(motion.heave.size, motion.fs)
    tf = analytic_transfer_function(freqs, cat, section, natural_frequency=0.12, sigma_velocity=0.3)
    stress = stress_history_from_motion(motion.heave, motion.fs, tf, section, scf=1.2)
    cycles = count_cycles(stress)
    return miner_damage(cycles, get_curve("D"))


def test_full_pipeline_is_deterministic():
    d1 = _pipeline_damage(seed=2024)
    d2 = _pipeline_damage(seed=2024)
    assert d1 == d2  # exact equality, not approx
    assert d1 > 0.0


def test_pipeline_seed_changes_result():
    assert _pipeline_damage(seed=1) != _pipeline_damage(seed=2)


def test_monte_carlo_byte_identical():
    a = run_monte_carlo(0.025, UncertaintyModel(), n_members=5000, seed=7)
    b = run_monte_carlo(0.025, UncertaintyModel(), n_members=5000, seed=7)
    np.testing.assert_array_equal(a.life_years, b.life_years)
    np.testing.assert_array_equal(a.damage_rate_per_year, b.damage_rate_per_year)
