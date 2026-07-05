"""scr_twin_core - framework-agnostic physics core for the SCR fatigue twin.

The package converts floating-unit Motion Reference Unit (MRU) time series into a
probabilistic estimate of steel catenary riser (SCR) fatigue life at the
touchdown point (TDP). It has no UI or web-framework dependencies so it can be
embedded in a desktop app, a WASM build, or a server identically.

Layer map (see the project spec):
    spectral / synthetic  -> sea-state and motion spectra
    catenary / transfer   -> Layer 1: MRU motion -> TDP bending moment H(f)
    stress                -> TDP hot-spot stress reconstruction
    rainflow / sn / miner -> Layer 2: cycle counting, S-N, cumulative damage
    spectral_damage       -> frequency-domain damage (Dirlik / Tovo-Benasciutti)
    environment           -> Arabian Gulf capacity knock-down
    montecarlo / bayesian -> Layer 3: probabilistic remaining life + updating
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    bayesian,
    catenary,
    config,
    constants,
    environment,
    ingest,
    inspection,
    miner,
    montecarlo,
    pipeline,
    rainflow,
    section,
    sn,
    spectral,
    spectral_damage,
    stress,
    synthetic,
    transfer,
    validation,
)

__all__ = [
    "bayesian",
    "catenary",
    "config",
    "constants",
    "environment",
    "ingest",
    "inspection",
    "miner",
    "montecarlo",
    "pipeline",
    "rainflow",
    "section",
    "sn",
    "spectral",
    "spectral_damage",
    "stress",
    "synthetic",
    "transfer",
    "validation",
    "__version__",
]
