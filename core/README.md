# scr_twin_core

Framework-agnostic physics core for a real-time **steel catenary riser (SCR)
fatigue digital twin**. It converts a floating unit's Motion Reference Unit (MRU)
recordings into a probabilistic estimate of SCR fatigue life at the touchdown
point (TDP). No UI or web-framework dependencies — the app is a thin shell over
this package.

## Layer map

| Stage | Module(s) | What it does |
|------|-----------|--------------|
| Sea state | `spectral`, `synthetic` | Welch PSD, spectral moments, JONSWAP/PM fit; deterministic synthetic MRU (demo only) |
| Layer 1 | `catenary`, `section`, `transfer` | Static catenary, pipe section, MRU→TDP bending-moment transfer function `H(f)` |
| Stress | `stress` | TDP hot-spot stress (time-domain FFT filtering **and** spectral propagation) |
| Layer 2 | `rainflow`, `sn`, `miner` | ASTM E1049 rainflow, DNV-RP-C203 S-N, Palmgren–Miner damage |
| Spectral damage | `spectral_damage` | Dirlik, Tovo–Benasciutti, narrow-band |
| Environment | `environment` | Arabian Gulf corrosion-fatigue knock-down |
| Layer 3 | `montecarlo`, `bayesian` | 10k-member Monte Carlo remaining-life posterior; Bayesian rate updating |
| Orchestration | `config`, `pipeline`, `validation` | Pydantic schemas, full-chain run + provenance, §5 gate checks |

## Design principles

- **Real physics only.** Every function docstring cites the governing equation
  or method. Illustrative/placeholder values are flagged in code (e.g.
  `is_reduced_order`, `is_synthetic`, `is_reference_preset`, and the lazy-wave
  solver raises `NotImplementedError` rather than faking a result).
- **Deterministic.** All stochastic paths take an explicit `seed`; identical
  seed+config → byte-identical output (`test_determinism.py`).
- **Validated.** 100% of the physics is covered by pytest, including analytical
  benchmarks and cross-checks against the `rainflow` and `fatpack` packages.

## Install & test

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows; use bin/activate on POSIX
pip install -e "core[test,dev]"
pytest core/tests -q          # run the suite
ruff check core/scr_twin_core # lint
mypy core/scr_twin_core       # type-check (run from core/)
python -m scr_twin_core.validation   # print the §5 acceptance-gate report
```

## Quick start

```python
import numpy as np
from scr_twin_core.config import AnalysisConfig, RiserConfig
from scr_twin_core.pipeline import run_full_analysis
from scr_twin_core.synthetic import synthetic_mru_motion

cfg = AnalysisConfig(riser=RiserConfig.reference_scr(), seed=0)
motion = synthetic_mru_motion(duration=1800, fs=4.0, hs=3.5, tp=11.0, seed=1)  # SYNTHETIC
result = run_full_analysis(cfg, motion.heave, motion.fs, motion_is_synthetic=True)

print(result.summary())        # P10/P50/P90 life, damage rates, provenance
```

## Plugging in real data and a project H(f)

- **Real MRU:** feed a measured heave (and, where available, pitch/surge) time
  series and its sample rate to `run_full_analysis`. The synthetic generator is
  a labelled demo/fallback only.
- **Project transfer function (Route 2):** import a magnitude/phase `H(f)` table
  exported from OrcaFlex/RIFLEX/DeepLines via
  `transfer.InterpolatedTransferFunction`; it is preferred over the Route-1
  analytic reduced-order model whenever available.

## Key references

- ASTM E1049-85 (rainflow); DNV-RP-C203 (S-N, thickness correction);
  DNV-RP-C205 (JONSWAP, sea states).
- Morison et al. (1950); stochastic linearisation: Borgman (1967).
- Dirlik (1985); Benasciutti & Tovo (2006).
- Quéau et al. (2015), *Ocean Engineering* — SCR TDP stress transfer function
  (independent validation reference for Route 1).
- Vosikovsky (1980) — corrosion-fatigue temperature dependence (Arabian Gulf
  temperature factor).

See module docstrings for the exact equation behind each function.
