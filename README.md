# SCR Fatigue Digital Twin

A physics-based digital twin that turns an FPSO/floating unit's existing **Motion
Reference Unit (MRU)** recordings into a continuously updated, probabilistic
estimate of **steel catenary riser (SCR)** fatigue life at the touchdown point
(TDP), and schedules inspection from that estimate.

Reference implementation of *"Real-Time Physics-Based SCR Fatigue Digital Twin
Using Existing FPSO Motion Reference Unit Data."*

## Status

| Component | State |
|-----------|-------|
| `core/` — `scr_twin_core` physics package | **Complete**: spectral, catenary/Morison `H(f)`, rainflow, S-N, Miner, spectral-damage, environment, Monte Carlo, Bayesian, ingestion, decision/economics, config, full-chain pipeline, validation. ~180 pytest tests, ruff + mypy clean; **9/9 acceptance gates pass**. |
| `server/` — FastAPI backend | REST + WebSocket exposing the core; typed Pydantic contracts; fuzz-safe ingestion; serves the built frontend offline. 12 API tests. |
| `app/` — React + TypeScript console | Instrument-grade dark console: MRU → H(f) → rainflow/S-N/Miner → posterior → decision, with the signature contracting posterior fan, live spectra, RBI + fleet economics, and an in-app validation view. Bespoke SVG charts, custom design system. |

Architecture note: the spec recommends a Tauri desktop shell (Option A). This
build uses **FastAPI + React served as one offline process** (spec Option C,
"reliability wins") because the Rust/Tauri toolchain is not provisioned here; the
core/server/app separation lets a Tauri wrapper be added later without touching
the physics.

## Architecture

```
core/                 pure, tested physics (no UI deps)  ── scr_twin_core
server/               FastAPI backend over the core (REST + WebSocket)
app/                  React + TypeScript console (bespoke SVG charts, dark HMI)
data/samples/         badged SYNTHETIC sample MRU + generator
.github/workflows/    CI: pytest (core+api) + ruff + mypy + gates + frontend build
run.ps1               one-command launcher
```

Non-negotiables: real physics only (every number traces to a cited equation),
deterministic (explicit seeds), must-not-crash (validated inputs), real data as
the primary path (synthetic is a labelled fallback).

## Run the console

One command (Windows PowerShell) — builds the frontend and serves the whole app
offline at <http://127.0.0.1:8000>:

```powershell
./run.ps1            # production: single process on :8000
./run.ps1 -Dev       # dev: backend :8000 + Vite HMR on :5173
```

Manual / cross-platform:

```bash
python -m venv .venv && .venv/Scripts/activate   # POSIX: source .venv/bin/activate
pip install -e core && pip install -r server/requirements.txt
(cd app && npm install && npm run build)          # build the console
uvicorn server.main:app --port 8000               # serve API + console
```

## Test & verify

```bash
pytest core/tests -q                   # ~180 physics unit + benchmark tests
pytest server/tests -q                 # API contract + fuzz + determinism tests
python -m scr_twin_core.validation     # print the §5 acceptance-gate report (9/9)
ruff check core/scr_twin_core && (cd core && mypy scr_twin_core)
python data/samples/generate_sample_mru.py   # regenerate the sample dataset
```

See [core/README.md](core/README.md) for the module map, physics references, and
how to plug in real MRU data and a project-specific transfer function.

## Acceptance gates (spec §5)

`python -m scr_twin_core.validation` checks each gate the core owns:

- JONSWAP `Hs` recovery from `4√m0`; catenary closed-form; DNV-RP-C203 S-N points.
- Dirlik / Tovo–Benasciutti / narrow-band / time-domain rainflow agreement.
- Arabian Gulf correction → 30–39% life reduction (combined factor 0.61–0.70).
- Year-15 design-vs-actual accumulated-damage divergence P10≈5% … P90≈28%.
- Bayesian 90% credible interval contracts as `1/√T` (halves by year 4).
- Determinism: identical seed+config → byte-identical output.
