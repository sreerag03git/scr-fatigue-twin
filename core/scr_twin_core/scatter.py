"""Long-term wave-scatter-diagram fatigue summation (DNV-RP-C203 Sec. 5).

Real SCR design fatigue is the sum of the damage from every sea state the riser
sees over its life, weighted by how often each occurs::

    D_annual = sum_ij  p_ij * D_rate(Hs_i, Tp_j)

where ``p_ij`` is the fraction of time spent in scatter cell ``(Hs_i, Tp_j)`` (the
occurrence probabilities sum to 1) and ``D_rate`` is the annual damage rate for
continuous exposure to that sea state (from the same JONSWAP -> transfer ->
Dirlik chain the single-sea-state pipeline already uses). This module is the pure
aggregation layer; the per-cell ``rate_fn`` (the sea-state physics) is supplied by
the caller so this stays framework-free and unit-testable.

The default :func:`example_scatter_diagram` is a documented ILLUSTRATIVE climate,
not project metocean data; :func:`load_scatter_csv` ingests a real table.

Reference
---------
DNV-RP-C203 Sec. 5 (long-term stress-range distribution as a sum over sea states);
DNV-RP-F204 (riser fatigue); Palmgren-Miner linear summation.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# rate_fn(hs, tp) -> annual damage rate [1/yr] for continuous exposure.
RateFn = Callable[[float, float], float]


@dataclass(frozen=True)
class ScatterCell:
    """One sea-state bin: significant wave height, peak period, occurrence prob."""

    hs: float
    tp: float
    probability: float


@dataclass(frozen=True)
class CellContribution:
    """Per-cell fatigue contribution for the driver heat map."""

    hs: float
    tp: float
    probability: float
    annual_rate: float  # damage rate if exposed continuously [1/yr]
    damage_fraction: float  # share of the total long-term annual damage


@dataclass(frozen=True)
class LongTermFatigue:
    """Long-term (scatter-summed) fatigue result."""

    annual_damage_rate: float
    life_years: float
    n_cells: int
    contributions: list[CellContribution]

    def as_dict(self) -> dict[str, object]:
        return {
            "annual_damage_rate": self.annual_damage_rate,
            "life_years": self.life_years,
            "n_cells": self.n_cells,
            "contributions": [
                {
                    "hs": c.hs, "tp": c.tp, "probability": c.probability,
                    "annual_rate": c.annual_rate, "damage_fraction": c.damage_fraction,
                }
                for c in self.contributions
            ],
        }


class ScatterDiagram:
    """A wave scatter diagram: sea-state cells with occurrence probabilities.

    Probabilities must be non-negative; they are renormalised to sum to 1 so a
    raw table given as occurrence counts, fractions, or percentages all work. A
    zero or negative total, or a non-positive Hs/Tp, is rejected loudly.
    """

    def __init__(self, cells: list[ScatterCell], *, source: str = "unspecified") -> None:
        if not cells:
            raise ValueError("scatter diagram has no cells")
        probs = np.array([c.probability for c in cells], dtype=np.float64)
        if np.any(probs < 0.0):
            raise ValueError("scatter probabilities must be non-negative")
        total = float(probs.sum())
        if total <= 0.0:
            raise ValueError("scatter probabilities sum to zero")
        for c in cells:
            if c.hs <= 0.0 or c.tp <= 0.0:
                raise ValueError("scatter Hs and Tp must be positive")
        self._cells = [
            ScatterCell(c.hs, c.tp, c.probability / total) for c in cells
        ]
        self.source = source

    @property
    def cells(self) -> list[ScatterCell]:
        return list(self._cells)

    def hs_values(self) -> list[float]:
        return sorted({c.hs for c in self._cells})

    def tp_values(self) -> list[float]:
        return sorted({c.tp for c in self._cells})


def aggregate_long_term(diagram: ScatterDiagram, rate_fn: RateFn) -> LongTermFatigue:
    """Sum the probability-weighted per-cell damage rates (DNV-RP-C203 Sec. 5).

    ``rate_fn(hs, tp)`` returns the annual damage rate for continuous exposure to
    that sea state; the long-term annual rate is ``sum_ij p_ij * rate_ij`` and the
    life is its reciprocal.
    """
    cells = diagram.cells
    rates = np.array([max(rate_fn(c.hs, c.tp), 0.0) for c in cells], dtype=np.float64)
    weighted = np.array([c.probability for c in cells]) * rates
    annual = float(weighted.sum())
    life = float("inf") if annual <= 0.0 else 1.0 / annual
    contribs = [
        CellContribution(
            hs=c.hs, tp=c.tp, probability=c.probability, annual_rate=float(r),
            damage_fraction=float(w / annual) if annual > 0.0 else 0.0,
        )
        for c, r, w in zip(cells, rates, weighted, strict=True)
    ]
    contribs.sort(key=lambda c: c.damage_fraction, reverse=True)
    return LongTermFatigue(
        annual_damage_rate=annual, life_years=life, n_cells=len(cells),
        contributions=contribs,
    )


def example_scatter_diagram() -> ScatterDiagram:
    """A documented ILLUSTRATIVE deep-water wave scatter diagram (NOT project data).

    A coarse Hs-Tp occurrence table with the bulk of the time in moderate seas and
    a decaying storm tail - representative in shape of a benign deep-water site.
    Occurrence values are illustrative; supply a project scatter table for real
    work via :func:`load_scatter_csv`.
    """
    # (Hs [m], Tp [s], occurrence [-]); renormalised in the constructor.
    rows = [
        (0.75, 5.0, 0.06), (0.75, 7.0, 0.10), (0.75, 9.0, 0.05),
        (1.5, 6.0, 0.09), (1.5, 8.0, 0.16), (1.5, 10.0, 0.10), (1.5, 12.0, 0.04),
        (2.5, 7.0, 0.06), (2.5, 9.0, 0.12), (2.5, 11.0, 0.07), (2.5, 13.0, 0.02),
        (3.5, 8.0, 0.02), (3.5, 10.0, 0.05), (3.5, 12.0, 0.03),
        (5.0, 10.0, 0.02), (5.0, 12.0, 0.015), (5.0, 14.0, 0.008),
        (7.0, 12.0, 0.004), (7.0, 14.0, 0.002),
    ]
    cells = [ScatterCell(hs, tp, p) for hs, tp, p in rows]
    return ScatterDiagram(cells, source="illustrative deep-water (NOT project data)")


_HS_COLS = {"hs", "hs_m", "hsig", "sig_wave_height", "significant_wave_height"}
_TP_COLS = {"tp", "tp_s", "peak_period", "tpeak"}
_PROB_COLS = {"prob", "probability", "occurrence", "p", "freq", "frequency", "weight"}


def load_scatter_csv(source: bytes | str, *, sep: str | None = None) -> ScatterDiagram:
    """Parse a scatter-diagram CSV with columns for Hs, Tp and occurrence.

    Accepts common column aliases (case-insensitive). ``# key: value`` header
    lines are ignored except that a ``# source:`` line is carried as provenance.
    Raises ``ValueError`` loudly on a malformed table.
    """
    text = source.decode("utf-8") if isinstance(source, bytes) else source
    provenance = "imported scatter table"
    data_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if body.lower().startswith("source:"):
                provenance = body.split(":", 1)[1].strip()
            continue
        data_lines.append(line)
    if not data_lines:
        raise ValueError("scatter CSV has no data rows")

    import pandas as pd

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=sep, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    def pick(aliases: set[str]) -> str:
        for col in df.columns:
            if col in aliases:
                return col
        raise ValueError(f"scatter CSV missing a column in {sorted(aliases)}; got {list(df.columns)}")

    hs_c, tp_c, p_c = pick(_HS_COLS), pick(_TP_COLS), pick(_PROB_COLS)
    cells = [
        ScatterCell(float(r[hs_c]), float(r[tp_c]), float(r[p_c]))
        for _, r in df.iterrows()
        if np.isfinite(r[hs_c]) and np.isfinite(r[tp_c]) and np.isfinite(r[p_c])
    ]
    if not cells:
        raise ValueError("scatter CSV parsed to zero valid cells")
    return ScatterDiagram(cells, source=provenance)
