"""Physical constants used throughout :mod:`scr_twin_core`.

All values are SI unless explicitly stated. Constants are documented with a
source so that every downstream number is traceable (see project principle:
"real physics only").
"""

from __future__ import annotations

# Standard gravity (CODATA / ISO 80000-3), m/s^2.
G: float = 9.80665

# Seawater density, kg/m^3. DNV-RP-C205 recommends 1025 kg/m^3 for open sea.
RHO_SEAWATER: float = 1025.0

# Air density at ~15 degC, kg/m^3 (used only for above-water segments).
RHO_AIR: float = 1.225

# Young's modulus of structural / API 5L line-pipe steel, Pa (DNV-OS-F201).
E_STEEL: float = 2.07e11

# Density of steel, kg/m^3.
RHO_STEEL: float = 7850.0

# Reference wall thickness for the DNV-RP-C203 thickness-correction term, m.
# Tubular joints / girth welds use 25 mm (DNV-RP-C203 Sec. 2.4.3).
T_REF_DNV: float = 0.025
