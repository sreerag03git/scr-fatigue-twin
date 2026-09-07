"""Vessel Response Amplitude Operator (RAO) import + spectrum-driven motion.

In linear seakeeping the vessel motion that drives the SCR is the wave field
shaped by the vessel RAOs. The synthetic generator's built-in RAOs are
ILLUSTRATIVE; importing a validated 6-DOF RAO matrix (WAMIT / AQWA / OrcaFlex
vessel type, for a heading and draft) makes the whole motion input defensible:

    x_dof(t) = sum_i |RAO_dof(f_i)| * a_i * cos(2 pi f_i t + phi_i + arg RAO_dof(f_i)),
    a_i = sqrt(2 S_wave(f_i) df_i)

so the motion PSD is ``|RAO(f)|^2 S_wave(f)``. Translational RAOs are in m/m,
rotational RAOs in deg/m (the ingest / hang-off convention). This is the motion-side
analogue of the validated H(f) import (see :mod:`scr_twin_core.transfer`).

Reference
---------
DNV-RP-C205 Sec. 7 (RAOs and linear response); DNV-RP-F204 Sec. 4 (wave-frequency
riser fatigue via RAO x spectrum).
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spectral import jonswap

DOF_NAMES = ("heave", "surge", "sway", "pitch", "roll", "yaw")
_ANGLE_DOF = frozenset({"pitch", "roll", "yaw"})


@dataclass(frozen=True)
class RAOProvenance:
    """Where an imported vessel-RAO matrix came from (badge + report metadata)."""

    source_tool: str = "unknown"     # WAMIT / AQWA / OrcaFlex vessel type ...
    tool_version: str = ""
    load_case: str = ""              # e.g. "heading=180 draft=survival"
    heading_deg: float = 0.0
    notes: str = ""
    is_validated: bool = True
    n_points: int = 0
    freq_min_hz: float = 0.0
    freq_max_hz: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "source_tool": self.source_tool, "tool_version": self.tool_version,
            "load_case": self.load_case, "heading_deg": self.heading_deg, "notes": self.notes,
            "is_validated": self.is_validated, "n_points": self.n_points,
            "freq_min_hz": self.freq_min_hz, "freq_max_hz": self.freq_max_hz,
        }


class VesselRAO:
    """A validated 6-DOF vessel RAO table (complex, per DOF) for one heading.

    ``channels`` maps a subset of the six DOF to complex RAOs on ``freqs`` (missing
    DOF are treated as zero). Magnitude units: m/m for translations, deg/m for
    rotations.
    """

    def __init__(
        self, freqs: ArrayLike, channels: Mapping[str, ArrayLike],
        *, provenance: RAOProvenance | None = None,
    ) -> None:
        f = np.asarray(freqs, dtype=np.float64)
        if f.size < 2:
            raise ValueError("need at least two RAO frequency points")
        if not np.all(np.isfinite(f)):
            raise ValueError("RAO frequencies contain non-finite values")
        order = np.argsort(f)
        self._f = f[order]
        self._ch: dict[str, NDArray[np.complex128]] = {}
        for name, arr in channels.items():
            if name not in DOF_NAMES:
                raise ValueError(f"unknown DOF {name!r}; expected {DOF_NAMES}")
            c = np.asarray(arr, dtype=np.complex128)
            if c.shape != f.shape:
                raise ValueError(f"RAO channel {name!r} length {c.size} != {f.size} frequencies")
            if not np.all(np.isfinite(c)):
                raise ValueError(f"RAO channel {name!r} contains non-finite values")
            self._ch[name] = c[order]
        if not self._ch:
            raise ValueError("RAO table has no recognised DOF channels")
        base = provenance or RAOProvenance()
        self.provenance = RAOProvenance(
            source_tool=base.source_tool, tool_version=base.tool_version,
            load_case=base.load_case, heading_deg=base.heading_deg, notes=base.notes,
            is_validated=base.is_validated, n_points=int(self._f.size),
            freq_min_hz=float(self._f.min()), freq_max_hz=float(self._f.max()),
        )

    @property
    def dof(self) -> list[str]:
        return [d for d in DOF_NAMES if d in self._ch]

    def evaluate(self, freqs: ArrayLike, dof: str) -> NDArray[np.complex128]:
        """Complex RAO for ``dof`` interpolated onto ``freqs`` (zero if absent)."""
        f = np.asarray(freqs, dtype=np.float64)
        if dof not in self._ch:
            return np.zeros(f.shape, dtype=np.complex128)
        c = self._ch[dof]
        mag = np.interp(f, self._f, np.abs(c))
        ph = np.interp(f, self._f, np.unwrap(np.angle(c)))
        return (mag * np.exp(1j * ph)).astype(np.complex128)

    def synthesize(
        self, *, hs: float, tp: float, gamma: float = 3.3, duration: float, fs: float,
        seed: int, n_components: int = 200, f_low: float = 0.02, f_high: float = 0.5,
    ) -> dict[str, NDArray[np.float64]]:
        """Random-phase 6-DOF motion from a JONSWAP wave field shaped by the RAOs.

        Deterministic for a ``seed``. Returns channels compatible with the hang-off
        resolver (translations [m], rotations [deg]).
        """
        if duration <= 0.0 or fs <= 0.0:
            raise ValueError("duration and fs must be positive")
        rng = np.random.default_rng(seed)
        edges = np.linspace(f_low, f_high, n_components + 1)
        df = np.diff(edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        freqs = centres + (rng.random(n_components) - 0.5) * df
        amps = np.sqrt(2.0 * jonswap(freqs, hs, tp, gamma=gamma, normalize=True) * df)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
        t = np.arange(0.0, duration, 1.0 / fs)

        out: dict[str, NDArray[np.float64]] = {}
        for name in self.dof:
            rao = self.evaluate(freqs, name)
            mag, arg = np.abs(rao), np.angle(rao)
            pm = 2.0 * np.pi * np.outer(t, freqs) + (phases + arg)
            out[name] = (np.cos(pm) * (mag * amps)).sum(axis=1).astype(np.float64)
        return out


# Column aliases (lower-cased, stripped) for imported RAO CSVs.
_RAO_FREQ_COLS = {"freq", "freq_hz", "frequency", "frequency_hz", "f", "hz", "f_hz"}


def load_rao_csv(source: object, **overrides: object) -> VesselRAO:
    """Load a 6-DOF vessel RAO table from a CSV (magnitude + phase per DOF).

    Expected columns: a frequency column plus, per DOF, ``<dof>_mag`` and
    ``<dof>_phase`` (or ``<dof>_phase_deg``) for any of heave/surge/sway/pitch/
    roll/yaw. Phase is radians unless the column is named ``*_deg``. Provenance may
    be supplied as ``# key: value`` header lines (``source_tool``, ``tool_version``,
    ``load_case``, ``heading_deg``, ``notes``) or via keyword overrides. Rejected
    loudly on malformed input.
    """
    import pandas as pd

    meta: dict[str, str] = {}
    text: str | None = None
    if hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    elif isinstance(source, bytes):
        text = source.decode("utf-8", "replace")
    elif isinstance(source, str):
        text = source
    if text is not None:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#") and ":" in s:
                k, _, v = s.lstrip("#").strip().partition(":")
                if k.strip().lower() in {"source_tool", "tool_version", "load_case", "heading_deg", "notes"}:
                    meta[k.strip().lower()] = v.strip()

    try:
        df = pd.read_csv(io.StringIO(text) if text is not None else source, comment="#", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse RAO CSV: {exc}") from exc
    if df.empty:
        raise ValueError("RAO CSV has no data rows")
    df.columns = [str(c).strip().lower() for c in df.columns]

    freq_col = next((c for c in df.columns if c in _RAO_FREQ_COLS), None)
    if freq_col is None:
        raise ValueError(f"RAO CSV missing a frequency column (one of {sorted(_RAO_FREQ_COLS)})")
    freqs = df[freq_col].to_numpy(dtype=np.float64)

    channels: dict[str, NDArray[np.complex128]] = {}
    for dof in DOF_NAMES:
        mag_col = f"{dof}_mag"
        if mag_col not in df.columns:
            continue
        mag = df[mag_col].to_numpy(dtype=np.float64)
        ph_deg_col, ph_col = f"{dof}_phase_deg", f"{dof}_phase"
        if ph_deg_col in df.columns:
            phase = np.deg2rad(df[ph_deg_col].to_numpy(dtype=np.float64))
        elif ph_col in df.columns:
            phase = df[ph_col].to_numpy(dtype=np.float64)
        else:
            phase = np.zeros_like(mag)
        channels[dof] = (mag * np.exp(1j * phase)).astype(np.complex128)
    if not channels:
        raise ValueError("RAO CSV has no <dof>_mag columns for any of "
                         "heave/surge/sway/pitch/roll/yaw")

    def _s(key: str, default: str = "") -> str:
        v: Any = overrides.get(key, meta.get(key, default))
        return str(v) if v is not None else default

    _hd: Any = overrides.get("heading_deg", meta.get("heading_deg", 0.0))
    try:
        heading = float(_hd)
    except (TypeError, ValueError):
        heading = 0.0
    prov = RAOProvenance(
        source_tool=_s("source_tool", "imported RAO"), tool_version=_s("tool_version"),
        load_case=_s("load_case"), heading_deg=heading, notes=_s("notes"),
        is_validated=bool(overrides.get("is_validated", True)),
    )
    return VesselRAO(freqs, channels, provenance=prov)
