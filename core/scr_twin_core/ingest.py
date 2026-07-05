"""MRU ingestion: parse -> validate -> resample -> detrend -> segment.

Real MRU recordings are the primary workflow. This pipeline is written to
**degrade gracefully**: any malformed, gappy, wrong-unit or empty input produces
a :class:`DataHealth` report with flags rather than an exception. Nothing is
silently discarded - every repair (gap fill, NaN interpolation, resample) is
recorded.

Canonical channels (SI, degrees for angles): ``heave`` [m], ``pitch`` [deg],
``surge`` [m], ``sway`` [m], ``roll`` [deg], ``yaw`` [deg]. Minimum useful set is
``heave`` (+ ``pitch``/``surge`` where present).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Column-name aliases (lower-cased, stripped) -> canonical channel name.
CHANNEL_ALIASES: dict[str, str] = {
    "heave": "heave", "heave_m": "heave", "z": "heave",
    "pitch": "pitch", "pitch_deg": "pitch", "pitch_rad": "pitch",
    "surge": "surge", "surge_m": "surge", "x": "surge",
    "sway": "sway", "sway_m": "sway", "y": "sway",
    "roll": "roll", "roll_deg": "roll",
    "yaw": "yaw", "yaw_deg": "yaw", "heading": "yaw",
}
TIME_ALIASES = {"time", "time_s", "t", "timestamp", "seconds", "sec"}

# Plausible RMS ranges for unit-sanity flags (not hard limits).
PLAUSIBLE_RMS = {
    "heave": (0.02, 15.0), "surge": (0.02, 20.0), "sway": (0.02, 20.0),
    "pitch": (0.05, 15.0), "roll": (0.05, 20.0), "yaw": (0.05, 30.0),
}


@dataclass(frozen=True)
class DataHealth:
    """Quality report for an ingested record. ``ok`` gates downstream analysis."""

    n_raw: int
    n_used: int
    fs_hz: float
    duration_s: float
    n_gaps: int
    max_gap_s: float
    nan_count: int
    clipped_fraction: float
    non_monotonic: bool
    channels: list[str]
    flags: list[str] = field(default_factory=list)
    ok: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "n_raw": self.n_raw, "n_used": self.n_used, "fs_hz": self.fs_hz,
            "duration_s": self.duration_s, "n_gaps": self.n_gaps,
            "max_gap_s": self.max_gap_s, "nan_count": self.nan_count,
            "clipped_fraction": self.clipped_fraction,
            "non_monotonic": self.non_monotonic, "channels": self.channels,
            "flags": self.flags, "ok": self.ok,
        }


@dataclass(frozen=True)
class IngestedMRU:
    """Uniformly-sampled, detrended MRU record split into analysis blocks."""

    time: NDArray[np.float64]
    channels: dict[str, NDArray[np.float64]]
    fs: float
    blocks: list[tuple[int, int]]
    health: DataHealth
    is_synthetic: bool = False

    @property
    def primary(self) -> NDArray[np.float64]:
        """The heave channel (the primary TDP-fatigue driver)."""
        return self.channels["heave"]


def _empty_health(n_raw: int, flags: list[str]) -> DataHealth:
    return DataHealth(
        n_raw=n_raw, n_used=0, fs_hz=0.0, duration_s=0.0, n_gaps=0, max_gap_s=0.0,
        nan_count=0, clipped_fraction=0.0, non_monotonic=False, channels=[],
        flags=flags, ok=False,
    )


def _clipped_fraction(x: NDArray[np.float64]) -> float:
    """Fraction of samples sitting at the signal's min or max (rail clipping)."""
    finite = x[np.isfinite(x)]
    if finite.size < 3:
        return 0.0
    lo, hi = finite.min(), finite.max()
    if hi <= lo:
        return 0.0
    at_rail = (np.isclose(x, lo) | np.isclose(x, hi)) & np.isfinite(x)
    # Only count rail values that repeat consecutively (a real clip, not an isolated peak).
    run = at_rail & np.concatenate(([False], at_rail[:-1]))
    return float(run.sum()) / float(x.size)


def ingest_mru(
    time: NDArray[np.float64] | None,
    raw_channels: dict[str, NDArray[np.float64]],
    *,
    assumed_fs: float = 4.0,
    block_duration_s: float = 1800.0,
    is_synthetic: bool = False,
) -> IngestedMRU:
    """Validate, resample to a uniform grid, detrend and segment MRU channels.

    Never raises on data content - problems are reported via
    :class:`DataHealth`. ``time`` may be ``None`` (then ``assumed_fs`` is used).
    """
    flags: list[str] = []
    n_raw = 0
    for v in raw_channels.values():
        n_raw = max(n_raw, np.asarray(v).size)

    # Canonicalise & coerce channels to float, dropping unknown names.
    channels: dict[str, NDArray[np.float64]] = {}
    for name, values in raw_channels.items():
        canon = CHANNEL_ALIASES.get(str(name).strip().lower())
        if canon is None:
            continue
        arr = np.asarray(values, dtype=np.float64).ravel()
        channels[canon] = arr

    if not channels:
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(n_raw, ["no recognised motion channels"]), is_synthetic)
    if "heave" not in channels:
        flags.append("no heave channel (primary driver missing)")
    if n_raw < 16:
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(n_raw, [*flags, "too few samples (<16)"]), is_synthetic)

    # --- Time base ---
    non_monotonic = False
    if time is None:
        flags.append(f"no timestamp column; assuming fs={assumed_fs} Hz")
        t = np.arange(n_raw, dtype=np.float64) / assumed_fs
    else:
        t = np.asarray(time, dtype=np.float64).ravel()
        if t.size != n_raw:
            n = min(t.size, n_raw)
            t = t[:n]
            channels = {k: v[:n] for k, v in channels.items()}
            n_raw = n
            flags.append("time/channel length mismatch; truncated to shortest")
        finite_t = np.isfinite(t)
        if not finite_t.all():
            flags.append("non-finite timestamps present")
        if np.any(np.diff(t[finite_t]) <= 0):
            non_monotonic = True
            flags.append("non-monotonic time; sorted and de-duplicated")
            order = np.argsort(t, kind="stable")
            t = t[order]
            channels = {k: v[order] for k, v in channels.items()}
            keep = np.concatenate(([True], np.diff(t) > 0))
            t = t[keep]
            channels = {k: v[keep] for k, v in channels.items()}
            n_raw = t.size

    if t.size < 16:
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(n_raw, [*flags, "too few valid time samples"]), is_synthetic)

    # --- Sample rate & gaps ---
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(n_raw, [*flags, "degenerate time base"]), is_synthetic)
    median_dt = float(np.median(dt))
    fs = 1.0 / median_dt if median_dt > 0 else assumed_fs
    gap_mask = dt > 1.5 * median_dt
    n_gaps = int(gap_mask.sum())
    max_gap_s = float(dt.max()) if dt.size else 0.0
    if n_gaps:
        flags.append(f"{n_gaps} gap(s) up to {max_gap_s:.1f}s; filled by interpolation")

    # --- NaN handling + clipping + unit sanity (per channel) ---
    nan_count = 0
    clipped = 0.0
    for name, arr in channels.items():
        n_nan = int(np.count_nonzero(~np.isfinite(arr)))
        nan_count += n_nan
        clipped = max(clipped, _clipped_fraction(arr))
        rms = float(np.sqrt(np.nanmean(arr[np.isfinite(arr)] ** 2))) if np.isfinite(arr).any() else 0.0
        lo, hi = PLAUSIBLE_RMS.get(name, (0.0, np.inf))
        if rms > 0 and not (lo <= rms <= hi):
            flags.append(f"{name} RMS={rms:.3g} outside plausible [{lo},{hi}] (units?)")
    if nan_count:
        flags.append(f"{nan_count} NaN sample(s) interpolated")
    if clipped > 0.01:
        flags.append(f"clipping detected ({clipped:.1%} of samples at a rail)")

    # --- Resample to a uniform grid at fs (fills gaps, repairs NaNs) ---
    t0, t1 = float(t[0]), float(t[-1])
    n_uniform = max(16, int(round((t1 - t0) * fs)) + 1)
    uni_t = t0 + np.arange(n_uniform) / fs
    out: dict[str, NDArray[np.float64]] = {}
    for name, arr in channels.items():
        valid = np.isfinite(t) & np.isfinite(arr)
        if valid.sum() < 2:
            out[name] = np.zeros(n_uniform)
            continue
        resampled = np.interp(uni_t, t[valid], arr[valid])
        resampled = resampled - np.mean(resampled)  # de-mean (detrend constant)
        out[name] = resampled.astype(np.float64)

    duration = float(uni_t[-1] - uni_t[0])

    # --- Segment into analysis blocks ---
    block_len = max(16, int(round(block_duration_s * fs)))
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < n_uniform:
        end = min(start + block_len, n_uniform)
        if end - start >= 16:
            blocks.append((start, end))
        start = end
    if not blocks:
        blocks = [(0, n_uniform)]

    health = DataHealth(
        n_raw=n_raw, n_used=n_uniform, fs_hz=fs, duration_s=duration,
        n_gaps=n_gaps, max_gap_s=max_gap_s, nan_count=nan_count,
        clipped_fraction=clipped, non_monotonic=non_monotonic,
        channels=sorted(out), flags=flags, ok="heave" in out and n_uniform >= 16,
    )
    return IngestedMRU(uni_t.astype(np.float64), out, fs, blocks, health, is_synthetic)


def _dataframe_to_arrays(df: object) -> tuple[NDArray[np.float64] | None, dict[str, NDArray[np.float64]]]:
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    time_arr: NDArray[np.float64] | None = None
    channels: dict[str, NDArray[np.float64]] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if key in TIME_ALIASES and time_arr is None:
            time_arr = series
        elif key in CHANNEL_ALIASES:
            channels[key] = series
    # Fallback: if no named time col but first column looks monotonic, treat it as time.
    if time_arr is None and len(df.columns) >= 2:
        first = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
        if np.all(np.diff(first[np.isfinite(first)]) > 0):
            time_arr = first
    return time_arr, channels


def load_mru_csv(source: object, **kwargs: object) -> IngestedMRU:
    """Load and ingest an MRU CSV (path or file-like). Comment lines (``#``) skipped."""
    import pandas as pd

    try:
        df = pd.read_csv(source, comment="#", skip_blank_lines=True)
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(0, [f"CSV parse error: {exc}"]), False)
    if df.empty:
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(0, ["empty file"]), False)
    time_arr, channels = _dataframe_to_arrays(df)
    return ingest_mru(time_arr, channels, **kwargs)  # type: ignore[arg-type]


def load_mru_parquet(source: object, **kwargs: object) -> IngestedMRU:
    """Load and ingest an MRU Parquet file (path or file-like)."""
    import pandas as pd

    try:
        df = pd.read_parquet(source)
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return IngestedMRU(np.empty(0), {}, 0.0, [], _empty_health(0, [f"Parquet parse error: {exc}"]), False)
    time_arr, channels = _dataframe_to_arrays(df)
    return ingest_mru(time_arr, channels, **kwargs)  # type: ignore[arg-type]
