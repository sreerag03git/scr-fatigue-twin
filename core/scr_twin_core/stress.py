"""TDP hot-spot stress reconstruction (project spec 4.3).

Turns hang-off motion into touchdown-point (TDP) hot-spot stress via the Layer-1
transfer function H(f) and the pipe section modulus, applying the stress-
concentration factor (SCF). Two consistent pathways are offered:

- time domain: filter the measured motion history through H(f) (FFT), giving a
  moment history and hence a stress history to rainflow-count;
- spectral: propagate the motion PSD through ``|H(f)|^2`` to a stress PSD whose
  spectral moments feed the frequency-domain damage methods.

Hot-spot stress = SCF * moment / Z, with Z the elastic section modulus.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .section import PipeSection
from .transfer import TransferFunction


def rfft_frequencies(n_samples: int, fs: float) -> NDArray[np.float64]:
    """One-sided FFT frequency grid for ``n_samples`` at sample rate ``fs``."""
    return np.fft.rfftfreq(n_samples, d=1.0 / fs).astype(np.float64)


def moment_history(
    motion: ArrayLike, fs: float, tf: TransferFunction
) -> NDArray[np.float64]:
    """TDP bending-moment history [N m] from a hang-off motion history.

    ``M(t) = IFFT{ H(f) * FFT[x(t)] }``. ``tf`` must be sampled on the one-sided
    FFT grid of the motion (see :func:`rfft_frequencies`), so both magnitude and
    phase of H(f) are applied.
    """
    x = np.asarray(motion, dtype=np.float64).ravel()
    n = x.size
    freqs = rfft_frequencies(n, fs)
    if tf.freqs.shape != freqs.shape or not np.allclose(tf.freqs, freqs, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "TransferFunction must be sampled on rfft_frequencies(len(motion), fs). "
            f"Expected {freqs.size} bins from 0 to {freqs[-1]:.4f} Hz."
        )
    spectrum = np.fft.rfft(x)
    moment = np.fft.irfft(spectrum * tf.value, n=n)
    return moment.astype(np.float64)


def stress_from_moment(
    moment: ArrayLike, section: PipeSection, *, scf: float = 1.0
) -> NDArray[np.float64]:
    """Hot-spot stress [Pa] from a moment history: ``sigma = SCF * M / Z``."""
    if scf <= 0.0:
        raise ValueError("scf must be positive")
    m = np.asarray(moment, dtype=np.float64)
    return (scf * m / section.section_modulus).astype(np.float64)


def stress_history_from_motion(
    motion: ArrayLike, fs: float, tf: TransferFunction, section: PipeSection, *, scf: float = 1.0
) -> NDArray[np.float64]:
    """Convenience: motion history -> TDP hot-spot stress history [Pa]."""
    return stress_from_moment(moment_history(motion, fs, tf), section, scf=scf)


def stress_psd_from_motion_psd(
    motion_psd: ArrayLike, tf: TransferFunction, section: PipeSection, *, scf: float = 1.0
) -> NDArray[np.float64]:
    """Hot-spot stress PSD [Pa^2/Hz] from a motion PSD.

    ``S_sigma(f) = (SCF / Z)^2 |H(f)|^2 S_motion(f)`` (linear random-vibration
    input-output relation combined with the moment-to-stress conversion).
    """
    if scf <= 0.0:
        raise ValueError("scf must be positive")
    s = np.asarray(motion_psd, dtype=np.float64)
    if s.shape != tf.freqs.shape:
        raise ValueError("motion_psd must be sampled on the transfer-function grid")
    gain = (scf / section.section_modulus) ** 2
    return (gain * tf.magnitude**2 * s).astype(np.float64)


def random_phase_timeseries(
    freqs: ArrayLike,
    onesided_psd: ArrayLike,
    *,
    fs: float,
    n_samples: int,
    seed: int,
) -> NDArray[np.float64]:
    """Reconstruct a zero-mean Gaussian time series with a target one-sided PSD.

    Random-phase spectral synthesis: component amplitudes ``a_k = sqrt(2 S df)``
    with uniform random phases, assembled via the inverse real FFT. The returned
    signal has (in expectation) the supplied PSD; deterministic for a ``seed``.
    ``freqs`` must be ``rfft_frequencies(n_samples, fs)``.
    """
    f_target = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(onesided_psd, dtype=np.float64)
    grid = rfft_frequencies(n_samples, fs)
    if f_target.shape != grid.shape or not np.allclose(f_target, grid, rtol=1e-6, atol=1e-9):
        raise ValueError("freqs must equal rfft_frequencies(n_samples, fs)")

    df = fs / n_samples
    amp = np.sqrt(np.clip(2.0 * psd * df, 0.0, None))
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=grid.size)

    # A real cosine component of amplitude a maps to |rfft bin| = a * n/2.
    spectrum = (n_samples / 2.0) * amp * np.exp(1j * phases)
    spectrum[0] = 0.0  # zero mean
    if n_samples % 2 == 0:
        spectrum[-1] = np.real(spectrum[-1])  # Nyquist bin must be real
    return np.fft.irfft(spectrum, n=n_samples).astype(np.float64)
