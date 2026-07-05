"""Layer 3b: Bayesian updating of the damage rate as monitoring accrues.

A conjugate Normal-Normal model on the annual damage rate ``lambda``:

    prior:       lambda ~ Normal(mu0, tau0^2)
    each block:  y_i = lambda + eps_i,   eps_i ~ Normal(0, s^2)   (a noisy per-block
                                                                   rate estimate)
    posterior after n blocks:
        precision  = 1/tau0^2 + n/s^2
        mean       = (mu0/tau0^2 + sum(y_i)/s^2) / precision
        std        = 1/sqrt(precision)

With a weakly-informative prior the posterior std -> s/sqrt(n); since the number
of blocks grows linearly with monitoring time ``T``, the 90% credible-interval
width contracts as ``1/sqrt(T)`` and halves by year 4 relative to year 1 (project
spec 4.6 / acceptance 5). This is genuine posterior updating from observed
per-block damage, not a cosmetic shrink.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-value for a two-sided 90% credible interval (5%/95% of the standard normal).
Z90: float = 1.6448536269514722


@dataclass(frozen=True)
class RatePosterior:
    """Posterior on the annual damage rate and the implied remaining life."""

    mean: float
    std: float
    ci90_low: float
    ci90_high: float
    n_blocks: int

    @property
    def ci90_width(self) -> float:
        return self.ci90_high - self.ci90_low

    def remaining_life(self, accumulated_damage: float = 0.0) -> tuple[float, float, float]:
        """Remaining life ``(low, median, high)`` [yr] from ``(1 - D)/lambda``.

        The rate CI maps inversely to life, so the rate upper bound gives the
        life lower bound. ``accumulated_damage`` is the fraction already spent.
        """
        remaining = max(0.0, 1.0 - accumulated_damage)
        hi_rate = max(self.ci90_high, 1e-30)
        lo_rate = max(self.ci90_low, 1e-30)
        med_rate = max(self.mean, 1e-30)
        return (remaining / hi_rate, remaining / med_rate, remaining / lo_rate)


class BayesianRateEstimator:
    """Sequential conjugate updater for the annual damage rate.

    Parameters
    ----------
    prior_mean:
        Prior mean annual damage rate ``mu0`` [1/yr] (e.g. the Monte Carlo P50).
    prior_std:
        Prior std ``tau0``. Default is deliberately weak (10x the mean) so a
        year of data dominates and the CI scales as ``1/sqrt(T)`` from year 1.
    block_obs_std:
        Per-block observation std ``s`` of the annual-rate estimate.
    """

    def __init__(self, prior_mean: float, block_obs_std: float, *, prior_std: float | None = None) -> None:
        if block_obs_std <= 0.0:
            raise ValueError("block_obs_std must be positive")
        if prior_mean < 0.0:
            raise ValueError("prior_mean must be non-negative")
        self._mu0 = prior_mean
        self._tau0 = prior_std if prior_std is not None else max(prior_mean * 10.0, 1e-9)
        self._s = block_obs_std
        self._n = 0
        self._sum_y = 0.0

    def update_block(self, observed_rate: float) -> None:
        """Ingest one per-block annual-rate estimate ``y_i``."""
        self._sum_y += float(observed_rate)
        self._n += 1

    def update_from_block_damage(self, block_damage: float, block_seconds: float) -> None:
        """Ingest a block by its damage and duration (converts to an annual rate)."""
        if block_seconds <= 0.0:
            raise ValueError("block_seconds must be positive")
        seconds_per_year = 365.25 * 24.0 * 3600.0
        self.update_block(block_damage / block_seconds * seconds_per_year)

    def posterior(self) -> RatePosterior:
        """Current posterior on the annual damage rate."""
        precision = 1.0 / self._tau0**2 + self._n / self._s**2
        mean = (self._mu0 / self._tau0**2 + self._sum_y / self._s**2) / precision
        std = 1.0 / math.sqrt(precision)
        return RatePosterior(
            mean=mean,
            std=std,
            ci90_low=mean - Z90 * std,
            ci90_high=mean + Z90 * std,
            n_blocks=self._n,
        )
