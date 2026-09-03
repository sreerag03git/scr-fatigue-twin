"""Structural reliability: fatigue reliability index beta and annual Pf.

Turns the Monte-Carlo remaining-life posterior into a reliability statement the
way DNV-RP-C210 / DNV-OS-F201 require: a reliability index ``beta``, a cumulative
and annual probability of failure, and a check against the target for the
selected safety class - plus FORM importance factors showing which uncertainty
source dominates.

The fatigue limit state is ``g = ln(L*Delta) - ln(T)`` where ``L`` is the fatigue
life, ``Delta`` the Miner capacity at failure (lognormal, median 1, COV ~0.3) and
``T`` the elapsed time. Because the basis is lognormal, ``g`` is normal and FORM
is exact: ``beta = (mu_lnL + mu_lnDelta - ln T)/sqrt(sigma_lnL^2 + sigma_lnDelta^2)``
and ``Pf = Phi(-beta)``. The life mean is shifted from the *characteristic*
(design, mean-minus-2-std) S-N basis to the *mean* S-N basis (DNV-RP-C210), so the
reliability is not double-conservative; the deterministic DFF check keeps using
the characteristic curve.

Reference
---------
DNV-RP-C210 (probabilistic methods for inspection planning; target Pf by safety
class); DNV-OS-F201 App. (safety classes); Madsen, Krenk & Lind, "Methods of
Structural Safety" (FORM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

# DNV target ANNUAL probability of failure by safety class (RP-C210 / OS-F201).
SAFETY_CLASS_TARGET_PF: dict[str, float] = {
    "low": 1.0e-3,
    "normal": 1.0e-4,
    "high": 1.0e-5,
}


@dataclass(frozen=True)
class ReliabilityResult:
    """Fatigue reliability outcome for a design life and safety class."""

    beta: float                 # reliability index at the design life
    pf_cumulative: float        # P(failure by the design life)
    pf_annual: float            # marginal annual Pf in the final year
    design_life_years: float
    safety_class: str
    target_pf: float            # target annual Pf for the class
    target_beta: float          # corresponding target index
    passes: bool                # pf_annual <= target_pf
    mean_curve_life_years: float  # median life on the mean S-N basis
    importance: dict[str, float]  # FORM alpha^2 importance factors (sum to 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "pf_cumulative": self.pf_cumulative,
            "pf_annual": self.pf_annual,
            "design_life_years": self.design_life_years,
            "safety_class": self.safety_class,
            "target_pf": self.target_pf,
            "target_beta": self.target_beta,
            "passes": self.passes,
            "mean_curve_life_years": self.mean_curve_life_years,
            "importance": self.importance,
        }


def _importance_factors(parameters: dict[str, float], miner_var: float) -> dict[str, float]:
    """FORM alpha^2 variance-share of ln(life) per uncertainty source."""
    ln10 = math.log(10.0)
    m = float(parameters.get("sn_slope_m", 3.0))
    var = {
        "S-N scatter": (ln10 * float(parameters.get("sn_logN_std", 0.2))) ** 2,
        "Transfer gain": m**2 * math.log(1.0 + float(parameters.get("tf_gain_cov", 0.1)) ** 2),
        "SCF": m**2 * math.log(1.0 + float(parameters.get("scf_cov", 0.1)) ** 2),
        "Environment": float(parameters.get("env_factor_cov", 0.05)) ** 2,
        "Miner capacity": miner_var,
    }
    total = sum(var.values())
    if total <= 0.0:
        return {k: 0.0 for k in var}
    return {k: v / total for k, v in var.items()}


def form_fatigue_reliability(
    life_samples: ArrayLike,
    design_life_years: float,
    parameters: dict[str, float],
    *,
    miner_cov: float = 0.3,
    safety_class: str = "normal",
) -> ReliabilityResult:
    """FORM fatigue reliability from the MC life posterior (DNV-RP-C210).

    ``life_samples`` are on the characteristic (design) S-N basis; the mean is
    shifted to the mean S-N basis by ``+2*sigma_logN*ln10`` before computing beta.
    """
    if safety_class not in SAFETY_CLASS_TARGET_PF:
        raise ValueError(f"safety_class must be one of {sorted(SAFETY_CLASS_TARGET_PF)}")
    life = np.asarray(life_samples, dtype=np.float64)
    life = life[np.isfinite(life) & (life > 0.0)]
    if life.size < 2:
        raise ValueError("need at least 2 positive, finite life samples")
    if design_life_years <= 0.0:
        raise ValueError("design_life_years must be positive")

    ln_life = np.log(life)
    sigma_l = float(np.std(ln_life))
    sigma_logn = float(parameters.get("sn_logN_std", 0.2))
    # Characteristic -> mean S-N basis (mean = characteristic + 2 std of log10 N).
    mu_l = float(np.mean(ln_life)) + 2.0 * sigma_logn * math.log(10.0)
    mean_curve_life = math.exp(mu_l)

    sigma_delta_sq = math.log(1.0 + miner_cov**2)
    sigma_tot = math.sqrt(sigma_l**2 + sigma_delta_sq)

    def beta_at(t: float) -> float:
        return (mu_l - math.log(t)) / sigma_tot

    beta = beta_at(design_life_years)
    pf_cum = float(norm.cdf(-beta))
    t_prev = max(design_life_years - 1.0, 1e-6)
    pf_annual = max(pf_cum - float(norm.cdf(-beta_at(t_prev))), 0.0)

    target_pf = SAFETY_CLASS_TARGET_PF[safety_class]
    target_beta = float(norm.ppf(1.0 - target_pf))
    importance = _importance_factors(parameters, sigma_delta_sq)

    return ReliabilityResult(
        beta=beta,
        pf_cumulative=pf_cum,
        pf_annual=pf_annual,
        design_life_years=design_life_years,
        safety_class=safety_class,
        target_pf=target_pf,
        target_beta=target_beta,
        passes=bool(pf_annual <= target_pf),
        mean_curve_life_years=mean_curve_life,
        importance=importance,
    )
