"""API request/response Pydantic schemas (typed contract with the frontend)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from scr_twin_core.config import AnalysisConfig


class SyntheticParams(BaseModel):
    """Parameters for the SYNTHETIC MRU generator (demo/fallback path)."""

    hs: float = Field(default=4.0, gt=0.0, le=20.0)
    tp: float = Field(default=11.0, gt=0.0, le=30.0)
    gamma: float = Field(default=2.5, ge=1.0, le=7.0)
    duration: float = Field(default=1800.0, gt=0.0, le=10800.0)
    fs: float = Field(default=4.0, gt=0.0, le=20.0)
    seed: int = Field(default=20240705, ge=0)


class AnalyzeSyntheticRequest(BaseModel):
    config: AnalysisConfig
    synthetic: SyntheticParams = Field(default_factory=SyntheticParams)
    transfer_token: str | None = None  # required when config.transfer.route == "imported"


class AnalyzeUploadRequest(BaseModel):
    config: AnalysisConfig
    token: str
    transfer_token: str | None = None  # required when config.transfer.route == "imported"


class EconomicsParams(BaseModel):
    """Editable conditional-economics inputs (mirror of core ConditionalEconomicsModel).

    ``phi`` (optional) fixes the fleet operating point P(asset ages slower than
    design); leave it null to report at the break-even phi.
    """

    discount_rate: float = 0.08
    inspection_cost_usd: float = 1.0e6
    baseline_interval_yr: float = 5.0
    cbm_interval_slow_yr: float = 8.0
    cbm_interval_fast_yr: float = 4.0
    sensor_capex_usd: float = 0.14e6
    sensor_opex_usd_per_yr: float = 0.026e6
    horizon_yr: float = 20.0
    n_units: int = 20
    cost_cov: float = 0.25
    phi: float | None = None


class HealthResponse(BaseModel):
    status: str
    core_version: str
    numpy_version: str
    scipy_version: str
