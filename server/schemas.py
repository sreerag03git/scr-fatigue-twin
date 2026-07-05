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


class AnalyzeUploadRequest(BaseModel):
    config: AnalysisConfig
    token: str


class EconomicsParams(BaseModel):
    """Editable fleet-economics inputs (mirror of core EconomicsModel)."""

    inspection_cost_usd: float = 1.0e6
    baseline_interval_yr: float = 5.0
    rbi_interval_conservative_yr: float = 6.5
    rbi_interval_optimistic_yr: float = 10.0
    monitoring_capex_usd: float = 0.14e6
    monitoring_opex_usd_per_yr: float = 0.026e6
    failure_consequence_usd: float = 50.0e6
    pof_reduction_optimistic: float = 0.01
    horizon_yr: float = 20.0
    n_units: int = 20


class HealthResponse(BaseModel):
    status: str
    core_version: str
    numpy_version: str
    scipy_version: str
