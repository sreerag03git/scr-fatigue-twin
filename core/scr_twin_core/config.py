"""Pydantic input/output schemas for the SCR fatigue twin.

These are the validated, serialisable contracts between the physics core and any
shell (desktop app, API, notebook). Every field carries an engineering bound so a
malformed configuration is rejected with a clear message rather than producing a
plausible-looking wrong answer. :meth:`RiserConfig.reference_scr` returns a
documented illustrative preset (clearly labelled, not project data).

Angle convention: ``hang_off_angle_deg`` is measured **from vertical** (the usual
SCR departure-angle convention, e.g. ~15-20 deg); it is converted to the
from-horizontal angle the catenary solver expects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .catenary import PlainCatenary, solve_plain_catenary
from .constants import E_STEEL
from .environment import (
    SALINITY_FACTOR_BOUNDS,
    TEMP_FACTOR_BOUNDS,
    EnvironmentCorrection,
)
from .section import PipeSection, submerged_weight
from .sn import DNV_C203_IN_AIR, MeanStressModel

SNClassName = Literal["B1", "B2", "C", "C1", "C2", "D", "E", "F", "F1", "F3", "G"]


class RiserConfig(BaseModel):
    """Steel catenary riser geometry, material and hot-spot definition."""

    model_config = {"extra": "forbid"}

    outer_diameter: float = Field(gt=0.0, le=2.0, description="Steel OD [m]")
    wall_thickness: float = Field(gt=0.0, le=0.2, description="Wall thickness [m]")
    youngs_modulus: float = Field(default=E_STEEL, gt=0.0, description="E [Pa]")
    material_grade: str = Field(default="API 5L X65", description="Line-pipe grade label")
    ultimate_strength: float = Field(
        default=531e6, gt=0.0, description="UTS [Pa] (X65 default), for mean-stress correction"
    )

    contents_density: float = Field(default=0.0, ge=0.0, le=20000.0, description="Bore fluid density [kg/m^3]")
    coating_thickness: float = Field(default=0.0, ge=0.0, le=0.5, description="Coating thickness [m]")
    coating_density: float = Field(default=0.0, ge=0.0, le=5000.0, description="Coating density [kg/m^3]")
    submerged_weight_override: float | None = Field(
        default=None, gt=0.0, description="If set, overrides the computed submerged weight [N/m]"
    )

    water_depth: float = Field(gt=0.0, le=4000.0, description="TDP-to-hangoff vertical distance [m]")
    hang_off_angle_deg: float = Field(
        gt=0.0, lt=89.0, description="Departure angle from vertical [deg]"
    )

    scf: float = Field(default=1.0, ge=1.0, le=10.0, description="Stress concentration factor")
    sn_class: SNClassName = Field(default="F1", description="DNV-RP-C203 S-N class")
    weld_thickness: float | None = Field(
        default=None, gt=0.0, description="Thickness for the (t/t_ref)^k correction [m]; defaults to wall_thickness"
    )
    mean_stress_model: MeanStressModel = Field(default=MeanStressModel.NONE)

    is_reference_preset: bool = Field(default=False, description="True marks an illustrative preset, not project data")

    @model_validator(mode="after")
    def _check_wall(self) -> RiserConfig:
        if self.wall_thickness >= self.outer_diameter / 2.0:
            raise ValueError("wall_thickness must be < OD/2")
        if self.sn_class not in DNV_C203_IN_AIR:
            raise ValueError(f"sn_class must be one of {sorted(DNV_C203_IN_AIR)}")
        return self

    @property
    def top_angle_from_horizontal_deg(self) -> float:
        return 90.0 - self.hang_off_angle_deg

    @property
    def thickness_for_correction(self) -> float:
        return self.weld_thickness if self.weld_thickness is not None else self.wall_thickness

    def pipe_section(self) -> PipeSection:
        return PipeSection(
            outer_diameter=self.outer_diameter,
            wall_thickness=self.wall_thickness,
            youngs_modulus=self.youngs_modulus,
        )

    def effective_submerged_weight(self) -> float:
        """Submerged weight per length [N/m] (override or computed from section)."""
        if self.submerged_weight_override is not None:
            return self.submerged_weight_override
        w = submerged_weight(
            self.pipe_section(),
            contents_density=self.contents_density,
            coating_thickness=self.coating_thickness,
            coating_density=self.coating_density,
        )
        if w <= 0.0:
            raise ValueError(
                "Computed submerged weight is non-positive (net-buoyant); set "
                "submerged_weight_override or revise the section/contents."
            )
        return w

    def catenary(self) -> PlainCatenary:
        return solve_plain_catenary(
            water_depth=self.water_depth,
            top_angle_deg=self.top_angle_from_horizontal_deg,
            submerged_weight=self.effective_submerged_weight(),
        )

    @classmethod
    def reference_scr(cls) -> RiserConfig:
        """Illustrative deep-water SCR preset (NOT project data).

        Representative of a 12.75 in (0.324 m) OD API 5L X65 steel catenary riser
        in ~1500 m water with a ~20 deg departure angle - a configuration in the
        range covered by Quéau et al. (2015)'s parametric SCR study. Values are
        illustrative defaults for demonstration only.
        """
        return cls(
            outer_diameter=0.3239,
            wall_thickness=0.0206,
            material_grade="API 5L X65",
            ultimate_strength=531e6,
            contents_density=800.0,
            coating_thickness=0.05,
            coating_density=900.0,
            water_depth=1500.0,
            hang_off_angle_deg=20.0,
            scf=1.15,
            sn_class="F1",
            mean_stress_model=MeanStressModel.NONE,
            is_reference_preset=True,
        )


class TransferConfig(BaseModel):
    """Layer-1 transfer-function options (Route 1 analytic parameters)."""

    model_config = {"extra": "forbid"}

    route: Literal["reference", "analytic", "imported"] = "reference"
    natural_frequency: float = Field(default=0.12, gt=0.0, le=2.0, description="TDP-region fn [Hz]")
    sigma_velocity: float = Field(default=0.3, ge=0.0, description="RMS relative velocity for drag linearisation [m/s]")
    drag_coefficient: float = Field(default=1.0, gt=0.0, le=3.0)
    added_mass_coefficient: float = Field(default=1.0, ge=0.0, le=3.0)
    structural_damping_ratio: float = Field(default=0.005, ge=0.0, le=0.5)


class EnvironmentConfig(BaseModel):
    """Arabian Gulf correction toggle and factors (with documented bounds)."""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    temperature_factor: float = Field(default=0.75, ge=TEMP_FACTOR_BOUNDS[0], le=TEMP_FACTOR_BOUNDS[1])
    salinity_factor: float = Field(default=0.875, ge=SALINITY_FACTOR_BOUNDS[0], le=SALINITY_FACTOR_BOUNDS[1])

    def correction(self) -> EnvironmentCorrection | None:
        if not self.enabled:
            return None
        return EnvironmentCorrection(
            temperature_factor=self.temperature_factor,
            salinity_factor=self.salinity_factor,
        )


class AnalysisConfig(BaseModel):
    """Top-level analysis configuration (deterministic; seed-driven)."""

    model_config = {"extra": "forbid"}

    riser: RiserConfig
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    block_duration_s: float = Field(default=1800.0, gt=0.0, description="Analysis block length [s]")
    n_monte_carlo: int = Field(default=10_000, ge=1, le=1_000_000)
    seed: int = Field(default=0, ge=0)
