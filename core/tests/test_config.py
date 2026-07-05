"""Config schema tests: validation bounds, reference preset, derived objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scr_twin_core.config import (
    AnalysisConfig,
    EnvironmentConfig,
    RiserConfig,
)


def test_reference_preset_is_valid_and_labelled():
    cfg = RiserConfig.reference_scr()
    assert cfg.is_reference_preset is True
    # derived engineering objects build without error
    sec = cfg.pipe_section()
    assert sec.outer_diameter == 0.3239
    cat = cfg.catenary()
    assert cat.tdp_curvature > 0.0
    assert cfg.effective_submerged_weight() > 0.0


def test_angle_convention_conversion():
    cfg = RiserConfig.reference_scr()
    # 20 deg from vertical -> 70 deg from horizontal
    assert cfg.top_angle_from_horizontal_deg == pytest.approx(70.0)


def test_wall_thickness_bound_enforced():
    with pytest.raises(ValidationError):
        RiserConfig(outer_diameter=0.3, wall_thickness=0.2, water_depth=1000, hang_off_angle_deg=20)


def test_out_of_range_fields_rejected():
    with pytest.raises(ValidationError):
        RiserConfig(outer_diameter=-1, wall_thickness=0.02, water_depth=1000, hang_off_angle_deg=20)
    with pytest.raises(ValidationError):
        RiserConfig(outer_diameter=0.3, wall_thickness=0.02, water_depth=1000, hang_off_angle_deg=95)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        RiserConfig(
            outer_diameter=0.3, wall_thickness=0.02, water_depth=1000,
            hang_off_angle_deg=20, bogus_field=1.0,
        )


def test_environment_config_bounds_and_toggle():
    ec = EnvironmentConfig()
    corr = ec.correction()
    assert corr is not None
    assert 0.61 <= corr.combined_factor <= 0.70
    assert EnvironmentConfig(enabled=False).correction() is None
    with pytest.raises(ValidationError):
        EnvironmentConfig(temperature_factor=0.5)


def test_analysis_config_round_trip_json():
    cfg = AnalysisConfig(riser=RiserConfig.reference_scr(), seed=7)
    restored = AnalysisConfig.model_validate_json(cfg.model_dump_json())
    assert restored.seed == 7
    assert restored.riser.outer_diameter == cfg.riser.outer_diameter


def test_net_buoyant_config_raises_on_weight():
    # Thin-wall, light contents, thick buoyant coating -> net buoyant.
    cfg = RiserConfig(
        outer_diameter=0.3, wall_thickness=0.008, water_depth=1000, hang_off_angle_deg=20,
        coating_thickness=0.15, coating_density=300.0,
    )
    with pytest.raises(ValueError):
        cfg.effective_submerged_weight()
