// Typed contracts mirroring the FastAPI backend (server/schemas.py + service.py).

export interface RiserConfig {
  outer_diameter: number;
  wall_thickness: number;
  youngs_modulus: number;
  material_grade: string;
  ultimate_strength: number;
  contents_density: number;
  coating_thickness: number;
  coating_density: number;
  submerged_weight_override: number | null;
  water_depth: number;
  hang_off_angle_deg: number;
  scf: number;
  sn_class: string;
  weld_thickness: number | null;
  mean_stress_model: string;
  is_reference_preset: boolean;
}

export interface TransferConfig {
  route: "reference" | "analytic" | "imported";
  natural_frequency: number;
  sigma_velocity: number;
  drag_coefficient: number;
  added_mass_coefficient: number;
  structural_damping_ratio: number;
}

export interface EnvironmentConfig {
  enabled: boolean;
  temperature_factor: number;
  salinity_factor: number;
}

export interface AnalysisConfig {
  riser: RiserConfig;
  transfer: TransferConfig;
  environment: EnvironmentConfig;
  block_duration_s: number;
  n_monte_carlo: number;
  seed: number;
}

export interface SyntheticParams {
  hs: number;
  tp: number;
  gamma: number;
  duration: number;
  fs: number;
  seed: number;
}

export interface Provenance {
  core_version: string;
  numpy_version: string;
  scipy_version: string;
  seed: number;
  config_sha256: string;
  n_samples: number;
  sample_rate_hz: number;
  transfer_is_reduced_order: boolean;
  motion_is_synthetic: boolean;
}

export interface DataHealth {
  n_raw: number;
  n_used: number;
  fs_hz: number;
  duration_s: number;
  n_gaps: number;
  max_gap_s: number;
  nan_count: number;
  clipped_fraction: number;
  non_monotonic: boolean;
  channels: string[];
  flags: string[];
  ok: boolean;
}

export interface AnalyzeResponse {
  sea_state: { hs: number; tp: number; tz: number; gamma: number };
  spectrum: { freq: number[]; motion_psd: number[]; stress_psd: number[] };
  damage: {
    annual_rate_time: number;
    annual_rate_spectral: number;
    deterministic_life_years: number;
    block_damage: number;
    block_seconds: number;
  };
  environment: { enabled: boolean; factor: number; temperature_factor: number; salinity_factor: number };
  posterior: {
    p10: number; p50: number; p90: number; n_members: number;
    hist_counts: number[]; hist_edges: number[]; cdf_x: number[]; cdf_p: number[];
  };
  bayesian_fan: { years: number[]; low: number[]; median: number[]; high: number[] };
  inspection: {
    next_inspection_year: number; target_pof: number; pof_at_next: number;
    limited_by_horizon: boolean; pof_years: number[]; pof_vals: number[];
  };
  economics: FleetEconomics;
  provenance: Provenance;
  data_health: DataHealth | null;
  trace: { time: number[]; heave: number[] };
}

export interface FleetEconomics {
  per_unit_saving_low_usd: number;
  per_unit_saving_high_usd: number;
  fleet_saving_low_usd: number;
  fleet_saving_high_usd: number;
  payback_low_yr: number;
  payback_high_yr: number;
  baseline_inspection_cost_usd: number;
  monitoring_cost_usd: number;
}

export interface Gate {
  name: string; category: string; passed: boolean; target: string; actual: string; detail: string;
}
export interface ValidationResponse { gates: Gate[]; passed: number; total: number }

export interface SNClass {
  name: string; m1: number; log_a1: number; m2: number; log_a2: number;
  thickness_exponent: number; fatigue_limit_mpa: number;
}

export interface IngestResponse {
  token: string;
  health: DataHealth;
  preview: { time: number[]; heave: number[] };
}
