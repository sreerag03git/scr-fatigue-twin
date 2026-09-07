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

export interface TransferPayload {
  freq: number[];
  stress_mag: number[];
  moment_mag: number[];
  phase: number[];
  route: string;
  is_validated: boolean;
  provenance: { is_validated?: boolean; notes?: string; source_tool?: string; [k: string]: unknown };
}

export interface CatenaryPayload {
  x: number[];
  y: number[];
  catenary_parameter: number;
  horizontal_span: number;
  arc_length: number;
  water_depth: number;
  tdp_curvature: number;
  top_angle_deg: number;
}

export interface VerificationPayload {
  hist_edges_mpa: number[];
  hist_counts: number[];
  moments: Record<string, number>;
  stress_to_mpa: number;
  sigma_mpa: number;
}

export interface VivMode {
  mode: number;
  frequency_hz: number;
  reduced_velocity: number;
  excited: boolean;
  a_over_d: number;
  stress_range_mpa: number;
  annual_damage_rate: number;
}

export interface VivPayload {
  enabled: boolean;
  annual_damage_rate?: number;
  life_years?: number;
  dominant_mode?: number;
  stability_parameter?: number;
  current_surface_velocity?: number;
  is_screening?: boolean;
  modes?: VivMode[];
  span_length?: number;
  dominant_shape?: { arc: number[]; disp: number[] };
  current_profile?: { height: number[]; speed: number[] };
  marine_growth?: {
    enabled: boolean;
    thickness_mm: number;
    density: number;
    effective_diameter_mm: number;
    base_diameter_mm: number;
    mass_per_length: number;
    submerged_weight_per_length: number;
  };
}

export interface CombinedPayload {
  wave_rate: number;
  viv_rate: number;
  annual_rate: number;
  life_years: number;
}

export interface CrackPayload {
  enabled: boolean;
  material?: string;
  equivalent_stress_range_mpa?: number;
  cycles_per_year?: number;
  initial_flaw_mm?: number;
  critical_depth_mm?: number;
  delta_k0?: number;
  delta_k_threshold?: number;
  propagates?: boolean;
  crack_life_years?: number;
  fraction_propagating?: number;
  crack_inspection_year?: number | null;
  a_of_t?: { years: number[]; depth_mm: number[] };
  pod?: { size_mm: number[]; prob: number[] };
}

export interface ReliabilityPayload {
  enabled: boolean;
  beta?: number;
  pf_cumulative?: number;
  pf_annual?: number;
  design_life_years?: number;
  safety_class?: string;
  target_pf?: number;
  target_beta?: number;
  passes?: boolean;
  mean_curve_life_years?: number;
  importance?: Record<string, number>;
}

export interface SeabedPayload {
  enabled: boolean;
  lambda_b?: number;
  base_life_years?: number;
  life_soft?: number;
  life_stiff?: number;
  k_v_kpa?: number[];
  correction?: number[];
  life_years?: number[];
}

export interface LongTermContribution {
  hs: number;
  tp: number;
  probability: number;
  annual_rate: number;
  damage_fraction: number;
}

export interface LongTermPayload {
  source: string;
  annual_damage_rate: number;
  life_years: number;
  n_cells: number;
  hs_values: number[];
  tp_values: number[];
  contributions: LongTermContribution[];
}

export interface AnalyzeResponse {
  sea_state: { hs: number; tp: number; tz: number; gamma: number };
  motion?: { source: string; is_validated: boolean; [k: string]: unknown };
  spectrum: { freq: number[]; motion_psd: number[]; stress_psd: number[] };
  transfer?: TransferPayload;
  catenary?: CatenaryPayload;
  verification?: VerificationPayload;
  long_term?: LongTermPayload;
  crack?: CrackPayload;
  reliability?: ReliabilityPayload;
  seabed?: SeabedPayload;
  viv?: VivPayload;
  combined?: CombinedPayload;
  dof_contributions?: Record<string, number>;
  damage: {
    annual_rate_time: number;
    annual_rate_spectral: number;
    deterministic_life_years: number;
    block_damage: number;
    block_seconds: number;
    sn_environment?: string;
    acceptance?: {
      utilisation?: number;
      passes?: boolean;
      design_fatigue_factor?: number;
      required_life_years?: number;
      [k: string]: unknown;
    };
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
  economics: ConditionalEconomics;
  provenance: Provenance;
  data_health: DataHealth | null;
  trace: { time: number[]; heave: number[] };
  run_id: number | null;
  source?: { kind: string; [k: string]: unknown };
  diagrams?: {
    cutaway: string;
    general_arrangement: string;
    configurations: string;
    platforms: string;
    flexible: string;
    architecture: string;
  };
}

export interface ConditionalEconomics {
  discount_rate: number;
  horizon_yr: number;
  n_units: number;
  breakeven_phi: number;
  phi: number;
  phi_is_endogenous: boolean;
  per_unit_delta_c_usd: number;
  fleet_delta_c_usd: number;
  net_positive: boolean;
  pv_baseline_usd: number;
  pv_sensor_usd: number;
  pv_cbm_slow_usd: number;
  pv_cbm_fast_usd: number;
  phi_grid: number[];
  fleet_delta_c_p50_usd: number[];
  fleet_delta_c_p5_usd: number[];
  fleet_delta_c_p95_usd: number[];
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

export interface RunSummary {
  id: number;
  created_at: string;
  source: string;
  is_synthetic: number;
  config_sha: string;
  det_life: number | null;
  life_p10: number | null;
  life_p50: number | null;
  life_p90: number | null;
  next_insp: number | null;
}

export interface RunListResponse {
  runs: RunSummary[];
  count: number;
}

export interface StoredRun {
  id: number;
  created_at: string;
  source: string;
  is_synthetic: number;
  config: AnalysisConfig;
  payload: AnalyzeResponse;
}
