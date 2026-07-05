import { useStore } from "../../state/store";
import { Field, Panel } from "../ui";

export function ConfigPanel() {
  const { config, snClasses, patchRiser, patchTransfer, patchEnv, patchConfig } = useStore();
  if (!config) return null;
  const { riser, transfer, environment } = config;

  return (
    <Panel index="CFG" title="Riser & analysis configuration">
      <div className="eyebrow" style={{ marginBottom: 6 }}>Steel catenary riser</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 10px" }}>
        <Field label="OD [m]">
          <input className="input" type="number" step={0.001} value={riser.outer_diameter}
            onChange={(e) => patchRiser({ outer_diameter: Number(e.target.value) })} />
        </Field>
        <Field label="Wall t [m]">
          <input className="input" type="number" step={0.001} value={riser.wall_thickness}
            onChange={(e) => patchRiser({ wall_thickness: Number(e.target.value) })} />
        </Field>
        <Field label="Water depth [m]">
          <input className="input" type="number" step={10} value={riser.water_depth}
            onChange={(e) => patchRiser({ water_depth: Number(e.target.value) })} />
        </Field>
        <Field label="Hang-off [° from vert]">
          <input className="input" type="number" step={1} value={riser.hang_off_angle_deg}
            onChange={(e) => patchRiser({ hang_off_angle_deg: Number(e.target.value) })} />
        </Field>
        <Field label="SCF">
          <input className="input" type="number" step={0.05} value={riser.scf}
            onChange={(e) => patchRiser({ scf: Number(e.target.value) })} />
        </Field>
        <Field label="S-N class">
          <select value={riser.sn_class} onChange={(e) => patchRiser({ sn_class: e.target.value })}>
            {snClasses.map((c) => (
              <option key={c.name} value={c.name}>{c.name} — Δσ*={c.fatigue_limit_mpa.toFixed(0)} MPa</option>
            ))}
          </select>
        </Field>
      </div>

      <hr className="rule" />
      <div className="eyebrow" style={{ marginBottom: 6 }}>Layer 1 — transfer function</div>
      <div className="seg" style={{ width: "100%" }}>
        <button style={{ flex: 1 }} aria-pressed={transfer.route === "reference"} onClick={() => patchTransfer({ route: "reference" })}>Reference H(f)</button>
        <button style={{ flex: 1 }} aria-pressed={transfer.route === "analytic"} onClick={() => patchTransfer({ route: "analytic" })}>Analytic (RO)</button>
      </div>
      <p className="tiny muted" style={{ marginTop: 6 }}>
        {transfer.route === "reference"
          ? "Illustrative reference table (Route 2). Import a project OrcaFlex/RIFLEX H(f) for rigor."
          : "Reduced-order catenary + linearised Morison (Route 1). Honest but under-predicts TDP stress."}
      </p>

      <hr className="rule" />
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
        <div className="eyebrow">Arabian Gulf correction</div>
        <div className="seg">
          <button aria-pressed={environment.enabled} onClick={() => patchEnv({ enabled: true })}>On</button>
          <button aria-pressed={!environment.enabled} onClick={() => patchEnv({ enabled: false })}>Off</button>
        </div>
      </div>
      {environment.enabled && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 10px" }}>
          <Field label="Temp factor">
            <input className="input" type="number" step={0.01} min={0.72} max={0.78} value={environment.temperature_factor}
              onChange={(e) => patchEnv({ temperature_factor: Number(e.target.value) })} />
          </Field>
          <Field label="Salinity factor">
            <input className="input" type="number" step={0.01} min={0.85} max={0.9} value={environment.salinity_factor}
              onChange={(e) => patchEnv({ salinity_factor: Number(e.target.value) })} />
          </Field>
        </div>
      )}

      <hr className="rule" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 10px" }}>
        <Field label="Monte Carlo N">
          <input className="input" type="number" step={1000} value={config.n_monte_carlo}
            onChange={(e) => patchConfig({ n_monte_carlo: Number(e.target.value) })} />
        </Field>
        <Field label="Seed">
          <input className="input" type="number" step={1} value={config.seed}
            onChange={(e) => patchConfig({ seed: Number(e.target.value) })} />
        </Field>
      </div>
    </Panel>
  );
}
