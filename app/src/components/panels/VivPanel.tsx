import { useStore } from "../../state/store";
import { fixed, sci, years } from "../../lib/format";
import { LineChart } from "../charts/LineChart";
import { Badge, EmptyState, Measured, Metric, Panel } from "../ui";

export function VivPanel() {
  const { result } = useStore();
  const viv = result?.viv;
  const combined = result?.combined;
  const enabled = !!viv?.enabled;

  const topModes = (viv?.modes ?? [])
    .filter((m) => m.excited)
    .sort((a, b) => b.annual_damage_rate - a.annual_damage_rate)
    .slice(0, 6);

  return (
    <Panel
      index="VIV"
      title="Cross-flow VIV screening · DNV-RP-F204"
      right={enabled && <Badge kind="badge--synthetic">SCREENING</Badge>}
    >
      {!result ? (
        <EmptyState text="Awaiting analysis" />
      ) : !enabled ? (
        <EmptyState text="No current set — VIV inactive (set surface current > 0)" />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            <Metric label="VIV screening life" value={years(viv!.life_years ?? Infinity)} unit="yr" tone="amber" size="lg" />
            <Metric label="Dominant mode" value={viv!.dominant_mode ?? 0} />
            <Metric label="Stability param Ks" value={fixed(viv!.stability_parameter ?? 0, 2)} />
            <Metric label="Surface current" value={fixed(viv!.current_surface_velocity ?? 0, 2)} unit="m/s" />
          </div>

          {combined && (
            <>
              <hr className="rule" />
              <div className="eyebrow" style={{ marginBottom: 6 }}>Combined wave + VIV (Miner sum)</div>
              <div className="metric-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                <Metric label="Wave rate" value={sci(combined.wave_rate)} unit="/yr" />
                <Metric label="VIV rate" value={sci(combined.viv_rate)} unit="/yr" tone="amber" />
                <Metric label="Combined life" value={years(combined.life_years)} unit="yr" tone="alarm" />
              </div>
            </>
          )}

          {viv!.dominant_shape && viv!.dominant_shape.arc.length > 1 && (
            <>
              <hr className="rule" />
              <div className="eyebrow" style={{ marginBottom: 4 }}>
                Dominant cross-flow mode shape · span {fixed(viv!.span_length ?? 0, 0)} m
              </div>
              <Measured height={140}>
                {(w) => (
                  <LineChart
                    x={viv!.dominant_shape!.arc}
                    y={viv!.dominant_shape!.disp}
                    width={w}
                    height={140}
                    color="var(--signal)"
                    xLabel="Arc length s [m]"
                    yLabel="φ(s)"
                    yFormat={(v) => v.toFixed(1)}
                    zeroBaseline
                  />
                )}
              </Measured>
            </>
          )}

          {topModes.length > 0 && (
            <>
              <hr className="rule" />
              <div className="eyebrow" style={{ marginBottom: 6 }}>Excited modes (top drivers)</div>
              <table className="gates">
                <thead>
                  <tr>
                    <td className="gates__name">Mode</td>
                    <td className="gates__actual">fn [Hz]</td>
                    <td className="gates__actual">Vr</td>
                    <td className="gates__actual">A/D</td>
                    <td className="gates__actual">Δσ [MPa]</td>
                    <td className="gates__actual">D/yr</td>
                  </tr>
                </thead>
                <tbody>
                  {topModes.map((m) => (
                    <tr key={m.mode}>
                      <td className="gates__name mono">{m.mode}</td>
                      <td className="gates__actual mono">{fixed(m.frequency_hz, 3)}</td>
                      <td className="gates__actual mono">{fixed(m.reduced_velocity, 1)}</td>
                      <td className="gates__actual mono">{fixed(m.a_over_d, 2)}</td>
                      <td className="gates__actual mono">{fixed(m.stress_range_mpa, 1)}</td>
                      <td className="gates__actual mono">{sci(m.annual_damage_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {viv!.marine_growth?.enabled && (
            <p className="tiny muted" style={{ marginTop: 8 }}>
              Marine growth {fixed(viv!.marine_growth.thickness_mm, 0)} mm → hydrodynamic Ø{" "}
              {fixed(viv!.marine_growth.effective_diameter_mm, 0)} mm (bare{" "}
              {fixed(viv!.marine_growth.base_diameter_mm, 0)} mm). Screening upper bound — design VIV
              needs Shear7 / VIVANA.
            </p>
          )}
        </>
      )}
    </Panel>
  );
}
