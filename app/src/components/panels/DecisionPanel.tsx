import { useStore } from "../../state/store";
import { pct, years } from "../../lib/format";
import { LineChart } from "../charts/LineChart";
import { EmptyState, Measured, Metric, Panel } from "../ui";

export function DecisionPanel() {
  const { result } = useStore();
  return (
    <Panel index="DEC" title="Risk-based inspection · economics">
      {!result ? (
        <EmptyState text="Awaiting posterior" />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginBottom: 8 }}>
            <Metric
              label="Next inspection"
              value={result.inspection.limited_by_horizon ? ">horizon" : years(result.inspection.next_inspection_year)}
              unit={result.inspection.limited_by_horizon ? "" : "yr"}
              tone="signal"
            />
            <Metric label="Target PoF" value={pct(result.inspection.target_pof, 1)} />
            <Metric label="PoF at inspection" value={pct(result.inspection.pof_at_next, 2)} />
          </div>
          <Measured height={140}>
            {(w) => (
              <LineChart width={w} height={140} x={result.inspection.pof_years} y={result.inspection.pof_vals}
                color="var(--amber)" xLabel="year" yLabel="P(fail)" yFormat={(v) => v.toFixed(2)} fill
                hmarker={{ value: result.inspection.target_pof, label: "target", color: "var(--alarm)" }}
                vmarker={result.inspection.limited_by_horizon ? undefined : { value: result.inspection.next_inspection_year, label: "inspect", color: "var(--signal-2)" }}
              />
            )}
          </Measured>

          <hr className="rule" />
          <div className="eyebrow" style={{ marginBottom: 6 }}>
            Conditional economics (Eq. 11) · {result.economics.n_units} units · {result.economics.horizon_yr.toFixed(0)} yr
          </div>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginBottom: 8 }}>
            <Metric
              label="Net value ΔC (fleet)"
              value={usdM(result.economics.fleet_delta_c_usd)}
              tone={result.economics.net_positive ? "signal" : "alarm"}
            />
            <Metric label="Fleet φ" value={result.economics.phi.toFixed(2)} unit={result.economics.phi_is_endogenous ? "post." : "ref"} />
            <Metric label="Break-even φ*" value={result.economics.breakeven_phi.toFixed(2)} />
          </div>
          <Measured height={140}>
            {(w) => (
              <LineChart width={w} height={140} x={result.economics.phi_grid}
                y={result.economics.fleet_delta_c_p50_usd.map((v) => v / 1e6)}
                color="var(--signal-2)" xLabel="φ = P(ages slower than design)" yLabel="ΔC [US$M]"
                yFormat={(v) => v.toFixed(0)} fill
                hmarker={{ value: 0, label: "break-even", color: "var(--alarm)" }}
                vmarker={{ value: result.economics.breakeven_phi, label: `φ*=${result.economics.breakeven_phi.toFixed(2)}`, color: "var(--alarm)" }}
              />
            )}
          </Measured>
          <div className="eyebrow" style={{ marginTop: 6, opacity: 0.7 }}>
            φ estimated from the remaining-life posterior; the sensor pays only for φ &gt; φ*. Discounting (r=
            {(result.economics.discount_rate * 100).toFixed(0)}%) replaces the retracted flat headline saving.
          </div>
        </>
      )}
    </Panel>
  );
}

function usdM(v: number): string {
  return `${v >= 0 ? "+" : "−"}$${Math.abs(v / 1e6).toFixed(1)}M`;
}
