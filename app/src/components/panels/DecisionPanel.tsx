import { useStore } from "../../state/store";
import { pct, usd, years } from "../../lib/format";
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
          <div className="eyebrow" style={{ marginBottom: 6 }}>Fleet business case · 20 units · 20 yr</div>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
            <Metric label="Net saving (range)" value={`${usd(result.economics.fleet_saving_low_usd)}–${usd(result.economics.fleet_saving_high_usd)}`} tone="signal" />
            <Metric label="Sensor payback" value={`${result.economics.payback_low_yr.toFixed(1)}–${result.economics.payback_high_yr.toFixed(1)}`} unit="yr" />
            <Metric label="Baseline inspection /unit" value={usd(result.economics.baseline_inspection_cost_usd)} />
            <Metric label="Monitoring cost /unit" value={usd(result.economics.monitoring_cost_usd)} />
          </div>
        </>
      )}
    </Panel>
  );
}
