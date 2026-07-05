import { useStore } from "../../state/store";
import { sci, years } from "../../lib/format";
import { EmptyState, Metric, Panel } from "../ui";

export function DamagePanel() {
  const { result } = useStore();
  return (
    <Panel index="L2" title="Rainflow · S-N · Miner">
      {!result ? (
        <EmptyState text="Awaiting stress reconstruction" />
      ) : (
        <div className="metric-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
          <Metric label="Deterministic life" value={years(result.damage.deterministic_life_years)} unit="yr" tone="amber" size="lg" />
          <Metric label="Annual damage (time)" value={sci(result.damage.annual_rate_time)} unit="/yr" />
          <Metric label="Spectral (Dirlik)" value={sci(result.damage.annual_rate_spectral)} unit="/yr" />
          <Metric
            label="Env. capacity factor"
            value={result.environment.enabled ? result.environment.factor.toFixed(3) : "1.000"}
            tone={result.environment.enabled ? "alarm" : undefined}
          />
        </div>
      )}
    </Panel>
  );
}
