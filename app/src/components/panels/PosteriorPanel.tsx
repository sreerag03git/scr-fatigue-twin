import { useStore } from "../../state/store";
import { years } from "../../lib/format";
import { FanChart } from "../charts/FanChart";
import { Histogram } from "../charts/Histogram";
import { Badge, EmptyState, Measured, Metric, Panel } from "../ui";

export function PosteriorPanel() {
  const { result } = useStore();
  return (
    <Panel
      index="L3"
      title="Remaining-life posterior"
      right={result && <Badge kind="badge--real">{result.posterior.n_members.toLocaleString()} MC members</Badge>}
    >
      {!result ? (
        <EmptyState text="Run the Monte Carlo to see the posterior" />
      ) : (
        <div className="posterior">
          <div className="posterior__fan">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 2 }}>
              <span className="eyebrow">90% credible band vs monitoring time — Bayesian contraction (1/√T)</span>
            </div>
            <Measured height={300}>
              {(w) => (
                <FanChart width={w} years={result.bayesian_fan.years} low={result.bayesian_fan.low}
                  median={result.bayesian_fan.median} high={result.bayesian_fan.high} />
              )}
            </Measured>
          </div>
          <div className="posterior__side">
            <div className="metric-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <Metric label="P10 (conservative)" value={years(result.posterior.p10)} unit="yr" tone="alarm" />
              <Metric label="P50 (median)" value={years(result.posterior.p50)} unit="yr" tone="signal" />
              <Metric label="P90" value={years(result.posterior.p90)} unit="yr" />
              <Metric label="P90/P10 spread" value={(result.posterior.p90 / Math.max(result.posterior.p10, 1e-9)).toFixed(1)} unit="×" />
            </div>
            <div style={{ marginTop: 10 }}>
              <span className="eyebrow">Life PDF</span>
              <Measured height={150}>
                {(w) => (
                  <Histogram width={w} counts={result.posterior.hist_counts} edges={result.posterior.hist_edges}
                    p10={result.posterior.p10} p50={result.posterior.p50} p90={result.posterior.p90} />
                )}
              </Measured>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
