import { useStore } from "../../state/store";
import { fixed } from "../../lib/format";
import { LineChart } from "../charts/LineChart";
import { Badge, EmptyState, Measured, Metric, Panel } from "../ui";

const ROUTE_LABEL: Record<string, string> = {
  reference: "Reference (Route 2)",
  analytic: "Analytic (Route 1)",
  imported: "Imported vendor H(f)",
};

export function TransferPanel() {
  const { result } = useStore();
  const tf = result?.transfer;
  const cat = result?.catenary;
  return (
    <Panel
      index="L1"
      title="Transfer function H(f) · catenary geometry"
      right={
        tf && (
          <Badge kind={tf.is_validated ? "badge--real" : "badge--synthetic"}>
            {tf.is_validated ? "VALIDATED" : "ILLUSTRATIVE"}
          </Badge>
        )
      }
    >
      {!tf || !cat ? (
        <EmptyState text="Awaiting Layer-1 solve" />
      ) : (
        <>
          <div className="row wrap" style={{ marginBottom: 8, gap: 6 }}>
            <span className="badge">{ROUTE_LABEL[tf.route] ?? tf.route}</span>
            <span className="tiny muted">{String(tf.provenance?.notes ?? "")}</span>
          </div>
          <Measured height={168}>
            {(w) => (
              <LineChart
                x={tf.freq}
                y={tf.stress_mag}
                width={w}
                height={168}
                color="var(--amber)"
                xLabel="Frequency [Hz]"
                yLabel="MPa per m heave"
                yFormat={(v) => v.toFixed(v < 10 ? 1 : 0)}
                fill
              />
            )}
          </Measured>
          <div
            className="metric-grid"
            style={{ gridTemplateColumns: "repeat(3, 1fr)", marginTop: 10 }}
          >
            <Metric label="Catenary a" value={fixed(cat.catenary_parameter, 0)} unit="m" />
            <Metric label="Horizontal span" value={fixed(cat.horizontal_span, 0)} unit="m" />
            <Metric label="Suspended length" value={fixed(cat.arc_length, 0)} unit="m" />
            <Metric label="Top angle (from horiz.)" value={fixed(cat.top_angle_deg, 1)} unit="°" />
            <Metric label="TDP curvature" value={cat.tdp_curvature.toExponential(2)} unit="1/m" />
            <Metric label="Water depth" value={fixed(cat.water_depth, 0)} unit="m" />
          </div>
        </>
      )}
    </Panel>
  );
}
