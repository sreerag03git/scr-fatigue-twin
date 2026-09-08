import { useStore } from "../../state/store";
import { fixed, sci, years } from "../../lib/format";
import { Badge, EmptyState, Metric, Panel } from "../ui";

export function ReliabilityPanel() {
  const { result } = useStore();
  const rel = result?.reliability;
  const enabled = !!rel?.enabled;
  const importance = rel?.importance ?? {};
  const drivers = Object.entries(importance).sort((a, b) => b[1] - a[1]);
  const maxShare = drivers.length ? drivers[0][1] : 1;

  return (
    <Panel
      index="REL"
      title="Structural reliability · FORM (DNV-RP-C210)"
      right={
        enabled && (
          <Badge kind={rel!.passes ? "badge--pass" : "badge--fail"}>
            {rel!.passes ? "MEETS TARGET" : "BELOW TARGET"}
          </Badge>
        )
      }
    >
      {!result ? (
        <EmptyState text="Awaiting analysis" />
      ) : !enabled ? (
        <EmptyState text="Reliability unavailable (insufficient life samples)" />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            <Metric
              label="Reliability index β (annual)"
              value={fixed(rel!.beta_annual ?? rel!.beta ?? 0, 2)}
              tone={rel!.passes ? "signal" : "alarm"}
              size="lg"
            />
            <Metric label="Annual Pf" value={sci(rel!.pf_annual ?? 0)} tone={rel!.passes ? undefined : "alarm"} />
            <Metric label={`Target (${rel!.safety_class})`} value={sci(rel!.target_pf ?? 0)} />
            <Metric label="Target β" value={fixed(rel!.target_beta ?? 0, 2)} />
          </div>
          <p className="tiny muted" style={{ marginTop: 8 }}>
            β and target are on the same annual basis (β_annual = Φ⁻¹(1−Pf_annual)); cumulative β at
            design life {fixed(rel!.beta ?? 0, 2)}. Mean-basis median life{" "}
            {years(rel!.mean_curve_life_years ?? Infinity)} yr at design life{" "}
            {years(rel!.design_life_years ?? 0)} yr. Acceptance is DNV annual Pf for safety class “{rel!.safety_class}”.
          </p>

          {drivers.length > 0 && (
            <>
              <hr className="rule" />
              <div className="eyebrow" style={{ marginBottom: 6 }}>FORM importance (variance share of ln life)</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                {drivers.map(([name, share]) => (
                  <div key={name} className="row" style={{ gap: 8 }}>
                    <span className="tiny" style={{ width: 120, color: "var(--text-mid)" }}>{name}</span>
                    <div style={{ flex: 1, height: 8, background: "var(--bg-2)", borderRadius: 3, overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${(share / (maxShare || 1)) * 100}%`,
                          height: "100%",
                          background: "var(--signal)",
                        }}
                      />
                    </div>
                    <span className="tiny mono" style={{ width: 44, textAlign: "right", color: "var(--text-mid)" }}>
                      {(share * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </Panel>
  );
}
