import { useStore } from "../../state/store";
import { fixed, pct, years } from "../../lib/format";
import { LineChart } from "../charts/LineChart";
import { Badge, EmptyState, Measured, Metric, Panel } from "../ui";

export function FracturePanel() {
  const { result } = useStore();
  const crack = result?.crack;
  const enabled = !!crack?.enabled;

  return (
    <Panel
      index="ECA"
      title="Fracture mechanics · BS 7910 crack growth"
      right={enabled && <Badge kind="badge--synthetic">SCREENING ECA</Badge>}
    >
      {!result ? (
        <EmptyState text="Awaiting analysis" />
      ) : !enabled ? (
        <EmptyState text="No propagating cycles — crack pathway inactive" />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            <Metric
              label="Crack-based life"
              value={years(crack!.crack_life_years ?? Infinity)}
              unit="yr"
              tone="amber"
              size="lg"
            />
            <Metric label="Equivalent Δσ" value={fixed(crack!.equivalent_stress_range_mpa ?? 0, 1)} unit="MPa" />
            <Metric label="a₀ → a_c" value={`${fixed(crack!.initial_flaw_mm ?? 0, 1)}→${fixed(crack!.critical_depth_mm ?? 0, 0)}`} unit="mm" />
            <Metric label="Fraction propagating" value={pct(crack!.fraction_propagating ?? 0, 0)} />
          </div>

          <div className="row wrap" style={{ gap: 16, marginTop: 12 }}>
            <div style={{ flex: "1 1 260px", minWidth: 0 }}>
              <div className="eyebrow" style={{ marginBottom: 4 }}>Crack depth a(t)</div>
              {crack!.a_of_t && (
                <Measured height={150}>
                  {(w) => (
                    <LineChart
                      x={crack!.a_of_t!.years}
                      y={crack!.a_of_t!.depth_mm}
                      width={w}
                      height={150}
                      color="var(--alarm)"
                      xLabel="Years"
                      yLabel="depth [mm]"
                      yFormat={(v) => v.toFixed(0)}
                      hmarker={{
                        value: crack!.critical_depth_mm ?? 0,
                        label: "wall breach",
                        color: "var(--alarm)",
                      }}
                      fill
                    />
                  )}
                </Measured>
              )}
            </div>
            <div style={{ flex: "1 1 260px", minWidth: 0 }}>
              <div className="eyebrow" style={{ marginBottom: 4 }}>Probability of detection</div>
              {crack!.pod && (
                <Measured height={150}>
                  {(w) => (
                    <LineChart
                      x={crack!.pod!.size_mm}
                      y={crack!.pod!.prob}
                      width={w}
                      height={150}
                      color="var(--signal)"
                      xLabel="Flaw size [mm]"
                      yLabel="POD"
                      yFormat={(v) => v.toFixed(1)}
                    />
                  )}
                </Measured>
              )}
            </div>
          </div>

          <p className="tiny muted" style={{ marginTop: 8 }}>
            {crack!.material}. {crack!.crack_inspection_year != null
              ? `Crack-based first inspection ≈ year ${fixed(crack!.crack_inspection_year, 1)}.`
              : "No crack-based inspection trigger within horizon."}{" "}
            Reduced-order flat-plate ECA — a design ECA needs the BS 7910 2-D (a/c) integration.
          </p>
        </>
      )}
    </Panel>
  );
}
