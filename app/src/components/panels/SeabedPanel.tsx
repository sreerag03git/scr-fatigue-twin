import { useStore } from "../../state/store";
import { fixed, years } from "../../lib/format";
import { LineChart } from "../charts/LineChart";
import { EmptyState, Measured, Metric, Panel } from "../ui";

export function SeabedPanel() {
  const { result } = useStore();
  const sb = result?.seabed;
  const enabled = !!sb?.enabled;

  return (
    <Panel index="TDP" title="Seabed-stiffness sensitivity">
      {!result ? (
        <EmptyState text="Awaiting analysis" />
      ) : !enabled ? (
        <EmptyState text="Seabed sensitivity unavailable" />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
            <Metric label="Soft seabed life" value={years(sb!.life_soft ?? Infinity)} unit="yr" tone="amber" />
            <Metric label="Rigid-base life" value={years(sb!.base_life_years ?? Infinity)} unit="yr" />
            <Metric label="Stiff seabed life" value={years(sb!.life_stiff ?? Infinity)} unit="yr" />
          </div>
          {sb!.k_v_kpa && sb!.life_years && (
            <Measured height={150}>
              {(w) => (
                <LineChart
                  x={sb!.k_v_kpa!}
                  y={sb!.life_years!}
                  width={w}
                  height={150}
                  color="var(--signal)"
                  xLabel="Vertical seabed stiffness k_v [kPa]"
                  yLabel="life [yr]"
                  yFormat={(v) => v.toFixed(0)}
                />
              )}
            </Measured>
          )}
          <p className="tiny muted" style={{ marginTop: 6 }}>
            Boundary-layer length λ_b = {fixed(sb!.lambda_b ?? 0, 1)} m. TDP fatigue about the
            (conservative) rigid-base assumption; a softer seabed lengthens the stress boundary layer
            and generally improves life.
          </p>
        </>
      )}
    </Panel>
  );
}
