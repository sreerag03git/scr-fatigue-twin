import { useStore } from "../../state/store";
import { LineChart } from "../charts/LineChart";
import { EmptyState, Measured, Panel } from "../ui";

export function TracePanel() {
  const { result } = useStore();
  return (
    <Panel index="MRU" title="Hang-off motion trace">
      {!result ? (
        <EmptyState text="No motion loaded" />
      ) : (
        <Measured height={210}>
          {(w) => (
            <LineChart width={w} height={210} x={result.trace.time} y={result.trace.heave}
              color="var(--signal)" xLabel="t [s]" yLabel="heave [m]" fill zeroBaseline
              yFormat={(v) => v.toFixed(1)} />
          )}
        </Measured>
      )}
    </Panel>
  );
}
