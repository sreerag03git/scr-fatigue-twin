import { useStore } from "../../state/store";
import { SpectrumChart } from "../charts/SpectrumChart";
import { EmptyState, Measured, Metric, Panel } from "../ui";

export function SeaStatePanel() {
  const { result, status } = useStore();
  return (
    <Panel index="L0" title="Sea state · spectral analysis">
      {!result ? (
        <EmptyState text={status === "loading" ? "Estimating spectrum…" : "Run an analysis"} />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 8 }}>
            <Metric label="Sig. heave Hm0" value={result.sea_state.hs.toFixed(2)} unit="m" />
            <Metric label="Tp" value={result.sea_state.tp.toFixed(1)} unit="s" />
            <Metric label="Tz" value={result.sea_state.tz.toFixed(1)} unit="s" />
            <Metric label="γ fit" value={result.sea_state.gamma.toFixed(1)} />
          </div>
          <Measured height={210}>
            {(w) => (
              <SpectrumChart width={w} freq={result.spectrum.freq}
                motionPsd={result.spectrum.motion_psd} stressPsd={result.spectrum.stress_psd} />
            )}
          </Measured>
        </>
      )}
    </Panel>
  );
}
