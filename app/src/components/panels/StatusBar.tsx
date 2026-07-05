import { useStore } from "../../state/store";

export function StatusBar() {
  const { result, status, error, config } = useStore();
  const p = result?.provenance;
  const dot = status === "error" ? "var(--alarm)" : status === "loading" ? "var(--amber)" : "var(--good)";
  const label = status === "error" ? "Error" : status === "loading" ? "Computing" : status === "ready" ? "Nominal" : "Idle";

  return (
    <footer className="statusbar mono">
      <span className="statusbar__item">
        <span className="statusbar__dot" style={{ background: dot, boxShadow: `0 0 6px ${dot}` }} />
        {label}
      </span>
      {error && <span className="statusbar__item" style={{ color: "var(--alarm)" }}>· {error}</span>}
      <span className="statusbar__spacer" />
      {p && (
        <>
          <span className="statusbar__item">fs {p.sample_rate_hz.toFixed(1)} Hz</span>
          <span className="statusbar__item">n {p.n_samples.toLocaleString()}</span>
          <span className="statusbar__item">seed {config?.seed ?? p.seed}</span>
          <span className="statusbar__item">H(f) {p.transfer_is_reduced_order ? "reduced-order" : "reference"}</span>
          <span className="statusbar__item" title="config hash">cfg {p.config_sha256.slice(0, 10)}</span>
          <span className="statusbar__item">core v{p.core_version}</span>
          <span className="statusbar__item">np {p.numpy_version} · sp {p.scipy_version}</span>
        </>
      )}
    </footer>
  );
}
