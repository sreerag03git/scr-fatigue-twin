import { api } from "../../api/client";
import { useStore } from "../../state/store";
import { Badge } from "../ui";

export function TopBar() {
  const { result, status, run, validation, theme, toggleTheme } = useStore();
  const synthetic = result?.provenance.motion_is_synthetic ?? true;
  const gatesOk = validation ? validation.passed === validation.total : false;
  const runId = result?.run_id ?? null;

  return (
    <header className="topbar">
      <div className="topbar__brand">
        <div className="topbar__mark" aria-hidden>
          <svg width="22" height="22" viewBox="0 0 22 22">
            <path d="M3 19 C 7 19, 8 6, 19 3" fill="none" stroke="var(--signal)" strokeWidth="1.6" />
            <circle cx="3" cy="19" r="2" fill="var(--amber)" />
            <circle cx="19" cy="3" r="1.6" fill="var(--signal-2)" />
          </svg>
        </div>
        <div>
          <div className="topbar__title">SCR·TWIN</div>
          <div className="topbar__sub">TDP Fatigue Integrity Console</div>
        </div>
      </div>

      <nav className="topbar__flow" aria-label="Data flow">
        {["MRU", "H(f)", "Rainflow · S-N", "Posterior", "Decision"].map((s, i) => (
          <span key={s} className="flowstep">
            {i > 0 && <span className="flowstep__arw">›</span>}
            {s}
          </span>
        ))}
      </nav>

      <div className="topbar__actions">
        {validation && (
          <Badge kind={gatesOk ? "badge--pass badge--dot" : "badge--fail badge--dot"}>
            {validation.passed}/{validation.total} gates
          </Badge>
        )}
        <Badge kind={synthetic ? "badge--synthetic badge--dot" : "badge--real badge--dot"}>
          {synthetic ? "Synthetic" : "Measured"}
        </Badge>
        <button className="btn btn--ghost btn--sm" onClick={toggleTheme} title="Toggle report theme">
          {theme === "dark" ? "Report" : "Console"}
        </button>
        <button
          className="btn btn--ghost btn--sm"
          onClick={() => runId != null && api.exportRun(runId)}
          disabled={runId == null}
          title="Download reproducible provenance bundle (config, seed, versions)"
        >
          Export
        </button>
        <button className="btn btn--primary" onClick={run} disabled={status === "loading"}>
          {status === "loading" ? "Computing…" : "Run analysis"}
        </button>
      </div>
    </header>
  );
}
