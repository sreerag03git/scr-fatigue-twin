import { useEffect } from "react";
import "./styles/layout.css";
import { useStore } from "./state/store";
import { ConfigPanel } from "./components/panels/ConfigPanel";
import { DamagePanel } from "./components/panels/DamagePanel";
import { DecisionPanel } from "./components/panels/DecisionPanel";
import { PosteriorPanel } from "./components/panels/PosteriorPanel";
import { RunHistoryPanel } from "./components/panels/RunHistoryPanel";
import { SeaStatePanel } from "./components/panels/SeaStatePanel";
import { SourcePanel } from "./components/panels/SourcePanel";
import { StatusBar } from "./components/panels/StatusBar";
import { TopBar } from "./components/panels/TopBar";
import { TracePanel } from "./components/panels/TracePanel";
import { ValidationPanel } from "./components/panels/ValidationPanel";

export default function App() {
  const { boot, booted, bootError } = useStore();
  useEffect(() => {
    boot();
  }, [boot]);

  if (bootError) {
    return (
      <div style={{ display: "grid", placeItems: "center", height: "100vh", padding: 24 }}>
        <div className="panel" style={{ maxWidth: 460, padding: 20 }}>
          <div className="eyebrow" style={{ color: "var(--alarm)" }}>Backend unavailable</div>
          <p style={{ color: "var(--text-mid)", fontSize: 13 }}>{bootError}</p>
          <p className="tiny muted">
            Start the physics backend, then reload:
            <code className="mono" style={{ display: "block", marginTop: 6, color: "var(--signal-2)" }}>
              uvicorn server.main:app --port 8000
            </code>
          </p>
          <button className="btn btn--primary" onClick={() => location.reload()}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <TopBar />
      <main className="console">
        <aside className="rail">
          <SourcePanel />
          <ConfigPanel />
          <RunHistoryPanel />
        </aside>
        <div className="stage">
          <div className="stage__row2">
            <SeaStatePanel />
            <TracePanel />
          </div>
          <DamagePanel />
          <PosteriorPanel />
          <div className="stage__row2">
            <DecisionPanel />
            <ValidationPanel />
          </div>
        </div>
      </main>
      <StatusBar />
      {!booted && (
        <div className="boot-veil">
          <div className="boot-veil__spinner" /> Initialising physics core…
        </div>
      )}
    </div>
  );
}
