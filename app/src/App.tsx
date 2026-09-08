import { useEffect } from "react";
import "./styles/layout.css";
import { useStore } from "./state/store";
import { LoadingScreen } from "./components/LoadingScreen";
import { ConfigPanel } from "./components/panels/ConfigPanel";
import { CircumferentialPanel } from "./components/panels/CircumferentialPanel";
import { DamagePanel } from "./components/panels/DamagePanel";
import { DecisionPanel } from "./components/panels/DecisionPanel";
import { DiagramsPanel } from "./components/panels/DiagramsPanel";
import { FracturePanel } from "./components/panels/FracturePanel";
import { LongTermPanel } from "./components/panels/LongTermPanel";
import { PosteriorPanel } from "./components/panels/PosteriorPanel";
import { ReliabilityPanel } from "./components/panels/ReliabilityPanel";
import { RunHistoryPanel } from "./components/panels/RunHistoryPanel";
import { SeaStatePanel } from "./components/panels/SeaStatePanel";
import { SeabedPanel } from "./components/panels/SeabedPanel";
import { SourcePanel } from "./components/panels/SourcePanel";
import { StatusBar } from "./components/panels/StatusBar";
import { TopBar } from "./components/panels/TopBar";
import { TracePanel } from "./components/panels/TracePanel";
import { TransferPanel } from "./components/panels/TransferPanel";
import { ValidationPanel } from "./components/panels/ValidationPanel";
import { VivPanel } from "./components/panels/VivPanel";

export default function App() {
  const { boot, bootError } = useStore();
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
          <TransferPanel />
          <DamagePanel />
          <LongTermPanel />
          <div className="stage__row2">
            <VivPanel />
            <SeabedPanel />
          </div>
          <CircumferentialPanel />
          <DiagramsPanel />
          <PosteriorPanel />
          <div className="stage__row2">
            <FracturePanel />
            <ReliabilityPanel />
          </div>
          <div className="stage__row2">
            <DecisionPanel />
            <ValidationPanel />
          </div>
        </div>
      </main>
      <StatusBar />
      <LoadingScreen />
    </div>
  );
}
