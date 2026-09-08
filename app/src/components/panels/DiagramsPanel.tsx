import { useState } from "react";
import type { AnalyzeResponse } from "../../api/types";
import { useStore } from "../../state/store";
import { EmptyState, Panel } from "../ui";

type DKey = keyof NonNullable<AnalyzeResponse["diagrams"]>;

const TABS: { key: DKey; label: string }[] = [
  { key: "cutaway", label: "Realistic cutaway" },
  { key: "general_arrangement", label: "GA drawing" },
  { key: "configurations", label: "SCR vs lazy-wave" },
  { key: "platforms", label: "Host platforms" },
  { key: "flexible", label: "Flexible riser" },
  { key: "architecture", label: "Processing chain" },
];

export function DiagramsPanel() {
  const { result, status } = useStore();
  const [tab, setTab] = useState<DKey>("cutaway");
  const diagrams = result?.diagrams;
  return (
    <Panel index="ST" title="System diagrams · technical drawings">
      {!diagrams ? (
        <EmptyState text={status === "loading" ? "Rendering diagrams…" : "Run an analysis to draw the SCR system"} />
      ) : (
        <>
          <div className="diagram-tabs">
            {TABS.map((t) => (
              <button
                key={t.key}
                className={"diagram-tab" + (tab === t.key ? " is-active" : "")}
                onClick={() => setTab(t.key)}
                type="button"
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="diagram-view" dangerouslySetInnerHTML={{ __html: diagrams[tab] }} />
          <p className="tiny muted" style={{ marginTop: 8 }}>
            Drawn to the solved catenary geometry. Schematic / alternative configurations are badged
            illustrative — the twin analyses the steel catenary riser.
          </p>
        </>
      )}
    </Panel>
  );
}
