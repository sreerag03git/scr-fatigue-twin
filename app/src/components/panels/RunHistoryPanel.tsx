import { api } from "../../api/client";
import { useStore } from "../../state/store";
import { Badge, EmptyState, Panel } from "../ui";

function fmtLife(v: number | null): string {
  if (v == null || !isFinite(v)) return "—";
  return v >= 999 ? "999+" : v.toFixed(0);
}

function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (isNaN(+d)) return iso;
  return d.toLocaleString(undefined, {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

export function RunHistoryPanel() {
  const { runs, activeRunId, loadRun } = useStore();
  return (
    <Panel index="LOG" title="Run history" right={<Badge>{runs.length}</Badge>}>
      {runs.length === 0 ? (
        <EmptyState text="No runs yet — run an analysis" />
      ) : (
        <ul className="runlist">
          {runs.map((r) => (
            <li key={r.id} className={"runrow" + (r.id === activeRunId ? " runrow--active" : "")}>
              <button
                className="runrow__main"
                onClick={() => loadRun(r.id)}
                title={`Re-open run #${r.id} (config ${r.config_sha.slice(0, 10)})`}
              >
                <span className="runrow__id mono">#{r.id}</span>
                <span className={"runrow__tag " + (r.is_synthetic ? "runrow__tag--syn" : "runrow__tag--real")}>
                  {r.is_synthetic ? "SYN" : "MEAS"}
                </span>
                <span className="runrow__life mono">
                  {fmtLife(r.life_p50)}
                  <em>yr</em>
                </span>
                <span className="runrow__time">{fmtTime(r.created_at)}</span>
              </button>
              <button
                className="runrow__exp"
                onClick={() => api.exportRun(r.id)}
                title="Download provenance bundle"
                aria-label={`Export run ${r.id}`}
              >
                ↓
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
