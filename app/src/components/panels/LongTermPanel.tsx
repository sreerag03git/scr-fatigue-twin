import type { LongTermContribution } from "../../api/types";
import { useStore } from "../../state/store";
import { pct, sci, years } from "../../lib/format";
import { EmptyState, Measured, Metric, Panel } from "../ui";

// Compact Hs × Tp driver heat map: cell shade = fatigue-damage contribution.
function DriverHeatMap({
  hs, tp, cells, width,
}: {
  hs: number[]; tp: number[]; cells: LongTermContribution[]; width: number;
}) {
  const m = { l: 34, r: 8, t: 8, b: 22 };
  const cw = Math.max(6, (width - m.l - m.r) / Math.max(tp.length, 1));
  const ch = 15;
  const height = m.t + m.b + ch * hs.length;
  const byKey = new Map<string, number>();
  let maxFrac = 0;
  for (const c of cells) {
    byKey.set(`${c.hs}|${c.tp}`, c.damage_fraction);
    if (c.damage_fraction > maxFrac) maxFrac = c.damage_fraction;
  }
  const shade = (f: number) => {
    const t = maxFrac > 0 ? f / maxFrac : 0;
    // off-white → teal → amber as contribution rises
    if (t < 0.5) {
      const u = t / 0.5;
      return `rgba(14,124,130,${(0.08 + 0.55 * u).toFixed(3)})`;
    }
    const u = (t - 0.5) / 0.5;
    return `rgb(${Math.round(14 + (176 - 14) * u)},${Math.round(124 + (125 - 124) * u)},${Math.round(130 + (26 - 130) * u)})`;
  };
  return (
    <svg width={width} height={height} role="img" aria-label="Hs–Tp fatigue driver heat map">
      {hs.map((h, i) => (
        <text
          key={"h" + h}
          className="tick-label"
          x={m.l - 5}
          y={m.t + ch * i + ch / 2 + 3}
          textAnchor="end"
        >
          {h.toFixed(1)}
        </text>
      ))}
      {tp.map((t, j) => (
        <text
          key={"t" + t}
          className="tick-label"
          x={m.l + cw * j + cw / 2}
          y={height - 8}
          textAnchor="middle"
        >
          {t.toFixed(0)}
        </text>
      ))}
      {hs.map((h, i) =>
        tp.map((t, j) => {
          const f = byKey.get(`${h}|${t}`) ?? 0;
          return (
            <rect
              key={`${h}-${t}`}
              x={m.l + cw * j}
              y={m.t + ch * i}
              width={cw - 1}
              height={ch - 1}
              fill={f > 0 ? shade(f) : "var(--bg-2)"}
            />
          );
        }),
      )}
      <text className="axis-label" x={m.l + (width - m.l - m.r) / 2} y={height} textAnchor="middle">
        Tp [s]
      </text>
      <text className="axis-label" x={10} y={m.t + (ch * hs.length) / 2} textAnchor="middle" transform={`rotate(-90 10 ${m.t + (ch * hs.length) / 2})`}>
        Hs [m]
      </text>
    </svg>
  );
}

export function LongTermPanel() {
  const { result } = useStore();
  const lt = result?.long_term;
  const drivers = [...(lt?.contributions ?? [])]
    .sort((a, b) => b.damage_fraction - a.damage_fraction)
    .slice(0, 5);

  return (
    <Panel index="LT" title="Long-term fatigue · DNV-RP-C203 §5 scatter">
      {!lt ? (
        <EmptyState text="Awaiting scatter summation" />
      ) : (
        <>
          <div className="metric-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
            <Metric label="Long-term life" value={years(lt.life_years)} unit="yr" tone="amber" size="lg" />
            <Metric label="Annual damage" value={sci(lt.annual_damage_rate)} unit="/yr" />
            <Metric label="Sea states" value={lt.n_cells} />
          </div>
          <div className="row wrap" style={{ gap: 16, marginTop: 12, alignItems: "flex-start" }}>
            <div style={{ flex: "1 1 240px", minWidth: 0 }}>
              <div className="eyebrow" style={{ marginBottom: 4 }}>Damage contribution by cell</div>
              <Measured height={38 + 15 * lt.hs_values.length}>
                {(w) => (
                  <DriverHeatMap hs={lt.hs_values} tp={lt.tp_values} cells={lt.contributions} width={w} />
                )}
              </Measured>
            </div>
            <div style={{ flex: "1 1 220px", minWidth: 0 }}>
              <div className="eyebrow" style={{ marginBottom: 6 }}>Top fatigue drivers</div>
              <table className="gates">
                <thead>
                  <tr>
                    <td className="gates__name">Hs / Tp</td>
                    <td className="gates__actual">P(cell)</td>
                    <td className="gates__actual">Damage share</td>
                  </tr>
                </thead>
                <tbody>
                  {drivers.map((d) => (
                    <tr key={`${d.hs}-${d.tp}`}>
                      <td className="gates__name mono">{d.hs.toFixed(1)} m / {d.tp.toFixed(0)} s</td>
                      <td className="gates__actual mono">{pct(d.probability, 0)}</td>
                      <td className="gates__actual mono">{pct(d.damage_fraction, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="tiny muted" style={{ marginTop: 8 }}>
            Source: {lt.source}. Probability-weighted Dirlik damage summed over the scatter diagram
            (omnidirectional; a directional scatter would resolve heading-dependent fatigue).
          </p>
        </>
      )}
    </Panel>
  );
}
