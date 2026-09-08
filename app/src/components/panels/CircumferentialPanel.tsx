import { useStore } from "../../state/store";
import { fixed, years } from "../../lib/format";
import { Badge, EmptyState, Metric, Panel } from "../ui";

// clock angle phi (0 = crown, at top) → screen point, clockwise.
function pt(cx: number, cy: number, r: number, phiDeg: number): [number, number] {
  const a = (phiDeg * Math.PI) / 180;
  return [cx + r * Math.sin(a), cy - r * Math.cos(a)];
}

function radar(
  angles: number[], values: number[], max: number,
  cx: number, cy: number, r0: number, r1: number,
): string {
  let d = "";
  for (let i = 0; i < angles.length; i++) {
    const norm = max > 0 ? Math.max(0, values[i]) / max : 0;
    const [x, y] = pt(cx, cy, r0 + norm * (r1 - r0), angles[i]);
    d += (d ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
  }
  return d + "Z";
}

export function CircumferentialPanel() {
  const { result } = useStore();
  const c = result?.circumferential;
  if (!result) return <Panel index="CIR" title="Circumferential fatigue · TDP girth weld"><EmptyState text="Awaiting analysis" /></Panel>;
  if (!c?.enabled || !c.angles_deg)
    return <Panel index="CIR" title="Circumferential fatigue · TDP girth weld"><EmptyState text="No rainflow cycles to distribute" /></Panel>;

  const angles = c.angles_deg;
  const total = c.damage_rate ?? [];
  const wave = c.wave_rate ?? [];
  const viv = c.viv_rate ?? [];
  const max = Math.max(...total, 1e-30);

  const S = 260;
  const cx = S / 2;
  const cy = S / 2;
  const r1 = S / 2 - 30; // max spoke
  const r0 = 34; // pipe wall radius
  const worst = c.worst_angle_deg ?? 0;
  const [wx, wy] = pt(cx, cy, r1 + 8, worst);

  const clock: { phi: number; label: string }[] = [
    { phi: 0, label: "Crown 0°" },
    { phi: 90, label: "90°" },
    { phi: 180, label: "Keel 180°" },
    { phi: 270, label: "270°" },
  ];

  return (
    <Panel
      index="CIR"
      title="Circumferential fatigue · TDP girth weld"
      right={<Badge kind="badge--synthetic">SCREENING</Badge>}
    >
      <div className="metric-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <Metric label="Worst clock position" value={`${fixed(worst, 0)}°`} tone="amber" size="lg" />
        <Metric label="Worst-position life (screening)" value={years(c.worst_life_years ?? Infinity)} unit="yr" />
        <Metric label="Crown φ=0 life" value={years(c.crown_life_years ?? Infinity)} unit="yr" />
        <Metric label="Worst vs crown" value={`${fixed(c.worst_vs_crown_ratio ?? 1, 1)}×`} />
      </div>

      <div className="row wrap" style={{ gap: 16, marginTop: 12, alignItems: "flex-start" }}>
        <svg width={S} height={S} role="img" aria-label="Circumferential damage clock map" style={{ flex: "0 0 auto" }}>
          {/* reference rings */}
          {[0.5, 1].map((f) => (
            <circle key={f} cx={cx} cy={cy} r={r0 + f * (r1 - r0)} fill="none" stroke="var(--grid)" strokeWidth={1} />
          ))}
          {/* pipe wall */}
          <circle cx={cx} cy={cy} r={r0} fill="var(--bg-2)" stroke="var(--line-strong)" strokeWidth={1.5} />
          {/* in-plane (vertical) axis guide */}
          <line x1={cx} y1={cy - r1} x2={cx} y2={cy + r1} stroke="var(--line)" strokeWidth={1} strokeDasharray="3 3" />
          <line x1={cx - r1} y1={cy} x2={cx + r1} y2={cy} stroke="var(--line)" strokeWidth={1} strokeDasharray="3 3" />
          {/* damage lobes */}
          <path d={radar(angles, wave, max, cx, cy, r0, r1)} fill="var(--amber)" fillOpacity={0.16} stroke="var(--amber)" strokeWidth={1.3} />
          {viv.some((v) => v > 0) && (
            <path d={radar(angles, viv, max, cx, cy, r0, r1)} fill="var(--signal)" fillOpacity={0.14} stroke="var(--signal)" strokeWidth={1.3} />
          )}
          <path d={radar(angles, total, max, cx, cy, r0, r1)} fill="none" stroke="var(--text-hi)" strokeWidth={1.6} />
          {/* worst-position marker */}
          <line x1={cx} y1={cy} x2={pt(cx, cy, r1, worst)[0]} y2={pt(cx, cy, r1, worst)[1]} stroke="var(--alarm)" strokeWidth={1.4} />
          <circle cx={pt(cx, cy, r1, worst)[0]} cy={pt(cx, cy, r1, worst)[1]} r={3.5} fill="var(--alarm)" />
          <text x={wx} y={wy} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--alarm)">worst</text>
          {/* clock labels */}
          {clock.map(({ phi, label }) => {
            const [lx, ly] = pt(cx, cy, r1 + 16, phi);
            return (
              <text key={phi} x={lx} y={ly + 3} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-lo)">
                {label}
              </text>
            );
          })}
        </svg>

        <div style={{ flex: "1 1 200px", minWidth: 0 }}>
          <div className="eyebrow" style={{ marginBottom: 6 }}>Reading the clock map</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="row" style={{ gap: 7 }}>
              <span style={{ width: 12, height: 3, background: "var(--amber)", borderRadius: 2 }} />
              <span className="tiny muted">Wave bending — in-plane; hot spot rotates to the heading ({fixed(c.heading_deg ?? 0, 0)}°). Crown 0° / keel 180° are the fixed pipe positions.</span>
            </div>
            {c.viv_included && (
              <div className="row" style={{ gap: 7 }}>
                <span style={{ width: 12, height: 3, background: "var(--signal)", borderRadius: 2 }} />
                <span className="tiny muted">Cross-flow VIV — out-of-plane, peaks at the 90°/270° saddles</span>
              </div>
            )}
            <div className="row" style={{ gap: 7 }}>
              <span style={{ width: 12, height: 3, background: "var(--text-hi)", borderRadius: 2 }} />
              <span className="tiny muted">Total D(φ) envelope (Miner sum) — the red spoke is the worst clock position</span>
            </div>
          </div>
          <p className="tiny muted" style={{ marginTop: 10 }}>
            Localizes girth-weld fatigue and flags the worst clock position for inspection. Screening
            idealizations: a single in-phase wave plane and pure cross-flow VIV — it does not model
            in-line (streamwise) VIV near the crown/keel or oblique-heading whirl, so it is not a
            life-extension basis; the conservative combined wave+VIV life governs. Wave and VIV share
            one hot-spot S-N basis (SCF, thickness, mean stress).
          </p>
        </div>
      </div>
    </Panel>
  );
}
