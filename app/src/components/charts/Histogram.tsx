import { linScale, niceTicks } from "./util";

interface Props {
  counts: number[];
  edges: number[];
  p10: number;
  p50: number;
  p90: number;
  width: number;
  height?: number;
}

// Remaining-life posterior PDF with P10/P50/P90 markers.
export function Histogram({ counts, edges, p10, p50, p90, width, height = 150 }: Props) {
  const m = { l: 40, r: 12, t: 10, b: 24 };
  const iw = Math.max(10, width - m.l - m.r);
  const ih = height - m.t - m.b;
  if (width < 40 || counts.length < 1) return <svg width={width} height={height} />;

  const xMin = edges[0];
  const xMax = Math.min(edges[edges.length - 1], p90 * 1.8);
  const sx = linScale(xMin, xMax, m.l, m.l + iw);
  const yMax = Math.max(...counts) * 1.1;
  const sy = linScale(0, yMax, m.t + ih, m.t);
  const xt = niceTicks(xMin, xMax, 5);

  const marker = (v: number, color: string, label: string) => (
    <g>
      <line x1={sx(v)} x2={sx(v)} y1={m.t} y2={m.t + ih} stroke={color} strokeWidth={1.2} strokeDasharray="4 3" />
      <text x={sx(v)} y={m.t - 1} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill={color}>
        {label}
      </text>
    </g>
  );

  return (
    <svg width={width} height={height} role="img" aria-label="Remaining-life distribution">
      {xt.map((v) => (
        <text key={v} className="tick-label" x={sx(v)} y={height - 8} textAnchor="middle">{v.toFixed(0)}</text>
      ))}
      {counts.map((c, i) => {
        const x0 = sx(edges[i]);
        const x1 = sx(edges[i + 1]);
        if (edges[i] > xMax) return null;
        return (
          <rect key={i} x={x0 + 0.5} y={sy(c)} width={Math.max(0.5, x1 - x0 - 1)} height={m.t + ih - sy(c)}
            fill="var(--violet)" opacity={0.42} />
        );
      })}
      {marker(p10, "var(--amber)", "P10")}
      {marker(p50, "var(--signal-2)", "P50")}
      {marker(p90, "var(--text-mid)", "P90")}
      <text className="axis-label" x={m.l + iw / 2} y={height - 0} textAnchor="middle">Remaining life [yr]</text>
    </svg>
  );
}
