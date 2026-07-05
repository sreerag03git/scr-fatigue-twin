import { extent, linScale, niceTicks, path } from "./util";

interface Props {
  x: number[];
  y: number[];
  width: number;
  height?: number;
  color?: string;
  xLabel?: string;
  yLabel?: string;
  yFormat?: (v: number) => string;
  fill?: boolean;
  vmarker?: { value: number; label: string; color: string };
  hmarker?: { value: number; label: string; color: string };
  zeroBaseline?: boolean;
}

export function LineChart({
  x, y, width, height = 150, color = "var(--signal)", xLabel, yLabel,
  yFormat = (v) => v.toFixed(1), fill = false, vmarker, hmarker, zeroBaseline = false,
}: Props) {
  const m = { l: 46, r: 14, t: 10, b: 24 };
  const iw = Math.max(10, width - m.l - m.r);
  const ih = height - m.t - m.b;
  if (width < 40 || x.length < 2) return <svg width={width} height={height} />;

  const [xlo, xhi] = extent(x);
  let [ylo, yhi] = extent(y);
  if (zeroBaseline) ylo = Math.min(0, ylo);
  if (yhi === ylo) yhi = ylo + 1;
  const pad = (yhi - ylo) * 0.08;
  const sx = linScale(xlo, xhi, m.l, m.l + iw);
  const sy = linScale(ylo - pad, yhi + pad, m.t + ih, m.t);
  const xt = niceTicks(xlo, xhi, 5);
  const yt = niceTicks(ylo, yhi, 4);

  const line = path(x, y, sx, sy);
  const area = fill ? `${line}L${sx(xhi).toFixed(2)} ${sy(ylo - pad).toFixed(2)}L${sx(xlo).toFixed(2)} ${sy(ylo - pad).toFixed(2)}Z` : "";

  return (
    <svg width={width} height={height} role="img" aria-label={yLabel || "trace"}>
      {yt.map((v) => (
        <g key={"y" + v}>
          <line className="gridline" x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} />
          <text className="tick-label" x={m.l - 6} y={sy(v) + 3} textAnchor="end">{yFormat(v)}</text>
        </g>
      ))}
      {xt.map((v) => (
        <text key={"x" + v} className="tick-label" x={sx(v)} y={height - 8} textAnchor="middle">{v.toFixed(v < 1 ? 2 : 0)}</text>
      ))}
      {fill && <path d={area} fill={color} opacity={0.1} />}
      <path d={line} fill="none" stroke={color} strokeWidth={1.4} />
      {hmarker && (
        <g>
          <line x1={m.l} x2={m.l + iw} y1={sy(hmarker.value)} y2={sy(hmarker.value)} stroke={hmarker.color} strokeWidth={1} strokeDasharray="4 3" />
          <text x={m.l + iw} y={sy(hmarker.value) - 3} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill={hmarker.color}>{hmarker.label}</text>
        </g>
      )}
      {vmarker && (
        <g>
          <line x1={sx(vmarker.value)} x2={sx(vmarker.value)} y1={m.t} y2={m.t + ih} stroke={vmarker.color} strokeWidth={1.2} strokeDasharray="4 3" />
          <text x={sx(vmarker.value)} y={m.t + 8} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill={vmarker.color}>{vmarker.label}</text>
        </g>
      )}
      {xLabel && <text className="axis-label" x={m.l + iw / 2} y={height - 0} textAnchor="middle">{xLabel}</text>}
      {yLabel && <text className="axis-label" x={12} y={m.t + ih / 2} textAnchor="middle" transform={`rotate(-90 12 ${m.t + ih / 2})`}>{yLabel}</text>}
    </svg>
  );
}
