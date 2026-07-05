import { linScale, logScale, logTicks, niceTicks, path } from "./util";

interface Props {
  freq: number[];
  motionPsd: number[];
  stressPsd: number[]; // Pa^2/Hz
  height?: number;
  width: number;
}

// Motion PSD [m^2/Hz] and stress PSD [MPa^2/Hz] on a shared log-y axis.
export function SpectrumChart({ freq, motionPsd, stressPsd, width, height = 210 }: Props) {
  const m = { l: 46, r: 14, t: 12, b: 26 };
  const iw = Math.max(10, width - m.l - m.r);
  const ih = height - m.t - m.b;
  if (width < 40 || freq.length < 2) return <svg width={width} height={height} />;

  const stressMpa = stressPsd.map((v) => v / 1e12);
  const fMax = 0.4;
  const sx = linScale(0, fMax, m.l, m.l + iw);

  const all = [...motionPsd, ...stressMpa].filter((v) => v > 0);
  const yMin = Math.max(1e-6, Math.min(...all) * 0.5);
  const yMax = Math.max(...all) * 2;
  const sy = logScale(yMin, yMax, m.t + ih, m.t);

  const xt = niceTicks(0, fMax, 5);
  const yt = logTicks(yMin, yMax);

  return (
    <svg width={width} height={height} role="img" aria-label="Response spectra">
      {yt.map((v) => (
        <g key={"y" + v}>
          <line className="gridline" x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} />
          <text className="tick-label" x={m.l - 6} y={sy(v) + 3} textAnchor="end">
            {v >= 1 ? v.toExponential(0) : v.toExponential(0)}
          </text>
        </g>
      ))}
      {xt.map((v) => (
        <g key={"x" + v}>
          <line className="gridline-2" x1={sx(v)} x2={sx(v)} y1={m.t} y2={m.t + ih} />
          <text className="tick-label" x={sx(v)} y={height - 10} textAnchor="middle">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      {/* wave band 0.05–0.30 Hz shading */}
      <rect x={sx(0.05)} y={m.t} width={sx(0.3) - sx(0.05)} height={ih} fill="#33b7c40a" />

      <path d={path(freq, motionPsd, sx, sy)} fill="none" stroke="var(--signal)" strokeWidth={1.5} opacity={0.85} />
      <path d={path(freq, stressMpa, sx, sy)} fill="none" stroke="var(--amber)" strokeWidth={1.6} />

      <text className="axis-label" x={m.l + iw / 2} y={height - 0} textAnchor="middle">
        Frequency [Hz]
      </text>
      <g fontFamily="var(--font-mono)" fontSize={10}>
        <rect x={m.l + 8} y={m.t + 4} width={9} height={2.5} fill="var(--signal)" />
        <text x={m.l + 21} y={m.t + 8} fill="var(--text-mid)">motion [m²/Hz]</text>
        <rect x={m.l + 8} y={m.t + 16} width={9} height={2.5} fill="var(--amber)" />
        <text x={m.l + 21} y={m.t + 20} fill="var(--text-mid)">stress [MPa²/Hz]</text>
      </g>
    </svg>
  );
}
