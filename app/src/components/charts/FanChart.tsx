import { band, linScale, niceTicks, path } from "./util";

interface Props {
  years: number[];
  low: number[];
  median: number[];
  high: number[];
  width: number;
  height?: number;
}

// The signature visual: remaining-life 90% credible band contracting as
// monitoring time accrues. Bold here; quiet everywhere else.
export function FanChart({ years, low, median, high, width, height = 300 }: Props) {
  const m = { l: 48, r: 16, t: 16, b: 30 };
  const iw = Math.max(10, width - m.l - m.r);
  const ih = height - m.t - m.b;
  if (width < 40 || years.length < 2) return <svg width={width} height={height} />;

  const xMax = years[years.length - 1];
  const sx = linScale(0, xMax, m.l, m.l + iw);
  // Robust y-ceiling: keep the wide prior visible but don't let it dominate.
  const yMax = Math.min(high[0], Math.max(median[0] * 3, high[Math.min(3, high.length - 1)] * 1.2));
  const sy = linScale(0, yMax, m.t + ih, m.t);

  const xt = niceTicks(0, xMax, 6);
  const yt = niceTicks(0, yMax, 5);
  const clipId = "fanclip";

  return (
    <svg width={width} height={height} role="img" aria-label="Remaining-life posterior fan">
      <defs>
        <clipPath id={clipId}>
          <rect x={m.l} y={m.t} width={iw} height={ih} />
        </clipPath>
        <linearGradient id="fanfill" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="var(--violet)" stopOpacity="0.30" />
          <stop offset="100%" stopColor="var(--signal)" stopOpacity="0.16" />
        </linearGradient>
      </defs>

      {yt.map((v) => (
        <g key={"y" + v}>
          <line className="gridline" x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} />
          <text className="tick-label" x={m.l - 7} y={sy(v) + 3} textAnchor="end">{v.toFixed(0)}</text>
        </g>
      ))}
      {xt.map((v) => (
        <g key={"x" + v}>
          <line className="gridline-2" x1={sx(v)} x2={sx(v)} y1={m.t} y2={m.t + ih} />
          <text className="tick-label" x={sx(v)} y={height - 12} textAnchor="middle">{v.toFixed(0)}</text>
        </g>
      ))}

      <g clipPath={`url(#${clipId})`}>
        <path d={band(years, low, high, sx, sy)} fill="url(#fanfill)" stroke="none" />
        <path d={path(years, high, sx, sy)} fill="none" stroke="var(--violet)" strokeWidth={1} opacity={0.55} strokeDasharray="3 3" />
        <path d={path(years, low, sx, sy)} fill="none" stroke="var(--signal)" strokeWidth={1} opacity={0.6} strokeDasharray="3 3" />
        <path d={path(years, median, sx, sy)} fill="none" stroke="var(--signal-2)" strokeWidth={2} />
      </g>

      {/* year-0 P50 marker */}
      <circle cx={sx(0)} cy={sy(Math.min(median[0], yMax))} r={3} fill="var(--signal-2)" />

      <text className="axis-label" x={m.l + iw / 2} y={height - 0} textAnchor="middle">
        Monitoring time [yr] →
      </text>
      <text className="axis-label" x={12} y={m.t + ih / 2} textAnchor="middle" transform={`rotate(-90 12 ${m.t + ih / 2})`}>
        Remaining life [yr]
      </text>
    </svg>
  );
}
