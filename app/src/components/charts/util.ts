// Minimal scale/tick/path helpers for the bespoke SVG charts.

export type Scale = (v: number) => number;

export function linScale(d0: number, d1: number, r0: number, r1: number): Scale {
  const m = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * m;
}

export function logScale(d0: number, d1: number, r0: number, r1: number): Scale {
  const l0 = Math.log10(Math.max(d0, 1e-30));
  const l1 = Math.log10(Math.max(d1, 1e-30));
  const m = l1 === l0 ? 0 : (r1 - r0) / (l1 - l0);
  return (v) => r0 + (Math.log10(Math.max(v, 1e-30)) - l0) * m;
}

export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (span / count) / step0;
  const step = err >= 7.5 ? step0 * 10 : err >= 3.5 ? step0 * 5 : err >= 1.5 ? step0 * 2 : step0;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(Math.max(min, 1e-30)));
  const hi = Math.ceil(Math.log10(Math.max(max, 1e-30)));
  const out: number[] = [];
  for (let e = lo; e <= hi; e++) out.push(Math.pow(10, e));
  return out;
}

export function path(xs: number[], ys: number[], sx: Scale, sy: Scale): string {
  let d = "";
  for (let i = 0; i < xs.length; i++) {
    const x = sx(xs[i]);
    const y = sy(ys[i]);
    if (!isFinite(x) || !isFinite(y)) continue;
    d += (d ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2);
  }
  return d;
}

export function band(xs: number[], lo: number[], hi: number[], sx: Scale, sy: Scale): string {
  let top = "";
  let bot = "";
  for (let i = 0; i < xs.length; i++) {
    const x = sx(xs[i]).toFixed(2);
    top += (top ? "L" : "M") + x + " " + sy(hi[i]).toFixed(2);
  }
  for (let i = xs.length - 1; i >= 0; i--) {
    const x = sx(xs[i]).toFixed(2);
    bot += "L" + x + " " + sy(lo[i]).toFixed(2);
  }
  return top + bot + "Z";
}

export function extent(a: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of a) {
    if (!isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!isFinite(lo)) return [0, 1];
  return [lo, hi];
}
