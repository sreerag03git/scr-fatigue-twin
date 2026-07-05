// Engineering number formatting. Terse, unit-aware, tabular.

export function years(v: number): string {
  if (!isFinite(v)) return "∞";
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`;
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function usd(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(v / 1e3).toFixed(0)}k`;
  return `$${v.toFixed(0)}`;
}

export function sci(v: number, digits = 2): string {
  if (v === 0) return "0";
  if (!isFinite(v)) return "∞";
  const exp = Math.floor(Math.log10(Math.abs(v)));
  if (exp >= -2 && exp <= 3) return v.toPrecision(digits + 1);
  const mant = v / Math.pow(10, exp);
  return `${mant.toFixed(digits)}e${exp >= 0 ? "+" : ""}${exp}`;
}

export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

export function fixed(v: number, d = 2): string {
  return isFinite(v) ? v.toFixed(d) : "—";
}
