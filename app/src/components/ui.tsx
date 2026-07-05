import type { ReactNode } from "react";
import { useSize } from "../hooks/useSize";

export function Panel({
  index, title, right, children, className,
}: { index?: string; title: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={"panel " + (className ?? "")}>
      <header className="panel__head">
        {index && <span className="panel__index">{index}</span>}
        <span className="panel__title">{title}</span>
        <span className="panel__spacer" />
        {right}
      </header>
      <div className="panel__body">{children}</div>
    </section>
  );
}

export function Metric({
  label, value, unit, tone, size,
}: {
  label: string; value: ReactNode; unit?: string;
  tone?: "amber" | "signal" | "alarm"; size?: "lg";
}) {
  const cls =
    "metric__value" +
    (size === "lg" ? " metric__value--lg" : "") +
    (tone ? ` metric__value--${tone}` : "");
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <span className={cls + " mono"}>
        {value}
        {unit && <span className="metric__unit">{unit}</span>}
      </span>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
    </label>
  );
}

export function Badge({ kind, children }: { kind?: string; children: ReactNode }) {
  return <span className={"badge " + (kind ?? "")}>{children}</span>;
}

// Render-prop that measures available width for a chart.
export function Measured({ height, children }: { height: number; children: (w: number) => ReactNode }) {
  const [ref, width] = useSize<HTMLDivElement>();
  return (
    <div ref={ref} style={{ width: "100%", height }}>
      {width > 0 ? children(width) : null}
    </div>
  );
}

export function EmptyState({ text }: { text: string }) {
  return (
    <div style={{ display: "grid", placeItems: "center", height: "100%", minHeight: 80, color: "var(--text-lo)", fontSize: 12 }}>
      {text}
    </div>
  );
}
