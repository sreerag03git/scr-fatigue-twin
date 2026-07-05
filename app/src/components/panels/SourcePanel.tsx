import { useRef } from "react";
import { useStore } from "../../state/store";
import { Badge, Field, Panel } from "../ui";

export function SourcePanel() {
  const { source, setSource, synthetic, patchSynthetic, uploadFile, uploadHealth, run } = useStore();
  const fileRef = useRef<HTMLInputElement>(null);

  const num = (label: string, key: keyof typeof synthetic, step = 0.1, min = 0) => (
    <Field label={label}>
      <input
        className="input" type="number" step={step} min={min} value={synthetic[key]}
        onChange={(e) => patchSynthetic({ [key]: Number(e.target.value) } as never)}
      />
    </Field>
  );

  return (
    <Panel index="00" title="Data source" right={<Badge kind={source === "synthetic" ? "badge--synthetic" : "badge--real"}>{source === "synthetic" ? "Synthetic" : "Measured"}</Badge>}>
      <div className="seg" style={{ width: "100%" }}>
        <button style={{ flex: 1 }} aria-pressed={source === "synthetic"} onClick={() => setSource("synthetic")}>Synthetic</button>
        <button style={{ flex: 1 }} aria-pressed={source === "upload"} onClick={() => setSource("upload")}>Upload MRU</button>
      </div>

      {source === "synthetic" ? (
        <>
          <p className="tiny muted" style={{ margin: "10px 0 8px" }}>
            Calibrated JONSWAP + RAO generator. Demo/fallback only — every derived value is badged synthetic.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 10px" }}>
            {num("Hs [m]", "hs", 0.1)}
            {num("Tp [s]", "tp", 0.5)}
            {num("γ (peak)", "gamma", 0.1, 1)}
            {num("Duration [s]", "duration", 60)}
            {num("fs [Hz]", "fs", 0.5)}
            {num("Seed", "seed", 1)}
          </div>
        </>
      ) : (
        <>
          <p className="tiny muted" style={{ margin: "10px 0 8px" }}>
            CSV / Parquet with a timestamp and motion channels (heave required). Auto-detects rate; reports data health.
          </p>
          <input ref={fileRef} type="file" accept=".csv,.parquet" style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && uploadFile(e.target.files[0])} />
          <button className="btn" style={{ width: "100%" }} onClick={() => fileRef.current?.click()}>
            Choose file…
          </button>
          {uploadHealth && (
            <div style={{ marginTop: 10 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="tiny muted">Data health</span>
                <Badge kind={uploadHealth.ok ? "badge--pass" : "badge--fail"}>{uploadHealth.ok ? "OK" : "Blocked"}</Badge>
              </div>
              <div className="mono tiny muted" style={{ marginTop: 4 }}>
                {uploadHealth.channels.join(", ") || "—"} · {uploadHealth.fs_hz.toFixed(2)} Hz · {uploadHealth.n_used.toLocaleString()} samples
              </div>
              {uploadHealth.flags.map((f, i) => (
                <div key={i} className="flag">⚑ {f}</div>
              ))}
            </div>
          )}
        </>
      )}
      <button className="btn btn--primary" style={{ width: "100%", marginTop: 12 }} onClick={run}>
        Ingest & run
      </button>
    </Panel>
  );
}
