// Thin typed fetch client. Every call surfaces backend error detail so panels
// can render a real error state instead of throwing to a white screen.

import type {
  AnalysisConfig,
  AnalyzeResponse,
  FleetEconomics,
  IngestResponse,
  RunListResponse,
  SNClass,
  StoredRun,
  SyntheticParams,
  ValidationResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Backend unreachable — is the server running?", 0);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep default */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => req<{ status: string; core_version: string }>("/api/health"),
  referenceConfig: () => req<AnalysisConfig>("/api/reference-config"),
  snClasses: () => req<SNClass[]>("/api/sn-classes"),
  validation: () => req<ValidationResponse>("/api/validation"),
  economics: () => req<FleetEconomics>("/api/economics"),

  analyzeSynthetic: (config: AnalysisConfig, synthetic: SyntheticParams) =>
    req<AnalyzeResponse>("/api/analyze/synthetic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, synthetic }),
    }),

  ingest: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return req<IngestResponse>("/api/ingest", { method: "POST", body: form });
  },

  analyzeUpload: (config: AnalysisConfig, token: string) =>
    req<AnalyzeResponse>("/api/analyze/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config, token }),
    }),

  runs: () => req<RunListResponse>("/api/runs"),
  getRun: (id: number) => req<StoredRun>(`/api/runs/${id}`),

  // Trigger a browser download of the reproducible provenance bundle for a run.
  exportRun: (runId: number) => {
    const a = document.createElement("a");
    a.href = `/api/runs/${runId}/export`;
    a.download = `scr-twin-run-${runId}-provenance.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  },
};
