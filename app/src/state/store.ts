import { create } from "zustand";
import { api, ApiError } from "../api/client";
import type {
  AnalysisConfig,
  AnalyzeResponse,
  DataHealth,
  RunSummary,
  SNClass,
  SyntheticParams,
  ValidationResponse,
} from "../api/types";

type Source = "synthetic" | "upload";
type Status = "idle" | "loading" | "ready" | "error";

const DEFAULT_SYNTH: SyntheticParams = {
  hs: 4.0, tp: 11.0, gamma: 2.5, duration: 1800, fs: 4.0, seed: 20240705,
};

interface Store {
  booted: boolean;
  bootError: string | null;
  config: AnalysisConfig | null;
  snClasses: SNClass[];
  validation: ValidationResponse | null;

  source: Source;
  synthetic: SyntheticParams;
  uploadToken: string | null;
  uploadHealth: DataHealth | null;

  status: Status;
  result: AnalyzeResponse | null;
  error: string | null;

  runs: RunSummary[];
  activeRunId: number | null;

  theme: "dark" | "light";

  boot: () => Promise<void>;
  run: () => Promise<void>;
  refreshRuns: () => Promise<void>;
  loadRun: (id: number) => Promise<void>;
  setSource: (s: Source) => void;
  patchSynthetic: (p: Partial<SyntheticParams>) => void;
  patchRiser: (p: Partial<AnalysisConfig["riser"]>) => void;
  patchTransfer: (p: Partial<AnalysisConfig["transfer"]>) => void;
  patchEnv: (p: Partial<AnalysisConfig["environment"]>) => void;
  patchConfig: (p: Partial<Pick<AnalysisConfig, "seed" | "n_monte_carlo">>) => void;
  uploadFile: (file: File) => Promise<void>;
  toggleTheme: () => void;
}

export const useStore = create<Store>((set, get) => ({
  booted: false,
  bootError: null,
  config: null,
  snClasses: [],
  validation: null,
  source: "synthetic",
  synthetic: DEFAULT_SYNTH,
  uploadToken: null,
  uploadHealth: null,
  status: "idle",
  result: null,
  error: null,
  runs: [],
  activeRunId: null,
  theme: "dark",

  boot: async () => {
    try {
      const [config, snClasses, validation] = await Promise.all([
        api.referenceConfig(),
        api.snClasses(),
        api.validation(),
      ]);
      set({ booted: true, config, snClasses, validation, bootError: null });
      await get().run();
    } catch (e) {
      set({ booted: true, bootError: e instanceof Error ? e.message : String(e) });
    }
  },

  run: async () => {
    const { config, source, synthetic, uploadToken } = get();
    if (!config) return;
    set({ status: "loading", error: null });
    try {
      const result =
        source === "synthetic"
          ? await api.analyzeSynthetic(config, synthetic)
          : uploadToken
            ? await api.analyzeUpload(config, uploadToken)
            : null;
      if (!result) {
        set({ status: "error", error: "No uploaded dataset selected." });
        return;
      }
      set({ status: "ready", result, activeRunId: result.run_id ?? null });
      void get().refreshRuns();
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      set({ status: "error", error: msg });
    }
  },

  refreshRuns: async () => {
    try {
      const r = await api.runs();
      set({ runs: r.runs });
    } catch {
      /* run history is non-critical; leave it as-is on error */
    }
  },

  loadRun: async (id) => {
    set({ status: "loading", error: null });
    try {
      const r = await api.getRun(id);
      const src = r.payload.source;
      set({
        result: r.payload,
        config: r.config,
        activeRunId: id,
        status: "ready",
        source: src?.kind === "upload" ? "upload" : "synthetic",
        synthetic:
          src?.kind === "synthetic"
            ? {
                hs: Number(src.hs), tp: Number(src.tp), gamma: Number(src.gamma),
                duration: Number(src.duration), fs: Number(src.fs), seed: Number(src.seed),
              }
            : get().synthetic,
      });
    } catch (e) {
      set({ status: "error", error: e instanceof Error ? e.message : String(e) });
    }
  },

  setSource: (s) => set({ source: s }),
  patchSynthetic: (p) => set((st) => ({ synthetic: { ...st.synthetic, ...p } })),
  patchRiser: (p) =>
    set((st) => (st.config ? { config: { ...st.config, riser: { ...st.config.riser, ...p } } } : {})),
  patchTransfer: (p) =>
    set((st) => (st.config ? { config: { ...st.config, transfer: { ...st.config.transfer, ...p } } } : {})),
  patchEnv: (p) =>
    set((st) => (st.config ? { config: { ...st.config, environment: { ...st.config.environment, ...p } } } : {})),
  patchConfig: (p) => set((st) => (st.config ? { config: { ...st.config, ...p } } : {})),

  uploadFile: async (file) => {
    set({ status: "loading", error: null });
    try {
      const res = await api.ingest(file);
      set({ source: "upload", uploadToken: res.token, uploadHealth: res.health });
      if (res.health.ok) await get().run();
      else set({ status: "error", error: "Data health check failed: " + res.health.flags.join("; ") });
    } catch (e) {
      set({ status: "error", error: e instanceof Error ? e.message : String(e) });
    }
  },

  toggleTheme: () => {
    const theme = get().theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);
    set({ theme });
  },
}));
