import { useEffect, useState } from "react";
import { useStore } from "../state/store";

// Boot stages tied to what the app is actually doing while it starts.
const STAGES = [
  "Initialising physics core",
  "Loading DNV-RP-C203 S-N library",
  "Solving reference catenary",
  "Deriving TDP transfer function",
  "Running validation gates",
  "Computing remaining-life posterior",
];

export function LoadingScreen() {
  const { booted, result, status, validation } = useStore();
  const [stage, setStage] = useState(0);
  const [gone, setGone] = useState(false);

  const done = !!result;

  // Advance the status lines until the first result lands.
  useEffect(() => {
    if (done) return;
    const t = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 620);
    return () => clearInterval(t);
  }, [done]);

  // Fade out, then unmount, once the first analysis is ready.
  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => setGone(true), 480);
    return () => clearTimeout(t);
  }, [done]);

  if (gone) return null;

  // Progress blends real milestones with the ticking stage.
  const progress = done
    ? 100
    : Math.min(
        94,
        12 + stage * 12 + (booted ? 12 : 0) + (validation ? 8 : 0) + (status === "loading" ? 10 : 0),
      );

  return (
    <div className={"splash" + (done ? " splash--hide" : "")} role="status" aria-live="polite">
      <div className="splash__inner">
        <div className="splash__mark" aria-hidden>
          <svg viewBox="0 0 130 120" width="96" height="88">
            <circle className="splash__tdp" cx="12" cy="104" r="9" fill="none" stroke="var(--amber)" strokeWidth="1" />
            <path
              className="splash__cat"
              d="M12 104 C 46 104, 52 26, 120 16"
              fill="none"
              stroke="var(--signal)"
              strokeWidth="2.6"
              strokeLinecap="round"
            />
            <circle cx="12" cy="104" r="4.5" fill="var(--amber)" />
            <circle cx="120" cy="16" r="3.6" fill="var(--signal-2)" />
          </svg>
        </div>

        <div className="splash__title">SCR·TWIN</div>
        <div className="splash__sub">TDP Fatigue Integrity Console</div>

        <div className="splash__bar" aria-hidden>
          <span className="splash__fill" style={{ width: progress + "%" }} />
        </div>
        <div className="splash__stage mono">{done ? "Ready" : STAGES[stage] + "…"}</div>

        <div className="splash__foot mono">
          Physics-based digital twin · DNV-RP-C203 · ASTM E1049 · Dirlik
        </div>
      </div>
    </div>
  );
}
