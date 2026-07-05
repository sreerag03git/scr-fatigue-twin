import { Component, type ErrorInfo, type ReactNode } from "react";

interface State { error: Error | null }

// Global boundary: a render error shows a recovery card, never a white screen.
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("UI error boundary:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ display: "grid", placeItems: "center", height: "100vh", padding: 24 }}>
          <div className="panel" style={{ maxWidth: 480, padding: 20 }}>
            <div className="eyebrow" style={{ color: "var(--alarm)" }}>Interface error</div>
            <p style={{ color: "var(--text-mid)", fontSize: 13 }}>
              A panel failed to render. The physics core and your data are unaffected.
            </p>
            <pre className="mono" style={{ fontSize: 11, color: "var(--text-lo)", whiteSpace: "pre-wrap" }}>
              {this.state.error.message}
            </pre>
            <button className="btn btn--primary" onClick={() => this.setState({ error: null })}>
              Reload panels
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
