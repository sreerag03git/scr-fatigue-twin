"""SCR-Twin - shareable Streamlit console.

A deployable (Streamlit Community Cloud) front-end over the *same* tested physics
core as the desktop app. It reuses ``server.service`` so every number matches the
FastAPI/React build exactly - this file only handles UI, animation and charts.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, point Streamlit Cloud at streamlit_app.py.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

# Make the local physics core and service layer importable without installation
# (so Streamlit Cloud works straight from the repo).
_ROOT = pathlib.Path(__file__).parent
for _p in (_ROOT / "core", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from fpdf import FPDF  # noqa: E402

from scr_twin_core import ingest as ingest_mod  # noqa: E402
from scr_twin_core import validation as validation_mod  # noqa: E402
from scr_twin_core.config import AnalysisConfig, EnvironmentConfig, RiserConfig, TransferConfig  # noqa: E402
from server import service  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette / theme
# --------------------------------------------------------------------------- #
SIGNAL, SIGNAL2, AMBER, ALARM = "#0f8f9c", "#0b7079", "#b4791a", "#c33d28"
GRID, TEXT, TEXTHI, PAPER = "#dce4e8", "#3f515c", "#1a2830", "rgba(0,0,0,0)"
GOOD = "#1f8a5b"
SN_CLASSES = ["B1", "B2", "C", "C1", "C2", "D", "E", "F", "F1", "F3", "G"]

st.set_page_config(
    page_title="SCR-Twin - TDP Fatigue Integrity Console",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background:
        radial-gradient(120% 80% at 50% -8%, #ffffff 0%, #eef1f3 62%); color:#1f2b33; }
      section[data-testid="stSidebar"] { background:#f5f7f8; border-right:1px solid #d3dbe0; }
      h1,h2,h3,h4 { letter-spacing:.01em; color:#1a2830; }
      .mono, code, [data-testid="stMetricValue"] {
        font-family:"JetBrains Mono","Cascadia Mono",Consolas,monospace !important; }
      .brand { display:flex; align-items:center; gap:12px; margin:-6px 0 2px; }
      .brand h1 { font-size:26px; margin:0; letter-spacing:.16em; color:#1a2830; font-weight:600; }
      .brand .sub { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:#6b7d88; }
      .kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:6px 0 4px; }
      .kpi { background:#ffffff; border:1px solid #e2e8ec; border-radius:8px; padding:11px 13px;
        box-shadow:0 1px 2px rgba(20,40,55,0.04); }
      .kpi .lab { font-size:9.5px; letter-spacing:.09em; text-transform:uppercase; color:#6b7d88; }
      .kpi .val { font-family:"JetBrains Mono",Consolas,monospace; font-size:22px; color:#1a2830;
        line-height:1.15; font-variant-numeric:tabular-nums; margin-top:3px; }
      .kpi .val small { font-size:11px; color:#6b7d88; margin-left:3px; }
      .kpi .val.sig { color:#0b7079; } .kpi .val.amber { color:#b4791a; } .kpi .val.alarm { color:#c33d28; }
      .tag { display:inline-block; font-family:monospace; font-size:10px; letter-spacing:.06em; padding:2px 8px;
        border-radius:5px; border:1px solid #c2ccd3; color:#3f515c; background:#ffffff; }
      .tag.syn { color:#b4791a; border-color:#dcbd86; } .tag.pass { color:#1f8a5b; border-color:#a7d3bd; }
      .tag.fail { color:#c33d28; border-color:#e2a99f; }
      .sec { font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:#56707d;
        border-bottom:1px solid #d3dbe0; padding-bottom:5px; margin:14px 0 8px; }
      .gate { display:flex; align-items:center; gap:9px; padding:5px 2px; border-bottom:1px solid #e6eaec; font-size:12.5px; }
      .gate .dot { width:8px; height:8px; border-radius:50%; flex:none; }
      .gate .actual { margin-left:auto; font-family:monospace; font-size:11px; color:#56707d; }
      .foot { color:#8496a0; font-size:10.5px; font-family:monospace; letter-spacing:.05em; }
      /* landing */
      .land { max-width:860px; margin:2vh auto 0; text-align:center; }
      .land h1 { font-size:44px; letter-spacing:.18em; margin:14px 0 2px; }
      .land .tagline { font-size:13px; letter-spacing:.16em; text-transform:uppercase; color:#56707d; }
      .land .lede { color:#3f515c; font-size:15px; line-height:1.6; max-width:640px; margin:20px auto 4px; }
      .chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:18px 0 6px; }
      .chip { font-family:monospace; font-size:11px; color:#3f515c; background:#ffffff; border:1px solid #d3dbe0;
        border-radius:20px; padding:4px 12px; }
      .flow { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; margin:22px 0 8px; }
      .flowcard { background:#ffffff; border:1px solid #e2e8ec; border-radius:8px; padding:12px 14px; width:150px; text-align:left; }
      .flowcard .n { font-family:monospace; font-size:10px; color:#0f8f9c; letter-spacing:.1em; }
      .flowcard .t { font-size:12.5px; color:#1a2830; margin-top:4px; font-weight:600; }
      .flowcard .d { font-size:10.5px; color:#6b7d88; margin-top:3px; line-height:1.4; }
      /* live run */
      .livebar { display:flex; align-items:center; gap:8px; font-family:monospace; font-size:12px;
        letter-spacing:.08em; color:#0b7079; text-transform:uppercase; margin:6px 0 8px; }
      .livedot { width:9px; height:9px; border-radius:50%; background:#c33d28;
        box-shadow:0 0 0 0 rgba(195,61,40,.5); animation:pulse 1.1s infinite; }
      @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(195,61,40,.45);} 70%{box-shadow:0 0 0 7px rgba(195,61,40,0);} 100%{box-shadow:0 0 0 0 rgba(195,61,40,0);} }
      .livestatus { font-family:monospace; font-size:12.5px; color:#3f515c; margin:2px 0 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session flow: landing -> console; ran gates the dashboard.
st.session_state.setdefault("launched", False)
st.session_state.setdefault("ran", False)


# --------------------------------------------------------------------------- #
# Cached compute (keyed on serialisable inputs -> efficient re-runs)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def gates() -> list[dict]:
    return [g.as_dict() for g in validation_mod.run_all_gates(seed=0)]


@st.cache_data(show_spinner=False)
def analyze_synthetic(config_json: str, hs: float, tp: float, gamma: float,
                      duration: float, fs: float, seed: int) -> dict:
    cfg = AnalysisConfig.model_validate_json(config_json)
    heave, fsr = service.make_synthetic(hs, tp, gamma, duration, fs, seed)
    return service.analyze(cfg, heave, fsr, is_synthetic=True)


@st.cache_data(show_spinner=False)
def analyze_upload(config_json: str, file_bytes: bytes) -> dict:
    cfg = AnalysisConfig.model_validate_json(config_json)
    rec = ingest_mod.load_mru_csv(io.BytesIO(file_bytes))
    if not rec.health.ok:
        return {"error": "; ".join(rec.health.flags) or "data health check failed",
                "health": rec.health.as_dict()}
    return service.analyze(cfg, rec.channels["heave"], rec.fs,
                           is_synthetic=False, data_health=rec.health.as_dict())


@st.cache_data(show_spinner=False)
def hires_heave(hs: float, tp: float, gamma: float, duration: float, fs: float, seed: int,
                target: int = 1800) -> tuple[list[float], list[float], float]:
    """High-resolution synthetic heave (decimated for smooth animation)."""
    heave, fsr = service.make_synthetic(hs, tp, gamma, duration, fs, seed)
    step = max(1, heave.size // target)
    h = heave[::step]
    t = np.arange(h.size) * step / fsr
    return [float(v) for v in t], [float(v) for v in h], float(fsr)


# --------------------------------------------------------------------------- #
# Formatting + Plotly helpers
# --------------------------------------------------------------------------- #
def life(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return "999+" if v >= 999 else f"{v:.0f}"


def kpi(label: str, value: str, unit: str = "", tone: str = "") -> str:
    u = f"<small>{unit}</small>" if unit else ""
    return f'<div class="kpi"><div class="lab">{label}</div><div class="val {tone}">{value}{u}</div></div>'


def kpi_row(items: list[str]) -> str:
    return '<div class="kpi-row">' + "".join(items) + "</div>"


def _fig(height: int) -> go.Figure:
    f = go.Figure()
    f.update_layout(
        height=height, margin=dict(l=56, r=18, t=14, b=42),
        paper_bgcolor=PAPER, plot_bgcolor=PAPER, showlegend=False,
        font=dict(color=TEXT, family="JetBrains Mono, Consolas, monospace", size=11),
        hoverlabel=dict(font_family="JetBrains Mono, monospace"),
    )
    return f


def spectra_fig(spec: dict) -> go.Figure:
    f = _fig(250)
    f.add_scatter(x=spec["freq"], y=spec["motion_psd"], line=dict(color=SIGNAL, width=1.8), yaxis="y")
    f.add_scatter(x=spec["freq"], y=spec["stress_psd"], line=dict(color=AMBER, width=1.8), yaxis="y2")
    f.update_layout(
        xaxis=dict(title="Frequency [Hz]", gridcolor=GRID, zeroline=False, range=[0, 0.4]),
        yaxis=dict(title="motion [m^2/Hz]", type="log", gridcolor=GRID, color=SIGNAL, zeroline=False),
        yaxis2=dict(title="stress [MPa^2/Hz]", type="log", overlaying="y", side="right", color=AMBER, showgrid=False),
    )
    return f


def fan_fig(fan: dict, p50: float, upto: int | None = None) -> go.Figure:
    f = _fig(330)
    n = len(fan["years"]) if upto is None else upto
    yrs = fan["years"][:n]
    f.add_scatter(x=yrs, y=fan["high"][:n], line=dict(width=0), hoverinfo="skip")
    f.add_scatter(x=yrs, y=fan["low"][:n], fill="tonexty", line=dict(width=0),
                  fillcolor="rgba(180,121,26,0.15)", hoverinfo="skip")
    f.add_scatter(x=yrs, y=fan["median"][:n], line=dict(color=SIGNAL2, width=2.4))
    f.add_hline(y=p50, line=dict(color=SIGNAL, width=0.8, dash="dot"))
    ymax = max(fan["high"]) * 1.05 if fan["high"] else None
    f.update_layout(
        xaxis=dict(title="Monitoring time [yr]", gridcolor=GRID, zeroline=False, range=[0, fan["years"][-1]]),
        yaxis=dict(title="Remaining life [yr]", gridcolor=GRID, zeroline=False, range=[0, ymax]),
    )
    return f


def pdf_hist_fig(post: dict) -> go.Figure:
    f = _fig(250)
    edges, counts = post["hist_edges"], post["hist_counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    f.add_bar(x=centers, y=counts, marker_color="rgba(15,143,156,0.35)",
              marker_line_color=SIGNAL, marker_line_width=0.3)
    for key, col in (("p10", AMBER), ("p50", SIGNAL2), ("p90", SIGNAL)):
        f.add_vline(x=post[key], line=dict(color=col, width=1.2, dash="dash"),
                    annotation_text=key.upper(), annotation_font_color=col, annotation_font_size=9)
    f.update_layout(xaxis=dict(title="Remaining life [yr]", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="MC members", gridcolor=GRID, zeroline=False))
    return f


def pof_fig(insp: dict) -> go.Figure:
    f = _fig(250)
    f.add_scatter(x=insp["pof_years"], y=insp["pof_vals"], line=dict(color=AMBER, width=2),
                  fill="tozeroy", fillcolor="rgba(180,121,26,0.10)")
    f.add_hline(y=insp["target_pof"], line=dict(color=ALARM, width=1, dash="dash"),
                annotation_text="target", annotation_font_color=ALARM, annotation_font_size=9)
    f.add_vline(x=insp["next_inspection_year"], line=dict(color=SIGNAL2, width=1.4, dash="dot"),
                annotation_text="inspect", annotation_font_color=SIGNAL2, annotation_font_size=9)
    f.update_layout(xaxis=dict(title="year", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="P(fail)", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


# --------------------------------------------------------------------------- #
# PDF report (matplotlib charts + fpdf2; ASCII text for the core fonts)
# --------------------------------------------------------------------------- #
def _mpl_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _mpl_spectra(spec: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.semilogy(spec["freq"], np.clip(spec["motion_psd"], 1e-12, None), color=SIGNAL, lw=1.4)
    ax2 = ax.twinx()
    ax2.semilogy(spec["freq"], np.clip(spec["stress_psd"], 1e-12, None), color=AMBER, lw=1.4)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.set_ylabel("motion PSD [m^2/Hz]", color=SIGNAL, fontsize=8)
    ax2.set_ylabel("stress PSD [MPa^2/Hz]", color=AMBER, fontsize=8)
    for a in (ax, ax2):
        a.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    return _mpl_png(fig)


def _mpl_fan(fan: dict, p50: float) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    yrs = fan["years"]
    ax.fill_between(yrs, fan["low"], fan["high"], color=AMBER, alpha=0.16, label="90% CI")
    ax.plot(yrs, fan["median"], color=SIGNAL2, lw=1.8, label="P50")
    ax.axhline(p50, color=SIGNAL, lw=0.7, ls=":")
    ax.set_xlabel("Monitoring time [yr]", fontsize=8)
    ax.set_ylabel("Remaining life [yr]", fontsize=8)
    ax.set_ylim(0, max(fan["high"]) * 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    return _mpl_png(fig)


def _mpl_hist(post: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    edges, counts = post["hist_edges"], post["hist_counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
    ax.bar(centers, counts, width=width, color=SIGNAL, alpha=0.35, edgecolor=SIGNAL, linewidth=0.3)
    for key, col in (("p10", AMBER), ("p50", SIGNAL2), ("p90", SIGNAL)):
        ax.axvline(post[key], color=col, ls="--", lw=1.1)
        ax.text(post[key], ax.get_ylim()[1] * 0.92, key.upper(), color=col, fontsize=7, ha="center")
    ax.set_xlabel("Remaining life [yr]", fontsize=8)
    ax.set_ylabel("MC members", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    return _mpl_png(fig)


@st.cache_data(show_spinner=False)
def build_pdf(config_json: str, source_kind: str, payload_json: str) -> bytes:
    """Assemble a one-page PDF integrity report (cached by inputs)."""
    cfg = AnalysisConfig.model_validate_json(config_json)
    p = json.loads(payload_json)
    dmg, post, insp, econ, prov = p["damage"], p["posterior"], p["inspection"], p["economics"], p["provenance"]
    sea, env = p["sea_state"], p["environment"]
    g = gates()

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(26, 40, 48)
    pdf.cell(0, 8, "SCR-TWIN  -  TDP Fatigue Integrity Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(90, 110, 122)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    badge = "SYNTHETIC (demo)" if source_kind == "synthetic" else "MEASURED MRU"
    pdf.cell(0, 5, f"Generated {stamp}   |   Source: {badge}   |   config {prov['config_sha256'][:12]}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    def section(title: str) -> None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 143, 156)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(40, 55, 65)
        pdf.set_font("Helvetica", "", 9)

    def kv(rows: list[tuple[str, str]]) -> None:
        for k, v in rows:
            pdf.cell(70, 5, k)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, v, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)

    r = cfg.riser
    section("Configuration")
    kv([
        ("Riser OD x WT [m]", f"{r.outer_diameter:.4f} x {r.wall_thickness:.4f}"),
        ("Water depth / hang-off", f"{r.water_depth:.0f} m  /  {r.hang_off_angle_deg:.0f} deg from vertical"),
        ("SCF / S-N class", f"{r.scf:.2f}  /  DNV {r.sn_class}"),
        ("Transfer route", cfg.transfer.route),
        ("Arabian Gulf correction", f"ON (factor {env['factor']:.3f})" if env["enabled"] else "OFF"),
        ("Monte Carlo / seed", f"{post['n_members']:,} members  /  seed {prov['seed']}"),
    ])
    pdf.ln(1)
    section("Sea state (identified)")
    kv([("Hs / Tp / Tz", f"{sea['hs']:.2f} m  /  {sea['tp']:.1f} s  /  {sea['tz']:.1f} s"),
        ("JONSWAP gamma", f"{sea['gamma']:.1f}")])
    pdf.ln(1)
    section("Fatigue result")
    kv([
        ("Deterministic life", f"{life(dmg['deterministic_life_years'])} yr"),
        ("Annual damage (time / spectral)", f"{dmg['annual_rate_time']:.2e}  /  {dmg['annual_rate_spectral']:.2e} /yr"),
        ("Remaining life P10 / P50 / P90", f"{life(post['p10'])} / {life(post['p50'])} / {life(post['p90'])} yr"),
    ])
    pdf.ln(1)
    section("Decision")
    kv([
        ("Next inspection", f"{insp['next_inspection_year']:.1f} yr  (target PoF {insp['target_pof']*100:.1f}%)"),
        ("Fleet saving (20u, 20yr)", f"${econ['fleet_saving_low_usd']/1e6:.1f}M - ${econ['fleet_saving_high_usd']/1e6:.1f}M"),
        ("Sensor payback", f"{econ['payback_low_yr']:.1f} - {econ['payback_high_yr']:.1f} yr"),
    ])
    pdf.ln(2)

    # Charts
    for png in (_mpl_fan(p["bayesian_fan"], post["p50"]), _mpl_spectra(p["spectrum"]), _mpl_hist(post)):
        pdf.image(io.BytesIO(png), w=185)
        pdf.ln(1)

    section("Validation gates (spec section 5)")
    for x in g:
        pdf.set_text_color(31, 138, 91) if x["passed"] else pdf.set_text_color(195, 61, 40)
        pdf.cell(8, 5, "PASS" if x["passed"] else "FAIL")
        pdf.set_text_color(40, 55, 65)
        pdf.cell(62, 5, x["name"][:38])
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, x["actual"][:70], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(120, 135, 145)
    pdf.multi_cell(0, 4,
                   "Physics-based digital twin. core v{cv}, numpy {nv}, scipy {sv}. Reproducible from config + "
                   "seed + versions. DNV-RP-C203 / ASTM E1049 / Dirlik. Reference SCR preset and reduced-order "
                   "H(f) are illustrative, not project data.".format(
                       cv=prov["core_version"], nv=prov["numpy_version"], sv=prov["scipy_version"]))
    return bytes(pdf.output())


# --------------------------------------------------------------------------- #
# Landing page
# --------------------------------------------------------------------------- #
def render_landing() -> None:
    st.markdown(
        '<div class="land">'
        '<svg width="72" height="72" viewBox="0 0 130 120">'
        '<circle cx="12" cy="104" r="9" fill="none" stroke="#b4791a" stroke-width="1"/>'
        '<path d="M12 104 C 46 104, 52 26, 120 16" fill="none" stroke="#0f8f9c" stroke-width="2.6" stroke-linecap="round"/>'
        '<circle cx="12" cy="104" r="4.5" fill="#b4791a"/><circle cx="120" cy="16" r="3.6" fill="#0b7079"/></svg>'
        '<h1>SCR&middot;TWIN</h1>'
        '<div class="tagline">TDP Fatigue Integrity Digital Twin</div>'
        '<div class="lede">Converts a floating unit&rsquo;s existing Motion Reference Unit recordings into a '
        'continuously-updated, probabilistic estimate of steel catenary riser fatigue life at the touchdown '
        'point &mdash; and schedules inspection from it. Real physics, deterministic, every number traceable.</div>'
        '<div class="chips">'
        + "".join(f'<span class="chip">{c}</span>' for c in
                  ["DNV-RP-C203", "ASTM E1049", "JONSWAP", "Morison H(f)", "Dirlik / Tovo-Benasciutti",
                   "Monte Carlo 10k", "Bayesian 1/sqrt(T)"])
        + '</div>'
        '<div class="flow">'
        + "".join(
            f'<div class="flowcard"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
            for n, t, d in [
                ("01", "MRU motion", "6-DOF hang-off record; synthetic or measured, health-checked"),
                ("02", "Layer 1 - H(f)", "Catenary + linearised Morison transfer to TDP bending moment"),
                ("03", "Layer 2 - damage", "Rainflow, DNV S-N, Miner; Dirlik cross-check"),
                ("04", "Layer 3 - posterior", "10k Monte Carlo + Bayesian remaining-life contraction"),
                ("05", "Decision", "Risk-based inspection schedule and fleet economics"),
            ])
        + '</div></div>',
        unsafe_allow_html=True,
    )
    c = st.columns([2, 1, 2])[1]
    if c.button("Launch console  →", type="primary", width="stretch"):
        st.session_state.launched = True
        st.rerun()
    st.markdown(
        '<div class="foot" style="text-align:center;margin-top:10px">Reference implementation &middot; '
        'reference SCR preset is illustrative, not project data</div>', unsafe_allow_html=True)


if not st.session_state.launched:
    render_landing()
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar - data source + configuration
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Data source")
source = st.sidebar.radio("source", ["Synthetic (demo)", "Upload MRU CSV"], label_visibility="collapsed")

synth: dict[str, float] = {}
upload_bytes: bytes | None = None
if source.startswith("Synthetic"):
    st.sidebar.caption("Calibrated JONSWAP + RAO generator - every derived value is badged **synthetic**.")
    c1, c2 = st.sidebar.columns(2)
    synth["hs"] = c1.number_input("Hs [m]", 0.5, 16.0, 4.0, 0.5)
    synth["tp"] = c2.number_input("Tp [s]", 4.0, 20.0, 11.0, 0.5)
    synth["gamma"] = c1.number_input("gamma peak", 1.0, 7.0, 2.5, 0.5)
    synth["duration"] = c2.number_input("Duration [s]", 300.0, 3600.0, 1800.0, 60.0)
    synth["fs"] = c1.number_input("fs [Hz]", 1.0, 10.0, 4.0, 1.0)
    synth["seed"] = c2.number_input("Seed", 0, 99_999_999, 20240705, 1)
else:
    up = st.sidebar.file_uploader("MRU CSV (time + heave/pitch...)", type=["csv"])
    if up is not None:
        upload_bytes = up.getvalue()
    st.sidebar.caption("Columns: `time_s, heave_m[, pitch_deg...]`. Malformed files degrade gracefully.")

st.sidebar.markdown("### Riser & analysis")
ref = RiserConfig.reference_scr()
with st.sidebar.expander("Steel catenary riser", expanded=True):
    od = st.number_input("Outer diameter [m]", 0.1, 1.5, ref.outer_diameter, 0.01, format="%.4f")
    wt = st.number_input("Wall thickness [m]", 0.005, 0.08, ref.wall_thickness, 0.001, format="%.4f")
    depth = st.number_input("Water depth [m]", 100.0, 3500.0, ref.water_depth, 50.0)
    ang = st.number_input("Hang-off [deg from vertical]", 1.0, 45.0, ref.hang_off_angle_deg, 1.0)
    scf = st.number_input("SCF", 1.0, 5.0, ref.scf, 0.05)
    sn_class = st.selectbox("DNV S-N class", SN_CLASSES, index=SN_CLASSES.index(ref.sn_class))

with st.sidebar.expander("Transfer function (Layer 1)"):
    route = st.radio("Route", ["reference", "analytic"], horizontal=True,
                     help="Reference = illustrative Route-2 table. Analytic = reduced-order Route-1.")

with st.sidebar.expander("Arabian Gulf correction", expanded=True):
    env_on = st.toggle("Apply correction", value=True)
    tfac = st.slider("Temperature factor", 0.72, 0.78, 0.75, 0.005, disabled=not env_on)
    sfac = st.slider("Salinity factor", 0.85, 0.90, 0.875, 0.005, disabled=not env_on)

with st.sidebar.expander("Probabilistic"):
    n_mc = st.select_slider("Monte Carlo members", [1000, 2000, 5000, 10000, 20000], 10000)
    seed = st.number_input("MC seed", 0, 1_000_000, 0, 1)

try:
    cfg = AnalysisConfig(
        riser=RiserConfig(
            outer_diameter=od, wall_thickness=wt, water_depth=depth,
            hang_off_angle_deg=ang, scf=scf, sn_class=sn_class,
            contents_density=ref.contents_density, coating_thickness=ref.coating_thickness,
            coating_density=ref.coating_density, is_reference_preset=False,
        ),
        transfer=TransferConfig(route=route),
        environment=EnvironmentConfig(enabled=env_on, temperature_factor=tfac, salinity_factor=sfac),
        n_monte_carlo=int(n_mc), seed=int(seed),
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Invalid configuration: {exc}")
    st.stop()

is_synth = source.startswith("Synthetic")

# --------------------------------------------------------------------------- #
# Header + run control
# --------------------------------------------------------------------------- #
g = gates()
gates_ok = sum(x["passed"] for x in g)
head_l, head_r = st.columns([3, 2])
with head_l:
    st.markdown(
        '<div class="brand">'
        '<svg width="30" height="30" viewBox="0 0 22 22"><path d="M3 19 C 7 19, 8 6, 19 3" '
        'fill="none" stroke="#0f8f9c" stroke-width="1.7"/><circle cx="3" cy="19" r="2.2" fill="#b4791a"/>'
        '<circle cx="19" cy="3" r="1.7" fill="#0b7079"/></svg>'
        '<div><h1>SCR&middot;TWIN</h1><div class="sub">TDP Fatigue Integrity Console</div></div></div>',
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(
        f'<div style="text-align:right;margin-top:10px">'
        f'<span class="tag {"pass" if gates_ok == len(g) else "fail"}">{gates_ok}/{len(g)} gates</span>&nbsp;'
        f'<span class="tag syn">{"SYNTHETIC" if is_synth else "MEASURED"}</span></div>',
        unsafe_allow_html=True,
    )

run_col, _ = st.columns([1, 3])
run_clicked = run_col.button("▶  Run analysis", type="primary", width="stretch",
                             help="Runs the full chain with a live, animated acquisition + posterior.")

# --------------------------------------------------------------------------- #
# Compute (cached, deterministic)
# --------------------------------------------------------------------------- #
if is_synth:
    payload = analyze_synthetic(cfg.model_dump_json(), synth["hs"], synth["tp"], synth["gamma"],
                                synth["duration"], synth["fs"], int(synth["seed"]))
elif upload_bytes is not None:
    payload = analyze_upload(cfg.model_dump_json(), upload_bytes)
else:
    st.info("Upload an MRU CSV in the sidebar, or switch to the synthetic demo generator, then press Run analysis.")
    st.stop()

if "error" in payload:
    st.error(f"Data health check failed - {payload['error']}")
    if payload.get("health"):
        st.json(payload["health"])
    st.stop()


# --------------------------------------------------------------------------- #
# Live, interactive run - streams acquisition, then contracts the posterior
# --------------------------------------------------------------------------- #
def run_live(pl: dict) -> None:
    if is_synth:
        t, h, fsr = hires_heave(synth["hs"], synth["tp"], synth["gamma"], synth["duration"],
                                synth["fs"], int(synth["seed"]))
        t, h = np.array(t), np.array(h)
    else:
        t, h = np.array(pl["trace"]["time"]), np.array(pl["trace"]["heave"])
        fsr = pl["provenance"]["sample_rate_hz"]
    if h.size < 2:
        return
    ymax = float(np.max(np.abs(h))) * 1.12 or 1.0
    tmax = float(t[-1]) or 1.0

    holder = st.empty()
    c = holder.container()
    c.markdown('<div class="livebar"><span class="livedot"></span> Live &middot; real-time analysis of the acquired record</div>',
               unsafe_allow_html=True)
    prog = c.progress(0)
    status = c.empty()
    kbox = c.empty()
    chart = c.empty()

    # Stage 1 - acquisition (stream in every fluctuation)
    n = 26
    ups = np.unique(np.linspace(2, h.size, n).astype(int))
    for k, upto in enumerate(ups):
        fig = _fig(240)
        fig.add_scatter(x=t[:upto], y=h[:upto], line=dict(color=SIGNAL, width=1))
        fig.update_layout(xaxis=dict(title="t [s]", range=[0, tmax], gridcolor=GRID, zeroline=False),
                          yaxis=dict(title="heave [m]", range=[-ymax, ymax], gridcolor=GRID, zeroline=False))
        chart.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"aq{k}")
        seg = h[:upto]
        kbox.markdown(kpi_row([
            kpi("Samples acquired", f"{int(upto):,}", f"/ {h.size:,}"),
            kpi("Running Hm0", f"{4.0*float(np.std(seg)):.2f}", "m", "sig"),
            kpi("Peak heave", f"{float(np.max(np.abs(seg))):.2f}", "m", "amber"),
            kpi("Sample rate", f"{fsr:.0f}", "Hz"),
        ]), unsafe_allow_html=True)
        status.markdown(f'<div class="livestatus">&#9656; Acquiring MRU frames &mdash; window {k+1}/{len(ups)}, '
                        f'{int(upto):,} samples</div>', unsafe_allow_html=True)
        prog.progress(int(4 + 50 * (k + 1) / len(ups)))
        time.sleep(0.045)

    # Stage 2 - spectral
    status.markdown('<div class="livestatus">&#9656; Estimating sea-state PSD (Welch) &middot; fitting JONSWAP&hellip;</div>',
                    unsafe_allow_html=True)
    chart.plotly_chart(spectra_fig(pl["spectrum"]), width="stretch", config={"displayModeBar": False}, key="live_spec")
    prog.progress(66)
    time.sleep(0.5)

    # Stage 3 - fatigue (ease the life readout in)
    status.markdown('<div class="livestatus">&#9656; TDP stress reconstruction &middot; rainflow &rarr; S-N &rarr; Miner&hellip;</div>',
                    unsafe_allow_html=True)
    dlife = pl["damage"]["deterministic_life_years"]
    for k in range(8):
        shown = dlife * (0.55 + 0.45 * (k + 1) / 8)
        kbox.markdown(kpi_row([
            kpi("Annual damage", f'{pl["damage"]["annual_rate_time"]:.2e}', "/yr", "amber"),
            kpi("Deterministic life", life(shown), "yr", "sig"),
            kpi("S-N class", cfg.riser.sn_class),
        ]), unsafe_allow_html=True)
        prog.progress(66 + int(14 * (k + 1) / 8))
        time.sleep(0.05)

    # Stage 4 - posterior contraction (the signature)
    status.markdown('<div class="livestatus">&#9656; Monte Carlo (10k) &middot; Bayesian posterior contraction '
                    '(90% CI ~ 1/&radic;T)&hellip;</div>', unsafe_allow_html=True)
    fan, p50 = pl["bayesian_fan"], pl["posterior"]["p50"]
    ny = len(fan["years"])
    for j in range(2, ny + 1):
        chart.plotly_chart(fan_fig(fan, p50, upto=j), width="stretch",
                           config={"displayModeBar": False}, key=f"fan{j}")
        prog.progress(80 + int(20 * (j - 1) / (ny - 1)))
        time.sleep(0.06)

    status.markdown('<div class="livestatus" style="color:#1f8a5b">&#10003; Analysis complete</div>',
                    unsafe_allow_html=True)
    prog.progress(100)
    time.sleep(0.4)
    holder.empty()


if run_clicked:
    st.session_state.ran = True
    run_live(payload)

if not st.session_state.ran:
    st.markdown(
        '<div style="margin-top:26px;text-align:center;color:#56707d">'
        '<div style="font-size:15px;color:#1a2830">Ready.</div>'
        '<div style="font-size:12.5px;margin-top:4px">Set the sea state and riser configuration in the sidebar, '
        'then press <b>&#9654; Run analysis</b> for a live, animated run.</div></div>',
        unsafe_allow_html=True)
    st.stop()


# --------------------------------------------------------------------------- #
# Dashboard (static result; also live-updates as you edit the sidebar)
# --------------------------------------------------------------------------- #
dmg, post, insp, econ, prov = (
    payload["damage"], payload["posterior"], payload["inspection"], payload["economics"], payload["provenance"],
)
sea, env = payload["sea_state"], payload["environment"]

st.markdown(kpi_row([
    kpi("Deterministic life", life(dmg["deterministic_life_years"]), "yr", "sig"),
    kpi("P10 (conservative)", life(post["p10"]), "yr", "amber"),
    kpi("P50 median", life(post["p50"]), "yr"),
    kpi("P90", life(post["p90"]), "yr"),
    kpi("Next inspection", f'{insp["next_inspection_year"]:.1f}', "yr", "sig"),
    kpi("Env. capacity factor", f'{env["factor"]:.3f}' if env["enabled"] else "-", "", "amber"),
]), unsafe_allow_html=True)

st.markdown('<div class="sec">Sea state &middot; spectral analysis &middot; hang-off motion</div>', unsafe_allow_html=True)
st.markdown(kpi_row([
    kpi("Sig. heave Hm0", f'{sea["hs"]:.2f}', "m"),
    kpi("Tp", f'{sea["tp"]:.1f}', "s"),
    kpi("Tz", f'{sea["tz"]:.1f}', "s"),
    kpi("gamma fit", f'{sea["gamma"]:.1f}'),
]), unsafe_allow_html=True)
sc1, sc2 = st.columns([3, 2])
sc1.plotly_chart(spectra_fig(payload["spectrum"]), width="stretch", config={"displayModeBar": False})
tf = _fig(180)
tf.add_scatter(x=payload["trace"]["time"], y=payload["trace"]["heave"], line=dict(color=SIGNAL, width=1))
tf.update_layout(xaxis=dict(title="t [s]", gridcolor=GRID, zeroline=False),
                 yaxis=dict(title="heave [m]", gridcolor=GRID, zeroline=False))
sc2.plotly_chart(tf, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="sec">Layer 2 - rainflow &middot; S-N &middot; Miner</div>', unsafe_allow_html=True)
st.markdown(kpi_row([
    kpi("Annual damage (time)", f'{dmg["annual_rate_time"]:.2e}', "/yr", "amber"),
    kpi("Spectral (Dirlik)", f'{dmg["annual_rate_spectral"]:.2e}', "/yr"),
    kpi("Block damage", f'{dmg["block_damage"]:.2e}'),
    kpi("S-N class", cfg.riser.sn_class),
]), unsafe_allow_html=True)

st.markdown(
    f'<div class="sec">Layer 3 - remaining-life posterior &middot; {post["n_members"]:,} MC members &middot; '
    'Bayesian contraction (90% CI ~ 1/&radic;T)</div>', unsafe_allow_html=True)
pc1, pc2 = st.columns([3, 2])
pc1.plotly_chart(fan_fig(payload["bayesian_fan"], post["p50"]), width="stretch", config={"displayModeBar": False})
pc2.plotly_chart(pdf_hist_fig(post), width="stretch", config={"displayModeBar": False})

st.markdown('<div class="sec">Decision - risk-based inspection &middot; fleet economics</div>', unsafe_allow_html=True)
dc1, dc2 = st.columns([3, 2])
dc1.plotly_chart(pof_fig(insp), width="stretch", config={"displayModeBar": False})
with dc2:
    st.markdown(kpi_row([
        kpi("Fleet saving (20u, 20yr)",
            f'${econ["fleet_saving_low_usd"]/1e6:.1f}-{econ["fleet_saving_high_usd"]/1e6:.1f}M', "", "sig"),
        kpi("Sensor payback", f'{econ["payback_low_yr"]:.1f}-{econ["payback_high_yr"]:.1f}', "yr"),
    ]), unsafe_allow_html=True)
    st.markdown(kpi_row([
        kpi("Target PoF", f'{insp["target_pof"]*100:.1f}', "%"),
        kpi("PoF at inspection", f'{insp["pof_at_next"]*100:.2f}', "%",
            "alarm" if insp["pof_at_next"] > insp["target_pof"] * 1.05 else ""),
    ]), unsafe_allow_html=True)

vc1, vc2 = st.columns([3, 2])
with vc1:
    st.markdown('<div class="sec">Validation gates (spec section 5)</div>', unsafe_allow_html=True)
    for x in g:
        dot = GOOD if x["passed"] else ALARM
        st.markdown(
            f'<div class="gate"><span class="dot" style="background:{dot}"></span>'
            f'<span>{x["name"]}</span><span class="actual">{x["actual"]}</span></div>',
            unsafe_allow_html=True,
        )
with vc2:
    st.markdown('<div class="sec">Reports & provenance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="foot">core v{prov["core_version"]} &middot; numpy {prov["numpy_version"]} &middot; '
        f'scipy {prov["scipy_version"]}<br>seed {prov["seed"]} &middot; '
        f'H(f) {"reduced-order" if prov["transfer_is_reduced_order"] else "reference"} &middot; '
        f'cfg {prov["config_sha256"][:12]}<br>'
        f'{prov["n_samples"]:,} samples @ {prov["sample_rate_hz"]:.1f} Hz</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    pdf_bytes = build_pdf(cfg.model_dump_json(), "synthetic" if is_synth else "upload", json.dumps(payload))
    st.download_button("⬇  Download PDF report", pdf_bytes,
                       file_name="scr-twin-integrity-report.pdf", mime="application/pdf", width="stretch")
    bundle = {
        "config": json.loads(cfg.model_dump_json()),
        "source": payload.get("source", {"kind": "synthetic" if is_synth else "upload"}),
        "provenance": prov,
        "summary": {"sea_state": sea, "damage": dmg,
                    "posterior": {k: post[k] for k in ("p10", "p50", "p90", "n_members")},
                    "inspection": insp["next_inspection_year"], "economics": econ},
    }
    st.download_button("⬇  Export provenance bundle (JSON)", json.dumps(bundle, indent=2),
                       file_name="scr-twin-provenance.json", mime="application/json", width="stretch")

st.markdown(
    '<div class="foot" style="margin-top:16px;text-align:center">Physics-based digital twin &middot; '
    'DNV-RP-C203 &middot; ASTM E1049 &middot; Dirlik / Tovo-Benasciutti &middot; reduced-order Morison H(f) &middot; '
    'reference SCR preset is illustrative, not project data</div>', unsafe_allow_html=True)
