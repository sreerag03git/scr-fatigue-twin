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
import streamlit.components.v1 as components  # noqa: E402
from fpdf import FPDF  # noqa: E402

from scr_twin_core import ingest as ingest_mod  # noqa: E402
from scr_twin_core import validation as validation_mod  # noqa: E402
from scr_twin_core.config import (  # noqa: E402
    AnalysisConfig,
    EnvironmentConfig,
    HangOffConfig,
    RiserConfig,
    TransferConfig,
    VivConfig,
)
from scr_twin_core.sn import (  # noqa: E402
    DNV_C203_IN_AIR,
    MeanStressModel,
    SNEnvironment,
    cycles_to_failure,
    get_curve,
)
from scr_twin_core.spectral import jonswap  # noqa: E402
from scr_twin_core.spectral_damage import dirlik_range_pdf  # noqa: E402
from server import service  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette / theme
# --------------------------------------------------------------------------- #
SIGNAL, SIGNAL2, AMBER, ALARM = "#16a6ac", "#0e7c82", "#b07d1a", "#c0523f"
GRID, TEXT, TEXTHI, PAPER = "#eaeeef", "#586a71", "#17242b", "rgba(0,0,0,0)"
GOOD = "#2f855a"
SN_CLASSES = ["B1", "B2", "C", "C1", "C2", "D", "E", "F", "F1", "F3", "G"]

# Mobile mode is read *before* the first render command so the page layout and
# sidebar state can flip between desktop (wide, multi-column) and mobile
# (centered, single-column, collapsed sidebar).
st.session_state.setdefault("mobile", False)
MOBILE = bool(st.session_state["mobile"])

st.set_page_config(
    page_title="SCR-Twin - TDP Fatigue Integrity Console",
    page_icon=("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
               "<rect width='32' height='32' rx='7' fill='%230e7c82'/>"
               "<path d='M6 10 Q16 26 26 10' fill='none' stroke='white' stroke-width='2.4' "
               "stroke-linecap='round'/></svg>"),
    layout="centered" if MOBILE else "wide",
    initial_sidebar_state="collapsed" if MOBILE else "expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      :root {
        --bg:#f4f6f6; --panel:#ffffff; --panel2:#eef2f2; --ink:#17242b; --sub:#586a71;
        --muted:#93a3a9; --line:#e7ecec; --line2:#d6dedf; --accent:#0e7c82; --accent2:#16a6ac;
        --amber:#b07d1a; --alarm:#c0523f; --good:#2f855a; --wash:rgba(14,124,130,0.055);
        --mono:'IBM Plex Mono','JetBrains Mono',Consolas,monospace;
        --sans:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
        --shadow:0 1px 2px rgba(23,36,43,0.04), 0 6px 22px rgba(23,36,43,0.05);
        --shadow-sm:0 1px 2px rgba(23,36,43,0.05);
        --r:9px;
      }
      html, body, .stApp, [class*="css"], p, span, div, label { font-family:var(--sans); }
      .stApp { color:var(--ink); background-color:var(--bg); }
      [data-testid="stHeader"] { background:transparent; }
      [data-testid="stMainBlockContainer"], .block-container { max-width:1180px; padding-top:2.2rem; }
      h1,h2,h3,h4 { letter-spacing:-.01em; color:var(--ink); font-weight:600; }
      .mono, code { font-family:var(--mono) !important; }
      [data-testid="stMetricValue"] { font-family:var(--sans) !important; font-variant-numeric:tabular-nums; }
      ::-webkit-scrollbar { width:10px; height:10px; }
      ::-webkit-scrollbar-thumb { background:#cdd6d7; border-radius:6px; }
      ::-webkit-scrollbar-thumb:hover { background:#b9c4c6; }
      ::-webkit-scrollbar-track { background:transparent; }

      /* ---- Masthead / title block ---- */
      .titleblock { display:grid; grid-template-columns:2.3fr repeat(4,1fr); border:1px solid var(--line);
        background:var(--panel); margin:0 0 14px; border-radius:var(--r); box-shadow:var(--shadow); overflow:hidden; }
      .titleblock > div { border-left:1px solid var(--line); padding:15px 18px; position:relative; }
      .titleblock > div:first-child { border-left:none; display:flex; flex-direction:column; justify-content:center; }
      .tb-name { font-size:21px; font-weight:700; letter-spacing:-.02em; color:var(--ink); }
      .tb-sub { font-size:9.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); margin-top:5px; font-weight:600; }
      .tb-k { font-size:9.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:500; }
      .tb-v { font-size:16px; font-weight:600; color:var(--ink); margin-top:5px; font-variant-numeric:tabular-nums; }
      .tb-v.pass { color:var(--good); } .tb-v.fail { color:var(--alarm); } .tb-v.sig { color:var(--accent); }

      /* ---- KPI cards ---- */
      .kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:12px; margin:10px 0 8px; }
      .kpi { background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:16px 17px 15px;
        box-shadow:var(--shadow-sm); transition:box-shadow .16s ease; }
      .kpi:hover { box-shadow:var(--shadow); }
      .kpi .lab { font-size:10.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); font-weight:500; }
      .kpi .val { font-size:28px; color:var(--ink); line-height:1.05; font-variant-numeric:tabular-nums;
        margin-top:9px; font-weight:600; letter-spacing:-.02em; }
      .kpi .val small { font-size:12.5px; color:var(--muted); margin-left:4px; font-weight:500; }
      .kpi .val.sig { color:var(--accent); } .kpi .val.amber { color:var(--amber); } .kpi .val.alarm { color:var(--alarm); }

      /* ---- tags ---- */
      .tag { display:inline-block; font-size:10.5px; font-weight:500; letter-spacing:.03em; padding:3px 10px;
        border-radius:100px; border:1px solid var(--line2); color:var(--sub); background:var(--panel); }
      .tag.syn,.tag.amber { color:var(--amber); border-color:color-mix(in srgb,var(--amber) 40%,transparent);
        background:color-mix(in srgb,var(--amber) 8%,transparent); }
      .tag.pass { color:var(--good); border-color:color-mix(in srgb,var(--good) 40%,transparent);
        background:color-mix(in srgb,var(--good) 8%,transparent); }
      .tag.fail { color:var(--alarm); border-color:color-mix(in srgb,var(--alarm) 40%,transparent);
        background:color-mix(in srgb,var(--alarm) 8%,transparent); }

      /* ---- section headers ---- */
      .sec { display:flex; align-items:center; gap:11px; margin:26px 0 13px;
        font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--sub); font-weight:600; }
      .sec[data-n]::before { content:attr(data-n); font-family:var(--mono); font-size:10.5px; color:var(--accent);
        letter-spacing:.02em; flex:none; font-weight:500; }
      .sec::after { content:""; flex:1; height:1px; background:var(--line); }

      /* ---- equation block ---- */
      .eq { background:var(--panel2); border:1px solid var(--line); border-left:2.5px solid var(--accent);
        padding:12px 15px; margin:10px 0 12px; border-radius:6px; font-family:var(--mono); font-size:13px;
        color:var(--ink); overflow-x:auto; }
      .eq .c { color:var(--muted); }

      /* ---- panels & notes ---- */
      .panel { background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:15px 17px; box-shadow:var(--shadow-sm); }
      .note { font-size:12.5px; color:var(--sub); line-height:1.6; }

      /* ---- gate rows ---- */
      .gate { display:flex; align-items:center; gap:11px; padding:9px 2px; border-bottom:1px solid var(--line); font-size:13px; }
      .gate .dot { width:8px; height:8px; border-radius:50%; flex:none; }
      .gate .actual { margin-left:auto; font-family:var(--mono); font-size:11.5px; color:var(--sub); }
      .foot { color:var(--muted); font-size:11px; letter-spacing:.01em; line-height:1.65; }

      /* ---- data tables (ledger / generic) ---- */
      table.ledger { width:100%; border-collapse:collapse; font-size:13px; }
      table.ledger th { text-align:left; font-size:10px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
        color:var(--muted); border-bottom:1px solid var(--line2); padding:9px 12px; }
      table.ledger td { padding:9px 12px; border-bottom:1px solid var(--line); color:var(--ink); vertical-align:top; }
      table.ledger td.lv { font-family:var(--mono); color:var(--accent); font-variant-numeric:tabular-nums; }
      table.ledger td.lb { color:var(--muted); font-size:11.5px; }
      table.ledger tr:last-child td { border-bottom:none; }
      table.ledger tr:hover td { background:var(--panel2); }

      /* ---- tabs ---- */
      [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--line); background:transparent; }
      [data-baseweb="tab"] { font-family:var(--sans) !important; font-size:13.5px !important; font-weight:500;
        letter-spacing:0; color:var(--muted); padding:11px 16px !important; }
      [data-baseweb="tab"]:hover { color:var(--ink); }
      [data-baseweb="tab"][aria-selected="true"] { color:var(--accent); font-weight:600; }
      [data-baseweb="tab-highlight"] { background:var(--accent) !important; height:2px; border-radius:2px; }

      /* ---- buttons ---- */
      .stButton button, .stDownloadButton button { border-radius:7px !important; font-family:var(--sans) !important;
        font-weight:500; letter-spacing:0; border:1px solid var(--line2) !important; color:var(--ink) !important;
        transition:all .15s ease; }
      .stButton button:hover, .stDownloadButton button:hover { border-color:var(--accent) !important; color:var(--accent) !important; }
      .stButton button[kind="primary"] { background:var(--accent) !important; border-color:var(--accent) !important;
        color:#fff !important; font-weight:600; box-shadow:0 2px 12px rgba(14,124,130,0.22) !important; }
      .stButton button[kind="primary"]:hover { background:#0a666b !important; border-color:#0a666b !important; color:#fff !important; }

      /* ---- sidebar control panel ---- */
      section[data-testid="stSidebar"] { background:#eef2f2; border-right:1px solid var(--line); }
      section[data-testid="stSidebar"] [data-testid="stExpander"] { border:1px solid var(--line); background:var(--panel);
        border-radius:8px; box-shadow:var(--shadow-sm); }
      section[data-testid="stSidebar"] summary { font-family:var(--sans) !important; font-size:12.5px !important;
        font-weight:600; letter-spacing:.01em; color:var(--ink) !important; text-transform:none; }
      section[data-testid="stSidebar"] label { font-size:12.5px !important; color:var(--sub) !important; }
      [data-testid="stWidgetLabel"] p { font-size:12.5px !important; }
      [data-baseweb="input"] input, [data-baseweb="select"] > div, .stNumberInput input { border-radius:6px !important; }

      /* ---- landing ---- */
      .land { max-width:900px; margin:2vh auto 0; text-align:center; }
      .land h1 { font-family:var(--sans); font-size:52px; font-weight:700; letter-spacing:-.025em; margin:16px 0 4px; color:var(--ink); }
      .land .tagline { font-size:11.5px; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); }
      .land .lede { color:var(--sub); font-size:16px; line-height:1.7; max-width:640px; margin:22px auto 4px; }
      .chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:22px 0 6px; }
      .chip { font-size:11.5px; font-weight:500; color:var(--sub); background:var(--panel); border:1px solid var(--line2);
        border-radius:100px; padding:5px 14px; }
      .flow { display:flex; flex-wrap:wrap; gap:12px; justify-content:center; margin:26px 0 8px; }
      .flowcard { background:var(--panel); border:1px solid var(--line); border-radius:var(--r); padding:16px 18px;
        width:168px; text-align:left; box-shadow:var(--shadow-sm); }
      .flowcard .n { font-family:var(--mono); font-size:11px; color:var(--accent); letter-spacing:.06em; font-weight:500; }
      .flowcard .t { font-size:13.5px; color:var(--ink); margin-top:7px; font-weight:600; }
      .flowcard .d { font-size:11.5px; color:var(--muted); margin-top:4px; line-height:1.5; }
      .livebar { display:flex; align-items:center; gap:9px; font-size:12px; font-weight:600;
        letter-spacing:.08em; color:var(--accent); text-transform:uppercase; margin:8px 0 10px; }
      .livedot { width:8px; height:8px; border-radius:50%; background:var(--accent);
        box-shadow:0 0 0 0 rgba(14,124,130,.4); animation:pulse 1.6s infinite; }
      @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(14,124,130,.4);} 70%{box-shadow:0 0 0 7px rgba(14,124,130,0);} 100%{box-shadow:0 0 0 0 rgba(14,124,130,0);} }
      @media (prefers-reduced-motion: reduce) { .livedot { animation:none; } }
      .livestatus { font-size:13px; color:var(--sub); margin:2px 0 6px; }
      .st-key-view_toggle { position:fixed !important; bottom:22px; right:22px; width:auto !important; z-index:1000; margin:0 !important; }
      .st-key-view_toggle button { border-radius:100px !important; padding:10px 20px !important;
        background:var(--ink) !important; color:#fff !important; border:none !important;
        font-weight:500 !important; box-shadow:0 6px 22px rgba(23,36,43,0.22) !important; min-height:0 !important; }
      .st-key-view_toggle button:hover { background:var(--accent) !important; color:#fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Mobile-mode responsive overrides (single column, tap-friendly, compact). On a
# real phone this just fills the screen; on a wide screen the media query below
# wraps the content in a centred phone-width device frame so desktop viewers see
# the handset experience rather than a stretched single column.
if MOBILE:
    st.markdown(
        """
        <style>
          .block-container { padding: 0.8rem 0.7rem 3rem !important; max-width: 100% !important; }
          .land { margin-top: 0; } .land h1 { font-size: 30px; letter-spacing:.1em; }
          .land .tagline { font-size: 11px; } .land .lede { font-size: 13.5px; }
          .flowcard { width: 100% !important; }
          .kpi-row { grid-template-columns: repeat(2, 1fr) !important; gap: 8px; }
          .kpi .val { font-size: 19px; } .kpi { padding: 9px 11px; }
          .brand h1 { font-size: 22px; } .sec { font-size: 10.5px; }
          .stButton button { min-height: 46px; font-size: 15px; }
          section[data-testid="stSidebar"] { min-width: 84vw !important; }

          /* Desktop viewers: render the mobile layout inside a phone chassis. */
          @media (min-width: 720px) {
            [data-testid="stMainBlockContainer"], .block-container {
              max-width: 430px !important;
              margin: 26px auto 46px !important;
              padding: 18px 17px 40px !important;
              background: #ffffff !important;
              border: 1px solid #d6dedf !important;
              border-radius: 40px !important;
              box-shadow: 0 0 0 11px #eef2f2, 0 26px 62px rgba(23,36,43,0.16) !important;
              min-height: 80vh !important;
            }
            /* speaker pill, so the frame reads as a handset */
            [data-testid="stMainBlockContainer"]::before, .block-container::before {
              content: ""; display: block; width: 46px; height: 5px; border-radius: 3px;
              background: #d6dedf; margin: 0 auto 14px !important;
            }
            /* keep the drawer phone-sized instead of 84vw of the desktop */
            section[data-testid="stSidebar"] { min-width: 360px !important; width: 360px !important; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# View toggle (mobile / desktop) - available on every page, top-right.
_tcols = st.columns([1, 1]) if MOBILE else st.columns([5, 1])
with _tcols[-1]:
    if st.button("Desktop view" if MOBILE else "Mobile view", key="view_toggle",
                 width="stretch",
                 help="Reflow the interface for phones and tablets: single column, collapsed sidebar."):
        st.session_state["mobile"] = not MOBILE
        st.rerun()


def dcols(spec: list[int]) -> list:
    """Streamlit columns on desktop; stacked full-width containers on mobile."""
    return [st.container() for _ in spec] if MOBILE else list(st.columns(spec))


# Session flow: landing -> console; ran gates the dashboard.
st.session_state.setdefault("launched", False)
st.session_state.setdefault("ran", False)


# --------------------------------------------------------------------------- #
# Cached compute (keyed on serialisable inputs -> efficient re-runs)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def gates() -> list[dict]:
    return [g.as_dict() for g in validation_mod.run_all_gates(seed=0)]


def _imported_tf(cfg: AnalysisConfig, transfer_bytes: bytes | None):
    """Parse the imported H(f) when the route needs it. Returns (tf, error)."""
    if cfg.transfer.route != "imported":
        return None, None
    if not transfer_bytes:
        return None, "Transfer route 'imported' selected but no validated H(f) table uploaded."
    try:
        return service.load_transfer(transfer_bytes), None
    except ValueError as exc:
        return None, f"Invalid H(f) table - {exc}"


def _scatter_diagram(scatter_bytes: bytes | None):
    """Parse an uploaded wave scatter-diagram CSV, or None to use the illustrative default."""
    if not scatter_bytes:
        return None, None
    try:
        return service.load_scatter(scatter_bytes), None
    except ValueError as exc:
        return None, f"Invalid scatter-diagram CSV - {exc}"


def _vessel_rao(rao_bytes: bytes | None):
    """Parse an uploaded vessel-RAO CSV, or None to use the illustrative generator."""
    if not rao_bytes:
        return None, None
    try:
        return service.load_rao(rao_bytes), None
    except ValueError as exc:
        return None, f"Invalid vessel-RAO CSV - {exc}"


@st.cache_data(show_spinner=False)
def analyze_synthetic(config_json: str, hs: float, tp: float, gamma: float,
                      duration: float, fs: float, seed: int, heading: float,
                      transfer_bytes: bytes | None = None,
                      scatter_bytes: bytes | None = None,
                      rao_bytes: bytes | None = None) -> dict:
    cfg = AnalysisConfig.model_validate_json(config_json)
    tf, err = _imported_tf(cfg, transfer_bytes)
    if err:
        return {"error": err}
    diagram, serr = _scatter_diagram(scatter_bytes)
    if serr:
        return {"error": serr}
    vessel_rao, rerr = _vessel_rao(rao_bytes)
    if rerr:
        return {"error": rerr}
    if vessel_rao is not None:
        channels, fsr = service.make_motion_from_rao(vessel_rao, hs, tp, gamma, duration, fs, seed)
        _p = vessel_rao.provenance.as_dict()
        motion_prov = {"source": f"validated RAO ({_p['source_tool']})", "is_validated": True,
                       "provenance": _p}
    else:
        channels, fsr = service.make_synthetic_6dof(hs, tp, gamma, duration, fs, seed, heading)
        motion_prov = {"source": "synthetic (illustrative RAO)", "is_validated": False}
    return service.analyze(cfg, channels["heave"], fsr, is_synthetic=True,
                           imported_tf=tf, channels=channels, scatter_diagram=diagram,
                           motion_provenance=motion_prov)


@st.cache_data(show_spinner=False)
def analyze_upload(config_json: str, file_bytes: bytes,
                   transfer_bytes: bytes | None = None,
                   scatter_bytes: bytes | None = None) -> dict:
    cfg = AnalysisConfig.model_validate_json(config_json)
    tf, err = _imported_tf(cfg, transfer_bytes)
    if err:
        return {"error": err}
    diagram, serr = _scatter_diagram(scatter_bytes)
    if serr:
        return {"error": serr}
    rec = ingest_mod.load_mru_csv(io.BytesIO(file_bytes))
    if not rec.health.ok:
        return {"error": "; ".join(rec.health.flags) or "data health check failed",
                "health": rec.health.as_dict()}
    # Uploaded multi-DOF records drive the full Eq.6 resolution; heave-only files
    # collapse to the heave path (dof channels dict has a single entry).
    return service.analyze(cfg, rec.channels["heave"], rec.fs,
                           is_synthetic=False, data_health=rec.health.as_dict(),
                           imported_tf=tf, channels=dict(rec.channels), scatter_diagram=diagram)


@st.cache_data(show_spinner=False)
def hires_heave(hs: float, tp: float, gamma: float, duration: float, fs: float, seed: int,
                target: int = 1800) -> tuple[list[float], list[float], float]:
    """High-resolution synthetic heave (decimated for smooth animation)."""
    heave, fsr = service.make_synthetic(hs, tp, gamma, duration, fs, seed)
    step = max(1, heave.size // target)
    h = heave[::step]
    t = np.arange(h.size) * step / fsr
    return [float(v) for v in t], [float(v) for v in h], float(fsr)


@st.cache_data(show_spinner=False)
def viv_knockdown(config_json: str) -> dict:
    """Live VIV-life sweep vs marine-growth thickness for the current riser/current.

    Keyed on a config whose marine-growth thickness is canonicalised to zero, so
    the sweep is cached across slider moves (only the current-thickness marker
    moves) and recomputes only when the riser or current actually changes.
    """
    cfg = AnalysisConfig.model_validate_json(config_json)
    grid = [t / 1000.0 for t in range(0, 151, 10)]
    return service.viv_life_sweep(cfg, grid)


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
        height=height, margin=dict(l=58, r=20, t=16, b=44),
        paper_bgcolor=PAPER, plot_bgcolor="#ffffff", showlegend=False,
        font=dict(color=TEXT, family="Inter, Segoe UI, system-ui, sans-serif", size=11),
        hoverlabel=dict(font_family="Inter, system-ui, sans-serif", bgcolor="#ffffff",
                        bordercolor="#dfe6e7", font_size=11),
        xaxis=dict(linecolor="#dfe6e7", ticks="outside", tickcolor="#dfe6e7", ticklen=3,
                   tickfont=dict(size=10), title_font=dict(size=11, color=TEXTHI)),
        yaxis=dict(linecolor="#dfe6e7", ticks="outside", tickcolor="#dfe6e7", ticklen=3,
                   tickfont=dict(size=10), title_font=dict(size=11, color=TEXTHI)),
    )
    return f


def spectra_fig(spec: dict) -> go.Figure:
    f = _fig(250)
    # Floor the PSDs before the log axes: a genuine zero in the spectrum maps to
    # log(0) = -inf, which Plotly then tries to place a <text> label at and throws
    # "<text> attribute y: -Infinity" for. Same 1e-12 floor the PDF's matplotlib
    # spectra already uses, so the two renderings agree.
    motion = np.clip(spec["motion_psd"], 1e-12, None)
    stress = np.clip(spec["stress_psd"], 1e-12, None)
    f.add_scatter(x=spec["freq"], y=motion, line=dict(color=SIGNAL, width=1.8), yaxis="y")
    f.add_scatter(x=spec["freq"], y=stress, line=dict(color=AMBER, width=1.8), yaxis="y2")
    f.update_layout(
        xaxis=dict(title="Frequency [Hz]", gridcolor=GRID, zeroline=False, range=[0, 0.4]),
        yaxis=dict(title="motion [m^2/Hz]", type="log", gridcolor=GRID, color=SIGNAL, zeroline=False),
        yaxis2=dict(title="stress [MPa^2/Hz]", type="log", overlaying="y", side="right", color=AMBER, showgrid=False),
    )
    return f


def transfer_fig(tf: dict) -> go.Figure:
    """Layer-1 |H(f)| stress transfer (Fig 4): MPa of TDP hot-spot stress per m heave."""
    f = _fig(250)
    f.add_scatter(x=tf["freq"], y=tf["stress_mag"], line=dict(color=SIGNAL2, width=2.2),
                  fill="tozeroy", fillcolor="rgba(15,143,156,0.08)", name="|H|")
    # Highlight the wave-frequency band the paper validates over (0.05-0.30 Hz).
    f.add_vrect(x0=0.05, x1=0.30, fillcolor="rgba(180,121,26,0.06)", line_width=0)
    f.update_layout(xaxis=dict(title="Frequency [Hz]", gridcolor=GRID, zeroline=False, range=[0, 0.4]),
                    yaxis=dict(title="|H| [MPa per m heave]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def dof_fig(contrib: dict) -> go.Figure:
    """Per-DOF share of the vertical hang-off motion variance (which DOF drives fatigue)."""
    items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    names = [k.upper() for k, _ in items]
    vals = [100.0 * v for _, v in items]
    colors = [SIGNAL2 if k == "heave" else AMBER if k in ("pitch", "roll") else "#8aa0ad"
              for k, _ in items]
    f = _fig(200)
    f.add_bar(y=names, x=vals, orientation="h", marker_color=colors,
              text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False)
    f.update_layout(
        xaxis=dict(title="% of hang-off motion variance", gridcolor=GRID, zeroline=False,
                   range=[0, (max(vals) * 1.28) if vals else 1.0]),
        yaxis=dict(autorange="reversed"), margin=dict(l=64, r=30, t=10, b=40))
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


def econ_fig(econ: dict) -> go.Figure:
    """Fig 11(b): conditional net fleet value dC vs phi, with a P5-P95 cost band.

    The sensor only pays where the curve is above zero - to the right of the
    break-even phi*. The fleet's own (endogenous) phi is marked on the x-axis.
    """
    f = _fig(250)
    phi = econ["phi_grid"]
    p5 = [v / 1e6 for v in econ["fleet_delta_c_p5_usd"]]
    p50 = [v / 1e6 for v in econ["fleet_delta_c_p50_usd"]]
    p95 = [v / 1e6 for v in econ["fleet_delta_c_p95_usd"]]
    f.add_scatter(x=phi, y=p95, line=dict(width=0), hoverinfo="skip")
    f.add_scatter(x=phi, y=p5, fill="tonexty", line=dict(width=0),
                  fillcolor="rgba(15,143,156,0.13)", hoverinfo="skip")
    f.add_scatter(x=phi, y=p50, line=dict(color=SIGNAL2, width=2.4), name="net dC")
    f.add_hline(y=0.0, line=dict(color=TEXT, width=0.9, dash="dot"))
    be = econ.get("breakeven_phi")
    if be is not None and np.isfinite(be):
        f.add_vline(x=be, line=dict(color=ALARM, width=1.2, dash="dash"),
                    annotation_text=f"break-even φ*={be:.2f}", annotation_font_color=ALARM,
                    annotation_font_size=9, annotation_position="top left")
    op = econ.get("phi")
    if op is not None:
        col = GOOD if econ.get("net_positive") else AMBER
        f.add_vline(x=op, line=dict(color=col, width=1.6),
                    annotation_text=f"fleet φ={op:.2f}", annotation_font_color=col,
                    annotation_font_size=9, annotation_position="bottom right")
    f.update_layout(
        xaxis=dict(title="φ = P(asset ages slower than design)", gridcolor=GRID,
                   zeroline=False, range=[0, 1]),
        yaxis=dict(title="Net fleet value ΔC [US$M, 20yr]", gridcolor=GRID, zeroline=False))
    return f


def catenary_fig(cat: dict) -> go.Figure:
    """Static catenary profile: riser shape from the TDP (origin) to the hang-off."""
    f = _fig(260)
    _wd = cat["water_depth"]
    f.add_hline(y=_wd, line=dict(color=SIGNAL, width=1.0, dash="dot"),
                annotation_text="MWL", annotation_font_size=9, annotation_font_color=SIGNAL,
                annotation_position="top left")
    f.add_hline(y=0.0, line=dict(color=AMBER, width=1.2),
                annotation_text="seabed / TDP", annotation_font_size=9, annotation_font_color=AMBER,
                annotation_position="bottom right")
    f.add_scatter(x=cat["x"], y=cat["y"], line=dict(color=SIGNAL2, width=2.4),
                  fill="tozeroy", fillcolor="rgba(15,143,156,0.06)", name="riser")
    f.add_scatter(x=[0.0], y=[0.0], mode="markers+text", text=["TDP"], textposition="top right",
                  marker=dict(color=AMBER, size=9), textfont=dict(color=AMBER, size=10))
    f.add_scatter(x=[cat["horizontal_span"]], y=[cat["water_depth"]], mode="markers+text",
                  text=["hang-off"], textposition="bottom left",
                  marker=dict(color=SIGNAL2, size=9), textfont=dict(color=SIGNAL2, size=10))
    f.update_layout(
        xaxis=dict(title="horizontal offset from TDP [m]", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="height above TDP [m]", gridcolor=GRID, zeroline=False,
                   scaleanchor="x", scaleratio=1.0))
    return f


def sn_family_fig(sn_class: str, environment: SNEnvironment) -> go.Figure:
    """DNV-RP-C203 S-N curve family (log-log), the selected class highlighted."""
    f = _fig(250)
    s_mpa = np.logspace(np.log10(8.0), np.log10(1200.0), 240)
    for name in DNV_C203_IN_AIR:
        n = cycles_to_failure(s_mpa * 1e6, get_curve(name, environment))
        sel = name == sn_class
        f.add_scatter(
            x=n, y=s_mpa, mode="lines",
            line=dict(color=SIGNAL if sel else "#cbd5d6", width=2.6 if sel else 1.0),
            name=name, hoverinfo="name" if not sel else "x+y+name",
        )
    f.update_layout(
        xaxis=dict(title="Cycles to failure N", type="log", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Stress range [MPa]", type="log", gridcolor=GRID, zeroline=False),
        annotations=[dict(x=0.03, y=0.06, xref="paper", yref="paper",
                          text=f"DNV {sn_class} - {'seawater/CP' if environment==SNEnvironment.SEAWATER_CP else 'in air'}",
                          showarrow=False, font=dict(color=SIGNAL, size=10))])
    return f


def dirlik_verify_fig(ver: dict) -> go.Figure:
    """Rainflow range histogram vs the Dirlik and narrow-band (Rayleigh) PDFs."""
    f = _fig(250)
    edges = np.asarray(ver["hist_edges_mpa"], dtype=float)
    counts = np.asarray(ver["hist_counts"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    total = counts.sum()
    density = counts / (total * widths) if total > 0 and np.all(widths > 0) else counts * 0.0
    f.add_bar(x=centres, y=density, marker_color="rgba(15,143,156,0.30)",
              marker_line_color=SIGNAL, marker_line_width=0.3, name="rainflow")
    moments = {int(k): float(v) for k, v in ver["moments"].items()}
    s = np.linspace(max(edges[1] * 0.05, 1e-3), edges[-1], 300)
    if moments.get(0, 0.0) > 0.0 and moments.get(2, 0.0) > 0.0 and moments.get(4, 0.0) > 0.0:
        dk = dirlik_range_pdf(s, moments, stress_to_mpa=ver["stress_to_mpa"])
        f.add_scatter(x=s, y=dk, line=dict(color=SIGNAL2, width=2.2), name="Dirlik")
        sigma = ver["sigma_mpa"]
        if sigma > 0.0:
            rayleigh = (s / (4.0 * sigma**2)) * np.exp(-(s**2) / (8.0 * sigma**2))
            f.add_scatter(x=s, y=rayleigh, line=dict(color=AMBER, width=1.6, dash="dash"),
                          name="narrow-band")
    f.update_layout(
        showlegend=True, legend=dict(font=dict(size=9), orientation="h", y=1.02, x=0.55),
        xaxis=dict(title="Stress range [MPa]", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="probability density", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def divergence_fan_fig(dfan: dict) -> go.Figure:
    """Accumulated-damage divergence (actual/design - 1) vs year, in percent."""
    f = _fig(250)
    yrs = dfan["years"]
    p10 = [100.0 * v for v in dfan["p10"]]
    p50 = [100.0 * v for v in dfan["p50"]]
    p90 = [100.0 * v for v in dfan["p90"]]
    f.add_scatter(x=yrs, y=p90, line=dict(width=0), hoverinfo="skip")
    f.add_scatter(x=yrs, y=p10, fill="tonexty", line=dict(width=0),
                  fillcolor="rgba(180,121,26,0.15)", hoverinfo="skip")
    f.add_scatter(x=yrs, y=p50, line=dict(color=SIGNAL2, width=2.4), name="P50")
    f.add_vline(x=15, line=dict(color=TEXT, width=0.8, dash="dot"),
                annotation_text="spec gate @yr15: P10≈5% / P90≈28%",
                annotation_font_size=9, annotation_font_color=TEXT, annotation_position="top left")
    f.update_layout(
        xaxis=dict(title="year", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="actual/design accumulated damage - 1 [%]", gridcolor=GRID, zeroline=False))
    return f


def architecture_svg() -> str:
    """Drafting-style signal-flow block diagram of the twin's processing chain."""
    INK, TEAL, TEAL2, AMBER, DIM, LINE = (
        "#182530", "#0e7c82", "#16a6ac", "#a86f16", "#8b9aa0", "#e7ecec")
    stages = [
        ("01", "SENSING", "MRU 6-DOF motion", "Eq.6 hang-off"),
        ("02", "TRANSFER", "H(f): MRU &rarr; TDP", "Morison / import"),
        ("03", "STRESS", "TDP hot-spot &#963;", "SCF &#183; M/Z"),
        ("04", "DETECTION", "rainflow &#183; S-N &#183; Miner", "DNV-RP-C203"),
        ("05", "POSTERIOR", "Monte-Carlo life", "10k members"),
        ("06", "ASSIMILATION", "Bayesian update", "AR(1) n_eff"),
        ("07", "DECISION", "RBI &#183; economics", "Eq.11 &#916;C"),
    ]
    W, H = 1200, 320
    bw, bh, gap, x0, ymid = 148, 66, 15, 30, 176
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         "font-family='Inter, Segoe UI, sans-serif'>",
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<rect x="8" y="8" width="{W-16}" height="{H-16}" fill="none" stroke="{INK}" stroke-width="0.7"/>',
         '<defs><marker id="af" markerWidth="10" markerHeight="9" refX="7.5" refY="3" orient="auto">'
         f'<path d="M0,0 L8,3 L0,6 Z" fill="{DIM}"/></marker>'
         '<marker id="afa" markerWidth="10" markerHeight="9" refX="7.5" refY="3" orient="auto">'
         f'<path d="M0,0 L8,3 L0,6 Z" fill="{AMBER}"/></marker>'
         '<marker id="aft" markerWidth="10" markerHeight="9" refX="7.5" refY="3" orient="auto">'
         f'<path d="M0,0 L8,3 L0,6 Z" fill="{TEAL2}"/></marker></defs>']

    def box(x, y, num, title, sub, std, accent=TEAL):
        return (
            f'<g>'
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="#fcfdfd" stroke="{INK}" stroke-width="1.1"/>'
            f'<text x="{x+10}" y="{y+16}" font-size="9" fill="{accent}" font-weight="600" letter-spacing="0.5">{num}</text>'
            f'<text x="{x+28}" y="{y+16}" font-size="8.5" fill="{DIM}" letter-spacing="0.9">{title}</text>'
            f'<line x1="{x}" y1="{y+23}" x2="{x+bw}" y2="{y+23}" stroke="{LINE}" stroke-width="0.7"/>'
            f'<text x="{x+10}" y="{y+42}" font-size="12" fill="{INK}" font-weight="600">{sub}</text>'
            f'<text x="{x+10}" y="{y+57}" font-size="9" fill="{DIM}">{std}</text>'
            f'</g>'
        )

    xs = []
    for i, (num, title, sub, std) in enumerate(stages):
        x = x0 + i * (bw + gap)
        xs.append(x)
        p.append(box(x, ymid, num, title, sub, std))
        if i > 0:
            xp = x0 + (i - 1) * (bw + gap) + bw
            p.append(f'<line x1="{xp}" y1="{ymid+bh/2}" x2="{x}" y2="{ymid+bh/2}" '
                     f'stroke="{DIM}" stroke-width="1.2" marker-end="url(#af)"/>')
    # Side inputs feeding DETECTION (box index 3).
    x_det = xs[3]
    env_x, viv_x, sy = x_det - bw - 4, x_det + bw + 4, 46
    p.append(box(env_x, sy, "+", "ENVIRONMENT", "wave scatter", "DNV-RP-C203 &#167;5", AMBER))
    p.append(box(viv_x, sy, "+", "VIV", "current lock-in", "DNV-RP-F204", AMBER))
    p.append(f'<line x1="{env_x+bw*0.5:.0f}" y1="{sy+bh}" x2="{x_det+bw*0.30:.0f}" y2="{ymid}" '
             f'stroke="{AMBER}" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#afa)"/>')
    p.append(f'<line x1="{viv_x+bw*0.5:.0f}" y1="{sy+bh}" x2="{x_det+bw*0.70:.0f}" y2="{ymid}" '
             f'stroke="{AMBER}" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#afa)"/>')
    # Feedback loop: decision -> sensing (continuous monitoring).
    x_last = xs[-1] + bw
    p.append(f'<path d="M{x_last-bw/2},{ymid+bh} L{x_last-bw/2},{ymid+bh+36} '
             f'L{x0+bw/2},{ymid+bh+36} L{x0+bw/2},{ymid+bh}" fill="none" stroke="{TEAL2}" '
             f'stroke-width="1.1" stroke-dasharray="5 4" marker-end="url(#aft)"/>')
    p.append(f'<text x="{(x0+x_last)/2}" y="{ymid+bh+52}" text-anchor="middle" font-size="9.5" '
             f'fill="{TEAL2}">continuous re-assimilation as monitoring data accrues</text>')
    p.append(f'<text x="{x0}" y="30" font-size="12.5" fill="{INK}" letter-spacing="0.6" '
             f'font-weight="600"><tspan fill="{TEAL}" font-weight="700">B&#160;&#160;</tspan>'
             f'DIGITAL-TWIN PROCESSING CHAIN</text>')
    p.append(f'<line x1="{x0}" y1="38" x2="{W-30}" y2="38" stroke="{LINE}" stroke-width="0.8"/>')
    p.append("</svg>")
    return "".join(p)


def _svg_arrow(x1, y1, x2, y2, color, width=1.0, both=False):
    """A dimension line with slim filled arrowheads (start optional)."""
    start = ' marker-start="url(#dimstart)"' if both else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="url(#dimend)"{start}/>')


def system_schematic_svg(payload: dict, riser) -> str:
    """Detailed to-scale general-arrangement side elevation of the SCR system.

    Driven by the actual solved catenary geometry and the riser section. Drafting-
    grade line-art: a turret-moored FPSO, the suspended catenary drawn as a
    double-line pipe, hatched seabed strata with a touchdown trench, a sheared
    current profile, full dimensioning with elevation markers, three enlarged
    detail callouts (turret & hang-off hull section, riser pipe section, TDP weld
    hot-spot) and a title block. Rendered to scale (V approx H).
    """
    import math

    cat = payload["catenary"]
    xs = [float(v) for v in cat["x"]]
    ys = [float(v) for v in cat["y"]]
    depth = float(cat["water_depth"])
    span = float(cat["horizontal_span"])
    a_cat = float(cat["catenary_parameter"])
    arc = float(cat["arc_length"])
    hang_from_vert = 90.0 - float(cat["top_angle_deg"])
    kappa_km = float(cat["tdp_curvature"]) * 1000.0
    od_mm = riser.outer_diameter * 1e3
    wt_mm = riser.wall_thickness * 1e3
    grade = riser.material_grade
    coat_mm = riser.coating_thickness * 1e3
    contents = riser.contents_density

    INK, INK2, DIM, WIT = "#182530", "#3a4c54", "#8b9aa0", "#c4ced1"
    TEAL, TEALD, AMBER, SAND1 = "#0e7c82", "#0a5c61", "#a86f16", "#9c8a5f"
    STEEL, STEELF, FAINT, HULL = "#ccd4d7", "#e9eeef", "#eef1f2", "#dfe6e7"

    VBW, VBH = 1440, 880
    ml, mtop, panelW = 118, 70, 880
    surf_pad = 92
    oy = mtop + surf_pad
    bed_max = VBH - 150 - 64
    ah = bed_max - oy
    aw = panelW - ml
    scale = min(aw / span, ah / depth)
    dw, dh = span * scale, depth * scale
    ox = ml

    def SX(cx: float) -> float:
        return ox + (span - cx) * scale

    def SY(cy: float) -> float:
        return oy + (depth - cy) * scale

    surf_y, bed_y = SY(depth), SY(0.0)
    left_x, right_x = SX(span) - 0.03 * dw, SX(0.0) + 0.02 * dw
    tdp = (SX(0.0), SY(0.0))
    hang = (SX(span), SY(depth))

    def T(x, y, t, fill=INK, size=11, anchor="start", weight=400, rot=None, ls=0):
        rr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ""
        ll = f' letter-spacing="{ls}"' if ls else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
                f'font-weight="{weight}" text-anchor="{anchor}"{ll}{rr}>{t}</text>')

    def LN(x1, y1, x2, y2, c, w, d=None, marker=""):
        da = f' stroke-dasharray="{d}"' if d else ""
        return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"{da}{marker}/>'

    def RC(x, y, w, h, fill="none", stroke=INK, sw=1.0, rx=0):
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')

    p = [f'<svg viewBox="0 0 {VBW} {VBH}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'preserveAspectRatio="xMidYMid meet" font-family="Inter, Segoe UI, sans-serif">']
    p.append(
        '<defs>'
        f'<marker id="a2" markerWidth="12" markerHeight="10" refX="9.5" refY="4" orient="auto"><path d="M0,0.5 L10,4 L0,7.5 L2.8,4 Z" fill="{DIM}"/></marker>'
        f'<marker id="a1" markerWidth="12" markerHeight="10" refX="1.5" refY="4" orient="auto"><path d="M10,0.5 L0,4 L10,7.5 L7.2,4 Z" fill="{DIM}"/></marker>'
        f'<marker id="cur" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{TEAL}"/></marker>'
        f'<pattern id="hatchS" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="{INK2}" stroke-width="0.5"/></pattern>'
        f'<pattern id="soil1" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="9" stroke="{SAND1}" stroke-width="0.8"/></pattern>'
        f'<pattern id="soil2" width="14" height="14" patternUnits="userSpaceOnUse"><circle cx="3" cy="3" r="0.8" fill="{SAND1}"/><circle cx="9" cy="9" r="0.8" fill="{SAND1}"/></pattern>'
        '</defs>'
    )
    p.append(RC(0, 0, VBW, VBH, fill="#ffffff", stroke="none"))
    p.append(RC(12, 12, VBW - 24, VBH - 24, sw=1.4))
    p.append(RC(17, 17, VBW - 34, VBH - 34, sw=0.6))

    # heading
    p.append(T(ml, 44, "A", fill=TEAL, size=13, weight=700))
    p.append(T(ml + 16, 44, "SYSTEM ELEVATION &mdash; TURRET-MOORED FPSO / STEEL CATENARY RISER / TOUCHDOWN",
               fill=INK, size=12.5, weight=600, ls=0.4))
    p.append(LN(ml, 52, ml + panelW, 52, WIT, 0.8))

    # to-scale grid + elevation markers (every 250 m)
    z = 0.0
    while z <= depth + 1:
        y = SY(depth - z)
        p.append(LN(left_x, y, right_x, y, FAINT, 1))
        p.append(T(left_x - 6, y + 3, f"EL {-int(z)}", fill=WIT, size=8.5, anchor="end"))
        z += 250.0
    gx = 0.0
    while gx <= span + 1:
        x = SX(span - gx)
        p.append(LN(x, surf_y, x, bed_y, FAINT, 1))
        gx += 250.0

    # seabed strata + mudline + touchdown trench
    bed_thk = 0.10 * dh
    p.append(RC(left_x, bed_y, right_x - left_x, bed_thk * 0.5, fill="url(#soil1)", stroke="none"))
    p.append(RC(left_x, bed_y + bed_thk * 0.5, right_x - left_x, bed_thk * 0.5, fill="url(#soil2)", stroke="none"))
    p.append(LN(left_x, bed_y, right_x, bed_y, SAND1, 1.6))
    p.append(LN(left_x, bed_y + bed_thk * 0.5, right_x, bed_y + bed_thk * 0.5, SAND1, 0.7, "6 3"))
    p.append(T(right_x - 4, bed_y + bed_thk * 0.28, "SOFT CLAY", fill=SAND1, size=8.5, anchor="end"))
    p.append(T(right_x - 4, bed_y + bed_thk * 0.78, "STIFF CLAY", fill=SAND1, size=8.5, anchor="end"))
    p.append(f'<path d="M{tdp[0] - 0.05 * dw:.1f},{bed_y:.1f} q{0.05 * dw:.1f},14 {0.10 * dw:.1f},0" '
             f'fill="none" stroke="{SAND1}" stroke-width="1" stroke-dasharray="3 2"/>')

    # mean water level + datum symbol
    p.append(LN(left_x, surf_y, right_x, surf_y, INK, 1.1))
    wlx = left_x
    wlp = []
    while wlx < right_x - 12:
        wlp.append(f'M{wlx:.1f},{surf_y:.1f} q6,-4 12,0')
        wlx += 12
    p.append(f'<path d="{" ".join(wlp)}" fill="none" stroke="{TEAL}" stroke-width="0.7" opacity="0.5"/>')
    dxm = SX(0.0) - 0.02 * dw
    p.append(f'<path d="M{dxm:.1f},{surf_y:.1f} l6,10 l-12,0 Z" fill="none" stroke="{INK}" stroke-width="1"/>')
    p.append(LN(dxm - 9, surf_y + 13, dxm + 9, surf_y + 13, INK, 0.8))
    p.append(T(dxm + 13, surf_y - 6, "MWL &nbsp;EL 0.0", fill=INK, size=10))

    # sheared current profile (real, if VIV/current active)
    viv = payload.get("viv")
    if viv and viv.get("enabled"):
        cp = viv["current_profile"]
        hh = [float(v) for v in cp["height"]]
        ss = [float(v) for v in cp["speed"]]
        umax = max(max(ss), 1e-6)
        u0 = float(viv.get("current_surface_velocity", ss[-1] if ss else 0.0))
        cur_x, cur_len = left_x + 6, 0.12 * dw
        env = [f'{cur_x:.1f},{surf_y:.1f}']
        for h, s in zip(hh, ss):
            env.append(f'{cur_x + (s / umax) * cur_len:.1f},{SY(h):.1f}')
        env.append(f'{cur_x:.1f},{bed_y:.1f}')
        p.append(f'<polygon points="{" ".join(env)}" fill="rgba(14,124,130,0.06)" '
                 f'stroke="{TEAL}" stroke-width="0.8" stroke-dasharray="3 3"/>')
        for i in range(3, len(hh) - 1, 7):
            s = ss[i]
            if s > 0.03 * umax:
                p.append(LN(cur_x, SY(hh[i]), cur_x + (s / umax) * cur_len, SY(hh[i]),
                            TEAL, 1.2, marker=' marker-end="url(#cur)"'))
        p.append(T(cur_x, surf_y - 19, "CURRENT U(z)", fill=TEAL, size=9.5, weight=600))
        p.append(T(cur_x, surf_y - 8, f"U_s = {u0:.2f} m/s &#183; 1/7 power law", fill=TEAL, size=9))

    # SCR as a double-line pipe (OD walls) to scale
    rp = [(SX(x), SY(y)) for x, y in zip(xs, ys)]
    od_px = 3.2

    def _offset(pts, off):
        out = []
        for i in range(len(pts)):
            a = pts[max(0, i - 1)]
            b = pts[min(len(pts) - 1, i + 1)]
            ddx, ddy = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(ddx, ddy) or 1.0
            out.append((pts[i][0] - ddy / ln * off, pts[i][1] + ddx / ln * off))
        return out

    up = _offset(rp, od_px)
    dn = _offset(rp, -od_px)
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in up)}" fill="none" stroke="{TEALD}" stroke-width="1.1"/>')
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in dn)}" fill="none" stroke="{TEALD}" stroke-width="1.1"/>')
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in rp)}" fill="none" stroke="{TEAL}" '
             f'stroke-width="{od_px * 2 - 1.2:.1f}" stroke-linecap="round" opacity="0.55"/>')
    p.append(LN(tdp[0], tdp[1], tdp[0] + 0.02 * dw, tdp[1], TEAL, od_px * 2 - 1.2))

    # ---- FPSO (turret-moored) side profile at the hang-off ----
    Lf = 0.34 * dw
    bx = hang[0] - Lf * 0.30
    fwd, aft = bx + Lf * 0.70, bx - Lf * 0.30
    deck_y, keel_y = surf_y - 0.035 * dh, surf_y + 0.045 * dh
    tur = hang[0]
    for f in (-0.16, 0.11):
        axp, ay = tur + f * dw, bed_y
        p.append(f'<path d="M{tur:.1f},{keel_y:.1f} Q{(tur + axp) / 2:.1f},{keel_y + (ay - keel_y) * 0.88:.1f} '
                 f'{axp:.1f},{ay:.1f}" fill="none" stroke="{INK2}" stroke-width="0.8" stroke-dasharray="5 3"/>')
        p.append(f'<path d="M{axp - 4:.1f},{ay:.1f} l4,7 l4,-7 Z" fill="{INK2}"/>')
    hull = (f'M{aft:.1f},{deck_y:.1f} L{fwd - 0.05 * Lf:.1f},{deck_y - 0.008 * dh:.1f} '
            f'Q{fwd:.1f},{deck_y - 0.006 * dh:.1f} {fwd:.1f},{surf_y - 0.004 * dh:.1f} '
            f'L{fwd - 0.02 * Lf:.1f},{keel_y:.1f} L{aft + 0.03 * Lf:.1f},{keel_y:.1f} '
            f'Q{aft:.1f},{keel_y:.1f} {aft:.1f},{keel_y - 0.02 * dh:.1f} Z')
    p.append(f'<path d="{hull}" fill="{HULL}" stroke="{INK}" stroke-width="1.6" stroke-linejoin="round"/>')
    p.append(LN(aft, surf_y, fwd, surf_y, INK2, 0.6))
    p.append(LN(aft + 2, surf_y - 2.5, fwd - 4, surf_y - 2.5, TEALD, 1.4))
    p.append(LN(aft, deck_y, fwd - 0.05 * Lf, deck_y, INK, 0.8))
    mod_y, mod_h = deck_y - 0.05 * dh, 0.045 * dh
    for i in range(4):
        mx = aft + 0.12 * Lf + i * 0.145 * Lf
        p.append(RC(mx, mod_y, 0.115 * Lf, mod_h, fill=STEELF, sw=0.9))
    p.append(RC(aft + 0.02 * Lf, mod_y - 0.05 * dh, 0.10 * Lf, 0.05 * dh + mod_h, fill=STEELF, sw=1))
    p.append(f'<ellipse cx="{aft + 0.07 * Lf:.1f}" cy="{mod_y - 0.05 * dh - 5:.1f}" '
             f'rx="{0.045 * Lf:.1f}" ry="2.6" fill="none" stroke="{INK}" stroke-width="0.9"/>')
    p.append(T(aft + 0.07 * Lf, mod_y - 0.05 * dh - 3, "H", fill=INK, size=6.5, anchor="middle", weight=700))
    p.append(f'<path d="M{fwd - 0.11 * Lf:.1f},{mod_y:.1f} L{fwd - 0.065 * Lf:.1f},{mod_y - 0.10 * dh:.1f} '
             f'L{fwd - 0.02 * Lf:.1f},{mod_y:.1f}" fill="none" stroke="{INK}" stroke-width="1"/>')
    for k in range(1, 4):
        t = k / 4.0
        p.append(LN(fwd - 0.11 * Lf + (0.045 * Lf) * t, mod_y - 0.10 * dh * t,
                    fwd - 0.02 * Lf - (0.045 * Lf) * t, mod_y - 0.10 * dh * t, INK2, 0.4))
    p.append(f'<path d="M{fwd - 0.065 * Lf:.1f},{mod_y - 0.10 * dh:.1f} q5,-7 11,-2 q-3,5 -9,3" fill="{AMBER}" opacity="0.7"/>')
    p.append(RC(tur - 0.026 * Lf, deck_y, 0.052 * Lf, keel_y - deck_y, fill="#ffffff", sw=1))
    p.append(LN(tur - 0.026 * Lf, keel_y, tur + 0.026 * Lf, keel_y, INK, 1.2))
    p.append(RC(tur - 0.02 * Lf, deck_y - 0.028 * dh, 0.04 * Lf, 0.028 * dh, fill=STEEL, sw=1))
    p.append(f'<circle cx="{tur:.1f}" cy="{keel_y:.1f}" r="3.2" fill="{AMBER}" stroke="#ffffff" stroke-width="1"/>')
    p.append(T((aft + fwd) / 2, mod_y - 0.10 * dh - 6, "FPSO (turret-moored)", fill=INK, size=10, weight=600, anchor="middle"))
    p.append(f'<circle cx="{tur + 16:.1f}" cy="{keel_y + 2:.1f}" r="8" fill="none" stroke="{TEAL}" stroke-width="1"/>')
    p.append(T(tur + 16, keel_y + 5, "B", fill=TEAL, size=9, anchor="middle", weight=700))

    # hang-off centreline + departure-angle arc (at the top)
    p.append(LN(hang[0], keel_y, hang[0], keel_y + 0.30 * dh, DIM, 0.9, "7 3 2 3"))
    p0 = (SX(xs[-1]), SY(ys[-1]))
    p1 = (SX(xs[-4]), SY(ys[-4]))
    tang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    r_arc = 0.15 * dh
    a0 = (hang[0] + r_arc * math.cos(math.pi / 2), keel_y + r_arc)
    a1 = (hang[0] + r_arc * math.cos(tang), keel_y + r_arc * math.sin(tang))
    p.append(f'<path d="M{a0[0]:.1f},{a0[1]:.1f} A{r_arc:.1f},{r_arc:.1f} 0 0 0 {a1[0]:.1f},{a1[1]:.1f}" '
             f'fill="none" stroke="{AMBER}" stroke-width="1.4"/>')
    p.append(T(hang[0] + r_arc * 0.7, keel_y + r_arc * 0.7, f"&#952;={hang_from_vert:.0f}&#176;", fill=AMBER, size=11, weight=600))

    # TDP hot-spot + leader + detail flag
    p.append(f'<circle cx="{tdp[0]:.1f}" cy="{tdp[1]:.1f}" r="6" fill="none" stroke="{AMBER}" stroke-width="1.6"/>')
    p.append(f'<circle cx="{tdp[0]:.1f}" cy="{tdp[1]:.1f}" r="2.5" fill="{AMBER}"/>')
    p.append(LN(tdp[0], tdp[1], tdp[0] - 0.09 * dw, tdp[1] - 0.09 * dh, AMBER, 0.8))
    p.append(T(tdp[0] - 0.09 * dw - 4, tdp[1] - 0.09 * dh,
               f"TOUCHDOWN POINT &nbsp;&#954;={kappa_km:.2f}/km", fill=AMBER, size=10, anchor="end", weight=600))
    p.append(f'<circle cx="{tdp[0] - 0.09 * dw - 92:.1f}" cy="{tdp[1] - 0.09 * dh + 12:.1f}" r="8" fill="none" stroke="{AMBER}" stroke-width="1"/>')
    p.append(T(tdp[0] - 0.09 * dw - 92, tdp[1] - 0.09 * dh + 15.5, "C", fill=AMBER, size=9, anchor="middle", weight=700))

    # water-depth dimension (left) + horizontal layback (bottom)
    wdx = left_x - 46
    p.append(LN(wdx, surf_y, wdx, bed_y, DIM, 0.9, marker=' marker-start="url(#a1)" marker-end="url(#a2)"'))
    p.append(LN(left_x, surf_y, wdx - 6, surf_y, WIT, 0.7))
    p.append(LN(left_x, bed_y, wdx - 6, bed_y, WIT, 0.7))
    p.append(T(wdx - 11, (surf_y + bed_y) / 2, f"WATER DEPTH &nbsp;d = {depth:.0f} m", fill=INK, size=10, anchor="middle", rot=90))
    lby = bed_y + bed_thk + 42
    p.append(LN(hang[0], lby, tdp[0], lby, DIM, 0.9, marker=' marker-start="url(#a1)" marker-end="url(#a2)"'))
    p.append(LN(hang[0], keel_y, hang[0], lby + 6, WIT, 0.7, "4 3"))
    p.append(LN(tdp[0], bed_y, tdp[0], lby + 6, WIT, 0.7))
    p.append(T((hang[0] + tdp[0]) / 2, lby - 8, f"HORIZONTAL LAYBACK &nbsp;X = {span:.0f} m", fill=INK, size=10, anchor="middle"))

    # ================= RIGHT: DETAIL PANELS =================
    rx0 = ml + panelW + 34
    rw = VBW - rx0 - 24
    p.append(LN(rx0 - 14, mtop - 6, rx0 - 14, VBH - 150, WIT, 0.8))

    # DETAIL B : turret & hang-off hull section
    ay0 = mtop + 2
    p.append(T(rx0, ay0, "B", fill=TEAL, size=12, weight=700))
    p.append(T(rx0 + 15, ay0, "DETAIL &mdash; TURRET & RISER HANG-OFF", fill=INK, size=10.5, weight=600))
    p.append(T(rx0 + rw, ay0, "SCALE 1:250", fill=DIM, size=8.5, anchor="end"))
    dA_x, dA_y, dA_w, dA_h = rx0, ay0 + 10, rw, 224
    p.append(RC(dA_x, dA_y, dA_w, dA_h, fill="#fcfdfd", stroke=WIT, sw=0.8))
    hsx, hsw = dA_x + 18, dA_w - 36
    hsy, hh2 = dA_y + 30, dA_h - 70
    p.append(RC(hsx, hsy, hsw, hh2, fill=HULL, sw=1.4))
    wlY = hsy + 0.30 * hh2
    p.append(LN(hsx - 6, wlY, hsx + hsw + 6, wlY, TEALD, 1.2))
    p.append(T(hsx - 8, wlY + 3, "WL", fill=TEALD, size=8, anchor="end"))
    for i, lb in enumerate(("MAIN DECK", "TWEEN DECK", "TANK TOP")):
        y = hsy + (0.16 + 0.24 * i) * hh2
        if i > 0:
            p.append(LN(hsx, y, hsx + hsw, y, INK2, 0.7))
        p.append(T(hsx + 3, y - 3, lb, fill=INK2, size=7))
    p.append(LN(hsx, hsy + hh2 - 0.10 * hh2, hsx + hsw, hsy + hh2 - 0.10 * hh2, INK, 0.9))
    p.append(T(hsx + 3, hsy + hh2 - 0.10 * hh2 + 9, "DOUBLE BOTTOM", fill=INK2, size=7))
    for i in range(1, 8):
        fx = hsx + i * hsw / 8
        p.append(LN(fx, hsy, fx, hsy + hh2, INK2, 0.3))
    mpx, mpw = hsx + hsw * 0.42, hsw * 0.16
    p.append(RC(mpx, hsy, mpw, hh2, fill="#ffffff", sw=1.1))
    p.append(RC(mpx + mpw * 0.15, hsy - 14, mpw * 0.7, 20, fill=STEEL, sw=1))
    p.append(T(mpx + mpw * 0.5, hsy - 17, "SWIVEL", fill=INK, size=7, anchor="middle"))
    p.append(f'<circle cx="{mpx + mpw * 0.5:.1f}" cy="{hsy + 0.10 * hh2:.1f}" r="4" fill="none" stroke="{INK}" stroke-width="1"/>')
    p.append(T(mpx + mpw + 4, hsy + 0.10 * hh2 + 3, "MAIN BEARING", fill=INK2, size=7))
    p.append(RC(mpx + mpw * 0.30, hsy + 0.20 * hh2, mpw * 0.4, 0.55 * hh2, fill=STEELF, sw=0.9))
    p.append(T(mpx + mpw * 0.5, hsy + 0.48 * hh2, "TURRET", fill=INK2, size=7, anchor="middle", rot=90))
    p.append(T(mpx + mpw * 0.5, hsy + hh2 + 10, "MOONPOOL", fill=INK2, size=7, anchor="middle"))
    fjx, fjy = mpx + mpw * 0.5, hsy + hh2
    p.append(f'<circle cx="{fjx:.1f}" cy="{fjy:.1f}" r="4.5" fill="{AMBER}" stroke="#ffffff" stroke-width="1"/>')
    p.append(f'<path d="M{fjx:.1f},{fjy:.1f} q22,26 46,40" fill="none" stroke="{TEAL}" stroke-width="2.6"/>')
    p.append(T(fjx + 50, fjy + 44, "FLEX JOINT", fill=AMBER, size=8, weight=600))
    p.append(T(fjx + 50, fjy + 55, "&rarr; SCR hang-off", fill=TEAL, size=8))

    # DETAIL A : riser pipe section
    by0 = dA_y + dA_h + 26
    p.append(T(rx0, by0, "A", fill=TEAL, size=12, weight=700))
    p.append(T(rx0 + 15, by0, "DETAIL &mdash; RISER PIPE SECTION", fill=INK, size=10.5, weight=600))
    pcx, pcy = rx0 + 70, by0 + 72
    r_o, r_i, r_c = 48.0, 48.0 - min(wt_mm / od_mm * 96.0, 30.0), 48.0 + max(coat_mm / od_mm * 96.0, 4.0)
    p.append(f'<circle cx="{pcx}" cy="{pcy}" r="{r_c:.1f}" fill="#f0ece0" stroke="{SAND1}" stroke-width="1"/>')
    p.append(f'<circle cx="{pcx}" cy="{pcy}" r="{r_o}" fill="{STEEL}" stroke="{INK}" stroke-width="1.4"/>')
    p.append(f'<circle cx="{pcx}" cy="{pcy}" r="{r_i:.1f}" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>')
    p.append(f'<path d="M {pcx - r_o} {pcy} A {r_o} {r_o} 0 0 1 {pcx} {pcy - r_o} L {pcx} {pcy - r_i:.1f} '
             f'A {r_i:.1f} {r_i:.1f} 0 0 0 {pcx - r_i:.1f} {pcy} Z" fill="url(#hatchS)" opacity="0.8"/>')
    p.append(LN(pcx - r_o, pcy + r_o + 14, pcx + r_o, pcy + r_o + 14, DIM, 0.8, marker=' marker-start="url(#a1)" marker-end="url(#a2)"'))
    p.append(T(pcx, pcy + r_o + 26, f"OD {od_mm:.0f} mm", fill=INK, size=9, anchor="middle"))
    lx2 = pcx + r_c + 18
    rows_a = [("OD steel", f"{od_mm:.1f} mm"), ("wall t", f"{wt_mm:.1f} mm"), ("grade", grade),
              ("coating", f"{coat_mm:.0f} mm"), ("contents", f"{contents:.0f} kg/m&#179;")]
    for i, (k, v) in enumerate(rows_a):
        yy = by0 + 34 + i * 15
        p.append(T(lx2, yy, k, fill=INK2, size=8.5))
        p.append(T(rx0 + rw, yy, v, fill=INK, size=9, anchor="end"))

    # DETAIL C : TDP weld hot-spot
    cy0 = by0 + 150
    p.append(T(rx0, cy0, "C", fill=AMBER, size=12, weight=700))
    p.append(T(rx0 + 15, cy0, "DETAIL &mdash; TDP WELD HOT-SPOT", fill=INK, size=10.5, weight=600))
    wx, wy, wl2 = rx0 + 16, cy0 + 56, rw - 40
    p.append(RC(wx, wy - 13, wl2 * 0.46, 26, fill=STEEL, sw=1.2))
    p.append(RC(wx + wl2 * 0.50, wy - 13, wl2 * 0.46, 26, fill=STEEL, sw=1.2))
    p.append(f'<path d="M{wx + wl2 * 0.46:.1f},{wy - 13:.1f} q{wl2 * 0.04:.1f},-6 {wl2 * 0.08:.1f},0 '
             f'l0,26 q{-wl2 * 0.04:.1f},6 {-wl2 * 0.08:.1f},0 Z" fill="{STEELF}" stroke="{INK}" stroke-width="1"/>')
    p.append(T(wx + wl2 * 0.5, wy - 18, "girth weld", fill=INK2, size=8, anchor="middle"))
    p.append(f'<path d="M{wx + wl2 * 0.46:.1f},{wy + 13:.1f} l3,7 l-2,1 l3,6" fill="none" stroke="{AMBER}" stroke-width="1.6"/>')
    p.append(T(wx + wl2 * 0.46 + 14, wy + 24, "fatigue crack (weld toe)", fill=AMBER, size=8))
    p.append(LN(wx - 4, wy + 15, wx + wl2 + 4, wy + 15, SAND1, 1.2))
    p.append(LN(wx - 2, wy, wx - 16, wy, INK2, 1, marker=' marker-end="url(#a1)"'))
    p.append(LN(wx + wl2 + 2, wy, wx + wl2 + 16, wy, INK2, 1, marker=' marker-end="url(#a2)"'))
    p.append(T(wx + wl2 * 0.5, wy + 40, "cyclic bending &#916;&#963; from vessel motion + VIV", fill=INK2, size=8, anchor="middle"))

    # ================= TITLE BLOCK =================
    tby, tbx = VBH - 138, ml
    tbw = VBW - 24 - ml
    p.append(RC(tbx, tby, tbw, 120, sw=1.2))
    c1, c2, c3 = tbx, tbx + tbw * 0.40, tbx + tbw * 0.70
    p.append(LN(c2, tby, c2, tby + 120, INK, 0.8))
    p.append(LN(c3, tby, c3, tby + 120, INK, 0.8))
    p.append(LN(c1, tby + 70, c2, tby + 70, INK, 0.6))
    p.append(T(c1 + 12, tby + 26, "STEEL CATENARY RISER &mdash; GENERAL ARRANGEMENT", fill=INK, size=13, weight=700))
    p.append(T(c1 + 12, tby + 46, "System side elevation &#183; vessel &rarr; suspended catenary &rarr; touchdown point", fill=INK2, size=10))
    p.append(T(c1 + 12, tby + 90, "PROJECT", fill=DIM, size=8, weight=600))
    p.append(T(c1 + 12, tby + 104, "SCR-Twin reference case (illustrative)", fill=INK, size=9.5))
    rows_t = [("Water depth", f"{depth:.0f} m"), ("Hang-off angle", f"{hang_from_vert:.0f}&#176; from vertical"),
              ("Catenary a=H/w", f"{a_cat:.0f} m"), ("Suspended arc length", f"{arc:.0f} m"),
              ("Horizontal layback", f"{span:.0f} m"), ("TDP curvature", f"{kappa_km:.2f} /km")]
    for i, (k, v) in enumerate(rows_t):
        col = c2 + 12 if i < 3 else c2 + tbw * 0.15 + 12
        yy = tby + 22 + (i % 3) * 30
        p.append(T(col, yy, k, fill=DIM, size=8, weight=600))
        p.append(T(col, yy + 13, v, fill=INK, size=10))
    p.append(T(c3 + 12, tby + 22, "SCALE", fill=DIM, size=8, weight=600))
    p.append(T(tbx + tbw - 12, tby + 22, "TO SCALE (V approx H), m", fill=INK, size=9, anchor="end"))
    p.append(T(c3 + 12, tby + 46, "UNITS", fill=DIM, size=8, weight=600))
    p.append(T(tbx + tbw - 12, tby + 46, "metres", fill=INK, size=9, anchor="end"))
    p.append(T(c3 + 12, tby + 70, "REV", fill=DIM, size=8, weight=600))
    p.append(T(tbx + tbw - 12, tby + 70, "B", fill=INK, size=9, anchor="end"))
    p.append(T(c3 + 12, tby + 94, "STATUS", fill=DIM, size=8, weight=600))
    p.append(T(tbx + tbw - 12, tby + 94, "ILLUSTRATIVE &mdash; NOT FOR CONSTRUCTION", fill=AMBER, size=8.5, anchor="end", weight=600))
    p.append(T(ml, tby - 8, "PLANE OF SECTION: vertical, in the riser departure azimuth", fill=DIM, size=8.5))

    p.append("</svg>")
    return "".join(p)


def along_riser_stress_fig(cat: dict, e_mod: float, od: float, scf: float) -> go.Figure:
    """Static bending-stress distribution along the riser arc: sigma = SCF·E·(D/2)·kappa(s).

    Shows the stress concentrating at the touchdown point where the curvature peaks
    - the reason SCR fatigue localises at the TDP.
    """
    a = float(cat["catenary_parameter"])
    span = float(cat["horizontal_span"])
    x = np.linspace(0.0, span, 260)
    kappa = 1.0 / (a * np.cosh(x / a) ** 2)          # curvature [1/m]
    arc = a * np.sinh(x / a)                          # arc length from TDP [m]
    sigma = scf * e_mod * (od / 2.0) * kappa / 1e6    # outer-fibre bending stress [MPa]
    f = _fig(250)
    f.add_scatter(x=arc, y=sigma, line=dict(color=AMBER, width=2.4),
                  fill="tozeroy", fillcolor="rgba(180,121,26,0.10)", name="σ_bend")
    f.add_scatter(x=[0.0], y=[float(sigma[0])], mode="markers+text", text=["TDP"],
                  textposition="top right", marker=dict(color=ALARM, size=9),
                  textfont=dict(color=ALARM, size=10))
    f.update_layout(
        xaxis=dict(title="arc length from TDP [m]", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="static bending stress [MPa]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def viv_mode_fig(viv: dict) -> go.Figure:
    """Dominant cross-flow VIV mode shape along the riser arc (the standing wave)."""
    f = _fig(250)
    sh = viv["dominant_shape"]
    arc = np.asarray(sh["arc"], dtype=float)
    disp = np.asarray(sh["disp"], dtype=float)
    f.add_scatter(x=arc, y=disp, line=dict(color=SIGNAL2, width=2.2), name="mode")
    f.add_scatter(x=arc, y=-disp, line=dict(color=SIGNAL2, width=0.8, dash="dot"),
                  hoverinfo="skip")  # envelope (the mode oscillates +/-)
    f.add_hline(y=0.0, line=dict(color=TEXT, width=0.6))
    f.update_layout(
        xaxis=dict(title="arc length from TDP [m]", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="cross-flow mode shape [-]", gridcolor=GRID, zeroline=False,
                   range=[-1.2, 1.2]))
    return f


def viv_vr_fig(viv: dict) -> go.Figure:
    """Reduced velocity Vr per mode with the cross-flow lock-in band shaded."""
    f = _fig(250)
    modes = viv["modes"]
    n = [m["mode"] for m in modes]
    vr = [m["reduced_velocity"] for m in modes]
    colors = [AMBER if m["excited"] else "#cbd5d6" for m in modes]
    f.add_hrect(y0=3.0, y1=9.0, fillcolor="rgba(180,121,26,0.10)", line_width=0,
                annotation_text="lock-in", annotation_font_size=9, annotation_font_color=AMBER)
    f.add_bar(x=n, y=vr, marker_color=colors, name="Vr")
    f.update_layout(
        xaxis=dict(title="cross-flow mode number", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="reduced velocity Vr = U / (fn D)", gridcolor=GRID, zeroline=False,
                   range=[0, max(12.0, min(30.0, max(vr) if vr else 12.0))]))
    return f


def knockdown_fig(sweep: dict, current_mm: float) -> go.Figure:
    """VIV screening life (log) vs marine-growth thickness, with a live marker."""
    thk = sweep["thickness_mm"]
    lifed = sweep["life_years"]
    f = _fig(300)
    f.add_scatter(x=thk, y=lifed, mode="lines+markers", line=dict(color=SIGNAL2, width=2.4),
                  marker=dict(color=SIGNAL2, size=5), name="VIV life")
    dl = sweep.get("design_life_years")
    if dl:
        f.add_hline(y=dl, line=dict(color=AMBER, width=1.2, dash="dash"),
                    annotation_text=f"{dl:.0f}-yr design life", annotation_font_color=AMBER,
                    annotation_font_size=9, annotation_position="top right")
    if thk:
        cur_life = float(np.interp(current_mm, thk, lifed))
        below = bool(dl and cur_life < dl)
        f.add_scatter(x=[current_mm], y=[cur_life], mode="markers+text",
                      marker=dict(color=ALARM if below else SIGNAL, size=11, symbol="circle",
                                  line=dict(color="#ffffff", width=1.6)),
                      text=[f"  {current_mm:.0f} mm"], textposition="middle right",
                      textfont=dict(color=ALARM if below else SIGNAL2, size=11), name="now")
    _fin = [v for v in lifed if np.isfinite(v) and v > 0]
    _dl = [dl] if dl else []
    _lo = min(_fin + _dl) if _fin else 1.0
    _hi = max(_fin + _dl) if _fin else 1e3
    f.update_layout(
        xaxis=dict(title="marine growth thickness [mm]", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="VIV screening life [yr]", type="log", gridcolor=GRID, zeroline=False,
                   range=[float(np.log10(max(1.0, _lo * 0.6))), float(np.log10(_hi * 1.6))]))
    return f


def scatter_fig(lt: dict) -> go.Figure:
    """Fatigue-driver heat map over the Hs-Tp scatter diagram (% of long-term damage)."""
    hs_vals = list(lt["hs_values"])
    tp_vals = list(lt["tp_values"])
    hi = {v: i for i, v in enumerate(hs_vals)}
    ti = {v: i for i, v in enumerate(tp_vals)}
    z = np.full((len(hs_vals), len(tp_vals)), np.nan)
    for c in lt["contributions"]:
        z[hi[c["hs"]], ti[c["tp"]]] = 100.0 * c["damage_fraction"]
    f = _fig(300)
    f.add_trace(go.Heatmap(
        x=tp_vals, y=hs_vals, z=z, colorscale=[[0, "rgba(15,143,156,0.05)"], [1, AMBER]],
        colorbar=dict(title="% dmg", thickness=10), hoverongaps=False,
        hovertemplate="Hs %{y} m, Tp %{x} s<br>%{z:.1f}% of damage<extra></extra>"))
    f.update_layout(
        xaxis=dict(title="Tp [s]", gridcolor=GRID, zeroline=False, dtick=2),
        yaxis=dict(title="Hs [m]", gridcolor=GRID, zeroline=False))
    return f


def data_table(headers: list[str], rows: list[list], value_cols: tuple[int, ...] = (1,)) -> str:
    """Render a technical HTML data table using the .ledger styling."""
    th = "".join(f"<th>{h}</th>" for h in headers)
    out = [f'<table class="ledger"><thead><tr>{th}</tr></thead><tbody>']
    for r in rows:
        tds = ""
        for i, c in enumerate(r):
            cls = "lv" if i in value_cols else ("lb" if i == len(r) - 1 else "")
            tds += f'<td class="{cls}">{c}</td>'
        out.append(f"<tr>{tds}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def jonswap_wave_fig(sea: dict) -> go.Figure:
    """The identified JONSWAP wave-elevation spectrum S(f) driving the response."""
    f = _fig(230)
    fr = np.linspace(0.02, 0.45, 400)
    s = jonswap(fr, sea["hs"], sea["tp"], gamma=max(sea["gamma"], 1.0), normalize=True)
    f.add_scatter(x=fr, y=s, line=dict(color=SIGNAL, width=2),
                  fill="tozeroy", fillcolor="rgba(16,162,170,0.08)")
    f.add_vline(x=1.0 / sea["tp"], line=dict(color=AMBER, width=1, dash="dot"),
                annotation_text="fp", annotation_font_size=9, annotation_font_color=AMBER)
    f.update_layout(xaxis=dict(title="frequency [Hz]", gridcolor=GRID, zeroline=False, range=[0, 0.45]),
                    yaxis=dict(title="S(f) [m²/Hz]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def hf_phase_fig(tf: dict) -> go.Figure:
    """Phase of the TDP transfer function H(f)."""
    f = _fig(230)
    ph = np.degrees(np.asarray(tf["phase"], dtype=float))
    f.add_scatter(x=tf["freq"], y=ph, line=dict(color=SIGNAL2, width=1.8))
    f.update_layout(xaxis=dict(title="frequency [Hz]", gridcolor=GRID, zeroline=False, range=[0, 0.4]),
                    yaxis=dict(title="phase ∠H(f) [deg]", gridcolor=GRID, zeroline=False))
    return f


def rainflow_hist_fig(ver: dict) -> go.Figure:
    """Rainflow stress-range histogram (counted cycles per range bin)."""
    f = _fig(230)
    edges = np.asarray(ver["hist_edges_mpa"], dtype=float)
    counts = np.asarray(ver["hist_counts"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    f.add_bar(x=centres, y=counts, marker_color="rgba(11,125,132,0.35)",
              marker_line_color=SIGNAL2, marker_line_width=0.4)
    f.update_layout(xaxis=dict(title="stress range [MPa]", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="counted cycles", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def cdf_fig(post: dict) -> go.Figure:
    """Empirical CDF of the Monte-Carlo remaining-life posterior with P10/50/90."""
    f = _fig(230)
    f.add_scatter(x=post["cdf_x"], y=post["cdf_p"], line=dict(color=SIGNAL2, width=2.2))
    for key, col in (("p10", AMBER), ("p50", SIGNAL2), ("p90", SIGNAL)):
        f.add_vline(x=post[key], line=dict(color=col, width=1, dash="dash"),
                    annotation_text=key.upper(), annotation_font_size=9, annotation_font_color=col)
    f.update_layout(xaxis=dict(title="remaining life [yr]", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="cumulative probability", gridcolor=GRID, zeroline=False, range=[0, 1]))
    return f


def current_profile_fig(viv: dict) -> go.Figure:
    """Sheared current speed vs height above the seabed."""
    f = _fig(230)
    cp = viv["current_profile"]
    f.add_scatter(x=cp["speed"], y=cp["height"], line=dict(color=SIGNAL, width=2),
                  fill="tozerox", fillcolor="rgba(16,162,170,0.08)")
    f.update_layout(xaxis=dict(title="current speed U [m/s]", gridcolor=GRID, zeroline=False, rangemode="tozero"),
                    yaxis=dict(title="height above seabed [m]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def riser_tension_fig(cat: dict, w: float) -> go.Figure:
    """Effective tension and axial stress mean along the riser arc."""
    a = float(cat["catenary_parameter"])
    span = float(cat["horizontal_span"])
    x = np.linspace(0.0, span, 240)
    arc = a * np.sinh(x / a)
    tension = w * a * np.cosh(x / a)   # T(x) = H cosh(x/a), H = w a
    f = _fig(230)
    f.add_scatter(x=arc, y=tension / 1e3, line=dict(color=SIGNAL2, width=2.2),
                  fill="tozeroy", fillcolor="rgba(11,125,132,0.06)")
    f.update_layout(xaxis=dict(title="arc length from TDP [m]", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="effective tension [kN]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def seabed_fig(sb: dict) -> go.Figure:
    """Fatigue life vs seabed vertical stiffness (rigid base = conservative)."""
    f = _fig(240)
    f.add_scatter(x=sb["k_v_kpa"], y=sb["life_years"], line=dict(color=SIGNAL2, width=2.4),
                  fill="tozeroy", fillcolor="rgba(11,125,132,0.06)")
    f.add_hline(y=sb["base_life_years"], line=dict(color=ALARM, width=1, dash="dash"),
                annotation_text="rigid seabed (base)", annotation_font_size=9, annotation_font_color=ALARM)
    f.update_layout(xaxis=dict(title="seabed stiffness k_v [kPa]", type="log", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="fatigue life [yr]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def reliability_fig(rel: dict) -> go.Figure:
    """FORM importance factors (alpha^2) - which uncertainty source drives Pf."""
    items = sorted(rel["importance"].items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [100.0 * v for _, v in items]
    colors = [SIGNAL2 if v == max(vals) else "#9db4bb" for v in vals]
    f = _fig(240)
    f.add_bar(y=names, x=vals, orientation="h", marker_color=colors,
              text=[f"{v:.0f}%" for v in vals], textposition="outside", cliponaxis=False)
    f.update_layout(
        xaxis=dict(title="importance α² [% of ln-life variance]", gridcolor=GRID, zeroline=False,
                   range=[0, max(vals) * 1.25 if vals else 1.0]),
        yaxis=dict(autorange="reversed"), margin=dict(l=90, r=30, t=10, b=40))
    return f


def crack_growth_ui_fig(crack: dict) -> go.Figure:
    """Paris-law crack depth a(t) vs year, with the critical depth and inspection."""
    at = crack["a_of_t"]
    f = _fig(240)
    f.add_scatter(x=at["years"], y=at["depth_mm"], line=dict(color=ALARM, width=2.4),
                  fill="tozeroy", fillcolor="rgba(192,67,47,0.07)", name="a(t)")
    f.add_hline(y=crack["critical_depth_mm"], line=dict(color=ALARM, width=1, dash="dash"),
                annotation_text="critical (through-wall)", annotation_font_size=9,
                annotation_font_color=ALARM)
    ci = crack.get("crack_inspection_year")
    if ci is not None:
        f.add_vline(x=ci, line=dict(color=SIGNAL2, width=1.2, dash="dot"),
                    annotation_text="inspect", annotation_font_size=9, annotation_font_color=SIGNAL2)
    f.update_layout(xaxis=dict(title="year", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="crack depth a [mm]", gridcolor=GRID, zeroline=False, rangemode="tozero"))
    return f


def pod_crack_fig(crack: dict) -> go.Figure:
    """Probability of detection vs crack size (subsea NDE)."""
    pod = crack["pod"]
    f = _fig(240)
    f.add_scatter(x=pod["size_mm"], y=pod["prob"], line=dict(color=SIGNAL2, width=2.2),
                  fill="tozeroy", fillcolor="rgba(11,125,132,0.06)")
    f.add_hline(y=0.9, line=dict(color=AMBER, width=1, dash="dash"),
                annotation_text="90% POD", annotation_font_size=9, annotation_font_color=AMBER)
    f.update_layout(xaxis=dict(title="crack size [mm]", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="probability of detection", gridcolor=GRID, zeroline=False, range=[0, 1]))
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
    _acc = dmg.get("acceptance")
    _rows = [
        ("Deterministic life", f"{life(dmg['deterministic_life_years'])} yr"),
        ("Annual damage (time / spectral)", f"{dmg['annual_rate_time']:.2e}  /  {dmg['annual_rate_spectral']:.2e} /yr"),
        ("Remaining life P10 / P50 / P90", f"{life(post['p10'])} / {life(post['p50'])} / {life(post['p90'])} yr"),
    ]
    if _acc is not None:
        _env = {"in_air": "in air", "seawater_cp": "seawater w/ CP"}.get(
            dmg.get("sn_environment", "in_air"), "in air")
        _rows.append(("S-N environment", _env))
        _rows.append((
            "DFF acceptance",
            f"util {_acc['utilisation']:.2f}  (DFF {_acc['dff']:.0f} x {_acc['design_service_life_years']:.0f} yr)"
            f"  ->  {'PASS' if _acc['passes'] else 'FAIL'}",
        ))
    kv(_rows)
    pdf.ln(1)
    section("Decision")
    _net = f"${econ['fleet_delta_c_usd']/1e6:+.1f}M ({econ['n_units']}u, {econ['horizon_yr']:.0f}yr)"
    _phi_src = "posterior" if econ.get("phi_is_endogenous") else "break-even ref"
    kv([
        ("Next inspection", f"{insp['next_inspection_year']:.1f} yr  (target PoF {insp['target_pof']*100:.1f}%)"),
        ("Conditional net value dC", f"{_net}  ->  {'net gain' if econ.get('net_positive') else 'net cost'}"),
        ("Fleet phi / break-even phi*", f"{econ['phi']:.2f} ({_phi_src})  /  {econ['breakeven_phi']:.2f}"),
    ])
    pdf.ln(2)

    # Charts
    for png in (_mpl_fan(p["bayesian_fan"], post["p50"]), _mpl_spectra(p["spectrum"]), _mpl_hist(post)):
        pdf.image(io.BytesIO(png), w=185)
        pdf.ln(1)

    section("Validation gates (spec section 5)")
    for x in g:
        if x["passed"]:
            pdf.set_text_color(31, 138, 91)
        else:
            pdf.set_text_color(195, 61, 40)
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
        '<circle cx="12" cy="104" r="9" fill="none" stroke="#b07d1a" stroke-width="1"/>'
        '<path d="M12 104 C 46 104, 52 26, 120 16" fill="none" stroke="#16a6ac" stroke-width="2.6" stroke-linecap="round"/>'
        '<circle cx="12" cy="104" r="4.5" fill="#b07d1a"/><circle cx="120" cy="16" r="3.6" fill="#0e7c82"/></svg>'
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
    c = st.container() if MOBILE else st.columns([2, 1, 2])[1]
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
# Console layout: left-rail section nav + header + remaining-life hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      /* --- left-rail section navigation --- */
      .navtitle { font-size:10px; font-weight:600; letter-spacing:.14em; text-transform:uppercase;
        color:var(--muted); margin:2px 0 8px 2px; }
      .st-key-navsec [role="radiogroup"] { gap:3px; }
      .st-key-navsec [role="radiogroup"] > label { display:flex; align-items:center; padding:8px 12px;
        border-radius:8px; margin:0; border:1px solid transparent; cursor:pointer;
        transition:background .13s ease, border-color .13s ease; }
      .st-key-navsec [role="radiogroup"] > label:hover { background:var(--panel); }
      /* hide the radio control (first child is a span or div depending on Streamlit build) */
      .st-key-navsec [role="radiogroup"] > label > :first-child { display:none !important; }
      .st-key-navsec [role="radiogroup"] > label,
      .st-key-navsec [role="radiogroup"] > label * { font-size:13.5px !important; font-weight:500 !important;
        color:var(--sub) !important; }
      .st-key-navsec [role="radiogroup"] > label:has(input:checked),
      .st-key-navsec [role="radiogroup"] > label:has([aria-checked="true"]) {
        background:var(--panel); border-color:var(--line2); box-shadow:var(--shadow-sm); }
      .st-key-navsec [role="radiogroup"] > label:has(input:checked) *,
      .st-key-navsec [role="radiogroup"] > label:has([aria-checked="true"]) * {
        color:var(--accent) !important; font-weight:600 !important; }
      .navrule { border:none; border-top:1px solid var(--line); margin:14px 0 10px; }

      /* --- remaining-life hero --- */
      .hero { display:grid; grid-template-columns:1.25fr 2fr; gap:22px; background:var(--panel);
        border:1px solid var(--line); border-radius:var(--r); box-shadow:var(--shadow);
        padding:20px 24px; margin:4px 0 16px; }
      .hero-main { border-right:1px solid var(--line); padding-right:22px; }
      .hero-top { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
      .hero-lab { font-size:10.5px; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); }
      .hero-life { font-size:58px; font-weight:700; letter-spacing:-.035em; line-height:1;
        margin:6px 0 2px; font-variant-numeric:tabular-nums; color:var(--ink); }
      .hero-life.sig { color:var(--accent); } .hero-life.alarm { color:var(--alarm); }
      .hero-life span { font-size:20px; font-weight:500; color:var(--muted); margin-left:7px; }
      .gauge { position:relative; height:8px; background:var(--panel2); border-radius:5px; margin:14px 0 8px; }
      .gauge-fill { position:absolute; top:0; left:0; height:100%; border-radius:5px; background:var(--accent); }
      .gauge-fill.alarm { background:var(--alarm); }
      .gauge-req { position:absolute; top:-4px; width:2px; height:16px; background:var(--ink); }
      .gauge-cap { font-size:11px; color:var(--sub); }
      .hbadge { display:inline-block; font-size:11.5px; font-weight:600; padding:3px 12px; border-radius:100px; }
      .hbadge.pass { color:var(--good); background:color-mix(in srgb,var(--good) 12%,transparent); }
      .hbadge.fail { color:var(--alarm); background:color-mix(in srgb,var(--alarm) 12%,transparent); }
      .hero-stats { display:grid; grid-template-columns:repeat(2,1fr); gap:14px 26px; align-content:center; }
      .hstat .k { font-size:9.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }
      .hstat .v { font-size:23px; font-weight:600; letter-spacing:-.01em; font-variant-numeric:tabular-nums;
        color:var(--ink); margin-top:3px; }
      .hstat .v small { font-size:12px; color:var(--muted); font-weight:500; margin-left:3px; }
      .hstat .v.sig { color:var(--accent); } .hstat .v.amber { color:var(--amber); } .hstat .v.alarm { color:var(--alarm); }
      @media (max-width:820px){ .hero{ grid-template-columns:1fr; } .hero-main{ border-right:none;
        border-bottom:1px solid var(--line); padding-right:0; padding-bottom:16px; } }

      /* --- section title band --- */
      .sectionhead { display:flex; align-items:baseline; gap:12px; margin:4px 0 14px; }
      .sectionhead .sx { font-family:var(--mono); font-size:12px; color:var(--accent); font-weight:500; }
      .sectionhead .st { font-size:19px; font-weight:600; letter-spacing:-.01em; color:var(--ink); }
      .sectionhead .sd { font-size:12.5px; color:var(--muted); margin-left:auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV_SECTIONS = ["Overview", "Structure", "Environment", "Sensing", "Detection",
                "Assimilation", "Economics", "Ledger", "Provenance"]
st.sidebar.markdown('<div class="navtitle">Sections</div>', unsafe_allow_html=True)
_section = st.sidebar.radio("Section", NAV_SECTIONS, label_visibility="collapsed", key="navsec")
st.sidebar.markdown('<hr class="navrule"/>', unsafe_allow_html=True)


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
    synth["heading"] = st.sidebar.slider(
        "Wave heading [deg] (0 = head, 90 = beam)", 0.0, 90.0, 20.0, 5.0,
        help="Drives the 6-DOF mix: head seas -> pitch/heave/surge; beam -> roll/sway.")
    _rao_up = st.sidebar.file_uploader("Validated vessel RAO CSV (optional)", type=["csv"], key="rao_csv")
    rao_bytes: bytes | None = _rao_up.getvalue() if _rao_up is not None else None
    if rao_bytes:
        st.sidebar.caption("Motion built from your **validated RAO** x the wave spectrum (badged).")
    else:
        st.sidebar.caption("Columns `freq_hz, heave_mag, heave_phase_deg, pitch_mag, ...`. "
                           "Without one, the built-in **illustrative** RAOs are used.")
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
    scf = st.number_input("Detail SCF", 1.0, 5.0, ref.scf, 0.05)
    hi_lo_mm = st.number_input("Girth-weld hi-lo misalignment [mm]", 0.0, 5.0, 0.0, 0.5,
                               help="DNV-RP-C203 App.3: adds a physics-derived SCF to the detail SCF.")
    sn_class = st.selectbox("DNV S-N class", SN_CLASSES, index=SN_CLASSES.index(ref.sn_class))
    _sn_env_label = st.selectbox(
        "S-N environment", ["in air (Table 2-1)", "seawater w/ CP (Table 2-2)"], index=0,
        help="Seawater-with-cathodic-protection moves the slope change to 1e6 cycles "
             "(DNV-RP-C203 Table 2-2) - more severe in the wave band than the in-air curve.",
    )
    sn_env = SNEnvironment.SEAWATER_CP if _sn_env_label.startswith("seawater") else SNEnvironment.IN_AIR

with st.sidebar.expander("Acceptance & mean stress"):
    dff = st.number_input("Design Fatigue Factor (DFF)", 1.0, 10.0, 3.0, 1.0,
                          help="DNV-OS-F201: predicted life must exceed DFF x design service life.")
    design_life = st.number_input("Design service life [yr]", 5.0, 60.0, 25.0, 5.0)
    safety_class = st.selectbox("Safety class (reliability target)", ["low", "normal", "high"], index=1,
                                help="DNV-RP-C210 target annual Pf: low 1e-3, normal 1e-4, high 1e-5.")
    _ms_label = st.selectbox("Mean-stress model", ["none", "goodman", "gerber", "swt"], index=0,
                             help="Rides on the static axial tension mean (T_TDP/A_steel).")
    ms_model = MeanStressModel(_ms_label)
    as_welded = st.checkbox("As-welded (DNV: no mean-stress benefit)", value=True,
                            help="Uncheck for base-material / stress-relieved details to let the model act.")

with st.sidebar.expander("Hang-off geometry (6-DOF, Eq. 6)"):
    st.caption("Resolves 6-DOF MRU motion to the porch: z_ho = heave - x_p*pitch + y_p*roll.")
    porch_x = st.number_input("Porch offset x [m] (+fwd)", -100.0, 100.0, 20.0, 1.0)
    porch_y = st.number_input("Porch offset y [m] (+port)", -50.0, 50.0, 0.0, 1.0)
    porch_z = st.number_input("Porch offset z [m] (+up)", -50.0, 50.0, 25.0, 1.0)
    azimuth = st.number_input("Riser azimuth [deg]", -180.0, 180.0, 0.0, 5.0)
    exact_rot = st.checkbox("Exact finite-rotation (vs small-angle Eq. 6)", value=False)

with st.sidebar.expander("Transfer function (Layer 1)", expanded=True):
    route = st.radio(
        "H(f) route", ["reference", "analytic", "imported"], horizontal=True,
        help="imported = validated vendor H(f) (OrcaFlex/RIFLEX/DeepLines). "
             "reference = illustrative table (NOT data). analytic = reduced-order Route-1.",
    )
    transfer_bytes: bytes | None = None
    if route == "imported":
        hf_up = st.file_uploader("Validated H(f) CSV", type=["csv"], key="hf_csv")
        if hf_up is not None:
            transfer_bytes = hf_up.getvalue()
        st.caption("Complex TDP moment transfer. Columns `freq_hz, magnitude, phase_rad` "
                   "(or `freq, re, im`); `# key: value` header lines carry provenance. "
                   "Example: `data/samples/example_transfer_function.csv`.")
    elif route == "reference":
        st.caption("Illustrative wave-band table - realistic magnitude but **not project data**.")
    else:
        st.caption("Reduced-order Morison model - a documented engineering approximation.")

with st.sidebar.expander("Current & VIV (DNV-RP-F204)"):
    viv_current = st.slider("Surface current [m/s]", 0.0, 3.0, 0.6, 0.1,
                            help="Sheared current driving cross-flow VIV. 0 disables VIV.")
    viv_damping = st.slider("Damping ratio", 0.005, 0.10, 0.02, 0.005)
    mg_thk_mm = st.slider("Marine growth thickness [mm]", 0.0, 150.0, 0.0, 10.0,
                          help="DNV-RP-C205: biofouling adds hydro diameter + mass, worsening VIV.")
    st.caption("VIV screening (Griffin A/D + lock-in). **Screening upper bound**, not design-grade.")

with st.sidebar.expander("Long-term wave climate (scatter)"):
    scatter_bytes: bytes | None = None
    sc_up = st.file_uploader("Scatter-diagram CSV", type=["csv"], key="scatter_csv")
    if sc_up is not None:
        scatter_bytes = sc_up.getvalue()
        st.caption("Loaded a project scatter table.")
    else:
        st.caption("Using an **illustrative** deep-water climate. Columns `Hs, Tp, prob` "
                   "(occurrence counts, fractions or %). Drives the long-term D = &Sigma; p&#8202;D fatigue.")

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
            hang_off_angle_deg=ang, scf=scf, hi_lo_misalignment=hi_lo_mm / 1e3, sn_class=sn_class,
            sn_environment=sn_env, design_fatigue_factor=dff,
            design_service_life_years=design_life, safety_class=safety_class,
            mean_stress_model=ms_model, as_welded=as_welded,
            contents_density=ref.contents_density, coating_thickness=ref.coating_thickness,
            coating_density=ref.coating_density, is_reference_preset=False,
        ),
        transfer=TransferConfig(route=route),
        hang_off=HangOffConfig(porch_x=porch_x, porch_y=porch_y, porch_z=porch_z,
                               riser_azimuth_deg=azimuth, exact_rotation=exact_rot),
        environment=EnvironmentConfig(enabled=env_on, temperature_factor=tfac, salinity_factor=sfac),
        viv=VivConfig(surface_current=viv_current, damping_ratio=viv_damping,
                      marine_growth_thickness=mg_thk_mm / 1e3),
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
_r = cfg.riser
_env_now = {"in_air": "in air", "seawater_cp": "seawater/CP"}.get(str(_r.sn_environment.value), "in air")
st.markdown(
    '<div class="titleblock">'
    '<div><div class="tb-name">SCR&middot;TWIN</div>'
    '<div class="tb-sub">TDP fatigue integrity digital twin</div></div>'
    f'<div><div class="tb-k">Riser section</div><div class="tb-v">{_r.outer_diameter*1e3:.0f}&times;{_r.wall_thickness*1e3:.1f} mm</div>'
    f'<div class="tb-k" style="margin-top:5px">grade</div><div class="tb-v">{_r.material_grade}</div></div>'
    f'<div><div class="tb-k">Water depth</div><div class="tb-v">{_r.water_depth:.0f} m</div>'
    f'<div class="tb-k" style="margin-top:5px">hang-off</div><div class="tb-v">{_r.hang_off_angle_deg:.0f}&deg; f/vert</div></div>'
    f'<div><div class="tb-k">S-N / SCF</div><div class="tb-v">DNV {_r.sn_class} &middot; {_r.scf:.2f}</div>'
    f'<div class="tb-k" style="margin-top:5px">environment</div><div class="tb-v">{_env_now}</div></div>'
    f'<div><div class="tb-k">DFF &middot; design life</div>'
    f'<div class="tb-v">{_r.design_fatigue_factor:.0f}&times; &middot; {_r.design_service_life_years:.0f} yr</div>'
    f'<div class="tb-k" style="margin-top:5px">gates &middot; source</div>'
    f'<div class="tb-v {"sig" if gates_ok==len(g) else "fail"}">{gates_ok}/{len(g)} &middot; {"SYN" if is_synth else "MRU"}</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

run_col, _ = dcols([1, 3])
run_clicked = run_col.button("Run analysis", type="primary", width="stretch",
                             help="Runs the full chain with a live, animated acquisition + posterior.")

# --------------------------------------------------------------------------- #
# Compute (cached, deterministic)
# --------------------------------------------------------------------------- #
if is_synth:
    payload = analyze_synthetic(cfg.model_dump_json(), synth["hs"], synth["tp"], synth["gamma"],
                                synth["duration"], synth["fs"], int(synth["seed"]),
                                synth["heading"], transfer_bytes, scatter_bytes, rao_bytes)
elif upload_bytes is not None:
    payload = analyze_upload(cfg.model_dump_json(), upload_bytes, transfer_bytes, scatter_bytes)
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

    status.markdown('<div class="livestatus" style="color:#2f855a">&#10003; Analysis complete</div>',
                    unsafe_allow_html=True)
    prog.progress(100)
    time.sleep(0.4)
    holder.empty()


if run_clicked:
    st.session_state.ran = True
    run_live(payload)

if not st.session_state.ran:
    st.markdown(
        '<div style="margin-top:26px;text-align:center;color:#586a71">'
        '<div style="font-size:15px;color:#17242b">Ready.</div>'
        '<div style="font-size:12.5px;margin-top:4px">Set the sea state and riser configuration in the sidebar, '
        'then press <b>Run analysis</b> for a live, animated run.</div></div>',
        unsafe_allow_html=True)
    st.stop()


# --------------------------------------------------------------------------- #
# Dashboard (static result; also live-updates as you edit the sidebar)
# --------------------------------------------------------------------------- #
dmg, post, insp, econ, prov = (
    payload["damage"], payload["posterior"], payload["inspection"], payload["economics"], payload["provenance"],
)
sea, env = payload["sea_state"], payload["environment"]

_acc = dmg.get("acceptance")
_cat = payload.get("catenary")
_tf = payload["transfer"]
_dof = payload.get("dof_contributions", {"heave": 1.0})
_ver = payload.get("verification")
_lt = payload.get("long_term")
_viv = payload.get("viv")
_comb = payload.get("combined")
_crack = payload.get("crack")
_rel = payload.get("reliability")
_seabed = payload.get("seabed")
_dfan = payload.get("divergence_fan")
_comb_life = _comb["life_years"] if _comb else dmg["deterministic_life_years"]

# --------------------------------------------------------------------------- #
# Remaining-life hero (always visible; section content is gated by the nav rail)
# --------------------------------------------------------------------------- #
_req_life = _r.design_fatigue_factor * _r.design_service_life_years
_p50, _p10 = post["p50"], post["p10"]
_passes = _acc["passes"] if _acc else (_p50 >= _req_life)
_util = _acc["utilisation"] if _acc else (_req_life / _p50 if _p50 else 0.0)
_gmax = max(_p50, _req_life, 1e-6) * 1.2
_fill = max(2.0, min(100.0, 100.0 * _p50 / _gmax))
_reqx = max(0.0, min(100.0, 100.0 * _req_life / _gmax))
_tone = "sig" if _passes else "alarm"
st.markdown(
    '<div class="hero"><div class="hero-main">'
    '<div class="hero-top"><span class="hero-lab">Remaining fatigue life &middot; P50</span>'
    f'<span class="hbadge {"pass" if _passes else "fail"}">{"ACCEPTABLE" if _passes else "BELOW TARGET"}</span></div>'
    f'<div class="hero-life {_tone}">{life(_p50)}<span>yr</span></div>'
    f'<div class="gauge"><div class="gauge-fill {_tone}" style="width:{_fill:.1f}%"></div>'
    f'<div class="gauge-req" style="left:{_reqx:.1f}%"></div></div>'
    f'<div class="gauge-cap">required (DFF &times; design) = {life(_req_life)} yr'
    f' &middot; utilisation {_util:.2f} &middot; DFF {_r.design_fatigue_factor:.0f}&times;</div>'
    '</div><div class="hero-stats">'
    f'<div class="hstat"><div class="k">Deterministic life</div><div class="v sig">{life(dmg["deterministic_life_years"])}<small>yr</small></div></div>'
    f'<div class="hstat"><div class="k">Combined wave + VIV</div><div class="v">{life(_comb_life)}<small>yr</small></div></div>'
    f'<div class="hstat"><div class="k">P10 conservative</div><div class="v amber">{life(_p10)}<small>yr</small></div></div>'
    f'<div class="hstat"><div class="k">Next inspection</div><div class="v sig">{insp["next_inspection_year"]:.1f}<small>yr</small></div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ========================== OVERVIEW ======================================= #
if _section == "Overview":
    st.markdown('<div class="sectionhead"><span class="sx">01</span>'
                '<span class="st">Acceptance &amp; headline result</span>'
                '<span class="sd">DNV-OS-F201 design-fatigue-factor check</span></div>',
                unsafe_allow_html=True)
    if _acc is not None:
        _env_label = {"in_air": "in air", "seawater_cp": "seawater w/ CP"}.get(
            dmg.get("sn_environment", "in_air"), dmg.get("sn_environment", "in_air"))
        _pass = _acc["passes"]
        st.markdown(kpi_row([
            kpi("S-N environment", _env_label),
            kpi("Design Fatigue Factor", f'{_acc["dff"]:.0f}', "x"),
            kpi("DFF utilisation", f'{_acc["utilisation"]:.2f}', "", "sig" if _pass else "alarm"),
            kpi("Acceptance", "PASS" if _pass else "FAIL", "", "sig" if _pass else "alarm"),
            kpi("Long-term (scatter) life", life(_lt["life_years"]) if _lt else "-", "yr"),
        ]), unsafe_allow_html=True)
    if _cat is not None:
        st.markdown('<div class="sec" data-n="A">System configuration &middot; SCR side elevation '
                    '(vessel &rarr; catenary &rarr; touchdown)</div>', unsafe_allow_html=True)
        components.html(
            f'<div style="width:100%;background:#fff">{system_schematic_svg(payload, cfg.riser)}</div>',
            height=720, scrolling=False,
        )
    st.markdown('<div class="sec" data-n="B">Digital-twin architecture &middot; processing chain</div>',
                unsafe_allow_html=True)
    components.html(f'<div style="width:100%;background:#fff">{architecture_svg()}</div>',
                    height=316, scrolling=False)

# ========================== STRUCTURE ====================================== #
if _section == "Structure":
    st.caption("Riser geometry and structural response: the solved catenary and the "
               "cross-flow modal shapes that carry VIV.")
    st.markdown('<div class="eq">y(x) = a&#183;(cosh(x/a) &minus; 1),&nbsp; a = H/w,&nbsp; '
                '&#954;(x) = 1/(a&#183;cosh&#178;(x/a)),&nbsp; &#963;<sub>bend</sub> = SCF&#183;E&#183;(D/2)&#183;&#954; '
                '<span class="c"># closed-form catenary + outer-fibre bending</span></div>',
                unsafe_allow_html=True)
    if _cat is not None:
        st.markdown('<div class="sec" data-n="01">Static catenary configuration &middot; riser shape &amp; touchdown</div>',
                    unsafe_allow_html=True)
        gc1, gc2 = dcols([3, 2])
        gc1.plotly_chart(catenary_fig(_cat), width="stretch", config={"displayModeBar": False})
        with gc2:
            st.markdown(kpi_row([
                kpi("Catenary parameter a", f'{_cat["catenary_parameter"]:.0f}', "m"),
                kpi("Horizontal span", f'{_cat["horizontal_span"]:.0f}', "m"),
            ]), unsafe_allow_html=True)
            st.markdown(kpi_row([
                kpi("Arc length", f'{_cat["arc_length"]:.0f}', "m"),
                kpi("TDP curvature", f'{_cat["tdp_curvature"]*1e3:.3f}', "1/km", "amber"),
            ]), unsafe_allow_html=True)
            st.caption("Closed-form catenary y(x)=a(cosh(x/a)-1); kappa_TDP = 1/a = w/H.")
        st.markdown('<div class="sec" data-n="02">Static bending-stress distribution along the riser</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            along_riser_stress_fig(_cat, cfg.riser.youngs_modulus, cfg.riser.outer_diameter, cfg.riser.scf),
            width="stretch", config={"displayModeBar": False})
        st.caption("Outer-fibre bending stress sigma=SCF·E·(D/2)·kappa(s) peaks at the touchdown "
                   "point - the physical reason SCR fatigue localises there.")
        _sc = cfg.riser.pipe_section()
        _w = cfg.riser.effective_submerged_weight()
        st.markdown('<div class="sec" data-n="03">Pipe section properties &middot; derived geometry</div>',
                    unsafe_allow_html=True)
        pp1, pp2 = dcols([1, 1])
        with pp1:
            st.markdown(data_table(
                ["Property", "Value", "Unit"],
                [["Outer diameter D", f"{cfg.riser.outer_diameter*1e3:.1f}", "mm"],
                 ["Wall thickness t", f"{cfg.riser.wall_thickness*1e3:.2f}", "mm"],
                 ["Inner diameter", f"{_sc.inner_diameter*1e3:.1f}", "mm"],
                 ["Steel area A", f"{_sc.steel_area*1e4:.1f}", "cm²"],
                 ["2nd moment I", f"{_sc.second_moment_area*1e8:.1f}", "cm⁴"],
                 ["Section modulus Z", f"{_sc.section_modulus*1e6:.1f}", "cm³"]],
                value_cols=(1,)), unsafe_allow_html=True)
        with pp2:
            st.markdown(data_table(
                ["Property", "Value", "Unit"],
                [["Bending stiffness EI", f"{_sc.bending_stiffness/1e6:.1f}", "MN·m²"],
                 ["Young's modulus E", f"{cfg.riser.youngs_modulus/1e9:.0f}", "GPa"],
                 ["Submerged weight w", f"{_w:.0f}", "N/m"],
                 ["Horizontal tension H", f"{_w*_cat['catenary_parameter']/1e3:.0f}", "kN"],
                 ["Top tension", f"{_w*_cat['catenary_parameter']*np.cosh(_cat['horizontal_span']/_cat['catenary_parameter'])/1e3:.0f}", "kN"],
                 ["Contents density", f"{cfg.riser.contents_density:.0f}", "kg/m³"]],
                value_cols=(1,)), unsafe_allow_html=True)
        st.markdown('<div class="sec" data-n="04">Effective-tension distribution along the riser</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(riser_tension_fig(_cat, _w), width="stretch", config={"displayModeBar": False})
        st.caption("T(s) = H·cosh(x/a) rises from the horizontal tension H at the TDP to the top "
                   "tension at hang-off; the axial mean stress T/A rides under the dynamic bending.")
    if _viv is not None and _viv.get("enabled"):
        st.markdown('<div class="sec" data-n="05">Cross-flow modal response &middot; tensioned-beam modes</div>',
                    unsafe_allow_html=True)
        vm1, vm2 = dcols([1, 1])
        vm1.plotly_chart(viv_mode_fig(_viv), width="stretch", config={"displayModeBar": False})
        vm1.caption(f"Dominant excited cross-flow mode {_viv['dominant_mode']} standing wave "
                    "(real tensioned-beam eigensolve).")
        vm2.plotly_chart(viv_vr_fig(_viv), width="stretch", config={"displayModeBar": False})
        vm2.caption("Reduced velocity per mode; amber = inside the lock-in band, i.e. excited.")
        _exc = [m for m in _viv["modes"] if m["excited"]]
        if _exc:
            st.markdown('<div class="sec" data-n="06">Excited-mode table &middot; lock-in cross-flow modes</div>',
                        unsafe_allow_html=True)
            st.markdown(data_table(
                ["Mode", "fn [Hz]", "Vr", "A/D", "Δσ [MPa]", "damage /yr"],
                [[str(m["mode"]), f"{m['frequency_hz']:.3f}", f"{m['reduced_velocity']:.1f}",
                  f"{m['a_over_d']:.2f}", f"{m['stress_range_mpa']:.1f}", f"{m['annual_damage_rate']:.2e}"]
                 for m in _exc[:10]], value_cols=(1, 2, 3, 4, 5)), unsafe_allow_html=True)
    if _seabed is not None and _seabed.get("enabled"):
        st.markdown('<div class="eq">&#955;<sub>b</sub> = &#8730;(EI/H),&nbsp; '
                    'l<sub>s</sub> = (4EI/k<sub>v</sub>)<sup>1/4</sup>,&nbsp; '
                    'C<sub>s</sub> = &#955;<sub>b</sub>/(&#955;<sub>b</sub>+l<sub>s</sub>) '
                    '<span class="c"># Pesce/Lenci TDP boundary-layer correction</span></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sec" data-n="07">Seabed-stiffness sensitivity &middot; touchdown interaction '
                    '<span class="tag amber">rigid base is conservative</span></div>', unsafe_allow_html=True)
        sb1, sb2 = dcols([3, 2])
        sb1.plotly_chart(seabed_fig(_seabed), width="stretch", config={"displayModeBar": False})
        with sb2:
            st.markdown(kpi_row([
                kpi("Rigid-seabed life (base)", life(_seabed["base_life_years"]), "yr", "alarm"),
                kpi("Bending layer λ_b", f'{_seabed["lambda_b"]:.1f}', "m"),
            ]), unsafe_allow_html=True)
            st.markdown(kpi_row([
                kpi("Soft-clay life", life(_seabed["life_soft"]), "yr", "sig"),
                kpi("Stiff-sand life", life(_seabed["life_stiff"]), "yr"),
            ]), unsafe_allow_html=True)
            st.caption("A compliant seabed distributes the TDP curvature (Cs<1), so the rigid-seabed "
                       "life is a conservative lower bound - the band shows the soft-to-stiff span. "
                       "TDP fatigue is very sensitive to seabed modelling; a project value of k_v "
                       "(or a nonlinear soil model) narrows it.")

# ========================== ENVIRONMENT ==================================== #
if _section == "Environment":
    st.caption("The metocean loading: the identified sea state, the spectra, and the "
               "long-term wave scatter climate that drives fatigue.")
    st.markdown(kpi_row([
        kpi("Sig. heave Hm0", f'{sea["hs"]:.2f}', "m"),
        kpi("Tp", f'{sea["tp"]:.1f}', "s"),
        kpi("Tz", f'{sea["tz"]:.1f}', "s"),
        kpi("gamma fit", f'{sea["gamma"]:.1f}'),
        kpi("Env. capacity factor", f'{env["factor"]:.3f}' if env["enabled"] else "-", "", "amber"),
    ]), unsafe_allow_html=True)
    st.markdown('<div class="eq">S(f) = &#945; g&#178; (2&#960;)<sup>&minus;4</sup> f<sup>&minus;5</sup> '
                'exp[&minus;1.25(f<sub>p</sub>/f)&#8308;] &#183; &#947;<sup>r</sup>&nbsp;&nbsp;'
                'H<sub>s</sub> = 4&#8730;m&#8320; <span class="c"># JONSWAP wave spectrum</span></div>',
                unsafe_allow_html=True)
    ev1, ev2 = dcols([1, 1])
    ev1.markdown('<div class="sec" data-n="01">Identified JONSWAP wave spectrum</div>', unsafe_allow_html=True)
    ev1.plotly_chart(jonswap_wave_fig(sea), width="stretch", config={"displayModeBar": False})
    ev2.markdown('<div class="sec" data-n="02">Response PSD &middot; motion &amp; stress</div>', unsafe_allow_html=True)
    ev2.plotly_chart(spectra_fig(payload["spectrum"]), width="stretch", config={"displayModeBar": False})
    if env["enabled"]:
        st.markdown('<div class="sec" data-n="03">Arabian-Gulf environmental knock-down</div>', unsafe_allow_html=True)
        st.markdown(kpi_row([
            kpi("Temperature factor", f'{env.get("temperature_factor", 0):.3f}'),
            kpi("Salinity factor", f'{env.get("salinity_factor", 0):.3f}'),
            kpi("Combined capacity factor", f'{env["factor"]:.3f}', "", "amber"),
            kpi("Life reduction", f'{(1-env["factor"])*100:.0f}', "%", "alarm"),
        ]), unsafe_allow_html=True)
        st.caption("Temperature + salinity severity multiply the S-N capacity (DNV-RP-C203); the "
                   "combined factor F scales the fatigue life.")
    if _viv is not None and _viv.get("enabled"):
        st.markdown('<div class="sec" data-n="04">Sheared current profile (drives VIV)</div>', unsafe_allow_html=True)
        cv1, cv2 = dcols([2, 3])
        cv1.plotly_chart(current_profile_fig(_viv), width="stretch", config={"displayModeBar": False})
        cv2.caption("Power-law current U(h)=U_s·(h/d)^(1/7) (DNV-RP-C205). The current sets the "
                    "vortex-shedding frequency and the reduced velocity that drives cross-flow VIV "
                    "lock-in (see the Detection tab).")
        _mg = _viv.get("marine_growth", {})
        if _mg.get("enabled"):
            cv2.markdown(kpi_row([
                kpi("Marine growth", f'{_mg["thickness_mm"]:.0f}', "mm", "amber"),
                kpi("Effective diameter", f'{_mg["effective_diameter_mm"]:.0f}', "mm"),
                kpi("Added mass", f'{_mg["mass_per_length"]:.0f}', "kg/m"),
            ]), unsafe_allow_html=True)
            cv2.caption(
                f'DNV-RP-C205 biofouling enlarges the hydrodynamic diameter '
                f'({_mg["base_diameter_mm"]:.0f} -> {_mg["effective_diameter_mm"]:.0f} mm, '
                f'D_eff = D + 2·t) and mass - shifting the vortex-shedding frequency and '
                f'worsening VIV.')
    if _lt is not None:
        st.markdown('<div class="sec" data-n="05">Long-term fatigue &middot; wave scatter-diagram summation '
                    '(DNV-RP-C203 &sect;5) &middot; D = &Sigma; p&#8202;D</div>', unsafe_allow_html=True)
        lt1, lt2 = dcols([3, 2])
        lt1.plotly_chart(scatter_fig(_lt), width="stretch", config={"displayModeBar": False})
        with lt2:
            _top = _lt["contributions"][0] if _lt["contributions"] else None
            st.markdown(kpi_row([
                kpi("Long-term life", life(_lt["life_years"]), "yr", "sig"),
                kpi("Sea-state cells", f'{_lt["n_cells"]}'),
            ]), unsafe_allow_html=True)
            if _top is not None:
                st.markdown(kpi_row([
                    kpi("Top driver cell", f'Hs {_top["hs"]:.1f} / Tp {_top["tp"]:.0f}', "m/s", "amber"),
                    kpi("its damage share", f'{100*_top["damage_fraction"]:.0f}', "%", "amber"),
                ]), unsafe_allow_html=True)
            st.caption(f"Damage summed over {_lt['n_cells']} sea states ({_lt['source']}).")
        st.markdown('<div class="sec" data-n="06">Fatigue-driver cells &middot; ranked by damage share</div>',
                    unsafe_allow_html=True)
        st.markdown(data_table(
            ["Hs [m]", "Tp [s]", "occurrence", "rate /yr", "damage share"],
            [[f"{c['hs']:.2f}", f"{c['tp']:.1f}", f"{100*c['probability']:.2f}%",
              f"{c['annual_rate']:.2e}", f"{100*c['damage_fraction']:.1f}%"]
             for c in _lt["contributions"][:12]], value_cols=(0, 1, 2, 3, 4)), unsafe_allow_html=True)

# ========================== SENSING ======================================== #
if _section == "Sensing":
    st.caption("From the vessel MRU recording to the touchdown stress: 6-DOF hang-off "
               "resolution and the motion&rarr;stress transfer function.")
    _mot = payload.get("motion", {})
    _mbadge = ('<span class="tag pass">VALIDATED motion</span>' if _mot.get("is_validated")
               else '<span class="tag syn">ILLUSTRATIVE motion</span>')
    st.markdown(f'<div class="sec">MRU hang-off motion &middot; {_mbadge} '
                f'<span class="foot">{_mot.get("source", "")}</span></div>', unsafe_allow_html=True)
    tf_trace = _fig(200)
    tf_trace.add_scatter(x=payload["trace"]["time"], y=payload["trace"]["heave"],
                         line=dict(color=SIGNAL, width=1))
    tf_trace.update_layout(xaxis=dict(title="t [s]", gridcolor=GRID, zeroline=False),
                           yaxis=dict(title="heave [m]", gridcolor=GRID, zeroline=False))
    st.plotly_chart(tf_trace, width="stretch", config={"displayModeBar": False})
    if len(_dof) > 1:
        st.markdown('<div class="sec">6-DOF hang-off resolution (Eq. 6) &middot; which DOF drives TDP fatigue</div>',
                    unsafe_allow_html=True)
        kc1, kc2 = dcols([3, 2])
        kc1.plotly_chart(dof_fig(_dof), width="stretch", config={"displayModeBar": False})
        with kc2:
            _topdof = max(_dof.items(), key=lambda kv: kv[1])
            st.markdown(kpi_row([
                kpi("Porch offset x", f'{cfg.hang_off.porch_x:.0f}', "m"),
                kpi("Dominant DOF", _topdof[0].upper(), f'{100*_topdof[1]:.0f}%', "amber"),
            ]), unsafe_allow_html=True)
            st.caption("z_ho = heave - x_p*pitch + y_p*roll (small-angle Eq. 6).")
    st.markdown('<div class="sec">Transfer function H(f) &middot; MRU motion &rarr; TDP stress</div>',
                unsafe_allow_html=True)
    _prov = _tf.get("provenance", {})
    if _tf["is_validated"]:
        _badge = '<span class="tag pass">VALIDATED (project)</span>'
        _src = f' &nbsp;<span class="foot">route: imported &middot; {_prov.get("source_tool", "")}</span>'
    else:
        _badge = '<span class="tag syn">ILLUSTRATIVE / approximate - NOT project data</span>'
        _src = f' &nbsp;<span class="foot">route: {_tf["route"]}</span>'
    st.markdown(f'<div style="margin:-2px 0 8px">{_badge}{_src}</div>', unsafe_allow_html=True)
    hm1, hm2 = dcols([1, 1])
    hm1.markdown('<div class="sec">magnitude |H(f)|</div>', unsafe_allow_html=True)
    hm1.plotly_chart(transfer_fig(_tf), width="stretch", config={"displayModeBar": False})
    hm2.markdown('<div class="sec">phase ∠H(f)</div>', unsafe_allow_html=True)
    hm2.plotly_chart(hf_phase_fig(_tf), width="stretch", config={"displayModeBar": False})
    _peak = max(_tf["stress_mag"]) if _tf["stress_mag"] else 0.0
    _ipk = _tf["stress_mag"].index(_peak) if _peak else 0
    st.markdown(kpi_row([
        kpi("Peak |H|", f"{_peak:.1f}", "MPa/m", "sig"),
        kpi("at frequency", f'{_tf["freq"][_ipk]:.3f}', "Hz"),
        kpi("route", _tf["route"]),
        kpi("validated", "YES" if _tf["is_validated"] else "NO", "", "sig" if _tf["is_validated"] else "amber"),
    ]), unsafe_allow_html=True)
    if not _tf["is_validated"]:
        st.caption("Import a validated OrcaFlex/RIFLEX/DeepLines H(f) (sidebar) for a "
                   "project-grade TDP stress transfer; the magnitude AND phase are both applied.")
    st.markdown('<div class="sec">Hot-spot SCF &middot; detail &times; misalignment (DNV-RP-C203 App.3)</div>',
                unsafe_allow_html=True)
    st.markdown(kpi_row([
        kpi("Detail (geometric) SCF", f'{prov.get("geometric_scf", cfg.riser.scf):.3f}'),
        kpi("Misalignment SCF", f'{prov.get("misalignment_scf", 1.0):.3f}',
            "", "amber" if prov.get("misalignment_scf", 1.0) > 1.001 else ""),
        kpi("Hi-lo eccentricity", f'{cfg.riser.hi_lo_misalignment*1e3:.1f}', "mm"),
        kpi("Effective hot-spot SCF", f'{prov.get("effective_scf", cfg.riser.scf):.3f}', "", "sig"),
    ]), unsafe_allow_html=True)
    st.caption("SCF_eff = detail SCF x [1 + 3(δ_m/t)·exp(−√(t/D))] - the hi-lo misalignment part is "
               "derived from the fabrication tolerance, not assumed.")
    _dh = payload.get("data_health")
    if _dh:
        st.markdown('<div class="sec">Data-health checks (ingest gate)</div>', unsafe_allow_html=True)
        st.markdown(kpi_row([
            kpi("Health", "OK" if _dh.get("ok") else "FAIL", "", "sig" if _dh.get("ok") else "alarm"),
        ] + [kpi(str(k), f"{v:.3g}" if isinstance(v, (int, float)) else str(v))
             for k, v in list(_dh.items())[:4] if k not in ("ok", "flags")]),
            unsafe_allow_html=True)

# ========================== DETECTION ====================================== #
if _section == "Detection":
    st.caption("Damage detection: rainflow + S-N + Miner, the spectral cross-check, and "
               "the VIV screening - the mechanisms that consume fatigue life.")
    st.markdown('<div class="eq">D = &#8721;<sub>i</sub> n<sub>i</sub>/N(&#916;&#963;<sub>i</sub>),&nbsp; '
                'log N = log a&#772; &minus; m&#183;log&#916;&#963;,&nbsp; life = 1/(D&#183;f<sub>yr</sub>) '
                '<span class="c"># Palmgren-Miner on the two-slope DNV S-N curve</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec" data-n="01">Rainflow &middot; S-N &middot; Miner (DNV-RP-C203 / ASTM E1049)</div>',
                unsafe_allow_html=True)
    st.markdown(kpi_row([
        kpi("Annual damage (time)", f'{dmg["annual_rate_time"]:.2e}', "/yr", "amber"),
        kpi("Spectral (Dirlik)", f'{dmg["annual_rate_spectral"]:.2e}', "/yr"),
        kpi("time/spectral ratio", f'{(dmg["annual_rate_spectral"]/dmg["annual_rate_time"]):.2f}'
            if dmg["annual_rate_time"] else "-", "", "sig"),
        kpi("Block damage", f'{dmg["block_damage"]:.2e}'),
        kpi("S-N class", cfg.riser.sn_class),
    ]), unsafe_allow_html=True)
    l2a, l2b = dcols([1, 1])
    l2a.markdown('<div class="sec" data-n="02">DNV S-N curve family</div>', unsafe_allow_html=True)
    l2a.plotly_chart(sn_family_fig(cfg.riser.sn_class, cfg.riser.sn_environment),
                     width="stretch", config={"displayModeBar": False})
    if _ver is not None:
        l2b.markdown('<div class="sec" data-n="03">Rainflow range histogram (counted cycles)</div>',
                     unsafe_allow_html=True)
        l2b.plotly_chart(rainflow_hist_fig(_ver), width="stretch", config={"displayModeBar": False})
    if _ver is not None:
        st.markdown('<div class="sec" data-n="04">Spectral cross-check &middot; Dirlik / narrow-band vs rainflow</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(dirlik_verify_fig(_ver), width="stretch", config={"displayModeBar": False})
        st.caption("The spectral (Dirlik) and time-domain (rainflow) pathways use the SAME S-N law "
                   "and agree within ~15% - the twin's internal fatigue verification.")
    st.markdown('<div class="sec" data-n="05">Mean-stress &amp; thickness corrections applied</div>',
                unsafe_allow_html=True)
    st.markdown(kpi_row([
        kpi("Mean-stress model", str(prov.get("mean_stress_model", "none"))),
        kpi("Applied", "YES" if prov.get("mean_stress_applied") else "NO", "",
            "amber" if prov.get("mean_stress_applied") else ""),
        kpi("Static mean stress", f'{prov.get("static_mean_stress_pa", 0)/1e6:.1f}', "MPa"),
        kpi("Thickness t_ref", f'{cfg.riser.thickness_for_correction*1e3:.1f}', "mm"),
    ]), unsafe_allow_html=True)
    if _viv is not None and _viv.get("enabled"):
        st.markdown('<div class="sec" data-n="06">Vortex-induced vibration (VIV) &middot; '
                    'DNV-RP-F204 <span class="tag amber">SCREENING</span></div>', unsafe_allow_html=True)
        st.markdown(kpi_row([
            kpi("Combined wave+VIV life", life(_comb_life), "yr", "sig"),
            kpi("VIV-only life", life(_viv["life_years"]), "yr", "amber"),
            kpi("Wave rate /yr", f'{_comb["wave_rate"]:.2e}' if _comb else "-"),
            kpi("VIV rate /yr", f'{_comb["viv_rate"]:.2e}' if _comb else "-", "", "amber"),
            kpi("Stability param Ks", f'{_viv["stability_parameter"]:.2f}'),
        ]), unsafe_allow_html=True)
        _excd = [m for m in _viv["modes"] if m["excited"]]
        if _excd:
            st.markdown(data_table(
                ["Mode", "fn [Hz]", "Vr", "A/D", "Δσ [MPa]", "damage /yr"],
                [[str(m["mode"]), f"{m['frequency_hz']:.3f}", f"{m['reduced_velocity']:.1f}",
                  f"{m['a_over_d']:.2f}", f"{m['stress_range_mpa']:.1f}", f"{m['annual_damage_rate']:.2e}"]
                 for m in _excd[:10]], value_cols=(1, 2, 3, 4, 5)), unsafe_allow_html=True)
        st.caption("Combined life adds the wave and VIV damage rates by Miner. VIV is a Griffin "
                   "A/D lock-in upper bound - design-grade VIV needs Shear7 / VIVANA.")
        st.markdown('<div class="sec" data-n="07">Marine-growth VIV knock-down '
                    '<span class="tag amber">live sweep</span></div>', unsafe_allow_html=True)
        _kd_cfg = cfg.model_copy(update={
            "viv": cfg.viv.model_copy(update={"marine_growth_thickness": 0.0})})
        _sweep = viv_knockdown(_kd_cfg.model_dump_json())
        _cur_mm = cfg.viv.marine_growth_thickness * 1e3
        if _sweep.get("enabled") and _sweep["thickness_mm"]:
            kd1, kd2 = dcols([3, 2])
            kd1.plotly_chart(knockdown_fig(_sweep, _cur_mm), width="stretch",
                             config={"displayModeBar": False})
            _clean_life = _sweep["life_years"][0]
            _cur_life = float(np.interp(_cur_mm, _sweep["thickness_mm"], _sweep["life_years"]))
            _dl = _sweep.get("design_life_years") or 0.0
            kd2.markdown(kpi_row([
                kpi("At current growth", life(_cur_life), "yr",
                    "alarm" if (_dl and _cur_life < _dl) else "sig"),
                kpi("Clean-riser life", life(_clean_life), "yr", "sig"),
                kpi("Growth setting", f'{_cur_mm:.0f}', "mm", "amber"),
                kpi("Knock-down", f'{_clean_life / _cur_life:.1f}' if _cur_life else "-", "x",
                    "amber"),
            ]), unsafe_allow_html=True)
            kd2.caption("VIV screening life as biofouling thickens: DNV-RP-C205 grows the "
                        "hydrodynamic diameter and added mass, which feed the DNV-RP-F204 lock-in "
                        "screen. The marker is the current growth setting - drag the sidebar "
                        "'Marine growth thickness' slider to move it along the curve.")
    if _crack is not None and _crack.get("enabled"):
        st.markdown('<div class="eq">da/dN = C(&#916;K)<sup>m</sup>,&nbsp; '
                    '&#916;K = Y&#183;&#916;&#963;&#183;&#8730;(&#960;a) '
                    '<span class="c"># BS 7910 Paris-law crack growth (parallel to S-N)</span></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sec" data-n="08">Fracture mechanics &middot; Paris-law crack growth '
                    '(BS 7910) <span class="tag amber">conservative ECA</span></div>', unsafe_allow_html=True)
        fr1, fr2 = dcols([1, 1])
        fr1.plotly_chart(crack_growth_ui_fig(_crack), width="stretch", config={"displayModeBar": False})
        fr1.caption(f"A {_crack['initial_flaw_mm']:.1f} mm postulated flaw grown under the equivalent "
                    f"stress range ({_crack['equivalent_stress_range_mpa']:.1f} MPa) to the "
                    f"{_crack['critical_depth_mm']:.1f} mm wall.")
        fr2.plotly_chart(pod_crack_fig(_crack), width="stretch", config={"displayModeBar": False})
        fr2.caption("Probability of detection vs crack size (subsea MPI/ACFM-class) - the basis for "
                    "crack-based inspection timing.")
        _cl = _crack["crack_life_years"]
        st.markdown(kpi_row([
            kpi("Crack-based life", life(_cl), "yr", "amber"),
            kpi("S-N deterministic life", life(dmg["deterministic_life_years"]), "yr", "sig"),
            kpi("Equivalent Δσ", f'{_crack["equivalent_stress_range_mpa"]:.1f}', "MPa"),
            kpi("Crack inspection", f'{_crack["crack_inspection_year"]:.1f}' if _crack.get("crack_inspection_year") else "-", "yr", "sig"),
            kpi("Material law", _crack["material"].split(",")[0]),
        ]), unsafe_allow_html=True)
        st.caption("A parallel fracture-mechanics pathway cross-checking the S-N life. Conservative "
                   "ECA: 1 mm postulated flaw, single membrane Y=1.12, no threshold benefit "
                   "(high-R tensioned riser). Design ECA needs the full BS 7910 2-D a/c integration.")

# ========================== ASSIMILATION =================================== #
if _section == "Assimilation":
    st.caption("Probabilistic remaining life: the Monte Carlo posterior, the Bayesian "
               "contraction as monitoring accrues, and the design-vs-actual divergence.")
    st.markdown('<div class="eq">&#955; ~ &#928;<sub>k</sub> m<sub>k</sub>&#183;&#955;&#8320;,&nbsp; '
                'life = (1&minus;D)/&#955;;&nbsp; posterior: prec = 1/&#964;&#8320;&#178; + n<sub>eff</sub>/s&#178; '
                '<span class="c"># MC uncertainty propagation + Bayesian AR(1) update</span></div>',
                unsafe_allow_html=True)
    st.markdown(kpi_row([
        kpi("P10", life(post["p10"]), "yr", "amber"),
        kpi("P50 median", life(post["p50"]), "yr", "sig"),
        kpi("P90", life(post["p90"]), "yr"),
        kpi("P90/P10 spread", f'{post["p90"]/post["p10"]:.1f}' if post["p10"] else "-", "x"),
        kpi("MC members", f'{post["n_members"]:,}'),
    ]), unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec" data-n="01">Remaining-life posterior &middot; {post["n_members"]:,} MC members &middot; '
        'Bayesian contraction (90% CI ~ 1/&radic;T)</div>', unsafe_allow_html=True)
    pc1, pc2 = dcols([3, 2])
    pc1.plotly_chart(fan_fig(payload["bayesian_fan"], post["p50"]), width="stretch", config={"displayModeBar": False})
    pc2.plotly_chart(pdf_hist_fig(post), width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="sec" data-n="02">Posterior CDF &middot; probability of exceeding a target life</div>',
                unsafe_allow_html=True)
    st.plotly_chart(cdf_fig(post), width="stretch", config={"displayModeBar": False})
    st.caption("Cumulative distribution of the Monte-Carlo remaining-life posterior - read off the "
               "probability the life falls below any design target.")
    if _rel is not None and _rel.get("enabled"):
        st.markdown('<div class="eq">g = ln(L&#183;&#916;) &minus; ln(T),&nbsp; '
                    '&#946; = (&#956;<sub>lnL</sub> &minus; ln T)/&#963;,&nbsp; '
                    'P<sub>f</sub> = &#934;(&minus;&#946;) '
                    '<span class="c"># FORM fatigue reliability (DNV-RP-C210)</span></div>',
                    unsafe_allow_html=True)
        _relpass = _rel["passes"]
        st.markdown('<div class="sec" data-n="04">Structural reliability &middot; FORM index &beta; vs DNV '
                    'safety class</div>', unsafe_allow_html=True)
        st.markdown(kpi_row([
            kpi("Reliability index β", f'{_rel["beta"]:.2f}', "", "sig" if _relpass else "alarm"),
            kpi("Target β", f'{_rel["target_beta"]:.2f}', f'{_rel["safety_class"]}'),
            kpi("Annual Pf", f'{_rel["pf_annual"]:.1e}'),
            kpi("Target annual Pf", f'{_rel["target_pf"]:.0e}'),
            kpi("Reliability check", "PASS" if _relpass else "FAIL", "", "sig" if _relpass else "alarm"),
        ]), unsafe_allow_html=True)
        rl1, rl2 = dcols([3, 2])
        rl1.plotly_chart(reliability_fig(_rel), width="stretch", config={"displayModeBar": False})
        with rl2:
            st.markdown(kpi_row([
                kpi("Mean-curve life", life(_rel["mean_curve_life_years"]), "yr"),
                kpi("Design life", f'{_rel["design_life_years"]:.0f}', "yr"),
            ]), unsafe_allow_html=True)
            st.caption("β on the mean S-N basis (DNV-RP-C210); the deterministic DFF check keeps the "
                       "characteristic curve. The importance factors show S-N scatter dominates the "
                       "fatigue uncertainty. Set the safety class in the sidebar.")
    if _dfan is not None:
        st.markdown('<div class="sec" data-n="03">Accumulated-damage divergence &middot; design vs actual wave climate '
                    '(spec &sect;5 gate)</div>', unsafe_allow_html=True)
        vd1, vd2 = dcols([3, 2])
        vd1.plotly_chart(divergence_fan_fig(_dfan), width="stretch", config={"displayModeBar": False})
        with vd2:
            _i15 = _dfan["years"].index(15.0) if 15.0 in _dfan["years"] else -1
            st.markdown(kpi_row([
                kpi("P10 @ yr 15", f'{100*_dfan["p10"][_i15]:.1f}', "%"),
                kpi("P90 @ yr 15", f'{100*_dfan["p90"][_i15]:.1f}', "%", "amber"),
            ]), unsafe_allow_html=True)
            st.caption("AR(1) wave-climate Monte Carlo. Spec gate: P10≈5%, P90≈28% at year 15.")

# ========================== ECONOMICS ====================================== #
if _section == "Economics":
    st.caption("The inspection decision and the conditional value of monitoring.")
    st.markdown('<div class="eq">&#916;C = &#8721;<sub>t</sub> [C<sub>base</sub>&minus;C<sub>cbm</sub>]/(1+r)<sup>t</sup> '
                '&minus; C<sub>sensor</sub>,&nbsp; C<sub>cbm</sub> = &#966;&#183;PV<sub>slow</sub> + '
                '(1&minus;&#966;)&#183;PV<sub>fast</sub> <span class="c"># Eq. 11 conditional CBM economics</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec" data-n="01">Risk-based inspection schedule</div>', unsafe_allow_html=True)
    dc1, dc2 = dcols([3, 2])
    dc1.plotly_chart(pof_fig(insp), width="stretch", config={"displayModeBar": False})
    with dc2:
        st.markdown(kpi_row([
            kpi("Next inspection", f'{insp["next_inspection_year"]:.1f}', "yr", "sig"),
            kpi("Target PoF", f'{insp["target_pof"]*100:.1f}', "%"),
        ]), unsafe_allow_html=True)
        st.markdown(kpi_row([
            kpi("PoF at inspection", f'{insp["pof_at_next"]*100:.2f}', "%",
                "alarm" if insp["pof_at_next"] > insp["target_pof"] * 1.05 else ""),
        ]), unsafe_allow_html=True)
    _net_tone = "sig" if econ.get("net_positive") else "alarm"
    _net = econ["fleet_delta_c_usd"] / 1e6
    _phi_src = "from posterior" if econ.get("phi_is_endogenous") else "break-even ref"
    st.markdown('<div class="sec" data-n="02">Conditional economics (Eq. 11) &middot; discounted value of monitoring vs &phi;</div>',
                unsafe_allow_html=True)
    ec1, ec2 = dcols([3, 2])
    ec1.plotly_chart(econ_fig(econ), width="stretch", config={"displayModeBar": False})
    with ec2:
        st.markdown(kpi_row([
            kpi(f"Net fleet value ΔC ({econ['n_units']}u, {econ['horizon_yr']:.0f}yr)",
                f'{_net:+.1f}', "US$M", _net_tone),
            kpi("per unit", f'{econ["per_unit_delta_c_usd"]/1e6:+.2f}', "US$M", _net_tone),
        ]), unsafe_allow_html=True)
        st.markdown(kpi_row([
            kpi(f"Fleet φ ({_phi_src})", f'{econ["phi"]:.2f}', "", "amber"),
            kpi("Break-even φ*", f'{econ["breakeven_phi"]:.2f}'),
        ]), unsafe_allow_html=True)
        st.caption(f"The sensor pays only when φ > φ*={econ['breakeven_phi']:.2f}; at r="
                   f"{econ['discount_rate']*100:.0f}% discount this fleet's φ={econ['phi']:.2f} makes it a "
                   f"net {'gain' if econ.get('net_positive') else 'cost'}.")
    st.markdown('<div class="sec" data-n="03">Discounted present-value cash-flow (per unit)</div>',
                unsafe_allow_html=True)
    _u = econ["n_units"]
    st.markdown(data_table(
        ["Cash-flow component", "PV per unit", "Basis"],
        [["Calendar inspections (baseline)", f"${econ['pv_baseline_usd']/1e6:.2f}M", "fixed interval"],
         ["CBM inspections — ages slower", f"${econ['pv_cbm_slow_usd']/1e6:.2f}M", "defer (φ)"],
         ["CBM inspections — ages faster", f"${econ['pv_cbm_fast_usd']/1e6:.2f}M", "tighten (1−φ)"],
         ["Sensor capex + PV opex", f"${econ['pv_sensor_usd']/1e6:.2f}M", "monitoring"],
         ["Net value ΔC per unit", f"${econ['per_unit_delta_c_usd']/1e6:+.2f}M", "Eq. 11"],
         [f"Net value ΔC — fleet ({_u} units)", f"${econ['fleet_delta_c_usd']/1e6:+.1f}M",
          "gain" if econ.get("net_positive") else "cost"]],
        value_cols=(1,)), unsafe_allow_html=True)

# ========================== LEDGER ========================================= #
if _section == "Ledger":
    st.caption("The auditable results ledger - every headline number with the standard it "
               "rests on, for this exact run.")
    _rows = [
        ("Deterministic fatigue life", f"{life(dmg['deterministic_life_years'])} yr", "DNV-RP-C203 Miner"),
        ("Remaining life P10 / P50 / P90", f"{life(post['p10'])} / {life(post['p50'])} / {life(post['p90'])} yr", "10k Monte Carlo"),
        ("Long-term (scatter) life", f"{life(_lt['life_years'])} yr" if _lt else "-", "DNV-RP-C203 Sec.5"),
        ("Combined wave+VIV life", f"{life(_comb_life)} yr", "Miner (wave + VIV)"),
        ("VIV-only life", f"{life(_viv['life_years'])} yr" if _viv and _viv.get('enabled') else "n/a", "DNV-RP-F204 screening"),
        ("DFF utilisation / acceptance", f"{_acc['utilisation']:.2f} -> {'PASS' if _acc['passes'] else 'FAIL'}" if _acc else "-", "DNV-OS-F201"),
        ("Reliability index β / annual Pf", f"{_rel['beta']:.2f} / {_rel['pf_annual']:.1e} -> {'PASS' if _rel['passes'] else 'FAIL'}" if _rel and _rel.get('enabled') else "-", "DNV-RP-C210 FORM"),
        ("Crack-based life (BS 7910 ECA)", f"{life(_crack['crack_life_years'])} yr" if _crack and _crack.get('enabled') else "-", "Paris-law"),
        ("Next inspection", f"{insp['next_inspection_year']:.1f} yr (target PoF {insp['target_pof']*100:.1f}%)", "RBI"),
        ("Net fleet value dC", f"{econ['fleet_delta_c_usd']/1e6:+.1f} US$M", "Eq. 11 discounted"),
        ("S-N class / environment", f"{cfg.riser.sn_class} / {dmg.get('sn_environment', 'in_air')}", "DNV-RP-C203"),
        ("Identified sea state Hs / Tp", f"{sea['hs']:.2f} m / {sea['tp']:.1f} s", "JONSWAP fit"),
        ("Annual damage (time / spectral)", f"{dmg['annual_rate_time']:.2e} / {dmg['annual_rate_spectral']:.2e} /yr", "rainflow / Dirlik"),
    ]
    st.markdown('<div class="sec" data-n="01">Results</div>', unsafe_allow_html=True)
    st.markdown(data_table(["Quantity", "Value", "Basis"],
                           [[q, v, b] for q, v, b in _rows], value_cols=(1,)), unsafe_allow_html=True)
    st.markdown('<div class="sec" data-n="02">Input configuration (this run)</div>', unsafe_allow_html=True)
    _rc = cfg.riser
    st.markdown(data_table(
        ["Input", "Value", "Field"],
        [["Riser OD × WT", f"{_rc.outer_diameter*1e3:.1f} × {_rc.wall_thickness*1e3:.2f} mm", "riser"],
         ["Material / UTS", f"{_rc.material_grade} / {_rc.ultimate_strength/1e6:.0f} MPa", "riser"],
         ["Water depth / hang-off", f"{_rc.water_depth:.0f} m / {_rc.hang_off_angle_deg:.0f}°", "riser"],
         ["SCF / S-N class / env", f"{_rc.scf:.2f} / {_rc.sn_class} / {_env_now}", "riser"],
         ["DFF / design life", f"{_rc.design_fatigue_factor:.0f}× / {_rc.design_service_life_years:.0f} yr", "riser"],
         ["Mean-stress / as-welded", f"{_rc.mean_stress_model.value} / {_rc.as_welded}", "riser"],
         ["Transfer route", cfg.transfer.route, "transfer"],
         ["Surface current", f"{cfg.viv.surface_current:.2f} m/s", "viv"],
         ["Monte-Carlo members / seed", f"{cfg.n_monte_carlo:,} / {cfg.seed}", "analysis"],
         ["Config SHA-256", prov["config_sha256"][:24] + "…", "provenance"]],
        value_cols=(1,)), unsafe_allow_html=True)

# ========================== PROVENANCE ===================================== #
if _section == "Provenance":
    st.caption("Reproducibility and verification: the acceptance gates and the exact "
               "inputs / library versions behind this run, plus downloadable reports.")
    st.markdown(f'<div class="sec" data-n="01">Acceptance gates &middot; {gates_ok}/{len(g)} passing '
                '(independent re-derivation, spec &sect;5)</div>', unsafe_allow_html=True)
    st.markdown(data_table(
        ["Gate", "Category", "Target", "Actual", "Status"],
        [[x["name"], x["category"], x["target"], x["actual"],
          '<span style="color:#2f855a">PASS</span>' if x["passed"] else '<span style="color:#c0523f">FAIL</span>']
         for x in g], value_cols=(3,)), unsafe_allow_html=True)
    st.markdown('<div class="sec" data-n="02">Claim → evidence map</div>', unsafe_allow_html=True)
    st.markdown(data_table(
        ["Paper claim", "Where it is proven", "Standard"],
        [["Hs recovered from motion spectrum", "JONSWAP gate + Environment tab", "4√m0"],
         ["Catenary shape / TDP curvature closed-form", "Catenary gate + Structure tab", "DNV-OS-F201"],
         ["Spectral & time-domain damage agree", "Dirlik-vs-rainflow gate + Detection tab", "<15% band"],
         ["Seawater-CP S-N knee at 1e6 cycles", "Seawater-CP gate", "DNV-RP-C203 T2-2"],
         ["Bayesian CI halves by year 4", "Bayesian 1/√T gate + Assimilation tab", "AR(1) n_eff"],
         ["Divergence P10≈5% / P90≈28% @ yr15", "Divergence gate + Assimilation tab", "spec §5"],
         ["Conditional economics break-even φ*", "Economics gate + Economics tab", "Eq. 11"],
         ["Byte-identical for identical seed", "Determinism gate", "reproducible"]],
        value_cols=(1,)), unsafe_allow_html=True)
    vc1, vc2 = dcols([3, 2])
    with vc1:
        st.markdown('<div class="sec">Reproducibility</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="note">Every result is reproducible from the exported bundle: the '
                    f'validated config (SHA-256 <code>{prov["config_sha256"][:16]}</code>), the seed '
                    f'(<code>{prov["seed"]}</code>) and the library versions. Re-running with the same '
                    'inputs is byte-identical (Determinism gate).</div>', unsafe_allow_html=True)
    with vc2:
        st.markdown('<div class="sec">Provenance</div>', unsafe_allow_html=True)
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
