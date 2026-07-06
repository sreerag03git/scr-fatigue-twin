"""SCR-Twin — shareable Streamlit console.

A deployable (Streamlit Community Cloud) front-end over the *same* tested physics
core as the desktop app. It reuses ``server.service`` so every number matches the
FastAPI/React build exactly — this file only handles UI and charts.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, point Streamlit Cloud at streamlit_app.py.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

# Make the local physics core and service layer importable without installation
# (so Streamlit Cloud works straight from the repo).
_ROOT = pathlib.Path(__file__).parent
for _p in (_ROOT / "core", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from scr_twin_core import ingest as ingest_mod  # noqa: E402
from scr_twin_core import validation as validation_mod  # noqa: E402
from scr_twin_core.config import AnalysisConfig, EnvironmentConfig, RiserConfig, TransferConfig  # noqa: E402
from server import service  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette / theme
# --------------------------------------------------------------------------- #
SIGNAL, SIGNAL2, AMBER, ALARM = "#0f8f9c", "#0b7079", "#b4791a", "#c33d28"
GRID, TEXT, TEXTHI, PAPER = "#dce4e8", "#3f515c", "#1a2830", "rgba(0,0,0,0)"
SN_CLASSES = ["B1", "B2", "C", "C1", "C2", "D", "E", "F", "F1", "F3", "G"]

st.set_page_config(
    page_title="SCR·Twin — TDP Fatigue Integrity Console",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --sig:#0f8f9c; --amber:#b4791a; }
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
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached compute (keyed on serialisable inputs -> efficient re-runs)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def gates() -> list[dict]:
    return [g.as_dict() for g in validation_mod.run_all_gates(seed=0)]


@st.cache_data(show_spinner="Running the full physics chain…")
def analyze_synthetic(config_json: str, hs: float, tp: float, gamma: float,
                      duration: float, fs: float, seed: int) -> dict:
    cfg = AnalysisConfig.model_validate_json(config_json)
    heave, fsr = service.make_synthetic(hs, tp, gamma, duration, fs, seed)
    return service.analyze(cfg, heave, fsr, is_synthetic=True)


@st.cache_data(show_spinner="Ingesting & analysing MRU record…")
def analyze_upload(config_json: str, file_bytes: bytes) -> dict:
    cfg = AnalysisConfig.model_validate_json(config_json)
    rec = ingest_mod.load_mru_csv(io.BytesIO(file_bytes))
    if not rec.health.ok:
        return {"error": "; ".join(rec.health.flags) or "data health check failed",
                "health": rec.health.as_dict()}
    return service.analyze(cfg, rec.channels["heave"], rec.fs,
                           is_synthetic=False, data_health=rec.health.as_dict())


# --------------------------------------------------------------------------- #
# Formatting + Plotly helpers
# --------------------------------------------------------------------------- #
def life(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return "999+" if v >= 999 else f"{v:.0f}"


def kpi(label: str, value: str, unit: str = "", tone: str = "") -> str:
    u = f"<small>{unit}</small>" if unit else ""
    return f'<div class="kpi"><div class="lab">{label}</div><div class="val {tone}">{value}{u}</div></div>'


def kpi_row(items: list[str]) -> None:
    st.markdown('<div class="kpi-row">' + "".join(items) + "</div>", unsafe_allow_html=True)


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
    f.add_scatter(x=spec["freq"], y=spec["motion_psd"], name="motion",
                  line=dict(color=SIGNAL, width=1.8), yaxis="y")
    f.add_scatter(x=spec["freq"], y=spec["stress_psd"], name="stress",
                  line=dict(color=AMBER, width=1.8), yaxis="y2")
    f.update_layout(
        xaxis=dict(title="Frequency [Hz]", gridcolor=GRID, zeroline=False, range=[0, 0.4]),
        yaxis=dict(title="motion [m²/Hz]", type="log", gridcolor=GRID, color=SIGNAL, zeroline=False),
        yaxis2=dict(title="stress [MPa²/Hz]", type="log", overlaying="y", side="right",
                    color=AMBER, showgrid=False),
    )
    return f


def trace_fig(trace: dict) -> go.Figure:
    f = _fig(180)
    f.add_scatter(x=trace["time"], y=trace["heave"], line=dict(color=SIGNAL, width=1))
    f.update_layout(xaxis=dict(title="t [s]", gridcolor=GRID, zeroline=False),
                    yaxis=dict(title="heave [m]", gridcolor=GRID, zeroline=False))
    return f


def fan_fig(fan: dict, p50: float) -> go.Figure:
    f = _fig(330)
    yrs = fan["years"]
    f.add_scatter(x=yrs, y=fan["high"], line=dict(width=0), hoverinfo="skip")
    f.add_scatter(x=yrs, y=fan["low"], fill="tonexty", line=dict(width=0),
                  fillcolor="rgba(180,121,26,0.15)", name="90% CI", hoverinfo="skip")
    f.add_scatter(x=yrs, y=fan["median"], line=dict(color=SIGNAL2, width=2.4), name="P50")
    f.add_hline(y=p50, line=dict(color=SIGNAL, width=0.8, dash="dot"))
    f.update_layout(
        xaxis=dict(title="Monitoring time [yr] →", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Remaining life [yr]", gridcolor=GRID, zeroline=False, rangemode="tozero"),
    )
    return f


def pdf_fig(post: dict) -> go.Figure:
    f = _fig(250)
    edges, counts = post["hist_edges"], post["hist_counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    f.add_bar(x=centers, y=counts, marker_color="rgba(15,143,156,0.35)",
              marker_line_color=SIGNAL, marker_line_width=0.3)
    for key, col in (("p10", AMBER), ("p50", SIGNAL2), ("p90", SIGNAL)):
        f.add_vline(x=post[key], line=dict(color=col, width=1.2, dash="dash"),
                    annotation_text=key.upper(), annotation_font_color=col,
                    annotation_font_size=9)
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
# Sidebar — data source + configuration
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Data source")
source = st.sidebar.radio("source", ["Synthetic (demo)", "Upload MRU CSV"], label_visibility="collapsed")

synth: dict[str, float] = {}
upload_bytes: bytes | None = None
if source.startswith("Synthetic"):
    st.sidebar.caption("Calibrated JONSWAP + RAO generator — every derived value is badged **synthetic**.")
    c1, c2 = st.sidebar.columns(2)
    synth["hs"] = c1.number_input("Hs [m]", 0.5, 16.0, 4.0, 0.5)
    synth["tp"] = c2.number_input("Tp [s]", 4.0, 20.0, 11.0, 0.5)
    synth["gamma"] = c1.number_input("γ peak", 1.0, 7.0, 2.5, 0.5)
    synth["duration"] = c2.number_input("Duration [s]", 300.0, 3600.0, 1800.0, 60.0)
    synth["fs"] = c1.number_input("fs [Hz]", 1.0, 10.0, 4.0, 1.0)
    synth["seed"] = c2.number_input("Seed", 0, 99_999_999, 20240705, 1)
else:
    up = st.sidebar.file_uploader("MRU CSV (time + heave/pitch…)", type=["csv"])
    if up is not None:
        upload_bytes = up.getvalue()
    st.sidebar.caption("Columns: `time_s, heave_m[, pitch_deg…]`. Malformed files degrade gracefully.")

st.sidebar.markdown("### Riser & analysis")
ref = RiserConfig.reference_scr()
with st.sidebar.expander("Steel catenary riser", expanded=True):
    od = st.number_input("Outer diameter [m]", 0.1, 1.5, ref.outer_diameter, 0.01, format="%.4f")
    wt = st.number_input("Wall thickness [m]", 0.005, 0.08, ref.wall_thickness, 0.001, format="%.4f")
    depth = st.number_input("Water depth [m]", 100.0, 3500.0, ref.water_depth, 50.0)
    ang = st.number_input("Hang-off [° from vertical]", 1.0, 45.0, ref.hang_off_angle_deg, 1.0)
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

# Build & validate the config (surface engineering-bound errors cleanly).
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

# --------------------------------------------------------------------------- #
# Header
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
        '<div><h1>SCR·TWIN</h1><div class="sub">TDP Fatigue Integrity Console</div></div></div>',
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(
        f'<div style="text-align:right;margin-top:10px">'
        f'<span class="tag {"pass" if gates_ok == len(g) else "fail"}">{gates_ok}/{len(g)} gates</span>&nbsp;'
        f'<span class="tag syn">{"SYNTHETIC" if source.startswith("Synthetic") else "MEASURED"}</span></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Run the analysis
# --------------------------------------------------------------------------- #
if source.startswith("Synthetic"):
    payload = analyze_synthetic(cfg.model_dump_json(), synth["hs"], synth["tp"], synth["gamma"],
                                synth["duration"], synth["fs"], int(synth["seed"]))
elif upload_bytes is not None:
    payload = analyze_upload(cfg.model_dump_json(), upload_bytes)
else:
    st.info("⬅ Upload an MRU CSV in the sidebar, or switch to the synthetic demo generator.")
    st.stop()

if "error" in payload:
    st.error(f"Data health check failed — {payload['error']}")
    if payload.get("health"):
        st.json(payload["health"])
    st.stop()

dmg, post, insp, econ, prov = (
    payload["damage"], payload["posterior"], payload["inspection"],
    payload["economics"], payload["provenance"],
)
sea, env = payload["sea_state"], payload["environment"]

# KPI band
kpi_row([
    kpi("Deterministic life", life(dmg["deterministic_life_years"]), "yr", "sig"),
    kpi("P10 (conservative)", life(post["p10"]), "yr", "amber"),
    kpi("P50 median", life(post["p50"]), "yr"),
    kpi("P90", life(post["p90"]), "yr"),
    kpi("Next inspection", f'{insp["next_inspection_year"]:.1f}', "yr", "sig"),
    kpi("Env. capacity factor", f'{env["factor"]:.3f}' if env["enabled"] else "—", "", "amber"),
])

# --------------------------------------------------------------------------- #
# Layer 0/1 — sea state + trace
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Sea state · spectral analysis · hang-off motion</div>', unsafe_allow_html=True)
kpi_row([
    kpi("Sig. heave Hm0", f'{sea["hs"]:.2f}', "m"),
    kpi("Tp", f'{sea["tp"]:.1f}', "s"),
    kpi("Tz", f'{sea["tz"]:.1f}', "s"),
    kpi("γ fit", f'{sea["gamma"]:.1f}'),
])
sc1, sc2 = st.columns([3, 2])
sc1.plotly_chart(spectra_fig(payload["spectrum"]), width="stretch", config={"displayModeBar": False})
sc2.plotly_chart(trace_fig(payload["trace"]), width="stretch", config={"displayModeBar": False})

# --------------------------------------------------------------------------- #
# Layer 2 — rainflow / S-N / Miner
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Layer 2 — rainflow · S-N · Miner</div>', unsafe_allow_html=True)
kpi_row([
    kpi("Annual damage (time)", f'{dmg["annual_rate_time"]:.2e}', "/yr", "amber"),
    kpi("Spectral (Dirlik)", f'{dmg["annual_rate_spectral"]:.2e}', "/yr"),
    kpi("Block damage", f'{dmg["block_damage"]:.2e}'),
    kpi("S-N class", cfg.riser.sn_class),
])

# --------------------------------------------------------------------------- #
# Layer 3 — posterior (signature fan) + PDF
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="sec">Layer 3 — remaining-life posterior · {post["n_members"]:,} MC members · '
    'Bayesian contraction (90% CI ~ 1/√T)</div>', unsafe_allow_html=True)
pc1, pc2 = st.columns([3, 2])
pc1.plotly_chart(fan_fig(payload["bayesian_fan"], post["p50"]), width="stretch",
                 config={"displayModeBar": False})
pc2.plotly_chart(pdf_fig(post), width="stretch", config={"displayModeBar": False})

# --------------------------------------------------------------------------- #
# Decision — RBI + fleet economics
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">Decision — risk-based inspection · fleet economics</div>', unsafe_allow_html=True)
dc1, dc2 = st.columns([3, 2])
dc1.plotly_chart(pof_fig(insp), width="stretch", config={"displayModeBar": False})
with dc2:
    kpi_row([
        kpi("Fleet saving (20u·20yr)",
            f'${econ["fleet_saving_low_usd"]/1e6:.1f}–{econ["fleet_saving_high_usd"]/1e6:.1f}M', "", "sig"),
        kpi("Sensor payback", f'{econ["payback_low_yr"]:.1f}–{econ["payback_high_yr"]:.1f}', "yr"),
    ])
    kpi_row([
        kpi("Target PoF", f'{insp["target_pof"]*100:.1f}', "%"),
        kpi("PoF at inspection", f'{insp["pof_at_next"]*100:.2f}', "%",
            "alarm" if insp["pof_at_next"] > insp["target_pof"] * 1.05 else ""),
    ])

# --------------------------------------------------------------------------- #
# Validation gates + provenance
# --------------------------------------------------------------------------- #
vc1, vc2 = st.columns([3, 2])
with vc1:
    st.markdown('<div class="sec">Validation gates (spec §5)</div>', unsafe_allow_html=True)
    for x in g:
        dot = "#56c08a" if x["passed"] else "#e75740"
        st.markdown(
            f'<div class="gate"><span class="dot" style="background:{dot}"></span>'
            f'<span>{x["name"]}</span><span class="actual">{x["actual"]}</span></div>',
            unsafe_allow_html=True,
        )
with vc2:
    st.markdown('<div class="sec">Provenance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="foot">core v{prov["core_version"]} · numpy {prov["numpy_version"]} · '
        f'scipy {prov["scipy_version"]}<br>seed {prov["seed"]} · '
        f'H(f) {"reduced-order" if prov["transfer_is_reduced_order"] else "reference"} · '
        f'cfg {prov["config_sha256"][:12]}<br>'
        f'{prov["n_samples"]:,} samples @ {prov["sample_rate_hz"]:.1f} Hz</div>',
        unsafe_allow_html=True,
    )
    bundle = {
        "config": json.loads(cfg.model_dump_json()),
        "source": payload.get("source", {"kind": "synthetic" if source.startswith("Synthetic") else "upload"}),
        "provenance": prov,
        "summary": {"sea_state": sea, "damage": dmg,
                    "posterior": {k: post[k] for k in ("p10", "p50", "p90", "n_members")},
                    "inspection": insp["next_inspection_year"], "economics": econ},
    }
    st.download_button("⬇ Export provenance bundle (JSON)", json.dumps(bundle, indent=2),
                       file_name="scr-twin-provenance.json", mime="application/json",
                       width="stretch")
    st.caption("Reproducible from config + seed + library versions.")

st.markdown(
    '<div class="foot" style="margin-top:16px;text-align:center">Physics-based digital twin · '
    'DNV-RP-C203 · ASTM E1049 · Dirlik / Tovo-Benasciutti · reduced-order Morison H(f) · '
    'reference SCR preset is illustrative, not project data</div>', unsafe_allow_html=True)
