"""Shared SVG diagram generators for the SCR twin (Streamlit + React frontends).

Pure string builders: each returns an SVG string driven by the solved payload /
riser config. Imported by streamlit_app.py and by server.service (which embeds the
SVGs in the analyze payload as a "diagrams" block for the React console).
"""

from __future__ import annotations


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
    pcx, pcy = rx0 + 76, by0 + 66
    # bonded material stack (bands exaggerated for legibility, not to scale)
    pipe_layers = [
        (46.0, "#c3ccce", "Solid PP (outer sheath)"),
        (41.0, "#dcd6b4", "Syntactic PP (insulation)"),
        (32.0, STEEL, "Carbon steel (X65)"),
        (22.0, "#b8a76a", "CRA liner (alloy 825)"),
        (17.0, "#ffffff", f"Bore / contents {contents:.0f} kg/m&#179;"),
    ]
    for r, fill, _lbl in pipe_layers:
        p.append(f'<circle cx="{pcx}" cy="{pcy}" r="{r:.1f}" fill="{fill}" stroke="{INK}" stroke-width="1"/>')
    p.append(f'<path d="M {pcx - 32} {pcy} A 32 32 0 0 1 {pcx} {pcy - 32} L {pcx} {pcy - 22} '
             f'A 22 22 0 0 0 {pcx - 22} {pcy} Z" fill="url(#hatchS)" opacity="0.7"/>')
    for i, (r, _f, lbl) in enumerate(pipe_layers):
        ly = pcy - 42 + i * 21
        lxr = pcx + 54
        p.append(LN(pcx + r * 0.70, pcy - r * 0.70 + i * 1.5, lxr - 4, ly, DIM, 0.7))
        p.append(T(lxr, ly + 3, lbl, fill=INK2, size=8))
    p.append(LN(pcx - 46, pcy + 58, pcx + 46, pcy + 58, DIM, 0.8, marker=' marker-start="url(#a1)" marker-end="url(#a2)"'))
    p.append(T(pcx, pcy + 70, f"OD {od_mm:.0f} mm &#183; wall t {wt_mm:.1f} mm &#183; {grade}", fill=INK, size=8.5, anchor="middle"))
    p.append(T(pcx, pcy + 82, "layers schematic, not to scale", fill=DIM, size=7.5, anchor="middle"))

    # DETAIL C : TDP weld hot-spot
    cy0 = by0 + 168
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


def system_cutaway_svg(payload: dict) -> str:
    """Realistic pictorial cutaway of the SCR system (rendered, gradient-shaded).

    A physically-scaled illustration companion to the drafting-grade GA drawing and
    the data-plot profile: a depth-graded water column with sunlight rays, a shaded
    turret-moored FPSO, the riser drawn as a cylindrical-shaded pipe on the real
    catenary, an amber touchdown hot-spot, and a textured seabed. Driven by the
    solved geometry (illustrative, not for construction).
    """
    import math

    cat = payload["catenary"]
    xs = [float(v) for v in cat["x"]]
    ys = [float(v) for v in cat["y"]]
    depth = float(cat["water_depth"])
    span = float(cat["horizontal_span"])
    a_cat = float(cat["catenary_parameter"])
    kappa_km = float(cat["tdp_curvature"]) * 1000.0
    hang_from_vert = 90.0 - float(cat["top_angle_deg"])
    viv = payload.get("viv") or {}
    us = float(viv.get("current_surface_velocity", 0.0)) if viv.get("enabled") else 0.0

    VBW, VBH = 1200, 820
    sky_h, ml, mr, bed_band = 150, 180, 64, 100
    mtop = sky_h + 34
    aw, ah = VBW - ml - mr, VBH - mtop - bed_band
    scale = min(aw / span, ah / depth)
    dw, dh = span * scale, depth * scale
    ox, oy = ml, mtop

    def SX(cx: float) -> float:
        return ox + (span - cx) * scale

    def SY(cy: float) -> float:
        return oy + (depth - cy) * scale

    surf_y, bed_y = SY(depth), SY(0.0)
    rp = [(SX(x), SY(y)) for x, y in zip(xs, ys)]
    hang = (SX(span), SY(depth))
    tdp = (SX(0.0), SY(0.0))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in rp)

    def chip(x, y, t, light=False, anchor="start"):
        fill = "#eef7f8" if light else "#12242b"
        stroke = "#0c353d" if light else "#ffffff"
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="12.5" font-weight="600" '
                f'text-anchor="{anchor}" style="paint-order:stroke;stroke:{stroke};stroke-width:3px;'
                f'stroke-linejoin:round;">{t}</text>')

    p = [f'<svg viewBox="0 0 {VBW} {VBH}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Inter, Segoe UI, sans-serif">']
    p.append(
        '<defs>'
        '<linearGradient id="cutwater" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#cfe7e9"/><stop offset="0.28" stop-color="#5fa6ac"/>'
        '<stop offset="0.62" stop-color="#217e86"/><stop offset="1" stop-color="#0c353d"/></linearGradient>'
        '<linearGradient id="cutsky" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#eef5f6"/><stop offset="1" stop-color="#dcebed"/></linearGradient>'
        '<linearGradient id="cutbed" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#c7bb96"/><stop offset="1" stop-color="#6f6038"/></linearGradient>'
        '<linearGradient id="cuthull" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#c3ccd0"/><stop offset="0.5" stop-color="#8a979d"/>'
        '<stop offset="1" stop-color="#515f65"/></linearGradient>'
        '<linearGradient id="cutdeck" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#dfe6e8"/><stop offset="1" stop-color="#aeb9bd"/></linearGradient>'
        '<radialGradient id="cuthaze" cx="0.5" cy="0.1" r="1.1">'
        '<stop offset="0" stop-color="#ffffff" stop-opacity="0.18"/>'
        '<stop offset="0.6" stop-color="#ffffff" stop-opacity="0"/></radialGradient>'
        '<filter id="cutsoft"><feGaussianBlur stdDeviation="3"/></filter>'
        '<filter id="cutray"><feGaussianBlur stdDeviation="6"/></filter>'
        '<marker id="cutflow" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3fa9ff"/></marker>'
        '<marker id="cutflowu" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e8a33a"/></marker>'
        '<pattern id="cutstip" width="10" height="10" patternUnits="userSpaceOnUse">'
        '<circle cx="2" cy="3" r="0.7" fill="#5c4f2c" opacity="0.5"/>'
        '<circle cx="7" cy="7" r="0.7" fill="#5c4f2c" opacity="0.4"/></pattern>'
        '</defs>'
    )

    # bands
    p.append(f'<rect x="0" y="0" width="{VBW}" height="{surf_y:.1f}" fill="url(#cutsky)"/>')
    p.append(f'<rect x="0" y="{surf_y:.1f}" width="{VBW}" height="{bed_y - surf_y:.1f}" fill="url(#cutwater)"/>')
    p.append(f'<rect x="0" y="{bed_y:.1f}" width="{VBW}" height="{VBH - bed_y:.1f}" fill="url(#cutbed)"/>')
    p.append(f'<rect x="0" y="{surf_y:.1f}" width="{VBW}" height="{0.5 * (bed_y - surf_y):.1f}" fill="url(#cuthaze)"/>')

    # sunlight rays
    for i in range(5):
        rx = 120 + i * 230
        w = 26 + i * 4
        y2 = surf_y + 0.62 * (bed_y - surf_y)
        p.append(f'<polygon points="{rx},{surf_y:.1f} {rx + w},{surf_y:.1f} {rx - 70 + w},{y2:.1f} '
                 f'{rx - 90},{y2:.1f}" fill="#ffffff" opacity="0.06" filter="url(#cutray)"/>')
    # current streaks
    if us > 0:
        for i in range(7):
            cyy = surf_y + 30 + i * ((bed_y - surf_y - 40) / 7)
            ln = 40 + (i % 3) * 24
            cxx = 90 + (i * 97) % 600
            p.append(f'<line x1="{cxx}" y1="{cyy:.1f}" x2="{cxx + ln}" y2="{cyy:.1f}" stroke="#eafcff" '
                     f'stroke-width="1.6" opacity="0.12" stroke-linecap="round"/>')
    # surface waves + specular
    wave = "".join(f'Q{(i * VBW / 40 + VBW / 80):.0f},{surf_y - 4:.1f} {((i + 1) * VBW / 40):.0f},{surf_y:.1f} '
                   for i in range(40))
    p.append(f'<path d="M0,{surf_y:.1f} {wave}" fill="none" stroke="#ffffff" stroke-width="1.6" opacity="0.5"/>')

    # seabed texture + mudline
    p.append(f'<rect x="0" y="{bed_y:.1f}" width="{VBW}" height="{VBH - bed_y:.1f}" fill="url(#cutstip)"/>')
    dune = "".join(f'Q{(i * VBW / 24 + VBW / 48):.0f},{bed_y - 2 - (2 if i % 2 else 0):.1f} '
                   f'{((i + 1) * VBW / 24):.0f},{bed_y:.1f} ' for i in range(24))
    p.append(f'<path d="M0,{bed_y:.1f} {dune}" fill="none" stroke="#efe6c8" stroke-width="1.4" opacity="0.6"/>')

    # ---- FPSO (rendered) ----
    lf = 0.30 * dw
    cx = hang[0]
    deck_y, keel_y = surf_y - 0.045 * dh, surf_y + 0.05 * dh
    aft, fwd = cx - lf * 0.62, cx + lf * 0.42
    p.append(f'<path d="M{aft:.1f},{surf_y + 2:.1f} L{fwd:.1f},{surf_y + 2:.1f} L{fwd:.1f},{surf_y + 18:.1f} '
             f'L{aft:.1f},{surf_y + 18:.1f} Z" fill="#ffffff" opacity="0.06" filter="url(#cutsoft)"/>')
    hull = (f'M{aft:.1f},{deck_y:.1f} L{fwd - 0.05 * lf:.1f},{deck_y - 0.006 * dh:.1f} '
            f'Q{fwd:.1f},{deck_y:.1f} {fwd:.1f},{surf_y:.1f} L{fwd - 0.03 * lf:.1f},{keel_y:.1f} '
            f'L{aft + 0.04 * lf:.1f},{keel_y:.1f} Q{aft:.1f},{keel_y:.1f} {aft:.1f},{keel_y - 0.02 * dh:.1f} Z')
    p.append(f'<path d="{hull}" fill="url(#cuthull)" stroke="#3a474d" stroke-width="1.2"/>')
    p.append(f'<line x1="{aft + 3:.1f}" y1="{surf_y:.1f}" x2="{fwd - 4:.1f}" y2="{surf_y:.1f}" stroke="#0a5c61" stroke-width="2.2" opacity="0.8"/>')
    p.append(f'<rect x="{aft:.1f}" y="{deck_y - 3:.1f}" width="{fwd - 0.05 * lf - aft:.1f}" height="4" fill="url(#cutdeck)"/>')
    for i in range(4):
        mx = aft + 0.14 * lf + i * 0.14 * lf
        p.append(f'<rect x="{mx:.1f}" y="{deck_y - 0.05 * dh:.1f}" width="{0.11 * lf:.1f}" height="{0.05 * dh:.1f}" '
                 f'fill="url(#cutdeck)" stroke="#5a686e" stroke-width="0.8"/>')
    p.append(f'<rect x="{aft + 0.03 * lf:.1f}" y="{deck_y - 0.10 * dh:.1f}" width="{0.10 * lf:.1f}" height="{0.10 * dh:.1f}" '
             f'fill="url(#cutdeck)" stroke="#5a686e" stroke-width="0.9"/>')
    p.append(f'<path d="M{fwd - 0.11 * lf:.1f},{deck_y - 0.05 * dh:.1f} L{fwd - 0.065 * lf:.1f},{deck_y - 0.14 * dh:.1f} '
             f'L{fwd - 0.02 * lf:.1f},{deck_y - 0.05 * dh:.1f}" fill="none" stroke="#6d7c82" stroke-width="1"/>')
    p.append(f'<path d="M{fwd - 0.065 * lf:.1f},{deck_y - 0.14 * dh:.1f} q5,-8 12,-2 q-3,6 -10,3" fill="#d98a2b" opacity="0.85"/>')
    p.append(f'<rect x="{cx - 0.02 * lf:.1f}" y="{deck_y - 0.03 * dh:.1f}" width="{0.04 * lf:.1f}" '
             f'height="{0.03 * dh + keel_y - deck_y:.1f}" fill="#7d8a90" stroke="#465257" stroke-width="0.8"/>')
    p.append(f'<circle cx="{cx:.1f}" cy="{keel_y:.1f}" r="4.5" fill="#d98a2b" stroke="#ffffff" stroke-width="1"/>')

    # ---- riser: cylindrical-shaded catenary pipe ----
    p.append(f'<polyline points="{poly}" fill="none" stroke="#06343a" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>')
    p.append(f'<polyline points="{poly}" fill="none" stroke="#0e7c82" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/>')

    def _off(pts, o):
        out = []
        for i in range(len(pts)):
            a = pts[max(0, i - 1)]
            b = pts[min(len(pts) - 1, i + 1)]
            ddx, ddy = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(ddx, ddy) or 1.0
            out.append((pts[i][0] - ddy / ln * o, pts[i][1] + ddx / ln * o))
        return out

    hi = _off(rp, 1.6)
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in hi)}" fill="none" '
             f'stroke="#7fdfe4" stroke-width="1.8" stroke-linecap="round" opacity="0.8"/>')
    p.append(f'<line x1="{tdp[0]:.1f}" y1="{tdp[1]:.1f}" x2="{tdp[0] + 0.03 * dw:.1f}" y2="{tdp[1]:.1f}" stroke="#06343a" stroke-width="9" stroke-linecap="round"/>')
    p.append(f'<line x1="{tdp[0]:.1f}" y1="{tdp[1]:.1f}" x2="{tdp[0] + 0.03 * dw:.1f}" y2="{tdp[1]:.1f}" stroke="#0e7c82" stroke-width="6.5" stroke-linecap="round"/>')
    p.append(f'<circle cx="{tdp[0]:.1f}" cy="{tdp[1]:.1f}" r="16" fill="#e8a33a" opacity="0.18" filter="url(#cutsoft)"/>')
    p.append(f'<circle cx="{tdp[0]:.1f}" cy="{tdp[1]:.1f}" r="5.5" fill="none" stroke="#f2c66b" stroke-width="2"/>')

    # ---- real-life engineering annotations (Buberg et al. conventions) ----
    def varrow(x, y1, y2, col, w=1.8):
        return (f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{w}"/>'
                f'<path d="M{x:.1f},{y1:.1f} l-4,7 l8,0 Z" fill="{col}"/>'
                f'<path d="M{x:.1f},{y2:.1f} l-4,-7 l8,0 Z" fill="{col}"/>')
    hvx = fwd + 24
    p.append(varrow(hvx, surf_y - 58, surf_y + 8, "#b23b3b"))
    p.append(chip(hvx + 8, surf_y - 44, "Heave motion"))
    p.append(chip(VBW * 0.42, surf_y - 8, "Wave"))
    # hang-off point + theta_HP from the vertical
    p.append(f'<line x1="{cx:.1f}" y1="{keel_y:.1f}" x2="{cx:.1f}" y2="{keel_y + 0.26 * dh:.1f}" '
             f'stroke="#12242b" stroke-width="1" stroke-dasharray="6 4"/>')
    _t0, _t1 = rp[-1], rp[max(0, len(rp) - 5)]
    _tang = math.atan2(_t1[1] - _t0[1], _t1[0] - _t0[0])
    _ra = 0.14 * dh
    p.append(f'<path d="M{cx:.1f},{keel_y + _ra:.1f} A{_ra:.1f},{_ra:.1f} 0 0 0 '
             f'{cx + _ra * math.cos(_tang):.1f},{keel_y + _ra * math.sin(_tang):.1f}" '
             f'fill="none" stroke="#e8a33a" stroke-width="1.6"/>')
    p.append(chip(cx + _ra * 0.8, keel_y + _ra * 0.9, "&#952;<tspan baseline-shift=\"sub\" font-size=\"9\">HP</tspan>", light=True))
    p.append(chip(cx + 12, keel_y - 4, "Hang-Off Point (HOP)", light=True))
    _ai = int(len(rp) * 0.72)
    p.append(chip(rp[_ai][0] + 10, rp[_ai][1], "Arc Length (AL)", light=True))
    # VIV callout: internal flow / time-varying external flow / vortex
    _mi = int(len(rp) * 0.42)
    _ex, _ey = rp[_mi][0] + 0.26 * dw, rp[_mi][1]
    _erx, _ery = 0.12 * dw, 0.075 * dh
    p.append(f'<line x1="{rp[_mi][0]:.1f}" y1="{rp[_mi][1]:.1f}" x2="{_ex - _erx:.1f}" y2="{_ey:.1f}" '
             f'stroke="#eef7f8" stroke-width="1" opacity="0.7"/>')
    p.append(f'<ellipse cx="{_ex:.1f}" cy="{_ey:.1f}" rx="{_erx:.1f}" ry="{_ery:.1f}" '
             f'fill="#0c353d" opacity="0.16" stroke="#cfeef0" stroke-width="1"/>')
    for _k in range(3):
        _ay = _ey - _ery * 0.42 + _k * _ery * 0.42
        p.append(f'<line x1="{_ex - _erx * 0.7:.1f}" y1="{_ay:.1f}" x2="{_ex - _erx * 0.05:.1f}" y2="{_ay:.1f}" '
                 f'stroke="#3fa9ff" stroke-width="1.6" marker-end="url(#cutflow)"/>')
    p.append(f'<line x1="{_ex + _erx * 0.15:.1f}" y1="{_ey + _ery * 0.55:.1f}" x2="{_ex + _erx * 0.15:.1f}" '
             f'y2="{_ey - _ery * 0.55:.1f}" stroke="#e8a33a" stroke-width="1.6" marker-end="url(#cutflowu)"/>')
    p.append(f'<path d="M{_ex + _erx * 0.5:.1f},{_ey + 2:.1f} a5,5 0 1 1 -3,-4" fill="none" stroke="#cfeef0" stroke-width="1.4"/>')
    p.append(chip(_ex, _ey - _ery - 6, "Time-varying external flow", light=True, anchor="middle"))
    p.append(chip(_ex - _erx * 0.6, _ey + _ery + 13, "internal flow", light=True))
    p.append(chip(_ex + _erx * 0.5, _ey + 4, "vortex", light=True))
    # coordinate axes O-X-Y-Z at the seabed
    _axx, _axy = tdp[0] + 0.16 * dw, bed_y - 8
    p.append(f'<line x1="{_axx:.1f}" y1="{_axy:.1f}" x2="{_axx:.1f}" y2="{_axy - 30:.1f}" stroke="#12242b" stroke-width="1.4"/>'
             f'<path d="M{_axx:.1f},{_axy - 30:.1f} l-3,6 l6,0 Z" fill="#12242b"/>')
    p.append(f'<line x1="{_axx:.1f}" y1="{_axy:.1f}" x2="{_axx + 30:.1f}" y2="{_axy:.1f}" stroke="#12242b" stroke-width="1.4"/>'
             f'<path d="M{_axx + 30:.1f},{_axy:.1f} l-6,-3 l0,6 Z" fill="#12242b"/>')
    p.append(f'<line x1="{_axx:.1f}" y1="{_axy:.1f}" x2="{_axx + 20:.1f}" y2="{_axy - 16:.1f}" stroke="#12242b" stroke-width="1.4"/>'
             f'<path d="M{_axx + 20:.1f},{_axy - 16:.1f} l-6,1 l3,5 Z" fill="#12242b"/>')
    p.append(chip(_axx - 4, _axy + 12, "O", anchor="end"))
    p.append(chip(_axx + 33, _axy + 3, "X"))
    p.append(chip(_axx + 22, _axy - 17, "Y"))
    p.append(chip(_axx - 3, _axy - 33, "Z", anchor="end"))

    # ---- labels ----
    mid = int(len(rp) * 0.55)
    p.append(chip(VBW - 14, surf_y - 8, "MEAN WATER LEVEL (MWL)", anchor="end"))
    p.append(chip(cx, deck_y - 0.14 * dh - 10, "Floating platform / FPSO", anchor="middle"))
    if us > 0:
        p.append(chip(14, surf_y - 8, f"Current + wind {us:.2f} m/s"))
    p.append(chip(14, (surf_y + bed_y) / 2, f"Water depth {depth:.0f} m", light=True))
    p.append(chip(rp[mid][0], rp[mid][1] - 12, "Steel catenary riser (SCR)", light=True))
    p.append(chip(tdp[0] - 14, tdp[1] - 16, f"Touch Down Point (TDP), &#954;={kappa_km:.2f}/km", light=True, anchor="end"))
    p.append(chip(VBW * 0.58, bed_y + 16, "mudline"))
    p.append(chip(14, bed_y + 34, "Seabed (linear stiffness k)"))
    p.append(f'<rect x="0" y="{VBH - 24}" width="{VBW}" height="24" fill="#0c353d"/>')
    p.append(f'<text x="14" y="{VBH - 8}" fill="#bfe0e3" font-size="11">Illustrative cutaway &mdash; '
             f'drawn to the solved catenary geometry (a={a_cat:.0f} m, layback {span:.0f} m, '
             f'{hang_from_vert:.0f}&#176; from vertical). Not for construction.</text>')

    p.append("</svg>")
    return "".join(p)


def riser_config_svg() -> str:
    """Schematic comparison of riser configurations: plain SCR vs steel lazy-wave (SLWR).

    Illustrative line-art (after Buberg et al. Fig. of SCR / SLWR): the twin solves the
    plain steel catenary; a lazy-wave / steep-wave configuration (buoyancy section,
    sag + hog) needs a dedicated solver and is shown here only for context.
    """
    INK, INK2, TEAL, AMBER, DIM = "#182530", "#3a4c54", "#0e7c82", "#a86f16", "#8b9aa0"
    SAND, WATER = "#9c8a5f", "#eef4f5"
    VBW, VBH = 1120, 520
    ywl, ybed = 92, 452

    def T(x, y, t, fill=INK, size=10.5, anchor="start", weight=400):
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
                f'font-weight="{weight}" text-anchor="{anchor}">{t}</text>')

    p = [f'<svg viewBox="0 0 {VBW} {VBH}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Inter, Segoe UI, sans-serif">',
         f'<rect width="{VBW}" height="{VBH}" fill="#ffffff"/>',
         f'<rect x="0" y="{ywl}" width="{VBW}" height="{ybed - ywl}" fill="{WATER}"/>',
         f'<pattern id="rcbed" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         f'<line x1="0" y1="0" x2="0" y2="10" stroke="{SAND}" stroke-width="0.9"/></pattern>',
         f'<rect x="0" y="{ybed}" width="{VBW}" height="{VBH - ybed}" fill="url(#rcbed)"/>',
         f'<line x1="0" y1="{ybed}" x2="{VBW}" y2="{ybed}" stroke="{SAND}" stroke-width="1.6"/>',
         f'<line x1="0" y1="{ywl}" x2="{VBW}" y2="{ywl}" stroke="{TEAL}" stroke-width="1.2"/>']
    p.append(T(10, ywl - 6, "waterline", fill=TEAL, size=9))
    p.append(T(VBW - 10, ybed + 18, "seabed / mudline", fill=SAND, size=9, anchor="end"))

    # floating facility (shared), two hang-off points
    fx = 545
    p.append(f'<rect x="{fx - 42}" y="{ywl - 16}" width="84" height="16" fill="#e9eeef" stroke="{INK}" stroke-width="1.2"/>')
    p.append(f'<line x1="{fx - 8}" y1="{ywl - 16}" x2="{fx - 8}" y2="{ywl - 34}" stroke="{INK}" stroke-width="1.1"/>')
    p.append(f'<line x1="{fx + 8}" y1="{ywl - 16}" x2="{fx + 8}" y2="{ywl - 30}" stroke="{INK}" stroke-width="1.1"/>')
    p.append(T(fx, ywl - 40, "Floating facility", fill=INK, size=10, anchor="middle", weight=600))
    hopl, hopr = (fx - 30, ywl + 4), (fx + 30, ywl + 4)
    for hp in (hopl, hopr):
        p.append(f'<circle cx="{hp[0]}" cy="{hp[1]}" r="3" fill="{AMBER}"/>')
    p.append(T(fx, ywl + 22, "Hang-Off Point (HOP)", fill=AMBER, size=9, anchor="middle"))

    # left: plain steel catenary riser (SCR)
    p.append(f'<path d="M{hopl[0]},{hopl[1]} C {hopl[0] - 40},{ywl + 150} {hopl[0] - 210},{ybed} 150,{ybed}" '
             f'fill="none" stroke="{TEAL}" stroke-width="3" stroke-linecap="round"/>')
    p.append(f'<line x1="150" y1="{ybed}" x2="120" y2="{ybed}" stroke="{TEAL}" stroke-width="3" stroke-linecap="round"/>')
    p.append(f'<circle cx="150" cy="{ybed}" r="5" fill="none" stroke="{AMBER}" stroke-width="1.5"/>')
    p.append(f'<circle cx="150" cy="{ybed}" r="2.2" fill="{AMBER}"/>')
    p.append(T(300, 200, "Steel catenary riser (SCR)", fill=INK, size=11, weight=600))
    p.append(T(150, ybed - 12, "Touch Down Point (TDP)", fill=AMBER, size=9, anchor="middle"))

    # right: steel lazy-wave riser (SLWR) - upper / buoyancy (hog) / lower sections
    slwr = (f'M{hopr[0]},{hopr[1]} C {hopr[0] + 55},{ywl + 170} {hopr[0] + 150},{ybed - 20} {hopr[0] + 205},{ybed - 22} '
            f'C {hopr[0] + 250},{ybed - 24} {hopr[0] + 285},{ybed - 120} {hopr[0] + 330},{ybed - 118} '
            f'C {hopr[0] + 375},{ybed - 116} {hopr[0] + 405},{ybed - 20} {hopr[0] + 430},{ybed}')
    p.append(f'<path d="{slwr}" fill="none" stroke="{TEAL}" stroke-width="3" stroke-linecap="round"/>')
    p.append(f'<line x1="{hopr[0] + 430}" y1="{ybed}" x2="{hopr[0] + 470}" y2="{ybed}" stroke="{TEAL}" stroke-width="3" stroke-linecap="round"/>')
    # buoyancy modules along the hog crest
    for i in range(6):
        bxp = hopr[0] + 292 + i * 13
        p.append(f'<ellipse cx="{bxp}" cy="{ybed - 118 - 5}" rx="6" ry="8" fill="#f2d38a" stroke="{AMBER}" stroke-width="1"/>')
    p.append(f'<circle cx="{hopr[0] + 430}" cy="{ybed}" r="5" fill="none" stroke="{AMBER}" stroke-width="1.5"/>')
    p.append(f'<circle cx="{hopr[0] + 430}" cy="{ybed}" r="2.2" fill="{AMBER}"/>')
    p.append(T(hopr[0] + 60, ywl + 120, "Upper section", fill=INK2, size=9))
    p.append(T(hopr[0] + 320, ybed - 138, "Buoyancy section (hog)", fill=INK2, size=9, anchor="middle"))
    p.append(T(hopr[0] + 405, ybed - 70, "Lower section", fill=INK2, size=9))
    p.append(T(hopr[0] + 250, 176, "Steel lazy-wave riser (SLWR)", fill=INK, size=11, weight=600))
    p.append(T(hopr[0] + 430, ybed - 12, "TDP", fill=AMBER, size=9, anchor="middle"))

    p.append(f'<rect x="0" y="{VBH - 22}" width="{VBW}" height="22" fill="#eef2f2"/>')
    p.append(T(10, VBH - 7, "Illustrative riser configurations (schematic). The twin solves the plain SCR; "
              "a lazy-wave / steep-wave configuration needs a dedicated solver.", fill=DIM, size=9.5))
    p.append("</svg>")
    return "".join(p)


def platform_types_svg() -> str:
    """Schematic of the common floating hosts and their riser configurations.

    Illustrative context (after typical riser-system figures): semisubmersible,
    FPSO, spar and TLP with top-tension / steel-catenary / lazy-wave risers. The
    twin analyses the steel catenary riser (SCR); the others are shown for context.
    """
    INK, INK2, TEAL, DIM, SAND, WATER = (
        "#182530", "#3a4c54", "#0e7c82", "#8b9aa0", "#9c8a5f", "#eef4f5")
    VBW, VBH = 1120, 420
    ywl, ybed = 74, 344

    def T(x, y, t, fill=INK, size=10, anchor="middle", weight=400):
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
                f'font-weight="{weight}" text-anchor="{anchor}">{t}</text>')

    p = [f'<svg viewBox="0 0 {VBW} {VBH}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Inter, Segoe UI, sans-serif">',
         f'<rect width="{VBW}" height="{VBH}" fill="#ffffff"/>',
         f'<rect x="0" y="{ywl}" width="{VBW}" height="{ybed - ywl}" fill="{WATER}"/>',
         f'<pattern id="ptbed" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         f'<line x1="0" y1="0" x2="0" y2="10" stroke="{SAND}" stroke-width="0.9"/></pattern>',
         f'<rect x="0" y="{ybed}" width="{VBW}" height="{VBH - ybed}" fill="url(#ptbed)"/>',
         f'<line x1="0" y1="{ybed}" x2="{VBW}" y2="{ybed}" stroke="{SAND}" stroke-width="1.5"/>',
         f'<line x1="0" y1="{ywl}" x2="{VBW}" y2="{ywl}" stroke="{TEAL}" stroke-width="1.1"/>',
         T(10, ywl - 6, "sea surface", fill=TEAL, size=9, anchor="start"),
         T(VBW - 10, ybed + 18, "seabed", fill=SAND, size=9, anchor="end")]

    def anchor_box(x):
        return f'<rect x="{x - 8:.1f}" y="{ybed - 4:.1f}" width="16" height="8" fill="{INK2}"/>'

    # 1) Semisubmersible + steel catenary riser
    cx = 150
    p.append(f'<rect x="{cx - 44}" y="{ywl - 26}" width="88" height="14" fill="#dfe6e7" stroke="{INK}" stroke-width="1.1"/>')
    p.append(f'<line x1="{cx-38}" y1="{ywl-12}" x2="{cx-38}" y2="{ywl+10}" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<line x1="{cx+38}" y1="{ywl-12}" x2="{cx+38}" y2="{ywl+10}" stroke="{INK}" stroke-width="3"/>')
    p.append(f'<rect x="{cx-50}" y="{ywl+10}" width="24" height="9" fill="#c3ccce" stroke="{INK}" stroke-width="1"/>')
    p.append(f'<rect x="{cx+26}" y="{ywl+10}" width="24" height="9" fill="#c3ccce" stroke="{INK}" stroke-width="1"/>')
    p.append(f'<path d="M{cx+40},{ywl+16} C {cx+70},{ywl+140} {cx+150},{ybed} {cx+165},{ybed}" fill="none" stroke="{TEAL}" stroke-width="2.4"/>')
    p.append(anchor_box(cx + 165))
    p.append(T(cx, ywl - 34, "Semisubmersible", fill=INK, size=11, weight=600))
    p.append(T(cx + 70, ybed - 8, "Steel catenary riser", fill=INK2, size=9))

    # 2) FPSO + catenary riser + top-tension riser
    cx = 430
    p.append(f'<path d="M{cx-64},{ywl-14} L{cx+58},{ywl-14} L{cx+70},{ywl-2} L{cx+58},{ywl+12} L{cx-58},{ywl+12} L{cx-70},{ywl} Z" fill="#dfe6e7" stroke="{INK}" stroke-width="1.2"/>')
    p.append(f'<rect x="{cx-6}" y="{ywl-24}" width="10" height="10" fill="none" stroke="{INK}" stroke-width="1"/>')
    p.append(f'<line x1="{cx-20}" y1="{ywl+12}" x2="{cx-20}" y2="{ybed}" stroke="{INK2}" stroke-width="2.2"/>')
    p.append(anchor_box(cx - 20))
    p.append(f'<path d="M{cx+30},{ywl+12} C {cx+70},{ywl+150} {cx+150},{ybed} {cx+165},{ybed}" fill="none" stroke="{TEAL}" stroke-width="2.4"/>')
    p.append(anchor_box(cx + 165))
    p.append(T(cx, ywl - 24, "FPSO", fill=INK, size=11, weight=600))
    p.append(T(cx - 20, ybed - 8, "top-tension riser", fill=INK2, size=8.5, anchor="end"))
    p.append(T(cx + 96, ybed - 8, "catenary riser", fill=INK2, size=9))

    # 3) Spar + catenary riser
    cx = 720
    p.append(f'<rect x="{cx-28}" y="{ywl-18}" width="56" height="12" fill="#dfe6e7" stroke="{INK}" stroke-width="1.1"/>')
    p.append(f'<rect x="{cx-14}" y="{ywl-6}" width="28" height="{ybed-ywl-90:.0f}" fill="#c3ccce" stroke="{INK}" stroke-width="1.1"/>')
    p.append(f'<path d="M{cx+14},{ywl+40} C {cx+60},{ywl+150} {cx+150},{ybed} {cx+165},{ybed}" fill="none" stroke="{TEAL}" stroke-width="2.4"/>')
    p.append(anchor_box(cx + 165))
    p.append(T(cx, ywl - 26, "Spar", fill=INK, size=11, weight=600))
    p.append(T(cx + 96, ybed - 8, "catenary riser", fill=INK2, size=9))

    # 4) TLP + top-tension risers on taut tendons
    cx = 1000
    p.append(f'<rect x="{cx-44}" y="{ywl-24}" width="88" height="13" fill="#dfe6e7" stroke="{INK}" stroke-width="1.1"/>')
    for dx in (-30, -10, 10, 30):
        p.append(f'<rect x="{cx+dx-3}" y="{ywl-11}" width="6" height="24" fill="#c3ccce" stroke="{INK}" stroke-width="0.8"/>')
    for dx in (-40, 40):
        p.append(f'<line x1="{cx+dx}" y1="{ywl+13}" x2="{cx+dx}" y2="{ybed}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="5 3"/>')
        p.append(anchor_box(cx + dx))
    for dx in (-12, 12):
        p.append(f'<line x1="{cx+dx}" y1="{ywl+13}" x2="{cx+dx}" y2="{ybed}" stroke="{TEAL}" stroke-width="2.2"/>')
        p.append(anchor_box(cx + dx))
    p.append(T(cx, ywl - 32, "TLP", fill=INK, size=11, weight=600))
    p.append(T(cx, ybed - 8, "top-tension risers", fill=INK2, size=9))

    p.append(f'<rect x="0" y="{VBH - 22}" width="{VBW}" height="22" fill="#eef2f2"/>')
    p.append(T(10, VBH - 7, "Illustrative host-platform / riser configurations. This twin analyses the steel "
              "catenary riser (SCR); other hosts and riser types are shown for context.",
              fill=DIM, size=9.5, anchor="start"))
    p.append("</svg>")
    return "".join(p)


def flexible_riser_svg() -> str:
    """Unbonded flexible-pipe cross-section (an alternative to the bonded steel SCR).

    Illustrative layer stack after typical flexible-riser figures: anti-collapse
    carcass, internal pressure sheath, hoop/pressure armour, cross-wound tensile
    armour wires, and the outer sheath. The twin models the steel catenary riser.
    """
    INK, INK2, DIM = "#182530", "#3a4c54", "#8b9aa0"
    VBW, VBH = 1120, 300
    cx, cy = 200, 150

    def T(x, y, t, fill=INK, size=9.5, anchor="start", weight=400):
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
                f'font-weight="{weight}" text-anchor="{anchor}">{t}</text>')

    layers = [
        (108, "#c3ccce", "Outer sheath (PA/PE)"),
        (98, "#9fb0b5", "Tensile armour (outer, cross-wound)"),
        (86, "#7d8f95", "Tensile armour (inner, cross-wound)"),
        (72, "#b8a76a", "Pressure / hoop armour"),
        (58, "#dcd6b4", "Internal pressure sheath"),
        (46, "#8a979d", "Anti-collapse carcass (interlocked)"),
        (34, "#ffffff", "Bore"),
    ]
    p = [f'<svg viewBox="0 0 {VBW} {VBH}" width="100%" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Inter, Segoe UI, sans-serif">',
         f'<rect width="{VBW}" height="{VBH}" fill="#ffffff"/>',
         T(20, 30, "FLEXIBLE (UNBONDED) RISER — PIPE CROSS-SECTION", fill=INK, size=11.5, weight=600)]
    for r, fill, _lbl in layers:
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{INK}" stroke-width="0.9"/>')
    # a small carcass interlock hint (dashes on the carcass ring)
    p.append(f'<circle cx="{cx}" cy="{cy}" r="40" fill="none" stroke="{INK2}" stroke-width="0.8" stroke-dasharray="3 3"/>')
    # leader labels to the right, stacked
    lxr = cx + 130
    for i, (r, _f, lbl) in enumerate(layers):
        ly = 66 + i * 26
        p.append(f'<line x1="{cx + r * 0.72:.1f}" y1="{cy - r * 0.72 + i * 3:.1f}" x2="{lxr - 6}" y2="{ly}" stroke="{DIM}" stroke-width="0.7"/>')
        p.append(f'<circle cx="{lxr - 6}" cy="{ly}" r="2" fill="{DIM}"/>')
        p.append(T(lxr, ly + 3, lbl, fill=INK2, size=9.5))
    p.append(f'<rect x="0" y="{VBH - 22}" width="{VBW}" height="22" fill="#eef2f2"/>')
    p.append(T(20, VBH - 7, "Illustrative unbonded flexible-pipe layers - an alternative riser type; the twin "
              "models the bonded steel catenary riser (see the pipe-section detail in the GA drawing).",
              fill=DIM, size=9.5))
    p.append("</svg>")
    return "".join(p)
