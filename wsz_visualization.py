"""Plotly-Figuren: Karte mit Zugkräften und Iterationspfad, Diagramme der Experimente. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import wsz_constants as C
import wsz_geomed as G

FONT = dict(size=13)


def _base(fig: go.Figure, height: int, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=55, r=15, t=30, b=45), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), legend=dict(orientation="h", y=-0.2), font=FONT)
    return fig


def _marker_size(w: np.ndarray) -> np.ndarray:
    """Flächentreu zum Gewicht: Durchmesser ∝ √w, zwischen 7 und 40 Pixel."""
    r = np.sqrt(w / w.max())
    return 7 + 33 * r


def _circle(center, radius, count=72):
    t = np.linspace(0, 2 * np.pi, count)
    return center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)


def build_map(a) -> go.Figure:
    """Karte der Kunden (Fläche = Gewicht), der vier Mitten, des Iterationspfads und der Zugkräfte auf den gewählten Kunden.

    Zugkräfte: von Kunde k geht für jeden anderen Kunden j ein Pfeil in Richtung j mit der Länge w_j (dünn); der dicke Pfeil ist ihre Summe. Der gestrichelte
    Kreis hat den Radius w_k. Liegt die Spitze des dicken Pfeils im Kreis, hält Kunde k den Standort fest (Optimum auf dem Kunden)."""
    p, w = a.points, a.weights
    fig = go.Figure()
    k = a.pull_customer
    span = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1e-9))
    labels = [str(i + 1) for i in range(len(w))] if len(w) <= 30 else None
    fig.add_trace(go.Scatter(
        x=p[:, 0], y=p[:, 1], mode="markers+text" if labels else "markers", text=labels, textposition="middle center", textfont=dict(size=10, color="white"),
        marker=dict(size=_marker_size(w), color=C.COLOR_POINT, line=dict(width=1, color="white")), name="Kunden (Fläche = Gewicht)",
        customdata=np.stack([np.arange(1, len(w) + 1), w, a.pull_share], axis=1),
        hovertemplate="Kunde %{customdata[0]:.0f}<br>Gewicht %{customdata[1]:.2f}<br>Zugkraft/Gewicht %{customdata[2]:.2f}<extra></extra>"))
    # Zugkräfte auf Kunde k
    vec = G.pull_vector(p, w, k)
    unit = 0.30 * span / max(float(np.linalg.norm(vec)), float(w[k]), float(w.max()), 1e-12)
    xk = p[k]
    ring_x, ring_y = _circle(xk, w[k] * unit)
    fig.add_trace(go.Scatter(x=ring_x, y=ring_y, mode="lines", line=dict(color=C.COLOR_PULL, dash="dash", width=1.5), name=f"Gewicht von Kunde {k + 1} (Kreisradius)", hoverinfo="skip"))
    diff = p - xk
    dist = np.linalg.norm(diff, axis=1)
    far = np.flatnonzero(dist > 0)
    if len(far) <= 60:
        for j in far:
            tip = xk + diff[j] / dist[j] * w[j] * unit
            fig.add_annotation(x=tip[0], y=tip[1], ax=xk[0], ay=xk[1], axref="x", ayref="y", xref="x", yref="y", showarrow=True, arrowhead=2, arrowsize=1,
                               arrowwidth=1, arrowcolor="rgba(224,123,0,0.45)")
    tip = xk + vec * unit
    fig.add_annotation(x=tip[0], y=tip[1], ax=xk[0], ay=xk[1], axref="x", ayref="y", xref="x", yref="y", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=3,
                       arrowcolor=C.COLOR_PULL)
    # Iterationspfad
    if a.result.path is not None and len(a.result.path) > 1:
        path = a.result.path
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", line=dict(color=C.COLOR_PATH, width=1), marker=dict(size=4), name=f"Iterationspfad (erste {len(path) - 1})",
                                 hovertemplate="Schritt %{pointNumber}<extra></extra>"))
    # Mitten
    def centre(name, c, symbol, color, size, hollow=False):
        marker = dict(symbol=symbol, size=size, color="rgba(0,0,0,0)" if hollow else color, line=dict(width=2.5 if hollow else 1.5, color=color))
        fig.add_trace(go.Scatter(x=[c[0]], y=[c[1]], mode="markers", marker=marker, name=name))
    centre("Schwerpunkt (Mittel)", a.centroid, "diamond", C.COLOR_MEAN, 13)
    centre("Koordinatenmedian", a.coord_median, "square", C.COLOR_COORD, 12)
    if a.tukey is not None:
        centre("Tiefster Datenpunkt (Tukey)", a.tukey, "x", C.COLOR_TUKEY, 13)
    if a.pre_center is not None:
        centre("Weber-Punkt der Ausgangsdaten, mitgestreckt/-gedreht", a.pre_center, "star", C.COLOR_WEBER, 17, hollow=True)
    centre("Weber-Punkt (Ergebnis des Lösers)", a.result.center, "star", C.COLOR_WEBER, 17)
    # Rand erzwingen (siehe DEMO-PLAYBOOK §3: scaleanchor nur mit autorange)
    pad = 0.12 * span
    fig.add_trace(go.Scatter(x=[p[:, 0].min() - pad, p[:, 0].max() + pad], y=[p[:, 1].min() - pad, p[:, 1].max() + pad], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    _base(fig, 560, "x (km)", "y (km)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_error_curve(errors: np.ndarray, title: str) -> go.Figure:
    """Abstand der Iterate zum Optimum, doppelt logarithmisch (Rundungsrauschen unter 1e-15 ausgeblendet)."""
    k = np.arange(len(errors))
    ok = errors > 1e-15
    fig = go.Figure(go.Scatter(x=k[ok], y=errors[ok], mode="lines", line=dict(color=C.COLOR_PATH), name=title))
    _base(fig, 340, "Iteration k", "Abstand zum Optimum (km)")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig


def build_share_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for (layout, kind) in dict.fromkeys((r["layout"], r["weights"]) for r in rows):
        sub = [r for r in rows if r["layout"] == layout and r["weights"] == kind]
        fig.add_trace(go.Scatter(x=[r["n"] for r in sub], y=[100 * r["mean"] for r in sub], mode="lines+markers", name=f"{layout}, {kind}",
                                 error_y=dict(type="data", array=[100 * r["std"] for r in sub], visible=True, thickness=1)))
    _base(fig, 420, "Kunden n", "Optimum liegt auf einem Kunden (%)")
    fig.update_xaxes(type="log", tickvals=[3, 5, 10, 20, 30, 50, 100], ticktext=["3", "5", "10", "20", "30", "50", "100"])
    fig.update_yaxes(range=[0, 100])
    return fig


def build_big_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[r["share"] for r in rows], y=[100 * r["mean"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_BIG),
                               error_y=dict(type="data", array=[100 * r["std"] for r in rows], visible=True, thickness=1), name="Standort auf dem Großkunden"))
    fig.add_vline(x=50, line_dash="dot", line_color="gray", annotation_text="50 %", annotation_position="top left")
    _base(fig, 360, "Gewichtsanteil des Großkunden (%)", "Standort auf dem Großkunden (%)")
    fig.update_yaxes(range=[0, 105])
    return fig


def build_iterations_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[r["f"] for r in rows], y=[r["median"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_PATH), name="Median der Iterationen"))
    fig.add_trace(go.Scatter(x=[r["f"] for r in rows], y=[r["p95"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_PULL, dash="dot"), name="95. Perzentil"))
    _base(fig, 380, "Gewicht des Kunden 1 in Vielfachen der Schwelle (f)", "Iterationen bis Schritt < 1e-9")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return fig


def build_highdim_chart(rows: list[dict], title: str) -> go.Figure:
    fig = go.Figure()
    names = (("mittel", "Mittel", C.COLOR_MEAN), ("koordinatenmedian", "Koordinatenmedian", C.COLOR_COORD), ("geometrischer_median", "Geometrischer Median", C.COLOR_WEBER), ("filter", "Einfacher Filter", C.COLOR_TUKEY))
    for key, name, color in names:
        fig.add_trace(go.Scatter(x=[r["d"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color)))
    _base(fig, 380, "Dimension d", "Fehler ‖Schätzer‖")
    fig.update_xaxes(type="log", tickvals=list(C.DIMS), ticktext=[str(d) for d in C.DIMS])
    fig.update_yaxes(type="log")
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return fig


def build_units_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[f"s = {r['s']:g}".replace(".", ",") for r in rows], y=[r["geo_mean"] for r in rows], marker_color=C.COLOR_WEBER, name="geometrischer Median"))
    _base(fig, 320, "Achsenfaktor", "Abstand in Streuungen der x-Werte")
    return fig
