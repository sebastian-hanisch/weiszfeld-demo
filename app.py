"""Geometrischer Median: Wo steht der Weber-Punkt? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Reihe „Mittelwerte und Lagemaße“. Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import wsz_constants as C
import wsz_equivariance as E
import wsz_evaluation as V
import wsz_experiments as X
import wsz_geomed as G
import wsz_results as R
from wsz_layout import dunkles_design, kopf_css
from wsz_pdf_export import build_pdf, centre_rows
from wsz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from wsz_visualization import (
    build_big_chart,
    build_error_curve,
    build_highdim_chart,
    build_iterations_chart,
    build_map,
    build_share_chart,
    build_units_chart,
)

st.set_page_config(page_title="Geometrischer Median – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


@st.cache_data(show_spinner=False)
def _share_live(layout, kind, n, share, runs):
    return X.share_on_point(layout, kind, n, runs, share=share / 100.0)


@st.cache_data(show_spinner=False)
def _error_curve(f, seed, steps):
    pts, w = X.threshold_instance(f, seed)
    ref = X.reference_center(pts, w, f)
    r = G.weiszfeld(pts, w, tol=0.0, maxit=steps, guard=1e-18, keep=steps + 1)
    stop = G.weiszfeld(pts, w, tol=X.STOP_TOL, maxit=X.MAXIT_SERIES, guard=1e-15)
    return np.linalg.norm(r.path - ref, axis=1), stop.iterations, stop.status, float(np.linalg.norm(stop.center - ref))


@st.cache_data(show_spinner=False)
def _highdim_live(case, d, mode):
    return X.highdim_cell(case, d, mode, 1)


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wsz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📍 Geometrischer Median: Wo steht der Weber-Punkt?")
st.markdown(
    """
**Stück 5 der Reihe „Mittelwerte und Lagemaße“.** Ein Lager soll gewichtete Kunden (Gewicht = Sendungsmenge) in der Ebene bedienen, und der **Gesamtweg** (Summe aus Gewicht mal Luftlinie) soll möglichst kurz sein:
Das ist der **geometrische Median**, in der Standortplanung der **Weber-Punkt**. Anders als Mittelwert und Koordinatenmedian hat er keine Formel; das **Weiszfeld-Verfahren** nähert ihn Schritt für Schritt an.
Die Demo zeigt, **wo der Punkt steht** (oft genau auf einem Kunden), **wann der Algorithmus dort stolpert** und warum **Achseneinheiten, Drehung und hohe Dimension** das Bild verändern.
Abgrenzung: Die Standortplanung wählt aus *diskreten* Kandidaten ([standortplanung-demo](https://github.com/sebastian-hanisch/standortplanung-demo)); hier ist der Standort ein *beliebiger* Punkt der Ebene.

Weiter unten: sechs Experimente (**🔬 E1 bis E6**), **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der geometrische Median", expanded=True):
    st.markdown(
        r"""
1. **Was gesucht ist.** Ein Punkt $c$, der die Summe $\sum_i w_i\,\|x_i - c\|$ der gewichteten Abstände zu allen Kunden klein macht. Der **Schwerpunkt** minimiert dagegen die Summe der *quadrierten* Abstände;
   der **Koordinatenmedian** nimmt jede Achse für sich.
2. **Zugkräfte.** Jeder Kunde zieht am Standort mit der Kraft $w_i$ in seine Richtung. Im Optimum heben sich alle Kräfte auf. Steht der Standort **auf** einem Kunden $k$, reicht es, dass die *übrigen* Kunden mit
   ihrer Zugkraft nicht stärker ziehen als $w_k$: Dann hält der Kunde den Standort fest. Auf der Karte sieht man das als Pfeil in einem Kreis (Radius $w_k$).
3. **Weiszfeld-Schritt.** Der neue Standort ist der Schwerpunkt mit Gewichten $w_i/\|x_i - c\|$: Wer nah ist, zählt mehr. Das lässt sich wiederholen, bis sich nichts mehr bewegt.
4. **Wo es kippt.** Steht ein Iterat *exakt* auf einem Kunden, ist der Schritt $0/0$. Und nahe der Schwelle zwischen „Standort zwischen den Kunden“ und „Standort auf dem Kunden“ wird das Verfahren sehr langsam.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🏭 Kunden")
    layout = st.selectbox("Layout", C.LAYOUTS, key="layout_select", help="Wie die Kunden in der Ebene liegen: normalverteilt, gleichmäßig im Quadrat, ein Außenposten bei (6; 6) oder eine Seitengruppe um (4; 1).")
    n = st.slider("Kunden n", *bounds("n_slider"), key="n_slider", help=f"Anzahl der Kunden. Über {C.TUKEY_LIVE_MAX} Kunden entfällt der tiefste Datenpunkt (Tukey), weil seine Rechnung mit n³ wächst.")
    weights = st.selectbox("Gewichte", C.WEIGHT_KINDS, key="weights_select", help="Gleich, lognormal gestreut, ein Großkunde mit gewähltem Gewichtsanteil oder ein Großkunde, dessen Gewicht ein Vielfaches seiner Schwelle ist.")
    share = st.slider("Gewichtsanteil des Großkunden (%)", *bounds("share_slider"), key="share_slider", disabled=weights != "Großkunde",
                      help="Nur bei Gewichtsart „Großkunde“: Anteil von Kunde 1 am Gesamtgewicht, die übrigen Kunden haben das Gewicht 1.")
    factor = st.slider("Gewicht in Vielfachen der Schwelle (f)", *bounds("factor_slider"), key="factor_slider", step=0.05, disabled=weights != "Großkunde an der Schwelle",
                       help="Nur bei Gewichtsart „Großkunde an der Schwelle“: Gewicht von Kunde 1 = f mal die Norm der Zugkraft der übrigen Kunden auf ihn. Ab f = 1 liegt der Standort auf Kunde 1.")
    st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.header("🧭 Einheiten und Drehung")
    s_factor = st.slider("Achsenfaktor s (y-Achse mal s)", *bounds("s_slider"), key="s_slider", step=0.5,
                         help="Streckt die y-Achse, etwa von Kilometern in Minuten. Der Weber-Punkt der gestreckten Daten weicht vom gestreckten Weber-Punkt ab.")
    angle = st.slider("Drehwinkel (Grad)", *bounds("angle_slider"), key="angle_slider", help="Dreht alle Kunden um den Ursprung. Der Weber-Punkt dreht mit, der Koordinatenmedian nicht.")
    st.header("⚙️ Löser")
    solver = st.selectbox("Löser", C.SOLVERS, key="solver_select", help="Klassisches Weiszfeld (bricht ab, wenn ein Iterat auf einem Kunden steht), mit Epsilon-Schutz (Abstände nach unten auf 1e-12 begrenzt) oder Vardi–Zhang (behandelt den Kunden am Iterat).")
    start = st.selectbox("Start", C.STARTS, key="start_select", help="Schwerpunkt, der dem Schwerpunkt nächste Kunde oder ein zufälliger Punkt im Rahmen der Kunden.")
    tol_exp = st.slider("Toleranz 10^e", *bounds("tol_slider"), key="tol_slider", help="Das Verfahren hält an, wenn der Schritt kürzer als 10^e ist.")
    maxit = st.select_slider("Iterationsgrenze", C.MAXIT_OPTIONS, key="maxit_select", help="Höchstzahl der Schritte.")
    st.header("🔍 Anzeige")
    if st.session_state.get("customer_slider", 1) > n:
        st.session_state["customer_slider"] = int(n)
    customer = st.slider("Kunde für die Zugkräfte", C.CUSTOMER_MIN, int(n), key="customer_slider", help="An diesem Kunden werden die Zugkräfte der übrigen eingezeichnet (Pfeile, Summenpfeil, Kreis vom Radius seines Gewichts).")

settings = V.Settings(layout, int(n), weights, int(share), float(factor), float(s_factor), int(angle), start, solver, int(tol_exp), int(maxit),
                      int(st.session_state["seed_input"]), int(customer))
sync_query_params({k: st.session_state[k] for k in ("layout_select", "n_slider", "weights_select", "share_slider", "factor_slider", "s_slider", "angle_slider",
                                                      "start_select", "solver_select", "tol_slider", "maxit_select", "seed_input", "customer_slider")})

with st.spinner("Rechne …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Gesamtweg am Ergebnis des Lösers", V.de(a.value, 2), delta=f"{V.de(a.value - a.ref_value, 3)} gegen das Optimum", delta_color="inverse" if a.value - a.ref_value > 5e-4 else "off")
m2.metric("Optimaler Gesamtweg", V.de(a.ref_value, 2), delta="Referenz", delta_color="off")
m3.metric("Iterationen", f"{a.result.iterations}", delta={"ok": "Schritt unter Toleranz", "maxit": "Grenze erreicht", "singulaer": "Abbruch auf Kunde", "datenpunkt": "Kunde hält fest"}[a.result.status], delta_color="off")
mean_value = G.objective(a.points, a.weights, a.centroid)
m4.metric("Gesamtweg am Schwerpunkt", V.de(mean_value, 2), delta=f"{V.de(mean_value - a.ref_value, 3)} gegen das Optimum", delta_color="inverse")
st.plotly_chart(build_map(a), width="stretch", key="map_chart")
st.caption("Fläche der Punkte = Gewicht. Orange: Zugkräfte der übrigen Kunden auf den gewählten Kunden (dünn: einzeln mit Länge = Gewicht, dick: Summe); der gestrichelte Kreis hat den Radius seines Gewichts. "
           "Liegt die Spitze des dicken Pfeils im Kreis, hält dieser Kunde den Standort fest. Blau: Pfad der ersten Iterate.")

rows = centre_rows(a)
st.dataframe(([{"Mitte": name, "x": c[0], "y": c[1], "Gesamtweg": value, "Mehrweg gegen das Optimum (%)": (100 * (value - a.ref_value) / a.ref_value if a.ref_value > 0 else 0.0)}
                           for name, c, value in rows]), hide_index=True, width="stretch",
             column_config={"x": st.column_config.NumberColumn(format="%.3f"), "y": st.column_config.NumberColumn(format="%.3f"), "Gesamtweg": st.column_config.NumberColumn(format="%.3f"),
                            "Mehrweg gegen das Optimum (%)": st.column_config.NumberColumn(format="%.3f")})
if a.tukey is None:
    st.caption(f"Der tiefste Datenpunkt (Tukey) wird nur bis {C.TUKEY_LIVE_MAX} Kunden berechnet.")

if a.pre_center is not None:
    st.info(f"**Streckung oder Drehung aktiv.** Der Weber-Punkt der Ausgangsdaten, mitgestreckt/-gedreht (hohler Stern), liegt {V.de(a.pre_distance, 2)} Streuungen der x-Werte vom Weber-Punkt der veränderten Daten; "
            f"wer die Mitte *vor* der Umrechnung bestimmt und dann die veränderten Wege fährt, hat {V.de(a.pre_excess_pct, 2)} % Mehrweg. "
            f"Der Koordinatenmedian weicht {V.de(a.coord_pre_distance, 2)} Streuungen ab" + (f", der tiefste Datenpunkt {V.de(a.tukey_pre_distance, 2)}." if a.tukey_pre_distance is not None else "."))

# PDF erst auf Klick
pdf_key = repr(settings)
if st.button("📄 PDF erstellen", key="pdf_make"):
    kind_loc, text_loc = V.verdict_location(a)
    kind_sol, text_sol = V.verdict_solver(solver, a.result.status, a.result.iterations, a.excess_pct, settings.maxit)
    st.session_state["pdf"] = (pdf_key, build_pdf(a, [(kind_loc, text_loc), (kind_sol, text_sol)]))
if st.session_state.get("pdf", (None, None))[0] == pdf_key:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="weber-punkt.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wo steht der Weber-Punkt, und kommt der Löser dort an?")
_verdict(V.verdict_location(a))
_verdict(V.verdict_solver(solver, a.result.status, a.result.iterations, a.excess_pct, settings.maxit))

with st.expander("🧰 Die drei Löser und der tiefste Datenpunkt erklärt"):
    st.markdown(
        r"""
| Verfahren | Was es tut | Was schiefgehen kann |
|---|---|---|
| **Klassisch (Weiszfeld)** | Schwerpunkt mit Gewichten $w_i/d_i$, $d_i$ = Abstand zu Kunde $i$ | Steht ein Iterat exakt auf einem Kunden, ist $d_i = 0$: Abbruch. Nahe der Schwelle sehr viele Schritte. |
| **Mit Epsilon-Schutz** | Wie klassisch, aber $d_i \ge 10^{-12}$ | Der Kunde zieht den Iterat mit dem Gewicht $w_i/10^{-12}$ an sich: Das Verfahren bleibt dort, meldet „konvergiert“ und liegt oft daneben. |
| **Vardi–Zhang** | Steht der Iterat auf Kunden mit Gesamtgewicht $\eta$, wird geprüft, ob die Zugkraft der übrigen höchstens $\eta$ ist (dann Optimum); sonst ein gedämpfter Schritt | Bleibt langsam nahe der Schwelle, bricht aber nicht ab und täuscht keine Konvergenz vor. |
| **Tiefster Datenpunkt (Tukey)** | Der Kunde, den die wenigsten Halbebenen „abschneiden“ können (Halbraumtiefe); bei Gleichstand der Schwerpunkt dieser Kunden | Kein Abstandsmaß, daher unabhängig von Einheiten und Drehung; aber ein Datenpunkt, nicht die Mitte der tiefsten Region (die einfache Fassung). |

Der Löser „Vardi–Zhang“ ist hier nach der Beschreibung des Verfahrens umgesetzt (die Originalarbeit wurde nicht eingesehen) und gegen `scipy.optimize` geprüft.
        """
    )

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); der Knopf „Nachmessen“ rechnet mit Ihren Einstellungen neu.")

# E1 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E1: Wie oft liegt der Standort auf einem Kunden?")
st.markdown("Gleich gewichtete oder gestreut gewichtete Zufallskunden; getestet wird je Instanz, ob die Zugkraft der übrigen auf mindestens einen Kunden höchstens dessen Gewicht ist.")
st.plotly_chart(build_share_chart(R.raster_rows()), width="stretch", key="e1_chart")
_verdict(V.verdict_e1(R.raster_cell("Normal", "gleich", 3), R.raster_cell("Normal", "gleich", 100), "Normalverteilt, gleiche Gewichte"))
_verdict(("info", f"Dreiecke (n = 3): Bei {V.pct(R.triangles()['Normal']['mean'])} % (normalverteilt) bzw. {V.pct(R.triangles()['Quadrat']['mean'])} % (Quadrat) der Dreiecke hat ein Winkel mindestens 120°; dann liegt "
                  f"der Standort auf dieser Ecke. Sonst findet die Torricelli-Konstruktion den Punkt: Sie weicht in {R.triangles()['torricelli']['instances']} Dreiecken höchstens {R.triangles()['torricelli']['max_distance']:.0e} vom Weiszfeld-Ergebnis ab."))
live_kind = weights if weights in ("gleich", "lognormal 0,5", "lognormal 1", "Großkunde") else None
live_runs = int(max(100, min(1000, C.LIVE_BUDGET // (4 * int(n) * int(n)))))
if st.button(f"Nachmessen mit Layout „{layout}“, n = {int(n)}, Gewichte „{weights}“ ({live_runs} Instanzen je Seed)", key="e1_live", disabled=live_kind is None):
    with st.spinner("Rechne vier Seeds …"):
        st.session_state["e1_result"] = (settings.layout, live_kind, int(n), int(share), live_runs, _share_live(layout, live_kind, int(n), int(share), live_runs))
if live_kind is None:
    st.caption("Für die Gewichtsart „Großkunde an der Schwelle“ gibt es keine Zufallsreihe: Das Gewicht hängt von der Instanz ab.")
res = st.session_state.get("e1_result")
if res is not None:
    st.info(f"Nachgemessen: Layout {res[0]}, Gewichte {res[1]}, n = {res[2]}: **{V.pct(res[5]['mean'])} %** der Instanzen mit Optimum auf einem Kunden "
            f"(Streuung zwischen den Seeds ±{V.pct(res[5]['std'])} Prozentpunkte, {res[4]} Instanzen je Seed).")

# E2 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E2: Ab welchem Anteil gehört der Standort dem Großkunden?")
st.markdown("20 normalverteilte Kunden, ein Großkunde mit dem angegebenen Gewichtsanteil am Gesamtgewicht, die übrigen Gewichte gleich. Die Preset-Knöpfe „Großkunde mit 40 %“ und „… 55 %“ zeigen je eine Instanz.")
st.plotly_chart(build_big_chart(R.big_rows()), width="stretch", key="e2_chart")
_verdict(V.verdict_e2(R.big_rows()))

# E3 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E3: Der Algorithmus an der Schwelle")
st.markdown(f"Sechs normalverteilte Kunden, Kunde 0 hat das Gewicht f mal seine Schwelle (ab f = 1 liegt das Optimum auf ihm), Start im Schwerpunkt, klassisches Verfahren mit Epsilon-Schutz 1e-15, Abbruch bei Schritt < 1e-9, Grenze {X.MAXIT_SERIES} Schritte; {R.factor_rows()[0]['instances']} Instanzen je Zeile.")
st.plotly_chart(build_iterations_chart(R.factor_rows()), width="stretch", key="e3_chart")
st.dataframe(([{"f": r["f"], "Median": r["median"], "95. Perzentil": r["p95"], "größter Wert": r["max"], "Anteil an der Grenze (%)": 100 * r["share_maxit"],
                            "Abstand zum Optimum beim Abbruch, Median (km)": r["median_error_at_stop"]} for r in R.factor_rows()]), hide_index=True, width="stretch",
             column_config={"f": st.column_config.NumberColumn(format="%.3f"), "Median": st.column_config.NumberColumn(format="%d"), "95. Perzentil": st.column_config.NumberColumn(format="%d"),
                            "größter Wert": st.column_config.NumberColumn(format="%d"), "Anteil an der Grenze (%)": st.column_config.NumberColumn(format="%.1f %%"),
                            "Abstand zum Optimum beim Abbruch, Median (km)": st.column_config.NumberColumn(format="%.2e")})
_verdict(V.verdict_e3(R.factor_rows()))
rate1, rate05, rate15 = R.rate_row(1.0), R.rate_row(0.5), R.rate_row(1.5)
st.markdown(
    f"**Fehlerkurve (Messung, keine Aussage über alle Instanzen).** Abstand der Iterate zum Optimum, {rate1['instances']} Instanzen, 20 000 Schritte. Weit von der Schwelle fällt er geometrisch: Der Quotient aufeinanderfolgender Fehler liegt bei k = 10 "
    f"im Median bei {V.de(rate05['quotient']['10'], 2)} (f = 0,5) bzw. {V.de(rate15['quotient']['10'], 2)} (f = 1,5), und bei k = 100 liegt der Fehler nur noch in {rate05['valid']['100']} bzw. {rate15['valid']['100']} von {rate1['instances']} Instanzen über 1e-9. "
    f"An der Schwelle (f = 1,0) ist der Quotient {V.de(rate1['quotient']['10'], 3)} (k = 10), {V.de(rate1['quotient']['100'], 3)} (k = 100), {V.de(rate1['quotient']['1000'], 4)} (k = 1000) und {V.de(rate1['quotient']['10000'], 4)} (k = 10000): er strebt gegen 1. "
    f"Die Steigung von log(Fehler) über log(k) zwischen k = 1000 und 10000 beträgt im Median {V.de(rate1['slope_1000_10000'], 2)}, der Fehler fällt also etwa wie 1/k."
)
e3_f = st.slider("Gewicht f für die eigene Fehlerkurve", 0.5, 3.0, 1.0, step=0.05, key="e3_f", help="Rechnet die Instanz mit dem Zufalls-Seed der Seitenleiste neu (sechs normalverteilte Kunden).")
if st.button("Fehlerkurve rechnen (20 000 Schritte)", key="e3_live"):
    with st.spinner("20 000 Schritte …"):
        st.session_state["e3_result"] = (float(e3_f), settings.seed, _error_curve(float(e3_f), settings.seed, 20000))
res = st.session_state.get("e3_result")
if res is not None:
    errors, its, status, err_stop = res[2]
    st.plotly_chart(build_error_curve(errors, f"f = {res[0]:g}"), width="stretch", key="e3_curve")
    st.caption(V.stop_caption(res[0], res[1], its, status, err_stop))

# E4 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E4: Der Start entscheidet")
st.markdown("Start auf dem Kunden, der dem Schwerpunkt am nächsten liegt (eine verbreitete Wahl), gleiche Gewichte, normalverteilte Kunden. „Nicht optimal“: Gesamtweg mehr als 1e-7 über dem Optimum.")
st.dataframe(([{"n": r["n"], "Instanzen": r["instances"], "klassisch bricht ab (%)": 100 * r["abort"], "klassisch nicht optimal (%)": 100 * r["classic_wrong"],
                            "mit Epsilon-Schutz nicht optimal (%)": 100 * r["guard_wrong"], "Vardi–Zhang nicht optimal (%)": 100 * r["vz_wrong"],
                            "klassisch, Start im Schwerpunkt, nicht optimal (%)": 100 * r["centroid_classic_wrong"]} for r in R.start_rows()]), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.2f %%") for c in ("klassisch bricht ab (%)", "klassisch nicht optimal (%)", "mit Epsilon-Schutz nicht optimal (%)",
                                                                                          "Vardi–Zhang nicht optimal (%)", "klassisch, Start im Schwerpunkt, nicht optimal (%)")})
_verdict(V.verdict_e4(R.start_row(10)))
if st.button("Die drei Löser auf Ihrer Instanz vergleichen (Start: nächster Kunde)", key="e4_live"):
    with st.spinner("Drei Läufe …"):
        st0 = X.nearest_point_start(a.points, a.weights)
        out = []
        for name in C.SOLVERS:
            r = V.run_solver(V.Settings(**{**settings.__dict__, "solver": name}), a.points, a.weights, st0)
            v = G.objective(a.points, a.weights, r.center)
            out.append({"Löser": name, "Status": r.status, "Iterationen": r.iterations, "Gesamtweg": v, "Mehrweg gegen das Optimum (%)": 100 * (v - a.ref_value) / a.ref_value if a.ref_value > 0 else 0.0})
        st.session_state["e4_result"] = (repr(settings), out)
res = st.session_state.get("e4_result")
if res is not None and res[0] == repr(settings):
    st.dataframe((res[1]), hide_index=True, width="stretch", column_config={"Gesamtweg": st.column_config.NumberColumn(format="%.3f"), "Mehrweg gegen das Optimum (%)": st.column_config.NumberColumn(format="%.3f")})

# E5 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E5: Einheiten und Drehung")
st.markdown(f"40 Kunden im Layout „Kern + Seitengruppe“ (30 um den Ursprung, 10 um (4; 1)), {R.unit_row(5.0)['instances']} Instanzen. Verglichen werden „Mitte bestimmen, dann y mit s strecken“ und „y strecken, dann die Mitte bestimmen“, gemessen in Streuungen der x-Werte. "
            "Der Mehrweg gilt, wenn die Wege wirklich mit s auf y gerechnet werden, die Mitte aber ohne Streckung gesucht wurde.")
st.plotly_chart(build_units_chart(R.unit_rows()), width="stretch", key="e5_chart")
st.dataframe(([{"Achsenfaktor s": r["s"], "Weber-Punkt: Abstand im Mittel (Streuungen)": r["geo_mean"], "Koordinatenmedian: größter Abstand": r["coord_max"],
                            "Mehrweg im Mittel (%)": r["excess_mean"], "Mehrweg größter Fall (%)": r["excess_max"]} for r in R.unit_rows()]), hide_index=True, width="stretch",
             column_config={"Achsenfaktor s": st.column_config.NumberColumn(format="%.1f"), "Weber-Punkt: Abstand im Mittel (Streuungen)": st.column_config.NumberColumn(format="%.3f"),
                            "Koordinatenmedian: größter Abstand": st.column_config.NumberColumn(format="%.1e"), "Mehrweg im Mittel (%)": st.column_config.NumberColumn(format="%.2f %%"),
                            "Mehrweg größter Fall (%)": st.column_config.NumberColumn(format="%.2f %%")})
_verdict(V.verdict_e5(R.unit_row(5.0), R.rotation()))
st.caption(f"Tiefster Datenpunkt (Tukey), s = 5, {R.unit_row(5.0)['tukey_runs']} Instanzen: größter Abstand {R.unit_row(5.0)['tukey_max']:.0e}. Der Standort wandert deutlich, die Kosten kaum: das Optimum ist flach. "
           "In der Seitenleiste lassen sich Achsenfaktor und Drehwinkel an der eigenen Instanz einstellen (Preset „Achsen in Minuten“).")

# E6 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E6: Hohe Dimension")
st.markdown("Standardnormale Punkte im $\\mathbb{R}^d$; 10 % davon sind durch einen engen Haufen ersetzt (Standardabweichung 0,5) auf der Diagonale. **Fall A:** Haufen weit weg (Abstand $5\\sqrt{d}$, das Mittel wandert mit). "
            f"**Fall B:** Haufen im festen Abstand 3. Fehler = Abstand des Schätzers vom wahren Zentrum (Ursprung), {R.highdim_rows('A', '1000')[0]['runs']} Läufe je Zelle. Der **einfache Filter** ist eine eigene vereinfachte Fassung (siehe Wo die Annahmen enden), kein Nachbau eines Literaturalgorithmus.")
mode = st.radio("Stichprobengröße", ("1000", "20d"), format_func=lambda m: "n = 1000" if m == "1000" else "n = 20·d (mindestens 200)", horizontal=True, key="e6_mode")
c1, c2 = st.columns(2)
c1.plotly_chart(build_highdim_chart(R.highdim_rows("A", mode), "Fall A: Haufen weit weg"), width="stretch", key="e6_a")
c2.plotly_chart(build_highdim_chart(R.highdim_rows("B", mode), "Fall B: Haufen im Abstand 3"), width="stretch", key="e6_b")
for case in ("A", "B"):
    st.dataframe(([{"Fall": case, "d": r["d"], "n": r["n"], "Mittel": r["mittel"], "Koordinatenmedian": r["koordinatenmedian"], "Geometrischer Median": r["geometrischer_median"],
                                "Einfacher Filter": r["filter"], "Geometrischer Median / (ε·√d)": r["quotient_geo"]} for r in R.highdim_rows(case, mode)]), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Mittel", "Koordinatenmedian", "Geometrischer Median", "Einfacher Filter", "Geometrischer Median / (ε·√d)")})
_verdict(V.verdict_e6(R.highdim_rows("A", "1000"), R.highdim_rows("B", "1000")))
d_live = st.select_slider("Dimension für den Nachlauf", (2, 5, 10, 20, 50, 100), value=20, key="e6_d")
case_live = st.radio("Fall für den Nachlauf", ("A", "B"), horizontal=True, key="e6_case")
if st.button("Nachlauf mit einem Lauf je Seed (4 Läufe)", key="e6_live"):
    with st.spinner("Rechne vier Läufe …"):
        st.session_state["e6_result"] = _highdim_live(case_live, int(d_live), "1000")
res = st.session_state.get("e6_result")
if res is not None:
    st.dataframe(([{"Fall": res["case"], "d": res["d"], "n": res["n"], "Läufe": res["runs"], "Mittel": res["mittel"], "Koordinatenmedian": res["koordinatenmedian"],
                                "Geometrischer Median": res["geometrischer_median"], "Einfacher Filter": res["filter"]}]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Luftlinie in der Ebene** | Straßen, Flüsse und Sperrungen fehlen; der Weber-Punkt kann in einem See oder auf einem Feld liegen. | Standortplanung mit diskreten Kandidaten |
| **Gewicht = Menge, keine Kapazität, keine Fixkosten** | Ein Lager hat Kosten, Kapazitäten und Mindestgrößen; der Gesamtweg ist nur ein Teil davon. | Kapazitierte Standortplanung |
| **Alle Abstände zählen linear (p = 1)** | Wer Ausreißer stärker oder schwächer gewichten will, braucht ein anderes p; die ganze Familie der Lp-Mitten ist ein eigenes Stück der Reihe. | Stück 3 der Reihe (Lp-Mitten) |
| **Simulierte Layouts** | Alle Häufigkeiten und Raten stammen aus den vier Layouts und Gewichtsarten dieser Demo; echte Kundenlagen können davon stark abweichen. | – |
| **Rate nur gemessen** | „Fehler fällt etwa wie 1/k an der Schwelle“ ist eine Messung an wenigen Instanzen, kein bewiesener Satz; Konvergenzbeweise werden nicht geführt. | – |
| **Der einfache Filter ist vereinfacht** | Er setzt voraus, dass die guten Punkte die Kovarianz I haben, und ist nicht der Algorithmus aus der Literatur; sein Abschneiden sagt nichts über dessen Güte. | Optimale Verfahren für hohe Dimension (nicht gebaut) |
| **Tiefster Datenpunkt statt Tukey-Median** | Der Tukey-Median im engeren Sinn ist die Mitte der tiefsten Region und muss kein Datenpunkt sein. | – |
| **Streuung zwischen den Seeds** | Jede Zelle nutzt vier feste Seeds; die angegebene Streuung ist die zwischen diesen Seeds, kein Konfidenzintervall. | – |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Kunden an Orten $x_1, \dots, x_n$ in der Ebene mit Gewichten $w_i > 0$. *Gesucht:* ein Punkt $c$, der den Gesamtweg minimiert.

| Symbol | Bedeutung |
|---|---|
| $x_i$ | Ort von Kunde $i$ (Kilometer, Luftlinie) |
| $w_i$ | Gewicht (Sendungsmenge) von Kunde $i$ |
| $c$ | gesuchter Standort |
| $u_{kj}$ | Einheitsvektor von $x_k$ zu $x_j$ |
| $\eta$ | Gesamtgewicht der Kunden am aktuellen Iterat |

**Problem.** Der Gesamtweg ist

$$F(c) = \sum_{i=1}^{n} w_i \,\|x_i - c\|.$$

Die Funktion $F$ ist konvex; liegen die Kunden nicht alle auf einer Geraden, ist das Minimum eindeutig.

**Optimalität zwischen den Kunden.** Steht das Minimum an keinem Kundenort, hebt sich die Zugkraft auf:

$$\sum_{i=1}^{n} w_i \,\frac{c - x_i}{\|c - x_i\|} = 0.$$

**Optimalität auf einem Kunden.** Das Minimum liegt genau dann an $x_k$, wenn die Zugkraft der übrigen höchstens das Gewicht von $k$ ist (Schwellentest):

$$\Big\| \sum_{j \ne k} w_j \, u_{kj} \Big\| \le w_k.$$

Wegen der Dreiecksungleichung gilt $\big\| \sum_{j \ne k} w_j u_{kj} \big\| \le \sum_{j \ne k} w_j$. Hat Kunde $k$ mindestens die Hälfte des Gesamtgewichts, ist $w_k \ge \sum_{j \ne k} w_j$, der Schwellentest also erfüllt: Der Standort liegt auf ihm.

**Weiszfeld-Schritt.**

$$c^{+} = \frac{\sum_i w_i \,x_i / \|x_i - c\|}{\sum_i w_i / \|x_i - c\|}.$$

Ist $c = x_k$ für einen Kunden, ist der Schritt nicht definiert.

**Vardi–Zhang-Schritt.** Sei $T(c)$ der Weiszfeld-Schritt über die Kunden, die nicht an $c$ stehen, $\eta$ das Gesamtgewicht der Kunden an $c$ und $r$ die Norm ihrer Zugkraft. Ist $r \le \eta$, ist $c$ optimal; sonst

$$c^{+} = (1 - \gamma)\,T(c) + \gamma\,c, \qquad \gamma = \min\!\Big(1, \frac{\eta}{r}\Big).$$

**Drei Kunden.** Hat das Dreieck einen Winkel von mindestens $120^\circ$, liegt der Standort auf dieser Ecke; sonst im Torricelli-Punkt, in dem alle drei Seiten unter $120^\circ$ erscheinen.

**Streckung der y-Achse um $s$.** Abstände ändern sich zu $\sqrt{\Delta x^2 + s^2 \Delta y^2}$, der Weber-Punkt der gestreckten Daten ist im Allgemeinen nicht der gestreckte Weber-Punkt. Bei einer Drehung bleiben alle Abstände erhalten, der Weber-Punkt dreht mit.

**Literatur.** Weiszfeld (1937), Kuhn (1973) und Vardi und Zhang (2000) gelten als die Grundlagen der Verfahren; die Originalarbeiten wurden für diese Demo **nicht eingesehen**. Die Verfahren sind nach ihrer Beschreibung umgesetzt
und gegen `scipy.optimize`, eine Gitter-Suche und die Torricelli-Konstruktion geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
