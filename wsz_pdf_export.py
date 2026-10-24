"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Mitten im Vergleich, Verdikte und eine einfache Karte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import wsz_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "√": "Wurzel", "‖": "|", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "ε": "eps"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def centre_rows(a: V.Analysis) -> list[tuple[str, np.ndarray, float]]:
    """(Name, Position, Gesamtweg) der Mitten, die die Karte zeigt."""
    import wsz_geomed as G
    rows = [("Schwerpunkt (Mittel)", a.centroid), ("Koordinatenmedian", a.coord_median)]
    if a.tukey is not None:
        rows.append(("Tiefster Datenpunkt (Tukey)", a.tukey))
    rows += [("Weber-Punkt (Löser)", a.result.center), ("Optimum", a.ref_center)]
    return [(name, c, G.objective(a.points, a.weights, c)) for name, c in rows]


def build_pdf(a: V.Analysis, verdicts: list[tuple[str, str]]) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Geometrischer Median: Wo steht der Weber-Punkt?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Layout {s.layout}, n = {s.n}, Gewichte {s.weights}, Löser {s.solver}, Start {s.start}, Toleranz 1e{s.tol_exp}, Iterationsgrenze {s.maxit}, "
                                f"Zufalls-Seed {s.seed}, Achsenfaktor {s.s:g}, Drehung {s.angle} Grad."))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Mitten im Vergleich", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    best = a.ref_value
    for name, c, value in centre_rows(a):
        extra = "" if best <= 0 else f", Mehrweg {V.de(100 * (value - best) / best, 3)} %"
        pdf.cell(0, 5.5, _latin(f"{name}: ({V.de(c[0], 3)}; {V.de(c[1], 3)}), Gesamtweg {V.de(value, 3)}{extra}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text))
        pdf.ln(1)
    # einfache Karte: Punkte mit Fläche ~ Gewicht, Mitten als Kreuze
    x0, y0, size = 30.0, pdf.get_y() + 4, 120.0
    if y0 + size > 280:
        pdf.add_page()
        y0 = 20.0
    p = a.points
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-9))

    def to_page(q):
        return x0 + (q[0] - lo[0]) / span * size, y0 + size - (q[1] - lo[1]) / span * size

    pdf.set_draw_color(180, 180, 180)
    pdf.rect(x0 - 3, y0 - 3, size + 6, size + 6)
    pdf.set_fill_color(110, 110, 110)
    for q, w in zip(p, a.weights):
        r = 0.8 + 3.2 * float(np.sqrt(w / a.weights.max()))
        px, py = to_page(q)
        pdf.ellipse(px - r, py - r, 2 * r, 2 * r, style="F")
    for name, c, _ in centre_rows(a):
        px, py = to_page(c)
        pdf.set_draw_color(47, 107, 101) if name.startswith(("Weber", "Optimum")) else pdf.set_draw_color(214, 39, 40)
        pdf.line(px - 2, py - 2, px + 2, py + 2)
        pdf.line(px - 2, py + 2, px + 2, py - 2)
    pdf.set_y(y0 + size + 6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Punktfläche ~ Gewicht; Kreuze: grün = Weber-Punkt und Optimum, rot = übrige Mitten. Luftlinie, Einheit km (bzw. Minuten auf der y-Achse nach Streckung)."))
    return bytes(pdf.output())
