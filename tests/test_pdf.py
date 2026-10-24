"""PDF-Export: erzeugt gültige PDF-Bytes mit Umlauten und Sonderzeichen, auch bei Streckung, ohne Tukey (n groß) und mit Abbruch."""

import wsz_constants as C
import wsz_evaluation as V
import wsz_pdf_export as P


def _verdicts(a):
    return [V.verdict_location(a), V.verdict_solver(a.settings.solver, a.result.status, a.result.iterations, a.excess_pct, a.settings.maxit)]


def test_pdf_for_every_preset_is_a_pdf():
    for name, preset in C.PRESETS.items():
        a = V.analyse(V.Settings(**preset))
        data = P.build_pdf(a, _verdicts(a))
        assert data[:5] == b"%PDF-" and len(data) > 1500, name


def test_pdf_without_tukey_and_with_a_long_text():
    a = V.analyse(V.Settings(n=C.TUKEY_LIVE_MAX + 20))
    assert [r[0] for r in P.centre_rows(a)] == ["Schwerpunkt (Mittel)", "Koordinatenmedian", "Weber-Punkt (Löser)", "Optimum"]
    assert P.build_pdf(a, _verdicts(a) * 5)[:5] == b"%PDF-"


def test_latin_replaces_characters_the_core_fonts_do_not_know():
    assert P._latin("Zugkraft ≤ Gewicht – ‖x‖ · 45° **fett**") == "Zugkraft <= Gewicht - |x| * 45 Grad fett"
    assert P._latin("Größe ä ö ü ß") == "Größe ä ö ü ß"


def test_centre_rows_by_hand():
    a = V.analyse(V.Settings(n=3))
    rows = P.centre_rows(a)
    assert rows[-1][0] == "Optimum" and abs(rows[-1][2] - a.ref_value) < 1e-12 and len(rows) == 5
