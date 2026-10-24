"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle steht hier. Verdikte sind reine Funktionen (Art, Text)."""

from types import SimpleNamespace

import numpy as np

import wsz_evaluation as ev


def _a(ref_on, weights, ratio, ref_value=12.3456):
    return SimpleNamespace(ref_on=ref_on, weights=np.array(weights), pull_share=np.array(ratio), ref_value=ref_value)


def test_de_and_pct_use_decimal_comma():
    assert ev.de(1.2345, 2) == "1,23" and ev.de(2.0, 0) == "2" and ev.pct(0.4321) == "43,2" and ev.pct(1.0, 0) == "100"


def test_location_on_a_customer_names_customer_weight_pull_and_cost():
    kind, text = ev.verdict_location(_a(1, [1.0, 2.5, 1.0], [1.5, 0.6, 2.0]))          # Zugkraft = 0,6 · 2,5 = 1,5
    assert kind == "success"
    assert "auf Kunde 2" in text and "Gewicht 2,50" in text and "Norm 1,50" in text and "12,35" in text


def test_location_between_customers_names_the_nearest_customer_to_the_threshold():
    kind, text = ev.verdict_location(_a(None, [1.0, 1.0, 1.0], [2.0, 1.25, 3.0]))
    assert kind == "info"
    assert "zwischen den Kunden" in text and "Kunde 2" in text and "1,25" in text and "12,35" in text


def test_solver_abort_reports_the_excess_or_a_lucky_optimum():
    kind, text = ev.verdict_solver("klassisch", "singulaer", 1, 2.644, 3000)
    assert kind == "error" and "nach 1 Schritten ab" in text and "2,64 %" in text
    kind, text = ev.verdict_solver("klassisch", "singulaer", 1, 0.0, 3000)
    assert kind == "error" and "Zufällig" in text and "über dem Optimum" not in text


def test_solver_iteration_limit_names_the_limit():
    kind, text = ev.verdict_solver("klassisch", "maxit", 100000, 0.00001234, 100000)
    assert kind == "warning" and "100000" in text and "0,0000 %" in text


def test_solver_vardi_zhang_stop_on_a_customer():
    kind, text = ev.verdict_solver("Vardi–Zhang", "datenpunkt", 1, 0.0, 3000)
    assert kind == "success" and "Vardi–Zhang erkennt nach 1 Schritten" in text


def test_solver_converged_but_wrong_is_an_error_and_converged_and_right_is_success():
    kind, text = ev.verdict_solver("mit Epsilon-Schutz", "ok", 2, 3.5, 3000)
    assert kind == "error" and "meldet Konvergenz" in text and "3,50 %" in text
    assert ev.verdict_solver("Vardi–Zhang", "ok", 47, 1e-4, 3000)[0] == "success"              # genau an der Schwelle 1e-4: noch Erfolg
    assert ev.verdict_solver("Vardi–Zhang", "ok", 47, 1.1e-4, 3000)[0] == "error"
    kind, text = ev.verdict_solver("Vardi–Zhang", "ok", 47, 0.0, 3000)
    assert "konvergiert nach 47 Schritten" in text and "0,000000 %" in text


def test_e1_names_both_ends_and_the_spread():
    kind, text = ev.verdict_e1({"n": 3, "mean": 0.432, "std": 0.004}, {"n": 100, "mean": 0.016, "std": 0.004}, "Beispiel")
    assert kind == "info" and "Beispiel" in text and "n = 3" in text and "43,2 %" in text and "n = 100" in text and "1,6 %" in text and "±0,4" in text


def test_e2_finds_the_first_share_with_certainty_and_the_last_below():
    rows = [{"share": 30, "mean": 0.22}, {"share": 40, "mean": 0.521}, {"share": 50, "mean": 1.0}, {"share": 60, "mean": 1.0}]
    kind, text = ev.verdict_e2(rows)
    assert kind == "success" and "**50 %**" in text and "Bei 40 % sind es 52,1 %" in text
    assert ev.verdict_e2([{"share": 30, "mean": 0.2}])[0] == "info"
    assert "Bei 30 %" not in ev.verdict_e2([{"share": 50, "mean": 1.0}])[1]            # ohne Zeile darunter kein Nachsatz


def test_e3_names_the_medians_the_share_at_the_limit_and_the_stop_error():
    rows = [{"f": 0.5, "median": 54.5, "instances": 96, "share_maxit": 0.0, "median_error_at_stop": 1e-9},
            {"f": 1.0, "median": 55970.5, "instances": 96, "share_maxit": 0.198, "median_error_at_stop": 5.6e-5},
            {"f": 3.0, "median": 19.0, "instances": 96, "share_maxit": 0.0, "median_error_at_stop": 1e-10}]
    kind, text = ev.verdict_e3(rows)
    assert kind == "warning" and "im Median 54 Iterationen" in text and "(f = 1,0) 55970" in text and "(f = 3,0) 19" in text
    assert "20 % der 96 Instanzen" in text and "56,0 Millionstel km" in text


def test_e4_names_the_shares():
    row = {"n": 10, "instances": 2000, "abort": 1.0, "classic_wrong": 0.877, "guard_wrong": 0.877, "vz_wrong": 0.0}
    kind, text = ev.verdict_e4(row)
    assert kind == "error" and "n = 10" in text and "**100 %**" in text and "2000 Instanzen" in text and "87,7 %" in text and "(87,7 %)" in text and "in 0,0 % daneben" in text


def test_e5_names_distance_excess_and_property_deviations():
    unit = {"s": 5.0, "geo_mean": 0.2558, "excess_mean": 0.4755, "excess_max": 3.07, "coord_max": 4.4e-16}
    rot = {"angle": 45.0, "coord_mean": 0.1642, "coord_max": 0.4702, "geo_max": 3.3e-15}
    kind, text = ev.verdict_e5(unit, rot)
    assert kind == "info"
    for part in ("s = 5", "0,26 Streuungen", "0,48 %", "3,1 %", "4e-16", "45°", "0,16 Streuungen", "0,47", "3e-15"):
        assert part in text, part


def test_e6_names_the_errors_and_quotients():
    a = [{"d": 2, "geometrischer_median": 0.18, "quotient_geo": 1.275}, {"d": 200, "geometrischer_median": 1.6351, "quotient_geo": 1.156}]
    b = [{"d": 2, "geometrischer_median": 0.179, "quotient_geo": 1.2651}, {"d": 200, "geometrischer_median": 0.666, "quotient_geo": 0.471}]
    kind, text = ev.verdict_e6(a, b)
    assert kind == "info"
    for part in ("0,18 (d = 2)", "1,64 (d = 200)", "zwischen 1,16 und 1,27", "von 0,18 auf 0,67", "von 1,27 auf 0,47"):
        assert part in text, part


def test_stop_caption_distinguishes_the_limit_from_a_normal_stop():
    text = ev.stop_caption(1.0, 3, 100000, "maxit", 1.5e-4)
    assert text == "f = 1,00, Zufalls-Seed 3: Das Verfahren erreicht die Grenze von 100000 Iterationen, ohne dass der Schritt unter 1e-9 fällt; der Iterat liegt dann noch 1,5e-04 km vom Optimum entfernt."
    text = ev.stop_caption(0.5, 7, 54, "ok", 2.0e-9)
    assert "hält bei Schritt < 1e-9 nach 54 Iterationen an" in text and "f = 0,50" in text and "2,0e-09 km" in text
