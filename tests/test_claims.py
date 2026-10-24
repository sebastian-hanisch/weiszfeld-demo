"""Jede Zahl der README wird hier aus der vorgerechneten Messreihe (data/wsz_results.json) gebildet und in der README gesucht (DEMO-PLAYBOOK §4).

Die Datei selbst ist durch tests/test_results.py gegen die Quelldateien abgesichert, und tests/test_results.py rechnet Zellen live nach."""

import re
from pathlib import Path

import pytest

import wsz_results as R

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
LAYOUT_LABEL = {"Normal": "Normal", "Quadrat": "Quadrat", "Kern + Außenposten": "Kern + Außenposten", "Kern + Seitengruppe": "Kern + Seitengruppe"}


def de(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return de(100 * x, nd)


def th(x, nd=0):
    """Deutsche Zahl mit Punkt als Tausendertrenner und Komma als Dezimalzeichen."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def sci(x, digits=2):
    """3,8·10⁻¹⁰ als Text."""
    mantissa, exp = f"{x:.{digits - 1}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa.replace('.', ',')}·10{str(int(exp)).translate(sup)}"


def has(text):
    assert text in README, text


def test_h1_and_the_e1_table():
    cells = [pct(R.raster_cell("Normal", "gleich", n)["mean"]) for n in (3, 5, 10, 30, 100)]
    has(" / ".join(f"{c} %" for c in cells))
    for layout, kind, label in (("Normal", "gleich", "Normal, gleich"), ("Normal", "lognormal 1", "Normal, lognormal 1"), ("Quadrat", "gleich", "Quadrat, gleich"),
                                ("Kern + Außenposten", "gleich", "Kern + Außenposten, gleich"), ("Kern + Seitengruppe", "gleich", "Kern + Seitengruppe, gleich")):
        row = " | ".join(pct(R.raster_cell(layout, kind, n)["mean"]) for n in (3, 5, 10, 20, 30, 50, 100))
        has(f"| {label} | {row} |")
    has(f"mit je 400 Instanzen, Streuung zwischen den Seeds {de(100 * R.raster_cell('Normal', 'gleich', 100)['std'])} Prozentpunkte")
    assert R.raster_cell("Normal", "gleich", 100)["runs_per_seed"] == 400


def test_h2_big_customer():
    parts = [f"{pct(R.big_cell(s)['mean'])} % ({s} %)" for s in (5, 10, 20, 30, 40)]
    has(", ".join(parts) + f", {pct(R.big_cell(50)['mean'])} % (50 % und 60 %)")
    assert R.big_cell(50)["mean"] == R.big_cell(60)["mean"] == 1.0


def test_h3_triangles():
    t = R.triangles()
    has(f"{pct(t['Normal']['mean'])} % und {pct(t['Quadrat']['mean'])} %")
    has(f"in {t['torricelli']['instances']:,} Dreiecken".replace(",", ".") + f" höchstens {sci(t['torricelli']['max_distance'], 2)}")
    assert t["torricelli"]["worse_objective"] == 0


def test_h4_iterations_and_the_e3_table():
    f = {r["f"]: r for r in R.factor_rows()}
    has(f"Median {th(f[0.5]['median'], 1)} (f = 0,5), {th(f[1.0]['median'], 1)} (f = 1,0), {th(f[3.0]['median'])} (f = 3,0); {pct(f[1.0]['share_maxit'])} % erreichen bei f = 1,0 die Grenze")
    cells = " | ".join(th(f[x]["median"], 1 if f[x]["median"] % 1 else 0) for x in (0.5, 0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1, 1.5, 3.0))
    has(f"| Median der Iterationen | {cells} |")
    has(f"mit 96 Instanzen sind es {th(f[1.0]['median'], 1)} und {pct(f[1.0]['share_maxit'])} %")
    assert f[1.0]["instances"] == 96 and all(r["instances"] == 96 for r in R.factor_rows())
    assert f[1.0]["max"] == 100000


def test_stop_criterion_deceives_at_the_threshold():
    f = {r["f"]: r for r in R.factor_rows()}
    has(f"noch {sci(f[1.0]['median_error_at_stop'])} km vom Optimum entfernt; bei f = 3,0 sind es {sci(f[3.0]['median_error_at_stop'])} km")
    assert 0.9e-6 < f[0.999]["median_error_at_stop"] < 1.1e-6 and 0.9e-6 < f[1.001]["median_error_at_stop"] < 1.1e-6


def test_h5_rate():
    r = R.rate_row(1.0)
    q = r["quotient"]
    has(f"{de(q['10'], 3)} (k = 10), {de(q['100'], 3)} (100), {de(q['1000'], 3)} (1.000), {de(q['10000'], 4)} (10.000); Steigung von log e über log k: {de(r['slope_1000_10000'], 2).replace('-', '−')}")
    assert r["instances"] == 24 and all(v == 24 for v in r["valid"].values())
    has("an 24 Instanzen")


def test_random_instances():
    rows = R.random_iteration_rows()
    has("Median " + " / ".join(str(int(r["median"])) for r in rows) + " Iterationen für n = 3 / 5 / 10 / 30 / 100")
    has("größter Wert " + " / ".join(th(r["max"]) for r in rows))
    assert all(r["instances"] == 2000 for r in rows)


def test_h6_and_e4():
    rows = R.start_rows()
    has(" / ".join(f"{pct(r['classic_wrong'], 2)} %" for r in rows) + " (n = 3 / 5 / 10 / 30), Vardi–Zhang 0,00 %")
    has("klassisch und mit Epsilon-Schutz " + " / ".join(pct(r["classic_wrong"], 2) for r in rows) + " % (n = 3 / 5 / 10 / 30), Vardi–Zhang 0,00 %")
    assert all(r["classic_wrong"] == r["guard_wrong"] and r["vz_wrong"] == 0 and r["abort"] == 1.0 and r["centroid_classic_wrong"] == 0 for r in rows)
    has("klassisch mit Start im Schwerpunkt 0,00 %")


def test_h7_h8_and_the_e5_table():
    u = {r["s"]: r for r in R.unit_rows()}
    has(f"{de(u[5.0]['geo_mean'], 2)} Streuungen; Mehrweg {de(u[5.0]['excess_mean'], 2)} % im Mittel, größter Fall {de(u[5.0]['excess_max'])} %")
    has("| " + " | ".join(de(u[s]["geo_mean"], 2) for s in (1.5, 2.0, 5.0, 10.0)) + " |")
    has("| " + " | ".join(de(u[s]["excess_mean"], 2) for s in (1.5, 2.0, 5.0, 10.0)) + " |")
    assert all(r["instances"] == 600 for r in u.values())
    assert max(u[s]["coord_max"] for s in u) < 2e-15 and u[5.0]["tukey_max"] < 2e-15 and u[5.0]["tukey_runs"] == 60
    has("Abweichung höchstens 2·10⁻¹⁵")
    rot = R.rotation()
    has(f"{de(rot['coord_mean'], 2)} Streuungen (größter Fall {de(rot['coord_max'], 2)}); geometrischer Median 2·10⁻¹⁵")
    assert 1.5e-15 <= rot["geo_max"] < 2.5e-15 and rot["angle"] == 45.0


def test_h9_h10_and_the_e6_table():
    a = R.highdim_rows("A", "1000")
    b = R.highdim_rows("B", "1000")
    qa = [r["quotient_geo"] for r in a]
    has(f"zwischen {de(min(qa), 2)} und {de(max(qa), 2)}; Fall B: {de(b[0]['quotient_geo'], 2)} → {de(b[-1]['quotient_geo'], 2)}")
    has(f"Fall A, d = 200, n = 1.000: Filter {de(a[-1]['filter'], 3)} gegen {de(a[-1]['geometrischer_median'], 3)} (geometrischer Median) und {de(a[-1]['mittel'], 3)} (Mittel)")
    for key, label, rows, ds in (("mittel", "Fall A: Mittel", a, (2, 20, 200)), ("koordinatenmedian", "Fall A: Koordinatenmedian", a, (2, 20, 200)),
                                 ("geometrischer_median", "Fall A: geometrischer Median", a, (2, 20, 200)), ("filter", "Fall A: einfacher Filter", a, (2, 20, 200)),
                                 ("mittel", "Fall B: Mittel", b, (2, 20, 200)), ("geometrischer_median", "Fall B: geometrischer Median", b, (2, 20, 200))):
        by_d = {r["d"]: r for r in rows}
        has(f"| {label} | " + " | ".join(de(by_d[d][key], 3) for d in ds) + " |")
    assert all(r["runs"] == 40 for r in a + b)
    has("40 Läufe je Zelle")


def test_case_b_mean_beats_the_geometric_median_from_d_10():
    b = {r["d"]: r for r in R.highdim_rows("B", "1000")}
    assert b[5]["geometrischer_median"] < b[5]["mittel"]
    for d in (10, 20, 50, 100, 200):
        assert b[d]["mittel"] < b[d]["geometrischer_median"], d
    has(f"n = 1.000: {de(b[10]['mittel'], 3)} gegen {de(b[10]['geometrischer_median'], 3)} bei d = 10 und {de(b[200]['mittel'], 3)} gegen {de(b[200]['geometrischer_median'], 3)} bei d = 200")


def test_filter_against_the_mean_in_case_b():
    b = {r["d"]: r for r in R.highdim_rows("B", "1000")}
    assert b[50]["filter"] < b[50]["mittel"] and b[100]["filter"] > b[100]["mittel"] and b[200]["filter"] > b[200]["mittel"]
    has(f"bis d = 50 ({de(b[50]['filter'], 3)} gegen {de(b[50]['mittel'], 3)})")
    has(f"({de(b[100]['filter'], 3)} gegen {de(b[100]['mittel'], 3)} bei d = 100, {de(b[200]['filter'], 3)} gegen {de(b[200]['mittel'], 3)} bei d = 200)")
    n20 = {r["d"]: r for r in R.highdim_rows("B", "20d")}
    has(f"bei d = 100 gleich gut wie das Mittel ({de(n20[100]['filter'], 3)} gegen {de(n20[100]['mittel'], 3)}) und bei d = 200 schlechter ({de(n20[200]['filter'], 3)} gegen {de(n20[200]['mittel'], 3)})")
    assert abs(n20[100]["filter"] - n20[100]["mittel"]) < 0.005 and n20[200]["filter"] > n20[200]["mittel"]


def test_the_scratch_oracle_run_numbers_are_documented_as_a_scratch_run():
    """Die Zahlen des Scratchpad-Laufs (nicht im Repo reproduzierbar) stehen mit dem Vermerk 'Scratchpad' in der README."""
    has("Größerer Lauf im Scratchpad vor dem Bau")


def test_numbers_in_the_text_without_a_table():
    assert len(re.findall(r"\b455\b", README)) == 1                      # Zahl der gelesenen READMEs aus dem Plan (Abgleich 2026-10-24), keine Messung dieser Demo
    assert R.meta()["stand"] == "2026-10-24"
    assert pytest.approx(R.raster_cell("Normal", "gleich", 3)["mean"], abs=0.02) == 0.43


def test_mutation_state_counts_and_test_volume():
    import json
    import subprocess
    import sys
    state = json.loads((Path(__file__).resolve().parent / "mutation_state.json").read_text(encoding="utf-8"))
    killed = sum(v == "killed" for v in state.values())
    equivalent = sum(v == "equivalent" for v in state.values())
    assert (killed, equivalent, len(state)) == (87, 15, 102)
    has("gesichert sind jetzt 102 Stellen (87 gefangen, 15 als äquivalent festgehalten")
    has("104 Stellen. Im ersten Lauf überlebten 29 Mutanten die ganze Suite (Rate 72 %, Werkzeugausgabe)")
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(Path(__file__).resolve().parent)], capture_output=True, text=True, timeout=120).stdout
    count = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert count > 200
    has("über 200 Tests")
