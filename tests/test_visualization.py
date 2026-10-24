"""Figuren: Spuren, Pfeile, feste Achsen und Randfälle (kein Pfad, kein Tukey bei großem n, keine Streckung)."""

from dataclasses import replace

import numpy as np
import pytest

import wsz_constants as C
import wsz_evaluation as V
import wsz_results as R
import wsz_visualization as W


def _names(fig):
    return [t.name for t in fig.data if t.name]


def test_map_has_customers_arrows_ring_path_and_the_four_centres():
    a = V.analyse(V.Settings(n=10))
    fig = W.build_map(a)
    names = _names(fig)
    for expected in ("Kunden (Fläche = Gewicht)", "Schwerpunkt (Mittel)", "Koordinatenmedian", "Tiefster Datenpunkt (Tukey)", "Weber-Punkt (Ergebnis des Lösers)"):
        assert expected in names
    assert any(n.startswith("Gewicht von Kunde 1") for n in names) and any(n.startswith("Iterationspfad") for n in names)
    assert len(fig.layout.annotations) == 9 + 1                   # neun Pfeile (je anderer Kunde) und der Summenpfeil
    assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.xaxis.range is None                          # Rand über unsichtbare Eckmarker, nicht über einen festen Bereich


def test_map_arrow_tip_of_the_resultant_is_the_pull_vector_times_the_scale():
    import wsz_geomed as G
    a = V.analyse(V.Settings(n=5))
    fig = W.build_map(a)
    k = a.pull_customer
    vec = G.pull_vector(a.points, a.weights, k)

    def arrow(ann):
        return np.array([ann.x - ann.ax, ann.y - ann.ay])

    resultant, first = arrow(fig.layout.annotations[-1]), arrow(fig.layout.annotations[0])
    j0 = next(j for j in range(len(a.weights)) if j != k)
    assert resultant[0] * vec[1] == pytest.approx(resultant[1] * vec[0], abs=1e-9)              # gleiche Richtung wie die Zugkraft
    assert np.linalg.norm(resultant) / np.linalg.norm(first) == pytest.approx(np.linalg.norm(vec) / a.weights[j0], rel=1e-9)   # Längen ~ Zugkraft bzw. Gewicht


def test_map_without_path_transform_and_large_n():
    big = V.analyse(V.Settings(n=C.TUKEY_LIVE_MAX + 5, customer=7))
    names = _names(W.build_map(big))
    assert "Tiefster Datenpunkt (Tukey)" not in names
    assert not any("Weber-Punkt der Ausgangsdaten" in n for n in names)
    assert len(W.build_map(big).layout.annotations) == 1               # über 60 Kunden nur der Summenpfeil
    stretched = V.analyse(V.Settings(n=10, s=3.0))
    assert any("Weber-Punkt der Ausgangsdaten" in n for n in _names(W.build_map(stretched)))
    no_path = replace(V.analyse(V.Settings(n=10)), result=V.analyse(V.Settings(n=10)).result._replace(path=None))
    assert not any(n.startswith("Iterationspfad") for n in _names(W.build_map(no_path)))


def test_error_curve_hides_rounding_noise():
    fig = W.build_error_curve(np.array([1.0, 0.1, 1e-16, 0.0]), "x")
    assert fig.data[0].y.tolist() == [1.0, 0.1] and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"


def test_result_charts_use_the_stored_numbers():
    share = W.build_share_chart(R.raster_rows())
    assert len(share.data) == len(C.LAYOUTS) * 3 and share.data[0].x == tuple(C.N_RASTER)
    big = W.build_big_chart(R.big_rows())
    assert big.data[0].y[-1] == 100.0
    it = W.build_iterations_chart(R.factor_rows())
    assert it.data[0].x == tuple(C.FACTORS)
    hd = W.build_highdim_chart(R.highdim_rows("A", "1000"), "Titel")
    assert [t.name for t in hd.data] == ["Mittel", "Koordinatenmedian", "Geometrischer Median", "Einfacher Filter"]
    un = W.build_units_chart(R.unit_rows())
    assert list(un.data[0].x) == ["s = 1,5", "s = 2", "s = 5", "s = 10"]


def test_marker_size_is_area_proportional_between_7_and_40():
    sizes = W._marker_size(np.array([1.0, 4.0]))
    assert sizes[1] == pytest.approx(40.0) and sizes[0] == pytest.approx(7 + 33 * 0.5)
