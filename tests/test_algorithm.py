"""Einheiten des Kerns an den Mini-Instanzen aus conftest.py, Erwartungswerte von Hand (siehe dort)."""

import math

import numpy as np
import pytest

import wsz_geomed as G


def test_objective_of_the_obtuse_triangle_at_the_corner_is_two_times_sqrt_1_04(obtuse):
    p, w = obtuse
    assert G.objective(p, w, p[2]) == pytest.approx(2 * math.sqrt(1.04))
    assert 2 * math.sqrt(1.04) == pytest.approx(2.0396, abs=1e-4)


def test_objective_uses_the_weights(abc):
    # c = A: 1·4 + 2·3 = 10 mit Gewichten (5, 1, 2)
    assert G.objective(abc, np.array([5.0, 1.0, 2.0]), abc[0]) == pytest.approx(4 + 6)


def test_weighted_centroid_by_hand(abc):
    # Gewichte (2, 1, 1): x = (0·2 + 4 + 0)/4 = 1, y = (0 + 0 + 3)/4 = 0,75
    assert G.weighted_centroid(abc, np.array([2.0, 1.0, 1.0])) == pytest.approx([1.0, 0.75])


def test_pull_at_a_is_norm_sqrt_2(abc):
    vec = G.pull_vector(abc, np.ones(3), 0)
    assert vec == pytest.approx([1.0, 1.0])
    norms, own = G.pull_norms(abc, np.array([1.0, 1.0, 1.0]))
    assert norms[0] == pytest.approx(math.sqrt(2))
    assert own.tolist() == [1.0, 1.0, 1.0]


def test_pull_at_the_obtuse_corner(obtuse):
    p, w = obtuse
    norms, _ = G.pull_norms(p, w)
    assert norms[2] == pytest.approx(0.4 / math.sqrt(1.04))


@pytest.mark.parametrize("w_a, on_a", [(1.40, False), (1.4142, False), (1.4143, True), (3.0, True)])
def test_threshold_of_weight_at_a_is_sqrt_2(abc, w_a, on_a):
    w = np.array([w_a, 1.0, 1.0])
    assert (G.on_point(abc, w) == 0) is on_a


def test_on_point_at_the_exact_threshold_counts_as_on_the_point(abc):
    assert G.on_point(abc, np.array([math.sqrt(2), 1.0, 1.0])) == 0


def test_obtuse_triangle_optimum_is_the_obtuse_corner(obtuse):
    p, w = obtuse
    assert G.on_point(p, w) == 2


def test_square_center_is_always_the_optimum(square_center):
    for w_center in (0.01, 1.0, 50.0):
        w = np.array([1.0, 1.0, 1.0, 1.0, w_center])
        assert G.on_point(square_center, w) == 4
        assert G.vardi_zhang(square_center, w, tol=1e-12).center == pytest.approx([0.0, 0.0])
    assert G.objective(square_center, np.ones(5), np.zeros(2)) == pytest.approx(4 * math.sqrt(2))


def test_collinear_points_median_is_the_middle_point(collinear):
    w = np.ones(3)
    assert G.on_point(collinear, w) == 1
    assert G.objective(collinear, w, collinear[1]) == pytest.approx(5.0)
    assert G.vardi_zhang(collinear, w, tol=1e-12).center == pytest.approx([1.0, 0.0])


def test_one_weiszfeld_step_by_hand(abc):
    """Ein Schritt ab c = (1, 1): c' = Σ x_i/d_i / Σ 1/d_i mit d = (√2, √10, √5)."""
    s = 1 / math.sqrt(2) + 1 / math.sqrt(10) + 1 / math.sqrt(5)
    expected = np.array([4 / math.sqrt(10) / s, 3 / math.sqrt(5) / s])
    r = G.weiszfeld(abc, np.ones(3), start=[1.0, 1.0], tol=1e30, maxit=1, keep=2)
    assert r.iterations == 1 and r.status == "ok"
    assert r.center == pytest.approx(expected)
    assert r.path.shape == (2, 2) and r.path[0] == pytest.approx([1.0, 1.0]) and r.path[1] == pytest.approx(expected)


def test_weiszfeld_finds_the_torricelli_point_of_the_equilateral_triangle(equilateral):
    r = G.weiszfeld(equilateral, np.ones(3), tol=1e-12)
    assert r.status == "ok"
    assert r.center == pytest.approx([1.0, math.sqrt(3) / 3], abs=1e-9)
    assert G.objective(equilateral, np.ones(3), r.center) == pytest.approx(2 * math.sqrt(3))
    assert G.fermat_point(*equilateral) == pytest.approx([1.0, math.sqrt(3) / 3])


def test_fermat_point_of_an_obtuse_triangle_is_the_corner(obtuse):
    assert G.fermat_point(*obtuse[0]) == pytest.approx([1.0, 0.2])


def test_classic_stops_on_a_data_point_vardi_zhang_does_not(abc):
    """Start auf C = (0, 3): ein Schritt des klassischen Verfahrens ist nicht definiert. Vardi–Zhang rechnet weiter und findet das Optimum."""
    w = np.ones(3)
    start = abc[2]
    classic = G.weiszfeld(abc, w, start=start, tol=1e-12)
    assert classic.status == "singulaer" and classic.center == pytest.approx(start)
    vz = G.vardi_zhang(abc, w, start=start, tol=1e-12)
    assert vz.status == "ok"
    assert G.objective(abc, w, vz.center) < G.objective(abc, w, start) - 0.1
    # A = (0, 0) ist hier nicht das Optimum (Gewicht 1 < √2): das Ergebnis ist ein innerer Punkt
    assert np.linalg.norm(vz.center - abc[0]) > 0.1


def test_epsilon_guard_does_not_rescue_the_start_on_a_data_point(abc):
    """Mit Absicherung auf 1e-12 zieht der Kunde mit Gewicht 1/1e-12 den Iterat an sich: das Verfahren bleibt auf C und meldet 'ok'."""
    r = G.weiszfeld(abc, np.ones(3), start=abc[2], tol=1e-10, guard=1e-12)
    assert r.status == "ok"
    assert np.linalg.norm(r.center - abc[2]) < 1e-9
    assert G.on_point(abc, np.ones(3)) is None


def test_vardi_zhang_stays_on_the_big_customer(abc):
    r = G.vardi_zhang(abc, np.array([2.0, 1.0, 1.0]), start=abc[0], tol=1e-12)
    assert r.status == "datenpunkt" and r.center == pytest.approx(abc[0])


def test_coincident_customers_hold_each_other_without_pull():
    p = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
    w = np.array([1.0, 1.0, 1.0])
    norms, own = G.pull_norms(p, w)
    assert own.tolist() == [2.0, 2.0, 1.0]
    assert norms.tolist() == pytest.approx([1.0, 1.0, 2.0])
    assert G.on_point(p, w) == 0


def test_path_keeps_only_the_first_iterates():
    p = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    r = G.weiszfeld(p, np.ones(3), tol=1e-12, keep=5)
    assert r.path.shape == (5, 2)
    assert G.weiszfeld(p, np.ones(3), tol=1e-12, keep=0).path is None
    assert G.vardi_zhang(p, np.ones(3), tol=1e-12, keep=7).path.shape == (7, 2)


def test_iteration_limit_is_reported():
    p = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    r = G.weiszfeld(p, np.ones(3), tol=0.0, maxit=7)
    assert r.status == "maxit" and r.iterations == 7


# --- Lücken aus dem ersten Mutationslauf (Überlebende der ganzen Suite) ------------------------------------------------------------------

def test_pull_vector_at_a_corner_away_from_the_origin(abc):
    """Zugkraft auf B = (4, 0): von A (−1, 0), von C (−4, 3)/5 = (−0,8; 0,6); Summe (−1,8; 0,6). Das fängt eine falsche Richtung (x_j − x_k statt x_k − x_j)."""
    assert G.pull_vector(abc, np.ones(3), 1) == pytest.approx([-1.8, 0.6])
    assert G.pull_vector(abc, np.array([1.0, 7.0, 2.0]), 1) == pytest.approx([-1.0 - 1.6, 1.2])      # Gewichte wirken: A mit 1, C mit 2


def test_pull_vector_counts_close_customers_and_skips_coincident_ones():
    p = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.0, 2.0]])           # Kunde 2 steht am Ort von Kunde 0, Kunde 1 ist nur 0,5 entfernt
    assert G.pull_vector(p, np.ones(4), 0) == pytest.approx([1.0, 1.0])
    assert np.isfinite(G.pull_vector(p, np.ones(4), 2)).all()


def test_pull_norms_in_chunks_equal_the_single_point_function():
    """n = 1500: die Zeilen werden in mehr als einem Stück gerechnet (2 000 000 // n Zeilen je Stück); die Grenzen zwischen den Stücken dürfen nichts verändern."""
    rng = np.random.default_rng(4)
    p, w = rng.normal(size=(1500, 2)), rng.lognormal(size=1500)
    norms, own = G.pull_norms(p, w)
    for k in (0, 1, 1332, 1333, 1334, 1499):
        assert norms[k] == pytest.approx(np.linalg.norm(G.pull_vector(p, w, k)))
    assert own.tolist() == w.tolist()


def test_excess_cost_with_a_total_path_below_one():
    from wsz_equivariance import excess_cost
    p = np.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0]])
    # Optimum 0,1 (Weg 0,3); bei 0,2 ist der Weg 0,2 + 0,1 + 0,1 = 0,4, also 1/3 mehr
    assert excess_cost(p, np.ones(3), np.array([0.2, 0.0]), np.array([0.1, 0.0])) == pytest.approx(100.0 / 3)


def test_vardi_zhang_damped_step_by_hand(abc):
    """Start auf A = (0, 0), Gewichte 1: η = 1, r = √2 > η, also γ = 1/√2. T(A) = Schwerpunkt von B und C mit Gewichten 1/4 und 1/3 = (12/7; 12/7).
    Erster Schritt: (1 − γ)·T(A) + γ·A = (1 − 1/√2)·12/7 in beiden Koordinaten."""
    r = G.vardi_zhang(abc, np.ones(3), start=abc[0], tol=0.0, maxit=1, keep=2)
    expected = (1 - 1 / math.sqrt(2)) * 12 / 7
    assert r.status == "maxit" and r.iterations == 1
    assert r.path[1] == pytest.approx([expected, expected])


def test_vardi_zhang_with_a_tie_between_pull_and_weight_stops_on_the_customer():
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    r = G.vardi_zhang(p, np.ones(2), start=p[0])
    assert r.status == "datenpunkt" and r.center.tolist() == [0.0, 0.0] and r.iterations == 1


def test_vardi_zhang_path_and_iteration_counts(abc):
    assert G.vardi_zhang(abc, np.ones(3), keep=0).path is None
    assert G.vardi_zhang(abc, np.ones(3), keep=1).path.shape == (1, 2)
    for limit in (1, 2, 7):
        r = G.vardi_zhang(abc, np.ones(3), tol=0.0, maxit=limit)
        assert r.iterations == limit and r.status == "maxit"
    assert G.weiszfeld(abc, np.ones(3), keep=1).path.shape == (1, 2)


def test_identical_customers_return_immediately_with_status_ok():
    p = np.array([[1.0, 1.0]] * 3)
    for r in (G.weiszfeld(p, np.ones(3)), G.vardi_zhang(p, np.ones(3))):
        assert r.status == "ok" and r.iterations == 0 and r.center.tolist() == [1.0, 1.0]


def test_vardi_zhang_damped_step_is_translation_equivariant(abc):
    """Dieselbe Handrechnung, alle Punkte um (10; −3) verschoben: der gedämpfte Schritt ist (1 − γ)·T + γ·c, nicht (1 − γ)·T − γ·c."""
    shift = np.array([10.0, -3.0])
    r = G.vardi_zhang(abc + shift, np.ones(3), start=abc[0] + shift, tol=0.0, maxit=1, keep=2)
    expected = (1 - 1 / math.sqrt(2)) * 12 / 7
    assert r.path[1] == pytest.approx(np.array([expected, expected]) + shift)
