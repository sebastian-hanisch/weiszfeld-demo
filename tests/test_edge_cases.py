"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n = 1, n = 2, alle Punkte gleich, Gewichtsgrenze, Gleichstände, Dubletten."""

import numpy as np
import pytest

import wsz_equivariance as E
import wsz_geomed as G


def test_single_customer_is_its_own_median():
    p, w = np.array([[2.0, 3.0]]), np.array([5.0])
    for r in (G.weiszfeld(p, w), G.vardi_zhang(p, w)):
        assert r.center.tolist() == [2.0, 3.0] and r.status == "ok" and r.iterations == 0
    assert G.on_point(p, w) == 0
    assert G.objective(p, w, p[0]) == 0.0


def test_two_customers_any_point_on_the_segment_is_optimal_and_the_heavier_one_wins():
    p = np.array([[0.0, 0.0], [4.0, 0.0]])
    assert G.on_point(p, np.array([1.0, 1.0])) == 0                 # Gleichstand: der kleinere Index, Zugkraft 1 ≤ Gewicht 1
    assert G.on_point(p, np.array([1.0, 3.0])) == 1                 # der schwerere Kunde hält den Standort
    assert G.on_point(p, np.array([3.0, 1.0])) == 0
    r = G.vardi_zhang(p, np.array([1.0, 3.0]), tol=1e-12)
    assert r.center == pytest.approx([4.0, 0.0])


def test_all_customers_at_the_same_place():
    p = np.array([[1.0, 1.0]] * 4)
    w = np.array([1.0, 2.0, 3.0, 4.0])
    assert G.weiszfeld(p, w).center.tolist() == [1.0, 1.0]
    assert G.on_point(p, w) == 0
    assert G.pull_norms(p, w)[0].tolist() == [0.0] * 4 and G.pull_norms(p, w)[1].tolist() == [10.0] * 4
    assert E.tukey_median_2d(p).tolist() == [1.0, 1.0]


def test_weight_bound_is_inclusive_in_the_threshold_test():
    """Zugkraft exakt gleich Gewicht: das Optimum liegt (auch) auf dem Kunden. Zwei Punkte, Gewichte 1 und 1."""
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert G.on_point(p, np.ones(2)) == 0


def test_iteration_limit_one_and_zero_tolerance():
    p = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert G.weiszfeld(p, np.ones(3), maxit=1, tol=0.0).iterations == 1
    assert G.vardi_zhang(p, np.ones(3), maxit=1, tol=0.0).status == "maxit"


def test_weighted_median_of_one_value_and_two_values():
    assert E.weighted_median_1d(np.array([4.0]), np.array([2.0])) == 4.0
    assert E.weighted_median_1d(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == 2.0
    assert E.tukey_depths(np.array([[0.0, 0.0], [1.0, 1.0]])).tolist() == [1, 1]
    assert E.tukey_depths(np.array([[0.0, 0.0]])).tolist() == [1]
