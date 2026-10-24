"""Hohe Dimension: Verunreinigung, geometrischer Median in R^d gegen scipy mit analytischem Gradienten, einfacher Filter."""

import numpy as np
import pytest

import wsz_geomed as G
import wsz_highdim as H
from wsz_scenario import Draws


def test_shift_norms_by_hand():
    assert H.shift_norm("A", 4) == pytest.approx(10.0)       # 5·√4
    assert H.shift_norm("A", 100) == pytest.approx(50.0)
    assert H.shift_norm("B", 4) == 3.0 and H.shift_norm("B", 100) == 3.0
    with pytest.raises(ValueError):
        H.shift_norm("C", 3)


def test_contaminate_replaces_the_requested_share_by_a_cluster_on_the_diagonal():
    d, n = 9, 2000
    p = H.contaminate("A", d, 0.1, n, Draws(1))
    assert p.shape == (n, d)
    cluster, rest = p[:200], p[200:]
    v = 15.0 * np.ones(d) / 3.0                                # ‖v‖ = 5·√9 = 15, Richtung (1, …, 1)/√9
    assert np.linalg.norm(cluster.mean(axis=0) - v) < 0.2
    assert np.linalg.norm(rest.mean(axis=0)) < 0.2
    assert cluster.std() == pytest.approx(0.5, abs=0.05) and rest.std() == pytest.approx(1.0, abs=0.05)
    assert H.contaminate("B", d, 0.0, 50, Draws(1)).shape == (50, d)


def test_mean_error_of_case_a_is_eps_times_shift():
    d = 16
    p = H.contaminate("A", d, 0.1, 20000, Draws(2))
    assert np.linalg.norm(p.mean(axis=0)) == pytest.approx(0.1 * 5 * np.sqrt(d), rel=0.05)      # 0,1 · 20 = 2


def test_geometric_median_nd_matches_scipy_with_analytic_gradient():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    for seed in range(1, 6):
        p = H.contaminate("B", 6, 0.1, 80, Draws(seed))

        def f(c):
            return np.linalg.norm(p - c, axis=1).sum()

        def grad(c):
            diff = c - p
            return (diff / np.linalg.norm(diff, axis=1)[:, None]).sum(axis=0)

        ref = scipy_optimize.minimize(f, np.median(p, axis=0), jac=grad, method="BFGS", options={"gtol": 1e-10})
        ours = H.geometric_median_nd(p, tol=1e-12, maxit=20000)
        assert f(ours) == pytest.approx(f(ref.x), rel=1e-10)
        assert np.linalg.norm(grad(ours)) < 1e-6


def test_geometric_median_nd_in_the_plane_agrees_with_vardi_zhang():
    p = np.random.default_rng(3).normal(size=(40, 2))
    assert H.geometric_median_nd(p, tol=1e-12, maxit=20000) == pytest.approx(G.vardi_zhang(p, np.ones(40), tol=1e-12).center, abs=1e-8)


def test_geometric_median_nd_resists_a_far_cluster_better_than_the_mean():
    p = H.contaminate("A", 10, 0.1, 1000, Draws(4))
    assert np.linalg.norm(H.geometric_median_nd(p)) < 0.5 * np.linalg.norm(p.mean(axis=0))


def test_filter_beats_mean_for_a_far_cluster_and_is_a_plain_mean_for_clean_data():
    p = H.contaminate("A", 20, 0.1, 1000, Draws(5))
    assert np.linalg.norm(H.simple_filter(p, 0.1)) < 0.3 * np.linalg.norm(p.mean(axis=0))
    clean = Draws(6).normal((500, 5))
    assert H.simple_filter(clean, 0.0) == pytest.approx(clean.mean(axis=0), abs=0.15)


def test_errors_returns_the_four_estimators():
    p = H.contaminate("B", 5, 0.1, 200, Draws(7))
    e = H.errors(p, 0.1)
    assert set(e) == {"mittel", "koordinatenmedian", "geometrischer_median", "filter"}
    assert e["mittel"] == pytest.approx(np.linalg.norm(p.mean(axis=0)))
    assert e["koordinatenmedian"] == pytest.approx(np.linalg.norm(np.median(p, axis=0)))
