"""Achsenstreckung, Drehung, Mehrweg, Koordinatenmedian und Halbraumtiefe: Handrechnungen und Eigenschaftstests (ORACLE-PLAYBOOK §5.2)."""

import math

import numpy as np
import pytest

import wsz_equivariance as E
import wsz_geomed as G
from wsz_scenario import Draws, layout_batch


def test_scale_axes_by_hand():
    p = np.array([[1.0, 2.0], [-3.0, 0.5]])
    assert E.scale_axes(p, 1.0, 5.0).tolist() == [[1.0, 10.0], [-3.0, 2.5]]
    assert E.scale_axes(p, 2.0, 1.0).tolist() == [[2.0, 2.0], [-6.0, 0.5]]


def test_rotation_by_hand():
    p = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert E.rotate(p, 90.0) == pytest.approx(np.array([[0.0, 1.0], [-2.0, 0.0]]), abs=1e-12)
    assert E.rotate(p, 180.0) == pytest.approx(-p, abs=1e-12)
    assert E.rotate(p, 0.0).tolist() == p.tolist()
    assert E.rotate(p, 45.0)[0] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])


def test_excess_cost_by_hand(collinear):
    """Punkte 0, 1, 5 auf der x-Achse: Optimum bei 1 (Weg 5). Bei 2 ist der Weg 2 + 1 + 3 = 6, also 20 % mehr."""
    w = np.ones(3)
    assert E.excess_cost(collinear, w, np.array([2.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(20.0)
    assert E.excess_cost(collinear, w, np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert E.excess_cost(np.zeros((2, 2)), np.ones(2), np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0   # Optimalweg 0


def test_weighted_median_1d_by_hand():
    assert E.weighted_median_1d(np.array([3.0, 1.0, 2.0]), np.ones(3)) == 2.0
    assert E.weighted_median_1d(np.array([3.0, 1.0, 2.0, 10.0]), np.ones(4)) == 2.5            # gerade Zahl: Mittelpunkt
    assert E.weighted_median_1d(np.array([1.0, 2.0, 10.0]), np.array([5.0, 1.0, 1.0])) == 1.0   # 5 von 7 liegen bei 1
    assert E.weighted_median_1d(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 2.0             # 3 von 4 liegen bei 2
    assert E.weighted_median_1d(np.array([1.0, 2.0]), np.array([2.0, 2.0])) == 1.5             # genau die Hälfte: Mittelpunkt


def test_weighted_median_matches_numpy_median_for_unit_weights_and_repeated_values():
    rng = np.random.default_rng(1)
    for n in range(1, 30):
        x = rng.normal(size=n)
        assert E.weighted_median_1d(x, np.ones(n)) == pytest.approx(np.median(x))
        k = rng.integers(1, 5, size=n)
        assert E.weighted_median_1d(x, k.astype(float)) == pytest.approx(np.median(np.repeat(x, k)))


def test_coordinate_median_is_not_a_data_point_in_general():
    p = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 1.0]])
    assert E.coordinate_median(p, np.ones(3)).tolist() == [1.0, 1.0]


def test_tukey_depths_by_hand(collinear, square_center):
    assert E.tukey_depths(collinear).tolist() == [1, 2, 1]
    assert E.tukey_depths(square_center).tolist() == [1, 1, 1, 1, 3]
    assert E.tukey_median_2d(square_center).tolist() == [0.0, 0.0]


def test_tukey_median_of_tied_points_is_their_centroid():
    p = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])      # alle vier Ecken haben Tiefe 1
    assert E.tukey_median_2d(p).tolist() == [1.0, 1.0]


def test_tukey_depth_of_coincident_points_counts_them_all():
    p = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
    assert E.tukey_depths(p).tolist() == [2, 2, 1]
    assert E.tukey_depths(np.array([[1.0, 1.0]] * 4)).tolist() == [4, 4, 4, 4]


def _sets(count, n, seed=1):
    d = Draws(seed)
    return layout_batch(d, "Normal", count, n)


def test_geometric_median_is_rotation_invariant_but_moves_under_axis_scaling():
    shifts_rot, shifts_scale = [], []
    for p in _sets(20, 12):
        w = np.ones(12)
        c = G.vardi_zhang(p, w, tol=1e-12).center
        c_rot = G.vardi_zhang(E.rotate(p, 37.0), w, tol=1e-12).center
        shifts_rot.append(np.linalg.norm(E.rotate(c[None], 37.0)[0] - c_rot))
        c_sc = G.vardi_zhang(E.scale_axes(p, 1.0, 5.0), w, tol=1e-12).center
        shifts_scale.append(np.linalg.norm(E.scale_axes(c[None], 1.0, 5.0)[0] - c_sc))
    assert max(shifts_rot) < 1e-8
    assert np.mean(shifts_scale) > 0.05


def test_coordinate_median_follows_axis_scaling_but_not_rotation():
    shifts_rot, shifts_scale = [], []
    for p in _sets(20, 12):
        w = np.ones(12)
        c = E.coordinate_median(p, w)
        shifts_scale.append(np.linalg.norm(E.scale_axes(c[None], 1.0, 5.0)[0] - E.coordinate_median(E.scale_axes(p, 1.0, 5.0), w)))
        shifts_rot.append(np.linalg.norm(E.rotate(c[None], 45.0)[0] - E.coordinate_median(E.rotate(p, 45.0), w)))
    assert max(shifts_scale) < 1e-12
    assert np.mean(shifts_rot) > 0.05


def test_tukey_median_follows_both_axis_scaling_and_rotation():
    for p in _sets(10, 15):
        c = E.tukey_median_2d(p)
        assert E.tukey_median_2d(E.scale_axes(p, 1.0, 5.0)) == pytest.approx(E.scale_axes(c[None], 1.0, 5.0)[0], abs=1e-9)
        assert E.tukey_median_2d(E.rotate(p, 45.0)) == pytest.approx(E.rotate(c[None], 45.0)[0], abs=1e-9)


def test_tukey_depth_is_invariant_under_a_shear():
    p = _sets(1, 25)[0]
    sheared = p @ np.array([[1.0, 0.0], [0.7, 1.0]])
    assert E.tukey_depths(p).tolist() == E.tukey_depths(sheared).tolist()
