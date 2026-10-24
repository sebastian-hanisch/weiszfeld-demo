"""Messreihen-Funktionen (wsz_experiments.py) in kleiner Größe: Reproduzierbarkeit, Handwerte, Orakel für die Stapelrechnung und die qualitativen Befunde."""

import numpy as np
import pytest

import wsz_constants as C
import wsz_experiments as X
import wsz_geomed as G
from wsz_scenario import Draws, layout_batch, weights_batch


def test_cell_seed_is_stable_and_separates_cells():
    assert X.cell_seed(11, "Normal", "gleich", 3) == X.cell_seed(11, "Normal", "gleich", 3)
    seeds = {X.cell_seed(s, lay, 5) for s in C.SEEDS for lay in C.LAYOUTS}
    assert len(seeds) == len(C.SEEDS) * len(C.LAYOUTS)
    assert X.cell_seed(1, 0.5) != X.cell_seed(1, 1.0)


def test_source_hash_is_a_stable_hex_string():
    h = X.source_hash()
    assert len(h) == 16 and int(h, 16) >= 0 and h == X.source_hash()


def test_mean_std_by_hand():
    assert X.mean_std([1.0, 3.0]) == (2.0, 1.0)


def test_batch_pull_norms_equal_the_single_instance_function():
    d = Draws(5)
    pts, w = layout_batch(d, "Normal", 8, 7), weights_batch(d, "lognormal 1", 8, 7)
    batch = G.pull_norms_batch(pts, w)
    for r in range(8):
        assert batch[r] == pytest.approx(G.pull_norms(pts[r], w[r])[0])


def test_share_on_point_is_reproducible_and_has_the_expected_size():
    a = X.share_on_point("Normal", "gleich", 3, 1500)
    assert a == X.share_on_point("Normal", "gleich", 3, 1500)
    assert 0.40 < a["mean"] < 0.47 and len(a["per_seed"]) == 4 and a["runs_per_seed"] == 1500


def test_share_on_point_agrees_with_the_single_instance_threshold_test():
    d = Draws(X.cell_seed(11, "Normal", "gleich", 5, 0.4))
    pts, w = layout_batch(d, "Normal", 300, 5), weights_batch(d, "gleich", 300, 5)
    hits = sum(G.on_point(pts[r], w[r]) is not None for r in range(300))
    assert X.share_on_point("Normal", "gleich", 5, 300, seeds=(11,))["mean"] == hits / 300


def test_big_customer_share_by_the_fifty_percent_rule():
    assert X.share_on_point("Normal", "Großkunde", 20, 600, share=0.5, only_first=True)["mean"] == 1.0
    assert X.share_on_point("Normal", "Großkunde", 20, 600, share=0.7, only_first=True)["mean"] == 1.0
    low = X.share_on_point("Normal", "Großkunde", 20, 600, share=0.05, only_first=True)["mean"]
    mid = X.share_on_point("Normal", "Großkunde", 20, 600, share=0.3, only_first=True)["mean"]
    assert low < mid < 1.0


def test_share_on_point_falls_with_n_and_rises_with_weight_spread():
    n3, n30 = (X.share_on_point("Normal", "gleich", n, 600)["mean"] for n in (3, 30))
    assert n3 > 4 * n30
    assert X.share_on_point("Normal", "lognormal 1", 10, 600)["mean"] > X.share_on_point("Normal", "gleich", 10, 600)["mean"]


def test_triangle_study_small():
    t = X.triangle_study(runs=3000, check=40)
    assert 0.40 < t["Normal"]["mean"] < 0.48 and 0.35 < t["Quadrat"]["mean"] < 0.45
    assert t["torricelli"]["max_distance"] < 1e-6 and t["torricelli"]["worse_objective"] == 0


def test_threshold_instance_pull_equals_weight_over_f():
    for f in (0.5, 1.0, 2.0):
        pts, w = X.threshold_instance(f, 99)
        norms, own = G.pull_norms(pts, w)
        assert norms[0] / own[0] == pytest.approx(1.0 / f) and len(w) == 6 and (w[1:] == 1).all()


def test_reference_center_is_the_optimum():
    pts, w = X.threshold_instance(1.5, 3)
    assert X.reference_center(pts, w, 1.5).tolist() == pts[0].tolist()
    pts, w = X.threshold_instance(0.5, 3)
    c = X.reference_center(pts, w, 0.5)
    diff = c - pts
    assert np.linalg.norm((w[:, None] * diff / np.linalg.norm(diff, axis=1)[:, None]).sum(axis=0)) < 1e-6


def test_iterations_peak_at_the_threshold():
    rows = {r["f"]: r for r in X.iterations_by_factor(factors=(0.5, 1.0, 3.0), per_seed=2)}
    assert rows[1.0]["median"] > 100 * rows[0.5]["median"] > 0 and rows[1.0]["median"] > 100 * rows[3.0]["median"]
    assert rows[1.0]["median_error_at_stop"] > 1000 * rows[3.0]["median_error_at_stop"]       # der Abbruch täuscht an der Schwelle


def test_rate_study_threshold_is_sublinear_and_far_from_it_is_linear():
    rows = {r["f"]: r for r in X.rate_study(factors=(1.0, 1.5), per_seed=1)}
    q = rows[1.0]["quotient"]
    assert q["10"] < q["100"] < q["1000"] < q["10000"] < 1.0
    assert -1.2 < rows[1.0]["slope_1000_10000"] < -0.8
    assert rows[1.5]["quotient"]["10"] < 0.8 and rows[1.5]["valid"]["100"] == 0


def test_random_iterations_have_a_long_tail():
    row = X.random_iterations(ns=(5,), per_seed=100)[0]
    assert row["median"] < 100 < row["max"] and row["instances"] == 400


def test_start_study_matches_the_documented_behaviour():
    r = X.start_study(5, per_seed=40)
    assert r["abort"] == 1.0 and r["classic_wrong"] == r["guard_wrong"] and r["classic_wrong"] > 0.5
    assert r["vz_wrong"] == 0.0 and r["centroid_classic_wrong"] == 0.0


def test_unit_study_geometric_median_moves_but_coordinate_median_does_not():
    r = X.unit_study(5.0, per_seed=10, tukey_runs=3)
    assert r["geo_mean"] > 0.05 and r["coord_max"] < 1e-12 and r["tukey_max"] < 1e-9 and r["excess_mean"] > 0
    assert X.unit_study(1.0, per_seed=5)["geo_mean"] < 1e-8


def test_rotation_study_by_property():
    r = X.rotation_study(per_seed=10)
    assert r["geo_max"] < 1e-9 and r["mean_max"] < 1e-9 and r["coord_mean"] > 0.05


def test_skewed_cloud_has_the_documented_composition():
    p = X.skewed_cloud(Draws(1), 400)
    assert p.shape == (400, 40, 2)
    assert np.allclose(p[:, 30:].mean(axis=(0, 1)), [4.0, 1.0], atol=0.1) and np.allclose(p[:, :30].mean(axis=(0, 1)), 0.0, atol=0.1)


def test_highdim_cell_orders_the_estimators_in_case_a():
    r = X.highdim_cell("A", 20, "1000", 1)
    assert r["n"] == 1000 and r["runs"] == 4
    assert r["mittel"] > r["koordinatenmedian"] > r["geometrischer_median"] > r["filter"]
    assert 0.9 < r["quotient_geo"] < 1.6
    assert X.highdim_cell("B", 5, "20d", 1)["n"] == 200 and X.highdim_cell("B", 50, "20d", 1)["n"] == 1000
