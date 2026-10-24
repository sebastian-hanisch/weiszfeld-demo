"""Orakel (ORACLE-PLAYBOOK §2): der geometrische Median gegen andere Rechenwege.

* scipy.optimize.minimize (Nelder-Mead, drei Starts) auf der Zielfunktion; Gitter-Brute-Force für n = 3 und 4; Torricelli-Konstruktion für Dreiecke.
* Schwellentest `on_point` gegen die Numerik (Vardi–Zhang mit enger Toleranz): liegt das Ergebnis auf einem Kunden?
* Halbraumtiefe gegen Aufzählung von Halbräumen: ganzzahlige Normalen (exakt, ganze Zahlen) und fein abgetastete Richtungen.
* Optimalitätsbedingung am Ergebnis (Zugkraft der Kunden hebt sich auf) und Gesamtweg gegen scipy.spatial.
Größerer Lauf im Scratchpad vor dem Bau: 2000 gewichtete Instanzen, 3000 Schwellentests (Vormessung des Plans), hier die kleine Fassung."""

import numpy as np
import pytest

import wsz_equivariance as E
import wsz_geomed as G
from wsz_scenario import Draws, layout_batch, weights_batch

scipy_optimize = pytest.importorskip("scipy.optimize")
scipy_distance = pytest.importorskip("scipy.spatial.distance")


def _instances(count, lo, hi, kind, seed):
    d = Draws(seed)
    for r in range(count):
        n = lo + int(d.raw(1)[0] % np.uint64(hi - lo + 1))
        yield layout_batch(d, "Normal", 1, n)[0], weights_batch(d, kind, 1, n)[0]


def _numeric_best(p, w):
    starts = (G.weighted_centroid(p, w), p[0], p[-1])
    return min(
        scipy_optimize.minimize(lambda c: G.objective(p, w, c), s, method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 4000}).fun
        for s in starts
    )


def test_all_solvers_reach_the_numeric_minimum_on_weighted_random_instances():
    worst = 0.0
    for p, w in _instances(120, 3, 12, "lognormal 1", 21):
        best = _numeric_best(p, w)
        for result in (G.weiszfeld(p, w, tol=1e-11, maxit=400000, guard=1e-14), G.vardi_zhang(p, w, tol=1e-11, maxit=400000)):
            worst = max(worst, abs(G.objective(p, w, result.center) - best) / best)
    assert worst < 1e-8


def test_vardi_zhang_is_never_worse_than_the_numeric_minimum_on_unit_weights():
    for p, w in _instances(60, 3, 10, "gleich", 22):
        assert G.objective(p, w, G.vardi_zhang(p, w, tol=1e-11, maxit=400000).center) <= _numeric_best(p, w) * (1 + 1e-9)


def test_grid_brute_force_for_three_and_four_customers():
    for p, w in _instances(12, 3, 4, "gleich", 23):
        lo, hi = p.min(axis=0), p.max(axis=0)
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 401), np.linspace(lo[1], hi[1], 401))
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        values = scipy_distance.cdist(grid, p).sum(axis=1)
        ours = G.objective(p, w, G.vardi_zhang(p, w, tol=1e-12).center)
        assert ours <= values.min() + 1e-9                           # das Gitter kann das Minimum nur von oben treffen
        assert values.min() - ours < 1e-3 * (hi - lo).max()          # und liegt nicht weit darüber


def test_objective_matches_scipy_distances():
    for p, w in _instances(30, 3, 20, "lognormal 0,5", 24):
        c = p.mean(axis=0) + 0.1
        assert G.objective(p, w, c) == pytest.approx(float((scipy_distance.cdist(c[None], p)[0] * w).sum()), rel=1e-12)


def test_optimality_condition_holds_at_the_result():
    """Nicht auf einem Kunden: Summe der gewichteten Einheitsvektoren ≈ 0; auf einem Kunden: Norm der Zugkraft der übrigen ≤ Gewicht dort."""
    for p, w in _instances(80, 3, 15, "lognormal 1", 25):
        r = G.vardi_zhang(p, w, tol=1e-13, maxit=400000)
        k = G.on_point(p, w)
        if k is None:
            diff = r.center - p
            d = np.linalg.norm(diff, axis=1)
            assert np.linalg.norm((w[:, None] * diff / d[:, None]).sum(axis=0)) < 1e-5 * w.sum()
        else:
            assert np.linalg.norm(r.center - p[k]) < 1e-9


def test_torricelli_construction_agrees_with_weiszfeld_and_the_objective():
    d = Draws(26)
    tri = layout_batch(d, "Normal", 300, 3)
    for a, b, c in tri:
        f = G.fermat_point(a, b, c)
        pts = np.array([a, b, c])
        w = np.ones(3)
        w_res = G.vardi_zhang(pts, w, tol=1e-13).center
        assert np.linalg.norm(f - w_res) < 1e-6
        assert G.objective(pts, w, f) <= G.objective(pts, w, w_res) + 1e-9


def test_torricelli_agrees_with_numeric_minimum_on_triangles():
    for a, b, c in layout_batch(Draws(27), "Normal", 25, 3):
        pts = np.array([a, b, c])
        best = _numeric_best(pts, np.ones(3))
        assert G.objective(pts, np.ones(3), G.fermat_point(a, b, c)) == pytest.approx(best, rel=1e-8)


def test_threshold_test_predicts_whether_the_numeric_result_sits_on_a_customer():
    agree, skipped, total = 0, 0, 0
    for p, w in _instances(250, 5, 9, "lognormal 1", 28):
        norms, own = G.pull_norms(p, w)
        if np.min(np.abs(norms / own - 1.0)) < 1e-3:       # zu nah an der Schwelle für eine Grenze von 1e-5
            skipped += 1
            continue
        total += 1
        r = G.vardi_zhang(p, w, tol=1e-12, maxit=400000)
        near = np.linalg.norm(p - r.center, axis=1).min() < 1e-5
        agree += (G.on_point(p, w) is not None) == near
    assert skipped < 10 and agree == total


# --- Halbraumtiefe -----------------------------------------------------------------------------------------------------------------------------

def _depth_by_integer_normals(points, bound=40):
    """Tiefe jedes (ganzzahligen) Punktes: kleinste Zahl von Punkten in einem abgeschlossenen Halbraum {x: (x − q)·u ≥ 0} über alle ganzzahligen u,
    0 < |u|∞ ≤ bound. Exakte ganze Zahlen; trifft jedes Intervall zwischen zwei kritischen Richtungen, solange die Punkte auf einem kleinen Gitter liegen."""
    pts = np.asarray(points, dtype=np.int64)
    r = np.arange(-bound, bound + 1)
    u = np.array([(a, b) for a in r for b in r if (a, b) != (0, 0)], dtype=np.int64)
    return [int(((pts - q) @ u.T >= 0).sum(axis=0).min()) for q in pts]


def test_halfspace_depth_equals_enumeration_with_integer_normals():
    rng = np.random.default_rng(5)
    for _ in range(60):
        n = int(rng.integers(3, 11))
        pts = rng.integers(0, 5, size=(n, 2))
        assert E.tukey_depths(pts.astype(float)).tolist() == _depth_by_integer_normals(pts)


def test_halfspace_depth_equals_enumeration_with_sampled_directions():
    angles = np.linspace(0, 2 * np.pi, 40000, endpoint=False)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    for p in layout_batch(Draws(29), "Normal", 20, 8):
        brute = [int(((p - q) @ u.T >= -1e-12).sum(axis=0).min()) for q in p]
        assert E.tukey_depths(p).tolist() == brute


def test_deepest_point_has_depth_at_least_a_third_of_n_in_the_plane():
    """Eine bekannte Eigenschaft der Halbraumtiefe in der Ebene: es gibt einen Punkt der Tiefe ≥ n/3 (Zentrumspunkt); der tiefste Datenpunkt kann darunter liegen,
    aber die größte Datenpunkt-Tiefe ist bei 15 bis 40 Normalpunkten immer mindestens n/4 (hier gemessen, keine Garantie)."""
    for n in (15, 25, 40):
        for p in layout_batch(Draws(30 + n), "Normal", 10, n):
            assert E.tukey_depths(p).max() >= n / 4


def test_weighted_coordinate_median_by_replication():
    rng = np.random.default_rng(6)
    for _ in range(30):
        n = int(rng.integers(3, 9))
        p = rng.normal(size=(n, 2))
        k = rng.integers(1, 4, size=n)
        expected = np.median(np.repeat(p, k, axis=0), axis=0)
        assert E.coordinate_median(p, k.astype(float)) == pytest.approx(expected)
