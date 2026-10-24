"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab."""

from dataclasses import replace

import numpy as np

import wsz_constants as C
import wsz_equivariance as E
import wsz_evaluation as V
import wsz_geomed as G
from wsz_scenario import Draws, layout_batch


def _start_on_customer_instances(count=40):
    for p in layout_batch(Draws(3), "Normal", count, 6):
        w = np.ones(6)
        yield p, w, p[int(np.argmin(np.linalg.norm(p - p.mean(axis=0), axis=1)))].copy()


def test_classic_differs_from_vardi_zhang_when_starting_on_a_customer():
    differ = sum(
        not np.allclose(G.weiszfeld(p, w, start=s, tol=1e-10).center, G.vardi_zhang(p, w, start=s, tol=1e-10).center, atol=1e-6)
        for p, w, s in _start_on_customer_instances()
    )
    assert differ > 20


def test_epsilon_guard_differs_from_vardi_zhang_when_starting_on_a_customer():
    differ = sum(
        not np.allclose(G.weiszfeld(p, w, start=s, tol=1e-10, guard=1e-12).center, G.vardi_zhang(p, w, start=s, tol=1e-10).center, atol=1e-6)
        for p, w, s in _start_on_customer_instances()
    )
    assert differ > 20


def test_all_solvers_agree_when_started_at_the_centroid():
    for p, w, _ in _start_on_customer_instances(15):
        results = [G.weiszfeld(p, w, tol=1e-12).center, G.weiszfeld(p, w, tol=1e-12, guard=1e-12).center, G.vardi_zhang(p, w, tol=1e-12).center]
        assert np.allclose(results[0], results[1], atol=1e-8) and np.allclose(results[0], results[2], atol=1e-8)


def test_every_solver_choice_in_the_app_changes_the_outcome_on_the_start_preset():
    base = V.Settings(**C.PRESETS["Start auf dem Kunden"])
    values = {name: V.analyse(replace(base, solver=name)) for name in C.SOLVERS}
    assert values["klassisch"].result.status == "singulaer" and values["Vardi–Zhang"].result.status != "singulaer"
    assert abs(values["klassisch"].value - values["Vardi–Zhang"].value) > 1e-3
    assert values["mit Epsilon-Schutz"].result.status == "ok" and abs(values["mit Epsilon-Schutz"].value - values["Vardi–Zhang"].value) > 1e-3


def test_every_start_choice_changes_the_first_iterate():
    base = V.Settings(solver="Vardi–Zhang")
    starts = {name: V.analyse(replace(base, start=name)).start for name in C.STARTS}
    assert not np.allclose(starts["Schwerpunkt"], starts["nächster Datenpunkt"]) and not np.allclose(starts["Schwerpunkt"], starts["zufälliger Punkt"])


def test_axis_factor_moves_the_geometric_median_but_not_the_tukey_median():
    base = V.Settings(layout="Kern + Seitengruppe", n=40, seed=2)
    plain, stretched = V.analyse(base), V.analyse(replace(base, s=5.0))
    assert stretched.pre_distance > 0.1 and plain.pre_distance is None
    assert stretched.tukey_pre_distance < 1e-9 and stretched.coord_pre_distance < 1e-9


def test_rotation_moves_the_coordinate_median_but_not_the_geometric_median():
    base = V.Settings(layout="Kern + Seitengruppe", n=40, seed=2, angle=45)
    a = V.analyse(base)
    assert a.coord_pre_distance > 0.05
    assert a.pre_distance < 1e-8 and a.tukey_pre_distance < 1e-9


def test_tukey_median_is_skipped_above_the_live_limit():
    assert V.analyse(V.Settings(n=C.TUKEY_LIVE_MAX + 1)).tukey is None
    assert V.analyse(V.Settings(n=C.TUKEY_LIVE_MAX)).tukey is not None


def test_depth_function_differs_between_collinear_and_general_points():
    assert E.tukey_depths(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])).max() == 2
