"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und jede Aussage der Hilfetexte stimmt mit der Rechnung überein (Hilfetexte nach der Messung, DEMO-PLAYBOOK §4)."""

import math
from dataclasses import replace

import numpy as np
import pytest

import wsz_constants as C
import wsz_evaluation as V
import wsz_presets as P

STANDARD = next(iter(C.PRESETS))


def _settings(name):
    return V.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert V.Settings(**C.PRESETS[STANDARD]) == V.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_standard_sits_between_the_customers():
    a = V.analyse(_settings(STANDARD))
    assert a.ref_on is None and a.result.status == "ok" and a.excess_pct < 1e-6


def test_obtuse_triangle_preset_has_an_angle_over_120_degrees_at_the_optimum():
    a = V.analyse(_settings("Stumpfes Dreieck"))
    k = a.ref_on
    assert k is not None
    u, v = a.points[(k + 1) % 3] - a.points[k], a.points[(k + 2) % 3] - a.points[k]
    assert math.degrees(math.acos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))) > 120


def test_big_customer_presets_sit_on_either_side_of_the_fifty_percent_rule():
    a40, a55 = V.analyse(_settings("Großkunde mit 40 %")), V.analyse(_settings("Großkunde mit 55 %"))
    assert a40.ref_on is None and a40.pull_share[0] > 1.0
    assert a55.ref_on == 0 and a55.pull_share[0] < 1.0
    assert a40.weights[0] / a40.weights.sum() == pytest.approx(0.40) and a55.weights[0] / a55.weights.sum() == pytest.approx(0.55)


def test_threshold_preset_hits_the_iteration_limit_with_an_optimal_cost():
    a = V.analyse(_settings("Genau an der Schwelle"))
    assert a.ref_on == 0
    assert a.result.status == "maxit" and a.result.iterations == 100000
    assert a.excess_pct < 1e-4          # die Kosten sind längst optimal


def test_start_preset_aborts_on_a_customer_and_is_not_optimal():
    a = V.analyse(_settings("Start auf dem Kunden"))
    assert a.result.status == "singulaer" and a.result.iterations == 1
    assert a.excess_pct > 0.5
    fixed = V.analyse(replace(_settings("Start auf dem Kunden"), solver="Vardi–Zhang"))
    assert fixed.excess_pct < 1e-6


def test_axes_preset_moves_the_weber_point_but_not_the_cost_much():
    a = V.analyse(_settings("Achsen in Minuten"))
    assert a.pre_distance > 0.1
    assert 0.1 < a.pre_excess_pct < 5.0
    assert a.coord_pre_distance < 1e-9               # der Koordinatenmedian geht bei der Streckung mit
    assert a.tukey_pre_distance < 1e-9


def test_every_layout_and_weight_kind_changes_the_instance_and_the_share_on_a_customer():
    from wsz_experiments import share_on_point
    shares = {lay: share_on_point(lay, "gleich", 10, 600)["mean"] for lay in C.LAYOUTS}
    assert len(set(shares.values())) == len(C.LAYOUTS)
    by_weights = {kind: share_on_point("Normal", kind, 10, 600, share=0.4)["mean"] for kind in ("gleich", "lognormal 0,5", "lognormal 1", "Großkunde")}
    assert len(set(by_weights.values())) == 4
