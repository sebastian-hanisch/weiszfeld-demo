"""Die vorgerechneten Messreihen (data/wsz_results.json): vollständig, aktuell gegenüber den Quelldateien und in sich stimmig."""

import json

import pytest

import wsz_constants as C
import wsz_experiments as X
import wsz_results as R


def test_results_file_is_current_with_the_generator_sources():
    """Schlägt fehl, sobald sich wsz_rng, wsz_scenario, wsz_geomed, wsz_equivariance, wsz_highdim, wsz_experiments oder tools/make_results.py ändern:
    dann `python tools/make_results.py --part <m1|m3|m4|m5|m6>` und `--merge` neu laufen lassen."""
    assert R.meta()["erzeuger_hash"] == X.source_hash()
    assert R.meta()["quellen"] == list(X.SOURCES) and R.meta()["seeds"] == list(C.SEEDS)


def test_results_file_is_valid_utf8_json_with_all_parts():
    data = json.loads(R.PATH.read_text(encoding="utf-8"))
    assert set(data) == {"meta", "m1", "m3", "m4", "m5", "m6"}
    assert "Kern + Außenposten" in R.PATH.read_text(encoding="utf-8")        # Umlaute stehen im Klartext (kein ASCII-Escape)


def test_raster_is_complete():
    rows = R.raster_rows()
    assert len(rows) == len(C.LAYOUTS) * 3 * len(C.N_RASTER)
    for r in rows:
        assert 0.0 <= r["mean"] <= 1.0 and len(r["per_seed"]) == len(C.SEEDS) and r["runs_per_seed"] == X.runs_for(r["n"])
    assert R.raster_rows("Normal", "gleich") == [r for r in rows if r["layout"] == "Normal" and r["weights"] == "gleich"]
    assert R.raster_cell("Quadrat", "lognormal 1", 30)["n"] == 30


def test_big_customer_rows_follow_the_fifty_percent_rule():
    assert [r["share"] for r in R.big_rows()] == list(C.SHARES)
    assert all(r["mean"] == 1.0 for r in R.big_rows() if r["share"] >= 50)
    assert all(r["mean"] < 1.0 for r in R.big_rows() if r["share"] < 50)
    assert R.big_cell(40)["share"] == 40


def test_other_parts_have_their_rows():
    assert [r["f"] for r in R.factor_rows()] == list(C.FACTORS) and R.factor_row(1.0)["instances"] == 96
    assert [r["f"] for r in R.rate_rows()] == [0.5, 1.0, 1.5] and R.rate_row(1.0)["instances"] == 24
    assert [r["n"] for r in R.start_rows()] == [3, 5, 10, 30] and R.start_row(10)["instances"] == 2000
    assert [r["s"] for r in R.unit_rows()] == list(C.S_VALUES) and R.unit_row(5.0)["tukey_runs"] == 60
    assert R.rotation()["angle"] == 45.0 and [r["n"] for r in R.random_iteration_rows()] == [3, 5, 10, 30, 100]
    for case in ("A", "B"):
        for mode in ("1000", "20d"):
            assert [r["d"] for r in R.highdim_rows(case, mode)] == list(C.DIMS)
    assert R.highdim_cell("A", "1000", 20)["n"] == 1000
    assert set(R.triangles()) == {"Normal", "Quadrat", "torricelli"}


def test_stored_cells_are_reproduced_by_the_live_functions():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds)."""
    cell = R.raster_cell("Normal", "gleich", 3)
    assert X.share_on_point("Normal", "gleich", 3, cell["runs_per_seed"]) == {k: cell[k] for k in ("mean", "std", "per_seed", "runs_per_seed")}
    assert X.start_study(3, per_seed=500) == R.start_row(3)
    hd = R.highdim_cell("B", "1000", 5)
    again = X.highdim_cell("B", 5, "1000", 10)
    assert again["geometrischer_median"] == pytest.approx(hd["geometrischer_median"]) and again["mittel"] == pytest.approx(hd["mittel"])
