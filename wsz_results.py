"""Vorgerechnete Messreihen (data/wsz_results.json, erzeugt von tools/make_results.py aus wsz_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results.py)."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import wsz_constants as C

PATH = Path(os.environ.get("WSZ_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / C.RESULTS_FILE)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import wsz_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


def raster_rows(layout=None, weights=None) -> list[dict]:
    rows = load()["m1"]["raster"]
    return [r for r in rows if (layout is None or r["layout"] == layout) and (weights is None or r["weights"] == weights)]


def raster_cell(layout: str, weights: str, n: int) -> dict:
    (row,) = [r for r in load()["m1"]["raster"] if r["layout"] == layout and r["weights"] == weights and r["n"] == n]
    return row


def big_rows() -> list[dict]:
    return load()["m1"]["big"]


def big_cell(share: int) -> dict:
    (row,) = [r for r in big_rows() if r["share"] == share]
    return row


def triangles() -> dict:
    return load()["m1"]["triangles"]


def factor_rows() -> list[dict]:
    return load()["m3"]["factors"]


def factor_row(f: float) -> dict:
    (row,) = [r for r in factor_rows() if r["f"] == f]
    return row


def rate_rows() -> list[dict]:
    return load()["m3"]["rate"]


def rate_row(f: float) -> dict:
    (row,) = [r for r in rate_rows() if r["f"] == f]
    return row


def random_iteration_rows() -> list[dict]:
    return load()["m3"]["random"]


def start_rows() -> list[dict]:
    return load()["m6"]["rows"]


def start_row(n: int) -> dict:
    (row,) = [r for r in start_rows() if r["n"] == n]
    return row


def unit_rows() -> list[dict]:
    return load()["m4"]["units"]


def unit_row(s: float) -> dict:
    (row,) = [r for r in unit_rows() if r["s"] == s]
    return row


def rotation() -> dict:
    return load()["m4"]["rotation"]


def highdim_rows(case: str, mode: str) -> list[dict]:
    return load()["m5"][case][mode]


def highdim_cell(case: str, mode: str, d: int) -> dict:
    (row,) = [r for r in highdim_rows(case, mode) if r["d"] == d]
    return row
