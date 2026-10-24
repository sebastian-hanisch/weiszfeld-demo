"""Einheiten, Drehung und die Mitten, die dabei mitgehen oder nicht: Achsenstreckung, Drehung, Mehrweg, Koordinatenmedian und tiefster Datenpunkt.

Der geometrische Median ist drehinvariant (Abstände bleiben), geht bei ungleicher Achsenstreckung aber nicht mit (Abstände ändern sich).
Der Koordinatenmedian geht bei Achsenstreckung mit (jede Achse für sich), bei Drehung nicht. Der tiefste Datenpunkt (Halbraumtiefe nach Tukey)
geht bei beidem mit, weil Halbräume unter jeder umkehrbaren linearen Abbildung Halbräume bleiben."""

from __future__ import annotations

import numpy as np

from wsz_geomed import objective


def scale_axes(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Achsen strecken: x mal sx, y mal sy (zum Beispiel y-Achse von Kilometern in Minuten)."""
    return np.asarray(points, dtype=float) * np.array([sx, sy], dtype=float)


def rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Alle Punkte um den Ursprung gegen den Uhrzeigersinn drehen."""
    t = np.radians(degrees)
    m = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.asarray(points, dtype=float) @ m.T


def excess_cost(points: np.ndarray, weights: np.ndarray, c_candidate: np.ndarray, c_optimal: np.ndarray) -> float:
    """Mehrweg in Prozent des optimalen Gesamtwegs, wenn statt des Optimums `c_candidate` gewählt wird (0, wenn der Optimalweg 0 ist)."""
    best = objective(points, weights, c_optimal)
    if best <= 0:
        return 0.0
    return 100.0 * (objective(points, weights, c_candidate) - best) / best


def weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    """Gewichteter Median einer Zahlenreihe; liegt die Hälfte des Gewichts genau auf einer Seite, der Mittelpunkt der beiden Nachbarn (wie numpy.median)."""
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, dtype=float)[order]
    cum = np.cumsum(np.asarray(weights, dtype=float)[order])
    half = cum[-1] / 2.0
    i = int(np.searchsorted(cum, half * (1.0 - 1e-12), side="left"))
    if i + 1 < len(v) and abs(cum[i] - half) <= 1e-12 * cum[-1]:
        return float((v[i] + v[i + 1]) / 2.0)
    return float(v[i])


def coordinate_median(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Koordinatenweiser (gewichteter) Median: jede Achse für sich."""
    return np.array([weighted_median_1d(points[:, a], weights) for a in range(points.shape[1])])


def tukey_depths(points: np.ndarray) -> np.ndarray:
    """Halbraumtiefe jedes Punktes in der Ebene: die kleinste Zahl von Punkten (den Punkt selbst und Punkte am selben Ort eingeschlossen) in einem abgeschlossenen
    Halbraum, dessen Rand durch diesen Punkt geht. Es genügen die Ränder durch einen zweiten Punkt, ein wenig nach beiden Seiten gedreht; Punkte auf dem Rand
    zählen je nach Drehrichtung mit oder nicht. Vektorisiert, O(n³) Rechenschritte (n bis 150 live)."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    depth = np.full(n, n)
    for q in range(n):
        d = p - p[q]
        length = np.linalg.norm(d, axis=1)
        idx = np.flatnonzero(length > 0)
        if len(idx) == 0:
            continue
        normals = np.stack([-d[idx, 1], d[idx, 0]], axis=1)                  # senkrecht zu den Richtungen d_i
        side = d @ normals.T                                                  # side[j, i] = d_j · n_i
        along = d @ d[idx].T                                                  # along[j, i] = d_j · d_i
        tol = 1e-9 * np.outer(length, length[idx])
        on_line = np.abs(side) <= tol
        best = n
        for sign in (1.0, -1.0):
            strict = (sign * side > tol).sum(axis=0)
            for turn in (1.0, -1.0):
                count = strict + (on_line & (turn * along >= -tol)).sum(axis=0)
                best = min(best, int(count.min()))
        depth[q] = best
    return depth


def tukey_median_2d(points: np.ndarray) -> np.ndarray:
    """Tiefster Datenpunkt: der Punkt mit der größten Halbraumtiefe, bei Gleichstand der Schwerpunkt dieser Punkte. (Der Tukey-Median im engeren Sinn ist die
    Mitte der tiefsten Region und muss kein Datenpunkt sein; hier wird die einfache Fassung gezeigt, die unter Achsenstreckung und Drehung mitgeht.)"""
    depth = tukey_depths(points)
    return np.asarray(points, dtype=float)[depth == depth.max()].mean(axis=0)
