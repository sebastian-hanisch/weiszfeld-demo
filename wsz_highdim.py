"""Hohe Dimension: verunreinigte Normalverteilung, geometrischer Median in beliebiger Dimension und ein einfacher Filter.

Daten: n Punkte aus N(0, I_d); der Anteil ε der Punkte wird durch einen engen Haufen N(v, 0,25·I_d) ersetzt, v liegt auf der Diagonale.
Fall A: ‖v‖ = 5·√d (der Haufen liegt weit weg, das Mittel wandert mit ε·‖v‖). Fall B: ‖v‖ = 3 fest (der Haufen liegt nah, gleich weit in jeder Dimension).
Das wahre Zentrum ist der Ursprung; der Fehler eines Schätzers ist seine Norm."""

from __future__ import annotations

import numpy as np

from wsz_scenario import Draws

CASES = ("A", "B")


def shift_norm(case: str, d: int) -> float:
    """Länge der Verschiebung v des Haufens: Fall A 5·√d, Fall B 3."""
    if case == "A":
        return 5.0 * np.sqrt(d)
    if case == "B":
        return 3.0
    raise ValueError(f"unbekannter Fall: {case}")


def contaminate(case: str, d: int, eps: float, n: int, draws: Draws) -> np.ndarray:
    """n Punkte im R^d: Standardnormal, die ersten round(ε·n) durch den Haufen N(v, 0,25·I) ersetzt."""
    points = draws.normal((n, d))
    m = int(round(eps * n))
    v = shift_norm(case, d) * np.ones(d) / np.sqrt(d)
    points[:m] = v + 0.5 * draws.normal((m, d))
    return points


def geometric_median_nd(points: np.ndarray, tol: float = 1e-9, maxit: int = 3000) -> np.ndarray:
    """Geometrischer Median in beliebiger Dimension (gleiche Gewichte): Weiszfeld ab dem Koordinatenmedian, Abstände nach unten auf 1e-12 begrenzt.
    In Dimension d ≥ 3 trifft der Iterat einen Datenpunkt praktisch nie; die Absicherung ist dort nur ein Rechenschutz."""
    c = np.median(points, axis=0)
    for _ in range(maxit):
        d = np.maximum(np.linalg.norm(points - c, axis=1), 1e-12)
        w = 1.0 / d
        c_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(c_new - c) < tol:
            return c_new
        c = c_new
    return c


def simple_filter(points: np.ndarray, eps: float, rounds: int = 40) -> np.ndarray:
    """Einfacher Filter (eigene vereinfachte Fassung, nicht der Algorithmus aus der Literatur): solange der größte Eigenwert der Kovarianz über 1 + 3ε liegt,
    wird 1 % der Punkte mit der größten Projektion auf diese Richtung entfernt; ausgegeben wird das Mittel der übrigen. Setzt voraus, dass die guten
    Punkte Kovarianz I haben (hier der Fall)."""
    pts = np.array(points, dtype=float)
    for _ in range(rounds):
        mu = pts.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(pts.T))
        if values[-1] < 1.0 + 3.0 * eps:
            break
        proj = np.abs((pts - mu) @ vectors[:, -1])
        drop = max(1, int(0.01 * len(pts)))
        pts = pts[np.argsort(proj)[:-drop]]
    return pts.mean(axis=0)


def errors(points: np.ndarray, eps: float) -> dict:
    """Fehler (Norm des Schätzers, wahres Zentrum = Ursprung) der vier Schätzer an einer Punktmenge."""
    return {
        "mittel": float(np.linalg.norm(points.mean(axis=0))),
        "koordinatenmedian": float(np.linalg.norm(np.median(points, axis=0))),
        "geometrischer_median": float(np.linalg.norm(geometric_median_nd(points))),
        "filter": float(np.linalg.norm(simple_filter(points, eps))),
    }
