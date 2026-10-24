"""Szenario: Kunden mit Gewicht in der Ebene (1 Einheit = 1 km), erzeugt aus dem eigenen SplitMix64-Strom (wsz_rng.py).

`Draws` zieht Zahlen blockweise (numpy, uint64 rechnet modulo 2^64) und liefert exakt den Strom der wörtlich kopierten Klasse `SplitMix64`;
tests/test_scenario.py vergleicht beide. Damit sind alle Instanzen auf jeder Plattform gleich, und Messreihen über viele Instanzen bleiben schnell.

Layouts: Normal (N(0, I)), Quadrat (gleichverteilt auf [-1, 1]²), Kern + Außenposten (Kunde 0 steht bei (6, 6), die übrigen sind normalverteilt),
Kern + Seitengruppe (ein Viertel der Kunden, die letzten, ist um (4; 1) verschoben).
Gewichte: gleich, lognormal(0; σ) mit σ = 0,5 oder 1, Großkunde (Kunde 0, gewählter Gewichtsanteil, übrige Gewichte 1) und Großkunde an der Schwelle
(Gewicht des Kunden 0 = f mal die Norm der Zugkräfte der übrigen Kunden auf ihn)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wsz_rng import MASK

GAMMA = 0x9E3779B97F4A7C15
OUTPOST = (6.0, 6.0)
SIDE_CENTER = (4.0, 1.0)
SIGMAS ={"lognormal 0,5": 0.5, "lognormal 1": 1.0}


class Draws:
    """SplitMix64-Strom, blockweise gezogen: gleicher Seed, gleiche Zahlen wie `wsz_rng.SplitMix64`, aber ohne Python-Schleife."""

    def __init__(self, seed: int):
        self.state = int(seed) & MASK

    def raw(self, count: int) -> np.ndarray:
        """Die nächsten `count` Ausgaben des Stroms als uint64-Feld."""
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.arange(1, count + 1, dtype=np.uint64) * np.uint64(GAMMA) + np.uint64(self.state)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + count * GAMMA) & MASK
        return z

    def uniform(self, shape) -> np.ndarray:
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        size = int(np.prod(shape))
        return ((self.raw(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))).reshape(shape)

    def normal(self, shape) -> np.ndarray:
        """Standardnormal nach Box-Muller: zwei gleichverteilte Zahlen ergeben ein Paar (Kosinus-Wert zuerst)."""
        size = int(np.prod(shape))
        pairs = (size + 1) // 2
        u = self.uniform((2, pairs))
        radius = np.sqrt(-2.0 * np.log(1.0 - u[0]))        # 1 - u liegt in (0, 1]: kein log(0)
        angle = 2.0 * math.pi * u[1]
        z = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return z[:size].reshape(shape)

    def lognormal(self, sigma: float, shape) -> np.ndarray:
        return np.exp(sigma * self.normal(shape))


def side_group_size(n: int) -> int:
    """Größe der Seitengruppe im Layout 'Kern + Seitengruppe': ein Viertel der Kunden (bei n = 40 sind es 10), mindestens einer."""
    return max(1, int(round(n / 4)))


def layout_batch(draws: Draws, layout: str, runs: int, n: int) -> np.ndarray:
    """`runs` Punktmengen mit je n Kunden, Form (runs, n, 2)."""
    if layout == "Normal":
        return draws.normal((runs, n, 2))
    if layout == "Quadrat":
        return draws.uniform((runs, n, 2)) * 2.0 - 1.0
    if layout == "Kern + Außenposten":
        points = draws.normal((runs, n, 2))
        points[:, 0] = OUTPOST
        return points
    if layout == "Kern + Seitengruppe":
        points = draws.normal((runs, n, 2))
        points[:, n - side_group_size(n):] += np.array(SIDE_CENTER)
        return points
    raise ValueError(f"unbekanntes Layout: {layout}")


def big_customer_weight(n: int, share: float) -> float:
    """Gewicht des Großkunden, wenn die übrigen n - 1 Kunden das Gewicht 1 haben und er den Anteil `share` (0 bis 1) am Gesamtgewicht hat."""
    if n < 2:
        raise ValueError("ein Großkunde braucht mindestens einen weiteren Kunden")
    if not 0.0 < share < 1.0:
        raise ValueError("der Gewichtsanteil liegt strikt zwischen 0 und 1")
    return share / (1.0 - share) * (n - 1)


def weights_batch(draws: Draws, kind: str, runs: int, n: int, share: float = 0.4) -> np.ndarray:
    """Gewichte (runs, n) für die Gewichtsarten ohne Schwelle. `share` gilt nur für 'Großkunde'."""
    if kind == "gleich":
        return np.ones((runs, n))
    if kind in SIGMAS:
        return draws.lognormal(SIGMAS[kind], (runs, n))
    if kind == "Großkunde":
        w = np.ones((runs, n))
        w[:, 0] = big_customer_weight(n, share)
        return w
    raise ValueError(f"unbekannte Gewichtsart: {kind}")


def threshold_weight(points: np.ndarray, others_weight: np.ndarray) -> float:
    """Schwelle des Kunden 0: Norm der Summe der gewichteten Einheitsvektoren von Kunde 0 zu den übrigen. Ab diesem Gewicht steht der Standort auf Kunde 0."""
    diff = points[1:] - points[0]
    dist = np.linalg.norm(diff, axis=1)
    keep = dist > 0
    return float(np.linalg.norm((others_weight[keep, None] * diff[keep] / dist[keep, None]).sum(axis=0)))


@dataclass(frozen=True)
class Instance:
    """Eine Instanz: Kundenpositionen (n, 2) in km und Gewichte (n,). Unveränderlich gedacht (Felder werden nicht beschrieben)."""
    points: np.ndarray
    weights: np.ndarray

    @property
    def n(self) -> int:
        return len(self.weights)


def make_instance(layout: str, n: int, kind: str, share: float, factor: float, seed: int) -> Instance:
    """Deterministisch aus den Reglern: gleiche Eingabe, gleiche Instanz. `share` in Prozent, `factor` in Vielfachen der Schwelle."""
    draws = Draws(seed)
    points = layout_batch(draws, layout, 1, n)[0]
    if kind == "Großkunde an der Schwelle":
        w = np.ones(n)
        w[0] = factor * threshold_weight(points, w[1:])
    else:
        w = weights_batch(draws, kind, 1, n, share / 100.0)[0]
    return Instance(points, w)
