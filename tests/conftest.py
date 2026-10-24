"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen (Gewichte 1, wenn nicht anders genannt):
* stumpfes Dreieck (0, 0), (2, 0), (1; 0,2): der Winkel an (1; 0,2) ist 180° − 2·arctan 0,2 = 157,4° ≥ 120°, also liegt das Optimum dort,
  Zielwert 2·√1,04 = 2,0396. Zugkraft der beiden anderen auf diese Ecke: (0; −0,4)/√1,04, Norm 0,3922 ≤ 1.
* A = (0, 0), B = (4, 0), C = (0, 3), Gewichte (w, 1, 1): Zugkraft auf A ist (1, 0) + (0, 1), Norm √2 = 1,4142; das Optimum liegt in A genau für w ≥ √2.
* Quadrat (±1, ±1) mit Mitte (0, 0): die Zugkräfte der Ecken auf die Mitte heben sich auf, das Optimum liegt in der Mitte für jedes Gewicht der Mitte.
* drei Punkte auf einer Geraden (0, 0), (1, 0), (5, 0): das Optimum ist der mittlere Punkt, Zielwert 1 + 4 = 5.
* gleichseitiges Dreieck (0, 0), (2, 0), (1; √3): Torricelli-Punkt = Schwerpunkt (1; √3/3), Zielwert 3·(2/√3) = 2·√3.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def obtuse():
    """Stumpfes Dreieck mit Winkel 157,4° an der dritten Ecke."""
    return np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.2]]), np.ones(3)


@pytest.fixture
def abc():
    """A = (0, 0), B = (4, 0), C = (0, 3); das Gewicht von A wird im Test gesetzt."""
    return np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])


@pytest.fixture
def square_center():
    return np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0]])


@pytest.fixture
def collinear():
    return np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])


@pytest.fixture
def equilateral():
    return np.array([[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3.0)]])
