"""Geometrischer Median (Weber-Punkt): min_c Σ w_i ‖x_i − c‖ in der Ebene, mit den Verfahren, die die Demo vergleicht.

Einheiten (je eine Regel, DEMO-PLAYBOOK §1b): `objective`, `weighted_centroid`, `pull_norms` (Zugkräfte auf jeden Kunden), `on_point` (Schwellentest),
`weiszfeld` (klassischer Schritt, optional mit Epsilon-Schutz), `vardi_zhang` (Schritt mit Behandlung des Kunden, auf dem der Iterat steht),
`fermat_point` (Torricelli-Konstruktion für drei Punkte). Kein Zugriff auf Streamlit, kein verstecktes Zustandsobjekt.

Status der Verfahren: "ok" (Schritt kürzer als die Toleranz), "singulaer" (klassisch: ein Iterat steht exakt auf einem Kunden, der Schritt ist nicht
definiert, das Verfahren bricht ab), "datenpunkt" (Vardi–Zhang: der Kunde, auf dem der Iterat steht, hält ihn fest, also Optimum), "maxit" (Grenze erreicht)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

COINCIDENT = 1e-13   # Abstand, unter dem Vardi–Zhang einen Iterat als „auf dem Kunden“ behandelt


class Result(NamedTuple):
    center: np.ndarray
    iterations: int
    status: str
    path: np.ndarray | None     # die ersten `keep` Iterate (inklusive Start), sonst None


def objective(points: np.ndarray, weights: np.ndarray, c: np.ndarray) -> float:
    """Gesamtweg Σ w_i ‖x_i − c‖."""
    return float((weights * np.linalg.norm(points - c, axis=1)).sum())


def weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gewichteter Schwerpunkt (Minimum der Summe der quadrierten Abstände)."""
    return (points * weights[:, None]).sum(axis=0) / weights.sum()


def pull_vector(points: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Summe der gewichteten Einheitsvektoren von Kunde k zu allen Kunden an anderen Orten: die Zugkraft der übrigen auf den Standort in x_k."""
    diff = points - points[k]
    dist = np.linalg.norm(diff, axis=1)
    far = dist > 0
    return (weights[far, None] * diff[far] / dist[far, None]).sum(axis=0)


def pull_norms(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Norm der Zugkraft der übrigen auf Kunde k, Gesamtgewicht am Ort von Kunde k) für jedes k. Kunden am selben Ort halten sich gegenseitig fest und
    zählen zum Gewicht am Ort, nicht zur Zugkraft. Zeilenweise in Stücken, damit n bis einige tausend den Speicher nicht sprengt."""
    n = len(weights)
    norms = np.empty(n)
    own = np.empty(n)
    step = max(1, 2_000_000 // max(n, 1))
    for a in range(0, n, step):
        b = min(n, a + step)
        diff = points[None, :, :] - points[a:b, None, :]            # diff[r, j] = x_j - x_k
        dist = np.linalg.norm(diff, axis=2)
        same = dist == 0
        unit = np.where(same[..., None], 0.0, diff / np.where(same, 1.0, dist)[..., None])
        norms[a:b] = np.linalg.norm((weights[None, :, None] * unit).sum(axis=1), axis=1)
        own[a:b] = (weights[None, :] * same).sum(axis=1)
    return norms, own


def on_point(points: np.ndarray, weights: np.ndarray, rel_tol: float = 1e-12) -> int | None:
    """Schwellentest: Index des kleinsten Kunden, auf dem das Optimum liegt (Norm der Zugkraft der übrigen ≤ Gewicht am Ort), sonst None.

    Das ist die Optimalitätsbedingung 0 ∈ Unterdifferential der konvexen Zielfunktion im Punkt x_k."""
    norms, own = pull_norms(points, weights)
    hit = np.flatnonzero(norms <= own * (1.0 + rel_tol))
    return int(hit[0]) if len(hit) else None


def _start_point(points, weights, start) -> np.ndarray:
    return weighted_centroid(points, weights) if start is None else np.array(start, dtype=float)


def weiszfeld(points, weights, start=None, tol: float = 1e-9, maxit: int = 100000, guard: float = 0.0, keep: int = 0) -> Result:
    """Klassisches Weiszfeld-Verfahren: c ← Σ (w_i/d_i) x_i / Σ (w_i/d_i) mit d_i = ‖x_i − c‖.

    guard = 0: steht ein Iterat exakt auf einem Kunden (d_i = 0), ist der Schritt nicht definiert: Abbruch mit Status "singulaer", Ergebnis ist dieser Iterat.
    guard > 0 (die verbreitete Epsilon-Absicherung): d_i wird nach unten auf `guard` begrenzt. Start None = gewichteter Schwerpunkt.
    Abbruch, wenn der Schritt kürzer als `tol` ist. `keep`: so viele Iterate (inklusive Start) werden im Pfad aufbewahrt."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    c = _start_point(points, weights, start)
    path = [c.copy()] if keep > 0 else None
    if np.all(points == points[0]):
        return Result(points[0].copy(), 0, "ok", np.array(path) if path is not None else None)
    status, it = "maxit", 0
    for it in range(1, maxit + 1):
        d = np.linalg.norm(points - c, axis=1)
        if guard == 0.0 and (d == 0).any():
            status = "singulaer"
            break
        if guard > 0.0:
            d = np.maximum(d, guard)
        wi = weights / d
        c_new = (points * wi[:, None]).sum(axis=0) / wi.sum()
        if path is not None and len(path) < keep:
            path.append(c_new.copy())
        step = float(np.linalg.norm(c_new - c))
        c = c_new
        if step < tol:
            status = "ok"
            break
    return Result(c, it, status, np.array(path) if path is not None else None)


def vardi_zhang(points, weights, start=None, tol: float = 1e-9, maxit: int = 100000, keep: int = 0) -> Result:
    """Weiszfeld-Schritt mit Behandlung des Kunden, auf dem der Iterat steht (Fassung des Plans; Originalarbeit nicht eingesehen, gegen scipy geprüft).

    Steht der Iterat auf Kunden mit Gesamtgewicht η und ist die Norm r der Zugkraft der übrigen höchstens η, ist er das Optimum (Status "datenpunkt").
    Sonst: c ← (1 − γ) T(c) + γ c mit T dem Weiszfeld-Schritt ohne die Kunden am Ort und γ = min(1, η / r)."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    c = _start_point(points, weights, start)
    path = [c.copy()] if keep > 0 else None
    if np.all(points == points[0]):
        return Result(points[0].copy(), 0, "ok", np.array(path) if path is not None else None)
    status, it = "maxit", 0
    for it in range(1, maxit + 1):
        d = np.linalg.norm(points - c, axis=1)
        on = d < COINCIDENT
        if on.any():
            eta = float(weights[on].sum())
            wi = weights[~on] / d[~on]
            t = (points[~on] * wi[:, None]).sum(axis=0) / wi.sum()
            r = float(np.linalg.norm(((points[~on] - c) * wi[:, None]).sum(axis=0)))
            if r <= eta:
                c = points[on][0].copy()
                status = "datenpunkt"
                break
            g = min(1.0, eta / r)
            c_new = (1.0 - g) * t + g * c
        else:
            wi = weights / d
            c_new = (points * wi[:, None]).sum(axis=0) / wi.sum()
        if path is not None and len(path) < keep:
            path.append(c_new.copy())
        step = float(np.linalg.norm(c_new - c))
        c = c_new
        if step < tol:
            status = "ok"
            break
    return Result(c, it, status, np.array(path) if path is not None else None)


def fermat_point(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Torricelli-Punkt dreier Punkte: ist ein Winkel ≥ 120°, die Ecke mit diesem Winkel; sonst der Schnitt der Geraden von zwei Ecken zur Spitze des
    jeweils gegenüber außen angesetzten gleichseitigen Dreiecks."""
    pts = [np.asarray(p, dtype=float) for p in (a, b, c)]
    for i in range(3):
        u = pts[(i + 1) % 3] - pts[i]
        v = pts[(i + 2) % 3] - pts[i]
        if u @ v <= -0.5 * np.linalg.norm(u) * np.linalg.norm(v):
            return pts[i].copy()

    def apex(p1, p2, p3):
        mid = (p1 + p2) / 2
        e = p2 - p1
        normal = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        h = np.sqrt(3.0) / 2.0 * np.linalg.norm(e)
        up, down = mid + normal * h, mid - normal * h
        return up if np.linalg.norm(up - p3) > np.linalg.norm(down - p3) else down

    a2 = apex(pts[1], pts[2], pts[0])
    b2 = apex(pts[2], pts[0], pts[1])
    m = np.array([a2 - pts[0], -(b2 - pts[1])]).T
    t = np.linalg.solve(m, pts[1] - pts[0])
    return pts[0] + t[0] * (a2 - pts[0])


def pull_norms_batch(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Norm der Zugkraft der übrigen auf jeden Kunden für einen Stapel von Instanzen: points (R, n, 2), weights (R, n) → (R, n).
    Setzt voraus, dass keine zwei Kunden am selben Ort stehen (bei Zufallsinstanzen mit Wahrscheinlichkeit 1); für Einzelinstanzen mit möglichen Dubletten `pull_norms`."""
    diff = points[:, None, :, :] - points[:, :, None, :]                    # diff[r, k, j] = x_j - x_k
    dist = np.linalg.norm(diff, axis=3)
    idx = np.arange(points.shape[1])
    dist[:, idx, idx] = np.inf
    unit = diff / dist[..., None]
    return np.linalg.norm((weights[:, None, :, None] * unit).sum(axis=2), axis=2)
