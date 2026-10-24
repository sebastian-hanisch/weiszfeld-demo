"""Kennzahlen der Karte, Verdikte der Kernfrage und der Experimente. Verdikte sind reine Funktionen (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

import wsz_constants as C
import wsz_equivariance as E
import wsz_geomed as G
from wsz_experiments import MAXIT_SERIES, cell_seed, nearest_point_start
from wsz_scenario import Draws, Instance, make_instance


def de(x: float, nd: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma, ohne Prozentzeichen."""
    return de(100.0 * x, nd)


@dataclass(frozen=True)
class Settings:
    layout: str = C._BASE["layout"]
    n: int = C._BASE["n"]
    weights: str = C._BASE["weights"]
    share: int = C._BASE["share"]
    factor: float = C._BASE["factor"]
    s: float = C._BASE["s"]
    angle: int = C._BASE["angle"]
    start: str = C._BASE["start"]
    solver: str = C._BASE["solver"]
    tol_exp: int = C._BASE["tol_exp"]
    maxit: int = C._BASE["maxit"]
    seed: int = C._BASE["seed"]
    customer: int = C._BASE["customer"]


assert [f.name for f in fields(Settings)] == list(C._BASE), "Settings und Presets müssen dieselben Regler in derselben Reihenfolge führen"


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    points: np.ndarray            # Kunden nach Streckung und Drehung (km bzw. Minuten)
    weights: np.ndarray
    pull_customer: int            # Index des Kunden, dessen Zugkräfte gezeichnet werden
    pull_share: np.ndarray        # Zugkraft der übrigen / Gewicht am Ort, je Kunde
    centroid: np.ndarray
    coord_median: np.ndarray
    tukey: np.ndarray | None      # nur bis TUKEY_LIVE_MAX Kunden
    start: np.ndarray             # Startpunkt des Lösers
    result: G.Result
    value: float                  # Gesamtweg am Ergebnis des Lösers
    ref_center: np.ndarray        # Optimum (auf einem Kunden exakt, sonst Vardi–Zhang mit enger Toleranz)
    ref_value: float
    ref_on: int | None            # Kunde, auf dem das Optimum liegt, sonst None
    excess_pct: float             # Mehrweg des Lösers gegen das Optimum in Prozent
    pre_center: np.ndarray | None  # Optimum der Ausgangsdaten, mitgestreckt/-gedreht (nur bei Streckung oder Drehung)
    pre_excess_pct: float | None   # Mehrweg, wenn die Mitte ohne Streckung/Drehung gesucht wurde
    pre_distance: float | None     # Abstand von pre_center zum Optimum, in Streuungen der x-Werte der Ausgangsdaten
    coord_pre_distance: float | None   # dasselbe für den Koordinatenmedian
    tukey_pre_distance: float | None


def optimum(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Optimum des Weber-Problems: liegt es auf einem Kunden (Schwellentest), exakt dort; sonst Vardi–Zhang mit enger Toleranz."""
    k = G.on_point(points, weights)
    if k is not None:
        return points[k].copy(), k
    return G.vardi_zhang(points, weights, tol=1e-12, maxit=C.REFERENCE_MAXIT).center, None


def transform(points: np.ndarray, s: float, angle: float) -> np.ndarray:
    """Erst die y-Achse mit s strecken, dann um `angle` Grad drehen."""
    return E.rotate(E.scale_axes(points, 1.0, s), angle)


def start_point(settings: Settings, points: np.ndarray, weights: np.ndarray):
    if settings.start == "Schwerpunkt":
        return None
    if settings.start == "nächster Datenpunkt":
        return nearest_point_start(points, weights)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return lo + Draws(cell_seed(settings.seed, "start")).uniform(2) * (hi - lo)


def run_solver(settings: Settings, points, weights, start) -> G.Result:
    tol = 10.0 ** settings.tol_exp
    if settings.solver == "klassisch":
        return G.weiszfeld(points, weights, start=start, tol=tol, maxit=settings.maxit, keep=C.PATH_KEEP)
    if settings.solver == "mit Epsilon-Schutz":
        return G.weiszfeld(points, weights, start=start, tol=tol, maxit=settings.maxit, guard=1e-12, keep=C.PATH_KEEP)
    return G.vardi_zhang(points, weights, start=start, tol=tol, maxit=settings.maxit, keep=C.PATH_KEEP)


def build_instance(settings: Settings) -> Instance:
    return make_instance(settings.layout, settings.n, settings.weights, settings.share, settings.factor, settings.seed)


def analyse(settings: Settings) -> Analysis:
    inst = build_instance(settings)
    w = inst.weights
    p = transform(inst.points, settings.s, settings.angle)
    norms, own = G.pull_norms(p, w)
    ref_center, ref_on = optimum(p, w)
    st = start_point(settings, p, w)
    result = run_solver(settings, p, w, st)
    ref_value = G.objective(p, w, ref_center)
    value = G.objective(p, w, result.center)
    transformed = settings.s != 1.0 or settings.angle != 0
    pre = pre_excess = pre_dist = coord_dist = tukey_dist = None
    tukey = E.tukey_median_2d(p) if settings.n <= C.TUKEY_LIVE_MAX else None
    if transformed:
        c0, _ = optimum(inst.points, w)
        pre = transform(c0[None], settings.s, settings.angle)[0]
        pre_excess = E.excess_cost(p, w, pre, ref_center)
        sd = float(np.std(inst.points[:, 0]))
        pre_dist = float(np.linalg.norm(pre - ref_center) / sd)
        coord_dist = float(np.linalg.norm(transform(E.coordinate_median(inst.points, w)[None], settings.s, settings.angle)[0] - E.coordinate_median(p, w)) / sd)
        if tukey is not None:
            tukey_dist = float(np.linalg.norm(transform(E.tukey_median_2d(inst.points)[None], settings.s, settings.angle)[0] - tukey) / sd)
    return Analysis(
        settings, p, w, min(settings.customer, settings.n) - 1, norms / own, G.weighted_centroid(p, w), E.coordinate_median(p, w), tukey,
        G.weighted_centroid(p, w) if st is None else st, result, value, ref_center, ref_value, ref_on,
        0.0 if ref_value <= 0 else 100.0 * (value - ref_value) / ref_value, pre, pre_excess, pre_dist, coord_dist, tukey_dist,
    )


# ---------------------------------------------------------------- Verdikte der Karte

def verdict_location(a: Analysis) -> tuple[str, str]:
    """Kernfrage 'Wo steht der Weber-Punkt?': auf einem Kunden oder zwischen den Kunden."""
    ratio = a.pull_share
    if a.ref_on is not None:
        k = a.ref_on
        return "success", (f"Der Weber-Punkt liegt **auf Kunde {k + 1}** (Gewicht {de(a.weights[k], 2)}): Die Zugkraft aller anderen Kunden auf diesen Ort hat nur die Norm "
                           f"{de(ratio[k] * a.weights[k], 2)} und reicht nicht, ihn zu verschieben. Der Gesamtweg beträgt {de(a.ref_value, 2)}.")
    j = int(np.argmin(ratio))
    return "info", (f"Der Weber-Punkt liegt **zwischen den Kunden**, Gesamtweg {de(a.ref_value, 2)}. Am nächsten an der Schwelle ist Kunde {j + 1}: "
                    f"Zugkraft durch Gewicht = {de(ratio[j], 2)} (ab höchstens 1,00 würde er den Standort festhalten).")


def verdict_solver(solver: str, status: str, iterations: int, excess_pct: float, maxit: int) -> tuple[str, str]:
    """Verdikt zum gewählten Löser: Abbruch, Grenze, falsche Konvergenz oder Erfolg."""
    if status == "singulaer":
        extra = (f"Gesamtweg {de(excess_pct, 2)} % über dem Optimum." if excess_pct > 1e-4 else "Zufällig ist der Kunde, auf dem er steht, hier das Optimum.")
        return "error", f"Das klassische Verfahren **bricht nach {iterations} Schritten ab**: Ein Iterat steht exakt auf einem Kunden, der nächste Schritt wäre 0/0. {extra}"
    if status == "maxit":
        return "warning", (f"Die **Iterationsgrenze von {maxit}** ist erreicht, bevor der Schritt unter die Toleranz fiel. Gesamtweg {de(excess_pct, 4)} % über dem Optimum. "
                           "Nahe der Schwelle konvergiert das Verfahren sehr langsam.")
    if status == "datenpunkt":
        return "success", f"{solver} erkennt nach {iterations} Schritten, dass der Kunde den Iterat festhält: Das Optimum liegt auf diesem Kunden."
    if excess_pct > 1e-4:
        return "error", (f"{solver} **meldet Konvergenz** nach {iterations} Schritten, aber der Gesamtweg liegt {de(excess_pct, 2)} % über dem Optimum: "
                         "Der Schritt wurde klein, weil der Iterat an einem Kunden hängt, nicht weil er am Optimum ist.")
    return "success", f"{solver} konvergiert nach {iterations} Schritten; der Gesamtweg liegt {de(max(excess_pct, 0.0), 6)} % über dem Optimum."


# ---------------------------------------------------------------- Verdikte der Experimente

def verdict_e1(first: dict, last: dict, label: str) -> tuple[str, str]:
    """E1: Anteil der Instanzen mit dem Optimum auf einem Kunden bei kleinem und großem n (Zeilen mit n, mean, std)."""
    return "info", (f"{label}: Bei n = {first['n']} Kunden liegt der Weber-Punkt in **{pct(first['mean'])} %** der Instanzen auf einem Kunden, bei n = {last['n']} nur noch in "
                    f"{pct(last['mean'])} % (Mittel über vier Zufalls-Seeds, Streuung zwischen den Seeds ±{pct(first['std'])} bzw. ±{pct(last['std'])} Prozentpunkte).")


def verdict_e2(rows: list[dict]) -> tuple[str, str]:
    """E2: Anteil des Standorts auf dem Großkunden nach Gewichtsanteil; ab 50 % zwingend (Dreiecksungleichung)."""
    always = [r for r in rows if r["mean"] >= 1.0]
    below = [r for r in rows if r["mean"] < 1.0]
    if not always:
        return "info", "In keiner Zeile liegt der Standort immer auf dem Großkunden."
    first, last_below = min(r["share"] for r in always), max(below, key=lambda r: r["share"]) if below else None
    text = f"Ab einem Gewichtsanteil von **{first} %** liegt der Standort in allen Instanzen auf dem Großkunden."
    if last_below is not None:
        text += f" Bei {last_below['share']} % sind es {pct(last_below['mean'])} %."
    return "success", text + " Bei 50 % und mehr ist das zwingend: Die Zugkraft aller anderen hat höchstens die Norm ihres Gesamtgewichts, und das ist höchstens das Gewicht des Großkunden."


def verdict_e3(rows: list[dict]) -> tuple[str, str]:
    """E3: Iterationen nach Gewicht f (Vielfaches der Schwelle)."""
    at = {r["f"]: r for r in rows}
    mid = at[1.0]
    return "warning", (f"Weit unter der Schwelle (f = 0,5) braucht das klassische Verfahren im Median {int(at[0.5]['median'])} Iterationen, an der Schwelle (f = 1,0) {int(mid['median'])}, "
                       f"weit darüber (f = 3,0) {int(at[3.0]['median'])}. Bei f = 1,0 erreichen {pct(mid['share_maxit'], 0)} % der {mid['instances']} Instanzen die Grenze von {MAXIT_SERIES} Schritten, ohne dass der Schritt unter die Toleranz fällt. "
                       f"Beim Abbruch (Schritt < 1e-9) liegt der Iterat dort im Median noch {de(mid['median_error_at_stop'] * 1e6, 1)} Millionstel km vom Optimum entfernt.")


def verdict_e4(row: dict) -> tuple[str, str]:
    """E4: Start auf dem Kunden, der dem Schwerpunkt am nächsten liegt."""
    return "error", (f"Bei n = {row['n']} bricht das klassische Verfahren vom Start auf einem Kunden in **{pct(row['abort'], 0)} %** der {row['instances']} Instanzen ab; in "
                     f"{pct(row['classic_wrong'], 1)} % ist das Ergebnis nicht optimal. Die Epsilon-Absicherung ändert daran nichts ({pct(row['guard_wrong'], 1)} %), "
                     f"Vardi–Zhang liegt in {pct(row['vz_wrong'], 1)} % daneben.")


def verdict_e5(unit: dict, rot: dict) -> tuple[str, str]:
    """E5: Achsenstreckung und Drehung."""
    return "info", (f"Bei s = {de(unit['s'], 0)} liegt der Weber-Punkt im Mittel {de(unit['geo_mean'], 2)} Streuungen anders, als wenn man ihn erst bestimmt und dann streckt; "
                    f"der Mehrweg beträgt im Mittel {de(unit['excess_mean'], 2)} % des Gesamtwegs (größter Fall {de(unit['excess_max'], 1)} %). Der Koordinatenmedian geht bei der Streckung mit "
                    f"(Abweichung {unit['coord_max']:.0e}), bei einer Drehung um {de(rot['angle'], 0)}° nicht (im Mittel {de(rot['coord_mean'], 2)} Streuungen, größte {de(rot['coord_max'], 2)}); "
                    f"der Weber-Punkt dreht mit (Abweichung {rot['geo_max']:.0e}).")


def verdict_e6(rows_a: list[dict], rows_b: list[dict]) -> tuple[str, str]:
    """E6: Fehler des geometrischen Medians in hoher Dimension, Fall A (weit entfernter Haufen) und Fall B (fester Abstand)."""
    a0, a1, b0, b1 = rows_a[0], rows_a[-1], rows_b[0], rows_b[-1]
    return "info", (f"Fall A (Haufen weit weg): Der Fehler des geometrischen Medians wächst von {de(a0['geometrischer_median'], 2)} (d = {a0['d']}) auf {de(a1['geometrischer_median'], 2)} (d = {a1['d']}); "
                    f"geteilt durch ε·√d bleibt er zwischen {de(min(r['quotient_geo'] for r in rows_a), 2)} und {de(max(r['quotient_geo'] for r in rows_a), 2)}. "
                    f"Fall B (fester Abstand 3): von {de(b0['geometrischer_median'], 2)} auf {de(b1['geometrischer_median'], 2)}, der Quotient fällt von {de(b0['quotient_geo'], 2)} auf {de(b1['quotient_geo'], 2)}.")


def stop_caption(f: float, seed: int, iterations: int, status: str, error: float) -> str:
    """Bildunterschrift der eigenen Fehlerkurve (E3): wie der Lauf endete und wie weit der Iterat dann noch vom Optimum entfernt war."""
    end = (f"erreicht die Grenze von {iterations} Iterationen, ohne dass der Schritt unter 1e-9 fällt" if status == "maxit"
           else f"hält bei Schritt < 1e-9 nach {iterations} Iterationen an")
    return f"f = {de(f, 2)}, Zufalls-Seed {seed}: Das Verfahren {end}; der Iterat liegt dann noch {f'{error:.1e}'.replace('.', ',')} km vom Optimum entfernt."
