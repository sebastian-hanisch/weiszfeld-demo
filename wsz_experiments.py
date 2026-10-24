"""Messreihen der Experimente E1 bis E6: reine Funktionen über feste Zufalls-Seeds, die sowohl die vorgerechneten Tabellen (tools/make_results.py → data/wsz_results.json)
als auch die Live-Knöpfe der App erzeugen. Jede Zelle nutzt vier feste Seeds (C.SEEDS); angegeben wird der Mittelwert mit der Standardabweichung über die Seeds.

Zufall: `Draws(cell_seed(...))` aus wsz_scenario.py; ein Seed gehört zu einer Zelle (Seed, Tags), nie zu einer Instanz allein."""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path

import numpy as np

import wsz_constants as C
import wsz_equivariance as E
import wsz_geomed as G
import wsz_highdim as H
from wsz_scenario import Draws, layout_batch, make_instance, threshold_weight, weights_batch

ROOT = Path(__file__).resolve().parent
SOURCES = ("wsz_rng.py", "wsz_scenario.py", "wsz_geomed.py", "wsz_equivariance.py", "wsz_highdim.py", "wsz_experiments.py", "tools/make_results.py")
STOP_TOL = 1e-9          # Abbruch der Messreihen: Schritt kürzer als diese Toleranz
MAXIT_SERIES = 100000    # Iterationsgrenze der Konvergenz-Messreihe (E3)


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Zelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte)."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


# ---------------------------------------------------------------- E1/E2: wie oft liegt das Optimum auf einem Kunden?

def _chunk(n: int) -> int:
    return max(1, 400_000 // (n * n))


def share_on_point(layout: str, kind: str, n: int, runs: int, seeds=C.SEEDS, share: float = 0.4, only_first: bool = False) -> dict:
    """Anteil der Zufallsinstanzen, in denen das Optimum auf einem Kunden liegt (Schwellentest ‖Zugkraft‖ ≤ Gewicht), je Seed `runs` Instanzen.
    `only_first`: nur der Großkunde (Kunde 0) zählt."""
    per_seed = []
    for seed in seeds:
        draws = Draws(cell_seed(seed, layout, kind, n, share))
        hit, done = 0, 0
        while done < runs:
            r = min(_chunk(n), runs - done)
            pts = layout_batch(draws, layout, r, n)
            w = weights_batch(draws, kind, r, n, share)
            ok = G.pull_norms_batch(pts, w) <= w * (1.0 + 1e-12)
            hit += int((ok[:, 0] if only_first else ok.any(axis=1)).sum())
            done += r
        per_seed.append(hit / runs)
    mean, std = mean_std(per_seed)
    return {"mean": mean, "std": std, "per_seed": per_seed, "runs_per_seed": runs}


def runs_for(n: int) -> int:
    return 4000 if n <= 30 else (1000 if n <= 50 else 400)


def run_m1() -> dict:
    raster = []
    for layout in C.LAYOUTS:
        for kind in ("gleich", "lognormal 0,5", "lognormal 1"):
            for n in C.N_RASTER:
                raster.append({"layout": layout, "weights": kind, "n": n, **share_on_point(layout, kind, n, runs_for(n))})
    big = []
    for pct in C.SHARES:
        big.append({"share": pct, **share_on_point("Normal", "Großkunde", 20, 4000, share=pct / 100.0, only_first=True)})
    return {"raster": raster, "big": big, "triangles": triangle_study()}


def triangle_study(runs: int = 100000, check: int = 2000) -> dict:
    """Anteil der Dreiecke mit einem Winkel ≥ 120° (Optimum auf der Ecke), je Layout; dazu Torricelli-Konstruktion gegen Vardi–Zhang an `check` Dreiecken."""
    out = {}
    for layout in ("Normal", "Quadrat"):
        per_seed = []
        for seed in C.SEEDS:
            tri = layout_batch(Draws(cell_seed(seed, layout, "dreieck")), layout, runs, 3)
            obtuse = np.zeros(runs, dtype=bool)
            for i in range(3):
                u, v = tri[:, (i + 1) % 3] - tri[:, i], tri[:, (i + 2) % 3] - tri[:, i]
                obtuse |= (u * v).sum(axis=1) <= -0.5 * np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            per_seed.append(float(obtuse.mean()))
        out[layout] = {"mean": mean_std(per_seed)[0], "std": mean_std(per_seed)[1], "runs_per_seed": runs}
    worst, worse = 0.0, 0
    for seed in C.SEEDS:
        for a, b, c in layout_batch(Draws(cell_seed(seed, "torricelli")), "Normal", check // len(C.SEEDS), 3):
            pts = np.array([a, b, c])
            f = G.fermat_point(a, b, c)
            v = G.vardi_zhang(pts, np.ones(3), tol=1e-13).center
            worst = max(worst, float(np.linalg.norm(f - v)))
            worse += G.objective(pts, np.ones(3), f) > G.objective(pts, np.ones(3), v) + 1e-9
    out["torricelli"] = {"instances": check, "max_distance": worst, "worse_objective": int(worse)}
    return out


# ---------------------------------------------------------------- E3: der Algorithmus an der Schwelle

def threshold_instance(factor: float, seed: int, n: int = 6):
    """Instanz für E3: n Normalpunkte, Kunde 0 mit dem Gewicht f mal Schwelle, übrige Gewichte 1."""
    inst = make_instance("Normal", n, "Großkunde an der Schwelle", 40, factor, seed)
    return inst.points, inst.weights


def reference_center(points, weights, factor: float) -> np.ndarray:
    """Referenzlösung der Schwellen-Instanz: ab f ≥ 1 liegt das Optimum auf Kunde 0, darunter wird Vardi–Zhang mit enger Toleranz gerechnet."""
    if factor >= 1.0:
        return points[0].copy()
    return G.vardi_zhang(points, weights, tol=1e-13, maxit=300000).center


def iterations_by_factor(factors=C.FACTORS, per_seed: int = 24, seeds=C.SEEDS) -> list[dict]:
    rows = []
    for f in factors:
        its, err, maxit_hit = [], [], 0
        for seed in seeds:
            for i in range(per_seed):
                pts, w = threshold_instance(f, cell_seed(seed, "schwelle", i))
                r = G.weiszfeld(pts, w, tol=STOP_TOL, maxit=MAXIT_SERIES, guard=1e-15)
                its.append(r.iterations)
                maxit_hit += r.status == "maxit"
                err.append(float(np.linalg.norm(r.center - reference_center(pts, w, f))))
        a = np.array(its)
        rows.append({"f": f, "instances": len(a), "median": float(np.median(a)), "p95": float(np.quantile(a, 0.95)), "max": int(a.max()),
                     "share_maxit": maxit_hit / len(a), "median_error_at_stop": float(np.median(err))})
    return rows


def rate_study(factors=(0.5, 1.0, 1.5), per_seed: int = 6, steps: int = 20000, ks=(10, 100, 1000, 10000)) -> list[dict]:
    """Fehler gegen Iteration: Quotient e(k+1)/e(k) (Median über die Instanzen, ohne Instanzen, deren Fehler bei k schon Rundungsrauschen ist) und die Steigung
    von log e über log k zwischen k = 1000 und 10000. Linear konvergent heißt Quotient dauerhaft unter 1; sublinear heißt Quotient strebt gegen 1."""
    rows = []
    for f in factors:
        errors = []
        for seed in C.SEEDS:
            for i in range(per_seed):
                pts, w = threshold_instance(f, cell_seed(seed, "rate", i))
                ref = reference_center(pts, w, f)
                r = G.weiszfeld(pts, w, tol=0.0, maxit=steps + 1, guard=1e-18, keep=steps + 2)
                errors.append(np.linalg.norm(r.path - ref, axis=1))
        e = np.array(errors)
        quot, valid = {}, {}
        for k in ks:
            ok = (e[:, k] > 1e-9) & (e[:, k + 1] > 0)
            valid[str(k)] = int(ok.sum())
            quot[str(k)] = float(np.median(e[ok, k + 1] / e[ok, k])) if ok.any() else None
        slope_ok = (e[:, 10000] > 1e-9)
        slope = float(np.median(np.log(e[slope_ok, 10000] / e[slope_ok, 1000]) / np.log(10.0))) if slope_ok.any() else None
        rows.append({"f": f, "instances": len(e), "quotient": quot, "valid": valid, "slope_1000_10000": slope,
                     "median_error": {str(k): float(np.median(e[:, k])) for k in ks}})
    return rows


def random_iterations(ns=C.N_RASTER[:3] + (30, 100), per_seed: int = 500) -> list[dict]:
    """Iterationen bis Schritt < 1e-9 an gleich gewichteten Normalpunkten (Start Schwerpunkt): Median, P95, größter Wert."""
    rows = []
    for n in ns:
        its = []
        for seed in C.SEEDS:
            draws = Draws(cell_seed(seed, "zufall", n))
            for p in layout_batch(draws, "Normal", per_seed, n):
                its.append(G.weiszfeld(p, np.ones(n), tol=STOP_TOL, maxit=300000, guard=1e-15).iterations)
        a = np.array(its)
        rows.append({"n": n, "instances": len(a), "median": float(np.median(a)), "p95": float(np.quantile(a, 0.95)), "max": int(a.max())})
    return rows


def run_m3() -> dict:
    return {"factors": iterations_by_factor(), "rate": rate_study(), "random": random_iterations()}


# ---------------------------------------------------------------- E4: der Start entscheidet

def nearest_point_start(points, weights) -> np.ndarray:
    """Der Kunde, der dem gewichteten Schwerpunkt am nächsten liegt (verbreitete Startwahl)."""
    return points[int(np.argmin(np.linalg.norm(points - G.weighted_centroid(points, weights), axis=1)))].copy()


def start_study(n: int, per_seed: int = 500, seeds=C.SEEDS, kind: str = "gleich") -> dict:
    """Start auf dem Kunden, der dem Schwerpunkt am nächsten liegt: Anteil der Instanzen, in denen das klassische Verfahren abbricht, und Anteil, in dem
    das Ergebnis nicht optimal ist (Gesamtweg um mehr als 1e-7 über dem Optimum): klassisch, mit Epsilon-Schutz 1e-12, Vardi–Zhang; zum Vergleich der Start im Schwerpunkt."""
    count = {"abort": 0, "classic_wrong": 0, "guard_wrong": 0, "vz_wrong": 0, "centroid_classic_wrong": 0}
    total = 0
    for seed in seeds:
        draws = Draws(cell_seed(seed, "start", n, kind))
        pts_all = layout_batch(draws, "Normal", per_seed, n)
        w_all = weights_batch(draws, kind, per_seed, n)
        for pts, w in zip(pts_all, w_all):
            ref = G.vardi_zhang(pts, w, tol=1e-12, maxit=300000)
            f_ref = G.objective(pts, w, ref.center)
            start = nearest_point_start(pts, w)
            classic = G.weiszfeld(pts, w, start=start, tol=1e-10, maxit=300000)
            guard = G.weiszfeld(pts, w, start=start, tol=1e-10, maxit=300000, guard=1e-12)
            vz = G.vardi_zhang(pts, w, start=start, tol=1e-10, maxit=300000)
            cent = G.weiszfeld(pts, w, tol=1e-10, maxit=300000)
            total += 1
            count["abort"] += classic.status == "singulaer"
            count["classic_wrong"] += G.objective(pts, w, classic.center) > f_ref + 1e-7
            count["guard_wrong"] += G.objective(pts, w, guard.center) > f_ref + 1e-7
            count["vz_wrong"] += G.objective(pts, w, vz.center) > f_ref + 1e-7
            count["centroid_classic_wrong"] += G.objective(pts, w, cent.center) > f_ref + 1e-7
    return {"n": n, "instances": total, **{k: v / total for k, v in count.items()}}


def run_m6() -> dict:
    return {"rows": [start_study(n) for n in (3, 5, 10, 30)]}


# ---------------------------------------------------------------- E5: Einheiten und Drehung

def skewed_cloud(draws: Draws, runs: int) -> np.ndarray:
    """40 Punkte je Instanz im Layout 'Kern + Seitengruppe': 30 aus N(0, I) und 10 aus N((4; 1), I), Form (runs, 40, 2)."""
    return layout_batch(draws, "Kern + Seitengruppe", runs, 40)


def _geomed(p) -> np.ndarray:
    return G.vardi_zhang(p, np.ones(len(p)), tol=1e-12, maxit=100000).center


def unit_study(s: float, per_seed: int = 150, seeds=C.SEEDS, tukey_runs: int = 0) -> dict:
    """y-Achse mal s. Abstand zwischen 'Mitte bestimmen, dann strecken' und 'strecken, dann Mitte bestimmen', geteilt durch die Standardabweichung der x-Werte
    (geometrischer Median, Koordinatenmedian, Tiefster Datenpunkt) und Mehrweg in Prozent des Gesamtwegs, wenn die Wege mit s auf y gerechnet werden, die Mitte aber
    ohne Streckung gesucht wurde."""
    geo, coord, mehr = [], [], []
    for seed in seeds:
        for p in skewed_cloud(Draws(cell_seed(seed, "einheit")), per_seed):
            w = np.ones(len(p))
            ps = E.scale_axes(p, 1.0, s)
            c_then_scale = E.scale_axes(_geomed(p)[None], 1.0, s)[0]
            c_of_scaled = _geomed(ps)
            geo.append(np.linalg.norm(c_then_scale - c_of_scaled) / p[:, 0].std())
            coord.append(np.linalg.norm(E.scale_axes(E.coordinate_median(p, w)[None], 1.0, s)[0] - E.coordinate_median(ps, w)))
            mehr.append(E.excess_cost(ps, w, c_then_scale, c_of_scaled))
    row = {"s": s, "instances": len(geo), "geo_mean": float(np.mean(geo)), "geo_std": float(np.std(geo)), "coord_max": float(np.max(coord)),
           "excess_mean": float(np.mean(mehr)), "excess_max": float(np.max(mehr))}
    if tukey_runs:
        d = []
        for p in skewed_cloud(Draws(cell_seed(1, "tukey-einheit")), tukey_runs):
            a = E.scale_axes(E.tukey_median_2d(p)[None], 1.0, s)[0]
            d.append(np.linalg.norm(a - E.tukey_median_2d(E.scale_axes(p, 1.0, s))))
        row["tukey_max"] = float(np.max(d))
        row["tukey_runs"] = tukey_runs
    return row


def rotation_study(angle: float = 45.0, per_seed: int = 150, seeds=C.SEEDS) -> dict:
    """Drehung um `angle` Grad: Abstand 'Mitte, dann drehen' zu 'drehen, dann Mitte' geteilt durch die Standardabweichung der x-Werte."""
    geo, coord, mean = [], [], []
    for seed in seeds:
        for p in skewed_cloud(Draws(cell_seed(seed, "drehung")), per_seed):
            w = np.ones(len(p))
            pr = E.rotate(p, angle)
            sc = p[:, 0].std()
            geo.append(np.linalg.norm(E.rotate(_geomed(p)[None], angle)[0] - _geomed(pr)) / sc)
            coord.append(np.linalg.norm(E.rotate(E.coordinate_median(p, w)[None], angle)[0] - E.coordinate_median(pr, w)) / sc)
            mean.append(np.linalg.norm(E.rotate(p.mean(axis=0)[None], angle)[0] - pr.mean(axis=0)) / sc)
    return {"angle": angle, "instances": len(geo), "geo_max": float(np.max(geo)), "coord_mean": float(np.mean(coord)), "coord_max": float(np.max(coord)),
            "mean_max": float(np.max(mean))}


def run_m4() -> dict:
    return {"units": [unit_study(s, tukey_runs=60 if s == 5.0 else 0) for s in C.S_VALUES], "rotation": rotation_study()}


# ---------------------------------------------------------------- E6: hohe Dimension

def highdim_cell(case: str, d: int, n_mode: str, runs_per_seed: int, eps: float = C.EPS_CONTAMINATION, seeds=C.SEEDS) -> dict:
    """Mittlerer Fehler (Norm des Schätzers) der vier Schätzer in einer Zelle; n_mode "1000" (n = 1000) oder "20d" (n = max(200, 20·d))."""
    n = 1000 if n_mode == "1000" else max(200, 20 * d)
    acc = {k: [] for k in ("mittel", "koordinatenmedian", "geometrischer_median", "filter")}
    for seed in seeds:
        draws = Draws(cell_seed(seed, "hochdim", case, d, n_mode))
        for _ in range(runs_per_seed):
            for key, value in H.errors(H.contaminate(case, d, eps, n, draws), eps).items():
                acc[key].append(value)
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    return {"case": case, "d": d, "n": n, "n_mode": n_mode, "runs": len(acc["mittel"]), "eps": eps, **mean,
            "se_geo": float(np.std(acc["geometrischer_median"]) / np.sqrt(len(acc["mittel"]))),
            "quotient_geo": mean["geometrischer_median"] / (eps * np.sqrt(d))}


def run_m5(runs_per_seed: int = 10) -> dict:
    return {case: {mode: [highdim_cell(case, d, mode, runs_per_seed) for d in C.DIMS] for mode in ("1000", "20d")} for case in H.CASES}


PARTS = {"m1": run_m1, "m3": run_m3, "m4": run_m4, "m5": run_m5, "m6": run_m6}
