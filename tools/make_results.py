"""Erzeugt data/wsz_results.json (die vorgerechneten Messreihen der Experimente E1 bis E6) aus wsz_experiments.py.

    python tools/make_results.py --part m1          # eine Messreihe nach data/_parts/m1.json (läuft allein, mehrere Teile können parallel laufen)
    python tools/make_results.py --merge            # alle Teile zu data/wsz_results.json zusammenfügen, mit Prüfsumme der Quelldateien

Teile: m1 (Anteil auf Kunden, Großkunde, Dreiecke), m3 (Iterationen nach f, Rate, Zufallsinstanzen), m4 (Einheiten, Drehung), m5 (Hohe Dimension), m6 (Startwahl).
Die Prüfsumme (`erzeuger_hash`) deckt wsz_rng, wsz_scenario, wsz_geomed, wsz_equivariance, wsz_highdim, wsz_experiments und dieses Skript ab;
tests/test_results.py schlägt fehl, wenn sich eine dieser Dateien nach dem Erzeugen geändert hat."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wsz_constants as C  # noqa: E402
import wsz_experiments as X  # noqa: E402

PARTS_DIR = ROOT / "data" / "_parts"
STAND = "2026-10-24"


def run_part(name: str) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = X.PARTS[name]()
    out = PARTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"hash": X.source_hash(), "data": result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: fertig in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)


def merge() -> None:
    parts, hashes = {}, set()
    for name in X.PARTS:
        raw = json.loads((PARTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        parts[name] = raw["data"]
        hashes.add(raw["hash"])
    if hashes != {X.source_hash()}:
        sys.exit(f"Teile sind mit anderem Quellstand erzeugt ({sorted(hashes)} gegen {X.source_hash()}); Teile neu erzeugen")
    meta = {"stand": STAND, "erzeuger_hash": X.source_hash(), "seeds": list(C.SEEDS), "quellen": list(X.SOURCES)}
    (ROOT / "data" / C.RESULTS_FILE).write_text(json.dumps({"meta": meta, **parts}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("data/wsz_results.json geschrieben", X.source_hash())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(X.PARTS))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.part:
        run_part(args.part)
    elif args.merge:
        merge()
    else:
        ap.error("--part oder --merge angeben")
