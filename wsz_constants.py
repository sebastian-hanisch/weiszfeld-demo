"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Einheit der Ebene: 1 Längeneinheit = 1 km (Luftlinie). Gewicht eines Kunden = Sendungsmenge (Stück je Woche), ohne weitere Bedeutung."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
LAYOUTS = ("Normal", "Quadrat", "Kern + Außenposten", "Kern + Seitengruppe")
WEIGHT_KINDS = ("gleich", "lognormal 0,5", "lognormal 1", "Großkunde", "Großkunde an der Schwelle")
STARTS = ("Schwerpunkt", "nächster Datenpunkt", "zufälliger Punkt")
SOLVERS = ("klassisch", "mit Epsilon-Schutz", "Vardi–Zhang")

N_MIN, N_MAX, DEFAULT_N = 3, 200, 10               # Karte, Zugkräfte und Tukey-Tiefe: O(n²) bis O(n³), siehe TUKEY_LIVE_MAX
SHARE_MIN, SHARE_MAX, DEFAULT_SHARE = 5, 95, 40    # Gewichtsanteil des Großkunden in Prozent
FACTOR_MIN, FACTOR_MAX, DEFAULT_FACTOR = 0.5, 3.0, 1.0   # Gewicht des Großkunden in Vielfachen der Schwelle
S_MIN, S_MAX, DEFAULT_S = 1.0, 10.0, 1.0           # Achsenfaktor: y-Achse mal s
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 180, 0    # Drehwinkel in Grad
TOL_EXP_MIN, TOL_EXP_MAX, DEFAULT_TOL_EXP = -12, -3, -9   # Toleranz 10^e: Abbruch, wenn der Schritt kürzer ist
MAXIT_OPTIONS = (100, 300, 1000, 3000, 10000, 30000, 100000)
DEFAULT_MAXIT = 3000
SEED_MAX, DEFAULT_SEED = 999999, 7
CUSTOMER_MIN, DEFAULT_CUSTOMER = 1, 1

# --- Rechenbudget (Slider-Maxima) ---------------------------------------------------------------------------------------------------------
TUKEY_LIVE_MAX = 150        # tiefster Datenpunkt: O(n³) vektorisiert; darüber wird er nicht berechnet
PATH_KEEP = 300             # so viele Iterate werden für den Pfad gezeichnet
REFERENCE_MAXIT = 20000     # Referenzlösung, wenn der Standort nicht auf einem Kunden liegt
LIVE_BUDGET = 10 ** 8       # Zufallsexperimente live: Instanzen · n² höchstens so groß

# --- Messreihen (vorgerechnet in data/wsz_results.json, erzeugt von tools/make_results.py) ----------------------------------------------
SEEDS = (11, 12, 13, 14)    # vier feste Zufalls-Seeds je Zelle: nie ein einzelner Seed als Befund
N_RASTER = (3, 5, 10, 20, 30, 50, 100)
FACTORS = (0.5, 0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1, 1.5, 3.0)
SHARES = (5, 10, 20, 30, 40, 50, 60)
S_VALUES = (1.5, 2.0, 5.0, 10.0)
DIMS = (2, 5, 10, 20, 50, 100, 200)
EPS_CONTAMINATION = 0.1
RESULTS_FILE = "wsz_results.json"

# --- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_POINT = "rgba(80,80,80,0.55)"
COLOR_BIG = "#b8860b"
COLOR_MEAN = "#7f7f7f"
COLOR_COORD = "#9467bd"
COLOR_WEBER = "#2F6B65"
COLOR_TUKEY = "#d62728"
COLOR_PATH = "#4c78a8"
COLOR_PULL = "#e07b00"
COLOR_NET = "#2F6B65"

# --- Presets (Werte für jeden Regler; die Seeds sind in tests/test_presets.py auf ihr Verhalten geprüft) ---------------------------------
_BASE = {
    "layout": "Normal", "n": DEFAULT_N, "weights": "gleich", "share": DEFAULT_SHARE, "factor": DEFAULT_FACTOR, "s": DEFAULT_S,
    "angle": DEFAULT_ANGLE, "start": "Schwerpunkt", "solver": "Vardi–Zhang", "tol_exp": DEFAULT_TOL_EXP, "maxit": DEFAULT_MAXIT,
    "seed": DEFAULT_SEED, "customer": DEFAULT_CUSTOMER,
}


def _preset(**changes):
    unknown = set(changes) - set(_BASE)
    assert not unknown, unknown
    return {**_BASE, **changes}


PRESETS = {
    "Standard (Normal, n = 10, gleiche Gewichte)": _preset(),
    "Stumpfes Dreieck": _preset(n=3, seed=2),
    "Großkunde mit 40 %": _preset(n=20, weights="Großkunde", share=40, seed=4),
    "Großkunde mit 55 %": _preset(n=20, weights="Großkunde", share=55, seed=4),
    "Genau an der Schwelle": _preset(n=6, weights="Großkunde an der Schwelle", factor=1.0, solver="mit Epsilon-Schutz", maxit=100000, seed=3),
    "Start auf dem Kunden": _preset(n=10, start="nächster Datenpunkt", solver="klassisch", seed=7),
    "Achsen in Minuten": _preset(layout="Kern + Seitengruppe", n=40, s=5.0, seed=2),
}
PRESET_HELP = {
    "Standard (Normal, n = 10, gleiche Gewichte)": "Zehn normalverteilte Kunden mit gleichem Gewicht, Vardi–Zhang ab dem Schwerpunkt. Der Weber-Punkt liegt zwischen den Kunden; die Zugkraft-Pfeile am gewählten Kunden sind stärker als sein Gewicht.",
    "Stumpfes Dreieck": "Drei Kunden, an einer Ecke ein Winkel von mehr als 120°. Der Weber-Punkt liegt genau auf dieser Ecke (Kunde 1): Die Zugkraft der beiden anderen reicht nicht, ihn herauszuziehen. Kein Punkt im Innern des Dreiecks ist besser.",
    "Großkunde mit 40 %": "20 Kunden, Kunde 1 hat 40 % des Gesamtgewichts. Hier reicht das noch nicht: Die Zugkraft der übrigen ist größer als sein Gewicht, der Standort liegt zwischen den Kunden. Erhöhen Sie den Anteil auf 55 %.",
    "Großkunde mit 55 %": "Dieselben 20 Kunden, Kunde 1 hat 55 % des Gesamtgewichts. Ab der Hälfte liegt der Standort in jeder Instanz auf dem Großkunden (Dreiecksungleichung).",
    "Genau an der Schwelle": "Sechs Kunden, das Gewicht von Kunde 1 ist genau seine Schwelle (f = 1). Der Standort liegt auf ihm, aber das Verfahren nähert sich nur noch sehr langsam: Die Iterationsgrenze von 100 000 Schritten wird erreicht. Die Kosten sind längst optimal, nur die Meldung „konvergiert“ bleibt aus.",
    "Start auf dem Kunden": "Das klassische Verfahren startet auf dem Kunden, der dem Schwerpunkt am nächsten liegt, und bricht im ersten Schritt ab: Der Iterat steht auf einem Kunden, der Schritt wäre 0/0. Der Ort ist nicht das Optimum. Wechseln Sie auf Vardi–Zhang oder den Schwerpunkt als Start.",
    "Achsen in Minuten": "40 Kunden mit Seitengruppe, die y-Achse mit 5 gestreckt (Kilometer in Minuten). Der Weber-Punkt der gestreckten Daten weicht vom mitgestreckten Weber-Punkt der Ausgangsdaten ab (hohler Stern); der Mehrweg bleibt aber gering, das Optimum ist flach.",
}
