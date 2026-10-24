# Geometrischer Median: Wo steht der Weber-Punkt? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-weiszfeld-demo.streamlit.app/)** -->

Stück 5 der Reihe **Mittelwerte und Lagemaße** (Arbeitsname; die Linie `lagemasse` ist auf der Website noch nicht angelegt) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Kante im Abhängigkeitsgraphen: Stück 3 (Lp-Mitten) → 5, mit der Schwäche des Vorgängers als Anlass: In der Ebene gibt es für den Abstand mit Exponent 1 keine Formel, und der koordinatenweise Median ist nicht drehinvariant. Stück 5 setzt den Medianbegriff aus Stück 2 und das p aus Stück 3 voraus und wiederholt sie nur in einem Satz.

Vehikel: **ein Lager für gewichtete Kunden in der Ebene** (Gewicht = Sendungsmenge, Einheit 1 km, Luftlinie). Gesucht ist der Punkt mit dem kürzesten Gesamtweg: der **geometrische Median** oder **Weber-Punkt**. Die Demo zeigt (a) die Zugkräfte der Kunden auf einen Standort und die Bedingung, ab der ein Kunde den Standort „besitzt“, (b) das Verhalten des Weiszfeld-Verfahrens an dieser Schwelle, (c) Startwahl und Absicherung, (d) Einheiten und Drehung, (e) hohe Dimension.

Abgrenzung: `standortplanung-demo`, `p-hub-median-demo` und `kapazitierte-standortplanung-demo` wählen aus **diskreten** Kandidaten; `kmeans-demo` rechnet den Medoid (einen Datenpunkt), nicht den stetigen Median; `p-center-demo` nennt das kontinuierliche p-Center nur für p = 1 ausdrücklich als nicht gebaut und behandelt Summe gegen Maximum diskret. Der Abgleich der READMEs (455 lokale und die per `gh api` gelesenen) fand zu Weiszfeld, Weber und geometrischer Median keinen Treffer.

## Warum dieses Problem

Die naheliegende Antwort „der Standort liegt in der Mitte der Kunden“ stimmt für den Schwerpunkt, nicht für den kürzesten Gesamtweg. Der Hook der Demo sind fünf gemessene Befunde statt eines Lehrbuchbilds: (1) das Optimum liegt **oft genau auf einem Kunden** und bei einem Gewichtsanteil von mindestens 50 % **immer** auf dem Großkunden; (2) das Standardverfahren **versagt dort, wo es stimmt** (Iterationen steigen an der Schwelle in die Zehntausende, ein Fünftel der Läufe erreicht die Grenze von 100.000 Schritten, ohne dass der Schritt unter die Toleranz fällt); (3) ein Start auf einem Kunden macht es in mehr als der Hälfte der Fälle falsch, **auch mit der üblichen Epsilon-Absicherung**; (4) der Weber-Punkt hängt an den **Achseneinheiten**, der tiefste Datenpunkt (Tukey) nicht; (5) in hoher Dimension hilft er weniger als erhofft.

Kernfrage: *Wo steht der Weber-Punkt, wovon hängt er ab, und warum ist der Algorithmus ausgerechnet dort schwierig, wo das Ergebnis eine Kundenposition ist?*

## Befunde und Korrekturen gegenüber dem Plan

Die Hypothesen des Einzelplans (`mittelwerte-planung/STUECK-5-weiszfeld.md`) wurden im Bau mit dem **eigenen SplitMix64-Strom** neu gemessen; die Vormessung des Plans (numpy-Zufall, Wegwerfskripte) liefert keine README-Zahl. Alle Zahlen: vier feste Zufalls-Seeds je Zelle, Mittel mit Streuung zwischen den Seeds.

| | Hypothese | Ergebnis | Zahl (diese Demo) |
|---|---|---|---|
| H1 | Das Optimum liegt bei n = 3 / 5 / 10 / 30 / 100 normalverteilten Kunden gleichen Gewichts in 43 / 26 / 13 / 4 / 1 % der Instanzen auf einem Kunden | **Ja**, bei n = 100 etwas höher als im Plan | 43,2 % / 26,5 % / 12,8 % / 4,2 % / 1,6 % |
| H2 | Bei einem Gewichtsanteil ab 50 % liegt der Standort immer auf dem Großkunden; darunter wächst der Anteil von 0,3 % (5 %) auf 51 % (40 %) | **Ja** | 0,4 % (5 %), 1,5 % (10 %), 7,1 % (20 %), 22,0 % (30 %), 52,1 % (40 %), 100,0 % (50 % und 60 %) |
| H3 | Dreiecke mit einem Winkel ab 120°: 44 % (Normal), 40 % (Quadrat); Weiszfeld = Torricelli-Konstruktion | **Ja** | 43,8 % und 40,0 %; Abstand Torricelli–Vardi–Zhang in 2.000 Dreiecken höchstens 3,8·10⁻¹⁰, nie ein schlechterer Gesamtweg |
| H4 | Die Iterationen steigen zur Schwelle (52 → 53.019) und fallen danach (7.956 → 19); 23 % konvergieren bei f = 1 nicht in 100.000 Schritten | **Ja**, mit anderen Zahlen (96 statt 48 Instanzen je Zeile) | Median 54,5 (f = 0,5), 55.970,5 (f = 1,0), 19 (f = 3,0); 19,8 % erreichen bei f = 1,0 die Grenze |
| H5 | Linear weg von der Schwelle, sublinear (etwa 1/k) an ihr | **Ja, als Messung** an 24 Instanzen, kein Beweis | Quotient e(k+1)/e(k) bei f = 1,0: 0,933 (k = 10), 0,991 (100), 0,999 (1.000), 0,9999 (10.000); Steigung von log e über log k: −1,00 |
| H6 | Start auf einem Kunden: klassisch und Epsilon-Schutz in 56 bis 96 % nicht optimal, Vardi–Zhang in 0 % | **Ja** | 56,55 % / 74,45 % / 87,70 % / 96,80 % (n = 3 / 5 / 10 / 30), Vardi–Zhang 0,00 % |
| H7 | Der Weber-Punkt wandert bei Achsenstreckung (0,24 Streuungen bei s = 5), der Tukey- und der Koordinatenmedian nicht; Mehrweg im Mittel 0,43 %, bis 3,1 % | **Ja**, Zahlen leicht anders | 0,25 Streuungen; Mehrweg 0,46 % im Mittel, größter Fall 2,5 % |
| H8 | Der Koordinatenmedian ist nicht drehinvariant (0,16 Streuungen bei 45°), der geometrische Median schon | **Ja** | 0,16 Streuungen (größter Fall 0,45); geometrischer Median 2·10⁻¹⁵ |
| H9 | Hohe Dimension: Der Fehler wächst mit √d bei weit entferntem Haufen (Fall A), sättigt bei festem Abstand (Fall B) | **Ja**, mit einem Zusatz in Fall B | Fall A (n = 1.000): geometrischer Median geteilt durch ε·√d zwischen 1,16 und 1,28; Fall B: 1,27 → 0,47 |
| H10 | Ein einfacher Filter schlägt Mittel und Median bei weit entferntem Haufen auch bei n/d = 5, bei nahem Haufen nur bei n ≫ d | **Ja**, die Aussage „mit n = 20·d brauchbar“ des Plans trägt **nicht** | Fall A, d = 200, n = 1.000: Filter 0,542 gegen 1,635 (geometrischer Median) und 7,080 (Mittel) |

### Korrekturen gegenüber dem Plan

- **Neue Zahlen, gleiches Bild.** Der Plan nannte 53.019 Iterationen und 22,9 % an der Grenze (48 Instanzen); mit 96 Instanzen sind es 55.970,5 und 19,8 %. Der Plan nannte 1,2 % bei n = 100, hier sind es 1,6 % (vier Seeds mit je 400 Instanzen, Streuung zwischen den Seeds 0,4 Prozentpunkte). Achsenstreckung: 0,25 statt 0,24 Streuungen. Fall A: Quotient 1,16 bis 1,28 (n = 1.000) statt 1,12 bis 1,34.
- **Neu gegenüber dem Plan: Das Abbruchkriterium täuscht an der Schwelle.** Der Abbruch bei Schritt < 10⁻⁹ ist dort kein Maß für den Fehler: Bei f = 1,0 liegt der Iterat im Median noch 5,6·10⁻⁵ km vom Optimum entfernt; bei f = 3,0 sind es 2,6·10⁻¹⁰ km. Mit f = 0,999 und 1,001 sind es etwa 10⁻⁶ km.
- **Neu: In Fall B ist das Mittel ab d = 10 besser als der geometrische Median** (n = 1.000: 0,306 gegen 0,317 bei d = 10 und 0,526 gegen 0,666 bei d = 200). Der Plan sagte „der Median hilft kaum“; bei festem Abstand des Haufens hilft er in hoher Dimension nicht und schadet sogar.
- **Der einfache Filter** schlägt das Mittel in Fall B bei n = 1.000 nur bis d = 50 (0,316 gegen 0,372) und liegt ab d = 100 darüber (0,532 gegen 0,433 bei d = 100, 0,706 gegen 0,526 bei d = 200). Mit n = 20·d ist er bei d = 100 gleich gut wie das Mittel (0,365 gegen 0,368) und bei d = 200 schlechter (0,408 gegen 0,369); der Plan nannte ihn dort „brauchbar“, das ist nicht belegt.
- **PLAN E3 (alt, ‖v‖ = √d)** vermischte Verschiebung und Dimension; hier steht Fall A (‖v‖ = 5·√d) neben Fall B (‖v‖ = 3).
- **Umsetzung weicht ab:** (1) Der Tukey-Median ist hier der **tiefste Datenpunkt** (Halbraumtiefe, Gleichstand: Schwerpunkt der tiefsten Punkte), vektorisiert in O(n³) statt O(n² log n); live bis 150 Kunden. (2) Ein viertes Layout „Kern + Seitengruppe“ (30 Kunden um den Ursprung, 10 um (4; 1) bei n = 40) und die Gewichtsart „Großkunde an der Schwelle“ kamen dazu. (3) Der Regler n geht bis 200 statt 5.000 (die Karte zeichnet Zugkräfte, die Tukey-Tiefe wächst mit n³). (4) Das Preset „Hohe Dimension“ entfällt, weil die Dimension nicht zur Karte gehört; E6 hat eigene Regler.

## Modell

- **Instanz.** Kunden an Orten x_i in der Ebene (Einheit 1 km), Gewichte w_i > 0, eigener SplitMix64-Generator (`wsz_rng.py` ist die wörtliche Kopie aus `reduktionskette-demo/red_scenario.py`, geprüft von `tests/test_copies.py`), damit alle Instanzen auf jeder Plattform gleich sind. Layouts: Normal, Quadrat, Kern + Außenposten, Kern + Seitengruppe. Gewichte: gleich, lognormal (σ = 0,5 oder 1), Großkunde mit gewähltem Anteil, Großkunde an der Schwelle.
- **Aufgabe.** Minimiere den Gesamtweg F(c) = Σ w_i ‖x_i − c‖. F ist konvex; liegen die Kunden nicht alle auf einer Geraden, ist das Minimum eindeutig.
- **Optimalität.** Liegt das Minimum an keinem Kundenort, heben sich die Zugkräfte auf. Liegt es auf Kunde k, ist die Norm der Summe der gewichteten Einheitsvektoren der übrigen höchstens w_k (**Schwellentest**). Hat Kunde k mindestens die Hälfte des Gesamtgewichts, folgt das aus der Dreiecksungleichung.
- **Verfahren.** Klassisches Weiszfeld-Verfahren (bricht ab, wenn ein Iterat exakt auf einem Kunden steht), mit Epsilon-Schutz (Abstände nach unten auf 10⁻¹² begrenzt), Vardi–Zhang (Behandlung des Kunden am Iterat), dazu Torricelli-Konstruktion für drei Punkte. Mathematik und Annahmentabelle stehen in den Expandern **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung** der App. Weiszfeld (1937), Kuhn (1973) und Vardi–Zhang (2000) wurden **nicht eingesehen**; die Verfahren sind nach ihrer Beschreibung umgesetzt und gegen unabhängige Rechenwege geprüft. Eine Konvergenzrate wird nur als Messung gezeigt, nicht als Satz.

## Methodik

| Baustein | Datei |
|---|---|
| Zielfunktion, Schwerpunkt, Zugkräfte, Schwellentest, Weiszfeld (klassisch, Epsilon-Schutz), Vardi–Zhang, Torricelli-Punkt | `wsz_geomed.py` |
| Generator (SplitMix64-Kopie), Layouts, Gewichte, Großkunde, Schwelle | `wsz_rng.py`, `wsz_scenario.py` |
| Achsenstreckung, Drehung, Mehrweg, gewichteter Koordinatenmedian, Halbraumtiefe, tiefster Datenpunkt | `wsz_equivariance.py` |
| Verunreinigte Normalverteilung, geometrischer Median in R^d, einfacher Filter | `wsz_highdim.py` |
| Messreihen E1 bis E6 (Funktionen für die Live-Knöpfe und die Vorrechnung) | `wsz_experiments.py`, `tools/make_results.py` → `data/wsz_results.json` |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `data/wsz_results.json` und sind in `tests/test_claims.py` nachgerechnet.

**E1 Wie oft liegt der Standort auf einem Kunden?** (Optimum auf einem Kunden, in % der Instanzen; Mittel über vier Seeds)

| Layout, Gewichte | n = 3 | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Normal, gleich | 43,2 | 26,5 | 12,8 | 6,8 | 4,2 | 2,2 | 1,6 |
| Normal, lognormal 1 | 86,3 | 64,0 | 36,2 | 18,0 | 12,0 | 6,9 | 3,9 |
| Quadrat, gleich | 40,1 | 22,1 | 10,9 | 5,5 | 3,2 | 2,0 | 0,8 |
| Kern + Außenposten, gleich | 61,5 | 32,3 | 14,0 | 6,4 | 4,3 | 2,5 | 1,2 |
| Kern + Seitengruppe, gleich | 53,8 | 30,4 | 14,6 | 7,5 | 5,1 | 3,1 | 1,4 |

Der Anteil fällt mit n, steigt mit der Gewichtsstreuung und mit einem Außenposten (die vollständige Rasterung mit lognormal 0,5 steht in der JSON-Datei).

**E3 Der Algorithmus an der Schwelle** (sechs Normalpunkte, Gewicht von Kunde 0 gleich f mal seine Schwelle, Start im Schwerpunkt, Schritt < 10⁻⁹, Grenze 100.000, 96 Instanzen je Zeile)

| f | 0,5 | 0,9 | 0,99 | 0,999 | 1,0 | 1,001 | 1,01 | 1,1 | 1,5 | 3,0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Median der Iterationen | 54,5 | 153,5 | 1.240,5 | 8.027,5 | 55.970,5 | 8.049,5 | 1.267,5 | 179 | 49 | 19 |

Zufallsinstanzen (gleiche Gewichte, 2.000 je n, Start im Schwerpunkt): Median 48 / 56 / 48 / 36 / 31 Iterationen für n = 3 / 5 / 10 / 30 / 100, größter Wert 10.000 / 9.275 / 3.797 / 1.137 / 695; wenige Instanzen liegen nahe an der Schwelle.

**E4 Der Start entscheidet** (Start auf dem Kunden, der dem Schwerpunkt am nächsten liegt; „nicht optimal“ = Gesamtweg mehr als 10⁻⁷ über dem Optimum): klassisch und mit Epsilon-Schutz 56,55 / 74,45 / 87,70 / 96,80 % (n = 3 / 5 / 10 / 30), Vardi–Zhang 0,00 %; klassisch mit Start im Schwerpunkt 0,00 %. Der klassische Lauf bricht vom Kunden aus immer ab (der Start ist ein Kunde); die Epsilon-Absicherung ändert am Ergebnis nichts, weil der Kunde den Iterat mit dem Gewicht w/10⁻¹² an sich zieht.

**E5 Einheiten und Drehung** (40 Kunden im Layout Kern + Seitengruppe, 600 Instanzen)

| Achsenfaktor s | 1,5 | 2 | 5 | 10 |
|---|---|---|---|---|
| Weber-Punkt: Abstand zwischen „Mitte, dann strecken“ und „strecken, dann Mitte“ (Streuungen der x-Werte) | 0,03 | 0,06 | 0,25 | 0,58 |
| Mehrweg im Mittel (% des Gesamtwegs) | 0,03 | 0,10 | 0,46 | 0,71 |

Koordinatenmedian und tiefster Datenpunkt (60 Instanzen bei s = 5): Abweichung höchstens 2·10⁻¹⁵. Der Standort wandert deutlich, die Kosten kaum: das Optimum ist flach.

**E6 Hohe Dimension** (10 % Haufen, 40 Läufe je Zelle, n = 1.000; Fehler des Schätzers)

| d | 2 | 20 | 200 |
|---|---|---|---|
| Fall A: Mittel | 0,706 | 2,236 | 7,080 |
| Fall A: Koordinatenmedian | 0,204 | 0,640 | 2,050 |
| Fall A: geometrischer Median | 0,180 | 0,519 | 1,635 |
| Fall A: einfacher Filter | 0,041 | 0,147 | 0,542 |
| Fall B: Mittel | 0,302 | 0,327 | 0,526 |
| Fall B: geometrischer Median | 0,179 | 0,406 | 0,666 |

## Ehrliche Grenzen

- **Luftlinie in der Ebene.** Keine Straßen, keine Kapazität, keine Fixkosten; der Weber-Punkt kann an einem unbrauchbaren Ort liegen. Dafür sind die Standortplanungs-Demos mit diskreten Kandidaten da.
- **Simulierte Layouts.** Alle Häufigkeiten und Raten gelten für die vier Layouts und fünf Gewichtsarten dieser Demo; echte Kundenlagen können abweichen.
- **Vier Seeds je Zelle.** Die angegebene Streuung ist die zwischen diesen Seeds, kein Konfidenzintervall. Die Rate (1/k an der Schwelle) ist an 24 Instanzen gemessen und nicht bewiesen; Konvergenzbeweise werden nicht geführt.
- **Der einfache Filter ist eine eigene vereinfachte Fassung** und setzt Kovarianz I der guten Punkte voraus; sein Abschneiden sagt nichts über den Literaturalgorithmus.
- **Tiefster Datenpunkt statt Tukey-Median.** Der Tukey-Median im engeren Sinn ist die Mitte der tiefsten Region.
- **Der Vardi–Zhang-Schritt ist nach der Beschreibung des Verfahrens umgesetzt,** die Originalarbeit wurde nicht eingesehen; geprüft ist er gegen `scipy.optimize`, eine Gitter-Suche und die Torricelli-Konstruktion.
- **Mit echtem Mausklick nicht geprüft:** Die Zugkräfte werden per Regler „Kunde für die Zugkräfte“ gewählt, nicht durch Anklicken eines Punktes.

## Tests

- **Kern gegen Handrechnung** (`tests/conftest.py`, `tests/test_algorithm.py`): stumpfes Dreieck (Zielwert 2·√1,04), Gewichtsschwelle √2, Quadrat mit Mitte, drei kollineare Punkte, ein Weiszfeld-Schritt von Hand, gleichseitiges Dreieck.
- **Orakel** (`tests/test_oracle_geometrischer_median.py`): Löser gegen `scipy.optimize` (Nelder-Mead, drei Starts), Gitter-Brute-Force für drei und vier Kunden, Torricelli-Konstruktion, Schwellentest gegen die Numerik, Halbraumtiefe gegen Aufzählung von Halbräumen mit ganzzahligen Normalen und fein abgetasteten Richtungen, Gesamtweg gegen `scipy.spatial`. Größerer Lauf im Scratchpad vor dem Bau: 2.000 gewichtete Instanzen (größte relative Abweichung gegen scipy 1,3·10⁻¹¹), 2.997 Schwellentests (alle übereinstimmend, 3 zu nah an der Schwelle übersprungen), 600 Halbraumtiefen (0 Abweichungen), 5.000 Dreiecke gegen die Torricelli-Konstruktion (größte Abweichung 3,5·10⁻¹⁰, nie ein schlechterer Gesamtweg).
- **Eigenschaftstests** (`tests/test_equivariance.py`): Drehinvarianz und Verschiebung des Weber-Punkts, Mitgehen des Koordinatenmedians bei Streckung und nicht bei Drehung, Invarianz der Halbraumtiefe unter Scherung.
- **Zweig-Tests** (`tests/test_zweige.py`): jede Löser-Variante und jede Startwahl weicht auf mindestens einer Instanz von der Nachbarvariante ab; jedes Layout und jede Gewichtsart verändert den Anteil auf Kunden (`tests/test_presets.py`).
- **Messreihen** (`tests/test_experiments.py`, `tests/test_results.py`): Reproduzierbarkeit, Stapelrechnung gegen Einzelrechnung, qualitative Befunde in kleiner Größe, vorgerechnete Datei aktuell gegen die Quelldateien (Prüfsumme).
- **Mutationsprüfung** (`oracle_mutation.py` aus dem Website-Repo, inkrementell, Zustand in `tests/mutation_state.json`): `wsz_geomed.py` und `wsz_equivariance.py`, 104 Stellen. Im ersten Lauf überlebten 29 Mutanten die ganze Suite (Rate 72 %, Werkzeugausgabe). Die Lücken (Zugkraft an einer Ecke abseits des Ursprungs, Blockgrenzen der zeilenweisen Rechnung, gedämpfter Vardi–Zhang-Schritt, Gleichstand zwischen Zugkraft und Gewicht, Iterationszähler, Pfadlänge, Mehrweg bei Gesamtweg unter 1) sind mit Tests geschlossen; gesichert sind jetzt 102 Stellen (87 gefangen, 15 als äquivalent festgehalten: Gleichstände innerhalb der Toleranz, Blockgrößen ohne Wirkung auf das Ergebnis), 2 Zeitüberschreitungen (Endlosschleife) zählen nicht. Nicht gezogen wurden `wsz_highdim.py` und `wsz_scenario.py` (das Werkzeug wählt sie nicht als Kernmodul), Auswertung, Darstellung und die kopierte `wsz_rng.py`. „Mutation sauber“ steht hier bewusst nicht.
- **Aussagen** (`tests/test_claims.py`): jede Zahl dieser README.
- **Umfang:** über 200 Tests, alle grün.
- **Oberfläche:** AppTest (jeder Knopf, jeder Regler am Minimum und Maximum, jede Option der Auswahlregler, Extremkombinationen), Layout, Footer, Presets, PDF, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App |
| `wsz_constants.py` | Regler-Grenzen, Messreihen-Konstanten, Farben, Presets mit Hilfetexten |
| `wsz_geomed.py` | Kern: Zugkräfte, Schwellentest, Weiszfeld, Vardi–Zhang, Torricelli |
| `wsz_scenario.py`, `wsz_rng.py` | Instanzen; SplitMix64 (Kopie) |
| `wsz_equivariance.py` | Streckung, Drehung, Mehrweg, Koordinatenmedian, Halbraumtiefe |
| `wsz_highdim.py` | Hohe Dimension: Verunreinigung, Median in R^d, einfacher Filter |
| `wsz_experiments.py` | Messreihen-Funktionen, Prüfsumme der Quellen |
| `wsz_results.py` | Laden der vorgerechneten Messreihen |
| `wsz_evaluation.py` | Kennzahlen der Karte, Verdikte |
| `wsz_visualization.py`, `wsz_pdf_export.py`, `wsz_presets.py`, `wsz_layout.py` | Figuren, PDF, Permalink und Presets, fester Kopf |
| `tools/make_results.py` | erzeugt `data/wsz_results.json` |
| `data/wsz_results.json` | die vorgerechneten Messreihen mit Prüfsumme |
| `tests/` | Tests (siehe oben) |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Laufzeit, Test und Gegenproben (scipy nur dort), CI |

## Bewusst nicht umgesetzt

- **Die ganze Familie der Lp-Mitten** (Stück 3), **der Median einer Zahlenreihe** (Stück 2): hier nur p = 1 in der Ebene und je ein Satz zur Einordnung.
- **Straßennetz, Kapazitäten, Fixkosten:** gehören zur Standortplanung mit diskreten Kandidaten.
- **Optimale Verfahren für hohe Dimension:** gezeigt wird nur ein eigener einfacher Filter; Literaturalgorithmen sind nicht nachgebaut.
- **Konvergenzbeweis und Satz über die Rate:** die Rate steht als Messung da; Quellen wurden nicht gelesen.
- **Der Tukey-Median im engeren Sinn** (Mitte der tiefsten Region) und eine Fassung für n über 150.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Streamlit-Cloud-Einstellung (Python-Version) ist erst beim Deploy festzulegen; lokal läuft die Demo mit Python 3.14, die CI mit 3.12. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Messreihen werden mit `python tools/make_results.py --part <m1|m3|m4|m5|m6>` (jeder Teil einzeln, parallel möglich) und `python tools/make_results.py --merge` neu erzeugt; `tests/test_results.py` meldet, wenn die Datei hinter den Quelldateien zurückliegt.

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
