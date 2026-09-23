# NNUE für eine Schach-Engine (C++20, AVX2/AVX512, int8-Quantisierung)

Ein eigenständiges NNUE (Efficiently Updatable Neural Network) für die
Stellungsbewertung: HalfKP-Features (40960 Eingänge, extrem dünnbesetzt),
inkrementell aktualisierter Akkumulator, quantisierte int8-Zwischenschichten
mit handgeschriebenen AVX2/AVX512-Kernen, sowie ein vollständiges Trainings-
modul mit dem "lazy sparse update"-Trick für den riesigen Embedding-Layer.

## Architektur

```
40960 Features (HalfKP, one-hot, ~30 aktiv)
        │  nur Zeilen addieren/subtrahieren (kein Matmul!)
        ▼
Accumulator[2][ACC_SIZE]   (int16, pro Perspektive: Weiß/Schwarz)
        │  ClippedReLU: clamp(x, 0, 127) → uint8
        ▼
concat(eigene Perspektive, gegnerische Perspektive)  → 2*ACC_SIZE uint8
        │  AffineLayer (int8 Gewichte, int32 Bias, AVX2/AVX512 Dot-Produkt)
        ▼
H1 (Standard: 128)  → ClippedReLU → H2 (32) → ClippedReLU → H3 (32) → ClippedReLU
        │  AffineLayer<H3,1>
        ▼
int32 Rohwert → /output_scale → Centipawns
```

Alle Schichtgrößen sind Template-Parameter
(`nnue::NNUE<ACC_SIZE, H1, H2, H3>`), Architekturen lassen sich also einfach
durch Ändern der Zahlen und Neukompilieren ausprobieren:

```cpp
nnue::NNUE<256, 128, 32, 32> stockfish_like;
nnue::NNUE<128, 64, 32, 16>  small;
```

## Warum das erste Layer kein Matmul ist

HalfKP-Index (exakt wie vorgegeben):

```
p_idx        = piece_type * 2 + piece_color
halfkp_idx   = piece_square + (p_idx + king_square * 10) * 64
```

Der Input-Vektor ist one-hot pro Figur (≈30 aktive von 40960 Einträgen).
Damit ist `Wᵀx` für das erste Layer nichts anderes als "addiere die Zeilen
von W, deren Feature aktiv ist" – siehe
[`include/nnue/feature_transformer.hpp`](include/nnue/feature_transformer.hpp).
Das macht zwei Dinge möglich:

1. **Forward Pass ist O(aktive Features)** statt O(40960 × ACC_SIZE).
2. **Inkrementelles Update bei `make_move`**: Ein Zug ändert nur 2–4
   Features (Wegzug, Hinzug, ggf. Schlagfigur, ggf. Turm bei Rochade) –
   der Akkumulator wird per `apply_add`/`apply_remove` in O(1) statt neu
   berechnet. Einzige Ausnahme: bewegt sich der **eigene König**, ändert
   sich `king_square` und damit *jedes* Feature dieser Perspektive → dafür
   ist `Accumulator::refresh()` nötig (nur für diese eine Perspektive, die
   gegnerische bleibt inkrementell). Siehe [`src/demo.cpp`](src/demo.cpp)
   für ein durchgerechnetes Beispiel inkl. Selbstcheck gegen einen vollen
   Refresh.

## Quantisierung

* **Feature Transformer**: Gewichte `int8`, Bias `int16`, Akkumulator
  `int16` (breiter als int8, weil ~30 Zeilen aufsummiert werden – sonst
  Overflow). Skala `QA = 127`.
* **Alle folgenden Schichten**: Gewichte `int8`, Bias `int32`, Skala
  `QB = 64` (Zweierpotenz ⇒ Requantisierung ist ein exakter Rechts-Shift,
  keine gerundete Division).
* Jedes ClippedReLU-Ergebnis liegt einheitlich in der "×QA"-Fixpunkt-Domäne
  (`[0,127]` uint8); jede Dense-Schicht multipliziert dabei einen Faktor
  `QB` hinzu, den `>> log2(QB) = >> 6` wieder herausrechnet. Details und
  Herleitung stehen in
  [`training/train_network.hpp`](training/train_network.hpp) (Kopfkommentar)
  und in den `ft_shift`/`hidden_shift`-Feldern von
  [`include/nnue/network.hpp`](include/nnue/network.hpp).

## AVX2/AVX512 – was ist vektorisiert

[`include/nnue/simd.hpp`](include/nnue/simd.hpp), drei Stufen je nach
Compiler-Flags (immer automatisch per Makro erkannt, kein Runtime-Dispatch
nötig da für eine Engine ohnehin fest für die Zielmaschine kompiliert wird):

| Kernel | AVX512VNNI | AVX2 | Scalar-Fallback |
|---|---|---|---|
| `dot_i8` (int8-Dot-Produkt für die Dense-Layer) | `_mm512_dpbusd_epi32` (1 Instruktion) | `_mm256_maddubs_epi16` + `_mm256_madd_epi16`-Trick | ✓ |
| `add_row_i8_to_i16` / `sub_row_i8_from_i16` (Akkumulator-Update) | – | `_mm256_cvtepi8_epi16` + add/sub | ✓ |
| `crelu_i16_to_u8` / `crelu_i32_to_u8` (ClippedReLU + Requantisierung) | – | `packs`+`permute`+`max` | ✓ |

Ohne `-mavx2`/`/arch:AVX2` fällt alles automatisch auf Skalar-Code zurück
(funktioniert auch auf Nicht-x86, z.B. zum Testen).

## Training ([`training/`](training))

Ein Float-Trainingsnetz ([`training/train_network.hpp`](training/train_network.hpp))
spiegelt die Architektur exakt, mit Forward/Backward-Pass (klassisches
Backprop, keine Autodiff-Bibliothek) und `quantize_into()`, das das
fertig trainierte Netz in die int8/int16-Inferenz-Struktur exportiert.

Der Feature-Transformer-Teil des Forward-Pass ist **quantisierungsbewusst**
(`fake_quantize()`, ein Straight-Through-Estimator): die Akkumulator-Summe
rundet schon während des Trainings so, wie `quantize_i8` es beim Export tun
wird. Ohne das hat Adam keinen Grund, ein Gewicht zu bevorzugen, das die
int8-Rundung übersteht, gegenüber einem beliebig kleineren, das genauso gut
den Float-Loss senkt – und `std::lround` rundet alles unter 1/127 beim
Export lautlos auf 0, wodurch das halbe gelernte Netz beim Quantisieren
verschwinden kann, ohne dass das Float-Training das je bemerkt.

### Der Sparse-Update-Trick

Die Feature-Transformer-Gewichtsmatrix hat 40960 × ACC_SIZE Einträge. Pro
Sample sind nur ~30 Zeilen aktiv → der *Gradient* ist natürlich dünnbesetzt.
Das Problem ist **Weight Decay** (L2-Regularisierung): ein normaler AdamW
zieht `lr * weight_decay * w` von *jedem* Gewicht in *jedem* Schritt ab –
auch von Zeilen ganz ohne Gradienten-Feedback in diesem Schritt. Naiv
implementiert müsste man also trotzdem alle 40960 Zeilen pro Schritt
anfassen und der Sparsity-Vorteil wäre dahin.

Der Trick ([`training/sparse_optimizer.hpp`](training/sparse_optimizer.hpp),
`LazySparseAdamW`): pro Zeile wird nur `last_step` gespeichert. Solange eine
Zeile nicht aktiv ist, passiert *nichts*. Wird sie wieder aktiv (oder beim
finalen `finalize()`-Aufruf am Trainingsende), wird der Decay für die
übersprungenen Schritte in geschlossener Form auf einmal nachgeholt:

```
w  *= (1 - lr·weight_decay)^Δt      // Decoupled Weight Decay
m  *= β1^Δt                         // Adams 1. Moment (Gradient=0 in der Lücke)
v  *= β2^Δt                         // Adams 2. Moment
```

Das ist – bis auf Float-Rundung – **exakt** das Ergebnis eines dichten
Optimierers, der wirklich jede der 40960 Zeilen jeden Schritt angefasst
hätte, kostet aber nur O(aktive Zeilen) pro Schritt statt O(40960).

### Trainings-Loop ausprobieren (synthetische Daten)

[`training/train_demo.cpp`](training/train_demo.cpp) trainiert auf
synthetischen Zufallsstellungen (Materialwert → Sigmoid-Zielwahrscheinlich-
keit) und exportiert `trained_net.nnue`. Nur zur Demonstration der
Trainingsmechanik – für echte Daten siehe den nächsten Abschnitt.

## Voller Trainingszyklus mit Stockfish-Daten

[`training/full_cycle.cpp`](training/full_cycle.cpp) ist das eigentliche
Trainings-Tool: erzeugt zufällige *legale* Stellungen mit dem Movegen des
umgebenden toMateTo-Engines (`../toMateTo_engine/move_generation/`), lässt
jede von Stockfish auf einstellbarer Tiefe bewerten, trainiert direkt
darauf, und speichert Datensatz + Netz unter einem gemeinsamen Namen.

```bash
./train.sh --depth 12 --samples 50000 --name my_net
# erzeugt: my_net.dataset (FEN;stockfish_cp;side_to_move, eine Zeile pro Stellung)
#          my_net.nnue     (fertig trainiertes, quantisiertes int8-Netz)
```

### Ausführungsmöglichkeiten (`--mode`)

`full_cycle` (bzw. `./train.sh`) kennt drei Modi, per `--mode`:

- **`full`** (Default): generieren + trainieren in einem Durchlauf, wie oben.
- **`gen_data`**: nur `--name.dataset` erzeugen, kein Netz, kein Training,
  kein Leaderboard-Eintrag – praktisch um einen Datensatz einmal zu
  generieren und später mehrfach für verschiedene Architekturen/Runs zu
  verwenden.
- **`only_train`**: kein Stockfish, keine Generierung – trainiert direkt auf
  einem vorhandenen `--dataset PATH` (z. B. aus `gen_data`), optional über
  mehrere `--epochs`. Hier zahlt sich Multithreading am meisten aus (siehe
  oben) – kein Stockfish-Aufruf, den man "verstecken" könnte.

```bash
./train.sh --mode gen_data --depth 12 --samples 200000 --name shared_data
./train.sh --mode only_train --dataset shared_data.dataset --epochs 3 --name my_net
```

Beide Kommandos landen (bis auf `gen_data`, das gar kein Netz erzeugt) am
Ende wieder im selben Leaderboard wie `full`.

**Architektur ist ein Compile-Time-Parameter, kein Runtime-Flag.**
`NNUE<ACC_SIZE,H1,H2,H3>` ist ein C++-Template – die Größen müssen beim
Kompilieren feststehen, das geht mit C++-Templates grundsätzlich nicht zur
Laufzeit (siehe Kopfkommentar in `full_cycle.cpp`). `train.sh` versteckt das:

```bash
./train.sh --arch 512,256,64,32 --depth 14 --samples 100000 --name big_net
# konfiguriert CMake mit -DNNUE_ACC_SIZE=512 -DNNUE_H1=256 ... neu,
# baut full_cycle neu, läuft dann ganz normal.
```

Ohne `--arch` bleibt die zuletzt gebaute Architektur (Default:
`256,128,32,32`) erhalten – kein unnötiger Rebuild.

**Jeder Lauf trägt sich in ein gemeinsames Leaderboard ein** – auch die im
nächsten Abschnitt beschriebenen Self-Play-Läufe landen in derselben Datei.
Name, Methode, Architektur, Tiefe, Sample-/Spielzahl, Trainings-Loss,
Rechenzeit und die (unten erklärte) Validierung gegen Stockfish landen als
eine Zeile in `training_results.csv` (append-only, im aktuellen
Arbeitsverzeichnis – CSV statt Datenbank, damit nichts weiter installiert
werden muss). Am Ende jedes Laufs UND jederzeit per `--report` wird daraus
eine Rangliste gedruckt, sortiert nach `val_loss` (siehe unten – nicht nach
dem rohen Trainings-Loss, der zwischen den beiden Methoden nicht direkt
vergleichbar ist):

```bash
./train.sh --report        # oder: ./self_play.sh --report
```
```
== Leaderboard (training_results.csv, 2 Läufe insgesamt) ==
momentan bester (val_loss): "selfplay_demo" [selfplay]  train_loss=0.00000  val_loss=0.088402 corr=0.058631  computation time: 29286ms  architecture=256,128,32,32
Platz 2: "stockfish_demo" [stockfish]  train_loss=0.07028  val_loss=0.102708 corr=0.000000  computation time: 732ms  architecture=256,128,32,32
```

**Vor dem eigentlichen Training läuft immer der Efficiency-Check** (misst,
wie lange ein voller `Accumulator::refresh()` gegenüber einem einzelnen
inkrementellen `apply_add`/`apply_remove` für die gewählte `ACC_SIZE`
braucht – genau der Vorteil, um den es bei NNUE geht):

```
== efficiency check (ACC_SIZE=256) ==
  full refresh (32 active features):    146.2 ns/call
  incremental update (add or remove):       1.8 ns/call  (83.0x faster than full)
```

**Kosten/Dauer**: jede generierte Stellung braucht einen echten
Stockfish-Aufruf auf der gewählten Tiefe – bei Tiefe 10-12 realistisch
20-100+ ms pro Stellung (stark hardwareabhängig), bei `--samples 100000`
also potenziell Stunden. Ein Fortschrittsbalken mit `ms/item` und ETA läuft
laufend mit; Tiefe und Sample-Zahl entsprechend wählen.

**Multithreading** (`--threads N`, Default: alle logischen Kerne):
Datengenerierung parallelisiert trivial – jede Stellung ist unabhängig, und
jeder Worker-Thread bekommt seinen eigenen Stockfish-Prozess (per
`setoption name Threads value 1` selbst auf 1 Thread beschränkt, damit sich
N Worker nicht gegenseitig die Kerne wegnehmen). Ein Produzenten/Konsumenten-
Queue (`training/full_cycle.cpp`, `WorkQueue`) sammelt die fertig
bewerteten Stellungen ein; das eigentliche `net.train_step()` bleibt
einzelner Thread (ein gemeinsamer Adam-Optimizer-State, siehe Kommentar
dort). Bei `--mode only_train` (unten) trägt Multithreading dagegen direkt
im Training selbst: `compute_gradients()` ist pro Stellung unabhängig und
lässt sich über alle Threads parallelisieren, `apply_gradients()` fasst die
Batch-Gradienten zu einem sequentiellen Adam-Schritt zusammen (siehe
`train_network.hpp`).

**Nur Linux/WSL**: `full_cycle` spricht Stockfish über `fork`+`exec`+Pipes
an ([`training/uci_engine.hpp`](training/uci_engine.hpp), gleicher Ansatz
wie das bereits vorhandene `testing/stockfish_perft.cpp` im toMateTo-Repo)
– nicht auf natives Windows portierbar. Der Rest der NNUE-Bibliothek
bleibt plattformunabhängig.

**Gefundener Bug im toMateTo-Engine (nicht in nnue-chess)**: `chess_board::
setup_chess_board()` initialisiert `castling_rights`, `ep_square` und die
Zug-Zähler nicht – ein `chess_board board;` (Default-Init) startet dort mit
Stack-Müll statt mit den Werten einer echten Startstellung. `full_cycle.cpp`
umgeht das lokal (`chess_board board{};` + explizite Zuweisung dieser drei
Felder, siehe Kommentar dort), aber jeder andere Code, der `chess_board`
genauso default-konstruiert statt per `setup_fen_position()`, dürfte
denselben Effekt zeigen (fiel hier zuerst als unplausible `FEN`s im
generierten Datensatz auf – 60-Ply-Random-Walks landeten bei Fullmove-
Zählern >50).

## Self-Play Reinforcement Learning (ohne Stockfish)

[`training/self_play.cpp`](training/self_play.cpp) ist die zweite
Trainingsmethode: **kein** externes Labeling – das Netz spielt gegen sich
selbst und lernt aus seinen eigenen Such-Ergebnissen. Die Idee, die du
beschrieben hast ("nach dem besten Zug sollte die Position dieselbe
Bewertung haben"), ist klassisches **TD-Bootstrapping** (TD-Leaf, wie z.B.
in KnightCap): eine kleine Alpha-Beta-Suche (das gerade trainierte Netz
selbst als Blattbewertung) sucht an jeder besuchten Stellung ein paar Züge
voraus; der von der Suche zurückgelieferte Wert wird zum Trainingsziel
*für die Stellung, an der gesucht wurde* – die statische (0-Ply-)Bewertung
lernt so schrittweise, das nachzubilden, was die tiefere Suche findet.
Danach wird der von der Suche bevorzugte Zug tatsächlich gespielt und die
Partie geht weiter.

```bash
./self_play.sh --games 500 --search-depth 2 --max-plies 150 --name sp_net
```

**Eigene Kopie der Engine, wie gewünscht**: anders als `full_cycle` (das den
Movegen des übergeordneten toMateTo-Repos referenziert) nutzt `self_play`
eine physische Kopie unter [`engine/toMateTo_engine/`](engine/toMateTo_engine)
– komplett in sich geschlossen innerhalb von `nnue-chess`, keine Datei
außerhalb dieses Ordners wird gelesen oder geschrieben (außer der
Stockfish-Binary, s.u.).

**Der Skalierungs-Filter, den du angesprochen hast.** Self-Play hat nichts,
was seine interne Bewertungsskala an echte Centipawns bindet (anders als
`full_cycle`, wo jedes Ziel buchstäblich `sigmoid(stockfish_cp/400)` ist) –
das Netz könnte "richtige" Zugpräferenzen lernen, aber auf einer völlig
beliebigen eigenen Skala. Deshalb ruft `self_play` **nach** dem Training
einmalig Stockfish auf (der einzige Ort, an dem dieses Tool Stockfish
überhaupt nutzt – nie während des eigentlichen Trainings), bewertet damit
eine frische, nie trainierte Menge an Stellungen, fittet per kleinste
Quadrate eine affine Korrektur (Skala + Offset) zwischen Netz-Rohwert und
Stockfish-`cp`, und berichtet den Fehler **nach** dieser Korrektur
(`calibrated_prob_mse`, in Wahrscheinlichkeitsraum – direkt vergleichbar mit
`full_cycle`s Trainings-Loss) plus die unkorrigierte Pearson-Korrelation:

```
== validation against a fresh Stockfish-labeled held-out set ==
  correlation=0.0586  calibrated_prob_mse=0.08840  (fit: scale=229273.859 offset=-112396.766)
```

Siehe [`training/validation.hpp`](training/validation.hpp) für die Details
und Begründung. Beide Zahlen landen auch im Leaderboard (`val_loss`,
`corr`), das deshalb `stockfish`- und `selfplay`-Läufe direkt gegeneinander
ranken kann, obwohl ihre rohen Trainings-Losses auf unterschiedlichen Skalen
leben.

### Kollaps-Risiko: was probiert wurde, ehrlicher Stand

Self-Play neigt dazu, in einen Zustand zu kollabieren, in dem das Netz für
(fast) jede Stellung dieselbe Bewertung ausgibt – Suche und Statik werden
sich selbst-referenziell einig, ohne dass echtes Schachwissen dahinter
steckt. Mehrere Gegenmaßnahmen sind eingebaut:

- **`randomize_ft_weights()`**: der Feature-Transformer startet normalerweise
  bei exakt 0 (sinnvoll, wenn Stockfish von Anfang an externe,
  positionsabhängige Ziele liefert). Bei reinem Self-Play ist das ein
  echter Fixpunkt – jede Stellung bekommt denselben Bias-Wert, die Suche
  kollabiert auf genau diesen konstanten Wert, Ziel und Vorhersage sind
  exakt identisch, Gradient exakt Null für immer. Fix: kleiner zufälliger
  Startwert vor dem Self-Play.
- **Target-Network** (`--sync-every`, siehe `main()`s Kommentar): die Suche
  bewertet Blätter mit einer periodisch eingefrorenen Kopie des Netzes,
  nicht mit dem live trainierten – sonst kann der Optimizer billig "Suche
  und Statik stimmen überein" lernen, ohne irgendwas über Schach zu lernen
  (derselbe Trick wie DQNs Target-Network gegen "moving target").
- **Exploration** (`kExplorationRate`): 10% der Züge werden zufällig
  gespielt statt immer der Suchbestzug – sonst laufen alle Partien schnell
  auf dieselbe schmale Zugfolge zusammen.
- **Warmstart** (`--init-from`, `dequantize_from()`): startet mit einem
  vorher Stockfish-trainierten Netz statt Zufallsgewichten.
- **LR-Schedule** (`--lr-half-life`, s.u.): sinkende Lernrate für mehr
  Stabilität in späteren Phasen.

**Ehrlich**: in ~200-Partien-Testläufen hat keine einzelne Maßnahme das
Kollaps-Risiko zuverlässig beseitigt – der [`eval_sanity_check.hpp`](training/eval_sanity_check.hpp)-Test
(läuft automatisch am Ende jedes Laufs) schlägt in kurzen Läufen weiterhin
gelegentlich an. Wahrscheinlichste Erklärung: ein reines Skalenproblem,
kein einzelner Bug mehr – `full_cycle` selbst brauchte 150.000 Schritte bis
zum Durchbruch (siehe oben), Self-Play erzeugt viel weniger Positionen pro
Zeiteinheit als das. Für belastbare Self-Play-Ergebnisse also: deutlich
mehr Partien einplanen und den Sanity-Check am Ende immer im Auge behalten.

### TD(λ)-Rollout-Variante (`--td-mode lambda`)

Alternative zum obigen TD-Leaf-Ansatz, klassisches forward-view TD(λ)
(Sutton 1988, Tesauro/TD-Gammon-Stil): statt einer breiten Suche über alle
Züge wird von jeder Stellung aus **greedy** (nur der 1-Ply-beste Zug laut
aktueller Bewertung, keine Baumsuche) bis `--rollout-depth` Halbzüge weit
ausgerollt. Jede Stellung entlang dieser Linie bekommt danach ein Ziel aus
der λ-gewichteten Summe **aller** nachfolgenden Stellungen auf ihrer
eigenen Restlinie – Einfluss nimmt mit `--td-lambda` pro Halbzug Abstand ab.
Rechnerisch günstiger pro Halbzug als die volle Suche (ein Eval pro
Kandidatenzug statt Zweig²), in Tests bisher genauso kollapsanfällig wie
TD-Leaf.

```bash
./self_play.sh --td-mode lambda --games 500 --rollout-depth 7 --td-lambda 0.7 --name sp_lambda
```

### Lernraten-Schedule

Beide Tools (`full_cycle` und `self_play`) unterstützen jetzt `--lr` und
`--lr-half-life`. Formel: `lr(t) = lr₀ / (1 + t/half_life)` – bewusst
**kein** Schedule, der bei einem festen Schritt-Horizont auf 0 herunterfährt
(z.B. linear über N Schritte): der geht "tot", sobald man länger als geplant
trainiert, ohne Weg zurück außer Neustart. Diese Formel nähert sich 0 nur
asymptotisch an – bei `t=half_life` ist die LR halbiert, bei `t=10×half_life`
immer noch bei `lr₀/11`, nie exakt bei 0. Lieber zu langsames Sinken
(einfach länger laufen lassen) als zu schnelles (nicht mehr umkehrbar) – der
Default (100.000) ist entsprechend konservativ gewählt.

## Bauen

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/test_simd          # SIMD-Kernel gegen Skalar-Referenz verifizieren
./build/nnue_demo           # Inferenz-Verkabelung, inkrementelles Update
./build/nnue_train_demo     # Trainings-Loop + Export
```

**Tatsächlich verifiziert, nicht nur durchgerechnet**: der mitgelieferte
clang in dieser Sandbox ist zu alt für die hier installierte MSVC-STL-Vorschau,
aber `cl.exe` (der echte MSVC-Compiler) war verfügbar, und alles wurde damit
auf echter AVX2-Hardware gebaut und laufen gelassen, nicht nur gelesen:

- `tests/test_simd.cpp`: die AVX2-Kerne in `simd.hpp` (insbesondere die
  Pack/Permute-Sequenz in `crelu_i32_to_u8`) gegen eine Skalar-Referenz,
  200 Zufallsdurchläufe pro Funktion – **ALL PASS (0 failures)**.
- `src/demo.cpp`: HalfKP-Indexierung + inkrementelles Akkumulator-Update –
  `eval after move (incremental)` und `eval after move (full refresh)`
  stimmen exakt überein.
- `training/train_demo.cpp`: 150 000 Trainingsschritte auf synthetischen
  Stellungen, Loss fällt von ~0.116 auf ~0.0008 (>140×), und die
  quantisierte int8-Auswertung trifft für alle 20 Testpositionen Vorzeichen
  und Größenordnung der float-Trainingsvorhersage.

Auf dem Weg dahin sind drei echte Bugs aufgefallen und wurden gefixt (nicht
nur beim Draufschauen, sondern weil sie beim Ausführen sichtbar wurden):
ein Stack-Overflow durch ein zu groß auf dem Stack angelegtes `NNUE`-Objekt,
tote Gradienten durch exakt-Null-initialisierte Biases direkt an der
ClippedReLU-Grenze, und eine "Dying-ReLU"-Kaskade durch unclipped Adam-Schritte
über viele Schritte hinweg (siehe Kommentare bei `relu01_grad`,
`randomize_dense_weights` und `fake_quantize` in
[`training/train_network.hpp`](training/train_network.hpp)). Letzteres hat
auch gezeigt, warum quantisierungsbewusstes Training (Fake-Quantization mit
Straight-Through-Estimator) hier nötig ist: ohne sie rundet `quantize_i8`
sehr kleine, aber echte gelernte Gewichte beim Export lautlos auf 0.
