# Dermasense Model

DenseNet-121, fine-tuned auf dem [HAM10000-Datensatz](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) zur Klassifikation von Hautläsionen in 7 Klassen.

---

## Schnellstart

### 1. Datensatz herunterladen

Der Datensatz muss manuell von Kaggle bezogen werden:

**[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**

Nach dem Download (Kaggle-Account erforderlich) liegen folgende Dateien vor:

```
HAM10000_images_part_1/   ← Ordner mit .jpg-Bildern (Teil 1)
HAM10000_images_part_2/   ← Ordner mit .jpg-Bildern (Teil 2)
HAM10000_metadata.csv     ← Metadaten-Tabelle
```

### 2. Dateien ablegen

Die Dateien müssen im Projektverzeichnis wie folgt abgelegt werden:

```
dhbw-task-integrationsseminar-skin-scanner-model/
└── raw_data/
    ├── HAM10000_metadata.csv
    └── image_data/
        ├── HAM10000_images_part_1/
        │   ├── ISIC_0024306.jpg
        │   └── ...
        └── HAM10000_images_part_2/
            ├── ISIC_0029306.jpg
            └── ...
```

> **Wichtig:** Der Ordner `raw_data/` ist nicht im Repository enthalten und muss manuell angelegt werden. Die Bilder aus beiden Kaggle-Teilordnern (`HAM10000_images_part_1` und `HAM10000_images_part_2`) bleiben in ihren jeweiligen Unterordnern — das Skript durchsucht alle Unterordner automatisch.

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Python 3.9 oder neuer wird empfohlen.

### 4. Training starten

```bash
python src/train.py
```

Das Skript erkennt automatisch die verfügbare Hardware (CUDA → MPS → CPU) und startet das Training für 50 Epochen. Das beste Modell wird unter `models/densenet121_ft.pth` gespeichert.

**Training fortsetzen** (nach Unterbrechung): Skript einfach erneut ausführen — der Checkpoint unter `models/checkpoint.pth` wird automatisch geladen.

**Training neu starten**: `models/checkpoint.pth` löschen, dann Skript ausführen.

#### Alternativ: Training per Docker (NVIDIA GPU)

```bash
docker build -t dermasense .
docker run --gpus all \
  -v $(pwd)/raw_data:/app/raw_data:ro \
  -v $(pwd)/models:/app/models \
  dermasense python src/train.py
```

Ohne GPU (CPU-only):

```bash
docker run \
  -v $(pwd)/raw_data:/app/raw_data:ro \
  -v $(pwd)/models:/app/models \
  dermasense python src/train.py
```

### 5. Inferenz auf einem einzelnen Bild

```bash
python src/infer.py /pfad/zum/bild.jpg
```

### 6. API-Server starten

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Oder per Docker Compose:

```bash
docker compose up api
```

Die API ist dann unter `http://localhost:8000` erreichbar.

---

## Klassen

| Code | Bezeichnung |
|------|-------------|
| `akiec` | Aktinische Keratose |
| `bcc` | Basalzellkarzinom |
| `bkl` | Benigne Keratose |
| `df` | Dermatofibrom |
| `mel` | Melanom |
| `nv` | Melanozytischer Nävus |
| `vasc` | Vaskuläre Läsion |

`mel`, `bcc` und `akiec` werden zur Laufzeit als **Hochrisiko** markiert.

---

## Wie das Training funktioniert

### 1. Normalisierung

Vor dem Training werden Mittelwert (`mean`) und Standardabweichung (`std`) pixelweise über alle ~10.000 HAM10000-Bilder berechnet — getrennt für jeden der drei RGB-Kanäle. Diese Werte werden zur Normalisierung aller Bilder verwendet.

Der Grund: Neuronale Netze lernen stabiler, wenn die Eingabedaten zentriert um 0 liegen. ImageNet-Normwerte würden hier zu einer systematischen Verschiebung führen, da Dermatoskopie-Aufnahmen farblich anders verteilt sind als allgemeine Fotos.

### 2. Train/Validation-Split

Der Datensatz wird 80/20 aufgeteilt. Dabei wird sichergestellt, dass nur Läsionen mit **genau einem Bild** im Validierungsset landen. HAM10000 enthält nämlich mehrere Aufnahmen derselben Läsion — würde man diese naiv aufteilen, könnte dieselbe Läsion in Training **und** Validierung erscheinen, was die gemessene Genauigkeit künstlich aufbläht (*Data Leakage*).

### 3. Datenaugmentierung

Nur auf den Trainingsdaten werden zufällige Transformationen angewendet:
- Horizontaler und vertikaler Flip
- Rotation ±20°
- Color Jitter (Helligkeit, Kontrast, Farbton)

Das Modell sieht jedes Bild in jeder Epoche leicht anders — es kann sich keine festen Muster "einprägen" und muss stattdessen robuste Features lernen. Das Validierungsset bleibt unverändert, damit die gemessene Genauigkeit real ist.

### 4. Modellarchitektur

DenseNet-121 ist ein CNN, das vortrainiert auf ImageNet geladen wird. Der originale 1000-Klassen-Klassifikationskopf wird durch eine einzelne lineare Schicht mit 7 Ausgaben ersetzt. Alle Parameter — also auch die tiefen Feature-Extraktionsschichten — sind von Epoche 1 an trainierbar (*Full Fine-Tuning*).

Das Vortraining auf ImageNet liefert dem Modell bereits ein Verständnis für grundlegende visuelle Strukturen (Kanten, Texturen, Formen), das auf medizinische Bilder übertragen wird.

### 5. Verlustfunktion und Optimierung

Als Verlustfunktion wird **CrossEntropyLoss** verwendet. Sie misst für jedes Bild, wie weit die vorhergesagten Wahrscheinlichkeiten vom tatsächlichen Label abweichen. Je sicherer das Modell bei der richtigen Klasse ist, desto niedriger der Loss.

Der **Adam-Optimizer** passt die Gewichte nach jedem Batch an. Die Updateformel lautet:

```
neues_gewicht = altes_gewicht - lernrate × gradient
```

Der Gradient gibt die Richtung des steilsten Anstiegs im Loss an — das Modell bewegt sich in die entgegengesetzte Richtung, also bergab.

### 6. Learning Rate Scheduler

Ein konstanter Lernrate führt in späteren Trainingsphasen dazu, dass das Modell das Optimum überspringt und der `val_loss` zu oszillieren beginnt, statt weiter zu fallen.

Der **`ReduceLROnPlateau`-Scheduler** beobachtet den `val_loss` nach jeder Epoche. Verbessert er sich 4 Epochen lang nicht, wird die Lernrate halbiert:

```
lr → lr × 0.5
```

Dies wiederholt sich bis zu einem Minimum von `1e-6`. Dadurch werden die Schritte automatisch kleiner, sobald sich das Modell dem Optimum nähert — präziseres Lernen ohne manuelles Eingreifen.

### 7. Modell-Speicherung

Nach jeder Epoche wird geprüft, ob die aktuelle `val_acc` die bisher beste übertrifft. Nur dann wird das Modell gespeichert. So enthält `models/densenet121_ft.pth` immer die Gewichte des Moments mit der besten Generalisierung — nicht zwingend der letzten Epoche.

Zusätzlich wird ein vollständiger Checkpoint gespeichert (`models/checkpoint.pth`), der Modellgewichte, Optimizer-Zustand, Scheduler-Zustand und die gesamte Trainingshistorie enthält. Das ermöglicht das nahtlose Fortsetzen des Trainings nach einer Unterbrechung.

---

## Hyperparameter

| Parameter | Wert |
|-----------|------|
| Bildgröße | 224×224 |
| Batch Size | 32 |
| Lernrate (initial) | 1e-3 |
| Epochen | 50 |
| Scheduler Patience | 4 Epochen |
| Scheduler Factor | 0.5 |
| Minimale Lernrate | 1e-6 |
