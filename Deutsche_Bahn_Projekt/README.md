# Deutsche Bahn – Zugverspätungsanalyse mit Wetterdaten

Dieses Projekt untersucht die Zusammenhänge zwischen **Zugverspätungen am Münchner Hauptbahnhof** und **Wetterbedingungen**. Ziel ist es, mit Hilfe von **Wetter- und Fahrplandaten ein Vorhersagemodell** zu trainieren, das Zugverspätungen besser erklärbar macht – z. B. durch Temperatur, Niederschlag, Wochentag oder Uhrzeit.

---

## Projektstruktur

```plaintext
bahn-delay-prediction/
├── data/                      # Rohdaten (.parquet) – nicht im Repo enthalten
│   └── data-2025-01.parquet
├── output/                    # Zwischenergebnisse und Exporte
│   └── kombined_data.csv
├── src/                       # Quellcode
│   └── train_model.py
├── README.md                  # Dieses Dokument
Datenquellen
Die historischen Zugverspätungsdaten stammen aus dem Open-Source-Projekt:
https://github.com/piebro/deutsche-bahn-data
Dieses Repository stellt monatlich aktualisierte .parquet-Dateien mit aggregierten Zugverspätungen der Deutschen Bahn zur Verfügung.

Beispieldatei: data-2025-01.parquet
Manuell herunterladen und im data/-Ordner ablegen

Wetterdaten (Open-Meteo)
Historische Wetterdaten für München (Temperatur, Niederschlag, Wind) über die Open-Meteo API.

Projektziele & Methodik
Vorverarbeitung der Fahrplandaten: nur München Hbf, relevante Spalten.

Download und Zusammenführung mit stündlichen Wetterdaten.

Feature Engineering: Wochentag, Stoßzeiten, Wetterkategorien, Zugausfälle.

Klassifizierung der Verspätung in:

-1: Ausgefallen

0: Pünktlich

1: ≤ 5 Min

2: ≤ 15 Min

3: > 15 Min

Training eines Random Forest Classifier auf den Feature-Daten.

Auswertung mit Confusion Matrix & Klassifikationsreport.

Beispiel-Ergebnisse
Durchschnittlicher Fehler (MAE): ca. 2 Minuten

Erklärte Varianz (R²): ~41 %

Temperatur war das relevanteste Wetter-Feature, Niederschlag hatte nur geringen Einfluss.


