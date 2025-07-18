# Deutsche Bahn Delay Prediction – Modelltraining mit Wetterdaten
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 1. Dateneinlesung & Vorverarbeitung
# ----------------------------

# Lade Zugdaten (.parquet) – Pfad anpassen
data_path = "data/data-2025-01.parquet"
df = pd.read_parquet(data_path)

# Filter auf München Hbf und relevante Spalten
df_filtered = df[df['station'] == 'München Hbf'][[
    'station', 'final_destination_station', 'delay_in_min', 'time',
    'is_canceled', 'train_type', 'train_line_station_num', 'departure_planned_time'
]]

# ----------------------------
# 2. Wetterdaten laden (Open-Meteo API)
# ----------------------------

weather_url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=48.1371&longitude=11.5754&start_date=2025-01-01&end_date=2025-01-30"
    "&hourly=temperature_2m,precipitation,windspeed_10m&timezone=Europe%2FBerlin"
)

weather_json = pd.read_json(weather_url)
weather_data = weather_json['hourly']

df_weather = pd.DataFrame({
    'time_wetter': pd.to_datetime(weather_data['time']),
    'temperature_2m': weather_data['temperature_2m'],
    'precipitation': weather_data['precipitation'],
    'windspeed_10m': weather_data['windspeed_10m']
})

# ----------------------------
# 3. Merge Zugdaten + Wetterdaten (zeitlich)
# ----------------------------

df_filtered.sort_values('time', inplace=True)
df_weather.sort_values('time_wetter', inplace=True)

df_merged = pd.merge_asof(df_filtered, df_weather, left_on='time', right_on='time_wetter')
df_merged.drop(columns='time_wetter', inplace=True)

# ----------------------------
# 4. Feature Engineering
# ----------------------------

df_merged['hour'] = df_merged['time'].dt.hour
df_merged['weekday'] = df_merged['departure_planned_time'].dt.weekday
df_merged['departure_hour'] = df_merged['departure_planned_time'].dt.hour
df_merged['time'] = df_merged['time'].dt.hour
df_merged['departure_planned_time'] = df_merged['departure_planned_time'].dt.hour

# Zugausfälle mit -1 kennzeichnen
df_merged.loc[df_merged['is_canceled'] == True, 'delay_in_min'] = -1

# Verspätungsklassen definieren
def delay_category(minute):
    if minute < 0: return -1     # Zug ausgefallen
    elif minute == 0: return 0
    elif minute <= 5: return 1
    elif minute <= 15: return 2
    else: return 3

df_merged['delay_class'] = df_merged['delay_in_min'].apply(delay_category)

# Weitere sinnvolle Features
df_merged['is_weekend'] = df_merged['weekday'] >= 5
df_merged['is_rush_hour'] = df_merged['departure_hour'].between(7, 9) | df_merged['departure_hour'].between(16, 18)
df_merged['precipitation_binary'] = df_merged['precipitation'] > 0

# Temperatur kategorisieren
def categorize_temp(t):
    if t < 0: return 'cold'
    elif t < 15: return 'mild'
    else: return 'warm'

df_merged['temp_category'] = df_merged['temperature_2m'].apply(categorize_temp)

# Optional: Speichern zur späteren Analyse
df_merged.to_csv("output/kombined_data.csv", index=False)

# ----------------------------
# 5. Modelltraining
# ----------------------------

# Zielvariable & Features vorbereiten
X = df_merged.drop(columns=['delay_in_min', 'is_canceled', 'delay_class'])
y = df_merged['delay_class']

# One-hot Encoding für kategorielle Features
X_encoded = pd.get_dummies(X, columns=[
    'station', 'final_destination_station', 'train_type',
    'train_line_station_num', 'temp_category'
], drop_first=True)

# Datensplitting
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Modelltraining (Random Forest)
clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Vorhersagen & Evaluation
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 6. (Optional) Feature Importance Plot
# ----------------------------
# import matplotlib.pyplot as plt
# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]
# feat_names = X_encoded.columns

# plt.figure(figsize=(10, 6))
# plt.title("Top Feature Importances")
# plt.bar(range(10), importances[indices[:10]], align='center')
# plt.xticks(range(10), [feat_names[i] for i in indices[:10]], rotation=45)
# plt.tight_layout()
# plt.show()
