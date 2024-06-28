import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import joblib

# Chemin vers le fichier CSV
csv_path = "C:/Users/hugob/OneDrive/Bureau/GII/stage_2e_année/App_example/data.csv"

# Charger les données
df = pd.read_csv(csv_path)
df.info()

# Séparation des caractéristiques et des étiquettes
features = df[["VOLUME", "TOTAL_STUDENTS", "OCCUPIED_TIME", "OPENING_SIZE_WINDOW", "OPENINNG_WINDOW_TIME", "OPENING_SIZE_DOOR", "OPENING_DOOR_TIME"]]
labels = df["IAQ_LEVEL"]

# Division des données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

# Entraînement du modèle
iaq = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, max_depth=10, random_state=42)
iaq.fit(train_features, train_labels)
rfr_y_pred = iaq.predict(test_features)

# Évaluation du modèle
print('Accuracy: %.4f' % metrics.accuracy_score(test_labels, rfr_y_pred))
print('R2: %.4f' % r2_score(test_labels, rfr_y_pred))

# Sauvegarde du modèle
joblib.dump(iaq, "model.pkl")
