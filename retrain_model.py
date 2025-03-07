import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import OneHotEncoder

# Charger les données d'entraînement
train_data = pd.read_csv(
    "/mnt/c/Users/azizk/Downloads/ML_Project_Files (1)11/archive (2)/churn-bigml-80.csv"
)  # Remplace par ton chemin de fichier

# Sélectionner les 14 premières features
features = train_data.iloc[:, :14]  # Prendre les 14 premières colonnes
labels = train_data["Churn"]  # Remplacer 'Churn' par la colonne cible

# Encodage des variables catégorielles (State, International plan, Voice mail plan)
categorical_cols = ["State", "International plan", "Voice mail plan"]

# Appliquer OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_columns = encoder.fit_transform(train_data[categorical_cols])

# Créer un DataFrame avec les nouvelles colonnes encodées
encoded_df = pd.DataFrame(
    encoded_columns, columns=encoder.get_feature_names_out(categorical_cols)
)  # Remplacer par get_feature_names_out

# Ajouter ces colonnes encodées au DataFrame d'origine
train_data_encoded = pd.concat(
    [train_data.drop(columns=categorical_cols), encoded_df], axis=1
)

# Sélectionner à nouveau les features
features = train_data_encoded.iloc[
    :, :14
]  # Assure-toi que seules les 14 premières features sont utilisées
labels = train_data_encoded["Churn"]  # Remplacer 'Churn' par la colonne cible

# Créer le modèle XGBoost
model = xgb.XGBClassifier()

# Entraîner le modèle
model.fit(features, labels)

# Sauvegarder le modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(
    "✅ Modèle réentrainé avec les variables catégorielles encodées et sauvegardé sous 'model.pkl'"
)
