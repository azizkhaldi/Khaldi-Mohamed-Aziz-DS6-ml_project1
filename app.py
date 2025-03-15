from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
#
# Charger le modèle sauvegardé
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Modèle chargé avec succès !")
except FileNotFoundError:
    print("⚠️ Erreur : Modèle non trouvé. Exécutez d'abord `python main.py --train`")

# Initialiser FastAPI
app = FastAPI()


# Définir le format des données d'entrée pour les prédictions
class PredictionInput(BaseModel):
    features: List[float]


# Fonction pour prétraiter les données avant la prédiction


def preprocess_input(features):
    """
    Applique le même encodage que celui utilisé pour entraîner le modèle.
    """
    categorical_cols = ["State", "International plan", "Voice mail plan"]

    # Créer un DataFrame avec les nouvelles données
    df_input = pd.DataFrame(
        [features], columns=[f"feature_{i}" for i in range(len(features))]
    )

    # Ajouter les colonnes catégoriques avec une valeur par défaut si elles ne sont pas présentes
    for col in categorical_cols:
        if col not in df_input.columns:
            df_input[col] = "unknown"  # ou une valeur par défaut

    # Appliquer One-Hot Encoding
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Vérifier si toutes les colonnes attendues sont présentes
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df_encoded:
            df_encoded[col] = 0  # Ajouter les colonnes manquantes avec des 0

    df_encoded = df_encoded[expected_columns]  # Réordonner les colonnes

    return df_encoded.values.flatten()


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        processed_features = preprocess_input(data.features)

        # Vérifier que le nombre de features correspond bien au modèle
        if len(processed_features) != len(model.feature_names_in_):
            raise HTTPException(
                status_code=400,
                detail=f"Feature shape mismatch, expected: {len(model.feature_names_in_)}, got {len(processed_features)}",
            )

        # Faire la prédiction
        prediction = model.predict([processed_features])

        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Point de terminaison pour réentraîner le modèle
@app.post("/retrain")
def retrain():
    """
    Endpoint pour réentraîner le modèle avec de nouveaux hyperparamètres.
    Cette méthode peut être étendue pour accepter des paramètres via POST.
    """
    try:
        # Charger de nouveaux hyperparamètres et réentraîner ici
        print("Réentrainement du modèle avec de nouveaux paramètres...")
        # Exemple : réentraîner le modèle avec un nouveau fichier de données
        # Enregistrer le modèle après réentrainement
        # model.fit(X_train, y_train)  # Cette ligne est à adapter selon ton code

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        return {"message": "Modèle réentraîné avec succès"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors du réentraînement : {str(e)}"
        )
