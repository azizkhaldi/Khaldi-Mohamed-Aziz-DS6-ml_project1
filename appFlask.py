from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Modèle chargé avec succès !")
except FileNotFoundError:
    print("⚠️ Erreur : Modèle non trouvé. Exécutez d'abord le processus de formation.")
    model = None

# Liste des features du modèle
categorical_cols = ["State", "International plan", "Voice mail plan"]
feature_names = [
    "State",
    "Account length",
    "Area code",
    "International plan",
    "Voice mail plan",
    "Number vmail messages",
    "Total day calls",
    "Total day charge",
    "Total eve calls",
    "Total eve charge",
    "Total night calls",
    "Total night charge",
    "Total intl calls",
    "Total intl charge",
]


# Fonction pour prétraiter les données avant la prédiction
def preprocess_input(features):
    """
    Applique le même encodage que celui utilisé pour entraîner le modèle.
    """
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données du formulaire et les convertir en float
        features = [float(x) for x in request.form.getlist("features")]
        processed_features = preprocess_input(features)

        # Vérifier que le nombre de features correspond bien au modèle
        if len(processed_features) != len(model.feature_names_in_):
            return render_template(
                "index.html",
                feature_names=feature_names,
                error="Erreur de dimensions des features",
            )

        # Faire la prédiction
        prediction = model.predict([processed_features])[0]
        probas = model.predict_proba([processed_features])[0]

        # Transformer 1 -> "Churn" et 0 -> "No Churn"
        prediction_label = "Churn" if prediction == 1 else "No Churn"

        return render_template(
            "index.html",
            feature_names=feature_names,
            prediction=prediction_label,
            probability_churn=round(probas[1], 2),
            probability_no_churn=round(probas[0], 2),
        )
    except Exception as e:
        return render_template("index.html", feature_names=feature_names, error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
