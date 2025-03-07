import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# Définition des 14 features à utiliser
SELECTED_FEATURES = [
    "Account length",
    "Area code",
    "Total day minutes",
    "Total day calls",
    "Total eve minutes",
    "Total eve calls",
    "Total night minutes",
    "Total night calls",
    "Total intl minutes",
    "Total intl calls",
    "Customer service calls",
    "International plan_Yes",
    "Voice mail plan_Yes",
    "State_CA",
]


def preprocess_data(df):
    """
    Convertit les variables catégoriques en numériques pour XGBoost.
    """
    categorical_cols = ["State", "International plan", "Voice mail plan"]

    # Vérifier quelles colonnes existent dans le dataset
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Vérifier si toutes les colonnes sont numériques après transformation
    print(f"✅ Nombre de features après encodage : {df.shape[1]}")
    print("✅ Vérification des types après transformation :")
    print(df.dtypes)

    return df


def align_features(train_df, test_df):
    """
    S'assure que le dataset de test a les mêmes colonnes que le dataset d'entraînement.
    Ajoute les colonnes manquantes avec des 0 et supprime les colonnes en trop.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Ajouter les colonnes manquantes dans le test avec des 0
    missing_cols = train_cols - test_cols
    for col in missing_cols:
        test_df[col] = 0

    # Supprimer les colonnes en trop dans le test
    extra_cols = test_cols - train_cols
    test_df = test_df.drop(columns=extra_cols, errors="ignore")

    # Réordonner les colonnes pour qu'elles soient identiques
    test_df = test_df[train_df.columns]

    return test_df


def prepare_data(train_path, test_path):
    """
    Charge et prépare les données d'entraînement et de test.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Appliquer la conversion des catégories en numériques
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Vérifier si la colonne cible "Churn" est bien présente
    if "Churn" not in train_df.columns or "Churn" not in test_df.columns:
        raise ValueError("⚠️ La colonne 'Churn' est manquante dans les datasets.")

    # Aligner les features entre train et test
    test_df = align_features(train_df, test_df)

    # Séparer les features (X) et la cible (y)
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # Sélectionner uniquement les 14 features
    X_train = X_train[SELECTED_FEATURES]
    X_test = X_test[SELECTED_FEATURES]

    return X_train, X_test, y_train, y_test


def train_model(train_path):
    """
    Entraîne un modèle XGBoost sur les données d'entraînement.
    """
    X_train, _, y_train, _ = prepare_data(train_path, train_path)

    # Création et entraînement du modèle XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
    model.fit(X_train, y_train)

    print("✅ Modèle XGBoost entraîné avec succès !")
    return model


def custom_accuracy_score(y_true, y_pred):
    """
    Fonction manuelle pour calculer la précision du modèle.
    """
    correct = sum(np.array(y_true) == np.array(y_pred))
    total = len(y_true)
    return correct / total


def evaluate_model(model, test_path):
    """
    Évalue le modèle sur les données de test.
    """
    _, X_test, _, y_test = prepare_data(test_path, test_path)

    # Vérifier que le nombre de features correspond bien au modèle
    if X_test.shape[1] != len(model.feature_names_in_):
        raise ValueError(
            f"⚠️ Mismatch des features : attendu {len(model.feature_names_in_)}, reçu {X_test.shape[1]}"
        )

    y_pred = model.predict(X_test)
    accuracy = custom_accuracy_score(y_test, y_pred)
    print(f"✅ Précision du modèle XGBoost : {accuracy:.2f}")
    return accuracy


def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Modèle sauvegardé sous {filename}")


def load_model(filename):
    """
    Charge un modèle à partir d'un fichier.
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"📂 Modèle chargé depuis {filename}")
    return model
