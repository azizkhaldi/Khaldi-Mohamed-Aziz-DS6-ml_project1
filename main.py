import argparse
import mlflow
import mlflow.xgboost
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)
import pandas as pd
import xgboost as xgb
import logging
import datetime
from elasticsearch import Elasticsearch

# 📌 Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5002")  # Assurez-vous que MLflow tourne sur ce port

# 📌 Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Configuration du logger pour envoyer les logs à Elasticsearch
logger = logging.getLogger("mlflow_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

# Fonction pour envoyer les logs vers Elasticsearch
def log_to_elasticsearch(message):
    try:
        es.index(index="mlflow-metrics", body={"message": message})
        logger.info(f"📊 Log envoyé à Elasticsearch : {message}")
    except Exception as e:
        logger.error(f"❌ Erreur d'envoi du log vers Elasticsearch: {e}")

# 📌 Fonction principale
def main():
    parser = argparse.ArgumentParser(description="Pipeline de machine learning")

    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--predict", action="store_true", help="Faire une prédiction")
    parser.add_argument("--train_path", type=str, help="Chemin du fichier d'entraînement")
    parser.add_argument("--test_path", type=str, help="Chemin du fichier de test")
    parser.add_argument("--input", type=str, help="Données pour la prédiction (ex: '5.1,3.5,1.4,0.2')")

    args = parser.parse_args()

    if args.prepare:
        logger.info("📌 Préparation des données...")
        try:
            X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
            logger.info("✅ Données préparées avec succès !")
        except Exception as e:
            logger.error(f"⚠️ Erreur lors de la préparation des données : {e}")

    if args.train:
        logger.info("📌 Entraînement du modèle...")
        try:
            X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)

            # 📌 Initialiser MLflow
            mlflow.set_experiment("Churn_Model_Experiment")  

            with mlflow.start_run() as run:
                logger.info("🚀 Début de l'entraînement du modèle...")

                # 📌 Définition du modèle XGBoost
                model = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )

                # 📌 Entraîner le modèle
                logger.info("🔄 Entraînement en cours...")
                model.fit(X_train, y_train)

                # 📌 Enregistrer les hyperparamètres et métriques
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("max_depth", 3)
                mlflow.log_param("learning_rate", 0.1)

                # 📌 Sauvegarde du modèle
                mlflow.xgboost.log_model(model, "xgboost_model")
                save_model(model, "xgboost_model.pkl")

                # 📌 Enregistrement du modèle dans le Model Registry
                run_id = run.info.run_id
                model_uri = f"runs:/{run_id}/xgboost_model"
                mlflow.register_model(model_uri, "XGBoost_Model")
                logger.info(f"✅ Modèle enregistré dans MLflow avec run ID: {run_id}")

                # 📌 Évaluation du modèle
                accuracy = evaluate_model(model, args.test_path)
                logger.info(f"🎯 Model Training Completed! Accuracy: {accuracy:.4f}")

                # 📌 Enregistrer les métriques dans MLflow et Elasticsearch
                mlflow.log_metric("accuracy", accuracy)
                log_to_elasticsearch(f"accuracy: {accuracy:.4f}")

        except Exception as e:
            logger.error(f"⚠️ Erreur durant l'entraînement : {e}")

    if args.evaluate:
        logger.info("📌 Évaluation du modèle...")
        try:
            model = load_model("xgboost_model.pkl")
            accuracy = evaluate_model(model, args.test_path)
            logger.info(f"✅ Précision du modèle : {accuracy:.4f}")

            # 📌 Loguer la métrique dans Elasticsearch
            log_to_elasticsearch(f"evaluation_accuracy: {accuracy:.4f}")

        except Exception as e:
            logger.error(f"⚠️ Erreur durant l'évaluation : {e}")

    if args.predict:
        logger.info("📌 Prédiction en cours...")
        try:
            if args.input:
                model = load_model("xgboost_model.pkl")
                data = list(map(float, args.input.split(",")))

                # 📌 Vérifier si les features correspondent (ajout d'une vérification)
                if hasattr(model, "feature_names_in_") and len(data) != len(model.feature_names_in_):
                    logger.error(
                        f"⚠️ Erreur : nombre de features incorrect. Attendu {len(model.feature_names_in_)}, reçu {len(data)}."
                    )
                else:
                    prediction = model.predict([data])[0]
                    logger.info(f"✅ Prédiction : {prediction}")

                    # 📌 Loguer la prédiction dans Elasticsearch
                    log_to_elasticsearch(f"prediction: {prediction}")

            else:
                logger.error("⚠️ Veuillez fournir des données avec --input.")
        except Exception as e:
            logger.error(f"⚠️ Erreur durant la prédiction : {e}")

if __name__ == "__main__":
    main()

