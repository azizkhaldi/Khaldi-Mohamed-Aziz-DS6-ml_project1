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

# 📌 Connexion sécurisée à Elasticsearch
ELASTICSEARCH_HOST = "http://localhost:9200"
ELASTICSEARCH_USER = "elastic"  # Remplace par ton utilisateur
ELASTICSEARCH_PASSWORD = "your_password"  # Remplace par ton mot de passe
ELASTICSEARCH_INDEX = "mlflow-metrics"

try:
    es = Elasticsearch(
        [ELASTICSEARCH_HOST],
        basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD),
        verify_certs=False
    )
    if not es.ping():
        raise ValueError("❌ Connexion à Elasticsearch échouée. Vérifiez que le service tourne.")
    print("✅ Connexion à Elasticsearch réussie !")
except Exception as e:
    print(f"❌ Erreur de connexion à Elasticsearch: {e}")

# 📌 Fonction pour envoyer des métriques à Elasticsearch
def log_metric_to_elasticsearch(metric_name, value):
    if es:
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metric_name": metric_name,
            "value": value
        }
        try:
            es.index(index=ELASTICSEARCH_INDEX, document=log_entry)
            logging.info(f"📊 Log envoyé à Elasticsearch : {log_entry}")
        except Exception as e:
            logging.error(f"❌ Erreur d'envoi du log vers Elasticsearch: {e}")

# 📌 Configuration des logs
logging.basicConfig(level=logging.INFO)

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
        logging.info("📌 Préparation des données...")
        try:
            X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
            logging.info("✅ Données préparées avec succès !")
        except Exception as e:
            logging.error(f"⚠️ Erreur lors de la préparation des données : {e}")

    if args.train:
        logging.info("📌 Entraînement du modèle...")
        try:
            X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)

            # 📌 Initialiser MLflow
            mlflow.set_experiment("Churn_Model_Experiment")  

            with mlflow.start_run() as run:
                logging.info("🚀 Début de l'entraînement du modèle...")

                # 📌 Définition du modèle XGBoost
                model = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )

                # 📌 Entraîner le modèle
                logging.info("🔄 Entraînement en cours...")
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
                logging.info(f"✅ Modèle enregistré dans MLflow avec run ID: {run_id}")

                # 📌 Évaluation du modèle
                accuracy = evaluate_model(model, args.test_path)
                logging.info(f"🎯 Model Training Completed! Accuracy: {accuracy:.4f}")

                # 📌 Enregistrer les métriques dans MLflow et Elasticsearch
                mlflow.log_metric("accuracy", accuracy)
                log_metric_to_elasticsearch("accuracy", accuracy)

        except Exception as e:
            logging.error(f"⚠️ Erreur durant l'entraînement : {e}")

    if args.evaluate:
        logging.info("📌 Évaluation du modèle...")
        try:
            model = load_model("xgboost_model.pkl")
            accuracy = evaluate_model(model, args.test_path)
            logging.info(f"✅ Précision du modèle : {accuracy:.4f}")

            # 📌 Loguer la métrique dans Elasticsearch
            log_metric_to_elasticsearch("evaluation_accuracy", accuracy)

        except Exception as e:
            logging.error(f"⚠️ Erreur durant l'évaluation : {e}")

    if args.predict:
        logging.info("📌 Prédiction en cours...")
        try:
            if args.input:
                model = load_model("xgboost_model.pkl")
                data = list(map(float, args.input.split(",")))

                # 📌 Vérifier si les features correspondent (ajout d'une vérification)
                if hasattr(model, "feature_names_in_") and len(data) != len(model.feature_names_in_):
                    logging.error(
                        f"⚠️ Erreur : nombre de features incorrect. Attendu {len(model.feature_names_in_)}, reçu {len(data)}."
                    )
                else:
                    prediction = model.predict([data])[0]
                    logging.info(f"✅ Prédiction : {prediction}")

                    # 📌 Loguer la prédiction dans Elasticsearch
                    log_metric_to_elasticsearch("prediction", prediction)

            else:
                logging.error("⚠️ Veuillez fournir des données avec --input.")
        except Exception as e:
            logging.error(f"⚠️ Erreur durant la prédiction : {e}")

def list_model_versions(model_name):
    client = mlflow.tracking.MlflowClient()
    try:
        model = client.get_registered_model(model_name)
        print(f"Model: {model.name}")
        for version in model.latest_versions:
            print(f"Version: {version.version}, Stage: {version.current_stage}")
    except Exception as e:
        print(f"⚠️ Erreur : {e}")

if __name__ == "__main__":
    main()

