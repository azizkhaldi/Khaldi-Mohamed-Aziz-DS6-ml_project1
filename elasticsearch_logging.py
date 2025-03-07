import logging
import datetime
from elasticsearch import Elasticsearch
import json

# 🔹 Connexion à Elasticsearch
ELASTICSEARCH_HOST = "http://localhost:9200"  # Assurez-vous qu'Elasticsearch tourne bien
ELASTICSEARCH_INDEX = "mlflow-logs"  # Nom de l'index pour les logs

# 🔹 Configuration de la connexion
try:
    es = Elasticsearch([ELASTICSEARCH_HOST])
    if not es.ping():
        raise ValueError("Connexion à Elasticsearch échouée. Vérifiez que le service tourne.")
except Exception as e:
    print(f"❌ Erreur de connexion à Elasticsearch: {e}")

# 🔹 Définir un logger personnalisé
class ElasticsearchHandler(logging.Handler):
    """ Handler personnalisé pour envoyer les logs vers Elasticsearch """
    
    def emit(self, record):
        log_entry = self.format(record)
        try:
            es.index(index=ELASTICSEARCH_INDEX, body=json.loads(log_entry))
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi du log vers Elasticsearch: {e}")

# 🔹 Configuration du format des logs
log_formatter = logging.Formatter(
    json.dumps({
        "timestamp": "%(asctime)s",
        "level": "%(levelname)s",
        "message": "%(message)s",
        "module": "%(module)s",
        "function": "%(funcName)s",
        "line": "%(lineno)d"
    })
)

# 🔹 Création du logger
logger = logging.getLogger("elasticsearch_logger")
logger.setLevel(logging.INFO)

# 🔹 Ajout du handler Elasticsearch
es_handler = ElasticsearchHandler()
es_handler.setFormatter(log_formatter)
logger.addHandler(es_handler)

# 🔹 Ajout d'un handler de console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# 🔹 Fonction pour envoyer un log de test
def log_test():
    logger.info("🚀 Elasticsearch Logging is working correctly!")
    logger.warning("⚠️ Attention, ceci est un warning !")
    logger.error("❌ Une erreur s'est produite !")

if __name__ == "__main__":
    log_test()

