# Variables
PYTHON = python3
PIP = pip3
TRAIN_PATH = "/mnt/c/Users/azizk/Downloads/ML_Project_Files (1)11/archive (2)/churn-bigml-80.csv"  # Remplacez par votre chemin
TEST_PATH = "/mnt/c/Users/azizk/Downloads/ML_Project_Files (1)11/archive (2)/churn-bigml-20.csv"   # Remplacez par votre chemin

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Prepare data
prepare:
	$(PYTHON) main.py --prepare --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Train model
train:
	$(PYTHON) main.py --train --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Evaluate model
evaluate:
	$(PYTHON) main.py --evaluate --train_path $(TRAIN_PATH) --test_path $(TEST_PATH)

# Run tests
test:
	$(PYTHON) -m pytest tests/

# Code quality checks
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check .

# Format code
format:
	black .

# Security check
security:
	bandit -r .

# Run all CI steps
ci: lint format security test

# Clean up
clean:
	rm -rf __pycache__
	rm -f model.pkl

# Phony targets
.PHONY: all install prepare train evaluate lint format security ci clean test test_api api mlflow docker-up docker-down docker-clean

# Default target
all: mlflow api

# Test API (Start FastAPI and Flask)
test_api:
	@echo "Starting FastAPI (Swagger UI) on port 8000 and Flask on port 5000"
	# Run FastAPI in the background
	uvicorn app:app --reload --host 0.0.0.0 --port 8000 & 
	# Run Flask in the foreground
	python flask_app.py

# Commande pour démarrer l'API (FastAPI et Flask)
api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000 &  # Démarrer FastAPI
	python flask_app.py  # Démarrer Flask

# Commande pour démarrer MLflow
mlflow:
	mlflow ui --backend-store-uri sqlite:////mnt/c/Users/azizk/Khaldi-Mohamed-Aziz-4DS6-ml_project/mlflow.db --host 0.0.0.0 --port 5000 &


# Variables Docker
DOCKER_COMPOSE = docker-compose

# Démarrer les services avec Docker Compose
docker-up:
	@echo "Démarrage de Docker Compose..."
	$(DOCKER_COMPOSE) up -d

# Arrêter les services avec Docker Compose
docker-down:
	@echo "Arrêt de Docker Compose..."
	$(DOCKER_COMPOSE) down

# Nettoyage des conteneurs et volumes
docker-clean:
	@echo "Nettoyage des conteneurs et volumes Docker..."
	$(DOCKER_COMPOSE) down -v
# Construire l’image Docker
docker-build:
	docker build -t mohamed_aziz_khaldi_mlops .

# Exécuter le conteneur
docker-run:
	docker run -d -p 8000:8000 mohamed_aziz_khaldi_mlops

# Se connecter à Docker Hub
docker-login:
	docker login

# Taguer l’image Docker
docker-tag:
	docker tag mohamed_aziz_khaldi_mlops azizkhaldi/mlops:latest

# Pousser l’image sur Docker Hub
docker-push:
	docker push azizkhaldi/mlops:latest
# Démarrer Elasticsearch et Kibana
start-monitoring:
	docker-compose up -d

# Démarrer MLflow
start-mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Vérifier si Elasticsearch fonctionne
check-es:
	curl -X GET "http://localhost:9200"

# Vérifier si Kibana fonctionne
check-kibana:
	curl -I "http://localhost:5601"

# Arrêter tout
stop-monitoring:
	docker-compose down

