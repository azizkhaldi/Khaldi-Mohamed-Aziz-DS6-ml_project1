import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données
data = load_iris()
X = data.data
y = data.target

# Diviser en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Sauvegarder le modèle
model.save_model('xgboost_model.json')

print("Modèle XGBoost sauvegardé avec succès.")

